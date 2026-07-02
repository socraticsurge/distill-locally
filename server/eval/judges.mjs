// Judge panel — 3 families, all API. Reads keys from .env at repo root.
// gpt-oss-120b (Groq) + llama-3.3-70b (Groq) + Gemini Flash Lite (Google AI Studio).
// Excludes Qwen (student) and DeepSeek (teacher) families by design.
// Rate-limit-aware: exponential backoff on 429/5xx.
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');

// minimal .env loader (no dependency)
export function loadEnv() {
  const p = join(ROOT, '.env');
  if (!existsSync(p)) return;
  for (const line of readFileSync(p, 'utf-8').split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '');
  }
}
loadEnv();

export const JUDGES = [
  { id: 'gpt-oss-120b',     provider: 'groq',   model: 'openai/gpt-oss-120b' },
  { id: 'llama-3.1-8b',     provider: 'groq',   model: 'llama-3.1-8b-instant' },   // swapped from llama-3.3-70b (TPD exhausted); much higher daily token budget
  { id: 'gemini-flash-lite', provider: 'gemini', model: process.env.GEMINI_MODEL || 'gemini-flash-lite-latest' },
  { id: 'nemotron-550b',    provider: 'nvidia', model: 'nvidia/nemotron-3-ultra-550b-a55b' },   // capable, credit-based (not TPD)
];

const sleep = ms => new Promise(r => setTimeout(r, ms));

// --- pacing: combined RPM + TPM gate, keyed PER MODEL (free-tier limits) ---
// Groq gpt-oss-120b free tier is the binding constraint: TPM=8000. Gemini flash-lite
// is RPM-bound (15 → we use 12). We track a rolling 60s window of requests+tokens per
// model and wait only as much as needed → never 413/429, "go slow, always finish".
// 2000: enough for the batched 7-arm checklist output (~63 booleans) + gpt-oss low
// reasoning. Note: max_completion counts toward per-request TPM but NOT daily TPD
// (TPD counts actual output), so a higher cap costs nothing on the daily budget.
const MAX_COMPLETION = 2000;
const LIMITS = {
  'openai/gpt-oss-120b':  { rpm: 18, tpm: 7500 },
  'llama-3.1-8b-instant': { rpm: 25, tpm: 5500 },   // high TPD; TPM-paced
  __gemini:               { rpm: 12, tpm: 200000 }, // Gemini: RPM-bound
  'nvidia/nemotron-3-ultra-550b-a55b': { rpm: 35, tpm: 1e9 }, // NVIDIA: 40 RPM cap -> pace 35; credit-based, not token-paced
};
const win = {};   // model -> { reqs:[ts], toks:[{t,n}] }
const estTokens = (system, user) => Math.ceil((system.length + user.length) / 4) + MAX_COMPLETION;
async function gate(model, tokens) {
  const lim = LIMITS[model] || LIMITS.__gemini;
  const s = (win[model] ||= { reqs: [], toks: [] });
  for (let guard = 0; guard < 100; guard++) {
    const now = Date.now(), cut = now - 60000;
    s.reqs = s.reqs.filter(t => t > cut); s.toks = s.toks.filter(x => x.t > cut);
    const usedTok = s.toks.reduce((a, x) => a + x.n, 0);
    const minGap = 60000 / lim.rpm;
    const sinceLast = now - (s.reqs[s.reqs.length - 1] || 0);
    let wait = 0;
    if (sinceLast < minGap) wait = minGap - sinceLast;                 // RPM
    if (usedTok + tokens > lim.tpm && s.toks.length) {                 // TPM: wait for oldest to age out
      wait = Math.max(wait, 60000 - (now - s.toks[0].t) + 250);
    }
    if (wait <= 0) break;
    await sleep(wait);
  }
  const t = Date.now(); s.reqs.push(t); s.toks.push({ t, n: tokens });
}

async function callGroq(model, system, user, json) {
  const key = process.env.GROQ_API_KEY;
  if (!key) throw new Error('GROQ_API_KEY missing in .env');
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    // NOTE: not using Groq's strict response_format:json_object — it hard-fails (400
    // json_validate_failed) on large/reasoned outputs. We instruct JSON in the prompt
    // and extract it ourselves (callJudge), which is far more robust.
    body: JSON.stringify({
      model, temperature: 0, max_completion_tokens: MAX_COMPLETION,
      // gpt-oss is a reasoning model; without capped effort it burns the budget on
      // reasoning and returns empty content on large prompts. 'low' keeps content intact.
      ...(model.includes('gpt-oss') ? { reasoning_effort: 'low' } : {}),
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    }),
  });
  if (!res.ok) { const e = new Error(`groq ${res.status}: ${(await res.text()).slice(0, 200)}`); e.status = res.status; throw e; }
  return (await res.json()).choices[0].message.content;
}

async function callGemini(model, system, user, json) {
  const key = process.env.GOOGLE_API_KEY;
  if (!key) throw new Error('GOOGLE_API_KEY missing in .env');
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;
  const res = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: system }] },
      contents: [{ role: 'user', parts: [{ text: user }] }],
      generationConfig: { temperature: 0, ...(json ? { responseMimeType: 'application/json' } : {}) },
    }),
  });
  if (!res.ok) { const e = new Error(`gemini ${res.status}: ${(await res.text()).slice(0, 200)}`); e.status = res.status; throw e; }
  const d = await res.json();
  return d.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
}

// NVIDIA NIM (integrate.api.nvidia.com) — OpenAI-compatible. enable_thinking:false for
// fast, clean JSON (probe: ~1s, no reasoning leak). Credit-based (no TPD), 40 RPM.
async function callNvidia(model, system, user) {
  const key = process.env.NVIDIA_API_KEY;
  if (!key) throw new Error('NVIDIA_API_KEY missing in .env');
  const res = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({ model, temperature: 0, max_tokens: MAX_COMPLETION,
      chat_template_kwargs: { enable_thinking: false },
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }] }),
  });
  if (!res.ok) { const e = new Error(`nvidia ${res.status}: ${(await res.text()).slice(0, 200)}`); e.status = res.status; throw e; }
  return (await res.json()).choices[0].message.content;
}

// call one judge with retry/backoff; returns text (or parsed object if json=true)
export async function callJudge(judge, { system, user, json = false }, { retries = 5 } = {}) {
  let delay = 2000;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      await gate(judge.provider === 'gemini' ? '__gemini' : judge.model, estTokens(system, user));
      const raw = judge.provider === 'groq' ? await callGroq(judge.model, system, user, json)
        : judge.provider === 'nvidia' ? await callNvidia(judge.model, system, user)
        : await callGemini(judge.model, system, user, json);
      if (!json) return raw;
      const t = raw.trim().replace(/^```json/i, '').replace(/^```/, '').replace(/```$/, '');
      const s = t.indexOf('{'), e = t.lastIndexOf('}');
      return JSON.parse(t.slice(s, e + 1));
    } catch (err) {
      // Only retry TRANSIENT errors (429 rate limit, 5xx). JSON-parse failures are
      // DETERMINISTIC at temp 0 — retrying the same prompt yields the same bad output,
      // so retrying just burns minutes (gate wait + backoff) for nothing. Fail-fast.
      const retriable = err.status === 429 || err.status >= 500;
      if (attempt === retries || !retriable) throw err;
      await sleep(delay + Math.floor((attempt * 500))); delay = Math.min(delay * 2, 30000);
    }
  }
}

// majority vote over an array of binary/label values
export function majority(values) {
  const counts = {};
  for (const v of values) { const k = JSON.stringify(v); counts[k] = (counts[k] || 0) + 1; }
  let best, n = -1;
  for (const [k, c] of Object.entries(counts)) if (c > n) { n = c; best = JSON.parse(k); }
  return best;
}
