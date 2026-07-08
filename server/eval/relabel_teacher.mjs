// Re-label the 401 TRAIN articles with a different (non-reasoning) teacher, to
// build a control training set. Holds the labeling procedure constant with the
// original R1 run (same system+user chat, temperature 0, free-form JSON per the
// system prompt) and changes only the teacher model.
//
// Usage:
//   node server/eval/relabel_teacher.mjs <ollama-model> <suffix>
//   node server/eval/relabel_teacher.mjs llama3.1:8b-instruct-q4_K_M llama
// Output: data/eval/train_chat_<suffix>.jsonl  (same {messages:[...]} format as
// train_chat.jsonl, drop-in for the Unsloth notebook) + a .report.json yield log.
// Resumable: appends per item, skips already-written records by index.
import { readFileSync, writeFileSync, existsSync, appendFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const OLLAMA = 'http://localhost:11434/api/chat';
const MODEL = process.argv[2];
const SUFFIX = process.argv[3];
if (!MODEL || !SUFFIX) { console.error('usage: relabel_teacher.mjs <ollama-model> <suffix>'); process.exit(1); }
const LIMIT = process.env.EVAL_LIMIT ? Number(process.env.EVAL_LIMIT) : Infinity;

const REQUIRED = ['summary', 'sentiment', 'urgency', 'frame', 'tone', 'depth', 'topics'];
const src = readFileSync(join(ROOT, 'data/eval/train_chat.jsonl'), 'utf-8').split('\n').filter(Boolean).map(l => JSON.parse(l));
const OUT = join(ROOT, `data/eval/train_chat_${SUFFIX}.jsonl`);
const REPORT = join(ROOT, `data/eval/train_chat_${SUFFIX}.report.json`);

// resume: how many records already written
let doneCount = 0;
if (existsSync(OUT)) doneCount = readFileSync(OUT, 'utf-8').split('\n').filter(Boolean).length;
const todo = src.slice(doneCount, doneCount + LIMIT === Infinity ? undefined : doneCount + LIMIT);
console.log(`model=${MODEL} suffix=${SUFFIX} | total ${src.length} | already ${doneCount} | to label ${todo.length}`);

// pull first {...} balanced JSON object out of a model response (strips ``` fences, prose)
function extractJson(text) {
  if (!text) return null;
  let s = text.replace(/```json/gi, '```').replace(/```/g, '').trim();
  const start = s.indexOf('{');
  if (start < 0) return null;
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (inStr) { if (esc) esc = false; else if (c === '\\') esc = true; else if (c === '"') inStr = false; }
    else if (c === '"') inStr = true;
    else if (c === '{') depth++;
    else if (c === '}') { depth--; if (depth === 0) { try { return JSON.parse(s.slice(start, i + 1)); } catch { return null; } } }
  }
  return null;
}

let ok = 0, bad = 0; const failures = [];
if (existsSync(REPORT)) { const r = JSON.parse(readFileSync(REPORT, 'utf-8')); ok = r.ok || 0; bad = r.bad || 0; }

for (let i = 0; i < todo.length; i++) {
  const idx = doneCount + i;
  const rec = todo[i];
  const sys = rec.messages.find(m => m.role === 'system');
  const usr = rec.messages.find(m => m.role === 'user');
  let assistantContent = null, raw = '';
  try {
    const res = await fetch(OLLAMA, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: MODEL, stream: false, think: false,
        options: { temperature: 0, num_ctx: 8192 },
        messages: [{ role: 'system', content: sys.content }, { role: 'user', content: usr.content }] }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    raw = d.message?.content || '';
    const obj = extractJson(raw);
    if (obj && REQUIRED.every(k => k in obj)) { assistantContent = JSON.stringify(obj, null, 2); ok++; }
    else { bad++; failures.push({ idx, reason: obj ? 'missing-fields' : 'no-json', head: raw.slice(0, 120) }); }
  } catch (e) { bad++; failures.push({ idx, reason: e.message, head: raw.slice(0, 120) }); }

  // Write a record for every item; assistant is null on failure (dropped at train-file build).
  appendFileSync(OUT, JSON.stringify({ idx, ok: assistantContent !== null,
    messages: assistantContent !== null
      ? [{ role: 'system', content: sys.content }, { role: 'user', content: usr.content }, { role: 'assistant', content: '\n' + assistantContent }]
      : null,
    raw: assistantContent === null ? raw.slice(0, 300) : undefined }) + '\n');
  writeFileSync(REPORT, JSON.stringify({ model: MODEL, suffix: SUFFIX, total: src.length, ok, bad,
    yield: +(ok / (ok + bad) * 100).toFixed(1), failures: failures.slice(-40) }, null, 2));
  if ((i + 1) % 20 === 0 || i === todo.length - 1) console.log(`[${idx + 1}/${src.length}] ok=${ok} bad=${bad}`);
}
console.log(`-> ${OUT}  (ok=${ok} bad=${bad}, yield ${(ok / (ok + bad) * 100).toFixed(1)}%)`);
console.log(`Note: build the Unsloth train file by keeping only records with ok=true and taking .messages.`);
