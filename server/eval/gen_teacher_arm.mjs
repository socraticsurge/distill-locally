// Generate a TEACHER's own outputs on the 93-item test set, so each distillation
// teacher has a self-score graded under the same rubric as its students.
// Same system prompt + article as every other arm, temperature 0. Resumable.
//   node server/eval/gen_teacher_arm.mjs teacher_llama  ollama llama3.1:8b-instruct-q4_K_M
//   node --env-file=.env server/eval/gen_teacher_arm.mjs teacher_gptoss groq openai/gpt-oss-120b
import { readFileSync, writeFileSync, existsSync, appendFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { callJudge } from './judges.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const [label, provider, model] = process.argv.slice(2);
if (!label || !provider || !model) { console.error('usage: gen_teacher_arm.mjs <label> <ollama|groq> <model>'); process.exit(1); }
const LIMIT = process.env.EVAL_LIMIT ? Number(process.env.EVAL_LIMIT) : Infinity;

const SYSTEM_PROMPT = readFileSync(join(ROOT, 'data/eval/system_prompt.txt'), 'utf-8');
const test = readFileSync(join(ROOT, 'data/eval/test.jsonl'), 'utf-8').split('\n').filter(Boolean).map(l => JSON.parse(l));
const OUT = join(ROOT, `data/eval/arm_${label}.jsonl`);

const done = new Set();
if (existsSync(OUT)) for (const l of readFileSync(OUT, 'utf-8').split('\n')) { if (l.trim()) try { done.add(JSON.parse(l).id); } catch {} }
const todo = test.filter(t => !done.has(t.id)).slice(0, LIMIT);
console.log(`arm=${label} provider=${provider} model=${model} | done ${done.size} | to generate ${todo.length}`);

async function ollama(system, user) {
  const res = await fetch('http://localhost:11434/api/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, stream: false, think: false, options: { temperature: 0, num_ctx: 8192 },
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }] }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()).message.content || '';
}

for (let i = 0; i < todo.length; i++) {
  const t = todo[i];
  try {
    const t0 = Date.now();
    const out = provider === 'ollama'
      ? await ollama(SYSTEM_PROMPT, t.input)
      : await callJudge({ id: `${label}-arm`, provider, model }, { system: SYSTEM_PROMPT, user: t.input, json: false });
    appendFileSync(OUT, JSON.stringify({ id: t.id, feed: t.feed, arm: label, output: out, durationMs: Date.now() - t0 }) + '\n');
    if ((i + 1) % 10 === 0 || i === todo.length - 1) console.log(`[${i + 1}/${todo.length}] id=${t.id}`);
  } catch (e) { console.log(`[FAIL] id=${t.id} ${e.message}`); }
}
console.log(`-> ${OUT}`);
