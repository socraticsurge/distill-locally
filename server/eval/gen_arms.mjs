// Generate arm outputs on the eval test set via Ollama, at temperature 0 (greedy).
// Arms: base | fewshot | constrained | tuned <ollama-model>.  (teacher outputs
// already live in test.jsonl.)  Resumable: appends per item, skips done ids.
// Usage:
//   node server/eval/gen_arms.mjs base
//   node server/eval/gen_arms.mjs fewshot
//   node server/eval/gen_arms.mjs constrained
//   node server/eval/gen_arms.mjs tuned rss-tuned-s42
// Optional smoke cap: EVAL_LIMIT=2 node server/eval/gen_arms.mjs base
import { readFileSync, writeFileSync, existsSync, appendFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const OLLAMA = 'http://localhost:11434/api/chat';
const arm = process.argv[2];
const tunedModel = process.argv[3];
const LIMIT = process.env.EVAL_LIMIT ? Number(process.env.EVAL_LIMIT) : Infinity;
if (!['base', 'fewshot', 'constrained', 'tuned'].includes(arm)) { console.error('arm must be base|fewshot|constrained|tuned'); process.exit(1); }
if (arm === 'tuned' && !tunedModel) { console.error('tuned needs an ollama model name'); process.exit(1); }

const SYSTEM_PROMPT = readFileSync(join(ROOT, 'data/eval/system_prompt.txt'), 'utf-8');
const SET = process.env.EVAL_SET || 'test';   // 'test' (scored) | 'pilot' (train subset)
const test = readFileSync(join(ROOT, `data/eval/${SET}.jsonl`), 'utf-8').split('\n').filter(Boolean).map(l => JSON.parse(l));

const MODEL = arm === 'tuned' ? tunedModel : 'qwen3:0.6b';
const label = arm === 'tuned' ? `tuned_${tunedModel.replace(/[^a-z0-9]+/gi, '_')}` : arm;
const OUT = join(ROOT, SET === 'test' ? `data/eval/arm_${label}.jsonl` : `data/eval/arm_${SET}_${label}.jsonl`);

// few-shot demos (2) drawn from the TRAIN split
const demos = arm === 'fewshot'
  ? readFileSync(join(ROOT, 'data/eval/train_chat.jsonl'), 'utf-8').split('\n').filter(Boolean).slice(0, 2)
      .map(l => JSON.parse(l).messages).flatMap(m => [m.find(x => x.role === 'user'), m.find(x => x.role === 'assistant')])
  : [];

// JSON schema for constrained decoding
const schema = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    sentiment: { type: 'string', enum: ['positive', 'neutral', 'negative'] },
    urgency: { type: 'string', enum: ['breaking', 'developing', 'evergreen'] },
    frame: { type: 'string', enum: ['conflict', 'human_interest', 'economic', 'analytical'] },
    tone: { type: 'string', enum: ['alarming', 'analytical', 'optimistic', 'opinion'] },
    depth: { type: 'string', enum: ['brief', 'standard', 'deep_dive'] },
    topics: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'sentiment', 'urgency', 'frame', 'tone', 'depth', 'topics'],
};

const done = new Set();
if (existsSync(OUT)) for (const l of readFileSync(OUT, 'utf-8').split('\n')) { if (l.trim()) try { done.add(JSON.parse(l).id); } catch {} }
const todo = test.filter(t => !done.has(t.id)).slice(0, LIMIT);
console.log(`arm=${label} model=${MODEL} | done ${done.size} | to generate ${todo.length}`);

for (let i = 0; i < todo.length; i++) {
  const t = todo[i];
  const messages = [{ role: 'system', content: SYSTEM_PROMPT }, ...demos, { role: 'user', content: t.input }];
  try {
    const t0 = Date.now();
    const res = await fetch(OLLAMA, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: MODEL, stream: false, think: false, options: { temperature: 0 },
        messages, ...(arm === 'constrained' ? { format: schema } : {}) }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    const out = d.message.content || d.message.thinking || '';
    appendFileSync(OUT, JSON.stringify({ id: t.id, feed: t.feed, arm: label, output: out,
      durationMs: d.total_duration ? Math.round(d.total_duration / 1e6) : (Date.now() - t0),
      evalCount: d.eval_count || 0 }) + '\n');
    console.log(`[${i + 1}/${todo.length}] id=${t.id} ${d.eval_count || 0}tok`);
  } catch (e) { console.log(`[FAIL] id=${t.id} ${e.message}`); }
}
console.log(`-> ${OUT}`);
