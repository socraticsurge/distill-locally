// Build a self-contained local inspector of the whole eval, from the FROZEN
// canonical artifacts (no API calls, no judge-cache replay):
//   - generations:  arm_*.jsonl  (+ teacher output in test.jsonl)
//   - grades:       scores_detail.json  (3-judge panel MAJORITY per check, per arm, per item)
//   - aggregates:   scores.json, detail_analyses.json, router_composite.json
//
//   node server/eval/build_viewer.mjs   ->  data/eval/viewer_data.json
// then: node server/eval/build_viewer_html.mjs  ->  data/eval/viewer.html
import { readFileSync, writeFileSync, existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { JUDGES } from './judges.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const D = join(ROOT, 'data/eval');
const REQUIRED = ['summary', 'sentiment', 'urgency', 'frame', 'tone', 'depth', 'topics'];
const CAT_FIELDS = ['sentiment', 'urgency', 'frame', 'tone', 'depth'];
const CHECKS = ['faithful', 'thesis', 'takeaway', 'length', 'opening', 'teacher', 'tech', 'tone', 'topics_cover'];
const CHECK_DESC = {
  faithful: 'Every factual claim is supported by the article (no hallucination).',
  thesis: 'Captures the article\'s central thesis / main finding.',
  takeaway: 'Includes a concrete, specific takeaway (not vague).',
  length: 'Summary is 3 to 4 sentences.',
  opening: 'Does NOT begin with "This/The article".',
  teacher: 'Teacher lens: explains a concept accessibly.',
  tech: 'Technologist lens: addresses the technical angle.',
  tone: 'Tone is direct, contemplative, not alarmist.',
  topics_cover: 'Listed topics capture the article\'s main themes.',
};

const jl = p => existsSync(p) ? readFileSync(p, 'utf-8').split('\n').filter(Boolean).map(l => JSON.parse(l)) : [];
const rj = p => JSON.parse(readFileSync(join(D, p), 'utf-8'));
const test = jl(join(D, 'test.jsonl'));

// generations by arm: id -> record
function loadArm(label) { const m = new Map(); for (const r of jl(join(D, `arm_${label}.jsonl`))) m.set(r.id, r); return m; }
const armRec = { teacher: new Map(test.map(t => [t.id, { output: t.teacher_output }])) };
for (const label of ['teacher_llama', 'distil']) { const m = loadArm(label); if (m.size) armRec[label] = m; }
for (const fam of ['tuned', 'llama']) for (const s of ['s1', 's2', 's3']) { const k = `tuned_rss_${fam}_${s}`; const m = loadArm(k); if (m.size) armRec[k] = m; }
for (const label of ['base', 'fewshot', 'constrained']) { const m = loadArm(label); if (m.size) armRec[label] = m; }
const armKeys = Object.keys(armRec);   // this is also the display order

const detail = rj('scores_detail.json');          // detail[arm][id][check] = 0/1 (panel majority)
const scores = rj('scores.json');
const analytics = {
  scores,
  detail: rj('detail_analyses.json'),
  router: existsSync(join(D, 'router_composite.json')) ? rj('router_composite.json') : null,
};

function parse(t) { if (!t) return null; const s = t.indexOf('{'), e = t.lastIndexOf('}'); if (s < 0 || e < 0) return null; try { return JSON.parse(t.slice(s, e + 1)); } catch { return null; } }
const SEED = { s1: 42, s2: 123, s3: 7 };
function armMeta(key) {
  if (key === 'teacher') return { label: 'R1-8B teacher', kind: 'teacher', note: 'reasoning (deepseek-r1:8b)' };
  if (key === 'teacher_llama') return { label: 'Llama-8B teacher', kind: 'teacher', note: 'non-reasoning (llama-3.1-8b-instruct)' };
  if (key === 'distil') return { label: 'Managed pipeline', kind: 'student', note: 'gpt-oss-120b + synthetic expansion' };
  if (key === 'base') return { label: 'Base', kind: 'baseline', note: 'Qwen3-0.6B zero-shot' };
  if (key === 'fewshot') return { label: 'Base + few-shot', kind: 'baseline', note: '2-3 in-context examples' };
  if (key === 'constrained') return { label: 'Base + constrained', kind: 'baseline', note: 'JSON-constrained decoding' };
  const m = key.match(/^tuned_rss_(tuned|llama)_(s\d)$/);
  if (m) { const fam = m[1] === 'tuned' ? 'R1' : 'Llama'; return { label: `${fam} student · seed ${SEED[m[2]] ?? m[2]}`, kind: 'student', note: `distilled from ${fam}-8B teacher` }; }
  return { label: key, kind: 'other', note: '' };
}

const articles = [];
for (const t of test) {
  const armsOut = {}, checklist = {}, classLabels = {};
  for (const a of armKeys) {
    const rec = armRec[a].get(t.id); if (!rec) continue;
    const p = parse(rec.output);
    armsOut[a] = {
      summary: p?.summary ?? null,
      labels: p ? Object.fromEntries(CAT_FIELDS.map(f => [f, p[f] ?? null])) : {},
      topics: p?.topics ?? [], raw: rec.output,
      durationMs: rec.durationMs ?? null, evalCount: rec.evalCount ?? null,
      schemaValid: !!(p && REQUIRED.every(k => k in p)),
    };
    classLabels[a] = armsOut[a].labels;
    const dg = detail[a]?.[t.id] ?? detail[a]?.[String(t.id)] ?? null;   // panel-majority grades
    if (dg) checklist[a] = Object.fromEntries(CHECKS.map(c => [c, dg[c] ?? null]));
  }
  const tp = parse(t.teacher_output);
  const teacherLabels = tp ? Object.fromEntries(CAT_FIELDS.map(f => [f, tp[f] ?? null])) : {};
  articles.push({
    id: t.id, feed: t.feed, title: t.title,
    chars: (t.input.match(/Content:\s*([\s\S]*)/)?.[1] ?? t.input).length,
    input: t.input, arms: armsOut, checklist, classLabels, teacherLabels,
  });
}

const meta = {
  builtAtNote: 'grades are the 3-judge panel MAJORITY (frozen canonical scorecard); per-individual-judge votes are not stored here',
  judges: JUDGES.map(j => ({ id: j.id, provider: j.provider, model: j.model })),
  checks: CHECKS.map(c => [c, CHECK_DESC[c]]),
  catFields: CAT_FIELDS,
  arms: armKeys.map(k => ({ key: k, ...armMeta(k) })),
};
writeFileSync(join(D, 'viewer_data.json'), JSON.stringify({ meta, articles, analytics }));
console.log(`-> data/eval/viewer_data.json  (${articles.length} articles, ${armKeys.length} arms)`);
// coverage sanity
const miss = [];
for (const a of armKeys) { const n = articles.filter(x => x.checklist[a]).length; if (n !== articles.length) miss.push(`${a}:${n}/${articles.length}`); }
console.log(miss.length ? `checklist grade coverage gaps: ${miss.join(', ')}` : `checklist grades present for all arms x all ${articles.length} articles`);
