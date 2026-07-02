// Attribution / audit trail for judge_cache.jsonl.
// The cache is content-addressed (k = sha1(judge.id + ' ' + system + ' ' + user)), so it's
// anonymous. This standalone tool RE-DERIVES every (judge x article x phase x arm) prompt
// exactly as score.mjs builds them, re-hashes, and matches back to the cache — giving a
// readable table of which judge said what, for which article, in which phase. Read-only:
// it does NOT call any API and does NOT touch the running scored job.
//
// Run AFTER (or during) the scored run:  node server/eval/attribute.mjs
// Prompt-building below is a VERBATIM copy of score.mjs (kept in sync); the match-rate
// printed at the end is the self-check that the copy is exact.
import { readFileSync, writeFileSync, existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createHash } from 'crypto';
import { JUDGES } from './judges.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const D = join(ROOT, 'data/eval');
const jl = p => existsSync(p) ? readFileSync(p, 'utf-8').split('\n').filter(Boolean).map(l => JSON.parse(l)) : [];

// ---- verbatim from score.mjs ----
const CAT_FIELDS = ['sentiment', 'urgency', 'frame', 'tone', 'depth'];
const CHECKLIST = [
  ['faithful', 'Every factual claim in the summary is supported by the article (no hallucination/contradiction).'],
  ['thesis', 'The summary captures the article\'s central thesis or main finding.'],
  ['takeaway', 'The summary includes a concrete, specific takeaway (not vague/generic).'],
  ['length', 'The summary is 3 to 4 sentences.'],
  ['opening', 'The summary does NOT begin with "This article" or "The article".'],
  ['teacher', 'Teacher lens: explains a concept accessibly or builds intuition.'],
  ['tech', 'Technologist lens: addresses the technical/engineering angle.'],
  ['tone', 'Tone is direct + contemplative + optimistic (not alarmist or generic).'],
];
const EXTRA = ['topics_cover'];
const hkey = s => createHash('sha1').update(s).digest('hex');
const keyOf = (judge, system, user) => createHash('sha1').update(judge.id + ' ' + system + ' ' + user).digest('hex');
function parse(t) { if (!t) return null; const s = t.indexOf('{'), e = t.lastIndexOf('}'); if (s < 0 || e < 0) return null; try { return JSON.parse(t.slice(s, e + 1)); } catch { return null; } }
function checklistPrompt(article, cands) {
  return `ARTICLE:\n${article.slice(0, 6000)}\n\nGrade EACH labeled candidate against the ARTICLE only.\n\nCHECKS (per candidate, true/false):\n` +
    CHECKLIST.map(([k, d]) => `- ${k}: ${d}`).join('\n') +
    `\n- topics_cover: the candidate's listed topics capture the article's main themes.\n\nCANDIDATES:\n` +
    cands.map(c => `[${c.label}] summary: ${c.summary}\n[${c.label}] topics: ${JSON.stringify(c.topics)}`).join('\n\n') +
    `\n\nReturn ONLY JSON keyed by label, each an object of the booleans above. Example: {"${cands[0]?.label}":{${[...CHECKLIST.map(([k]) => `"${k}":true`), ...EXTRA.map(k => `"${k}":true`)].join(',')}}}`;
}
const SYS_CHECK = 'You are a strict grader. Judge each labeled summary against the article only. Output strict JSON keyed by label.';
const SYS_CLASS = 'You label a news article. Choose exactly one option per field. Output strict JSON.';
const OPTS = { sentiment: 'positive|neutral|negative', urgency: 'breaking|developing|evergreen', frame: 'conflict|human_interest|economic|analytical', tone: 'alarming|analytical|optimistic|opinion', depth: 'brief|standard|deep_dive' };
function classPrompt(input) {
  return `ARTICLE:\n${input.slice(0, 6000)}\n\nLabel these fields:\n` +
    CAT_FIELDS.map(f => `- ${f}: one of ${OPTS[f]}`).join('\n') + `\n\nReturn ONLY JSON: {${CAT_FIELDS.map(f => `"${f}":"..."`).join(',')}}`;
}

// ---- load data exactly as score.mjs (test set) ----
const test = jl(join(D, 'test.jsonl'));
const arms = { teacher: new Map(test.map(t => [t.id, t.teacher_output])) };
for (const label of ['base', 'fewshot', 'constrained']) { const m = new Map(jl(join(D, `arm_${label}.jsonl`)).map(r => [r.id, r.output])); if (m.size) arms[label] = m; }
for (const f of readdirSync(D).filter(f => f.startsWith('arm_tuned_'))) { const label = f.slice('arm_'.length).replace(/\.jsonl$/, ''); arms[label] = new Map(jl(join(D, f)).map(r => [r.id, r.output])); }
const armNames = Object.keys(arms);

const cache = new Map();
for (const r of jl(join(D, 'judge_cache.jsonl'))) cache.set(r.k, r.v);

const rows = [];
let matched = 0;
function tryMatch(judge, system, user, meta) {
  const k = keyOf(judge, system, user);
  if (cache.has(k)) { matched++; rows.push({ ...meta, judge: judge.id, v: cache.get(k) }); return true; }
  return false;
}

// classification: one call per (article, judge)
for (const t of test) for (const judge of JUDGES) tryMatch(judge, SYS_CLASS, classPrompt(t.input), { phase: 'classification', articleId: t.id, feed: t.feed });

// checklist: one call per (article, judge), all arms graded under shuffled labels
for (const t of test) {
  const ordered = armNames.filter(a => arms[a].get(t.id)).slice().sort((x, y) => hkey(t.id + x) < hkey(t.id + y) ? -1 : 1);
  const cands = ordered.map((arm, i) => { const p = parse(arms[arm].get(t.id)); return { arm, label: String.fromCharCode(65 + i), summary: p?.summary ?? arms[arm].get(t.id), topics: p?.topics ?? [] }; });
  const labelMap = Object.fromEntries(cands.map(c => [c.label, c.arm]));
  for (const judge of JUDGES) tryMatch(judge, SYS_CHECK, checklistPrompt(t.input, cands), { phase: 'checklist', articleId: t.id, feed: t.feed, labelToArm: labelMap });
}

// negative control: teacher summary vs mismatched article (sample 30)
const items = test.slice(0, Math.min(30, test.length));
for (let i = 0; i < items.length; i++) {
  const t = items[i], wrong = items[(i + 1) % items.length];
  const p = parse(arms.teacher.get(t.id)); if (!p?.summary) continue;
  const cand = [{ label: 'A', summary: p.summary, topics: p.topics ?? [] }];
  for (const judge of JUDGES) tryMatch(judge, SYS_CHECK, checklistPrompt(wrong.input, cand), { phase: 'negcontrol', articleId: t.id, mismatchedWith: wrong.id });
}

writeFileSync(join(D, 'attribution.jsonl'), rows.map(r => JSON.stringify(r)).join('\n') + '\n');
const byPhase = {}; for (const r of rows) byPhase[r.phase] = (byPhase[r.phase] || 0) + 1;
console.log(`cache entries: ${cache.size} | attributed (test-set, current prompts): ${matched}`);
console.log(`by phase:`, byPhase);
console.log(`unmatched cache entries: ${cache.size - matched}  (expected: pilot(train) + old-checklist + smoke leftovers)`);
console.log(`-> data/eval/attribution.jsonl  (judge / article / feed / phase / arm-label map per response)`);
