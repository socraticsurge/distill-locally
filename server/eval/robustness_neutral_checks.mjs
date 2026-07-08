// Robustness cut for the reasoning-teacher control (local, NO API).
// The frozen summary checklist mixes ARTICLE-GROUNDED checks (faithful/thesis/takeaway)
// with PERSONA/STYLE checks (tone/teacher-lens/tech-lens/opening/length) that encode the
// teacher's target voice. Concern: the +22pp R1-vs-Llama gap could be partly "R1 student
// matches the R1-shaped rubric." This recomputes the gap on the teacher-NEUTRAL subset.
//
// Reads the per-item per-check detail dumped by score.mjs (DETAIL_OUT), so the numbers are
// exactly the frozen 2-judge majorities behind the scorecard — no prompt reconstruction.
//   Build it: JUDGES_EXCLUDE=gpt-oss-120b,llama-3.1-8b ARTICLE_CHARS=100000 ONLY_CHECKLIST=1 \
//             DETAIL_OUT=checklist_detail_11arm.json node --env-file=.env server/eval/score.mjs
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const file = process.argv[2] || 'data/eval/scores_detail.json';
const D = JSON.parse(readFileSync(join(ROOT, file), 'utf-8'));

const ALL = ['faithful', 'thesis', 'takeaway', 'length', 'opening', 'teacher', 'tech', 'tone'];
const NEUTRAL = ['faithful', 'thesis', 'takeaway'];   // article-grounded, persona-free
const STYLE = ['length', 'opening', 'teacher', 'tech', 'tone'];
const R1 = ['tuned_rss_tuned_s1', 'tuned_rss_tuned_s2', 'tuned_rss_tuned_s3'];
const LL = ['tuned_rss_llama_s1', 'tuned_rss_llama_s2', 'tuned_rss_llama_s3'];

const IDS = Object.keys(D.teacher);   // stable item order (keys aligned across arms)
const N = IDS.length;
const mean = xs => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
// per-item score for an arm over a subset = fraction of the subset's checks that pass
const subsetPerItem = (arm, subset) => IDS.map(id => mean(subset.map(k => D[arm][id][k])));
const poolPerItem = (seeds, subset) => IDS.map((id, i) => mean(seeds.map(a => subsetPerItem(a, subset)[i])));
const poolRate = (seeds, subset) => mean(poolPerItem(seeds, subset)) * 100;

const B = 20000;
function bootstrap(aArr, bArr) {
  const d = aArr.map((x, i) => x - bArr[i]);
  const obs = mean(d), means = new Array(B);
  for (let b = 0; b < B; b++) { let s = 0; for (let i = 0; i < d.length; i++) s += d[(Math.random() * d.length) | 0]; means[b] = s / d.length; }
  means.sort((x, y) => x - y);
  const lo = means[Math.floor(0.025 * B)], hi = means[Math.floor(0.975 * B)];
  const below = means.filter(m => m <= 0).length / B; const p = 2 * Math.min(below, 1 - below);
  return { obs: obs * 100, lo: lo * 100, hi: hi * 100, p, sig: (lo > 0 || hi < 0) };
}
const fmt = r => `Δ=${r.obs >= 0 ? '+' : ''}${r.obs.toFixed(1)}pp  95% CI [${r.lo.toFixed(1)}, ${r.hi.toFixed(1)}]  p=${r.p < 0.001 ? '<0.001' : r.p.toFixed(3)}  ${r.sig ? 'SIGNIFICANT' : 'n.s.'}`;

console.log(`Teacher-neutral robustness cut  (N=${N}, B=${B})   source: ${file}\n`);
console.log(`SELF-CHECK — reconstructed ALL-8-check pass-rate (should match scorecard 72.6 / 50.8):`);
console.log(`  R1-pooled ${poolRate(R1, ALL).toFixed(1)}  |  Llama-pooled ${poolRate(LL, ALL).toFixed(1)}\n`);

console.log(`Per-check pass-rate, R1-pooled vs Llama-pooled (where the +22 lives):`);
console.log(`  check        R1     Llama     Δ      kind`);
for (const k of ALL) {
  const r = poolRate(R1, [k]), l = poolRate(LL, [k]);
  console.log(`  ${k.padEnd(10)} ${r.toFixed(1).padStart(5)}  ${l.toFixed(1).padStart(5)}  ${((r - l >= 0 ? '+' : '') + (r - l).toFixed(1)).padStart(6)}   ${NEUTRAL.includes(k) ? 'NEUTRAL' : 'style'}`);
}

console.log(`\n=== TEACHER-NEUTRAL subset {${NEUTRAL.join(', ')}} ===`);
console.log(`  pass-rates: R1 ${poolRate(R1, NEUTRAL).toFixed(1)} | Llama ${poolRate(LL, NEUTRAL).toFixed(1)} | base ${poolRate(['base'], NEUTRAL).toFixed(1)} | teacher ${poolRate(['teacher'], NEUTRAL).toFixed(1)}`);
console.log(`  R1 vs Llama:    ${fmt(bootstrap(poolPerItem(R1, NEUTRAL), poolPerItem(LL, NEUTRAL)))}`);
console.log(`  R1 vs base:     ${fmt(bootstrap(poolPerItem(R1, NEUTRAL), subsetPerItem('base', NEUTRAL)))}`);
console.log(`  Llama vs base:  ${fmt(bootstrap(poolPerItem(LL, NEUTRAL), subsetPerItem('base', NEUTRAL)))}`);

console.log(`\n=== faithfulness ALONE {faithful} — the hallucination check ===`);
console.log(`  pass-rates: R1 ${poolRate(R1, ['faithful']).toFixed(1)} | Llama ${poolRate(LL, ['faithful']).toFixed(1)} | base ${poolRate(['base'], ['faithful']).toFixed(1)}`);
console.log(`  R1 vs Llama:    ${fmt(bootstrap(poolPerItem(R1, ['faithful']), poolPerItem(LL, ['faithful'])))}`);

console.log(`\n=== PERSONA/STYLE subset {${STYLE.join(', ')}} (for contrast) ===`);
console.log(`  pass-rates: R1 ${poolRate(R1, STYLE).toFixed(1)} | Llama ${poolRate(LL, STYLE).toFixed(1)}`);
console.log(`  R1 vs Llama:    ${fmt(bootstrap(poolPerItem(R1, STYLE), poolPerItem(LL, STYLE)))}`);
