// Paired bootstrap on the per-item summary-checklist scores (local, no API).
// Reads the compiled 2-judge scorecard and runs the pre-registered comparisons:
//   PRIMARY:   tuned vs base+constrained
//   SECONDARY: tuned vs base, tuned vs few-shot, tuned vs teacher
// Tuned is reported per-seed AND pooled (per-item mean across the 3 seeds).
// 95% CI + two-sided bootstrap p on the mean per-item difference over the 93 test items.
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const file = process.argv[2] || 'data/eval/scores.json';
const S = JSON.parse(readFileSync(join(ROOT, file), 'utf-8'));
const CL = S.checklist;

const seeds = ['tuned_rss_tuned_s1', 'tuned_rss_tuned_s2', 'tuned_rss_tuned_s3'];
const per = a => CL[a].perItem;
const N = per('teacher').length;
// pooled tuned = per-item mean across the 3 seeds
const tunedPooled = Array.from({ length: N }, (_, i) =>
  seeds.reduce((s, a) => s + per(a)[i], 0) / seeds.length);

const B = 20000;
function bootstrap(aArr, bArr) {
  const d = aArr.map((x, i) => x - bArr[i]);
  const obs = d.reduce((s, x) => s + x, 0) / d.length;
  const means = new Array(B);
  for (let b = 0; b < B; b++) {
    let s = 0;
    for (let i = 0; i < d.length; i++) s += d[(Math.random() * d.length) | 0];
    means[b] = s / d.length;
  }
  means.sort((x, y) => x - y);
  const lo = means[Math.floor(0.025 * B)], hi = means[Math.floor(0.975 * B)];
  // two-sided bootstrap p: 2× the tail mass on the far side of 0
  const below = means.filter(m => m <= 0).length / B;
  const p = 2 * Math.min(below, 1 - below);
  return { obs: obs * 100, lo: lo * 100, hi: hi * 100, p, sig: (lo > 0 || hi < 0) };
}

const fmt = r => `Δ=${r.obs >= 0 ? '+' : ''}${r.obs.toFixed(1)}pp  95% CI [${r.lo.toFixed(1)}, ${r.hi.toFixed(1)}]  p=${r.p < 0.001 ? '<0.001' : r.p.toFixed(3)}  ${r.sig ? 'SIGNIFICANT' : 'n.s.'}`;

const rate = a => (a === 'pooled' ? (tunedPooled.reduce((s, x) => s + x, 0) / N * 100) : CL[a].checklistPassRate);
const arr = a => (a === 'pooled' ? tunedPooled : per(a));

console.log(`Paired bootstrap on summary-checklist per-item scores  (N=${N} items, B=${B} resamples)`);
console.log(`Source: ${file}\n`);
console.log(`Pass-rates: teacher ${rate('teacher').toFixed(1)} | base ${rate('base').toFixed(1)} | fewshot ${rate('fewshot').toFixed(1)} | constrained ${rate('constrained').toFixed(1)} | tuned-pooled ${rate('pooled').toFixed(1)}`);
console.log(`Tuned seeds: s1 ${rate(seeds[0]).toFixed(1)} | s2 ${rate(seeds[1]).toFixed(1)} | s3 ${rate(seeds[2]).toFixed(1)}\n`);

const comparisons = [
  ['PRIMARY  tuned(pooled) vs constrained', 'pooled', 'constrained'],
  ['         tuned-s1      vs constrained', seeds[0], 'constrained'],
  ['         tuned-s2      vs constrained', seeds[1], 'constrained'],
  ['         tuned-s3      vs constrained', seeds[2], 'constrained'],
  ['SECOND.  tuned(pooled) vs base',        'pooled', 'base'],
  ['SECOND.  tuned(pooled) vs fewshot',     'pooled', 'fewshot'],
  ['SECOND.  tuned(pooled) vs teacher',     'pooled', 'teacher'],
  ['ref      fewshot       vs constrained', 'fewshot', 'constrained'],
];
for (const [label, a, b] of comparisons) {
  console.log(`${label}\n   ${fmt(bootstrap(arr(a), arr(b)))}`);
}
