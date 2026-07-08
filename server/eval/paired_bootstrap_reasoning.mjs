// Paired bootstrap for the REASONING-vs-NON-REASONING teacher control (local, no API).
// Same method as paired_bootstrap.mjs, but the contrast is the distillation teacher:
//   R1-distilled student  (DeepSeek-R1:8b, reasoning)      = tuned_rss_tuned_s{1,2,3}
//   Llama-distilled student (Llama-3.1-8B-Instruct, plain) = tuned_rss_llama_s{1,2,3}
// Identical recipe/seeds/split; ONLY the teacher differs. Tests whether the paper's
// "reasoning teacher" claim is actually supported on the summary sub-task.
// Metric = per-item summary-checklist score over the 93 test items.
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const file = process.argv[2] || 'data/eval/scores.json';
const S = JSON.parse(readFileSync(join(ROOT, file), 'utf-8'));
const CL = S.checklist;

const R1 = ['tuned_rss_tuned_s1', 'tuned_rss_tuned_s2', 'tuned_rss_tuned_s3'];
const LL = ['tuned_rss_llama_s1', 'tuned_rss_llama_s2', 'tuned_rss_llama_s3'];
const per = a => CL[a].perItem;
const N = per('teacher').length;
const pool = seeds => Array.from({ length: N }, (_, i) => seeds.reduce((s, a) => s + per(a)[i], 0) / seeds.length);
const r1Pooled = pool(R1), llPooled = pool(LL);

const B = 20000;
function bootstrap(aArr, bArr) {
  const d = aArr.map((x, i) => x - bArr[i]);
  const obs = d.reduce((s, x) => s + x, 0) / d.length;
  const means = new Array(B);
  for (let b = 0; b < B; b++) { let s = 0; for (let i = 0; i < d.length; i++) s += d[(Math.random() * d.length) | 0]; means[b] = s / d.length; }
  means.sort((x, y) => x - y);
  const lo = means[Math.floor(0.025 * B)], hi = means[Math.floor(0.975 * B)];
  const below = means.filter(m => m <= 0).length / B;
  const p = 2 * Math.min(below, 1 - below);
  return { obs: obs * 100, lo: lo * 100, hi: hi * 100, p, sig: (lo > 0 || hi < 0) };
}
const fmt = r => `Δ=${r.obs >= 0 ? '+' : ''}${r.obs.toFixed(1)}pp  95% CI [${r.lo.toFixed(1)}, ${r.hi.toFixed(1)}]  p=${r.p < 0.001 ? '<0.001' : r.p.toFixed(3)}  ${r.sig ? 'SIGNIFICANT' : 'n.s.'}`;
const arr = a => a === 'R1' ? r1Pooled : a === 'LL' ? llPooled : per(a);
const rate = a => a === 'R1' ? r1Pooled.reduce((s, x) => s + x, 0) / N * 100 : a === 'LL' ? llPooled.reduce((s, x) => s + x, 0) / N * 100 : CL[a].checklistPassRate;

console.log(`Reasoning-teacher control — paired bootstrap on summary-checklist per-item scores (N=${N}, B=${B})`);
console.log(`Source: ${file}\n`);
console.log(`Summary-checklist pass-rates:`);
console.log(`  R1-distilled (reasoning):    pooled ${rate('R1').toFixed(1)}  | seeds ${R1.map(s => CL[s].checklistPassRate).join(' / ')}`);
console.log(`  Llama-distilled (non-reas.): pooled ${rate('LL').toFixed(1)}  | seeds ${LL.map(s => CL[s].checklistPassRate).join(' / ')}`);
console.log(`  reference: teacher ${rate('teacher').toFixed(1)} | base ${rate('base').toFixed(1)} | constrained ${rate('constrained').toFixed(1)}\n`);

const comparisons = [
  ['HEADLINE  R1-pooled     vs Llama-pooled  (reasoning gap)', 'R1', 'LL'],
  ['          R1-s1         vs Llama-s1',                       R1[0], LL[0]],
  ['          R1-s2         vs Llama-s2',                       R1[1], LL[1]],
  ['          R1-s3         vs Llama-s3',                       R1[2], LL[2]],
  ['does R1 beat base?      R1-pooled     vs base',             'R1', 'base'],
  ['does Llama beat base?   Llama-pooled  vs base',             'LL', 'base'],
];
for (const [label, a, b] of comparisons) console.log(`${label}\n   ${fmt(bootstrap(arr(a), arr(b)))}`);

// classification is a closed-set point estimate (no per-item array stored) — report the means only
const CLS = S.classification.perArm;
const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
console.log(`\nClassification macro-acc (point estimates; not bootstrapped — no per-item array):`);
console.log(`  R1-distilled    pooled-mean ${mean(R1.map(a => CLS[a].macroAcc)).toFixed(1)}  | seeds ${R1.map(a => CLS[a].macroAcc).join(' / ')}`);
console.log(`  Llama-distilled pooled-mean ${mean(LL.map(a => CLS[a].macroAcc)).toFixed(1)}  | seeds ${LL.map(a => CLS[a].macroAcc).join(' / ')}`);
