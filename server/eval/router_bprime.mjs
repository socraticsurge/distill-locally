// Router composites recomputed from the canonical per-item detail (scores_detail.json)
// plus article lengths (test.jsonl). Reproduces routers A/B/C from the paper and adds
// Router B' (short -> Llama-3.1-8B-Instruct teacher), per reviewer request.
//
// A router is a deterministic per-article engine selection, so its per-item grades are
// spliced from already-graded arms; composites are exact arithmetic. No API calls.
//   node server/eval/router_bprime.mjs
import { readFileSync } from "node:fs";

const ROOT = decodeURIComponent(new URL("../..", import.meta.url).pathname);
const detail = JSON.parse(readFileSync(ROOT + "data/eval/scores_detail.json", "utf8"));
const test = readFileSync(ROOT + "data/eval/test.jsonl", "utf8").split("\n").filter(Boolean).map(JSON.parse);

const SHORT = 1200;
const CHECKS = ["faithful", "thesis", "takeaway", "length", "opening", "teacher", "tech", "tone"];
const SEEDS = ["tuned_rss_tuned_s1", "tuned_rss_tuned_s2", "tuned_rss_tuned_s3"];

// article ids in test order + short flag from body length (mirrors §6.2.1)
const ids = test.map((t) => String(t.id));
const bodyLen = (t) => { const m = t.input.match(/Content:\s*([\s\S]*)/); return (m ? m[1] : t.input).length; };
const short = test.map((t) => bodyLen(t) <= SHORT);
const nShort = short.filter(Boolean).length;

const checklistItem = (arm, id) => { const r = detail[arm][id]; return CHECKS.reduce((a, c) => a + (r[c] ? 1 : 0), 0) / CHECKS.length; };
const faithfulItem = (arm, id) => (detail[arm][id].faithful ? 1 : 0);

const armChecklist = (arm) => ids.reduce((a, id) => a + checklistItem(arm, id), 0) / ids.length * 100;
const armFaithful = (arm) => ids.reduce((a, id) => a + faithfulItem(arm, id), 0) / ids.length * 100;

// sanity: reproduce published arm aggregates
console.log("== arm self-check (should match scores.json) ==");
for (const a of ["teacher", "teacher_llama", "base", "fewshot", "distil"])
  console.log(`${a.padEnd(14)} checklist ${armChecklist(a).toFixed(1)}  faithful ${armFaithful(a).toFixed(1)}`);
const tunedMean = SEEDS.reduce((a, s) => a + armChecklist(s), 0) / 3;
const tunedFaith = SEEDS.reduce((a, s) => a + armFaithful(s), 0) / 3;
console.log(`tuned (3-seed) checklist ${tunedMean.toFixed(1)}  faithful ${tunedFaith.toFixed(1)}`);

// faithful by length for any arm
const faithByLen = (arm) => {
  let sS = 0, nS = 0, sL = 0, nL = 0;
  ids.forEach((id, i) => { const v = faithfulItem(arm, id); if (short[i]) { sS += v; nS++; } else { sL += v; nL++; } });
  return { short: +(sS / nS * 100).toFixed(1), long: +(sL / nL * 100).toFixed(1) };
};
console.log("\n== faithful by length ==");
for (const a of ["teacher", "teacher_llama", "distil", ...SEEDS])
  console.log(`${a.padEnd(20)} short ${faithByLen(a).short}  long ${faithByLen(a).long}`);
const tunedLongFaith = SEEDS.reduce((a, s) => a + faithByLen(s).long, 0) / 3;

// composite: tuned on long, fallback arm on short
function composite(shortArm, perShortMs) {
  const perSeedChk = SEEDS.map((s) =>
    ids.reduce((a, id, i) => a + (short[i] ? checklistItem(shortArm, id) : checklistItem(s, id)), 0) / ids.length * 100);
  const chkMean = perSeedChk.reduce((a, b) => a + b) / 3;
  // faithful composite = short items from fallback + long items from tuned mean
  const fbShortFaith = faithByLen(shortArm).short;
  const faith = (nShort * fbShortFaith + (93 - nShort) * tunedLongFaith) / 93;
  // paired bootstrap vs all-tuned on per-item checklist (seed-averaged diffs), fixed seed
  const diffs = ids.map((id, i) => short[i] ? (checklistItem(shortArm, id) - SEEDS.reduce((a, s) => a + checklistItem(s, id), 0) / 3) : 0);
  let seed = 7; const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const B = 20000, means = [];
  for (let b = 0; b < B; b++) { let sum = 0; for (let k = 0; k < diffs.length; k++) sum += diffs[(rand() * diffs.length) | 0]; means.push(sum / diffs.length); }
  means.sort((a, b) => a - b);
  const delta = diffs.reduce((a, b) => a + b) / diffs.length * 100;
  const batch = ((500 * nShort / 93) * perShortMs + (500 * (93 - nShort) / 93) * 0.8) / 60;
  return { chk: +chkMean.toFixed(1), delta: +delta.toFixed(1),
    ci: [+(means[(0.025 * B) | 0] * 100).toFixed(1), +(means[(0.975 * B) | 0] * 100).toFixed(1)],
    faith: +faith.toFixed(1), batchMin: Math.round(batch) };
}

console.log(`\n== routers (nShort=${nShort}) ==`);
const routers = [
  ["A short->few-shot", "fewshot", 0.8],
  ["C short->base", "base", 0.8],
  ["B short->R1 teacher", "teacher", 39.2],
  ["B' short->Llama teacher", "teacher_llama", 4.713],
];
console.log(`all-tuned         chk ${tunedMean.toFixed(1)}  faith ${tunedFaith.toFixed(1)}  batch ~7 min`);
for (const [name, arm, ms] of routers) {
  const c = composite(arm, ms);
  console.log(`${name.padEnd(24)} chk ${c.chk}  Δvs-tuned ${c.delta} [${c.ci}]  faith ${c.faith}  batch ~${c.batchMin} min`);
}
