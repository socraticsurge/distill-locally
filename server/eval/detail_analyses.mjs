// Item-level latent analyses (paper §6.3 confusion, §6.6 seed-agreement signal,
// §7 judge-disagreement decomposition + truncation flips, §6.2 length mechanism,
// verbosity direction). Reads the DETAIL_DUMP files produced by compile.mjs:
//
//   ARTICLE_CHARS=100000 JUDGES_EXCLUDE=gpt-oss-120b,llama-3.1-8b DETAIL_DUMP=1 node server/eval/compile.mjs
//   ARTICLE_CHARS=1200   JUDGES_EXCLUDE=gpt-oss-120b,llama-3.1-8b DETAIL_DUMP=1 node server/eval/compile.mjs
//   node server/eval/detail_analyses.mjs
//   -> prints all analyses, writes data/eval/detail_analyses.json
//
// Arms are discovered dynamically (tuned seeds by prefix), so re-running after new
// arms are added (e.g. a platform-distilled student) extends every table automatically.

import { readFileSync, writeFileSync } from "node:fs";

const D = "data/eval";
const full = JSON.parse(readFileSync(`${D}/detail_dump_100000.json`, "utf8"));
const lead = JSON.parse(readFileSync(`${D}/detail_dump_1200.json`, "utf8"));
const test = readFileSync(`${D}/test.jsonl`, "utf8").split("\n").filter(Boolean).map(JSON.parse);
const CHECKS = ["faithful", "thesis", "takeaway", "length", "opening", "teacher", "tech", "tone"];
const FIELDS = ["sentiment", "urgency", "frame", "tone", "depth"];

const { ids, gold, pred } = full.classification;
const SEEDS = Object.keys(pred).filter((a) => a.startsWith("tuned_")).sort();
const ARMS = Object.keys(pred).filter((a) => a !== "constrained");
const out = {};

// ── 1. Per-class confusion for urgency (paper §6.3) ─────────────────────────
const gdist = {};
for (const g of gold) gdist[g.urgency] = (gdist[g.urgency] || 0) + 1;
const majorityBaseline = +((Math.max(...Object.values(gdist)) / gold.length) * 100).toFixed(1);
out.urgency = { goldDist: gdist, majorityBaseline, perArm: {} };
console.log(`\n1. URGENCY per-class analysis — gold ${JSON.stringify(gdist)}, majority baseline ${majorityBaseline}%`);
for (const arm of ARMS) {
  const recall = {}, total = {}, predDist = {};
  for (let i = 0; i < gold.length; i++) {
    const p = pred[arm][i]; if (!p) continue;
    const g = gold[i].urgency;
    total[g] = (total[g] || 0) + 1;
    if (p.urgency === g) recall[g] = (recall[g] || 0) + 1;
    predDist[p.urgency] = (predDist[p.urgency] || 0) + 1;
  }
  const rec = Object.fromEntries(Object.keys(gdist).map((g) => [g, +((100 * (recall[g] || 0)) / (total[g] || 1)).toFixed(0)]));
  out.urgency.perArm[arm] = { recallPct: rec, predDist };
  console.log(`  ${arm.padEnd(22)} recall ${JSON.stringify(rec)}  predicts ${JSON.stringify(predDist)}`);
}

// ── 2. Seed-agreement as a per-item confidence signal (§6.6) ────────────────
out.seedAgreement = {};
console.log(`\n2. SEED AGREEMENT signal (${SEEDS.length} seeds: ${SEEDS.join(", ")})`);
for (const f of FIELDS) {
  let uT = 0, uC = 0, sT = 0, sC = 0;
  for (let i = 0; i < gold.length; i++) {
    const ps = SEEDS.map((s) => pred[s][i]?.[f]).filter(Boolean);
    if (ps.length < SEEDS.length) continue;
    const g = gold[i][f];
    if (new Set(ps).size === 1) { uT++; if (ps[0] === g) uC++; }
    else { sT++; sC += ps.filter((x) => x === g).length / ps.length; }
  }
  out.seedAgreement[f] = { unanimousN: uT, unanimousAcc: +((100 * uC) / (uT || 1)).toFixed(1), splitN: sT, splitAcc: +((100 * sC) / (sT || 1)).toFixed(1) };
  console.log(`  ${f.padEnd(10)} unanimous ${String(uT).padStart(2)} items acc ${out.seedAgreement[f].unanimousAcc}%   split ${String(sT).padStart(2)} items acc ${out.seedAgreement[f].splitAcc}%`);
}

// ── 3. Judge disagreement by check (§7) ─────────────────────────────────────
const dis = {};
for (const it of full.checklistItems)
  for (const arm of Object.keys(it.arms))
    for (const c of CHECKS) {
      dis[c] ||= [0, 0]; dis[c][1]++;
      if (new Set(it.arms[arm][c].votes).size > 1) dis[c][0]++;
    }
out.judgeDisagreementByCheck = Object.fromEntries(Object.entries(dis).map(([c, [d, t]]) => [c, +((100 * d) / t).toFixed(1)]));
console.log(`\n3. JUDGE DISAGREEMENT by check:`, out.judgeDisagreementByCheck);

// ── 4. Truncation flips on faithful, lead -> full (§7) ──────────────────────
const leadById = Object.fromEntries(lead.checklistItems.map((it) => [it.id, it]));
out.truncationFlips = { perArm: {}, byLength: { long: {}, short: {} } };
let up = 0, down = 0;
for (const it of full.checklistItems) {
  const li = leadById[it.id]; if (!li) continue;
  const bucket = it.artLen > 1200 ? "long" : "short";
  for (const arm of Object.keys(it.arms)) {
    const a = li.arms[arm]?.faithful.maj, b = it.arms[arm].faithful.maj;
    const k = `${a}->${b}`;
    out.truncationFlips.byLength[bucket][k] = (out.truncationFlips.byLength[bucket][k] || 0) + 1;
    const pa = (out.truncationFlips.perArm[arm] ||= {});
    pa[k] = (pa[k] || 0) + 1;
    if (!a && b) up++; if (a && !b) down++;
  }
}
out.truncationFlips.totalUp = up; out.truncationFlips.totalDown = down;
console.log(`\n4. TRUNCATION FLIPS (faithful): ${up} up vs ${down} down; by length:`, out.truncationFlips.byLength);

// ── 5 & 6. Verbosity direction + sentence-count mechanism (§6.2) ────────────
const summaryOf = (o) => { try { return JSON.parse(o.slice(o.indexOf("{"), o.lastIndexOf("}") + 1)).summary || ""; } catch { return ""; } };
const outputsOf = (arm) => arm === "teacher"
  ? Object.fromEntries(test.map((t) => [t.id, t.teacher_output]))
  : Object.fromEntries(readFileSync(`${D}/arm_${arm}.jsonl`, "utf8").split("\n").filter(Boolean).map((l) => { const r = JSON.parse(l); return [r.id, r.output]; }));
function spearman(x, y) {
  const rank = (v) => { const s = [...v.keys()].sort((a, b) => v[a] - v[b]); const r = Array(v.length); s.forEach((idx, j) => (r[idx] = j)); return r; };
  const rx = rank(x), ry = rank(y), n = x.length, mx = (n - 1) / 2;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { num += (rx[i] - mx) * (ry[i] - mx); dx += (rx[i] - mx) ** 2; dy += (ry[i] - mx) ** 2; }
  return num / Math.sqrt(dx * dy || 1);
}
out.verbosity = {}; out.sentences = {};
console.log(`\n5/6. VERBOSITY + SENTENCE COUNTS`);
for (const arm of ARMS) {
  const outs = outputsOf(arm);
  const xs = [], ys = []; const cnt = { "<3": 0, "3-4": 0, ">4": 0 };
  for (const it of full.checklistItems) {
    if (!it.arms[arm] || !outs[it.id]) continue;
    const summ = summaryOf(outs[it.id]); if (!summ) continue;
    xs.push(summ.length);
    ys.push(CHECKS.filter((c) => it.arms[arm][c].maj).length / CHECKS.length);
    const ns = summ.split(/[.!?]+(?:\s|$)/).filter((s) => s.trim()).length;
    cnt[ns < 3 ? "<3" : ns <= 4 ? "3-4" : ">4"]++;
  }
  const t = cnt["<3"] + cnt["3-4"] + cnt[">4"] || 1;
  out.verbosity[arm] = +spearman(xs, ys).toFixed(2);
  out.sentences[arm] = Object.fromEntries(Object.entries(cnt).map(([k, v]) => [k, +((100 * v) / t).toFixed(1)]));
  console.log(`  ${arm.padEnd(22)} spearman(len,pass) ${out.verbosity[arm] >= 0 ? "+" : ""}${out.verbosity[arm]}   sentences ${JSON.stringify(out.sentences[arm])}`);
}

writeFileSync(`${D}/detail_analyses.json`, JSON.stringify(out, null, 2));
console.log(`\nwrote ${D}/detail_analyses.json`);
