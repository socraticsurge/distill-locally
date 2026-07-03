// Router composite scoring (paper §8.1) — post-hoc, computed entirely from the
// canonical scorecard. A router is a deterministic per-article / per-field engine
// selection, so its output for every test item is identical to one already-graded
// arm output; composite metrics are therefore exact arithmetic over existing
// per-item grades. No generation, no judging, no API keys.
//
//   node server/eval/router_composite.mjs
//   -> prints the §8.1 table, writes data/eval/router_composite.json

import { readFileSync, writeFileSync } from "node:fs";

const SCORES = "data/eval/scores_gemini_nemotron_n93_fullctx.json";
const TEST = "data/eval/test.jsonl";
const SHORT_CHARS = 1200; // same threshold as §6.2.1
const SEEDS = ["tuned_rss_tuned_s1", "tuned_rss_tuned_s2", "tuned_rss_tuned_s3"];
const B = 20000;

const d = JSON.parse(readFileSync(SCORES, "utf8"));
const test = readFileSync(TEST, "utf8").split("\n").filter(Boolean).map(JSON.parse);
const ck = d.checklist, byLen = ck.__faithfulByLen, perArm = d.classification.perArm;

// Short/long flags: article body length, matching the §6.2.1 subgroup (22 short / 71 long)
const short = test.map((t) => {
  const m = t.input.match(/Content:\s*([\s\S]*)/);
  return (m ? m[1] : t.input).length <= SHORT_CHARS;
});
const nShort = short.filter(Boolean).length;
if (nShort !== byLen.teacher.shortN) throw new Error(`short reconstruction mismatch: ${nShort} vs ${byLen.teacher.shortN}`);

// ── Composite checklist: splice per-item grades (tuned on long, fallback arm on short) ──
function compositeChecklist(shortArm) {
  const perSeed = SEEDS.map((s) => {
    const tu = ck[s].perItem, fb = ck[shortArm].perItem;
    return (tu.reduce((a, v, i) => a + (short[i] ? fb[i] : v), 0) / tu.length) * 100;
  });
  return { perSeed: perSeed.map((v) => +v.toFixed(1)), mean: +(perSeed.reduce((a, b) => a + b) / 3).toFixed(1) };
}

// ── Composite faithful: exact re-weighting of the published length-split aggregates ──
const tunedLong = SEEDS.reduce((a, s) => a + byLen[s].long, 0) / 3;
const compositeFaithful = (shortArm) =>
  +(((nShort * byLen[shortArm].short) + ((93 - nShort) * tunedLong)) / 93).toFixed(1);

// ── Paired bootstrap: router vs all-tuned on per-item checklist diffs (seed-averaged) ──
function pairedBootstrap(shortArm) {
  const diffs = test.map((_, i) => {
    if (!short[i]) return 0; // router == tuned on long items
    return SEEDS.reduce((a, s) => a + (ck[shortArm].perItem[i] - ck[s].perItem[i]), 0) / 3;
  });
  const n = diffs.length;
  let seed = 7;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const means = [];
  for (let b = 0; b < B; b++) {
    let sum = 0;
    for (let k = 0; k < n; k++) sum += diffs[(rand() * n) | 0];
    means.push(sum / n);
  }
  means.sort((a, b) => a - b);
  const mean = diffs.reduce((a, b) => a + b) / n;
  return { delta: +(mean * 100).toFixed(1), ci95: [+(means[(0.025 * B) | 0] * 100).toFixed(1), +(means[(0.975 * B) | 0] * 100).toFixed(1)] };
}

// ── Per-field classification composite (wholesale: field -> best arm per §8) ──
const tunedField = (f) => SEEDS.reduce((a, s) => a + perArm[s].byField[f], 0) / 3;
const routerCls = {
  urgency: +tunedField("urgency").toFixed(1),
  frame: +tunedField("frame").toFixed(1),
  sentiment: +tunedField("sentiment").toFixed(1),
  tone: +perArm.fewshot.byField.tone.toFixed(1),
};
const macro4 = (vals) => +(vals.reduce((a, b) => a + b) / 4).toFixed(1);
const cls4 = {
  router: macro4(Object.values(routerCls)),
  teacher: macro4(["urgency", "frame", "sentiment", "tone"].map((f) => perArm.teacher.byField[f])),
  tuned: macro4(["urgency", "frame", "sentiment", "tone"].map(tunedField)),
  fewshot: macro4(["urgency", "frame", "sentiment", "tone"].map((f) => perArm.fewshot.byField[f])),
};

// ── Batch latency estimate for 500 articles (p50s from §6.5) ──
const latency500 = (shortArm) => {
  const perShort = shortArm === "teacher" ? 39.2 : 0.8;
  return +(((500 * (nShort / 93)) * perShort + (500 * (1 - nShort / 93)) * 0.8) / 60).toFixed(0);
};

const routers = {
  "A_short_fewshot": { shortArm: "fewshot" },
  "B_short_teacher": { shortArm: "teacher" },
  "C_short_base": { shortArm: "base" },
};
const out = { nShort, shortChars: SHORT_CHARS, classification: { routerFields: routerCls, macro4: cls4 }, routers: {} };
console.log(`Router composites (post-hoc splice of existing per-item grades; ${nShort} short / ${93 - nShort} long)\n`);
console.log("router              checklist  faithful  cls-macro4  batch500   vs all-tuned (paired)");
for (const [name, { shortArm }] of Object.entries(routers)) {
  const c = compositeChecklist(shortArm), f = compositeFaithful(shortArm), p = pairedBootstrap(shortArm);
  out.routers[name] = { shortArm, checklist: c, faithful: f, cls4: cls4.router, batch500min: latency500(shortArm), pairedVsTuned: p };
  console.log(`${name.padEnd(18)}  ${String(c.mean).padEnd(9)}  ${String(f).padEnd(8)}  ${String(cls4.router).padEnd(10)}  ~${String(latency500(shortArm)).padEnd(3)} min   Δ${p.delta} [${p.ci95}]`);
}
console.log(`\nreference: all-tuned checklist ${(SEEDS.reduce((a, s) => a + ck[s].checklistPassRate, 0) / 3).toFixed(1)}, teacher ${ck.teacher.checklistPassRate}, cls-macro4 teacher ${cls4.teacher} / tuned ${cls4.tuned} / fewshot ${cls4.fewshot}`);
writeFileSync("data/eval/router_composite.json", JSON.stringify(out, null, 2));
console.log("\nwrote data/eval/router_composite.json");
