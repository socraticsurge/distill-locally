# Review Manifest — DeepSeek-R1 → Qwen3-0.6B Distillation Study

This file is the entry point for reviewing the paper and the data behind it. Everything the analysis rests on is in-repo; **all headline metrics can be recomputed offline from the judge cache with no API keys** (see §Reproduce).

Paths are relative to the repo root: `/Users/vinaychaganti/Documents/VibeCodedApps/distill-locally`. (This study was extracted from the "Atlas Pulse" RSS reader app into its own repository; see the top-level `README.md`.)

---

## 1. What to review
- **The paper:** [`paper/distillation_paper.md`](distillation_paper.md) (or the rendered [`paper/distillation_paper.pdf`](distillation_paper.pdf)) — the final study (pre-registered, 7 arms, 2-judge panel, N=93, **full-text grading canonical**). **This is the deliverable.**
- **Frozen design (pre-registration):** [`paper/PREREGISTRATION.md`](PREREGISTRATION.md) (frozen 2026-07-02, before scoring) and [`paper/evaluation_design.md`](evaluation_design.md) (full design v0.8). The pipeline diagram: [`paper/evaluation_pipeline.mmd`](evaluation_pipeline.mmd).
- *Note:* an earlier v1 pilot (20 articles, single judge, saturated rubric) is not part of this repo; the current paper supersedes it in every respect.

---

## 2. Data files (`data/eval/`) — what constitutes the analysis

### Inputs / ground truth
| File | Rows | Role | Schema |
|---|---|---|---|
| `test.jsonl` | 93 | Held-out test set + teacher gold | `id, feed, title, input (article text), teacher_output (JSON string)` |
| `train_chat.jsonl` | 401 | Training set (teacher traces, chat format) | training pairs |
| `split.json` | — | The exact 401/93 stratified-random split manifest | ids per split, per feed |
| `pilot.jsonl` | 12 | Pilot set (train subset) used to freeze the checklist | same as test |

`teacher_output` and every arm output is a JSON object with 7 fields: `summary, sentiment, urgency, frame, tone, depth, topics`.

### Arm outputs (the models' actual generations, temperature 0)
| File | Rows | Arm |
|---|---|---|
| `arm_base.jsonl` | 93 | Base `Qwen3-0.6B` zero-shot |
| `arm_fewshot.jsonl` | 93 | Base + few-shot (control: prompting) |
| `arm_constrained.jsonl` | 93 | Base + constrained JSON decoding (control: formatting) |
| `arm_tuned_rss_tuned_s1/s2/s3.jsonl` | 93 each | Distilled student, 3 seeds |
| (teacher arm) | 93 | Read from `test.jsonl` `teacher_output`, not a separate file |
| `arm_pilot_*.jsonl` | 12 each | Pilot-set generations (used only for the freeze) |

Arm record schema: `id, feed, arm, output (JSON string), durationMs, evalCount`.

### Judge data (the grading)
| File | Rows | Role |
|---|---|---|
| `judge_cache.jsonl` | 664 | **Content-addressed cache of every judge response.** Each line `{k: sha1-key, v: parsed-judge-JSON}`. This is the raw grading data; the scorecards are computed from it. Covers classification (per field), the batched checklist (labels A–G per item), and the negative control. |
| `system_prompt.txt` | — | The generation system prompt used for every arm (read by `gen_arms.mjs`). |

### Scorecards (computed results)
| File | Role |
|---|---|
| `scores_gemini_nemotron_n93_fullctx.json` | **THE result (canonical)** — full-text-grading (`ARTICLE_CHARS=100000`) 2-judge (Gemini + Nemotron) N=93 scorecard the paper reports. Contains structure, classification (per-field + agreement), checklist (per-arm, per-item, `__perCheck`, `__perCheckByArm`, `__faithfulByLen`, `__disagreeRate`), negative control. Regenerate: `ARTICLE_CHARS=100000 JUDGES_EXCLUDE=gpt-oss-120b,llama-3.1-8b node server/eval/compile.mjs`. |
| `scores_gemini_nemotron_n93.json` | **Sensitivity** — the earlier lead-only (1200-char) scorecard, kept to reproduce the truncation-artifact analysis (§6.2.1, §7). Not the reported numbers. |

### Visualization (not an analysis input)
`outputs_comparison.html` — side-by-side viewer of all seven arms' outputs, for eyeballing.

---

## 3. Analysis source code (`server/eval/`)
| File | Role |
|---|---|
| `judges.mjs` | Judge panel definition + API clients (Groq / Gemini / NVIDIA), rate-limit gate, `keyOf` cache key, `majority()`. **`JUDGES` array = the panel.** |
| `compile.mjs` | **The analysis engine.** Cache-only (never calls the API): reconstructs judge prompts, looks them up in `judge_cache.jsonl`, and computes every metric. This produced the scorecard. Panel selected via env vars (below). |
| `paired_bootstrap.mjs` | Pre-registered paired-bootstrap significance test on the checklist per-item scores (reads the scorecard). Produces the §5.4 numbers. |
| `score.mjs` | The online (API-calling) version of `compile.mjs` — used to *generate* the judge cache. Same metrics; writes to cache. |
| `gen_arms.mjs` | Generated the arm outputs (base/fewshot/constrained/tuned) on the test set, temp 0, resumable. |
| `ping_judges.mjs` | Judge connectivity check. |
| `attribute.mjs`, `score_peek.mjs` | Exploratory/abandoned; not part of the reported pipeline. |
| `README.md` | Script-level notes. |

---

## 4. Reproduce the results (offline, no API keys)
```bash
cd "/Users/vinaychaganti/Documents/VibeCodedApps/RSS Feed Reader"

# Full 2-judge scorecard (Gemini + Nemotron) — recomputes every metric from the cache:
JUDGES_EXCLUDE=gpt-oss-120b,llama-3.1-8b node server/eval/compile.mjs

# Per-judge robustness check (paper §5.5) — each judge alone:
JUDGES_ONLY=gemini node server/eval/compile.mjs
JUDGES_ONLY=nvidia node server/eval/compile.mjs      # nvidia = Nemotron-550B

# Pre-registered significance test (paper §5.4):
node server/eval/paired_bootstrap.mjs
```
No `.env` / keys needed — `compile.mjs` only reads `judge_cache.jsonl`. Keys are only required to *regenerate* the cache (`score.mjs`), which a reviewer should not need to do.

---

## 5. Known caveats a reviewer should scrutinize (all disclosed in the paper)
1. **Two judges, by choice.** Every *direction* is confirmed in each judge alone (§5.5); a third same-family LLM judge was deliberately not added. Residual risk = *shared* bias, resolvable only by human labels.
2. **`majority()` tie-break.** With exactly 2 judges, ties break toward the first-listed judge (Gemini). The single "consensus" per-check table (§5.2) therefore ≈ Gemini on the 26.8% of split checks; §5.5 reports each judge separately. See the `keyOf` note below.
3. **`keyOf` is byte-sensitive.** The cache key function in `compile.mjs`/`judges.mjs` contains a non-ASCII byte; a *retyped* ASCII copy silently misses the cache 100%. Reviewers extending the analysis must reuse `compile.mjs`, not reimplement `keyOf`.
4. **Proxy-gold ≠ truth.** Classification accuracy is vs. judge consensus (agreement, not correctness); `tone`/`depth` have the lowest inter-judge agreement.
5. **Judges saw the first 1200 chars** of each article (free-tier token budget), uniform across arms.
6. **Efficiency numbers (§5.6)** are model-intrinsic figures carried from v1, not re-measured per-arm on the 93-set.
7. **Not yet run:** pairwise win-rate; human spot-check.

---

## 6. Headline claims and where each is backed
| Claim | Source |
|---|---|
| Primary: tuned > constrained on checklist, +11.6pp, p<0.001, every seed | `paired_bootstrap.mjs` output; §5.4 |
| Tuned ties few-shot on checklist (composition trade, not equivalence) | `scores_*.json` `__perCheckByArm`; §5.2 |
| Distillation win = `takeaway`; loss = `faithful` | `__perCheckByArm`; §5.2 |
| Urgency: tuned beats teacher | `classification.perArm.byField`; §5.3, confirmed both judges §5.5 |
| Tone-labeling collapse (seed-unstable) | `classification.perArm`; §5.3, §5.5 |
| Negative control 0% faithful on mismatch | `scores_*.json` `negative_control`; §4 |
| Every direction holds in each judge alone | `JUDGES_ONLY=` runs; §5.5 |
