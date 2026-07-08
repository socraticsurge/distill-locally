# Different Teachers, Different Capabilities — Sub-1B On-Device Distillation

![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Reproducible](https://img.shields.io/badge/analysis-reproduces%20offline%20from%20the%20scorecard-1a3c6e)
![Local](https://img.shields.io/badge/stack-Ollama%20%2B%20Unsloth-8A2BE2)

Can an 8B **reasoning** teacher (`deepseek-r1:8b`) be distilled into a 0.6B student (`Qwen3-0.6B`) for a structured text-enrichment task — one JSON object per article with a free-text summary, five categorical labels, and open-set topics — and *what specifically* does the distillation transfer? The study answers per sub-task, with a control-anchored, three-judge protocol, and settles the central question with a **same-size non-reasoning-teacher control**: an equal-size instruction teacher (`llama-3.1-8b-instruct`) trained under the identical recipe, plus a larger managed pipeline (`gpt-oss-120b` + synthetic-data expansion) as a systems comparison.

*The task was originally drawn from an RSS-reader side project; the repository is the study, standing on its own.*

![Experimental pipeline: 494 labeled articles split 401/93; three teachers (reasoning 8B, same-size non-reasoning 8B, managed 120B pipeline) distill the same Qwen3-0.6B base; twelve arms scored on two sub-tasks by a blinded three-judge panel against the full article](paper/figures/fig2_pipeline.png)

## Start here
- **The paper:** [`paper/arxiv/paper.pdf`](paper/arxiv/paper.pdf) (12 pp, IEEEtran) — source [`paper/arxiv/paper.tex`](paper/arxiv/paper.tex).
- **Browse it yourself:** open [`data/eval/viewer.html`](data/eval/viewer.html) locally — every arm's generation, the panel grades per check, the scorecard, and the analytics, in one offline page.
- **Citing this work:** [`CITATION.cff`](CITATION.cff).

## Headline findings
Canonical grading is **full-text** (the judge sees the whole article) by a **three-judge panel** (Gemini Flash Lite · Nemotron-550B · Mistral-Small-119B), reference-free, blinded, majority vote; **N = 93** held-out articles, **twelve arms**, three training seeds. Significance is paired-bootstrap (20 000 resamples).

- **Speed:** the 0.6B student runs at **~0.8 s/article** vs the teacher's **~39 s** — a 500-item batch collapses from **5.4 h to ~7 min** on a laptop.
- **Summary quality:** the tuned student scores **75.1** on the checklist — beating its primary baseline (constrained decoding) by **+16.8** `[11.4, 22.5]`, p<0.001, and few-shot prompting by a secondary **+4.9** `[1.6, 8.2]` — closing **58%** of the base→teacher gap.
- **It's the teacher's *reasoning nature*, not its scale (the control that anchors the title):** a same-size **non-reasoning** teacher, distilled by the identical recipe, trains a student **no better than the untuned base** (+0.6 `[−5.2, 6.4]`, n.s.), while the reasoning-teacher student beats base by **+18.7** `[13.1, 24.4]`. Reasoning-vs-instruction student gap: **+18.1** `[14.4, 22.0]`, p<0.001, across all three seed pairs.
- **Different teachers transfer different capabilities:** reasoning → **writing quality**; scale + synthetic data → **label diversity** (the managed pipeline is the best small-model classifier, macro 66.3, and the only arm above the `depth` majority baseline); the same-size instruction teacher → **thin-source grounding** (its students hold 74% faithful on the 22 short articles where the reasoning students fall to 55% and the managed student collapses to 36%). *This last effect is a consistent ordering, not a significant aggregate difference (n = 22) — reported as a direction.*
- **Classification, honestly:** fine-tuning does **not** beat prompting on macro accuracy (55.3 vs 58.5); the `urgency` "win" is mostly majority-class alignment; `depth` sits below its majority baseline for every small-model arm; `tone` is cued far better by a few examples than baked into weights.
- **Routing, scored as a system (§8):** send only the 22 short articles to a larger engine — to the **R1 teacher** (Router B, +4.6 checklist, ~82 min/500) for writing quality, or to the **Llama-8B teacher** (Router B′, the highest faithfulness of any bulk-on-device config, 82.8, at ~14 min/500) for grounding. Per-field classification routing reaches macro 76.1 — labeled as an **oracle upper bound**, not a held-out result.

The paper argues against its own convenience in several places (the `urgency` demotion, the thin-source caveat, and the full-source-grading fix that removed a spurious faithfulness "regression") — which is the point.

## Reproduce (offline, no API keys)
Every reported metric recomputes from the **committed scorecard** (`data/eval/scores.json`, `data/eval/scores_detail.json`) using only Node built-ins — no `npm install`, no keys.
```bash
# Primary summary win (tuned vs constrained / few-shot), paired bootstrap:
node server/eval/paired_bootstrap.mjs

# The reasoning-teacher control (R1-distilled vs Llama-distilled vs base):
node server/eval/paired_bootstrap_reasoning.mjs

# Teacher-neutral robustness cut (faithful/thesis/takeaway only):
node server/eval/robustness_neutral_checks.mjs

# Routing table scored as a system, incl. Router B′ (short → Llama-8B teacher):
node server/eval/router_bprime.mjs

# Local inspector: generations + panel grades + scorecard + analytics -> data/eval/viewer.html
node server/eval/build_viewer.mjs && node server/eval/build_viewer_html.mjs
```
Regenerating the figures needs Python with matplotlib (Times New Roman for the paper font):
```bash
python3 paper/figures/gen_fig2.py            # pipeline diagram
python3 paper/figures/gen_fig3_fig4_fig5.py  # gap-closure, checklist, efficiency
python3 paper/figures/gen_fig7.py            # three-teacher comparison
```
**Regenerating the judge grades from scratch** (`server/eval/score.mjs`) is the only step that needs API keys (`--env-file=.env`); reproducing the reported metrics from the committed scorecard does not.

## Layout
```
paper/arxiv/      the paper (paper.tex + paper.pdf), arXiv submission package (ARXIV_UPDATE.md,
                  arxiv_submission.tar.gz, arxiv_abstract.txt)
paper/figures/    figure generators (gen_fig*.py) + rendered PNGs (Times New Roman + STIX)
server/eval/      scoring engine (score.mjs), judge panel + clients (judges.mjs),
                  significance (paired_bootstrap*.mjs), routing (router_bprime.mjs),
                  robustness (robustness_neutral_checks.mjs), item-level analyses (detail_analyses.mjs),
                  local inspector builders (build_viewer*.mjs), teacher/arm generators
data/eval/        test/train sets, per-arm outputs (arm_*.jsonl), judge cache, canonical scorecards
                  (scores.json, scores_detail.json), split, system_prompt.txt, viewer.html
notebooks/        Unsloth QLoRA training notebooks (3 seeds; R1 and Llama teachers)
models/           trained GGUFs (~1.5 GB, git-ignored — regenerate via notebooks/)
```

## Toolchain (all open source)
Inference: **Ollama**. Fine-tuning: **Unsloth** QLoRA on a free Colab T4. Models: **Qwen3-0.6B**, **DeepSeek-R1:8b**, **Llama-3.1-8B-Instruct** (open weights). Judges (reference-free, hosted): **Gemini Flash Lite** (Google AI Studio), **Nemotron-550B** and **Mistral-Small-119B** (NVIDIA NIM). The core study uses no proprietary training platform; the managed-pipeline comparison arm was trained through a commercial managed service (`gpt-oss-120b` teacher + synthetic-data expansion), and its outputs and scores are in-repo.

## Notes
- **Tag `arxiv-v1-submitted`** pins the exact source submitted to arXiv v1; `main` carries the current revision.
- **`.env` is git-ignored** (judge API keys). Needed only to regenerate the judge cache; reproducing the reported metrics needs no keys.
- **`models/` is git-ignored** (1.5 GB of GGUFs) — regenerate from `notebooks/`.
- **Do not reimplement `keyOf`** in `server/eval/judges.mjs` — the cache key is byte-sensitive; a retyped copy silently misses the cache. Extend the scorer in place.
- **The judge cache is a speed cache, not the record.** Panel-**majority** grades are frozen in `scores_detail.json`; individual per-judge votes are not persisted (see the local inspector's Analytics note). A future revision should log raw per-judge grades with provenance.
- The datasets in `data/eval/` are provided pre-built; the reported results reproduce from `server/eval/` + `data/eval/` alone.
