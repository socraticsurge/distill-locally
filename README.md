# Distill Locally — DeepSeek-R1:8b → Qwen3-0.6B

![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Pre-specified](https://img.shields.io/badge/design-pre--specified%2C%20frozen%20before%20scoring-1a3c6e)
![Reproducible](https://img.shields.io/badge/reproduces-offline%2C%20no%20API%20keys-1a3c6e)
![Local](https://img.shields.io/badge/stack-Ollama%20%2B%20Unsloth-8A2BE2)

A local knowledge-distillation study: can an 8B reasoning teacher (`deepseek-r1:8b`) be distilled into a 600M student (`Qwen3-0.6B`) for a structured RSS-article-summarization task — a JSON object with a free-text summary, five categorical fields, and open-set topics — and *what specifically* does the distillation buy? Evaluated with a pre-specified (frozen-before-scoring), control-anchored, multi-judge protocol — and extended, under a dated amendment, with a managed-platform distillation arm (Distil Labs) as a system-level comparison.

This repository was extracted from the "Atlas Pulse" RSS reader app so the study can live, version, and be reviewed on its own.

## Start here
- **The paper:** [`paper/distillation_paper.md`](paper/distillation_paper.md) — or the rendered arXiv-style [`paper/distillation_paper.pdf`](paper/distillation_paper.pdf) (with figures).
- **Reviewer's map (every claim → the file that backs it):** [`paper/REVIEW_MANIFEST.md`](paper/REVIEW_MANIFEST.md)
- **Frozen analysis plan + dated amendments:** [`paper/PREREGISTRATION.md`](paper/PREREGISTRATION.md) · full design: [`paper/evaluation_design.md`](paper/evaluation_design.md)
- **What actually ships (measurement vs. solution design):** paper §8.2 — two engines and one rule.
- **Practitioner lessons (the decision surface — teacher/student choice, quantization, tuning recipe, data size, controls, judging):** paper §9.1.
- **Citing this work:** see [`CITATION.cff`](CITATION.cff).

## Headline findings
All numbers are from **full-text judge grading** (the judge sees the whole article; N = 93, eight arms).

- **Summary quality:** DIY distillation significantly beats the formatting control (+15.5 pts checklist, p<0.001) **and** few-shot prompting (+5.9, p<0.001), closing ~59% of the base→teacher gap. The soft spot is faithfulness on short-source articles — a localized fabrication mode (§6.2.1).
- **Classification, honestly:** the tuned student tops every arm — including its own teacher — on `urgency` agreement, but the promised per-class confusion analysis **demotes that headline**: most of the margin is majority-class alignment, +4.3 points over a constant guess (§6.3). The macro ties few-shot.
- **The routing table, scored as a system (§8.1):** the paper's own short-source advice did not survive its own measurement — prompting fallbacks barely restore faithfulness; only a teacher fallback does (+6.1 checklist, paired [3.1, 9.2], at ~82 min per 500 articles).
- **A managed-platform arm (Amendment 2, §6.8):** the same 401 training articles through Distil Labs (its own 120B teacher, synthetic-data expansion) **ties** the DIY QLoRA on summary quality (Δ−3.6 [−8.5, +1.1]) while producing the **best small-model classifier** and the **worst thin-source faithfulness** (36.4%) — the two training recipes transfer different capabilities.
- **What ships (§8.2):** measure every field; ship only the routing that pays — for this product, two engines and one rule.

The paper contains three self-corrections found by its own instruments (§8.1, §6.3, and the §7 prompt-composition check) — which is the point. See §6–§9.

## Reproduce the results (offline, no API keys)
Core analysis uses only Node built-ins — no `npm install` required. Full-text grading (`ARTICLE_CHARS=100000`) is canonical throughout the paper.
```bash
# Canonical full-text 2-judge (Gemini + Nemotron) scorecard, recomputed from the judge cache:
ARTICLE_CHARS=100000 JUDGES_EXCLUDE=gpt-oss-120b,llama-3.1-8b node server/eval/compile.mjs
#   -> data/eval/scores.json  (== data/eval/scores_gemini_nemotron_n93_fullctx.json)

# Pre-specified paired-bootstrap significance test (on the full-text scorecard):
node server/eval/paired_bootstrap.mjs data/eval/scores_gemini_nemotron_n93_fullctx.json

# Per-judge robustness (paper §7) — each judge alone:
ARTICLE_CHARS=100000 JUDGES_ONLY=gemini node server/eval/compile.mjs
ARTICLE_CHARS=100000 JUDGES_ONLY=nvidia node server/eval/compile.mjs

# Fresh 8-arm scorecard incl. the platform arm (Amendment 2) + drift check vs the file above:
#   -> data/eval/scores_gemini_nemotron_n93_fullctx_8arm.json (committed)

# The routing table scored as a system (paper §8.1):
node server/eval/router_composite.mjs

# Item-level analyses — urgency confusion, seed-agreement signal, judge-noise
# decomposition, truncation flips (paper §6.3 / §6.6 / §7):
ARTICLE_CHARS=100000 JUDGES_EXCLUDE=gpt-oss-120b,llama-3.1-8b DETAIL_DUMP=1 node server/eval/compile.mjs
ARTICLE_CHARS=1200   JUDGES_EXCLUDE=gpt-oss-120b,llama-3.1-8b DETAIL_DUMP=1 node server/eval/compile.mjs
node server/eval/detail_analyses.mjs

# Regenerate Figure 2 (pipeline diagram) and Figure 6 (8-arm checklist):
python3 paper/figures/gen_fig2_fig6.py

# Rebuild the paper PDF (needs Python with markdown + weasyprint):
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python3 paper/build_pdf.py
```
`compile.mjs` only reads `data/eval/judge_cache.jsonl`; it never calls an API. **API keys (`.env`) are needed only to *regenerate* the cache** (`server/eval/score.mjs`, `gen_arms.mjs`), which a reviewer does not need to do. Figures 2 and 6 regenerate from data (`paper/figures/gen_fig2_fig6.py`); the remaining figures are author-provided PNGs in `paper/figures/`.

## Layout
```
paper/            the paper, review manifest, pre-registration, design, pipeline diagram
server/eval/      analysis engine (compile.mjs), judge panel + clients (judges.mjs),
                  significance (paired_bootstrap.mjs), router scoring (router_composite.mjs),
                  item-level analyses (detail_analyses.mjs), cache/arm generators (score.mjs, gen_arms.mjs)
data/eval/        test/train sets, per-arm outputs (incl. arm_distil.jsonl), judge cache, scorecards, split,
                  and system_prompt.txt (the generation prompt)
notebooks/        Unsloth QLoRA training notebook (3 seeds)
models/           trained GGUFs (~1.5 GB, git-ignored — regenerate via notebooks/)
```

## Toolchain (all open source)
Teacher & student inference: **Ollama**. Fine-tuning: **Unsloth** QLoRA on a free Colab T4. Models: **Qwen3-0.6B** and **DeepSeek-R1:8b** (open weights). Judges: **Gemini Flash Lite** (Google AI Studio) and **Nemotron** (NVIDIA NIM), reference-free. The core study uses no proprietary training platform; the Amendment 2 comparison arm was trained on the **Distil Labs** managed pipeline, and its outputs and scores are in-repo.

## Notes
- **`models/` is git-ignored** (1.5 GB of GGUF binaries). Regenerate from `notebooks/unsloth_qwen3_0_6b_3seeds.ipynb`.
- **`.env` is git-ignored** and holds judge API keys (Groq / Google AI Studio / NVIDIA NIM). Only needed to regenerate the judge cache; reproducing the reported metrics needs no keys.
- **Do not reimplement `keyOf`** in `server/eval/judges.mjs` — the cache key is byte-sensitive; a retyped copy silently misses the cache 100%. Extend `compile.mjs` in place instead.
- The datasets in `data/eval/` are provided pre-built; the raw data-generation/sampling scripts are not part of this repo (the reported results reproduce from `server/eval/` + `data/eval/` alone).
