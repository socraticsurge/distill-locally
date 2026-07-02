# Distill Locally — DeepSeek-R1:8b → Qwen3-0.6B

A local knowledge-distillation study: can an 8B reasoning teacher (`deepseek-r1:8b`) be distilled into a 600M student (`Qwen3-0.6B`) for a structured RSS-article-summarization task — a JSON object with a free-text summary, five categorical fields, and open-set topics — and *what specifically* does the distillation buy? Evaluated with a pre-registered, control-anchored, multi-judge protocol.

This repository was extracted from the "Atlas Pulse" RSS reader app so the study can live, version, and be reviewed on its own.

## Start here
- **The paper:** [`paper/distillation_paper_v2.md`](paper/distillation_paper_v2.md)
- **Reviewer's map (every claim → the file that backs it):** [`paper/REVIEW_MANIFEST.md`](paper/REVIEW_MANIFEST.md)
- **Pre-registration (frozen before scoring):** [`paper/PREREGISTRATION.md`](paper/PREREGISTRATION.md) · full design: [`paper/evaluation_design.md`](paper/evaluation_design.md)
- v1 effort, superseded but kept for the record: [`paper/distillation_working_paper.md`](paper/distillation_working_paper.md)

## Headline finding
On the pre-registered primary comparison, distillation significantly beats the formatting control (+11.6pp checklist pass-rate, p<0.001, every seed) and improves classification. But the per-check breakdown shows the aggregate tie with few-shot prompting is a *trade*: fine-tuning buys concrete takeaways and best-in-class urgency labeling (beating even the teacher) while **regressing on faithfulness** and collapsing on tone-labeling. The actionable conclusion is a **per-field engine assignment**, not a single winner. See the paper §6–§7.

## Reproduce the results (offline, no API keys)
Core analysis uses only Node built-ins — no `npm install` required.
```bash
# Full 2-judge (Gemini + Nemotron) scorecard, recomputed from the judge cache:
npm run compile            # == JUDGES_EXCLUDE=gpt-oss-120b,llama-3.1-8b node server/eval/compile.mjs

# Pre-registered paired-bootstrap significance test:
npm run significance       # == node server/eval/paired_bootstrap.mjs

# Per-judge robustness (paper §5.5) — each judge alone:
npm run judge:gemini
npm run judge:nemotron
```
`compile.mjs` only reads `data/eval/judge_cache.jsonl`; it never calls an API. **API keys (`.env`) are needed only to *regenerate* the cache** (`server/eval/score.mjs`, `gen_arms.mjs`), which a reviewer does not need to do.

## Layout
```
paper/            the paper, review manifest, pre-registration, design, pipeline diagram
server/eval/      analysis engine (compile.mjs), judge panel + clients (judges.mjs),
                  significance (paired_bootstrap.mjs), cache/arm generators (score.mjs, gen_arms.mjs)
data/eval/        test/train sets, per-arm model outputs, judge cache, scorecards, split,
                  and system_prompt.txt (the generation prompt)
notebooks/        Unsloth QLoRA training notebook (3 seeds)
models/           trained GGUFs (~1.5 GB, git-ignored — regenerate via notebooks/)
```

## Toolchain (all open source)
Teacher & student inference: **Ollama**. Fine-tuning: **Unsloth** QLoRA on a free Colab T4. Models: **Qwen3-0.6B** and **DeepSeek-R1:8b** (open weights). Judges: **Gemini Flash Lite** (Google AI Studio) and **Nemotron** (NVIDIA NIM), reference-free. No proprietary training platform is involved.

## Notes
- **`models/` is git-ignored** (1.5 GB of GGUF binaries). Regenerate from `notebooks/unsloth_qwen3_0_6b_3seeds.ipynb`.
- **`.env` is git-ignored** and holds judge API keys (Groq / Google AI Studio / NVIDIA NIM). Only needed to regenerate the judge cache; reproducing the reported metrics needs no keys.
- **Do not reimplement `keyOf`** in `server/eval/judges.mjs` — the cache key is byte-sensitive; a retyped copy silently misses the cache 100%. Extend `compile.mjs` in place instead.
- The datasets in `data/eval/` are provided pre-built; the raw data-generation/sampling scripts are not part of this repo (the reported results reproduce from `server/eval/` + `data/eval/` alone).
