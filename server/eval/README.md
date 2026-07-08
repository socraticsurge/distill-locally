# Evaluation harness — reproduce the scorecard

All arms are generated locally at **temperature 0**; judge calls are **cached**
(`data/eval/judge_cache.jsonl`), so the entire scorecard recomputes **offline with no API
keys** — only regenerating the cache needs keys.

## The one canonical result

```
node --env-file=.env server/eval/score.mjs
```

writes **`data/eval/scores.json`** (structure, classification vs. panel proxy-gold + agreement,
summary-checklist pass-rate ± 95% CI, per-arm × per-check matrix, negative control) and
**`data/eval/scores_detail.json`** (per-item per-check majorities, used by the length and
neutral-subset analyses). Full-source grading is the default (`ARTICLE_CHARS=100000`).

With the committed cache this runs in a fraction of a second and needs **no keys**. To regrade
from scratch you need `.env` with `GOOGLE_API_KEY` (Gemini) and `NVIDIA_API_KEY` (Nemotron +
Mistral-Small).

## Judge panel (canonical)

Three independent model families, none sharing the teacher's (DeepSeek-R1-Distill-Llama,
Llama) or student's (Qwen) lineage — a true 3-way majority:

| judge | provider | key |
|---|---|---|
| `gemini-flash-lite` | Google AI Studio | `GOOGLE_API_KEY` |
| `nemotron-550b` | NVIDIA `integrate` | `NVIDIA_API_KEY` |
| `mistral-small-119b` | NVIDIA `integrate` | `NVIDIA_API_KEY` |

Groq `gpt-oss-120b` / `llama-3.1-8b` were evaluated during panel selection but dropped
(free-tier instability). A fully-local `phi4-mini` cross-check scorecard is kept at
`data/eval/scores_phi_crosscheck.json` (produced by temporarily swapping the third judge in
`judges.mjs` for `{id:'phi4-mini', provider:'ollama', model:'phi4-mini:3.8b'}`).

## Regenerating the cache (needs keys)

1. **Arms** (local Ollama, no keys) — one JSON object per test article, temp 0:
   ```
   node server/eval/gen_arms.mjs base
   node server/eval/gen_arms.mjs fewshot
   node server/eval/gen_arms.mjs constrained
   node server/eval/gen_arms.mjs tuned rss-tuned-s42     # R1-teacher student, seed 42 (also s123, s7)
   node server/eval/gen_arms.mjs tuned rss-llama-s42     # Llama-teacher student, seed 42 (also s123, s7)
   ```
   Teacher self-arm (each 8B teacher graded on the test set under the same rubric):
   ```
   node server/eval/gen_teacher_arm.mjs teacher_llama ollama llama3.1:8b-instruct-q4_K_M
   ```
   (The R1 teacher's outputs ship in `test.jsonl`; the managed-pipeline student `arm_distil` is
   provided.)

2. **Score** (needs keys — populates the judge cache, then writes the scorecard):
   ```
   node --env-file=.env server/eval/score.mjs
   ```

## Analysis over the scorecard (no keys)

All default to `scores.json` / `scores_detail.json`:

- **Primary + secondary significance:** `node server/eval/paired_bootstrap.mjs`
- **Reasoning-teacher contrast** (R1 vs Llama students): `node server/eval/paired_bootstrap_reasoning.mjs`
- **Teacher-neutral + length robustness:** `node server/eval/robustness_neutral_checks.mjs`
- **Per-judge robustness:** `JUDGES_ONLY=gemini …` / `JUDGES_ONLY=nvidia …`
- **Truncation-artifact analysis:** re-run the score with `ARTICLE_CHARS=1200`.

## Figures

`python3 paper/figures/gen_teachers_figure.py`, `gen_fig3_fig4_fig5.py`, `gen_fig2_fig6.py`,
`gen_fig7.py` — all read `data/eval/scores.json` and write to `paper/figures/`.
