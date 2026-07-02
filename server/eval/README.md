# Eval harness — run sequence

Implements `paper/evaluation_design.md`. All arms generated at **temp 0**; judge calls are **cached** (`data/eval/judge_cache.jsonl`) so rate-limit interruptions never lose work.

## Prereqs
- Split + datasets are provided pre-built in `data/eval/{test.jsonl, train_chat.jsonl, split.json}` (train 401 / test 93).
- `.env` at repo root with `GROQ_API_KEY` + `GOOGLE_API_KEY` + `NVIDIA_API_KEY`. Needed only to (re)generate the judge cache — **reproducing the reported metrics from the cached responses needs no keys** (`node server/eval/compile.mjs`).
- Tuned models imported to Ollama as `rss-tuned-s42`, `rss-tuned-s123`, `rss-tuned-s7` (from the 3-seed Colab notebook).

## Steps
1. **Generate arm outputs** (local Ollama, no keys):
   ```
   node server/eval/gen_arms.mjs base
   node server/eval/gen_arms.mjs fewshot
   node server/eval/gen_arms.mjs constrained
   node server/eval/gen_arms.mjs tuned rss-tuned-s42
   node server/eval/gen_arms.mjs tuned rss-tuned-s123
   node server/eval/gen_arms.mjs tuned rss-tuned-s7
   ```
   (teacher outputs already in `test.jsonl`.)

2. **Pilot the checklist** on ~15 items (needs `.env`):
   ```
   EVAL_LIMIT=15 node server/eval/score.mjs
   ```
   Inspect: per-check headroom, field agreement (subjective if < 0.40), and the negative control (faithful-on-mismatch should be LOW). Revise the `CHECKLIST` in `score.mjs` if a check saturates or judges can't agree.

3. **FREEZE** (dated commit): checklist, judge prompts, headline metric, primary comparison (tuned vs base+constrained), κ=0.40.

4. **Scored run** (full test set):
   ```
   node server/eval/score.mjs
   ```
   → `data/eval/scores.json` (structure, length, classification+agreement, checklist pass-rate ± 95% CI, negative control).

## Analysis over the cache (no keys)
- **Full scorecard:** `JUDGES_EXCLUDE=gpt-oss-120b,llama-3.1-8b node server/eval/compile.mjs` → `data/eval/scores.json` (structure, classification + agreement, checklist pass-rate ± 95% CI, per-arm×per-check matrix, negative control).
- **Significance:** `node server/eval/paired_bootstrap.mjs` (primary + secondary paired-bootstrap).
- **Per-judge robustness:** `JUDGES_ONLY=gemini …` / `JUDGES_ONLY=nvidia …` (paper §5.5).

## Still to wire
- **Pairwise win-rate** (tuned vs base / base+constrained / teacher, both orders) and **topics** relevance beyond coverage.
