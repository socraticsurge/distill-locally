# Pre-registration — 500-article distillation eval

**Frozen: 2026-07-02**, after the Groq-only pilot on 12 train articles, **before** the scored run.
Nothing below is revised after seeing scored results. Companion to `evaluation_design.md`.

## Dataset
- Stratified-random split: **401 train / 93 test** (`data/eval/split.json`), every feed represented; test = new articles from known feeds (deployment-realistic).
- Teacher = `deepseek-r1:8b`; student = `Qwen3-0.6B`.

## Arms (all generated at temperature 0)
teacher · base zero-shot · base+few-shot · base+constrained-decoding · tuned ×3 seeds (rss-tuned-s1/s2/s3).

## Primary comparison (pre-registered)
**Tuned vs. base+constrained**, on the summary **checklist pass-rate**, via paired bootstrap (95% CI).
Secondary (not headline): tuned vs. base, tuned vs. few-shot, and per-field classification.

## Frozen summary checklist — 8 binary checks
1. **faithful** — every factual claim is supported by the article (no hallucination/contradiction).
2. **thesis** — captures the article's central thesis/main finding.
3. **takeaway** — includes a concrete, specific takeaway (not vague).
4. **length** — 3–4 sentences.
5. **opening** — does NOT begin with "This article"/"The article".
6. **teacher** — teacher lens: explains a concept accessibly / builds intuition.
7. **tech** — technologist lens: addresses the technical/engineering angle.
8. **tone** — direct + contemplative + optimistic (not alarmist/generic).

Plus **topics_cover** (do the listed topics capture the article's main themes).

**Dropped at freeze (pilot showed no headroom):** `exec` (8.3% pass — near-broken) and `topics_relevant` (95% — saturated). Dropped, not reworded, to avoid any post-hoc wording bias.

## Classification
5 categorical fields (sentiment/urgency/frame/tone/depth) scored as macro-accuracy vs. **panel-consensus proxy-gold** (majority of the 3 judges). Any field with inter-judge **κ < 0.40** is reported separately as *irreducibly subjective at 0.6B*, not fixed by widening definitions.

## Judge panel — 3 families
`openai/gpt-oss-120b` (Groq) · `llama-3.3-70b-versatile` (Groq) · `gemini-3.1-flash-lite` (Google AI Studio). Excludes Qwen (student) and DeepSeek (teacher) families. Grading is reference-free (against the article), blinded (shuffled anonymous labels), temperature 0.

## Grader validation
**Negative control passed in the pilot: 0% faithful on mismatched articles** — the grader demonstrably discriminates real from mismatched summaries.

## Amendment — 2026-07-02 (resource-driven, not result-driven)
- **Judge panel:** `llama-3.3-70b-versatile` → **`llama-3.1-8b-instant`**. Reason: llama-3.3-70b free-tier **TPD = 100k** was exhausted; 8b-instant has a far larger daily token budget so the run completes same-day. Still 3 distinct families (OpenAI / Meta / Google); still excludes Qwen (student) & DeepSeek (teacher). Caveat: 8b is a weaker judge — reported per-judge κ + the negative control will flag if its votes are unreliable.
- **Token trims (to fit free-tier daily token caps):** article context in judge prompts 6000→1200 chars; `max_completion` 2000→800. Uniform across all arms, so comparisons stay fair; absolute faithfulness may read slightly low since judges see only the article lead.
- **N:** scored on a **stratified N=40** subsample of the 93-item test set (still 2× the original n=20). These are pre-committed *before* seeing scores.

## Notes
- Pairwise win-rate runs on a separate day / Groq-only (Gemini daily budget).
- Pilot diagnostics that informed this freeze are in `data/eval/pilot_score.log` / `scores.json`.
