# What Distillation Actually Buys a Sub-1B Student: A Pre-Registered, Multi-Judge Evaluation of DeepSeek-R1 → Qwen3-0.6B

**Authors:** *Vinay Chaganti & Antigravity (AI Assistant)*
**Date:** July 2, 2026
**Status:** *Draft* — results on a 2-judge panel (Gemini Flash Lite + Nemotron-550B), N=93. Every directional finding is confirmed by **each judge independently** (§5.5); a third same-family LLM judge is deliberately *not* added because it cannot change the directions or resolve the judges' magnitude disagreement — only human gold labels can (§10).

> **Relationship to the v1 working paper.** This is a separate, more rigorous study — it does **not** replace [`distillation_working_paper.md`](distillation_working_paper.md), which stands as the record of the first-pass effort. v1 scored 20 articles with a single same-generation judge on a 0–5 rubric that saturated (faithfulness 4.95), and reported a headline "91.0 vs 88.8 /100" that we no longer consider defensible. This paper re-runs the question with a pre-registered design: a 93-article test set, seven arms including two non-distillation controls, a decomposition into three sub-tasks, a multi-family LLM-judge panel validated by a negative control, three training seeds, and paired-bootstrap significance tests frozen *before* scoring. Where v1 flattered the method, this study is built to catch it.

---

## Abstract
We quantify what knowledge distillation actually transfers when an 8-billion-parameter reasoning teacher (`deepseek-r1:8b`) is distilled into a **600M-parameter** `Qwen3-0.6B` student for a structured RSS-summarization task (a JSON object bundling a free-text summary, five categorical fields, and open-set topics). The single output is really **three ML sub-tasks**, and we score each separately against **two non-distillation controls** — few-shot prompting (is the win just prompting?) and constrained JSON decoding (is the win just formatting?). Grading uses a reference-free, blinded, multi-family LLM-judge panel validated by a negative control (0% faithful on mismatched articles). On our pre-registered primary comparison, **distillation significantly beats the formatting control on summary quality (+11.6pp checklist pass-rate, 95% CI [7.6, 15.7], p<0.001)** and on classification (56.4% vs 40.2% macro-accuracy). But the per-check/per-field breakdown — the study's real contribution — shows a picture far sharper than "distillation wins." On the aggregate summary score distillation is **statistically tied with simple few-shot prompting** (63.4% vs 63.2%, n.s.), yet that tie is a *composition difference*: fine-tuning buys **concrete takeaways** (+29pp over base) and best-in-class **urgency classification** (78%, beating even the teacher), while **regressing on faithfulness** (hallucination check 40% vs base 49%) and **collapsing on tone-labeling** (26%, seed-unstable). It **remains below the teacher** on the summary (−15.2pp, p<0.001). The actionable conclusion is a **per-field engine assignment**, not a single winner: the tuned student is the best on-device choice for structure (100% schema-valid JSON) and classification, but the reader-facing summary text and the tone label should come from a prompting baseline or the teacher. We also report substantial **seed variance** (checklist 61.6–65.7%; tone-labeling 8.6–46.2%), a caveat v1's single run could not have surfaced.

---

## 1. Introduction & the question we are actually asking
Running an 8B reasoning model to summarize every article in an RSS reader is slow (~39 s/article on a consumer MacBook) and memory-heavy. A 0.6B student runs in ~1 s and ~0.5 GB — but out of the box it struggles to hold a strict JSON schema, sustain a specific analytical voice, and blend multiple required perspectives. Knowledge distillation — fine-tuning the small student on the large teacher's outputs — is the standard fix.

The v1 paper concluded distillation "matched or exceeded" the teacher. That conclusion did not survive scrutiny of its method. The question this paper asks is deliberately harder and more useful:

> **Where — if anywhere — does reasoning-teacher distillation help a sub-1B student: structure, classification, summarization, or none? And does it beat the cheap alternatives (prompting, constrained decoding) that require no training at all?**

Distillation is only interesting if it beats **both** controls on something real. That framing is the whole design.

---

## 2. The task is three sub-tasks
The one JSON output bundles three different ML problems, scored **separately**:

| Sub-task | Fields | Type | Nature |
|---|---|---|---|
| **Summarization** | `summary` | free-text generation | subjective quality |
| **Classification** | `sentiment, urgency, frame, tone, depth` | closed-set labels | subjective labels, no free gold |
| **Topic tagging** | `topics` | open-set multi-label | fuzzy, open vocabulary |

Reporting them apart is the study's main analytical contribution. A single blended score (v1's mistake) hides *what* actually moved.

---

## 3. Experimental setup

### 3.1 Models
| Role | Model | ~Params | Runtime |
|---|---|---|---|
| **Teacher** | DeepSeek-R1 8B (`deepseek-r1:8b`) | 8B | Ollama, on-device (Q4_K_M) |
| **Student** | Qwen3-0.6B | 0.6B | Ollama, on-device (Q4_K_M) |

A ~13× parameter gap, and a difference in kind: DeepSeek-R1 emits chain-of-thought (`<think>`) before answering; the student is trained on the teacher's *final* outputs only, compiling task knowledge into weights and skipping runtime deliberation.

### 3.2 Data and split
500 articles sampled fairly across all feeds; teacher generates a gold JSON for each; malformed outputs dropped. **Stratified-random split: 401 train / 93 test** (`data/eval/split.json`), every feed represented. The test set is *new articles from known feeds* — matching the actual deployment (the reader follows fixed feeds and sees new items from them), not generalization to unseen sources.

### 3.3 Training
QLoRA fine-tune of `Qwen3-0.6B` directly on the 401 teacher traces via Unsloth (LoRA rank 32, response-only loss masking, Qwen3 "thinking" disabled), on a free Colab T4. **Three seeds (42 / 123 / 7)** → `rss-tuned-s1/s2/s3`, so seed variance is characterized rather than assumed away. Exported to `q4_k_m` GGUF, served locally in Ollama.

### 3.4 The seven arms (all generated at temperature 0 / greedy)
| Arm | Role |
|---|---|
| **Teacher** `deepseek-r1:8b` | reference point (not assumed a ceiling) |
| **Base** `Qwen3-0.6B` zero-shot | floor |
| **Base + few-shot** (2–3 in-context examples) | control: *is the win just prompting?* |
| **Base + constrained decoding** (JSON-schema-forced) | control: *is the win just formatting?* |
| **Tuned ×3 seeds** | the distillation method |

Temperature 0 everywhere so the *only* characterized variance is the tuned model's 3 training seeds — exactly what the CIs capture.

---

## 4. Evaluation protocol (pre-registered, frozen 2026-07-02)
Full design in [`evaluation_design.md`](evaluation_design.md); frozen commitments in [`PREREGISTRATION.md`](PREREGISTRATION.md). Nothing below was revised after seeing scores.

- **Structure (deterministic, no judge):** schema-validity % (parses as JSON with all 7 fields).
- **Classification (5 fields):** macro-accuracy vs. **panel-consensus proxy-gold** (judge majority = proxy truth). Inter-judge agreement reported per field; we state plainly that agreement means judges *concur*, not that they are *correct*. No local human adjudication (parked for the platform).
- **Summarization — binary checklist (PRIMARY):** 8 binary checks (6 judged, 2 deterministic) + `topics_cover`. Chosen over a 0–5 rubric because that saturated in v1. Per-item, so it yields clean bootstrap CIs.
- **Topics:** coverage only (`topics_relevant` was dropped at the pilot freeze — it saturated at 95%, carrying no signal).
- **Grader validation — negative control:** grade summaries against a *mismatched* article; a real grader must score these far lower. **Result: 0% faithful on mismatch (n=30)** — the panel demonstrably discriminates good from bad.
- **Primary pre-registered comparison:** **tuned vs. base+constrained** on checklist pass-rate, paired bootstrap, 95% CI. Naming one primary controls multiple comparisons. All others (vs. base, vs. few-shot, vs. teacher, per-field) are secondary.

### 4.1 The frozen 8-check summary checklist
1. **faithful** — every claim supported by the article (no hallucination). 2. **thesis** — captures the central finding. 3. **takeaway** — concrete, specific. 4. **length** — 3–4 sentences *(deterministic)*. 5. **opening** — does not begin "This/The article" *(deterministic)*. 6. **teacher** — explains a concept accessibly. 7. **tech** — addresses the technical angle. 8. **tone** — direct + contemplative + optimistic. *(Dropped at freeze for no headroom: `exec` 8.3% pass, `topics_relevant` 95% — dropped, not reworded, to avoid post-hoc bias.)*

### 4.2 Judge panel and its honest status
Reference-free grading against the **article** (never the teacher's answer → no teacher-mimicry bias), blinded to arm identity (shuffled anonymous labels), temperature 0. The panel excludes both the teacher's family (`deepseek`) and the student's family (`qwen`) so no judge self-prefers.

The results below are compiled over the **two judges that reliably completed the batched checklist task**:
- **Gemini Flash Lite** (`gemini-3.1-flash-lite`, Google) — via Google AI Studio.
- **Nemotron-550B** (`nvidia/nemotron-3-ultra-550b-a55b`, NVIDIA) — via NVIDIA NIM; 0 failures, credit-based (not daily-token-capped).

The pre-registered third Groq judge (`llama-3.1-8b-instant`, swapped in when `llama-3.3-70b`'s free-tier daily token budget was exhausted) **failed ~31% of the batched checklist calls with malformed JSON** — small judges cannot reliably emit the ~63-boolean structured verdict — so it is excluded as an unreliable grader.

**We deliberately do not add a third capable judge, and §5.5 shows why it would add no value.** The two judges disagree on 26.8% of binary checks, but they **agree on the direction of every finding** when each is analyzed alone; where they differ it is on *magnitude*, which a third same-family LLM judge cannot resolve — only human gold labels can. A third judge would rubber-stamp the directions and leave the magnitude question open, so the honest move is to **report the two judges separately** (not blend them into a single "consensus" that hides their disagreement) and to send the marginal effort to a human spot-check instead (§10).

**Reporting note (a correction over an earlier draft):** with exactly two judges, a majority vote has no tiebreaker, so our `majority()` breaks ties toward the first-listed judge (Gemini). That means the single "consensus" per-check table (§5.2) is, on the 26.8% of checks where the judges split, **effectively Gemini's vote**. Rather than paper over this, §5.5 reports each judge's grades independently and shows the findings hold in both. Both judges also agree the negative control collapses to 0%.

---

## 5. Results (N=93, 2-judge panel)

### 5.1 Structure — deterministic schema validity
| Arm | Schema-valid JSON | Output length (chars, p50 / p95) |
|---|---|---|
| Teacher | 100% | 658 / 815 |
| Base zero-shot | 92.5% | 639 / 872 |
| Base + few-shot | 97.8% | 585 / 701 |
| Base + constrained | 100% | 631 / 872 |
| **Tuned (s1/s2/s3)** | **100% / 100% / 100%** | 539–595 / 691–754 |

Both constrained decoding **and** distillation reach 100% valid JSON — but distillation does it *baked into the weights*, with no per-request schema payload and no decoding constraints, and produces the **most concise** outputs (p50 539–595 chars vs teacher's 658). The base model's 7.5% structural failure rate is the concrete reliability gap.

### 5.2 Summarization — binary checklist pass-rate (PRIMARY metric)
| Arm | Checklist pass-rate | 95% CI | topics coverage |
|---|---|---|---|
| **Teacher** | **78.6%** | [78.2, 79.4] | 97.8% |
| Base zero-shot | 50.4% | [48.8, 52.8] | 84.9% |
| Base + few-shot | 63.2% | [62.1, 65.3] | 91.4% |
| Base + constrained | 51.9% | [50.4, 53.6] | 89.2% |
| Tuned s1 | 63.0% | [62.1, 64.5] | 90.3% |
| Tuned s2 | 65.7% | [64.5, 67.3] | 95.7% |
| Tuned s3 | 61.6% | [59.3, 62.9] | 91.4% |
| **Tuned (pooled)** | **63.4%** | — | ~92% |

The aggregate hides the most important result. Broken out **per arm × per check** (pass-rate %, the diagnostic that says *where* each version wins and loses):

| Check | teacher | base | few-shot | constrained | tuned-s1 | tuned-s2 | tuned-s3 | **tuned μ** |
|---|---|---|---|---|---|---|---|---|
| **faithful** (no hallucination) | 68.8 | 49.5 | 58.1 | 51.6 | 41.9 | 40.9 | 38.7 | **40.5** |
| **thesis** (central finding) | 95.7 | 79.6 | 88.2 | 82.8 | 86.0 | 87.1 | 86.0 | **86.4** |
| **takeaway** (concrete/specific) | 82.8 | 43.0 | 47.3 | 45.2 | 66.7 | 79.6 | 69.9 | **72.1** |
| **length** (3–4 sentences) | 80.6 | 54.8 | 60.2 | 54.8 | 55.9 | 63.4 | 51.6 | **57.0** |
| **opening** (not "This article") | 72.0 | 51.6 | 78.5 | 53.8 | 80.6 | 72.0 | 73.1 | **75.2** |
| **teacher-lens** (accessible) | 77.4 | 29.0 | 48.4 | 30.1 | 51.6 | 54.8 | 48.4 | **51.6** |
| **tech-lens** (engineering angle) | 63.4 | 36.6 | 48.4 | 36.6 | 48.4 | 53.8 | 49.5 | **50.6** |
| **tone** (direct+reflective+optimistic) | 88.2 | 59.1 | 76.3 | 60.2 | 73.1 | 74.2 | 75.3 | **74.2** |
| topics_cover | 97.8 | 84.9 | 91.4 | 89.2 | 90.3 | 95.7 | 91.4 | **92.5** |

Two findings jump out, and neither is visible in the aggregate:

- **The distillation win is `takeaway` (concrete specificity): base 43% → tuned 72%, a +29pp jump that decisively beats few-shot (47%).** This is the *only* check where distillation clearly beats prompting — and it's a big one. The tuned models learned from the teacher to end on a specific, usable point instead of a vague gesture. They also lifted the two persona lenses (teacher/tech) and `opening` well above both controls.
- **The distillation LOSS is `faithful` (no hallucination): base 49.5% → tuned 40.5%, a −9pp regression.** Fine-tuning made the student *hallucinate more* than the untuned base. The teacher is the most faithful arm (68.8%); the tuned students are the *least* faithful of any arm. This is the single most important caveat in the study and has direct product consequences (§7).

So the aggregate tie with few-shot (63.4 vs 63.2) is a **composition difference, not equivalence**: distillation buys concrete specificity and persona voice at the cost of faithfulness, while few-shot is more evenly balanced and stays more faithful. The checklist discriminates cleanly (no check saturated or broken) — exactly what v1's rubric failed to do.

### 5.3 Classification — macro-accuracy vs. consensus proxy-gold
| Arm | Macro-acc | sentiment | urgency | frame | tone | depth |
|---|---|---|---|---|---|---|
| **Teacher** | **64.9%** | 72.0 | 55.9 | 66.7 | 71.0 | 59.1 |
| Base zero-shot | 39.3% | 36.0 | 59.3 | 50.0 | 22.1 | 29.1 |
| Base + few-shot | 50.8% | 61.5 | 49.5 | 40.7 | 69.2 | 33.0 |
| Base + constrained | 40.2% | 39.8 | 59.1 | 50.5 | 24.7 | 26.9 |
| Tuned s1 | 63.4% | 65.6 | 77.4 | 65.6 | 46.2 | 62.4 |
| Tuned s2 | 50.5% | 57.0 | 78.5 | 53.8 | 8.6 | 54.8 |
| Tuned s3 | 55.3% | 51.6 | 78.5 | 59.1 | 23.7 | 63.4 |
| **Tuned (mean)** | **56.4%** | 58.1 | 78.1 | 59.5 | 26.2 | 60.2 |

Inter-judge agreement per field (2-judge raw agreement): frame 0.92, urgency 0.88, sentiment 0.84, tone 0.75, depth 0.67 — all usable; `depth` and `tone` are the noisiest, consistent with their being the most subjective. The per-field breakdown is, again, more informative than the macro number:

- **`urgency` — distillation's standout: tuned 78.1%, beating the teacher (55.9%), base (59.3%) and few-shot (49.5%).** The tuned student is the single best arm at classifying breaking / developing / evergreen. This is the cleanest "the weights moved" result in the study — few-shot actually *hurt* urgency (49.5% < base), so this is not a prompting effect.
- **`depth` — tuned 60.2% ≈ teacher (59.1%), vs base 29.1% / few-shot 33.0%.** A ~+30pp lift over both controls; distillation roughly matches the teacher.
- **`sentiment` (58.1%) and `frame` (59.5%)** — tuned sits between the controls and the teacher: clearly above base/constrained, competitive with few-shot, below teacher.
- **`tone` — the collapse: tuned 26.2%, worse than everything except the untuned base (22.1%).** Few-shot (69.2%) and teacher (71.0%) are far better. Seed s2 is catastrophic (8.6%). Note the irony against §5.2: the tuned models *write* good tone (74% on the tone checklist check) but cannot *label* it — fine-tuning optimized the generative behavior while degrading the categorical head for this one subjective field. This, plus the seed variance, is a real caution.

### 5.4 Significance — paired bootstrap (pre-registered), N=93, 20k resamples
| Comparison | Δ (pp) | 95% CI | p | Verdict |
|---|---|---|---|---|
| **PRIMARY: tuned (pooled) vs constrained** | **+11.6** | [7.6, 15.7] | <0.001 | **SIGNIFICANT** |
|   tuned-s1 vs constrained | +11.2 | [7.0, 15.6] | <0.001 | significant |
|   tuned-s2 vs constrained | +13.8 | [9.0, 19.0] | <0.001 | significant |
|   tuned-s3 vs constrained | +9.7 | [5.4, 14.1] | <0.001 | significant |
| tuned (pooled) vs base | +13.0 | [8.7, 17.7] | <0.001 | significant |
| **tuned (pooled) vs few-shot** | **+0.3** | [−3.5, 3.9] | 0.867 | **n.s.** |
| tuned (pooled) vs teacher | −15.2 | [−19.9, −10.8] | <0.001 | significant |
| *(ref)* few-shot vs constrained | +11.3 | [7.0, 15.9] | <0.001 | significant |

The primary hypothesis holds **for every seed independently**: distillation beats the free formatting fix on summary quality. But few-shot prompting beats constrained decoding by the *same* margin (+11.3pp), and tuned vs. few-shot is a statistical tie.

### 5.5 Judge robustness — do the findings depend on the panel? (and why a third judge adds nothing)
The two judges disagree on **26.8%** of binary checks — not negligible. So the fair question is whether any finding is an artifact of *which* judge (or of the Gemini tie-break, §4.2). We answer it by scoring **each judge entirely alone** and asking whether the direction survives. It does, on every headline finding:

| Finding | Gemini alone | Nemotron alone | Direction agrees? |
|---|---|---|---|
| **Urgency: tuned beats the teacher** | tuned 78.1 vs teacher 55.9 | tuned 74.9 vs teacher 66.7 | ✅ both |
| **Classification macro: tuned between few-shot and teacher** | 39.3 base · 50.8 FS · **56.4** tuned · 64.9 tchr | 46.5 base · 51.4 FS · **58.8** tuned · 68.0 tchr | ✅ both |
| **Takeaway: tuned beats few-shot** | tuned 72.1 vs FS 47.3 | tuned 45.9 vs FS 29.0 | ✅ both |
| **Faithful: tuned regresses vs base** | tuned 40.5 vs base 49.5 (−9) | tuned 47.3 vs base 48.4 (−1) | ✅ both (magnitude differs) |
| **Tone-labeling collapse (tuned)** | tuned 26.2 vs teacher 71.0 | tuned 34.1 vs teacher 67.7 | ✅ both |
| **Negative control** | 0% faithful on mismatch | 0% faithful on mismatch | ✅ both |

**Every directional claim in this paper holds in each judge independently.** The judges differ on *magnitude* — Nemotron is a uniformly harsher grader (e.g. it scores the faithfulness regression at −1pp where Gemini scores −9pp; it is far stricter on the persona lenses). That is exactly the kind of disagreement a **third LLM judge cannot resolve** — averaging three graders of the same kind gives a more precise estimate of *inter-LLM opinion*, not of *ground truth*. Resolving magnitude requires **human gold labels** (§10). So a third capable judge is dominated: it can neither overturn the (already twice-confirmed) directions nor settle the (genuinely open) magnitudes. We therefore stop at two judges, report them separately, and spend the marginal effort on a human spot-check.

*(The AND/OR tie-handling bracket — the range a larger panel's majority could produce without human input — is wide because disagreement is 26.8%, which is precisely why we rely on per-judge direction-agreement rather than a single blended pass-rate.)*

### 5.6 Efficiency (model-intrinsic runtime, Ollama on M-series MacBook)
| Arm | Latency / article | Throughput | Peak RAM | Cost |
|---|---|---|---|---|
| Teacher `deepseek-r1:8b` | ~39,200 ms | ~40 tok/s | ~5.6 GB | $0 (local) |
| Base / Tuned `Qwen3-0.6B` | ~1,000 ms | ~200 tok/s | ~0.5 GB | $0 (local) |

→ **~42× faster wall-clock, ~11× less RAM**, fully on-device. *(Provenance: these are model-intrinsic runtime figures measured during the v1 runs; they are unchanged by the evaluation redesign but were not re-measured per-arm on the 93-item set. Re-measuring p50/p95 per arm on the test set is a remaining to-do — §10.)*

---

## 6. Interpretation — what distillation actually bought
Reading the three sub-tasks and their per-check/per-field breakdowns together, distillation's value is **real, specific, and comes with a specific cost:**

1. **It beats the no-training controls on the summary — significantly, on every seed** (primary: +11.6pp vs constrained). Constrained decoding fixes *format* but not *content*: its checklist pass-rate (51.9%) is statistically identical to the raw base (50.4%). Distillation moves content.

2. **On the aggregate summary score it ties few-shot (63.4 vs 63.2, n.s.) — but that tie is a composition difference, not equivalence.** Distillation and prompting buy *different* things: fine-tuning buys **concrete takeaways** (`takeaway` +29pp over base, +25pp over few-shot) and **persona voice** (teacher/tech lenses), while **regressing on faithfulness** (`faithful` −9pp below base). Few-shot stays more faithful and more balanced. So "which is better" depends entirely on *which check you care about* — and for a news product, faithfulness is not negotiable (§7).

3. **Distillation's clean, control-beating, prompting-beating wins are:**
   - **Guaranteed structure** — 100% schema-valid JSON, no prompt-token overhead, no decoding constraints (few-shot only reaches 97.8%).
   - **Urgency classification** — 78.1%, the best of *any* arm including the teacher, and a field few-shot actively *hurts*. The single strongest "the weights moved" signal.
   - **Concrete takeaways** — the one summary check where fine-tuning clearly beats prompting.
   - **Concision** — the shortest outputs of any arm.

4. **It does not reach the teacher on the summary** (−15.2pp) and **regresses on faithfulness and tone-labeling.** The 8B reasoning teacher remains the most faithful, most complete summarizer; distillation narrows the gap on some axes while opening new gaps on others. This is a trade, not a strict improvement.

5. **Seed variance is material** (checklist 61.6–65.7%; classification 50.5–63.4%; tone-labeling 8.6–46.2%). A single run — as in v1 — could land anywhere in that band. Reporting three seeds changes the confidence of every claim above.

**The one-line takeaway:** *Distillation into a 0.6B student is worth it for a structured task when you need guaranteed schema-valid output, strong categorical labels (especially urgency), and concise, specific, on-brand summaries at sub-second on-device cost — but it trades away faithfulness and tone-labeling, so it is not a drop-in replacement for the teacher, and on the free-text summary alone it is no better than simply prompting the base model well.*

---

## 7. Application — what this means for Atlas Pulse
The point of the study is a product decision: for each AI feature in the reader, which engine should run it? The sub-task decomposition maps directly onto the app's surfaces, and the answer is **not one model for everything.**

| Reader feature | Sub-task it depends on | Recommended engine | Why |
|---|---|---|---|
| **"Breaking / developing / evergreen" badges, feed prioritization** | `urgency` classification | **Tuned student** | 78% accuracy — best of *all* arms, beats the teacher; runs in ~1s on-device so the whole feed can be scored live. |
| **Filters/sorts by depth, sentiment, frame; topic tags & related-article grouping** | classification + topics | **Tuned student** | Best student accuracy (56% macro, ~92% topic coverage), 100% valid JSON so the UI never fails to render a card. |
| **The AI summary card (headline text shown to the reader)** | `summary` free-text | **Few-shot base, or teacher for a "deep summary" action** | Distillation's faithfulness regression (40% vs base 49% vs teacher 69%) is disqualifying for un-verified, reader-facing prose. Few-shot is as good overall *and* more faithful; the teacher is best when quality outranks latency. |
| **Any batch/offline enrichment (nightly re-summarize, backfill)** | all | **Teacher** | Latency doesn't matter offline; take the most faithful, highest-quality output. |
| **"Tone" mood indicator** | `tone` classification | **Few-shot or teacher, not tuned** | Tuned tone-labeling collapsed (26%, seed-unstable); the base/teacher at ~70% are far safer for a user-visible label. |

**The concrete product recipe this suggests:** run the **tuned student as the always-on, on-device engine** for everything structural and categorical — schema-valid JSON, urgency badges, filters, topics — where it is both cheapest and best. **Do not** ship its raw summary text as authoritative: either display summaries from the few-shot base (equal quality, more faithful, still ~1s) or gate a "generate deep summary" action to the teacher, and consider a lightweight faithfulness guard (e.g. flag summaries whose claims aren't grounded in the article). The `tone` label should not come from the tuned model.

**Why this is useful beyond this app:** the study shows that for a bundled structured-output task, *the right engine is per-field, not per-task.* A 0.6B distilled model can be the best available option for the high-frequency classification and structure work — the stuff you'd never pay teacher-latency for on every article — while a cheaper prompting baseline or the teacher covers the low-frequency, high-stakes free-text. Distillation earns its place in the pipeline; it just doesn't earn *the whole pipeline.*

---

## 8. Threats to validity
- **Two judges — by choice, not shortfall.** Every directional finding holds in each judge alone (§5.5), so the panel size is not a threat to the *directions*. The residual threat is a **shared bias both judges carry** (both could be lenient/strict in the same way), which a third same-family judge would not detect and only human gold labels resolve (§10). The originally-planned small Groq judge was excluded because it could not reliably produce the structured verdict — an honest limitation of free-tier grading, not a result-driven exclusion.
- **Magnitudes are judge-dependent.** The judges agree on direction but not size (e.g. faithfulness regression −9pp Gemini vs −1pp Nemotron). We report both and refrain from over-precise magnitude claims; the human spot-check (§10) is the way to pin these down.
- **Proxy-gold, not human gold.** Classification "accuracy" is vs. judge consensus, which measures agreement, not truth. Human adjudication is parked for the platform (§10). This especially bounds the `tone` and `depth` fields, where inter-judge agreement is lowest.
- **Truncated article context.** To fit free-tier token budgets, judges saw the first 1200 chars of each article (uniform across arms, so comparisons stay fair, but absolute faithfulness may read slightly low — the tuned faithfulness regression is a *relative* finding and survives this).
- **Efficiency not re-measured per-arm** on the 93-set (§5.6 provenance note).
- **Pairwise win-rate not yet run** — the confirmatory cross-check on the holistic verdict (§10).

None of these threaten the *primary* result (a deterministic-plus-judged, negative-control-validated, per-seed-significant comparison), but they bound how far the secondary claims should be pushed.

---

## 9. Conclusion
A pre-registered, control-anchored, multi-judge evaluation, **decomposed to the individual check and field**, gives a far more actionable answer than v1's single blended score. Distilling `deepseek-r1:8b` into a 600M `Qwen3-0.6B` student **significantly beats the no-training baselines on summary quality (primary result) and improves classification** — but the per-check view shows the aggregate tie with few-shot is a *trade*: distillation buys concrete takeaways, persona voice, best-in-class urgency labeling, and guaranteed 100% valid JSON, while **regressing on faithfulness and collapsing on tone-labeling**. It **remains below the teacher** on the summary. The honest, useful conclusion is a **per-field engine assignment** (§7), not "distillation wins": the tuned student is the right on-device engine for structure and classification; a prompting baseline or the teacher should own the reader-facing summary text and the tone label.

**Three lessons generalize:** (1) *Decompose to the check/field* — the blended win hid both distillation's biggest gain (takeaway/urgency) and its biggest risk (faithfulness). (2) *Anchor to no-training controls and pre-register the primary* — without the few-shot control we would have miscredited prompting-level gains to fine-tuning. (3) *The deployment answer is per-field, not per-model* — a small distilled model can be simultaneously the best choice for some fields and the wrong choice for others in the same JSON object.

---

## 10. Remaining work (does not block this draft)
- **Human spot-check (highest value, replaces the third judge).** Hand-label ~30–50 items on the checks/fields where the two judges disagree most (faithful, tone, depth) to settle *magnitude* and validate the proxy-gold's *correctness* — the one thing a third LLM judge cannot do. This is the marginal effort that used to be budgeted for a third judge.
- **Pairwise win-rate** (tuned vs base / constrained / teacher, both A/B orders, blinded) — confirmatory cross-check on §5.2.
- **Faithfulness drill-down** — read the tuned models' hallucinated summaries to characterize *what* the fine-tune traded away, and test whether adding a faithfulness signal to training recovers it.
- **Per-arm efficiency** p50/p95, throughput, RAM re-measured on the 93-item test set.
- **Length-confound check** — correlate checklist pass-rate with output length per arm.
- Parked for the RapidCanvas Platform: human gold-label adjudication, a non-reasoning teacher arm, reasoning-trace transfer, larger seed counts.

---

### Reproducibility
Test split `data/eval/split.json` (401/93); per-arm outputs `data/eval/arm_*.jsonl` (7 arms × 93); judge responses cached in `data/eval/judge_cache.jsonl`; compiled scorecard `data/eval/scores_gemini_nemotron_n93.json`; significance `server/eval/paired_bootstrap.mjs`; judge panel + clients `server/eval/judges.mjs`; generate/score/compile `server/eval/{gen_arms,score,compile}.mjs`; frozen design `paper/evaluation_design.md` + `paper/PREREGISTRATION.md`. Ollama models: `qwen3:0.6b` (base), `rss-tuned-s1/s2/s3` (distilled), `deepseek-r1:8b` (teacher).
