#!/usr/bin/env python3
"""Regenerate fig_study_at_a_glance.png (v2: platform branch, eight arms) and
fig6_platform_checklist.png (fresh 8-arm checklist comparison) from the canonical
8-arm scorecard. Run from repo root: python3 paper/figures/gen_fig2_fig6.py"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Figure 2: study at a glance, v2 ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 11.6), dpi=133)
ax.set_xlim(0, 100); ax.set_ylim(0, 104); ax.axis("off")

def box(x, y, w, h, text, fc="#eceff3", ec="#5a6675", lw=1.8, fs=13, bold_first=False, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                       fc=fc, ec=ec, lw=lw, linestyle="--" if dashed else "-")
    ax.add_patch(p)
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x + w/2, y + h/2 + 1.3, lines[0], ha="center", va="center", fontsize=fs, fontweight="bold")
        ax.text(x + w/2, y + h/2 - 1.6, "\n".join(lines[1:]), ha="center", va="center", fontsize=fs - 1.5)
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold_first else "normal")

def arrow(x1, y1, x2, y2, color="#4a5462", lw=2.2, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22,
                                 color=color, lw=lw, linestyle="--" if dashed else "-"))

def step(n, y):
    ax.add_patch(mp.Ellipse((4, y), 3.6, 4.4, fc="#2b2f36", ec="none"))
    ax.text(4, y, str(n), color="white", ha="center", va="center", fontsize=15, fontweight="bold")

ax.text(50, 102.5, "Study at a glance:  distilling into a 0.6B on-device student — two training pipelines",
        ha="center", va="center", fontsize=19, fontweight="bold")

step(1, 96);  box(12, 93, 76, 6, "500 news articles  ·  sampled across all subscribed RSS feeds", fs=14, bold_first=True)
arrow(50, 92.4, 50, 89.2)
step(2, 85);  box(12, 81.5, 76, 7, "TEACHER  —  DeepSeek-R1 8B  (reasoning model, runs on-device)\nlabels every article → one gold JSON  ·  ~39 s / article", fc="#dbe5f4", ec="#2f5ba8")
arrow(50, 80.9, 50, 77.7)
step(3, 74);  box(24, 70.5, 52, 6.5, "Stratified split  (every feed represented)", fs=14, bold_first=True)
arrow(38, 69.9, 29, 66.7); arrow(62, 69.9, 71, 66.7)
box(10, 60.5, 38, 6, "401 training articles\n(teacher's labels)")
box(54, 60.5, 36, 6, "93 held-out test articles")

step(4, 51)
arrow(22, 59.9, 15.5, 55.2, color="#7b4fb5"); arrow(36, 59.9, 42.5, 55.2, color="#c07a1f")
box(5, 45.5, 21.5, 9, "TRAIN (DIY)  —  QLoRA\nQwen3-0.6B · 3 seeds\n(Unsloth, free Colab T4)\nteacher outputs, verbatim", fc="#e9dff3", ec="#7b4fb5", fs=11.5)
box(30.5, 45.5, 24, 9, "TRAIN (PLATFORM)  —  managed service\nsingle run\nteacher gpt-oss-120b → synthetic\ndata expansion", fc="#f7e8d4", ec="#c07a1f", fs=11.5, dashed=True)
arrow(15.5, 44.9, 32, 38.0, color="#7b4fb5")
arrow(42.5, 44.9, 46, 38.0, color="#c07a1f", dashed=True)
arrow(72, 59.9, 72, 38.0)

step(5, 33)
box(9, 29.5, 82, 8, "EIGHT ARMS generate the full JSON on the 93 test articles  (temperature 0 / greedy)\n"
    "Teacher (R1-8B) | Base 0.6B zero-shot | + few-shot | + constrained JSON | Tuned × 3 seeds | Platform-distilled ×1",
    fc="#e6efe0", ec="#4a7c3f", fs=13.5, bold_first=True)
arrow(50, 28.9, 50, 25.7)
ax.text(50, 24.2, "Each JSON is split into THREE sub-tasks and scored separately",
        ha="center", va="center", fontsize=14, fontweight="bold")

step(6, 18)
box(8, 14, 26, 6.5, "SUMMARY\n8-item checklist (pass-rate)", fc="#e6efe0", ec="#4a7c3f", fs=12)
box(37, 14, 26, 6.5, "CLASSIFICATION\n5 labels vs. panel proxy-gold", fc="#e6efe0", ec="#4a7c3f", fs=12)
box(66, 14, 26, 6.5, "TOPICS\nopen-set coverage", fc="#e6efe0", ec="#4a7c3f", fs=12)
for x in (21, 50, 79): arrow(x, 23.4, x, 21.2, color="#4a7c3f")
box(8, 4.5, 55, 6.5, "JUDGE PANEL  —  Gemini Flash Lite + Nemotron-550B  (2 families)\nblinded · reference-free · grades vs. FULL article · negative control (0% on mismatch)",
    fc="#f7ecd4", ec="#b98a1e", fs=11.5)
box(66, 4.5, 26, 6.5, "STATS\npaired bootstrap +\ngap-closure to teacher", fc="#f4dede", ec="#b13a3a", fs=11.5)
plt.savefig("paper/figures/fig_study_at_a_glance.png", bbox_inches="tight", facecolor="white")
plt.close()
print("fig_study_at_a_glance.png written")

# ── Figure 6: fresh 8-arm checklist comparison ──────────────────────────────
d = json.load(open("data/eval/scores_gemini_nemotron_n93_fullctx_8arm.json"))
ck = d["checklist"]
seeds = ["tuned_rss_tuned_s1", "tuned_rss_tuned_s2", "tuned_rss_tuned_s3"]
tuned_vals = [ck[s]["checklistPassRate"] for s in seeds]
rows = [
    ("Teacher (R1-8B)", ck["teacher"]["checklistPassRate"], ck["teacher"]["ci95"], "#2f5ba8"),
    ("DIY tuned (3-seed mean)", sum(tuned_vals)/3, [min(tuned_vals), max(tuned_vals)], "#7b4fb5"),
    ("Platform-distilled", ck["distil"]["checklistPassRate"], ck["distil"]["ci95"], "#c07a1f"),
    ("Base + few-shot", ck["fewshot"]["checklistPassRate"], ck["fewshot"]["ci95"], "#8a929e"),
    ("Base + constrained", ck["constrained"]["checklistPassRate"], ck["constrained"]["ci95"], "#8a929e"),
    ("Base zero-shot", ck["base"]["checklistPassRate"], ck["base"]["ci95"], "#8a929e"),
]
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
ys = range(len(rows))[::-1]
for y, (label, v, ci, c) in zip(ys, rows):
    ax.barh(y, v, color=c, alpha=0.85, height=0.62)
    ax.errorbar(v, y, xerr=[[v - ci[0]], [ci[1] - v]], color="#2b2f36", capsize=4, lw=1.6)
    ax.text(ci[1] + 1.2, y, f"{v:.1f}", va="center", fontsize=11, fontweight="bold")
ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.set_xlim(0, 100); ax.set_xlabel("Summary checklist pass-rate (%) — fresh 8-arm grading, N = 93", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Platform-distilled vs. DIY: a statistical tie on summary quality (Δ−3.6 [−8.5, +1.1])",
             fontsize=12.5, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("paper/figures/fig6_platform_checklist.png", bbox_inches="tight", facecolor="white")
print("fig6_platform_checklist.png written")
