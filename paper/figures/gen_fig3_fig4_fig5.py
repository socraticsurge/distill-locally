#!/usr/bin/env python3
"""Regenerate fig3 (gap-closure), fig4 (checklist), fig5 (efficiency) in the shared
serif style, from the primary-grading numbers reported in the paper tables.
Run from repo root: python3 paper/figures/gen_fig3_fig4_fig5.py"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS
apply_style()

# ---------- Fig 4: summary checklist pass-rate by arm (primary grading) ----------
rows = [  # (label, value, ci_lo, ci_hi, color) — canonical 3-judge grading
    ("Teacher (R1-8B)",      88.6, 86.3, 91.5, COLORS["r1"]),
    ("Tuned (distilled)",    75.1, 73.9, 76.5, COLORS["accent"]),
    ("Base + few-shot",      70.2, 69.4, 71.0, COLORS["base"]),
    ("Base + constrained",   58.2, 54.8, 60.9, COLORS["base"]),
    ("Base zero-shot",       56.3, 53.6, 58.1, COLORS["base"]),
]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ys = list(range(len(rows)))[::-1]
for y, (lab, v, lo, hi, c) in zip(ys, rows):
    ax.barh(y, v, color=c, alpha=0.9, height=0.62)
    ax.errorbar(v, y, xerr=[[v-lo], [hi-v]], color="#2b2f36", capsize=4, lw=1.4)
    ax.text(hi+1.2, y, f"{v:.1f}", va="center", fontsize=10, fontweight="bold")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 100); ax.set_xlabel("Summary checklist pass-rate (%), full-text grading, N = 93")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Distillation beats both non-distillation controls on summaries", fontweight="bold", pad=8)
plt.tight_layout(); plt.savefig("paper/figures/fig4_checklist.png"); plt.close()
print("fig4_checklist.png")

# ---------- Fig 3: per-axis gap-closure toward the teacher ----------
# (base, tuned, teacher) per axis; gap-closure = (tuned-base)/(teacher-base)*100
data = [  # (name, base, tuned, teacher) — canonical 3-judge grading
    ("takeaway",     37.6, 68.5, 89.2), ("topics_cover", 78.5, 82.8, 96.8),
    ("tech-lens",    33.3, 48.4, 57.0), ("teacher-lens", 21.5, 50.2, 76.3),
    ("tone (summ.)", 57.0, 79.2, 94.6), ("thesis",       80.6, 87.1, 97.8),
    ("opening",      80.6, 98.2, 100.0), ("faithful",    69.9, 73.1, 95.7),
    ("length",       69.9, 95.7, 97.8),
    ("urgency",      55.8, 72.4, 67.7), ("frame",        51.2, 58.4, 63.4),
    ("sentiment",    48.8, 69.2, 81.7), ("depth",        51.2, 42.7, 48.4),
    ("tone (label)", 32.6, 34.1, 77.4),
]
labels, gcs, colors, notes = [], [], [], []
for name, b, t, te in data:
    denom = te - b
    if denom <= 0:                      # teacher <= base: gap-closure undefined
        gc = 100 if t > b else 0
        col = COLORS["r1"] if t > b else COLORS["base"]; note = "beats teacher" if t > b else ""
    else:
        gc = (t - b) / denom * 100
        col = "#3f7d3f" if gc > 100 else (COLORS["llama"] if gc < 0 else COLORS["r1"])
        note = ""
    labels.append(name); gcs.append(gc); colors.append(col); notes.append(note)
fig, ax = plt.subplots(figsize=(7.4, 5.4))
ys = list(range(len(labels)))[::-1]
ax.axvline(0, color="#8a929e", lw=1); ax.axvline(100, color="#8a929e", lw=1, ls="--")
for y, g, c, n in zip(ys, gcs, colors, notes):
    ax.barh(y, g, color=c, alpha=0.9, height=0.66)
    ax.text(g + (2 if g >= 0 else -2), y, f"{g:.0f}" + (f"  {n}" if n else ""),
            va="center", ha="left" if g >= 0 else "right", fontsize=8.5)
ax.set_yticks(ys); ax.set_yticklabels(labels)
ax.axhline(4.5, color="#c8ccd2", lw=0.8)                       # summary | classification divider
ax.text(150, 9.3, "summary", fontsize=9, style="italic", color="#5a6675")
ax.text(150, 3.6, "classification", fontsize=9, style="italic", color="#5a6675")
ax.set_xlim(-40, 170); ax.set_xlabel("Gap-closure toward teacher (%)   0 = base, 100 = teacher")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Every summary axis recovers over base; classification is mixed", fontweight="bold", pad=8)
plt.tight_layout(); plt.savefig("paper/figures/fig3_gap_closure.png"); plt.close()
print("fig3_gap_closure.png")

# ---------- Fig 5: efficiency (batch collapse + quality-cost plane) ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.8))
vols = np.array([100, 200, 300, 400, 500])
axL.plot(vols, vols*39.2/3600, "-o", color=COLORS["r1"], label="Teacher (R1-8B), ~39 s/article")
axL.plot(vols, vols*0.8/3600, "-s", color="#3f7d3f", label="Any 0.6B arm, ~0.8 s/article")
axL.set_xlabel("Articles in batch"); axL.set_ylabel("Wall-clock (hours)")
axL.annotate("5.4 h", (500, 5.44), fontsize=9, color=COLORS["r1"], ha="right", va="bottom")
axL.annotate("~7 min", (500, 0.11), fontsize=9, color="#3f7d3f", ha="right", va="bottom")
axL.legend(frameon=False, fontsize=8.5, loc="upper left")
axL.spines[["top", "right"]].set_visible(False)
axL.set_title("Batch wall-clock vs. volume", fontweight="bold", fontsize=11)
pts = [("Teacher", 39200, 84.8, COLORS["r1"]), ("Tuned", 774, 72.5, COLORS["accent"]),
       ("Few-shot", 856, 66.7, COLORS["base"]), ("Constrained", 824, 57.0, COLORS["base"]),
       ("Base", 845, 55.0, COLORS["base"])]
for name, ms, q, c in pts:
    axR.scatter(ms, q, s=60, color=c, zorder=3)
    axR.annotate(name, (ms, q), fontsize=8.5, xytext=(4, 4), textcoords="offset points")
axR.set_xscale("log"); axR.set_xlabel("Latency per article (ms, log)"); axR.set_ylabel("Summary quality (%)")
axR.spines[["top", "right"]].set_visible(False)
axR.set_title("Quality-cost plane: size is the horizontal move", fontweight="bold", fontsize=11)
plt.tight_layout(); plt.savefig("paper/figures/fig5_efficiency.png"); plt.close()
print("fig5_efficiency.png")
