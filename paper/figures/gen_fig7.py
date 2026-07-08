#!/usr/bin/env python3
"""Fig 7: non-reasoning-teacher control. Run from repo root: python3 paper/figures/gen_fig7.py"""
import matplotlib.pyplot as plt
# checklist pass-rate, 11-arm re-grade, N=93
rows = [
    ("Teacher (R1-8B)",                 85.9, None,          "#2f5ba8"),
    ("Reasoning-teacher student\n(R1-8B, distilled)",     72.6, (71.5, 74.6), "#7b4fb5"),
    ("Base 0.6B (no distillation)",     54.7, None,          "#8a929e"),
    ("Non-reasoning-teacher student\n(Llama-3.1-8B-Instruct)", 50.8, (49.7, 51.9), "#c0603a"),
]
fig, ax = plt.subplots(figsize=(10, 4.4), dpi=160)
ys = list(range(len(rows)))[::-1]
for y, (label, v, rng, c) in zip(ys, rows):
    ax.barh(y, v, color=c, alpha=0.88, height=0.6)
    if rng:
        ax.errorbar(v, y, xerr=[[v - rng[0]], [rng[1] - v]], color="#2b2f36", capsize=4, lw=1.6)
    ax.text(v + 1.0, y, f"{v:.1f}", va="center", fontsize=11, fontweight="bold")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xlim(0, 100)
ax.set_xlabel("Summary checklist pass-rate (%) — 11-arm re-grade, N = 93", fontsize=11)
ax.axvline(54.7, ls="--", lw=1, color="#8a929e", alpha=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Only the reasoning teacher beats no-distillation: +21.9 pts [18.3, 25.5], p<0.001",
             fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("paper/figures/fig7_reasoning_control.png", bbox_inches="tight", facecolor="white")
print("fig7_reasoning_control.png written")
