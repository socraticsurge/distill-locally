#!/usr/bin/env python3
"""Master 'three teachers, their students' figure. Run from repo root:
   python3 paper/figures/gen_teachers_figure.py
Data-driven from the 12-arm scorecard."""
import json, sys, os
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS
apply_style()

S = json.load(open("data/eval/scores_gemini_nemotron_n93_fullctx_12arm.json"))
CL = S["checklist"]; CLS = S["classification"]["perArm"]
mean = lambda xs: sum(xs) / len(xs)
R1 = ["tuned_rss_tuned_s1", "tuned_rss_tuned_s2", "tuned_rss_tuned_s3"]
LL = ["tuned_rss_llama_s1", "tuned_rss_llama_s2", "tuned_rss_llama_s3"]
chk = lambda a: CL[a]["checklistPassRate"]
mac = lambda a: CLS[a]["macroAcc"]

# rows: (label, teacher-self, student-pooled, color)  teacher None where not graded
rows = [
    ("R1-8B\n(reasoning)",        chk("teacher"),        mean([chk(a) for a in R1]), mean([mac(a) for a in R1]), mac("teacher"),        COLORS["r1"]),
    ("Llama-3.1-8B\n(non-reasoning)", chk("teacher_llama"), mean([chk(a) for a in LL]), mean([mac(a) for a in LL]), mac("teacher_llama"), COLORS["llama"]),
    ("gpt-oss-120B\n(managed+synthetic)", None,           chk("distil"),              mac("distil"),              None,                  COLORS["gptoss"]),
]
base_chk, few_chk, base_mac = chk("base"), chk("fewshot"), mac("base")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
y = np.arange(len(rows))[::-1]
h = 0.34

def panel(ax, teach_idx, stud_idx, title, ref_lines):
    for yi, r in zip(y, rows):
        t, c = r[teach_idx], r[5]
        s = r[stud_idx]
        if t is not None:
            ax.barh(yi + h/2, t, height=h, color=c, alpha=0.95, edgecolor="white", lw=0.5)
            ax.text(t + 0.8, yi + h/2, f"{t:.0f}", va="center", fontsize=9, fontweight="bold", color=c)
        ax.barh(yi - h/2, s, height=h, color=c, alpha=0.5, edgecolor="white", lw=0.5, hatch="///")
        ax.text(s + 0.8, yi - h/2, f"{s:.0f}", va="center", fontsize=9, color=c)
    for x, lab, ls in ref_lines:
        ax.axvline(x, ls=ls, lw=1, color="#8a929e", alpha=0.8)
        ax.text(x, len(rows) - 0.35, lab, fontsize=8, color="#5a6675", ha="center")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 100); ax.set_xlabel(title)
    ax.spines[["top", "right"]].set_visible(False)

panel(axL, 1, 2, "Summary checklist pass-rate (%)", [(base_chk, "base", "--"), (few_chk, "few-shot", ":")])
panel(axR, 4, 3, "Classification macro-accuracy (%)", [(base_mac, "base", "--")])
axR.set_yticklabels([])

# legend
from matplotlib.patches import Patch
leg = [Patch(fc="#2b2f36", alpha=0.95, label="teacher (its own output)"),
       Patch(fc="#2b2f36", alpha=0.5, hatch="///", label="its distilled 0.6B student")]
axR.legend(handles=leg, loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=9)
fig.suptitle("Three teachers and their 0.6B students: reasoning wins summaries; scale+synthetic wins labels",
             fontsize=12.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("paper/figures/fig7_teachers.png")
print("fig7_teachers.png written | R1", chk("teacher"), mean([chk(a) for a in R1]),
      "| Llama", chk("teacher_llama"), mean([chk(a) for a in LL]), "| platform", chk("distil"))
