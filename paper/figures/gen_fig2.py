#!/usr/bin/env python3
"""Fig 2: study-at-a-glance, three-teacher pipeline. Run from repo root:
   python3 paper/figures/gen_fig2.py"""
import sys, os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS
apply_style()

fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=12, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def arr(x1, y1, x2, y2, c="#4a5462"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20, color=c, lw=2))

ax.text(50, 97, "Study at a glance: three teachers, one 0.6B student, per-field scoring",
        ha="center", fontsize=15, fontweight="bold")

# corpus + split
box(30, 87, 40, 6.5, "494 news articles  ·  24 feeds", "#eceff3", "#5a6675", fs=12.5, bold=True)
arr(50, 86.8, 50, 83)
box(26, 76, 48, 6.5, "stratified split:  401 train  /  93 held-out test", "#eceff3", "#5a6675", fs=12)

ax.text(50, 71.5, "the 401 training articles are labeled by three teachers, each distilling the same Qwen3-0.6B base:",
        ha="center", fontsize=11.5, style="italic", color="#3a4250")

# three teachers
tx = [4, 37, 70]; tw = 26
teach = [("R1-8B\nreasoning", COLORS["r1"], "#dbe5f4"),
         ("Llama-3.1-8B\nnon-reasoning", COLORS["llama"], "#f4e2da"),
         ("gpt-oss-120B\nmanaged + synthetic data", COLORS["gptoss"], "#f7ecd4")]
for x, (label, ec, fc) in zip(tx, teach):
    box(x, 60, tw, 8, label, fc, ec, fs=11.5, bold=True)
    arr(x + tw/2, 59.8, x + tw/2, 55)
stu = ["0.6B student\n(QLoRA, 3 seeds)", "0.6B student\n(QLoRA, 3 seeds)", "0.6B student\n(1 managed run)"]
for x, s, (_, ec, fc) in zip(tx, stu, teach):
    box(x, 46.5, tw, 8, s, "#f3eff8", COLORS["accent"], fs=10.5)

for x in tx: arr(x + tw/2, 46.3, 50, 37.5)

# generation (arm composition folded in)
box(9, 29, 82, 8, "12 arms generate one JSON per held-out article, temperature 0:\n7 distilled students, the two 8B teachers, few-shot, constrained decoding, and the untuned base",
    "#e6efe0", "#4a7c3f", fs=11, bold=False)
arr(50, 28.8, 50, 26)

# judge panel
box(10, 18.5, 80, 7, "blinded, reference-free 3-judge panel  ·  Gemini Flash Lite · Nemotron-550B · Mistral-Small-119B  ·  graded against the full article",
    "#f7ecd4", "#b98a1e", fs=11)
arr(50, 18.3, 50, 15)

# two sub-tasks
box(12, 5, 36, 7.5, "Summarization\n8-item binary checklist (pass-rate)", "#e6efe0", "#4a7c3f", fs=11)
box(52, 5, 36, 7.5, "Classification\n5 labels vs. panel proxy-gold", "#e6efe0", "#4a7c3f", fs=11)
ax.text(50, 1, "reported per field as gap-closure toward the teacher", ha="center", fontsize=10, style="italic", color="#5a6675")

plt.tight_layout()
plt.savefig("paper/figures/fig2_pipeline.png")
print("fig2_pipeline.png written")
