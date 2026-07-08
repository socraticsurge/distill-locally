#!/usr/bin/env python3
"""Fig 1: neutral task schematic (replaces the product-UI panel mockup).
Run from repo root: python3 paper/figures/gen_fig1.py"""
import sys, os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(__file__))
from _style import apply_style, COLORS
apply_style()

fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")

def arrow(x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=16,
                                 color="#4a5462", lw=1.8))

# input -> model -> structured output
box(1, 16, 20, 12, "News article\n(plain text)", "#eceff3", "#5a6675")
arrow(21.5, 29, 22)
box(29, 16, 20, 12, "0.6B student\n(on-device,\ntemp 0)", "#e9dff3", COLORS["accent"], bold=True)
arrow(49.5, 57, 22)

# JSON output with the three field groups and their stakes
box(57, 31, 42, 10, "summary  (free text)   [HIGH stakes]", "#f4dede", "#b13a3a", fs=9.5, bold=True)
box(57, 14.5, 42, 9, "5 categorical labels   [low stakes]\nsentiment · urgency · frame · tone · depth", "#e6efe0", "#4a7c3f", fs=8.5)
box(57, 3.5, 42, 8, "open-set topics   [low stakes]", "#e6efe0", "#4a7c3f", fs=9)
ax.text(78, 43, "one JSON object per article", ha="center", fontsize=9, style="italic", color="#5a6675")

ax.text(50, -2, "The asymmetric failure cost across fields motivates a per-field engine assignment.",
        ha="center", fontsize=8.5, color="#2b2f36")
plt.tight_layout()
plt.savefig("paper/figures/fig1_task.png")
print("fig1_task.png written")
