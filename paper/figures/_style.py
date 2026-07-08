"""Shared figure style so every plot matches the IEEEtran (Times/serif) body font.
Import from any generator:  from _style import apply_style, COLORS ; apply_style()
"""
import matplotlib as mpl

# arm/teacher palette, reused across figures for consistency
COLORS = {
    "r1":        "#2f5ba8",   # reasoning teacher / its students (blue family)
    "r1_student":"#5b7fc4",
    "llama":     "#c0603a",   # non-reasoning teacher / its students (terracotta)
    "gptoss":    "#c07a1f",   # large managed teacher / its student (amber)
    "base":      "#8a929e",   # non-distillation baselines (grey)
    "teacher":   "#2b2f36",   # generic teacher ink
    "accent":    "#7b4fb5",
}

def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",   # Times-compatible math, to match the IEEEtran (Times) body
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.edgecolor": "#2b2f36",
        "axes.linewidth": 0.9,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })
