# Distillation Study: Complete Setup Guide 📚

## Overview 🔍

This guide helps you set up and understand the DeepSeek-R1:8B → Qwen3-0.6B knowledge distillation study for structured news enrichment tasks.

**Project Goal:** Create a fast, faithful 600M student model that matches an 8B reasoning teacher's output quality while running ~40× faster on-device (~0.8s vs ~39s per article).

---

## Step-by-Step Setup 🛠️

### 1. Initial Clone
```bash
git clone <repo-url> distill-local 
cd /Users/vinaychaganti/.openclaw/workspace/distill-locally/
npm install               # Installs @mizchi/mem-adapter and other dependencies
```

---

## Key Resources 📖

### Documentation Index: `WIKI.md`
The main wiki hub at `/Users/vinaychaganti/.openclaw/workspace/distill-local/WIKI.md`.

This auto-generated documentation includes cross-referenced sections from your paper, figures in `.fig/`, and evaluation results.

--- 

### Evaluation Engine: `server/eval/*.mjs`
Run full-text scoring using the canonical script runner:
```bash
node server/eval/compile.mjs --help
node server/eval/run.py        # if Python available (optional)
```

---

## Quick Start Commands 📝

### Run Evaluation Scoring:
```bash
# Full text scorecard with judge panel scoring  
cd /Users/vinaychaganti/.openclaw/workspace/distill-locally/server/eval
node compile.mjs --dataset data/test.jsonl \
    --scoring "<model-a>,<model-b>"  # Judge panel config
```

### Read Documentation:
```bash
# Main wiki hub  
cat WIKI.md | less 

# Review paper sections and figures at /paper/*.md 
cd paper && ls -la .fig/   # Figure directory contents from paper
```

---

## Common Tasks 🎯

| Task | Command Pattern | Notes |
|------|-----------------|-------|
| **Run full evaluation** | `node compile.mjs` | Reads config from `.openclaw/workspace/distill-local/data/*.jsonl` |  
| **Train student model** | Check notebooks/ directory | See README.md for Colab T4 QLoRA examples in notebook files (unsloth)  |
| **View study results** | `cd paper && cat distillation_paper.md` | Full breakdown by field/arm at this location with figures available (.fig/) |  

---

## Troubleshooting 💡

### Issue: "npm ERR! missing peer..." 
Solution: Run `npm install @mizchi/mem-adapter --save-exact` per README.md instructions for correct versions.

See `/Users/vinaychaganti/.openclaw/workspace/distill-local/setup/GUIDE.md#common-tasks-table` above for command patterns!

