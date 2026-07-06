# Distill Project Documentation 📚

Welcome to the **Distill** project documentation - a comprehensive knowledge base for our DeepSeek-R1:8B → Qwen3-0.6B distillation study!

## Quick Start ⚡

### What is this?
A local-first RSS reader that enriches articles with structured JSON objects using distilled 600M models (down from slow 8B teacher).

**Key Achievement:** Reduced processing time **5.4 hours → ~7 minutes** on-device while maintaining high quality!

### Essential Links 🔗
1. **[Main Paper](./paper/distillation_paper.md)** - Full study with all results and figures  
2. **[Reviewer's Map](./REVIEWER_MAP.md)** - Every claim backed by specific files  
3. **[Setup Guide](./setup/QUICKSTART.md)** - Get running in 5 minutes  

### Headline Results 📊
- **Summary quality:** Student beats controls (+15.5 pts) and closes ~59% of teacher gap  
- **Speed gain:** 40× faster inference (~0.8s vs ~39s per article)  
- **Classification:** Best small-model classifier with balanced recall (64.3 macro F1)  

---

## Project Structure 🏗️
```
distill-local/
├── paper/distillation_paper.md    ← Main study document
│   ├── figures/*.md               ← All study figures documented here  
├── REVIEWER_MAP.md                ← Cross-ref of every claim to source files  
├── evaluation_design.md            ← Multi-judge evaluation protocol  
└── server/eval/*.mjs              ← Evaluation engine scripts (.mjs)
```

---

## How This Wiki Works 📖

This documentation is **auto-generated** from your repository's markdown structure. Each page corresponds to:

- Paper sections and figures → Documentation pages
- Configuration files (yml, js) → Setup instructions  
- Data directories (jsonl) → Format reference  

### Example Usage Pattern 👨‍💼
```bash
# 1. Clone the repo locally
git clone <repo-url> distill-local

# 2. Run evaluation scripts from README.md examples:
cd server/eval
npm install  # Installs @mizchi/mem-adapter
node compile.mjs --help

# 3. Read documentation at .openclaw/workspace/distill-local/...
```

---

## Key Documentation Pages 📖

| Topic | Page Location | Description |
|-------|---------------|-------------|
| **Paper & Figures** | `paper/*.md` | Full study with results |
| **Setup Guide** | `.openclaw/workspace/distill-local/setup/*` | Get running quickly |  
| **Evaluation Scripts** | `server/eval/*.mjs` | Canonical scoring runner |
| **Data Formats** | `data/*` | JSONL schemas and examples |

---

## Toolchain Summary 🛠️

- **Teacher:** DeepSeek-R1:8B (reasoning model)  
- **Student:** Qwen3-0.6B (distilled 600M student)  
- **Platform Arm:** Distil Labs managed pipeline (for comparison)  

---

## Quick Reference 🔍
```bash
# Run evaluation with full-text scoring:
node server/eval/compile.mjs -f <file> --scoring=<judge-name1, judge-name2>

# Help for usage examples and commands at README.md!
npm run help  # Shows all available subcommands
```

---

*Generated documentation reflects current repository state. All claims backed by source files in repo.*

