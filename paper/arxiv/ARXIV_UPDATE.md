# arXiv replacement — submission package & checklist

This is a **replacement** (new version) of an existing arXiv submission, not a new one.
Upload the source tarball, refresh the metadata, add a version note, verify the preview,
and submit.

## 1. What to upload

**File:** `arxiv_submission.tar.gz` (in this directory, ~588 KB)

Contents (flat, exactly six files):
- `paper.tex` — main source (`graphicspath` set to `{{./}}` so figures resolve flat)
- `fig1_task.png`, `fig2_pipeline.png`, `fig3_gap_closure.png`, `fig5_efficiency.png`, `fig7_teachers.png`

Deliberately **excluded**: `paper.pdf` (arXiv builds its own; a same-named PDF confuses its
classifier), `paper_autoconv_backup.tex`, `README.md`, and `fig4_checklist.png` (generated but
never `\includegraphics`'d).

No `IEEEtran.cls` is bundled — it ships with arXiv's TeX Live. No `.bib`/`.bbl` is needed —
the bibliography is a manual `thebibliography` environment.

## 2. Pre-flight checks already done

- **Compiles clean standalone** from the flat directory (tectonic): 12 pages, US-letter, 0 overfull boxes.
- **pdflatex-safe** (arXiv's default engine): the body is pure ASCII. Every non-ASCII glyph
  in the file appears only inside its own `\newunicodechar{...}` declaration; those mappings are
  defensive and unused by the text, so pdflatex will not trip on an unmapped character.
- Only the 5 referenced figures are included; all `\ref`/`\label` resolve; no duplicate labels.

## 3. Metadata to update on the web form

The title and abstract both changed substantially since the last version, so update them.
arXiv metadata is plain text (light math OK), not LaTeX.

**Title**
```
Different Teachers, Different Capabilities: Sub-1B On-Device Distillation for Structured Text Enrichment
```

**Abstract** — see `arxiv_abstract.txt` in this directory (paste verbatim).

**Comments** (suggested version note):
```
12 pages, 5 figures. Substantial revision: adds a same-size non-reasoning-teacher control,
a three-judge LLM-as-a-judge panel with a negative control, full-source faithfulness grading,
and a per-field routing analysis; title and framing updated accordingly.
```

**Categories:** keep whatever the existing submission uses (likely cs.CL primary, cs.LG cross).
Do not change on a replacement unless you mean to.

**Authors / license:** carry over from the existing version; no change needed.

## 4. Step-by-step

1. Sign in at arxiv.org and open the existing article (your user dashboard lists it; the
   identifier is on that page if you need it).
2. Click **Replace** (this starts version v-next; the old version stays public and linked).
3. Upload `arxiv_submission.tar.gz`. Let AutoTeX process it (it runs pdflatex).
4. **Inspect the generated preview PDF** page by page — confirm all five figures render and
   the tables are intact.
5. Update **Title** and **Abstract** (Section 3), add the **Comments** note.
6. Submit. Replacements go on hold and announce at the next scheduled mailing (usually the
   next business day, 20:00 ET cutoff).

## 5. If AutoTeX errors

- "File `figX.png` not found" → a figure name/case mismatch; confirm the tarball is flat and
  names match the `\includegraphics` calls exactly.
- Unicode / inputenc complaint (unlikely) → add `\usepackage[utf8]{inputenc}` right after the
  `\documentclass` line and re-tar.
- To rebuild the tarball after any edit:
  `cd paper/arxiv && tar czf arxiv_submission.tar.gz -C arxiv_upload .`
