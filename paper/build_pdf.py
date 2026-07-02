#!/usr/bin/env python3
"""Render paper/distillation_paper_v4.md -> distillation_paper_v4.pdf, arXiv-style.
Run: DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python3 paper/build_pdf.py"""
import os, re, sys
import markdown, weasyprint

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "distillation_paper_v4.md")
OUT  = os.path.join(HERE, "distillation_paper_v4.pdf")

md_text = open(SRC).read()
html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc", "sane_lists"])

# --- title block: everything before the first <hr> is centered title/author matter ---
parts = html.split("<hr />", 1)
if len(parts) == 2:
    head, body = parts
    head = head.replace("<strong>Date:", "<br><strong>Date:").replace("<strong>Status:", "<br><strong>Status:")
    html = f'<div class="titleblock">{head}</div><hr />{body}'

# --- tag figure captions (paragraphs starting with <strong>Figure) ---
html = re.sub(r'<p><strong>(Figure \d+\.)</strong>', r'<p class="caption"><strong>\1</strong>', html)
# --- tag the abstract paragraph (first <p> after the Abstract heading) ---
html = re.sub(r'(<h2[^>]*>Abstract</h2>\s*)<p>', r'\1<p class="abstract">', html, count=1)

CSS = """
@page { size: letter; margin: 1in 0.9in;
        @bottom-center { content: counter(page); font-family: 'Times New Roman', serif; font-size: 9pt; color:#555; } }
html { font-family: 'Times New Roman', 'Times', 'DejaVu Serif', serif; font-size: 10.5pt; line-height: 1.34; color:#111; }
body { text-align: justify; hyphens: auto; }
.titleblock { text-align: center; margin-bottom: 0.4rem; }
.titleblock h1 { font-size: 19pt; line-height:1.2; margin: 0 0 .3rem; font-weight: bold; }
.titleblock h3 { font-size: 12pt; font-weight: normal; font-style: italic; color:#333; margin: 0 0 .6rem; }
.titleblock p { font-size: 10pt; line-height:1.5; margin:.2rem 0; }
h1 { font-size: 19pt; } h2 { font-size: 13pt; margin: 1.05rem 0 .35rem; border-bottom:1px solid #ddd; padding-bottom:2px; }
h3 { font-size: 11pt; margin: .7rem 0 .3rem; } h4 { font-size: 10.5pt; margin:.6rem 0 .25rem; font-style:italic; }
h2, h3, h4 { font-weight: bold; text-align:left; page-break-after: avoid; }
p { margin: 0 0 .5rem; }
p.abstract { font-size: 9.7pt; margin: 0 1.1rem .6rem; text-align: justify; }
blockquote { margin: .6rem 1.2rem; padding-left:.7rem; border-left:3px solid #cbd5e1; color:#1f2937; font-style:italic; }
a { color:#1d4ed8; text-decoration:none; }
code { font-family:'DejaVu Sans Mono',monospace; font-size:8.6pt; background:#f4f4f6; padding:0 2px; }
pre { background:#f6f7f9; border:1px solid #e5e7eb; border-radius:4px; padding:.55rem .7rem; font-size:8.3pt;
      line-height:1.28; overflow-x:auto; white-space:pre-wrap; page-break-inside: avoid; }
pre code { background:none; padding:0; }
img { display:block; margin:.5rem auto .2rem; max-width:100%; }
p.caption { font-size:8.8pt; color:#374151; text-align:left; margin:.1rem .4rem .9rem; line-height:1.3; }
table { border-collapse: collapse; width:100%; margin:.5rem 0 .8rem; font-size:8.9pt; page-break-inside: avoid; }
thead th { border-top:1.3px solid #111; border-bottom:1px solid #111; padding:3px 6px; text-align:left; }
tbody td { border:none; padding:2.5px 6px; }
tbody tr:last-child td { border-bottom:1.3px solid #111; }
table td:not(:first-child), table th:not(:first-child) { text-align:right; }
hr { border:none; border-top:1px solid #e5e7eb; margin:1rem 0; }
strong { font-weight: bold; }
"""

doc = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
weasyprint.HTML(string=doc, base_url=HERE + "/").write_pdf(OUT)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
