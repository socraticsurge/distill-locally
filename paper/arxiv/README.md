# arXiv source

`paper.tex` is the LaTeX version of `../distillation_paper.md`, converted for
arXiv submission (pdflatex-compatible preamble; unicode mapped via
`newunicodechar`). Figures are shared with the paper: copy `../figures/*.png`
into `figures/` (or symlink) before compiling.

Compile locally (Tectonic, no TeX Live install needed):

    cd paper/arxiv && mkdir -p figures && cp ../figures/*.png figures/ && tectonic paper.tex

To build the arXiv upload tarball:

    tar -czf distillation_arxiv.tar.gz paper.tex figures/

Two conversion bugs were fixed relative to the original auto-conversion
(2026-07-04): doubled backslashes inside figure environments, and pandoc's
`\LTcaptype{none}` wrapper conflicting with the `caption` package.

Note: verified with Tectonic (XeTeX engine). arXiv compiles with pdflatex by
default; the preamble targets pdflatex, and the fixes are engine-agnostic, but
expect arXiv's autotex log to be the final word.
