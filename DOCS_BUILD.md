# Document build notes

The academic manuscript is maintained in Markdown (`perceptibility_gap_paper_v1_4.md`) as the single source of truth. The `.tex` and `.docx` are generated from it with pandoc.

## Regenerate the manuscript formats

```bash
# LaTeX
pandoc perceptibility_gap_paper_v1_4.md -s --from gfm --to latex -o perceptibility_gap_paper_v1_4.tex

# Word
pandoc perceptibility_gap_paper_v1_4.md -s --from gfm --to docx -o perceptibility_gap_paper_v1_4.docx
```

Requires `pandoc` (built with 3.9.0.2). The figure `shared_normalized_distortion_axisb.png` is embedded into the `.docx` automatically and referenced by relative path in the `.tex`, so keep it in the same directory when compiling the LaTeX.

## PDF

`pdflatex` is **not installed** in this environment, so no PDF is checked in. To produce one once a TeX distribution is available:

```bash
pdflatex perceptibility_gap_paper_v1_4.tex
pdflatex perceptibility_gap_paper_v1_4.tex   # second pass for cross-references
```

(Install e.g. MacTeX or BasicTeX first: `brew install --cask mactex-no-gui`.)

## Versions

- `v1_4` — current. Adds cross-family generalization (Section 5.1, Appendix F.6).
- `v1_3` — integrity revision (leakage fix, hardened controls, LOCO/null).
- `v1_2b` — prior baseline (kept for provenance).

Only `v1_4` is exported to `.tex`/`.docx`; regenerate older versions on demand if needed.
