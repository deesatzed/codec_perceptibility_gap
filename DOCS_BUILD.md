# Document build notes

The academic manuscript is maintained in Markdown (`perceptibility_gap_paper_v1_5.md`) as the single source of truth. The `.tex` and `.docx` are generated from it with pandoc.

## Regenerate the manuscript formats

```bash
# LaTeX — the markdown reader + low --columns forces pandoc to emit wrapping
# p{} columns so the wide gate-matrix / claim-ladder tables fit the page
# (gfm reader leaves them as non-wrapping `l` columns and they overflow).
pandoc perceptibility_gap_paper_v1_5.md -s \
  --from markdown+pipe_tables --to latex \
  -V geometry:margin=0.85in -V fontsize=11pt \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  -H header.tex --columns=48 \
  -o perceptibility_gap_paper_v1_5.tex

# Word
pandoc perceptibility_gap_paper_v1_5.md -s --from gfm --to docx -o perceptibility_gap_paper_v1_5.docx
```

`header.tex` (small font inside tables, in this directory):

```latex
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize}
\AtBeginEnvironment{tabular}{\footnotesize}
```

Requires `pandoc` (built with 3.9.0.2). The figure `shared_normalized_distortion_axisb.png` is embedded into the `.docx` automatically and referenced by relative path in the `.tex`, so keep it in the same directory when compiling.

## PDF

Built with **tectonic** (`brew install tectonic`; built with 0.16.9). Tectonic auto-fetches packages and runs the needed passes:

```bash
tectonic perceptibility_gap_paper_v1_5.tex
# -> perceptibility_gap_paper_v1_5.pdf (21 pages; only sub-pixel 0.12pt alignment notices remain)
```

The committed `perceptibility_gap_paper_v1_5.pdf` was produced this way. Remaining LaTeX warnings, if any, are cosmetic line-break (under/overfull \hbox in prose) notices, not layout errors.

## Versions

- `v1_5` — current. External-critique mitigations: parallel verification tracks (Section 5.2), estimator-class bridging validation (Section 5.3, Appendix F.7), and a non-oscillatory discrete-time chaotic family (Hénon) extending cross-family generalization (Section 5.1.1, Appendix F.6).
- `v1_4` — added cross-family generalization with the Duffing nonlinear family (Section 5.1).
- `v1_3` — integrity revision (leakage fix, hardened controls, LOCO/null).
- `v1_2b` — prior baseline (kept for provenance).

Only `v1_5` is exported to `.tex`/`.docx`; regenerate older versions on demand if needed.
