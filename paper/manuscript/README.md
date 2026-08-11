# AQNG manuscript

This directory contains the canonical AQNG manuscript source.

Build locally with a REVTeX-capable TeX installation:

```bash
cd paper/manuscript
latexmk -pdf -halt-on-error -interaction=nonstopmode main.tex
```

The GitHub Actions workflow `.github/workflows/build-paper.yml` validates the manuscript and stores the compiled PDF artifact. On pushes to `main`, it also updates `AQNG_paper.pdf` when the compiled output changes.

Scientific source and citation constraints are documented in `SOURCE_POLICY.md`.
