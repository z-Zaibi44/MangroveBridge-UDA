# Reproducibility and release provenance

## What was supplied

The source was `MangroveBridge_UDA_11 (1)(2).ipynb` (29 cells, 19 code cells).

Original file SHA-256:

```text
ac4e6c55f9f361b85934524a641a340ca4a99cbae149e5ee64f581efcaa3ffae
```

Repository preparation cleared code outputs and execution counts, removed execution metadata, replaced eight local absolute paths with editable relative examples, and revised the introductory release documentation. Every code cell outside the path-configuration cell was checked against the supplied source and retained unchanged. The Python script is an ordered export of all notebook code cells.

## Checks performed

- Parsed every code cell and the Python export with Python's syntax parser.
- Confirmed that notebook outputs and execution counts were cleared.
- Confirmed that original Windows absolute paths were removed from notebook source.
- Compared code outside Section 0 byte-for-byte with the source cell text.
- Checked that the Python export contains the same ordered code as the cleaned notebook.

These are static checks. PyTorch, PROSAIL, Rasterio and GeoPandas were unavailable in the preparation runtime, and study datasets were not supplied. The network self-test, dependency installation, physical simulation, training, checkpoint resumption, spatial overlay and raster inference were not executed. Stored output in the original notebook is not treated as independent verification. The environment files have not been solved or tested as a lockfile.

## Running an experiment

Record the repository commit, Python/package versions, hardware and CUDA versions, input file hashes, wavelength grid, sample grouping, image CRS, field-holdout assignment, flight geometry, band selection and every Section 0 setting.

The supplied defaults train seeds 42, 123 and 2024 and perform label-free tuning. Preserve per-seed reports and mean/standard-deviation summaries. Seeds alone do not guarantee identical GPU results across hardware or package versions.

Start a fresh kernel when changing inputs or settings. For a fresh experiment, use a new `OUT_DIR` or set `RESUME_SEEDS=False`; reused outputs must correspond to the intended inputs and settings. Preserve the trained artifacts separately from Git source control.

The synthetic self-test in Section 4c exercises forward/backward propagation, ordinal probabilities and brightness invariance using tensors. It needs earlier setup variables and is part of the notebook run order, not a standalone benchmark.

## Scientific evaluation

Check leaf-group separation, the PROSPECT fit, the crown inventory and held-out field-crown export before interpreting results. DEV is the configured label-free selection criterion. Leaf validation, seed agreement, evidential vacuity and REP/PRI plausibility do not establish field accuracy. Validate the exported crown predictions independently against field assessments.

The optional `prospect_version="PRO"` branch assumes APIs beyond the standard PROSPECT-5/D interface documented by the linked PROSAIL bindings. The supplied default is `"D"`; verify compatibility before enabling that branch.

Separate ablation/figure notebooks, raw study data and trained weights are outside this initial release. Do not report them as bundled artifacts.
