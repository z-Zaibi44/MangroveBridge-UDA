# MangroveBridge-UDA

**Physics-guided leaf-to-UAV canopy unsupervised domain adaptation for mangrove health assessment.**

MangroveBridge-UDA transfers condition-discriminative information from labelled laboratory leaf spectra to unlabelled UAV hyperspectral imagery. It combines PROSPECT-D/4SAIL canopy simulation, a wavelength-aware hybrid state-space/attention encoder, evidential ordinal learning, class-prior-aware adversarial and covariance alignment, and uncertainty-gated mean-teacher self-training.

The framework distinguishes Healthy, Transitional, and Stressed conditions using a controlled laboratory senescence gradient. Pixel-wise evidential predictions are aggregated within LiDAR-derived individual crown polygons to produce crown-level condition maps with associated uncertainty. The manuscript evaluates predictions against independent in situ physiological reference labels.

**Author:** Zohaib Raza · Department of Land Surveying and Geoinformatics · The Hong Kong Polytechnic University

## Method overview

1. Align labelled leaf spectra to UAV wavelength centres and split by leaf/sample groups.
2. Pool crown-aware UAV samples across study plots while holding field-assessed crowns aside.
3. Invert PROSPECT and generate canopy spectra with 4SAIL, scene-derived backgrounds and noise.
4. Encode spectral views with wavelength-aware tokens, domain-specific BatchNorm, bidirectional selective state-space blocks and attention.
5. Train an evidential ordinal CORN head with trait prediction, importance-weighted CDAN/CORAL and mean-teacher self-training.
6. Select checkpoints and tune settings using label-free DEV risk; train the final setting with every configured seed.
7. Map pixels strip by strip, fuse evidence per crown, and export seed-agreement and plausibility summaries.

The configured ordered classes are **Healthy → Transitional → Stressed**. The notebook pools plots for one model and writes maps separately for each plot.

## Repository contents

| File | Purpose |
|---|---|
| [MangroveBridge_UDA.ipynb](MangroveBridge_UDA.ipynb) | Main workflow; start here |
| [mangrovebridge_uda.py](mangrovebridge_uda.py) | Sequential Python export of the notebook's code cells |
| [environment.yml](environment.yml) | Conda environment specification |
| [requirements.txt](requirements.txt) | Python dependencies for a pip-based installation |
| [DATA_FORMAT.md](DATA_FORMAT.md) | Input formats, class encoding and spatial assumptions |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Run settings, validation scope and release provenance |
| [.gitignore](.gitignore) | Excludes local inputs, output maps, checkpoints and environment files |

The separate ablation notebook, publication-figure notebook and design document mentioned in the original upload were not provided. The main notebook retains its ablation switches and figure-data exports.

## Installation

Python 3.11 is the suggested starting environment from the supplied notebook. Dependency files are starting specifications, not a tested environment lock.

### Conda (recommended for geospatial dependencies on Windows)

```bash
conda env create -f environment.yml
conda activate mangrovebridge
jupyter lab
```

### Pip alternative

From a Python 3.11 virtual environment:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

For CUDA, choose the appropriate PyTorch installation command from the [official selector](https://pytorch.org/get-started/locally/). The workflow falls back to CPU when CUDA is unavailable; full training and tuning can be expensive on CPU. The selective state-space blocks use PyTorch directly; the notebook does not import `mamba-ssm` or Triton.

## Quick start

1. Download or clone the repository and open a terminal in its root directory.
2. Install the environment and place your own data in the local paths described in [DATA_FORMAT.md](DATA_FORMAT.md).
3. Open `MangroveBridge_UDA.ipynb`. Edit Section 0 input paths, the field-holdout configuration, flight solar zenith angle, leaf grouping, and any band selection needed for your data.
4. Restart the kernel and run cells in order. Inspect the crown inventory in Section 2b before training. The synthetic forward/backward self-test in Section 4c runs after the preceding setup cells.
5. Review label-free tuning and multi-seed summaries, then inspect per-plot crown maps. Perform independent validation against field assessments outside this notebook.

The default paths are relative to the repository root. Start Jupyter there. `FIELD_POINTS_SHP` points to an example local field file; either provide it or explicitly set it to `None` if field data are unavailable, noting that no field crowns can then be held out.

To run the complete workflow as a script, edit the paths in the script's Section 0 and run:

```bash
python mangrovebridge_uda.py
```

The script starts the entire pipeline, including tuning and training. The notebook is the primary editable source; regenerate the script after editing notebook code cells to keep both versions synchronized.

## Configuration to review

| Setting | Supplied default / meaning |
|---|---|
| `SEEDS` | `[42, 123, 2024]` |
| `RUN_TUNING` | `True`; one-at-a-time search followed by multi-seed ranking |
| `MAP_SEED` | `"best_dev"`; chosen with label-free DEV risk |
| `SELECTED_BANDS` | 506–646, 690–742 and 918–934 nm; change for your experiment |
| `FIELD_HOLDOUT` | `True`; excludes matching assessed crowns from adaptation/selection samples |
| `leaf_group_mode` | `"column"`; change to `"regex"` or `"every_n"` for repeated scans |
| `prospect_version` | `"D"`; the optional `"PRO"` branch requires a compatible PROSAIL implementation |
| `RESUME_SEEDS` | `True`; consult the reproducibility notes before reusing run folders |
| `OUT_DIR` | `outputs`, with `selected_bands` appended under the supplied selection |

## Outputs

Outputs are written below `OUT_DIR` and excluded from Git tracking.

| Output | Contents |
|---|---|
| `crown_inventory/` | Classifiable-crown inventory and overview; unclassifiable crowns are listed separately |
| `runs/` | Per-setting/per-seed teacher checkpoints and training/selection reports |
| `seed_results.csv`, `seed_results_summary.csv` | Per-seed metrics and aggregate summaries |
| `mangrovebridge_teacher.pt`, `mangrovebridge_meta.json` | Selected teacher and metadata |
| Per-plot `pixel_class.tif`, `pixel_layers.tif`, `crown_class.tif` | Raster classifications and auxiliary layers |
| `crown_condition.csv` / `.shp` and pooled crown tables | Crown predictions, uncertainty and evidence-source information |
| Seed-agreement exports | Cross-seed agreement and consensus products |
| `figure_data/` | Arrays, tables and manifest for downstream figures |

For precise locations and raster encodings, inspect Section 7 and its saved metadata. Field accuracy is not computed by this workflow. Leaf validation and spectral-index plausibility are distinct from independent crown-condition accuracy.

## References and related implementations

The notebook's section markdown contains method-specific literature links. Repository organization was informed by these published implementations; their code is not bundled here:

- [THUML Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) — domain-adaptation examples, documentation, installation and dataset organization.
- [PROSAIL Python bindings](https://github.com/jgomezdans/prosail) — PROSPECT and 4SAIL interfaces used by the physics bridge.

