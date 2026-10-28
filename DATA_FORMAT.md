# Data formats and assumptions

Study data are not included. Example paths below are placeholders; provide your real files locally and edit Section 0.

| Setting | Example local path | Required content |
|---|---|---|
| `PLOT_IMAGES_DIR` | `data/plot_images` | One georeferenced hyperspectral image per plot; `.tif`, `.tiff`, `.img` or `.dat` |
| `CROWN_SHP` | `data/crowns/final_crowns.shp` | Segmented crown polygons |
| `COMPLETE_WL_PATH` | `data/wavelengths/Cubert_Wavelengths.csv` | One wavelength per raster band in raster order |
| `LEAF_FILES[0]` | `data/leaf/Day_1.csv` | Healthy leaf spectra, class index 0 |
| `LEAF_FILES[1]` | `data/leaf/Day_3.csv` | Transitional leaf spectra, class index 1 |
| `LEAF_FILES[2]` | `data/leaf/Day7.csv` | Stressed leaf spectra, class index 2 |
| `FIELD_POINTS_SHP` | `data/field/Li600_labels.shp` | Field-assessed crown points; set to `None` only when unavailable |

All rasters need a consistent band/wavelength interpretation. Keep ENVI header files beside `.dat` or `.img` data. Keep all shapefile companion files together, including `.shx`, `.dbf` and `.prj` where applicable.

## Wavelength CSV

Include a header. The first column holds wavelengths in nanometres; other columns are ignored. There must be one row for each raster band, in exactly the raster's band order. The following is a layout illustration, not a complete wavelength file:

```csv
wavelength
418.0
422.0
426.0
```

## Leaf CSVs

Include a header. The first column contains wavelengths in nanometres; every remaining column contains one spectrum. All class files should share a wavelength grid. Column headers become sample identifiers used for grouping.

```csv
wavelength,leaf12_1,leaf12_2,leaf13_1
418.0,0.051,0.049,0.055
422.0,0.053,0.052,0.058
426.0,0.056,0.054,0.060
```

These numbers illustrate the layout only and are not supplied training data. The notebook detects common reflectance scales, but verify the inferred scale against your actual input. Leaf spectra are either matched to UAV centres within tolerance or resampled with Gaussian spectral responses. Selection limits the network bands; physical inversion, screening and plausibility checks still use the available full aligned spectrum.

If several columns are repeated measurements of the same leaf or tree, configure `leaf_group_mode="regex"` with a suitable expression or `"every_n"` with the correct repeat count. The default `"column"` treats each column as a different group and cannot prevent repeated-measurement leakage by itself.

## Crown polygons and field points

The crown layer may have a `crown_id`; otherwise IDs are assigned by file order. An internal `uid` distinguishes crowns across plots. Keep the `plot`, original `crown_id` and internal `uid` columns when joining outputs.

The implementation assumes that imagery, crown polygons and field points already have matching projected coordinates. It assigns the image CRS label to vectors without transforming coordinates. **Reproject and verify overlays before running**; changing a CRS label does not correct coordinate differences. `FIELD_SNAP_M` is interpreted in map units, so confirm metre-based coordinates if using its default 1.0 value.

The field-point attribute is configured as `hlth_id`:

| Field label | Model index | Class |
|---|---|---|
| 1 | 0 | Healthy |
| 2 | 1 | Transitional |
| 3 | 2 | Stressed |

With field holdout enabled, matched assessed crowns are excluded from adaptation, selection and threshold samples. The notebook does not calculate field accuracy. Check the held-out crown export before using field data for independent validation.

The crown inventory lists crowns without usable imagery separately. Classification outputs therefore represent classifiable crowns rather than every source polygon. Training screening and final map eligibility differ: final mapping can classify valid shaded or non-leaf pixels within a crown, with evidence selection handled by the crown-fusion rule.
