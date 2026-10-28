# -*- coding: utf-8 -*-
"""MangroveBridge-UDA: sequential Python export of MangroveBridge_UDA.ipynb.

Run from the repository root after editing Section 0 paths. Execution starts
the entire research pipeline, including tuning and multi-seed training.
The notebook is the primary editable source.
"""


# %%
# ==========================================================
# SECTION 0 — IMPORTS, PATHS, CONFIGURATION
# ==========================================================
import copy, json, math, os, random, re, time, warnings
from pathlib import Path

import numpy as np
import importlib, sys as _sys
if not hasattr(np.random, "mtrand"):
    # numpy.random half-loaded in this kernel (e.g. an earlier run of this cell was interrupted or failed while
    # importing): re-attach the sub-module, otherwise SciPy fails with "numpy.random has no attribute 'mtrand'"
    np.random.mtrand = _sys.modules.get("numpy.random.mtrand") or importlib.import_module("numpy.random.mtrand")
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.features import shapes as rio_shapes
from rasterio.transform import Affine
from shapely.geometry import shape as shp_shape
from shapely.ops import unary_union

from scipy.optimize import least_squares
from scipy.signal import savgol_filter, savgol_coeffs
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import prosail                                   # PROSPECT + 4SAIL (Numba-compiled; works on Windows)

# ------------------------------ INPUTS ---------------------------------------
# Study plots: a FOLDER with one hyperspectral image per plot (9 plots -> 9 images; the file name is the plot name).
# All plots are treated as ONE image: one pooled pixel sample, one model; maps are written per plot.
PLOT_IMAGES_DIR  = Path(r"data/plot_images")                              # <-- SET
PLOT_IMAGE_EXT   = (".tif", ".tiff", ".img", ".dat")
CROWN_SHP        = Path(r"data/crowns/final_crowns.shp")
COMPLETE_WL_PATH = Path(r"data/wavelengths/Cubert_Wavelengths.csv")  # UAV band centres, 1st column
LEAF_FILES = [  # (path, class index, class name)
    (Path(r"data/leaf/Day_1.csv"), 0, "Healthy"),
    (Path(r"data/leaf/Day_3.csv"), 1, "Transitional"),
    (Path(r"data/leaf/Day7.csv"),  2, "Stressed"),
]
CLASS_NAMES = ["Healthy", "Transitional", "Stressed"]   # ordinal; same names as the field labels
K = len(CLASS_NAMES)

# Field-assessed crowns: POINT shapefile, one point per assessed crown, health class in attribute "hlth_id".
# Used ONLY to keep these crowns OUT of training (their pixels are removed from every training, selection and threshold
# sample). This notebook never computes field accuracy: you validate the per-seed crown maps yourself at the end.
FIELD_POINTS_SHP  = Path(r"data/field/Li600_labels.shp")                  # <-- SET (None = no field data)
FIELD_LABEL_FIELD = "hlth_id"
FIELD_LABEL_MAP   = {1: 0, 2: 1, 3: 2}   # hlth_id: 1 = Healthy, 2 = Transitional, 3 = Stressed -> class index 0, 1, 2
FIELD_SNAP_M      = 1.0       # a point that falls just outside every crown is assigned to the nearest crown within this
                              # distance (map units, m); 0 = only points inside a crown polygon
FIELD_HOLDOUT     = True      # field crowns are kept aside: their pixels never enter training, checkpoint selection or
                              # thresholds, so your own validation at the end stays independent

# ------------------------------ SEEDS ----------------------------------------
SEEDS        = [42, 123, 2024]   # training seeds (fixed a priori): results are reported for every seed and as mean ± SD
MAP_SEED     = "best_dev"        # seed for the main maps: "best_dev" = lowest DEV risk (label-free), or a seed number
RESUME_SEEDS = True              # reuse saved runs (OUT_DIR/runs/<setting>/seed_<s>/teacher.pt) instead of retraining

# ------------------------------ PARAMETER OPTIMISATION (label-free) ----------
# Stage 1: each value below is tried ONE AT A TIME (all other settings at their CFG value) with seed TUNE_SEED.
# Stage 2: the TUNE_TOP best settings are trained with all SEEDS; the setting with the lowest mean DEV risk wins and
# becomes the final model. Ranking = DEV risk (lower is better); a setting is only eligible if its held-out leaf
# macro-F1 is no more than TUNE_LEAF_F1_TOL below the default. No field data are used. Epochs stay 60 x 100.
RUN_TUNING       = True
TUNE_SEED        = 42
TUNE_GRID        = dict(lr=[1e-4, 1e-3], w_adv=[0.5], w_coral=[0.25, 1.0], w_pl=[0.5], dim=[32])
TUNE_TOP         = 2
TUNE_LEAF_F1_TOL = 0.05

# ------------------------------ BAND SELECTION -------------------------------
# Bands chosen beforehand by your lab-vs-field spectral analysis (None = all bands). Give a list of wavelengths in nm
# and/or (lo, hi) ranges, e.g. [531, 570, (700, 750), 800], or a CSV file whose FIRST column holds the wavelengths (nm).
# Each wavelength is matched to the nearest model band (within SELECTED_BANDS_TOL_NM nm). Only these bands enter the
# network; the full spectrum is still used for PROSPECT / 4SAIL, pixel screening (NDVI, NDWI, shadow) and REP / PRI.
SELECTED_BANDS        = [(506, 646), (690, 742), (918, 934)]   # your ranges (nm): 36 + 14 + 5 = 55 bands
SELECTED_BANDS_TOL_NM = 3.0
BAND_RUN_NAME         = "selected_bands"         # with a selection, results go to OUT_DIR/<BAND_RUN_NAME>; all-band results stay

# ------------------------------ OUTPUTS (new folder) -------------------------
OUT_DIR = Path(r"outputs")
if SELECTED_BANDS is not None and BAND_RUN_NAME:
    OUT_DIR = OUT_DIR / BAND_RUN_NAME            # selected-band run in its own folder (does not overwrite all-band models)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DATA = OUT_DIR / "figure_data"           # everything the figure notebook needs (no model re-run required)
FIG_DATA.mkdir(parents=True, exist_ok=True)

CFG = dict(
    seed=42,
    # bands
    band_min=None, band_max=None,      # None = use the full leaf range (418-982 nm); set numbers to narrow it
    match_tol_nm=1.0,                  # max |leaf wavelength - UAV band centre| accepted when matching bands
    patch=4, fwhm_nm=10.0, reflectance_scale=None,
    # leaf groups — CHECK (Section 1)
    leaf_group_mode="column", leaf_group_regex=r"^(.*?)[_\-\s\.]?\d+$", n_replicates=1,
    # UAV sampling and screening
    rows_per_strip=64, n_target=150_000, n_background=20_000, n_screened=20_000,
    bg_ndvi_max=0.30, min_background=20,   # background = non-crown pixels with NDVI below this (see Section 3 fallbacks)
    ndvi_min=0.30, ndwi_max=0.0,
    # PROSPECT
    prospect_version="D", cw_prior=0.030, cm_prior=0.010, prot_prior=0.0010, cbc_prior=0.009,
    inv_range=(450.0, 950.0),
    # 4SAIL library
    n_sim_per_leaf=60, lai=(1.0, 6.0), ala=(40.0, 75.0), hspot=(0.05, 0.5),
    sza_flight=35.0, sza_jitter=5.0,               # <-- SET sza_flight FROM YOUR FLIGHT TIME/LOCATION
    vza=(0.0, 16.0), raa=(0.0, 180.0), n_bg_clusters=4, noise_scale=1.0,
    # views
    sg_window=7, sg_order=2,
    # network
    dim=64, depth=2, heads=4, d_state=8, dropout=0.1, grad_checkpoint=True, use_ssm=True,
    # training
    epochs=60, iters_per_epoch=100, warmup_epochs=10,
    src_batch=126, leaf_frac=0.2, tgt_crowns=24, tgt_per_crown=8,
    lr=3e-4, weight_decay=1e-2, ema=0.995, kl_anneal_epochs=10, grad_clip=1.0,
    w_trait=0.1, w_adv=1.0, w_coral=0.5, w_pl=1.0,
    pl_tau_min=0.60, pl_tau_max=0.95, pl_max_vacuity=0.50,
    prior_floor=0.03, prior_momentum=0.5,
    prior_freeze=True,                 # estimate the UAV class mix once, at the end of warm-up, then keep it fixed
    pl_class_balance=True,             # every class contributes equally to the pseudo-label loss (stops drift)
    ckpt_every=5, lso_mode=None,                  # None | "LAI"  (leave-structure-out experiment)
    # inference
    vac_percentile=95.0, maha_percentile=99.0, crown_min_px=10, infer_batch=2048,
)
CFG_BASE = None                    # filled in Section 6c (settings before the winning tuning setting is applied)
ABLATION = dict(physics_bridge=True, augmentation=True, dsbn=True, adversarial=True, coral=True,
                self_training=True, label_shift=True, trait_head=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(CFG["seed"])
rng = np.random.default_rng(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, "| torch", torch.__version__)

# %%
# ==========================================================
# SECTION 1 — LEAF DATA, BANDS, GROUPS, SPLIT
# ==========================================================
# Settings added in later versions: filled in here if an older Section 0 is still in memory
for _k, _v in dict(band_min=None, band_max=None, match_tol_nm=1.0, bg_ndvi_max=0.30, min_background=20,
                   prior_freeze=True, pl_class_balance=True, use_ssm=True).items():
    CFG.setdefault(_k, _v)
CFG["crown_min_px"] = min(CFG["crown_min_px"], 10)

def load_wavelengths(path):
    """UAV band-centre CSV: wavelengths (nm) in the FIRST column, one row per raster band, in raster band order."""
    return pd.read_csv(path).iloc[:, 0].astype(float).to_numpy()


def load_leaf_csv(path, label, class_name):
    """Leaf CSV: first column = wavelength (nm); every other column = one spectrum (header = sample name)."""
    df = pd.read_csv(path)
    wl = df.iloc[:, 0].astype(float).to_numpy()
    order = np.argsort(wl)
    R = df.iloc[order, 1:].to_numpy(dtype=float).T
    names = [f"{class_name}:{c}" for c in df.columns[1:]]
    return wl[order], R, np.full(len(R), label, dtype=np.int64), names


def align_leaf_to_uav(wl_leaf, R, wl_uav, tol, fwhm, band_min=None, band_max=None, patch=4):
    """Put leaf spectra on the UAV band grid.
    Mode 'matched' (expected: leaves already resampled to Cubert): each leaf wavelength is paired with the UAV band
    whose centre is within `tol` nm. Mode 'resampled' (fallback, e.g. 1-nm spectrometer data): leaf spectra are
    convolved with Gaussian band responses (FWHM = `fwhm`) at every UAV band centre inside the leaf range.
    Returns 0-based UAV band indices, leaf spectra on those bands, and the mode used."""
    lo = max(wl_leaf.min(), band_min if band_min is not None else -np.inf)
    hi = min(wl_leaf.max(), band_max if band_max is not None else np.inf)
    sel = np.flatnonzero((wl_leaf >= lo) & (wl_leaf <= hi))
    idx = np.array([int(np.argmin(np.abs(wl_uav - w))) for w in wl_leaf[sel]])
    err = np.abs(wl_uav[idx] - wl_leaf[sel])
    if err.max() <= tol and len(np.unique(idx)) == len(idx):
        mode, band_idx, Ra = "matched", idx, R[:, sel]
    else:
        band_idx = np.flatnonzero((wl_uav >= lo + fwhm / 2) & (wl_uav <= hi - fwhm / 2))
        sig = fwhm / 2.3548
        M = np.exp(-0.5 * ((wl_leaf[None, :] - wl_uav[band_idx][:, None]) / sig) ** 2)
        M /= M.sum(1, keepdims=True)
        mode, Ra = "resampled", R @ M.T
    n_drop = len(band_idx) % patch                       # whole patches only: drop extra bands at the noisier NIR end
    if n_drop:
        band_idx, Ra = band_idx[:-n_drop], Ra[:, :-n_drop]
    if np.any(np.diff(band_idx) != 1):
        warnings.warn("The matched UAV bands are not consecutive raster bands; patches and the derivative view "
                      "assume a continuous spectrum. Check the wavelength files.")
    return band_idx, Ra, mode, n_drop, float(err.max())


def detect_scale(values, what):
    """Median-based: robust to a few glint pixels above 1.0 (fixes CanopyBridge's p99 rule)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v) & (v > 0)]
    med = float(np.median(v)) if v.size else 0.0
    scale = 10000.0 if med > 100 else (100.0 if med > 1.5 else 1.0)
    print(f"[{what}] median value {med:.4g} -> dividing by {scale:g}")
    return scale


def make_groups(names, mode, regex, n_rep):
    if mode == "column":
        warnings.warn("leaf_group_mode='column': each spectrum is its own group. "
                      "If leaves were scanned more than once, use 'regex' or 'every_n'.")
        return np.array(names)
    out, counters = [], {}
    for nm in names:
        cls, col = nm.split(":", 1)
        if mode == "regex":
            m = re.match(regex, col)
            out.append(f"{cls}:{m.group(1) if m else col}")
        elif mode == "every_n":
            i = counters.get(cls, 0); counters[cls] = i + 1
            out.append(f"{cls}:{i // n_rep}")
        else:
            raise ValueError(mode)
    return np.array(out)


wl_full = load_wavelengths(COMPLETE_WL_PATH)                       # all UAV bands (e.g. 350-1002 nm)

# Leaf spectra: all three files must share one wavelength grid (418-982 nm, resampled to Cubert)
parts = [load_leaf_csv(p, k, n) for p, k, n in LEAF_FILES]
wl_leaf = parts[0][0]
R_parts = []
for (wl_i, R_i, _, _), (p, _, _) in zip(parts, LEAF_FILES):
    if len(wl_i) != len(wl_leaf) or not np.allclose(wl_i, wl_leaf, atol=0.05):
        if wl_i.min() > wl_leaf.min() + 0.05 or wl_i.max() < wl_leaf.max() - 0.05:
            raise ValueError(f"{Path(p).name} covers {wl_i.min()}-{wl_i.max()} nm, narrower than the first leaf file.")
        print(f"{Path(p).name}: different wavelength grid -> interpolated onto the first leaf file's grid")
        R_i = np.stack([np.interp(wl_leaf, wl_i, r) for r in R_i])
    R_parts.append(R_i)

R_leaf_full = np.vstack(R_parts)
scale_leaf = CFG["reflectance_scale"] or detect_scale(R_leaf_full, "leaf")
band_idx, R_leaf, align_mode, n_drop, max_err = align_leaf_to_uav(
    wl_leaf, R_leaf_full / scale_leaf, wl_full, CFG["match_tol_nm"], CFG["fwhm_nm"],
    CFG["band_min"], CFG["band_max"], CFG["patch"])
WL = wl_full[band_idx]                                                  # model wavelengths = UAV band centres
B = len(WL)
print(f"UAV wavelength file: {len(wl_full)} bands ({wl_full.min():.1f}-{wl_full.max():.1f} nm) | "
      f"leaf grid: {len(wl_leaf)} wavelengths ({wl_leaf.min():.1f}-{wl_leaf.max():.1f} nm)")
print(f"Leaf -> UAV alignment: {align_mode}"
      + (f" (max offset {max_err:.3f} nm)" if align_mode == "matched" else " (Gaussian band responses)")
      + f" | raster bands used: {band_idx[0] + 1}-{band_idx[-1] + 1}"
      + (f" | dropped {n_drop} NIR-end band(s) to fill whole patches" if n_drop else ""))
y_leaf = np.concatenate([p[2] for p in parts])
leaf_names = sum([p[3] for p in parts], [])
leaf_groups = make_groups(leaf_names, CFG["leaf_group_mode"], CFG["leaf_group_regex"], CFG["n_replicates"])

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG["seed"])
tr_leaf, va_leaf = next(sgkf.split(R_leaf, y_leaf, leaf_groups))
leaf_val = np.zeros(len(R_leaf), bool); leaf_val[va_leaf] = True
assert not set(leaf_groups[tr_leaf]) & set(leaf_groups[va_leaf]), "group leakage"

def band_near(w):
    return int(np.argmin(np.abs(WL - w)))


def resolve_band_selection(sel, wl, tol):
    """Selected wavelengths / (lo, hi) ranges / CSV (first column, nm) -> sorted indices into the model bands WL."""
    if sel is None:
        return None
    if isinstance(sel, (str, Path)):
        t = pd.read_csv(sel, header=None)
        sel = list(pd.to_numeric(t.iloc[:, 0], errors="coerce").dropna().astype(float))
        if not sel:
            raise ValueError(f"No wavelengths found in the first column of {sel}")
    idx, rows, missed = [], [], []
    for item in sel:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            lo, hi = sorted(float(v) for v in item)
            hit = np.flatnonzero((wl >= lo) & (wl <= hi))
            if not len(hit):
                missed.append(f"{lo:g}-{hi:g} nm"); continue
            idx += hit.tolist(); rows.append((f"{lo:g}-{hi:g}", f"{wl[hit[0]]:.1f}-{wl[hit[-1]]:.1f}", len(hit)))
        else:
            w = float(item); j = int(np.argmin(np.abs(wl - w)))
            if abs(wl[j] - w) > tol:
                missed.append(f"{w:g} nm"); continue
            idx.append(j); rows.append((f"{w:g}", f"{wl[j]:.1f}", 1))
    idx = sorted(set(idx))
    if missed:
        warnings.warn(f"Not in the model band range {wl[0]:.0f}-{wl[-1]:.0f} nm (or > {tol} nm from a band centre), "
                      f"ignored: {', '.join(missed)}")
    if len(idx) < 3:
        raise ValueError(f"Band selection gives {len(idx)} model band(s); at least 3 are needed.")
    print("Band selection (requested -> model band centre(s), n bands):")
    for r_ in rows:
        print(f"  {r_[0]:>11s} -> {r_[1]:>13s} ({r_[2]})")
    return idx


BAND_SEL = resolve_band_selection(SELECTED_BANDS, WL, SELECTED_BANDS_TOL_NM)
if BAND_SEL is None:
    warnings.warn("SELECTED_BANDS is None: the model runs on ALL bands. Set SELECTED_BANDS in Section 0 for the "
                  "selected-band model.")
CFG["band_sel"] = BAND_SEL                                       # read by the network (Section 4a)
if BAND_SEL is None:
    print(f"Bands: {B} ({WL[0]:.1f}-{WL[-1]:.1f} nm) -> {B // CFG['patch']} tokens of {CFG['patch']} bands")
else:
    print(f"Bands: {len(BAND_SEL)} selected of {B} -> {len(BAND_SEL)} tokens of 1 band "
          f"({', '.join(f'{w:.0f}' for w in WL[BAND_SEL])} nm) | results in {OUT_DIR}")
print(f"Leaf spectra: {len(R_leaf)} in {len(np.unique(leaf_groups))} groups")
for k in range(K):
    print(f"  {CLASS_NAMES[k]:9s} train {np.sum((y_leaf == k) & ~leaf_val):4d} | val {np.sum((y_leaf == k) & leaf_val):4d}")

# %%
# ==========================================================
# SECTION 2 — PLOT IMAGES, CROWN RASTERS (NO MASKING), STRIP-WISE SAMPLING, SCREENING
# ==========================================================
# Settings added in later versions: filled in here if an older Section 0 is still in memory
for _k, _v in dict(band_min=None, band_max=None, match_tol_nm=1.0, bg_ndvi_max=0.30, min_background=20).items():
    CFG.setdefault(_k, _v)

# Coordinates: the plot images, the crown shapefile and the field points are already in the same projected system
# (checked by overlaying them). They are therefore NEVER reprojected; the crowns and points simply take the CRS label of
# the plot images, so no coordinate transformation (and no PROJ/GDAL datum lookup) is ever needed.
def use_image_crs(gdf, crs):
    """Same coordinates, image CRS label (no reprojection)."""
    try:
        return gdf.set_crs(crs, allow_override=True)
    except Exception:
        return gdf                      # keep the coordinates and the original CRS label


def otsu(x, nbins=256):
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, bins=nbins)
    p = hist / max(hist.sum(), 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p); w1 = 1 - w0
    m0 = np.cumsum(p * mids) / np.clip(w0, 1e-12, None)
    m1 = (np.sum(p * mids) - np.cumsum(p * mids)) / np.clip(w1, 1e-12, None)
    return float(mids[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def spectral_indices(S):
    """S: (n, B) reflectance -> NDVI, NDWI, NIR brightness."""
    red, nir, green, nir2 = S[:, band_near(670)], S[:, band_near(800)], S[:, band_near(550)], S[:, band_near(860)]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    ndwi = (green - nir2) / np.maximum(green + nir2, 1e-6)
    bright = S[:, (WL >= 780) & (WL <= 900)].mean(1)
    return ndvi, ndwi, bright


# bands that are actually used: the network bands (all, or the selected ones) + the bands of NDVI / NDWI / brightness
USED_IDX = np.array(sorted(set(range(B) if CFG.get("band_sel") is None else CFG["band_sel"])
                           | {band_near(w) for w in (550, 670, 800, 860)}
                           | set(np.flatnonzero((WL >= 780) & (WL <= 900)).tolist())))
NIR_IDX = band_near(800)


def spectra_ok(raw, nodata):
    """raw: (n, B) raster values of the model bands -> pixels with a valid spectrum (finite in the used bands, not the
    nodata value there, NIR > 0). Values <= 0 in single bands are allowed (fixed by `clean_spectra`)."""
    u = raw[:, USED_IDX]
    ok = np.all(np.isfinite(u), 1) & (raw[:, NIR_IDX] > 0)
    if nodata is not None and np.isfinite(nodata) and nodata != 0:
        ok &= ~np.any(u == nodata, 1)
    return ok


def has_image(raw, nodata):
    """Pixels that carry image data at all (any used band finite, > 0 and not nodata). Crown parts beyond the image's
    data area (the empty border of a plot image) are not part of the map."""
    u = raw[:, USED_IDX]
    good = np.isfinite(u) & (u > 0)
    if nodata is not None and np.isfinite(nodata):
        good &= u != nodata
    return good.any(1)


def clean_spectra(raw, scale):
    """Scale to reflectance; non-finite values and values <= 0 become 1e-4 (the network uses log reflectance)."""
    S = raw.astype(np.float64) / scale
    S[~np.isfinite(S)] = 1e-4
    return np.maximum(S, 1e-4)


def discover_plot_images(folder):
    """One hyperspectral image per study plot (file name = plot name)."""
    folder = Path(folder)
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in PLOT_IMAGE_EXT)
    if not files:
        raise ValueError(f"No plot images ({', '.join(PLOT_IMAGE_EXT)}) found in {folder}")
    return [dict(name=f.stem, hsi=f) for f in files]


def read_crowns():
    """All segmented crowns, read once; crown_id from the attribute table or 1..n in file order."""
    c = gpd.read_file(CROWN_SHP)
    c = c[c.geometry.notnull() & ~c.geometry.is_empty].copy()
    if "crown_id" not in c.columns:
        c["crown_id"] = np.arange(1, len(c) + 1)
    c["crown_id"] = c["crown_id"].astype(int)
    if c["crown_id"].duplicated().any():
        warnings.warn("crown_id values in CROWN_SHP are not unique; field crowns may be matched ambiguously.")
    return c


def make_crown_raster(p, crowns_all, uid_offset):
    """Burn every crown that overlaps the plot image into a crown raster on the image's own grid (full polygons, no
    masking or clipping). Returns (crowns, raster path, next free uid). uids are unique across all plots."""
    with rasterio.open(p["hsi"]) as src:
        if src.count != len(wl_full):
            raise ValueError(f"{p['name']}: image has {src.count} bands, wavelength file has {len(wl_full)}.")
        H, W, transform, crs, b = src.height, src.width, src.transform, src.crs, src.bounds
        meta = src.meta.copy()
    c = crowns_all.cx[b.left:b.right, b.bottom:b.top].copy()       # crowns overlapping the image extent (same coordinates)
    if not len(c):
        x0, y0, x1, y1 = crowns_all.total_bounds
        raise ValueError(f"{p['name']}: no crowns over the image. Image extent x {b.left:.1f}-{b.right:.1f}, "
                         f"y {b.bottom:.1f}-{b.top:.1f}; crown shapefile extent x {x0:.1f}-{x1:.1f}, y {y0:.1f}-{y1:.1f}.")
    c = use_image_crs(c, crs)
    c["plot"] = p["name"]
    c["uid"] = uid_offset + np.arange(1, len(c) + 1)
    next_uid = uid_offset + len(c) + 1                              # every uid handed out here is reserved for this plot
    arr = rasterize([(g, int(u)) for g, u in zip(c.geometry, c["uid"])], out_shape=(H, W), transform=transform,
                    fill=0, dtype="int32")
    n_hit = len(np.unique(arr[arr > 0]))
    path = OUT_DIR / f"{p['name']}_crown_id.tif"
    meta.update(count=1, dtype="int32", nodata=0, driver="GTiff", compress="deflate")
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr, 1)
    print(f"{p['name']}: {W} x {H} px | {len(c):,} crowns over the image extent, {n_hit:,} cover at least one pixel "
          f"({int((arr > 0).sum()):,} crown pixels)")
    return c, path, next_uid


def sample_uav(seed):
    """Sunlit crown pixels, background and screened-out pixels from all plots, at one common sampling rate."""
    r = np.random.default_rng(seed)
    bands = [int(i) + 1 for i in band_idx]
    n_crown = n_other = 0
    for p in PLOT_INFO:
        with rasterio.open(p["crown_raster"]) as csrc:
            H, W = csrc.height, csrc.width
            c = sum(int((csrc.read(1, window=Window(0, r0, W, min(512, H - r0))) > 0).sum()) for r0 in range(0, H, 512))
        n_crown += c; n_other += H * W - c
    q_t = min(1.0, 3.0 * CFG["n_target"] / max(n_crown, 1))
    q_b = min(1.0, 20.0 * CFG["n_background"] / max(n_other, 1))
    tgt, tid, bg = [], [], []
    for p in PLOT_INFO:
        n_before = sum(len(t) for t in tgt)
        with rasterio.open(p["hsi"]) as src, rasterio.open(p["crown_raster"]) as csrc:
            H, W, nodata = src.height, src.width, src.nodata
            for r0 in range(0, H, CFG["rows_per_strip"]):
                win = Window(0, r0, W, min(CFG["rows_per_strip"], H - r0))
                cr = csrc.read(1, window=win).ravel()
                pt = (cr > 0) & (r.random(cr.size) < q_t)
                pb = (cr == 0) & (r.random(cr.size) < q_b)
                if not (pt.any() or pb.any()):
                    continue
                spec = src.read(bands, window=win).astype(np.float32).reshape(len(bands), -1).T
                ok = spectra_ok(spec, nodata)
                tgt.append(spec[pt & ok]); tid.append(cr[pt & ok]); bg.append(spec[pb & ok])
        print(f"  {p['name']}: {sum(len(t) for t in tgt) - n_before:,} crown pixels sampled (before screening)")
    tgt, tid = np.concatenate(tgt), np.concatenate(tid)
    bg = np.concatenate(bg) if bg else np.zeros((0, len(bands)), np.float32)
    scale = CFG["reflectance_scale"] or detect_scale(tgt[:, USED_IDX], "UAV")
    tgt, bg = clean_spectra(tgt, scale), clean_spectra(bg, scale)

    ndvi, ndwi, bright = spectral_indices(tgt)
    veg = (ndvi >= CFG["ndvi_min"]) & (ndwi < CFG["ndwi_max"])
    shadow_thr = otsu(bright[veg]) if veg.sum() > 100 else 0.0
    keep = veg & (bright > shadow_thr)
    print(f"Sampled crown pixels: {len(tgt):,} | sunlit vegetation kept for TRAINING: {keep.mean():.1%} "
          f"(NDVI >= {CFG['ndvi_min']}, NDWI < {CFG['ndwi_max']}, NIR > {shadow_thr:.3f})")
    scr = tgt[~keep]
    if len(scr) > CFG["n_screened"]:
        scr = scr[r.choice(len(scr), CFG["n_screened"], replace=False)]
    tgt, tid = tgt[keep], tid[keep]
    if len(tgt) > CFG["n_target"]:
        s = r.choice(len(tgt), CFG["n_target"], replace=False); tgt, tid = tgt[s], tid[s]
    # Background (mud / water / debris) outside crowns but inside the plot images. Dense canopies can leave few or
    # no such pixels, so report what was found and relax the rule step by step (Section 3).
    print(f"Non-crown pixels sampled with valid data: {len(bg):,}")
    if len(bg):
        bndvi = spectral_indices(bg)[0]
        print("  their NDVI percentiles 5/25/50/75/95: " + " / ".join(f"{v:.2f}" for v in np.percentile(bndvi, [5, 25, 50, 75, 95])))
        if np.mean(bndvi >= 0.6) > 0.5:
            warnings.warn(f"{np.mean(bndvi >= 0.6):.0%} of the image pixels OUTSIDE the crown polygons are dense canopy "
                          "(NDVI >= 0.6): the crown polygons do not cover most trees in the plot images. Check in QGIS "
                          "that the crown shapefile covers these plots and that its CRS is the SAME as the images' CRS "
                          "(Layer Properties > Source; on-the-fly reprojection can hide a mismatch).")
        bg = bg[bndvi < CFG["bg_ndvi_max"]]          # never fall back to vegetation: it would be mistaken for soil
        print(f"  with NDVI < {CFG['bg_ndvi_max']} (mud / water / debris): {len(bg):,}")
    if len(bg) > CFG["n_background"]:
        bg = bg[r.choice(len(bg), CFG["n_background"], replace=False)]
    per_plot = {p["name"]: int(np.isin(tid, crowns_gdf.loc[crowns_gdf["plot"] == p["name"], "uid"]).sum()) for p in PLOT_INFO}
    print(f"Training target pixels: {len(tgt):,} in {len(np.unique(tid))} crowns ("
          + ", ".join(f"{k} {v:,}" for k, v in per_plot.items()) + f") | background: {len(bg):,} | screened-out sample: {len(scr):,}")
    mask_params = dict(scale=scale, ndvi_min=CFG["ndvi_min"], ndwi_max=CFG["ndwi_max"], shadow_thr=shadow_thr)
    return tgt.astype(np.float64), tid.astype(np.int64), bg.astype(np.float64), scr.astype(np.float64), mask_params


def pixel_masks(S):
    """Leaf (vegetation) and sunlit-leaf pixels of reflectance spectra S, with the thresholds of the training screen."""
    ndvi, ndwi, bright = spectral_indices(S)
    veg = (ndvi >= MASK["ndvi_min"]) & (ndwi < MASK["ndwi_max"])
    return veg, veg & (bright > MASK["shadow_thr"])


PLOTS = discover_plot_images(PLOT_IMAGES_DIR)
print(f"{len(PLOTS)} plot images: " + ", ".join(p["name"] for p in PLOTS))
_crowns_all = read_crowns()
print(f"Crowns: {len(_crowns_all):,} polygons (used in the coordinates of the plot images, no reprojection)")
PLOT_INFO, _parts, _uid0 = [], [], 0
for _p in PLOTS:
    _cg, _cr, _uid0 = make_crown_raster(_p, _crowns_all, _uid0)
    PLOT_INFO.append(dict(name=_p["name"], hsi=_p["hsi"], crown_raster=_cr))
    _parts.append(_cg)
_crs0 = _parts[0].crs
crowns_gdf = use_image_crs(gpd.GeoDataFrame(pd.concat([g for g in _parts if len(g)], ignore_index=True),
                                            geometry=_parts[0].geometry.name), _crs0)
del _crowns_all
if crowns_gdf["uid"].duplicated().any():
    raise RuntimeError("Crown uids are not unique across plots; delete the *_crown_id.tif files in OUT_DIR and re-run Section 2.")
CROWN_INFO = crowns_gdf[["uid", "plot", "crown_id"]].set_index("uid")      # uid -> plot, original crown_id (one row per uid)
R_tgt, tgt_crown, R_bg, R_screened, MASK = sample_uav(CFG["seed"])

# %%
# ==========================================================
# SECTION 2b — THE CROWNS AND PIXELS THAT WILL BE CLASSIFIED (before training)
# ==========================================================
INV_DIR = OUT_DIR / "crown_inventory"; INV_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_TYPES = {1: "sunlit leaf", 2: "shaded leaf", 3: "non-leaf (bark, soil, water inside the crown)"}


def crown_inventory(p):
    """Per crown of plot p: image pixels, pixels with a valid spectrum (classified by the model) and their type.
    Crown pixels beyond the image's data area are removed from the crown raster (they are outside the image)."""
    bands = [int(i) + 1 for i in band_idx]
    with rasterio.open(p["hsi"]) as src, rasterio.open(p["crown_raster"]) as csrc:
        H, W, nod = src.height, src.width, src.nodata
        arr, prof = csrc.read(1), csrc.profile
        C = int(crowns_gdf.loc[crowns_gdf["plot"] == p["name"], "uid"].max()) + 1
        n_px, n_val, n_sun, n_shd, n_out = (np.zeros(C, np.int64) for _ in range(5))
        for r0 in range(0, H, CFG["rows_per_strip"]):
            win = Window(0, r0, W, min(CFG["rows_per_strip"], H - r0)); h = int(win.height)
            cr = arr[r0:r0 + h].ravel()
            inside = np.flatnonzero(cr > 0)
            if not len(inside):
                continue
            raw = src.read(bands, window=win).astype(np.float32).reshape(len(bands), -1).T[inside]
            img = has_image(raw, nod)
            np.add.at(n_out, cr[inside][~img], 1)
            arr[r0:r0 + h].reshape(-1)[inside[~img]] = 0                    # crown part beyond the image data
            inside, raw = inside[img], raw[img]
            np.add.at(n_px, cr[inside], 1)
            ok = spectra_ok(raw, nod)
            if not ok.any():
                continue
            ids = cr[inside][ok]
            veg, sun = pixel_masks(clean_spectra(raw[ok], MASK["scale"]))
            np.add.at(n_val, ids, 1); np.add.at(n_sun, ids, sun.astype(int)); np.add.at(n_shd, ids, (veg & ~sun).astype(int))
    u = crowns_gdf.loc[crowns_gdf["plot"] == p["name"], "uid"].to_numpy()
    df = pd.DataFrame(dict(uid=u, plot=p["name"], crown_id=CROWN_INFO.loc[u, "crown_id"].to_numpy(), n_pixels=n_px[u],
                           n_valid=n_val[u], n_sunlit_leaf=n_sun[u], n_shaded_leaf=n_shd[u],
                           n_non_leaf=n_val[u] - n_sun[u] - n_shd[u], n_invalid=n_px[u] - n_val[u],
                           n_outside_image=n_out[u]))
    return df, arr, prof


def rgb_quicklook(path, max_px=900):
    """True-colour preview (decimated read, 2-98 % stretch) and its extent."""
    bl = [int(np.argmin(np.abs(wl_full - w))) + 1 for w in (640, 550, 470)]
    with rasterio.open(path) as src:
        f = max(1, int(math.ceil(max(src.height, src.width) / max_px)))
        h, w = int(math.ceil(src.height / f)), int(math.ceil(src.width / f))
        a = src.read(bl, out_shape=(3, h, w)).astype(np.float64)
        b = src.bounds
    img = np.ones((h, w, 3))
    good = np.all(np.isfinite(a), 0) & np.any(a > 0, 0)
    for i in range(3):
        v = a[i][good]
        lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 1)
        img[..., i] = np.clip((a[i] - lo) / max(hi - lo, 1e-9), 0, 1)
    img[~good] = 1.0
    return img, (b.left, b.right, b.bottom, b.top), (h, w)


_inv = []
for p in PLOT_INFO:
    df_, arr_, prof_ = crown_inventory(p)
    bad_ = df_.loc[df_["n_valid"] == 0, "uid"].to_numpy()
    if len(bad_):
        arr_[np.isin(arr_, bad_)] = 0        # crowns without any valid pixel cannot be classified: removed everywhere
    with rasterio.open(p["crown_raster"], "w", **prof_) as d_:
        d_.write(arr_, 1)
    _inv.append(df_); del arr_
inv = pd.concat(_inv, ignore_index=True)
not_cls = inv[inv["n_valid"] == 0]
inv_ok = inv[inv["n_valid"] > 0]
not_cls.to_csv(INV_DIR / "crowns_not_classifiable.csv", index=False)

# crowns without any valid pixel are removed from the crown table too: no "no data" crowns in the outputs
if len(not_cls):
    drop = set(not_cls["uid"])
    crowns_gdf = crowns_gdf[~crowns_gdf["uid"].isin(drop)].reset_index(drop=True)
    CROWN_INFO = crowns_gdf[["uid", "plot", "crown_id"]].set_index("uid")
    tgt_keep = ~np.isin(tgt_crown, list(drop)); R_tgt, tgt_crown = R_tgt[tgt_keep], tgt_crown[tgt_keep]
inv_ok.to_csv(INV_DIR / "crowns_to_classify.csv", index=False)

PLOT_INV = inv.groupby("plot").agg(
    crowns_over_image=("uid", "size"), crowns_to_classify=("n_valid", lambda s: int((s > 0).sum())),
    crowns_not_classifiable=("n_valid", lambda s: int((s == 0).sum())), crown_pixels=("n_pixels", "sum"),
    pixels_to_classify=("n_valid", "sum"), sunlit_leaf=("n_sunlit_leaf", "sum"), shaded_leaf=("n_shaded_leaf", "sum"),
    non_leaf=("n_non_leaf", "sum"), invalid_pixels=("n_invalid", "sum"),
    outside_image_pixels=("n_outside_image", "sum")).reindex([p["name"] for p in PLOT_INFO])
PLOT_INV.loc["All plots"] = PLOT_INV.sum()
for c_ in ("sunlit_leaf", "shaded_leaf", "non_leaf"):
    PLOT_INV[f"{c_}_pct"] = 100 * PLOT_INV[c_] / PLOT_INV["pixels_to_classify"].clip(lower=1)
PLOT_INV.to_csv(INV_DIR / "plots_to_classify.csv")
print("Crowns and pixels that WILL BE CLASSIFIED at the end (before any training):")
print(PLOT_INV[["crowns_over_image", "crowns_to_classify", "crowns_not_classifiable", "pixels_to_classify",
                "sunlit_leaf_pct", "shaded_leaf_pct", "non_leaf_pct", "invalid_pixels"]].round(1).to_string())
print(f"Every crown pixel with a valid spectrum gets a class; the {int(PLOT_INV.loc['All plots', 'invalid_pixels']):,} "
      "crown pixels with image data but no valid spectrum take the class of their crown; "
      f"{int(PLOT_INV.loc['All plots', 'outside_image_pixels']):,} crown pixels beyond the images' data area are outside "
      "the map. Lists: " + str(INV_DIR))

# overview figure: true colour + crowns that will be classified (blue) and crowns that cannot be (red)
n = len(PLOT_INFO); nc = min(3, n); nr = int(math.ceil(n / nc))
fig, axes = plt.subplots(nr, nc, figsize=(5.0 * nc, 5.0 * nr), squeeze=False)
for ax, p in zip(axes.ravel(), PLOT_INFO):
    img, ext, (h, w) = rgb_quicklook(p["hsi"])
    from rasterio.enums import Resampling
    with rasterio.open(p["crown_raster"]) as s_:
        cr_small = s_.read(1, out_shape=(h, w), resampling=Resampling.nearest)
    ax.imshow(img, extent=ext, interpolation="nearest")
    ov = np.zeros((h, w, 4)); ov[cr_small > 0] = (0.16, 0.47, 0.84, 0.45)
    ax.imshow(ov, extent=ext, interpolation="nearest")
    nb = not_cls[not_cls["plot"] == p["name"]]
    if len(nb):
        try:                                                        # outlines of the crowns that cannot be classified
            _g = _parts[[q["name"] for q in PLOT_INFO].index(p["name"])].set_index("uid").loc[nb["uid"]]
            gpd.GeoSeries(_g.geometry).boundary.plot(ax=ax, color="red", lw=0.6)
        except Exception:
            pass
    r_ = PLOT_INV.loc[p["name"]]
    ax.set_title(f"{p['name']}: {int(r_['crowns_to_classify'])} crowns, {int(r_['pixels_to_classify']):,} px"
                 + (f"  ({int(r_['crowns_not_classifiable'])} not classifiable)" if r_["crowns_not_classifiable"] else ""),
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.suptitle("Crowns that will be classified (blue) — before training", fontsize=11)
plt.tight_layout(); plt.savefig(INV_DIR / "crowns_to_classify_overview.png", dpi=200); plt.show()

# ---- field crowns: kept aside (never used for training, checkpoint selection or thresholds) ----


def load_field_points():
    """Field points -> crown labels. Returns (Series label by crown uid, point table, crown table)."""
    pts = gpd.read_file(FIELD_POINTS_SHP)
    pts = use_image_crs(pts, crowns_gdf.crs)              # same projected coordinates, no reprojection
    if FIELD_LABEL_FIELD not in pts.columns:
        raise ValueError(f"Attribute '{FIELD_LABEL_FIELD}' not found in {FIELD_POINTS_SHP.name}: {list(pts.columns)}")
    raw = pd.to_numeric(pts[FIELD_LABEL_FIELD], errors="coerce")
    vals = set(raw.dropna().astype(int))
    lab_map = FIELD_LABEL_MAP or ({1: 0, 2: 1, 3: 2} if vals <= {1, 2, 3} else {0: 0, 1: 1, 2: 2} if vals <= {0, 1, 2} else None)
    if lab_map is None:
        raise ValueError(f"{FIELD_LABEL_FIELD} has values {sorted(vals)}; set FIELD_LABEL_MAP, e.g. {{1: 0, 2: 1, 3: 2}}.")
    print(f"Field points: {len(pts)} | {FIELD_LABEL_FIELD} -> class: "
          + ", ".join(f"{k} -> {CLASS_NAMES[v]} (n = {int((raw == k).sum())})" for k, v in sorted(lab_map.items())))
    pts = pts.assign(label=raw.map(lab_map), x=pts.geometry.x, y=pts.geometry.y)
    pts = pts[pts["label"].notna()].copy()
    cr = crowns_gdf[["uid", "plot", "crown_id", "geometry"]]
    j = gpd.sjoin(pts[["label", "x", "y", "geometry"]], cr, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].copy()          # overlapping crowns: keep the first match
    j["match"] = np.where(j["uid"].notna(), "inside crown", "none")
    miss = j["uid"].isna()
    if miss.any() and FIELD_SNAP_M > 0:
        jn = gpd.sjoin_nearest(pts.loc[j.index[miss], ["label", "geometry"]], cr, how="inner",
                               max_distance=FIELD_SNAP_M, distance_col="dist")
        jn = jn[~jn.index.duplicated(keep="first")]
        if len(jn):
            j.loc[jn.index, ["uid", "plot", "crown_id"]] = jn[["uid", "plot", "crown_id"]].to_numpy()
            j.loc[jn.index, "match"] = f"nearest crown <= {FIELD_SNAP_M} m"
    pt_tab = pd.DataFrame(j.drop(columns=["geometry", "index_right"], errors="ignore"))
    pt_tab["label"] = pt_tab["label"].astype(int)
    print("  matched: " + ", ".join(f"{k} {v}" for k, v in pt_tab["match"].value_counts().items())
          + " (unmatched points lie outside every plot crown and are ignored)")
    if pt_tab["uid"].notna().mean() < 0.5:
        warnings.warn(f"Only {int(pt_tab['uid'].notna().sum())} of {len(pt_tab)} field points fall in (or within "
                      f"{FIELD_SNAP_M} m of) a plot crown. Points outside the plot images are expected; if the points "
                      "lie inside the plots, the crown polygons are missing there (see the crown counts above).")
    ok = pt_tab[pt_tab["uid"].notna()].copy(); ok["uid"] = ok["uid"].astype(int)
    rows = []
    for uid, gg in ok.groupby("uid"):
        cnt = gg["label"].value_counts()
        tie = len(cnt) > 1 and cnt.iloc[0] == cnt.iloc[1]
        rows.append(dict(uid=uid, plot=gg["plot"].iloc[0], crown_id=int(gg["crown_id"].iloc[0]),
                         label=int(cnt.index[0]), n_points=len(gg), conflicting=len(cnt) > 1, tie=tie))
    crown_tab = pd.DataFrame(rows, columns=["uid", "plot", "crown_id", "label", "n_points", "conflicting", "tie"])
    n_tie = int(crown_tab["tie"].sum()) if len(crown_tab) else 0
    if n_tie:
        warnings.warn(f"{n_tie} crowns have points with tied, different labels; they are excluded from the hold-out list.")
    crown_tab = crown_tab[~crown_tab["tie"].astype(bool)] if len(crown_tab) else crown_tab
    print(f"  field crowns: {len(crown_tab)} (" + ", ".join(
        f"{CLASS_NAMES[k]} {int((crown_tab['label'] == k).sum())}" for k in range(K)) + ")")
    return pd.Series(crown_tab["label"].to_numpy(), index=crown_tab["uid"].to_numpy()), pt_tab, crown_tab


field = None                     # crowns with a field point (uid index); only their LOCATION is used, to hold them out
if FIELD_POINTS_SHP:
    field, _, _ = load_field_points()
    if FIELD_HOLDOUT:
        _held = np.isin(tgt_crown, field.index.to_numpy())
        R_tgt, tgt_crown = R_tgt[~_held], tgt_crown[~_held]
        print(f"Held out {len(field)} field crowns: {int(_held.sum()):,} of their sampled pixels removed from the "
              f"training sample ({len(R_tgt):,} target pixels remain). No field accuracy is computed in this notebook.")
    CROWN_INFO.loc[field.index, ["plot", "crown_id"]].reset_index().to_csv(OUT_DIR / "held_out_field_crowns.csv", index=False)

# %%
# ==========================================================
# SECTION 3 — PROSPECT INVERSION, BACKGROUND, NOISE, 4SAIL LIBRARY
# ==========================================================
GRID = np.arange(400, 2501, dtype=np.float64)
_sig = CFG["fwhm_nm"] / 2.3548
SRF = np.exp(-0.5 * ((GRID[None, :] - WL[:, None]) / _sig) ** 2)
SRF /= SRF.sum(1, keepdims=True)
INV = (WL >= CFG["inv_range"][0]) & (WL <= CFG["inv_range"][1])
TRAIT_NAMES = ["N", "Cab", "Car", "Ant", "Cbrown"]
LB, UB = np.array([1.0, 0, 0, 0, 0]), np.array([3.5, 120, 35, 15, 1.5])
X0 = np.array([1.8, 40, 10, 1, 0.05])


def prospect_bands(p):
    n, cab, car, ant, cbr = p
    if CFG["prospect_version"].upper() == "PRO":
        _, r, t = prosail.run_prospect(n, cab, car, cbr, CFG["cw_prior"], 0.0, ant=ant, prot=CFG["prot_prior"],
                                       cbc=CFG["cbc_prior"], prospect_version="PRO")
    else:
        _, r, t = prosail.run_prospect(n, cab, car, cbr, CFG["cw_prior"], CFG["cm_prior"], ant=ant,
                                       prospect_version=CFG["prospect_version"])
    return SRF @ r, SRF @ t


def invert_leaf(r_meas):
    sol = least_squares(lambda p: (prospect_bands(p)[0] - r_meas)[INV], X0, bounds=(LB, UB),
                        x_scale=[0.5, 20, 5, 2, 0.2], max_nfev=300)
    rb, tb = prospect_bands(sol.x)
    return sol.x, rb, tb, float(np.sqrt(np.mean((rb - r_meas)[INV] ** 2)))


t0 = time.time()
TRAITS = np.zeros((len(R_leaf), 5)); R_fit = np.zeros_like(R_leaf); T_fit = np.zeros_like(R_leaf)
fit_rmse = np.zeros(len(R_leaf))
for i in range(len(R_leaf)):
    TRAITS[i], R_fit[i], T_fit[i], fit_rmse[i] = invert_leaf(R_leaf[i])
    if (i + 1) % 25 == 0:
        print(f"  inverted {i+1}/{len(R_leaf)} ({time.time()-t0:.0f} s)")
print("PROSPECT fit RMSE: median %.4f | 95th pct %.4f" % (np.median(fit_rmse), np.percentile(fit_rmse, 95)))
print(pd.DataFrame(TRAITS, columns=TRAIT_NAMES).groupby(np.array(CLASS_NAMES)[y_leaf]).median().round(2))

T_leaf = T_fit * (1 - R_leaf) / np.clip(1 - R_fit, 1e-3, None)
T_leaf = np.clip(T_leaf, 0, np.clip(0.98 - R_leaf, 0, None))

# Scene background endmembers (wet mud, water, debris) and per-band noise
bg_source = "non-crown pixels"
if len(R_bg) < CFG["min_background"] and len(R_screened):
    # Fallback 1: non-vegetation pixels inside crown polygons (low NDVI or open water), e.g. when the raster is
    # masked outside the crowns or the canopy is closed
    nd_s, nw_s, _ = spectral_indices(R_screened)
    cand = R_screened[(nd_s < CFG["bg_ndvi_max"]) | (nw_s >= CFG["ndwi_max"])]
    if len(cand) >= 20:
        R_bg, bg_source = cand, "non-vegetation pixels inside crown polygons"
if len(R_bg) < 20:
    # Fallback 2: PROSAIL's built-in dry and wet soil spectra resampled to the Cubert bands
    warnings.warn("No usable mud/water pixels in the image; using PROSAIL's default dry and wet soil spectra as "
                  "background. Better: plot images that include some canopy gaps with mud or water.")
    soils = np.stack([SRF @ prosail.spectral_lib.soil.rsoil1, SRF @ prosail.spectral_lib.soil.rsoil2])
    R_bg = np.vstack([soils * f for f in (0.5, 0.75, 1.0)])           # a few brightness levels
    bg_source = "PROSAIL default soils (fallback)"
n_cl = int(min(CFG["n_bg_clusters"], len(np.unique(np.round(R_bg, 5), axis=0))))
km = KMeans(n_clusters=n_cl, n_init=10, random_state=CFG["seed"]).fit(R_bg)
BG = np.clip(km.cluster_centers_, 1e-3, 1.0)
print(f"Background endmembers: {n_cl} from {len(R_bg):,} spectra ({bg_source})")
_sub = R_tgt[rng.choice(len(R_tgt), min(20000, len(R_tgt)), replace=False)]
_res = _sub - savgol_filter(_sub, CFG["sg_window"], CFG["sg_order"], axis=1)
_h0 = savgol_coeffs(CFG["sg_window"], CFG["sg_order"])[CFG["sg_window"] // 2]
NOISE_SIGMA = 1.4826 * np.median(np.abs(_res - np.median(_res, 0)), 0) / np.sqrt(1 - _h0) * CFG["noise_scale"]

# 4SAIL library (canopy reflectance only — pixel-level shade/background mixing is done on the fly in Section 4)
def U(a):
    return rng.uniform(*a)

sim_R, sim_leaf, sim_meta = [], [], []
t0 = time.time()
if ABLATION["physics_bridge"]:
    for i in range(len(R_leaf)):
        r_i = np.ascontiguousarray(R_leaf[i], dtype=np.float64)
        t_i = np.ascontiguousarray(T_leaf[i], dtype=np.float64)
        for _ in range(CFG["n_sim_per_leaf"]):
            lai, ala, hs = U(CFG["lai"]), U(CFG["ala"]), U(CFG["hspot"])
            sza = CFG["sza_flight"] + rng.uniform(-CFG["sza_jitter"], CFG["sza_jitter"])
            vza, raa = U(CFG["vza"]), U(CFG["raa"])
            soil = np.ascontiguousarray(BG[rng.integers(len(BG))])
            rc = prosail.run_sail(r_i, t_i, lai, ala, hs, sza, vza, raa, typelidf=2, rsoil0=soil)
            sim_R.append(np.clip(rc, 1e-4, 1.0)); sim_leaf.append(i); sim_meta.append((lai, ala, sza, vza))
        if (i + 1) % 25 == 0:
            print(f"  4SAIL: leaves {i+1}/{len(R_leaf)} ({time.time()-t0:.0f} s)")
sim_R = np.array(sim_R, dtype=np.float64).reshape(-1, B)
sim_leaf = np.array(sim_leaf, dtype=np.int64)
sim_meta = pd.DataFrame(sim_meta, columns=["lai", "ala", "sza", "vza"])
print("4SAIL canopy library:", sim_R.shape)

fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
for k in range(K):
    i = np.where(y_leaf == k)[0][0]
    ax[0].plot(WL, R_leaf[i], label=f"{CLASS_NAMES[k]} measured"); ax[0].plot(WL, R_fit[i], "--", lw=0.8)
    if len(sim_R):
        ax[2].plot(WL, np.median(sim_R[y_leaf[sim_leaf] == k], 0), label=CLASS_NAMES[k])
ax[0].set_title("Leaf R (solid) vs PROSPECT fit (dashed)"); ax[0].legend(fontsize=7)
for b in BG: ax[1].plot(WL, b)
ax[1].set_title("Scene background endmembers"); ax[2].set_title("Median 4SAIL canopy by class"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.show()

# %%
# ==========================================================
# SECTION 4a — MODULES
# ==========================================================
SRC, TGT = 0, 1


def nonuniform_derivative(w):
    """(n, n) matrix: first derivative d/dλ on an irregular wavelength grid (second-order central differences,
    one-sided at the ends), used when only selected bands enter the network."""
    w = np.asarray(w, float); n = len(w); D = np.zeros((n, n))
    D[0, :2] = [-1.0, 1.0]; D[0] /= w[1] - w[0]
    D[-1, -2:] = [-1.0, 1.0]; D[-1] /= w[-1] - w[-2]
    for i in range(1, n - 1):
        a, b = w[i] - w[i - 1], w[i + 1] - w[i]
        D[i, i - 1], D[i, i], D[i, i + 1] = -b / (a * (a + b)), (b - a) / (a * b), a / (b * (a + b))
    return D


class SpectralViews(nn.Module):
    """Three brightness-invariant views. With a band selection (`sel`) only the selected bands are used: the input
    is reduced to them first and the derivative view is taken on their (irregular) wavelength grid."""
    def __init__(self, wl, sg_window, sg_order, sel=None):
        super().__init__()
        wl = np.asarray(wl, float)
        self.has_sel = sel is not None
        if self.has_sel:
            sel = np.asarray(sel, int)
            self.register_buffer("sel", torch.as_tensor(sel, dtype=torch.long))
            self.register_buffer("dmat", torch.tensor(nonuniform_derivative(wl[sel]), dtype=torch.float32))
            nb = len(sel)
        else:
            step = float(np.mean(np.diff(wl)))
            k = savgol_coeffs(sg_window, sg_order, deriv=1, use="dot") / step
            self.pad = sg_window // 2
            self.register_buffer("sg", torch.tensor(k, dtype=torch.float32).view(1, 1, -1))
            nb = len(wl)
        self.register_buffer("mu", torch.zeros(3, nb))
        self.register_buffer("sd", torch.ones(3, nb))

    def raw(self, R):
        if self.has_sel:
            R = R.index_select(1, self.sel)
        r = R.clamp_min(1e-4)
        lr = torch.log(r)
        v1 = lr - lr.mean(1, keepdim=True)
        if self.has_sel:
            v2 = (lr @ self.dmat.T) * 10.0
        else:
            v2 = F.conv1d(F.pad(lr.unsqueeze(1), (self.pad, self.pad), mode="replicate"), self.sg).squeeze(1) * 10.0
        v3 = r / r.norm(dim=1, keepdim=True) * math.sqrt(r.shape[1])
        return torch.stack([v1, v2, v3], 1)                                  # (n, 3, B)

    @torch.no_grad()
    def fit(self, R):
        v = self.raw(R)
        self.mu.copy_(v.mean(0)); self.sd.copy_(v.std(0).clamp_min(1e-6))

    def forward(self, R):
        return (self.raw(R) - self.mu) / self.sd


class WavelengthEncoding(nn.Module):
    def __init__(self, dim, n_freq=16, pmin=5.0, pmax=2000.0):
        super().__init__()
        periods = torch.logspace(math.log10(pmin), math.log10(pmax), n_freq)
        self.register_buffer("omega", 2 * math.pi / periods)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq + 1, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, centers, widths):
        a = centers[:, None] * self.omega[None, :]
        return self.mlp(torch.cat([torch.sin(a), torch.cos(a), widths[:, None] / 50.0], 1))


class SelectiveSSM(nn.Module):
    def __init__(self, dim, d_state=8, expand=2, d_conv=3, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        E = expand * dim
        self.E, self.N, self.dt_rank = E, d_state, max(1, math.ceil(dim / 16))
        self.in_proj = nn.Linear(dim, 2 * E)
        self.conv = nn.Conv1d(E, E, d_conv, padding=d_conv - 1, groups=E)
        self.x_proj = nn.Linear(E, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, E)
        dt = torch.exp(torch.rand(E) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))      # inverse softplus (Mamba init)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(E, 1)))
        self.D = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, dim)

    def forward(self, x):
        n, L, _ = x.shape
        xs, z = self.in_proj(x).chunk(2, dim=-1)
        xs = F.silu(self.conv(xs.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = torch.split(self.x_proj(xs), [self.dt_rank, self.N, self.N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        h = x.new_zeros(n, self.E, self.N)
        ys = []
        for t in range(L):
            dtt = dt[:, t].unsqueeze(-1)
            h = torch.exp(dtt * A) * h + dtt * Bm[:, t].unsqueeze(1) * xs[:, t].unsqueeze(-1)
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + xs * self.D
        return self.out_proj(y * F.silu(z))


class HybridBlock(nn.Module):
    """Bidirectional SSM (spectral continuity) + attention (non-local band interactions) + FFN.
    use_ssm=False removes the SSM path (ablation 'attention only')."""
    def __init__(self, dim, heads, d_state, drop, use_ssm=True):
        super().__init__()
        self.use_ssm = use_ssm
        if use_ssm:
            self.n1, self.fwd, self.bwd = nn.LayerNorm(dim), SelectiveSSM(dim, d_state), SelectiveSSM(dim, d_state)
        self.n2, self.attn = nn.LayerNorm(dim), nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.use_ssm:
            h = self.n1(x)
            x = x + self.drop(self.fwd(h) + self.bwd(h.flip(1)).flip(1))
        h = self.n2(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop(self.ffn(self.n3(x)))


class MangroveBridgeNet(nn.Module):
    def __init__(self, wl, cfg, n_classes=3, n_traits=4):
        super().__init__()
        d, sel = cfg["dim"], cfg.get("band_sel")
        self.views = SpectralViews(wl, cfg["sg_window"], cfg["sg_order"], sel)
        step = float(np.mean(np.diff(wl)))
        if sel is not None:                     # selected bands: one token per band (bands need not be adjacent)
            P = 1
            w = torch.tensor(np.asarray(wl, float)[np.asarray(sel, int)], dtype=torch.float32).view(-1, 1)
        else:                                   # all bands: tokens of `patch` adjacent bands
            P = cfg["patch"]
            w = torch.tensor(wl, dtype=torch.float32).view(-1, P)
        self.register_buffer("centers", w.mean(1))
        self.register_buffer("widths", w[:, -1] - w[:, 0] + step)
        self.patch = nn.Conv1d(3, d, kernel_size=P, stride=P)
        self.wenc = WavelengthEncoding(d)
        self.dsbn = nn.ModuleList([nn.BatchNorm1d(d), nn.BatchNorm1d(d)])
        self.blocks = nn.ModuleList([HybridBlock(d, cfg["heads"], cfg["d_state"], cfg["dropout"], cfg.get("use_ssm", True))
                                     for _ in range(cfg["depth"])])
        self.norm = nn.LayerNorm(d)
        self.pool_attn, self.pool_proj = nn.Linear(d, 1), nn.Linear(2 * d, d)
        self.ev_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.2), nn.Linear(d, 2 * (n_classes - 1)))
        self.trait_head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, n_traits))
        self.use_ckpt, self.use_dsbn, self.Km1 = cfg["grad_checkpoint"], True, n_classes - 1

    def features(self, R, domain):
        x = self.patch(self.views(R))                                       # (n, d, T)
        x = x + self.wenc(self.centers, self.widths).T.unsqueeze(0)
        x = self.dsbn[domain if self.use_dsbn else SRC](x).transpose(1, 2)  # (n, T, d)
        for blk in self.blocks:
            if self.use_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        a = torch.softmax(self.pool_attn(x), dim=1)
        return self.pool_proj(torch.cat([x.mean(1), (a * x).sum(1)], -1))

    def forward(self, R, domain):
        f = self.features(R, domain)
        ev = F.softplus(self.ev_head(f)).view(-1, self.Km1, 2)              # Beta evidence per CORN task
        return {"f": f, "ev": ev}


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c):
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.c * g, None


class OrdinalCDAN(nn.Module):
    """Discriminator on f ⊗ [1, P(y>Healthy), P(y>Transitional)] behind gradient reversal."""
    def __init__(self, dim, n_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * n_classes, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, f, cum, coeff):
        c = torch.cat([torch.ones_like(cum[:, :1]), cum.detach()], 1)
        op = torch.bmm(f.unsqueeze(2), c.unsqueeze(1)).flatten(1)
        return self.net(GradReverse.apply(op, coeff)).squeeze(1)


class Augmenter(nn.Module):
    """Physically motivated pixel-level effects applied on the fly to reflectance:
    sunlit/shaded mixing with bluish diffuse skylight, scene-background mixing, brightness and calibration tilt,
    image-estimated band noise. (Canopy structure itself comes from 4SAIL, not from here.)"""
    def __init__(self, wl, bg, noise_sigma):
        super().__init__()
        self.register_buffer("wl", torch.tensor(wl, dtype=torch.float32))
        self.register_buffer("bg", torch.tensor(bg, dtype=torch.float32))
        self.register_buffer("sig", torch.tensor(noise_sigma, dtype=torch.float32))

    def _u(self, n, lo, hi):
        return torch.empty(n, 1, device=self.wl.device).uniform_(lo, hi)

    @torch.no_grad()
    def forward(self, R, mode):
        n, lam = R.shape[0], self.wl[None, :]
        if mode == "weak":
            out = R * self._u(n, 0.9, 1.1)
            return (out + torch.randn_like(out) * self.sig).clamp_min(1e-4)
        if mode == "canopy":            # source 4SAIL spectra -> UAV-like pixels
            bg_max, bg_prob, sh = 0.40, 0.7, True
        elif mode == "strong":          # target pixels: already canopies, lighter perturbation
            bg_max, bg_prob, sh = 0.10, 0.3, False
        else:
            raise ValueError(mode)
        out = R
        if sh:
            a = self._u(n, 0.5, 2.5)
            kappa = (lam / 550.0) ** (-a); kappa = kappa / kappa.mean(1, keepdim=True)
            f_sun, rho_sh = self._u(n, 0.5, 1.0), self._u(n, 0.05, 0.35)
            out = f_sun * out + (1 - f_sun) * rho_sh * kappa * out
        if self.bg.shape[0] > 0:
            f_bg = self._u(n, 0.0, bg_max) * (torch.rand(n, 1, device=R.device) < bg_prob)
            out = (1 - f_bg) * out + f_bg * self.bg[torch.randint(0, self.bg.shape[0], (n,), device=R.device)]
        out = out * torch.exp(self._u(n, math.log(0.7), math.log(1.3))) * (lam / 650.0) ** self._u(n, -0.15, 0.15)
        return (out + torch.randn_like(out) * self.sig * self._u(n, 0.5, 1.5)).clamp_min(1e-4)

# %%
# ==========================================================
# SECTION 4b — EVIDENTIAL CORN, LOSSES, LABEL-SHIFT TOOLS
# ==========================================================
def ecorn_ab(ev):
    return ev[..., 0] + 1.0, ev[..., 1] + 1.0                       # alpha ("worse than k"), beta ("not worse")


def ecorn_cum(ev):
    a, b = ecorn_ab(ev)
    return torch.cumprod(a / (a + b), dim=1)                        # P(y > k), rank-consistent


def class_probs_from_cum(cum):
    one, zero = torch.ones_like(cum[:, :1]), torch.zeros_like(cum[:, :1])
    return (torch.cat([one, cum], 1) - torch.cat([cum, zero], 1)).clamp_min(0)


def ecorn_vacuity(ev):
    """Task vacuities 2/(a+b), weighted by how relevant each task is: task k matters only if y >= k."""
    a, b = ecorn_ab(ev)
    u = 2.0 / (a + b)
    cum = ecorn_cum(ev)
    w = torch.cat([torch.ones_like(cum[:, :1]), cum[:, :-1]], 1)
    return (w * u).sum(1) / w.sum(1)


def kl_beta_uniform(a, b):
    s = a + b
    return (torch.lgamma(s) - torch.lgamma(a) - torch.lgamma(b)
            + (a - 1) * (torch.digamma(a) - torch.digamma(s)) + (b - 1) * (torch.digamma(b) - torch.digamma(s)))


def ecorn_loss(ev, y, kl_coeff, weight=None, per_sample=False):
    """CORN conditional training sets + evidential (Beta) log loss + KL on misleading evidence."""
    a, b = ecorn_ab(ev)
    w = torch.ones_like(y, dtype=a.dtype) if weight is None else weight
    tot, den = a.new_zeros(len(y)), a.new_zeros(len(y))
    for k in range(ev.shape[1]):
        m = (y >= k).float()
        t = (y > k).float()
        s = a[:, k] + b[:, k]
        nll = t * (torch.digamma(s) - torch.digamma(a[:, k])) + (1 - t) * (torch.digamma(s) - torch.digamma(b[:, k]))
        at, bt = t + (1 - t) * a[:, k], (1 - t) + t * b[:, k]
        tot = tot + m * (nll + kl_coeff * kl_beta_uniform(at, bt))
        den = den + m
    if per_sample:
        return tot
    return (w * tot).sum() / (w * den).sum().clamp_min(1e-6)


def entropy_weight(p):
    ent = -(p * torch.log(p + 1e-8)).sum(1)
    return 1.0 + torch.exp(-ent)


def cdan_loss(d_s, d_t, ws, wt):
    ls = F.binary_cross_entropy_with_logits(d_s, torch.ones_like(d_s), reduction="none")
    lt = F.binary_cross_entropy_with_logits(d_t, torch.zeros_like(d_t), reduction="none")
    return (ws * ls).sum() / ws.sum() + (wt * lt).sum() / wt.sum()


def weighted_coral(fs, ft, ws):
    ws = ws / ws.sum()
    ms = (ws[:, None] * fs).sum(0, keepdim=True)
    cs = (ws[:, None] * (fs - ms)).T @ (fs - ms)
    ct = torch.cov(ft.T)
    d = fs.shape[1]
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def fit_bcts(p, y, iters=200):
    """Bias-corrected temperature scaling on log class-probabilities (Alexandari et al. 2020)."""
    lp = torch.log(torch.as_tensor(p, dtype=torch.float32) + 1e-8)
    yt = torch.as_tensor(y, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True); bias = torch.zeros(lp.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([logT, bias], lr=0.1, max_iter=iters)
    def closure():
        opt.zero_grad()
        l = F.cross_entropy(lp / logT.exp() + bias, yt); l.backward(); return l
    opt.step(closure)
    return float(logT.exp()), bias.detach().numpy()


def apply_bcts(p, T, bias):
    z = np.log(p + 1e-8) / T + bias
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def em_prior(q, src_prior, iters=500, tol=1e-8):
    """Saerens, Latinne & Decaestecker (2002) EM estimate of target class priors from calibrated posteriors."""
    pi = src_prior.copy()
    for _ in range(iters):
        r = q * (pi / src_prior); r /= r.sum(1, keepdims=True)
        new = r.mean(0)
        if np.abs(new - pi).max() < tol:
            break
        pi = new
    return pi


class AdaptiveThresholds:
    """FreeMatch-style self-adaptive, class-wise confidence thresholds (Wang et al. 2023)."""
    def __init__(self, K, m=0.999):
        self.tau, self.p, self.m = 1.0 / K, np.full(K, 1.0 / K), m

    def update(self, probs):
        p = probs.detach().cpu().numpy()
        self.tau = self.m * self.tau + (1 - self.m) * float(p.max(1).mean())
        self.p = self.m * self.p + (1 - self.m) * p.mean(0)

    def get(self):
        return np.clip(self.tau * self.p / self.p.max(), CFG["pl_tau_min"], CFG["pl_tau_max"])


class CrownSampler:
    """Target batches of G crowns × m pixels: every batch covers many crowns, not one big crown."""
    def __init__(self, crown_ids, G, m, seed):
        self.r = np.random.default_rng(seed)
        _, inv = np.unique(crown_ids, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(inv))]
        self.groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
        self.G, self.m = G, m

    def sample(self):
        pick = self.r.choice(len(self.groups), size=min(self.G, len(self.groups)), replace=False)
        return np.concatenate([self.r.choice(self.groups[c], self.m, replace=len(self.groups[c]) < self.m) for c in pick])


@torch.no_grad()
def predict(model, R, domain, batch=None):
    """Numpy reflectance (n, B) -> dict of numpy outputs."""
    model.eval(); batch = batch or CFG["infer_batch"]
    out = {k: [] for k in ("cum", "probs", "vac", "f", "ev")}
    for i in range(0, len(R), batch):
        o = model(torch.as_tensor(R[i:i + batch], dtype=torch.float32, device=DEVICE), domain)
        cum = ecorn_cum(o["ev"])
        out["cum"].append(cum.cpu()); out["probs"].append(class_probs_from_cum(cum).cpu())
        out["vac"].append(ecorn_vacuity(o["ev"]).cpu()); out["f"].append(o["f"].cpu()); out["ev"].append(o["ev"].cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}

# %%
# ==========================================================
# SECTION 4c — SELF-TEST ON RANDOM DATA (run this first)
# ==========================================================
_m = MangroveBridgeNet(WL, CFG, K).to(DEVICE)
_d = OrdinalCDAN(CFG["dim"], K).to(DEVICE)
_aug = Augmenter(WL, BG, NOISE_SIGMA).to(DEVICE)
_R = torch.rand(16, B, device=DEVICE) * 0.4 + 0.02
_m.train()
_o = _m(_aug(_R, "canopy"), SRC); _ot = _m(_aug(_R, "strong"), TGT)
assert _o["f"].shape == (16, CFG["dim"]) and _o["ev"].shape == (16, K - 1, 2)
_cum = ecorn_cum(_o["ev"]); _p = class_probs_from_cum(_cum)
assert torch.allclose(_p.sum(1), torch.ones(16, device=DEVICE), atol=1e-5)
assert torch.all(_cum[:, 1] <= _cum[:, 0] + 1e-6), "rank consistency"
_y = torch.randint(0, K, (16,), device=DEVICE)
_l = (ecorn_loss(_o["ev"], _y, 1.0) + _m.trait_head(_o["f"]).pow(2).mean()
      + cdan_loss(_d(_o["f"], _cum, 1.0), _d(_ot["f"], ecorn_cum(_ot["ev"]), 1.0), entropy_weight(_p), entropy_weight(_p))
      + weighted_coral(_o["f"], _ot["f"], torch.ones(16, device=DEVICE)))
_l.backward()
_v = _m.views.raw(_R[:4]); _v2 = _m.views.raw(_R[:4] * 0.37)
assert torch.allclose(_v, _v2, atol=1e-4), "views must be brightness-invariant"
_np = sum(p.numel() for p in _m.parameters()); _nd = sum(p.numel() for p in _d.parameters())
print(f"Self-test passed. Parameters: network {_np/1e6:.3f} M + discriminator {_nd/1e6:.3f} M")
del _m, _d, _aug, _R, _o, _ot, _cum, _p, _y, _l, _v, _v2

# %%
# ==========================================================
# SECTION 5a — DATA TENSORS AND SAMPLERS
# ==========================================================
TRAIT_USE = [TRAIT_NAMES.index(t) for t in ["Cab", "Car", "Ant", "N"]]

src_R = np.vstack([R_leaf, sim_R]) if len(sim_R) else R_leaf.copy()
src_y = np.concatenate([y_leaf, y_leaf[sim_leaf]]) if len(sim_R) else y_leaf.copy()
src_is_sim = np.r_[np.zeros(len(R_leaf), bool), np.ones(len(sim_R), bool)]
src_val = np.r_[leaf_val, leaf_val[sim_leaf]] if len(sim_R) else leaf_val.copy()
src_tr_raw = np.vstack([TRAITS[:, TRAIT_USE], TRAITS[sim_leaf][:, TRAIT_USE]]) if len(sim_R) else TRAITS[:, TRAIT_USE]
TR_MU, TR_SD = src_tr_raw[~src_val].mean(0), src_tr_raw[~src_val].std(0) + 1e-6
src_trait = (src_tr_raw - TR_MU) / TR_SD
src_lai = np.r_[np.full(len(R_leaf), np.nan), sim_meta["lai"].to_numpy()] if len(sim_R) else np.full(len(R_leaf), np.nan)

train_ok = ~src_val
if CFG["lso_mode"] == "LAI":
    train_ok &= ~src_is_sim | (src_lai < 4.0)
cls_leaf = [np.flatnonzero(train_ok & ~src_is_sim & (src_y == k)) for k in range(K)]
cls_sim = [np.flatnonzero(train_ok & src_is_sim & (src_y == k)) for k in range(K)]
SRC_PRIOR = np.full(K, 1.0 / K)                          # the source batch is class-balanced by construction

Rs_t = torch.tensor(src_R, dtype=torch.float32, device=DEVICE)
ys_t = torch.tensor(src_y, dtype=torch.long, device=DEVICE)
trs_t = torch.tensor(src_trait, dtype=torch.float32, device=DEVICE)
sim_t = torch.tensor(src_is_sim, device=DEVICE)
Rt_np = R_tgt.astype(np.float32)
sampler = CrownSampler(tgt_crown, CFG["tgt_crowns"], CFG["tgt_per_crown"], CFG["seed"])
monitor = rng.choice(len(R_tgt), min(20000, len(R_tgt)), replace=False)
val_idx = np.flatnonzero(src_val)


def sample_source(n, r=None):
    """Class-balanced source batch: leaves + 4SAIL canopies (leaves only when ABLATION['physics_bridge'] is False)."""
    r = rng if r is None else r
    idx = []
    for k in range(K):
        nk = n // K
        use_sim = ABLATION["physics_bridge"] and len(cls_sim[k]) > 0
        n_leaf = int(round(nk * CFG["leaf_frac"])) if use_sim else nk
        idx.append(r.choice(cls_leaf[k], n_leaf, replace=True))
        if nk - n_leaf > 0:
            idx.append(r.choice(cls_sim[k], nk - n_leaf, replace=True))
    return np.concatenate(idx)


def source_val_idx():
    """Held-out source samples of the source domain actually trained on (leaves only without the physics bridge)."""
    return val_idx if ABLATION["physics_bridge"] else val_idx[~src_is_sim[val_idx]]


def uses_target():
    """False for SourceOnly: no component of the loss sees UAV pixels."""
    return ABLATION["adversarial"] or ABLATION["coral"] or ABLATION["self_training"]


print(f"Source train: {train_ok.sum():,} (leaves {sum(len(c) for c in cls_leaf)}, 4SAIL {sum(len(c) for c in cls_sim):,}) | "
      f"source val: {len(val_idx):,} | target: {len(R_tgt):,} pixels / {len(sampler.groups)} crowns")

# %%
# ==========================================================
# SECTION 5b — TRAINING (one function; runs are managed in Section 6c)
# ==========================================================
def ramp(p):
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def train_one(seed, verbose=True):
    """Train MangroveBridge-UDA with one training seed. Data (leaf split, 4SAIL library, UAV sample) are fixed by
    CFG['seed']; the seed changes initialisation, batches and augmentation. Field crowns are never seen (Section 2)."""
    set_seed(seed)
    r = np.random.default_rng(seed)
    model = MangroveBridgeNet(WL, CFG, K).to(DEVICE)
    model.use_dsbn = ABLATION["dsbn"]
    aug = Augmenter(WL, BG, NOISE_SIGMA).to(DEVICE)
    disc = OrdinalCDAN(CFG["dim"], K).to(DEVICE)
    sampler = CrownSampler(tgt_crown, CFG["tgt_crowns"], CFG["tgt_per_crown"], seed)

    with torch.no_grad():                    # two-domain standardisation of the views
        n_fit = min(5000, len(R_tgt))
        s_fit = Rs_t[torch.as_tensor(sample_source(n_fit, r), device=DEVICE)]
        s_fit = aug(s_fit, "canopy") if ABLATION["augmentation"] else s_fit
        if uses_target():
            t_fit = torch.tensor(Rt_np[r.choice(len(Rt_np), n_fit, replace=False)], device=DEVICE)
            model.views.fit(torch.cat([s_fit, t_fit]))
        else:                                # SourceOnly: standardisation from source spectra only
            model.views.fit(s_fit)

    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(list(model.parameters()) + list(disc.parameters()), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    total = CFG["epochs"] * CFG["iters_per_epoch"]
    warm = max(1, min(300, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    thr = AdaptiveThresholds(K)
    pi_t = SRC_PRIOR.copy()
    prior_frozen = False
    history, checkpoints, step = [], [], 0
    t0 = time.time()

    for epoch in range(CFG["epochs"]):
        model.train(); disc.train()
        kl = min(1.0, epoch / max(1, CFG["kl_anneal_epochs"]))
        self_train = ABLATION["self_training"] and epoch >= CFG["warmup_epochs"]
        w_cls = torch.tensor(pi_t / SRC_PRIOR if ABLATION["label_shift"] else np.ones(K), dtype=torch.float32, device=DEVICE)
        acc = dict(src=0., trait=0., adv=0., coral=0., pl=0., mask=0.)
        for _ in range(CFG["iters_per_epoch"]):
            coeff = CFG["w_adv"] * ramp(step / total)

            # ---- source ----
            bi = torch.as_tensor(sample_source(CFG["src_batch"], r), device=DEVICE)
            xs, ys = Rs_t[bi], ys_t[bi]
            if ABLATION["augmentation"]:
                xs = torch.where(sim_t[bi].unsqueeze(1), aug(xs, "canopy"), aug(xs, "weak"))
            out_s = model(xs, SRC)
            loss_src = ecorn_loss(out_s["ev"], ys, kl)
            loss = loss_src
            loss_trait = torch.zeros((), device=DEVICE)
            if ABLATION["trait_head"]:
                loss_trait = F.mse_loss(model.trait_head(out_s["f"]), trs_t[bi])
                loss = loss + CFG["w_trait"] * loss_trait

            # ---- target (strong view; DSBN target branch); skipped when nothing uses it (SourceOnly) ----
            need_t = ABLATION["adversarial"] or ABLATION["coral"] or self_train
            if need_t:
                ti = sampler.sample()
                xt = torch.tensor(Rt_np[ti], device=DEVICE)
                out_t = model(aug(xt, "strong") if ABLATION["augmentation"] else xt, TGT)
                cum_t = ecorn_cum(out_t["ev"])
            cum_s = ecorn_cum(out_s["ev"])
            ws = w_cls[ys]                                                   # importance weights (label shift)

            loss_adv = torch.zeros((), device=DEVICE)
            if ABLATION["adversarial"]:
                ps, pt = class_probs_from_cum(cum_s).detach(), class_probs_from_cum(cum_t).detach()
                loss_adv = cdan_loss(disc(out_s["f"], cum_s, coeff), disc(out_t["f"], cum_t, coeff),
                                     entropy_weight(ps) * ws, entropy_weight(pt))
                loss = loss + loss_adv
            loss_coral = torch.zeros((), device=DEVICE)
            if ABLATION["coral"]:
                loss_coral = weighted_coral(out_s["f"], out_t["f"], ws)
                loss = loss + CFG["w_coral"] * loss_coral

            # ---- mean-teacher pseudo-labels (class-balanced; no prior multiplication) ----
            loss_pl, mrate = torch.zeros((), device=DEVICE), 0.0
            if self_train:
                with torch.no_grad():
                    oa, ob = teacher(aug(xt, "weak"), TGT), teacher(aug(xt, "strong"), TGT)
                    pa = class_probs_from_cum(ecorn_cum(oa["ev"]))
                    pb = class_probs_from_cum(ecorn_cum(ob["ev"]))
                    pm = 0.5 * (pa + pb)
                    conf, pl = pm.max(1)
                    thr.update(pm)
                    tau = torch.tensor(thr.get(), dtype=torch.float32, device=DEVICE)
                    vac = 0.5 * (ecorn_vacuity(oa["ev"]) + ecorn_vacuity(ob["ev"]))
                    mask = (conf >= tau[pl]) & (pa.argmax(1) == pb.argmax(1)) & (vac <= CFG["pl_max_vacuity"])
                mrate = float(mask.float().mean())
                if mask.any():
                    w_pl = mask.float() * conf
                    if CFG["pl_class_balance"]:
                        cnt = torch.bincount(pl[mask], minlength=K).float()
                        cw = torch.where(cnt > 0, cnt.sum() / ((cnt > 0).sum() * cnt.clamp_min(1)), torch.zeros_like(cnt))
                        w_pl = w_pl * cw[pl]
                    loss_pl = ecorn_loss(out_t["ev"], pl, kl, weight=w_pl)
                    loss = loss + CFG["w_pl"] * loss_pl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(disc.parameters()), CFG["grad_clip"])
            opt.step(); sched.step(); step += 1
            with torch.no_grad():
                for pt_, ps_ in zip(teacher.parameters(), model.parameters()):
                    pt_.mul_(CFG["ema"]).add_(ps_.detach(), alpha=1 - CFG["ema"])
                for bt, bs in zip(teacher.buffers(), model.buffers()):
                    bt.copy_(bs)
            for k_, v_ in (("src", loss_src), ("trait", loss_trait), ("adv", loss_adv), ("coral", loss_coral), ("pl", loss_pl)):
                acc[k_] += float(v_) / CFG["iters_per_epoch"]
            acc["mask"] += mrate / CFG["iters_per_epoch"]

        # ---- epoch end: label-free diagnostics only (no field data), calibrated label-shift estimate ----
        teacher.use_dsbn = model.use_dsbn
        pv = predict(teacher, src_R[val_idx], SRC)
        yv = src_y[val_idx]; sv = src_is_sim[val_idx]                       # diagnostics: leaves and 4SAIL separately
        lab_v = (pv["cum"] > 0.5).sum(1)
        f1_leaf = f1_score(yv[~sv], lab_v[~sv], average="macro")
        f1_sim = f1_score(yv[sv], lab_v[sv], average="macro") if sv.any() else float("nan")
        pm_t = predict(teacher, R_tgt[monitor], TGT)
        if ABLATION["label_shift"] and epoch + 1 >= CFG["warmup_epochs"] and not prior_frozen:
            _m = np.ones(len(yv), bool) if ABLATION["physics_bridge"] else ~sv     # calibrate on the trained source domain
            Tc, bc = fit_bcts(pv["probs"][_m], yv[_m])
            q_t = apply_bcts(pm_t["probs"], Tc, bc)
            val_prior = np.maximum(np.bincount(yv[_m], minlength=K) / _m.sum(), 1e-6); val_prior /= val_prior.sum()
            est = em_prior(q_t, val_prior)
            est = np.maximum(est, CFG["prior_floor"]); est /= est.sum()
            if CFG["prior_freeze"]:
                pi_t, prior_frozen = est, True
                if verbose:
                    print("UAV class mix estimated and frozen: " + ", ".join(f"{n} {v:.2f}" for n, v in zip(CLASS_NAMES, pi_t)))
            else:
                pi_t = CFG["prior_momentum"] * pi_t + (1 - CFG["prior_momentum"]) * est
        frac = np.bincount((pm_t["cum"] > 0.5).sum(1), minlength=K) / len(monitor)
        row = dict(epoch=epoch + 1, **{f"loss_{k}": v for k, v in acc.items() if k != "mask"}, pl_mask=acc["mask"],
                   f1_leaf=f1_leaf, f1_sim=f1_sim, tgt_vacuity=float(pm_t["vac"].mean()),
                   **{f"tgt_frac_{n}": float(v) for n, v in zip(CLASS_NAMES, frac)},
                   **{f"prior_{n}": float(v) for n, v in zip(CLASS_NAMES, pi_t)})
        history.append(row)
        if epoch + 1 > CFG["warmup_epochs"] and frac.min() < 0.02:
            warnings.warn(f"Epoch {epoch+1}: UAV share of {CLASS_NAMES[int(frac.argmin())]} fell below 2 %.")
        if verbose:
            print(f"[seed {seed}] ep {epoch+1:03d} | src {acc['src']:.3f} adv {acc['adv']:.3f} coral {acc['coral']:.4f} "
                  f"pl {acc['pl']:.3f} mask {acc['mask']:.2f} | F1 leaf {f1_leaf:.3f} sim {f1_sim:.3f} | UAV "
                  + "/".join(f"{v:.2f}" for v in frac) + f" | vac {row['tgt_vacuity']:.2f} | {time.time()-t0:.0f}s")
        if (epoch + 1) % CFG["ckpt_every"] == 0 or epoch + 1 == CFG["epochs"]:
            checkpoints.append(dict(epoch=epoch + 1, state=copy.deepcopy(teacher.state_dict()), pi_t=pi_t.copy()))
    return dict(teacher=teacher, checkpoints=checkpoints, history=pd.DataFrame(history), minutes=(time.time() - t0) / 60)


def plot_history(hist, path, show=True):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    hist.plot(x="epoch", y=["f1_leaf", "f1_sim"], ax=ax[0]); ax[0].set_title("Macro-F1 (held-out leaf groups / their 4SAIL canopies)")
    hist.plot(x="epoch", y=[f"tgt_frac_{n}" for n in CLASS_NAMES], ax=ax[1]); ax[1].axvline(CFG["warmup_epochs"], ls=":", c="0.5")
    ax[1].set_title("Predicted UAV class shares")
    hist.plot(x="epoch", y=[f"prior_{n}" for n in CLASS_NAMES], ax=ax[2]); ax[2].set_title("Estimated UAV class prior π_t")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    plt.show() if show else plt.close(fig)

# %%
# ==========================================================
# SECTION 6a — DEV CHECKPOINT SELECTION (function)
# ==========================================================
_r6 = np.random.default_rng(CFG["seed"] + 6)
dev_src = _r6.choice(np.flatnonzero(train_ok), min(10000, int(train_ok.sum())), replace=False)
dev_tgt = _r6.choice(len(R_tgt), min(10000, len(R_tgt)), replace=False)        # field crowns are not in R_tgt


def dev_risk(teacher, state):
    """Deep Embedded Validation risk of one checkpoint (label-free on the UAV side; lower is better)."""
    teacher.load_state_dict(state)
    ds = dev_src if ABLATION["physics_bridge"] else dev_src[~src_is_sim[dev_src]]
    vi = source_val_idx()
    fs = predict(teacher, src_R[ds], SRC)["f"]
    ft = predict(teacher, R_tgt[dev_tgt], TGT)["f"]
    ov = predict(teacher, src_R[vi], SRC)
    clf = LogisticRegression(max_iter=2000).fit(np.vstack([fs, ft]), np.r_[np.zeros(len(fs)), np.ones(len(ft))])
    p = np.clip(clf.predict_proba(ov["f"])[:, 1], 1e-4, 1 - 1e-4)
    w = p / (1 - p) * len(fs) / len(ft)
    L = ecorn_loss(torch.tensor(ov["ev"]), torch.tensor(src_y[vi]), 0.0, per_sample=True).numpy()
    wl = w * L
    eta = -np.cov(wl, w)[0, 1] / max(np.var(w, ddof=1), 1e-12)
    return float(wl.mean() + eta * w.mean() - eta)


def select_checkpoint(teacher, checkpoints, verbose=True):
    scores = [(c["epoch"], dev_risk(teacher, c["state"])) for c in checkpoints]
    best = int(np.argmin([s for _, s in scores]))
    teacher.load_state_dict(checkpoints[best]["state"])
    if verbose:
        print("DEV risk: " + ", ".join(f"ep{e} {s:.3f}" for e, s in scores) + f" -> selected epoch {checkpoints[best]['epoch']}")
    dev = pd.DataFrame(dict(epoch=[e for e, _ in scores], dev_risk=[s for _, s in scores],
                            selected=[i == best for i in range(len(scores))]))
    return dict(epoch=int(checkpoints[best]["epoch"]), pi_t=checkpoints[best]["pi_t"], dev=dev,
                dev_risk=float(scores[best][1]))

# %%
# ==========================================================
# SECTION 6b — REJECT THRESHOLDS AND LABEL-FREE VALIDATION (function)
# ==========================================================
def maha_dist(f, mu, prec):
    return np.sqrt(np.min(np.stack([np.einsum("ij,jk,ik->i", f - mu[k], prec, f - mu[k]) for k in range(K)], 1), 1))


def calibrate(teacher, verbose=True):
    """Reject thresholds from held-out SOURCE data only, vacuity sanity check, held-out leaf / 4SAIL accuracy."""
    r = np.random.default_rng(CFG["seed"] + 7)
    ov = predict(teacher, src_R[val_idx], SRC)
    thr_m = np.isin(val_idx, source_val_idx())                     # thresholds from the trained source domain only
    vac_thr = float(np.percentile(ov["vac"][thr_m], CFG["vac_percentile"]))
    pool = np.flatnonzero(train_ok if ABLATION["physics_bridge"] else train_ok & ~src_is_sim)
    fit_i = r.choice(pool, min(20000, len(pool)), replace=False)
    f_fit = predict(teacher, src_R[fit_i], SRC)["f"]; y_fit = src_y[fit_i]
    mu = np.stack([f_fit[y_fit == k].mean(0) for k in range(K)])
    prec = np.linalg.inv(np.cov((f_fit - mu[y_fit]).T) + 1e-3 * np.eye(f_fit.shape[1]))
    maha_thr = float(np.percentile(maha_dist(ov["f"][thr_m], mu, prec), CFG["maha_percentile"]))
    om = predict(teacher, R_tgt[monitor], TGT)
    ok_v = om["vac"]
    lowconf = float(np.mean((om["vac"] > vac_thr) | (maha_dist(om["f"], mu, prec) > maha_thr)))
    uav_frac = np.bincount((om["cum"] > 0.5).sum(1), minlength=K) / len(monitor)
    scr_v = predict(teacher, R_screened, TGT)["vac"] if len(R_screened) else np.array([])
    auc = roc_auc_score(np.r_[np.zeros(len(ok_v)), np.ones(len(scr_v))], np.r_[ok_v, scr_v]) if len(scr_v) else np.nan
    yv, sv = src_y[val_idx], src_is_sim[val_idx]
    lab_v = (ov["cum"] > 0.5).sum(1)
    out = dict(vac_thr=vac_thr, mu=mu, prec=prec, maha_thr=maha_thr, ov=ov, ok_v=ok_v, scr_v=scr_v, auroc=auc,
               lowconf_frac=lowconf, uav_frac=uav_frac,
               leaf_macro_f1=f1_score(yv[~sv], lab_v[~sv], average="macro"),
               sim_macro_f1=f1_score(yv[sv], lab_v[sv], average="macro") if sv.any() else np.nan)
    if verbose:
        print(f"Reject thresholds: vacuity > {vac_thr:.3f} or Mahalanobis > {maha_thr:.2f} | vacuity AUROC "
              f"(screened-out vs sunlit) {auc:.2f} | held-out leaf F1 {out['leaf_macro_f1']:.3f}, 4SAIL F1 {out['sim_macro_f1']:.3f}")
    return out


def print_heldout_report(cal):
    ov = cal["ov"]; yv, sv = src_y[val_idx], src_is_sim[val_idx]; lab_v = (ov["cum"] > 0.5).sum(1)
    print("\n=== Held-out LEAF groups ===")
    print(confusion_matrix(yv[~sv], lab_v[~sv], labels=range(K)))
    print(classification_report(yv[~sv], lab_v[~sv], labels=range(K), target_names=CLASS_NAMES, zero_division=0))
    if sv.any():
        print("=== 4SAIL canopies of held-out leaves ===")
        print(classification_report(yv[sv], lab_v[sv], labels=range(K), target_names=CLASS_NAMES, zero_division=0))

# %%
# ==========================================================
# SECTION 6c — RUN MANAGER, CROWN RULE, LABEL-FREE REPORTS (functions)
# ==========================================================
import gc
import hashlib
from contextlib import contextmanager
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, recall_score, precision_score

for _k, _v in dict(crown_min_px=10).items():
    CFG.setdefault(_k, _v)
if not CROWN_INFO.index.is_unique:      # crowns from an older Section 2 still in memory
    raise RuntimeError("Crown uids repeat across plots: this kernel still holds crowns from an older Section 2. "
                       "Restart the kernel and run this notebook from Section 0 (Section 2 must contain 'next_uid').")

LEVELS = ["confident sunlit pixels", "all sunlit pixels", "sunlit + shaded leaf pixels", "all valid crown pixels"]


def crown_fusion(ev_lv, n_lv, ids, vac_thr):
    """Crown rule of the maps: Beta evidence averaged over the first pixel set with >= crown_min_px pixels
    (confident sunlit -> sunlit -> sunlit + shaded leaf -> all valid), else the largest set."""
    L = ev_lv.shape[0]
    cnt = n_lv[:, ids]
    level = np.full(len(ids), -1)
    for lv in range(L):
        level[(level < 0) & (cnt[lv] >= CFG["crown_min_px"])] = lv
    rest = np.flatnonzero(level < 0)
    if len(rest):
        has = cnt[:, rest].max(0) > 0
        level[rest[has]] = cnt[:, rest[has]].argmax(0)
    nodata = level < 0
    lv_safe = np.where(nodata, 0, level)
    n_used = np.where(nodata, 0, cnt[lv_safe, np.arange(len(ids))])
    ev = ev_lv[lv_safe, ids] / np.maximum(n_used, 1)[:, None, None]
    a, b = 1.0 + ev[..., 0], 1.0 + ev[..., 1]
    cum = np.cumprod(a / (a + b), 1)
    u = 2.0 / (a + b); wgt = np.c_[np.ones(len(ids)), cum[:, :-1]]
    vac = (wgt * u).sum(1) / wgt.sum(1)
    cls_c = (cum > 0.5).sum(1) + 1
    cls_c[nodata] = 0
    reliability = np.where(nodata, "no data", np.where((level == 0) & (vac <= vac_thr), "high",
                                                        np.where(level <= 1, "medium", "low")))
    return dict(level=level, nodata=nodata, n_used=n_used, cum=cum, vac=vac, cls=cls_c, reliability=reliability)


def class_metrics(y, p):
    """OA, balanced accuracy, macro-F1, kappa, QWK, MAE and per-class PA / UA / F1 (+ confusion matrix)."""
    lab = list(range(K))
    cm = confusion_matrix(y, p, labels=lab)
    pa = recall_score(y, p, labels=lab, average=None, zero_division=0)
    ua = precision_score(y, p, labels=lab, average=None, zero_division=0)
    f1 = f1_score(y, p, labels=lab, average=None, zero_division=0)
    m = dict(n=len(y), OA=float(np.mean(y == p)), balanced_acc=balanced_accuracy_score(y, p),
             macro_F1=f1_score(y, p, labels=lab, average="macro", zero_division=0),
             kappa=cohen_kappa_score(y, p, labels=lab), QWK=cohen_kappa_score(y, p, labels=lab, weights="quadratic"),
             MAE=float(np.mean(np.abs(y - p))))
    for k, n in enumerate(CLASS_NAMES):
        m[f"PA_{n}"], m[f"UA_{n}"], m[f"F1_{n}"] = pa[k], ua[k], f1[k]
    return m, cm


def cm_table(cm, title=None):
    """Confusion matrix (counts) with reference (row) and predicted (column) totals."""
    t = pd.DataFrame(cm, index=[f"ref {n}" for n in CLASS_NAMES], columns=[f"pred {n}" for n in CLASS_NAMES])
    t["n (reference)"] = t.sum(1)
    t.loc["n (predicted)"] = t.sum(0)
    if title:
        print(f"\n{title}\n{t.to_string()}")
    return t


def mean_sd_text(df, cols):
    return {c: (f"{df[c].mean():.3f} ± {df[c].std(ddof=1):.3f}" if len(df) > 1 else f"{df[c].mean():.3f}") for c in cols}


@contextmanager
def overrides(cfg=None, abl=None):
    """Temporarily change CFG / ABLATION entries (one tuning setting or one ablation variant)."""
    cfg, abl = dict(cfg or {}), dict(abl or {})
    old_c = {k: CFG.get(k, "__missing__") for k in cfg}; old_a = {k: ABLATION[k] for k in abl}
    CFG.update(cfg); ABLATION.update(abl)
    try:
        yield
    finally:
        for k, v in old_c.items():
            if v == "__missing__":
                CFG.pop(k, None)
            else:
                CFG[k] = v
        ABLATION.update(old_a)


def run_tag(cfg=None, abl_name=None):
    """Folder name of a setting: 'default', 'lr0.001', 'w_coral0.25', ... or 'abl_<variant>'."""
    if abl_name:
        return f"abl_{abl_name}"
    if not cfg:
        return "default"
    return "_".join(f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}" for k, v in sorted(cfg.items()))


_INFER_KEYS = {"crown_min_px", "infer_batch", "vac_percentile", "maha_percentile", "rows_per_strip"}


def run_signature():
    """Data + effective training settings + ablation switches (inside an `overrides` block)."""
    h = hashlib.sha1()
    h.update(CROWN_INFO.reset_index()[["uid", "plot", "crown_id"]].to_csv(index=False).encode())
    h.update(np.asarray(sorted(field.index) if (field is not None and FIELD_HOLDOUT) else [], np.int64).tobytes())
    h.update(np.asarray(band_idx, np.int64).tobytes())
    h.update(json.dumps({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                         for k, v in sorted(CFG.items()) if k not in _INFER_KEYS}, sort_keys=True).encode())
    h.update(json.dumps(ABLATION, sort_keys=True).encode())
    return h.hexdigest()


RUNS_DIR = OUT_DIR / "runs"; RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_dir(tag, seed):
    return RUNS_DIR / tag / f"seed_{seed}"


def build_model():
    m = MangroveBridgeNet(WL, CFG, K).to(DEVICE)
    m.use_dsbn = ABLATION["dsbn"]
    return m


def label_free_report(teacher, cal, sel, d, seed, minutes):
    """Held-out leaf and 4SAIL accuracy, DEV risk, uncertainty checks and UAV class shares of one run -> files in d."""
    ov = cal["ov"]; yv, sv = src_y[val_idx], src_is_sim[val_idx]; lab = (ov["cum"] > 0.5).sum(1)
    m_leaf, cm_leaf = class_metrics(yv[~sv], lab[~sv])
    row = dict(seed=seed, selected_epoch=sel["epoch"], dev_risk=sel["dev_risk"], train_minutes=minutes,
               **{f"leaf_{k}": v for k, v in m_leaf.items()})
    cm_table(cm_leaf).to_csv(d / "heldout_confusion_leaf.csv")
    pd.DataFrame(classification_report(yv[~sv], lab[~sv], labels=range(K), target_names=CLASS_NAMES,
                                       zero_division=0, output_dict=True)).T.to_csv(d / "heldout_report_leaf.csv")
    if sv.any():
        m_sim, cm_sim = class_metrics(yv[sv], lab[sv])
        row.update({f"sim_{k}": v for k, v in m_sim.items()})
        cm_table(cm_sim).to_csv(d / "heldout_confusion_4sail.csv")
    row.update(vacuity_AUROC=cal["auroc"], vac_thr=cal["vac_thr"], maha_thr=cal["maha_thr"],
               uav_lowconf_frac=cal["lowconf_frac"],
               **{f"uav_frac_{n}": float(v) for n, v in zip(CLASS_NAMES, cal["uav_frac"])},
               **{f"pi_t_{n}": float(v) for n, v in zip(CLASS_NAMES, np.asarray(sel["pi_t"]))})
    pd.DataFrame([row]).to_csv(d / "label_free_metrics.csv", index=False)
    return row


def load_run(tag, seed, cfg=None, abl=None):
    """Trained model of one run if it exists and matches the current data/settings, else None (no training)."""
    d = run_dir(tag, seed); f = d / "teacher.pt"
    with overrides(cfg, abl):
        if not f.exists():
            return None
        ck = torch.load(f, map_location=DEVICE, weights_only=False)
        if ck.get("run_sig") != run_signature():
            return None
        t = build_model(); t.load_state_dict(ck["state"]); t.eval()
        cal = calibrate(t, verbose=False)
    return dict(teacher=t, cal=cal, ck=ck, dir=d)


def run_or_load(tag, seed, cfg=None, abl=None, verbose=False):
    """Train (or reload) one run = one setting x one seed, pick its checkpoint by DEV, write its label-free results."""
    d = run_dir(tag, seed); d.mkdir(parents=True, exist_ok=True)
    with overrides(cfg, abl):
        sig = run_signature()
        ck = torch.load(d / "teacher.pt", map_location=DEVICE, weights_only=False) \
            if (RESUME_SEEDS and (d / "teacher.pt").exists()) else None
        if ck is not None and ck.get("run_sig") != sig:
            print(f"  {tag} / seed {seed}: saved model used other data or settings -> retraining")
            ck = None
        if ck is not None:
            teacher = build_model(); teacher.load_state_dict(ck["state"]); teacher.eval()
            sel = dict(epoch=ck["selected_epoch"], pi_t=np.asarray(ck["pi_t"]), dev=pd.read_csv(d / "dev_scores.csv"),
                       dev_risk=ck["dev_risk"])
            hist, minutes = pd.read_csv(d / "training_history.csv"), ck.get("minutes", np.nan)
            print(f"  {tag} / seed {seed}: loaded (selected epoch {sel['epoch']}, DEV {sel['dev_risk']:.4f})", flush=True)
        else:
            print(f"  {tag} / seed {seed}: training ...", flush=True)
            run = train_one(seed, verbose=verbose)
            teacher, hist, minutes = run["teacher"], run["history"], run["minutes"]
            sel = select_checkpoint(teacher, run["checkpoints"], verbose=verbose)
            hist.to_csv(d / "training_history.csv", index=False); sel["dev"].to_csv(d / "dev_scores.csv", index=False)
            torch.save(dict(state=teacher.state_dict(), selected_epoch=sel["epoch"], pi_t=np.asarray(sel["pi_t"]).tolist(),
                            dev_risk=sel["dev_risk"], minutes=minutes, run_sig=sig, cfg_over=dict(cfg or {}),
                            abl_over=dict(abl or {})), d / "teacher.pt")
            plot_history(hist, d / "training_diagnostics.png", show=False)
            del run
            print(f"  {tag} / seed {seed}: done in {minutes:.0f} min (selected epoch {sel['epoch']}, "
                  f"DEV {sel['dev_risk']:.4f})", flush=True)
        cal = calibrate(teacher, verbose=False)
        row = label_free_report(teacher, cal, sel, d, seed, minutes)
    row.update(tag=tag)
    return dict(teacher=teacher, cal=cal, sel=sel, hist=hist, row=row, dir=d)


def free(res):
    """Release a run's model from memory (results stay on disk)."""
    res.pop("teacher", None); gc.collect()
    if str(DEVICE).startswith("cuda"):
        torch.cuda.empty_cache()


if CFG_BASE is None:
    CFG_BASE = dict(CFG)                       # settings of Sections 0-1, before the winning setting is applied
print("Run manager ready. Runs are stored in", RUNS_DIR)

# %%
# ==========================================================
# SECTION 6d — PARAMETER OPTIMISATION (label-free, DEV risk)
# ==========================================================
TUNE_DIR = OUT_DIR / "tuning"; TUNE_DIR.mkdir(exist_ok=True)
BEST_FILE = TUNE_DIR / "best_setting.json"
SHOW = ["tag", "seed", "selected_epoch", "dev_risk", "leaf_macro_F1", "sim_macro_F1", "vacuity_AUROC", "uav_lowconf_frac"]

CFG.clear(); CFG.update(CFG_BASE)               # start from the Section 0 settings (also when this cell is re-run)
if RUN_TUNING:
    cands = [dict()] + [{k: v} for k, vals in TUNE_GRID.items() for v in vals if v != CFG.get(k)]
    print(f"Stage 1: {len(cands)} settings x seed {TUNE_SEED} (each 60 x 100 iterations)")
    s1 = []
    for c in cands:
        r_ = run_or_load(run_tag(c), TUNE_SEED, c)
        s1.append(dict(r_["row"], setting=json.dumps(c))); free(r_)
    s1 = pd.DataFrame(s1)
    leaf0 = float(s1.loc[s1["tag"] == "default", "leaf_macro_F1"].iloc[0])
    s1["eligible"] = s1["leaf_macro_F1"] >= leaf0 - TUNE_LEAF_F1_TOL
    s1 = s1.sort_values(["eligible", "dev_risk"], ascending=[False, True]).reset_index(drop=True)
    s1["rank"] = np.arange(1, len(s1) + 1)
    s1.to_csv(TUNE_DIR / "stage1_one_at_a_time.csv", index=False)
    print("\nStage 1 (ranked by DEV risk; eligible = leaf macro-F1 within "
          f"{TUNE_LEAF_F1_TOL} of the default {leaf0:.3f}):\n" + s1[["rank", "eligible"] + SHOW].round(4).to_string(index=False))

    top = s1[s1["eligible"]].head(TUNE_TOP)
    print(f"\nStage 2: {len(top)} best settings x seeds {SEEDS}")
    s2 = []
    for _, t in top.iterrows():
        c = json.loads(t["setting"])
        for seed in SEEDS:
            r_ = run_or_load(t["tag"], seed, c)
            s2.append(dict(r_["row"], setting=t["setting"])); free(r_)
    s2 = pd.DataFrame(s2)
    s2.to_csv(TUNE_DIR / "stage2_runs.csv", index=False)
    agg = s2.groupby(["tag", "setting"]).agg(dev_mean=("dev_risk", "mean"), dev_sd=("dev_risk", "std"),
                                             leaf_F1_mean=("leaf_macro_F1", "mean"), leaf_F1_sd=("leaf_macro_F1", "std"),
                                             sim_F1_mean=("sim_macro_F1", "mean"), n_seeds=("seed", "nunique"))
    agg = agg.reset_index()
    agg["_default"] = agg["tag"] != "default"                # tie on DEV risk -> keep the default setting
    agg = agg.sort_values(["dev_mean", "_default"], kind="mergesort").drop(columns="_default").reset_index(drop=True)
    agg.to_csv(TUNE_DIR / "stage2_summary.csv", index=False)
    print("\nStage 2 (mean over seeds, lowest DEV risk wins):\n" + agg.round(4).to_string(index=False))
    best = agg.iloc[0]
    BEST_SETTING = dict(tag=best["tag"], cfg=json.loads(best["setting"]))
    BEST_FILE.write_text(json.dumps(BEST_SETTING, indent=2))
elif BEST_FILE.exists():
    BEST_SETTING = json.loads(BEST_FILE.read_text())
    print("Parameter optimisation skipped: using the saved best setting", BEST_SETTING)
else:
    BEST_SETTING = dict(tag="default", cfg={})
    print("Parameter optimisation skipped and no saved result: the default setting is used.")
print(f"\nFinal setting: {BEST_SETTING['tag']} {BEST_SETTING['cfg']}  ->  {BEST_FILE}")

# %%
# ==========================================================
# SECTION 6e — FINAL MODEL: EVERY SEED, MEAN ± SD, BEST (DEV) SEED
# ==========================================================
FINAL_TAG, FINAL_CFG = BEST_SETTING["tag"], dict(BEST_SETTING["cfg"])
CFG.update(FINAL_CFG)                          # the winning setting is now the model's setting (Sections 7-9)
SEED_DIR = RUNS_DIR / FINAL_TAG                # OUT_DIR/runs/<final setting>/seed_<s>/
DATA_SIG = run_signature()

final_rows = []
for seed in SEEDS:
    r_ = run_or_load(FINAL_TAG, seed)
    final_rows.append(r_["row"]); free(r_)
SEED_RESULTS = pd.DataFrame(final_rows)
metric_cols = ["selected_epoch", "dev_risk", "leaf_OA", "leaf_balanced_acc", "leaf_macro_F1", "leaf_kappa", "leaf_QWK",
               "sim_OA", "sim_macro_F1", "vacuity_AUROC", "uav_lowconf_frac"] + [f"uav_frac_{n}" for n in CLASS_NAMES]
metric_cols = [c for c in metric_cols if c in SEED_RESULTS]

if MAP_SEED == "best_dev" or MAP_SEED is None:
    MAP_SEED = int(SEED_RESULTS.loc[SEED_RESULTS["dev_risk"].idxmin(), "seed"])
    _why = "lowest DEV risk (label-free)"
else:
    MAP_SEED = int(MAP_SEED); _why = "fixed in Section 0"
SEED_RESULTS["best_seed"] = SEED_RESULTS["seed"] == MAP_SEED
SEED_RESULTS.to_csv(OUT_DIR / "seed_results.csv", index=False)
summ = pd.DataFrame(dict(mean=SEED_RESULTS[metric_cols].mean(), sd=SEED_RESULTS[metric_cols].std(ddof=1),
                         min=SEED_RESULTS[metric_cols].min(), max=SEED_RESULTS[metric_cols].max(),
                         best_seed=SEED_RESULTS.set_index("seed").loc[MAP_SEED, metric_cols]))
summ["mean ± SD"] = [f"{a:.3f} ± {b:.3f}" if np.isfinite(b) else f"{a:.3f}" for a, b in zip(summ["mean"], summ["sd"])]
summ.to_csv(OUT_DIR / "seed_results_summary.csv")
print(f"\n=========== Final model ({FINAL_TAG}), {len(SEEDS)} seeds — label-free results ===========")
print(SEED_RESULTS[["seed", "best_seed"] + metric_cols].round(3).to_string(index=False))
print("\nMean ± SD over seeds (and the best seed):\n" + summ[["mean ± SD", "min", "max", "best_seed"]].round(3).to_string())
print(f"\nBest seed = {MAP_SEED} ({_why}); it writes the main maps. Field crowns were not used for this choice.")

# ---- the best seed becomes the global model for Sections 7-9 ----
_b = load_run(FINAL_TAG, MAP_SEED)
teacher, _c, _ck = _b["teacher"], _b["cal"], _b["ck"]
hist = pd.read_csv(SEED_DIR / f"seed_{MAP_SEED}" / "training_history.csv")
_dev = pd.read_csv(SEED_DIR / f"seed_{MAP_SEED}" / "dev_scores.csv")
VAC_THR, MU_K, PREC, MAHA_THR = _c["vac_thr"], _c["mu"], _c["prec"], _c["maha_thr"]
ov, ok_v, scr_v = _c["ov"], _c["ok_v"], _c["scr_v"]
PI_T, SELECTED_EPOCH = np.asarray(_ck["pi_t"]), int(_ck["selected_epoch"])
_dev.to_csv(FIG_DATA / "dev_scores.csv", index=False); hist.to_csv(FIG_DATA / "training_history.csv", index=False)
hist.to_csv(OUT_DIR / "training_history.csv", index=False)


def maha(f):
    return maha_dist(f, MU_K, PREC)


print_heldout_report(_c)
plot_history(hist, OUT_DIR / "training_diagnostics.png")
meta = dict(band_idx=band_idx.tolist(), wavelengths=WL.tolist(), mask=MASK, classes=CLASS_NAMES, cfg=CFG, seed=MAP_SEED,
            setting=BEST_SETTING, ablation=ABLATION, vac_thr=VAC_THR, maha_thr=MAHA_THR, pi_t=PI_T.tolist(),
            selected_epoch=SELECTED_EPOCH,
            selected_wavelengths=None if CFG.get("band_sel") is None else WL[CFG["band_sel"]].tolist())
torch.save(dict(state=teacher.state_dict(), meta=meta, mu_k=MU_K, prec=PREC), OUT_DIR / "mangrovebridge_teacher.pt")
(OUT_DIR / "mangrovebridge_meta.json").write_text(json.dumps(meta, indent=2, default=float))

# %%
# ==========================================================
# SECTION 7a — MAPPING FUNCTIONS
# ==========================================================
import shutil


def predict_strip(spec, mdl):
    """(n, B) reflectance -> numpy dict with the 'unknown' flag of model `mdl`; replaceable for testing."""
    o = predict(mdl["teacher"], spec, TGT)
    o["unknown"] = (o["vac"] > mdl["vac_thr"]) | (maha_dist(o["f"], mdl["mu"], mdl["prec"]) > mdl["maha_thr"])
    return o


IND_BANDS = {w: band_near(w) for w in (531, 570, 670, 700, 740, 780)}


def map_uav(p, out, mdl):
    """Map one plot with model `mdl` into folder `out`. EVERY crown pixel gets a class (1-3): pixels with a valid
    spectrum are classified by the model; the few without one take the class of their crown. Every crown in the crown
    raster has at least one valid pixel (Section 2b), so every crown gets a class: no 'no data' pixels or crowns."""
    bands = [int(i) + 1 for i in band_idx]
    L = len(LEVELS)
    with rasterio.open(p["hsi"]) as src, rasterio.open(p["crown_raster"]) as csrc:
        H, W, nod = src.height, src.width, src.nodata
        max_id = max(int(csrc.read(1, window=Window(0, r0, W, min(512, H - r0))).max()) for r0 in range(0, H, 512))
        C = max_id + 1
        # per crown and per evidence level (confident sunlit / sunlit / leaf / all valid): evidence, counts, statistics
        ev_lv = np.zeros((L, C, K - 1, 2)); n_lv = np.zeros((L, C), np.int64)
        votes_lv = np.zeros((L, C, K), np.int64); s1_lv = np.zeros((L, C)); s2_lv = np.zeros((L, C))
        flag_lv = np.zeros((L, C), np.int64); ind_lv = np.zeros((L, C, len(IND_BANDS)))
        n_tot = np.zeros(C, np.int64); n_type = np.zeros((4, C), np.int64)          # pixel types 1-3, 0 = invalid
        base = dict(driver="GTiff", height=H, width=W, crs=src.crs, transform=src.transform,
                    compress="deflate", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        tmp = dict(driver="GTiff", height=H, width=W, crs=src.crs, transform=src.transform, BIGTIFF="IF_SAFER")
        f_cls, f_lay = out / "_tmp_pixel_class.tif", out / "_tmp_pixel_layers.tif"     # first pass (uncompressed)
        names = ["stress_score", "p_gt_healthy", "p_gt_moderate", "vacuity", "low_confidence_flag", "pixel_type"]
        with rasterio.open(f_cls, "w", **tmp, count=1, dtype="uint8", nodata=0) as dc, \
             rasterio.open(f_lay, "w", **tmp, count=len(names), dtype="float32", nodata=-9999) as dl:
            for r0 in range(0, H, CFG["rows_per_strip"]):
                win = Window(0, r0, W, min(CFG["rows_per_strip"], H - r0)); h = int(win.height)
                cr = csrc.read(1, window=win).ravel()
                cls = np.zeros(h * W, np.uint8); lay = np.full((len(names), h * W), -9999, np.float32)
                inside = np.flatnonzero(cr > 0)
                if len(inside):
                    np.add.at(n_tot, cr[inside], 1)
                    raw = src.read(bands, window=win).astype(np.float32).reshape(len(bands), -1).T[inside]
                    ok = spectra_ok(raw, nod)
                    np.add.at(n_type[0], cr[inside][~ok], 1)
                    if ok.any():
                        S = clean_spectra(raw[ok], MASK["scale"]); pix = inside[ok]; ids = cr[pix]
                        veg, sun = pixel_masks(S)
                        ptype = np.where(sun, 1, np.where(veg, 2, 3))
                        o = predict_strip(S, mdl)                                 # every valid crown pixel
                        lab = (o["cum"] > 0.5).sum(1); score = o["cum"].sum(1); flag = o["unknown"]
                        cls[pix] = lab + 1
                        lay[:, pix] = np.vstack([score, o["cum"][:, 0], o["cum"][:, 1], o["vac"], flag, ptype])
                        for t_ in (1, 2, 3):
                            np.add.at(n_type[t_], ids[ptype == t_], 1)
                        Sind = S[:, list(IND_BANDS.values())]
                        for lv, m in enumerate([sun & ~flag, sun, veg, np.ones_like(sun)]):
                            np.add.at(ev_lv[lv], ids[m], o["ev"][m]); np.add.at(n_lv[lv], ids[m], 1)
                            np.add.at(votes_lv[lv], (ids[m], lab[m]), 1); np.add.at(flag_lv[lv], ids[m], flag[m].astype(int))
                            np.add.at(s1_lv[lv], ids[m], score[m]); np.add.at(s2_lv[lv], ids[m], score[m] ** 2)
                            np.add.at(ind_lv[lv], ids[m], Sind[m])
                dc.write(cls.reshape(1, h, W), window=win)
                dl.write(lay.reshape(len(names), h, W), window=win)

    # ---- crown fusion (crown_fusion, Section 6c) ----
    ids = np.flatnonzero(n_tot > 0)
    fu = crown_fusion(ev_lv, n_lv, ids, mdl["vac_thr"])
    if fu["nodata"].any():                     # cannot happen when Section 2b ran; dropped rather than written as 'no data'
        warnings.warn(f"{p['name']}: {int(fu['nodata'].sum())} crowns without a valid pixel were left out of the outputs.")
        keep_c = ~fu["nodata"]; ids = ids[keep_c]; fu = {k: (v[keep_c] if hasattr(v, "__len__") else v) for k, v in fu.items()}
    level, n_used, cum, vac, cls_c, reliability = (fu[k] for k in ("level", "n_used", "cum", "vac", "cls", "reliability"))
    ar = np.arange(len(ids))
    nu = np.maximum(n_used, 1)
    votes = votes_lv[level, ids]; s1 = s1_lv[level, ids]; s2 = s2_lv[level, ids]; nflag = flag_lv[level, ids]
    R_ind = ind_lv[level, ids] / nu[:, None]
    r = {w: R_ind[:, j] for j, w in enumerate(IND_BANDS)}
    with np.errstate(divide="ignore", invalid="ignore"):
        rep_ = 700 + 40 * ((r[670] + r[780]) / 2 - r[700]) / (r[740] - r[700])
        pri_ = (r[531] - r[570]) / (r[531] + r[570])
    mean_s = s1 / nu
    table = pd.DataFrame(dict(
        plot=p["name"], crown_id=CROWN_INFO.loc[ids, "crown_id"].to_numpy(), uid=ids,
        n_pixels=n_tot[ids], n_valid=n_tot[ids] - n_type[0, ids], n_sunlit=n_type[1, ids], n_shaded=n_type[2, ids],
        n_non_leaf=n_type[3, ids], n_filled=n_type[0, ids],
        evidence_source=[LEVELS[l] for l in level], n_pixels_used=n_used,
        p_gt_healthy=cum[:, 0], p_gt_moderate=cum[:, 1], crown_stress_score=cum.sum(1), crown_vacuity=vac,
        reliability=reliability, frac_low_confidence=nflag / nu, pixel_score_mean=mean_s,
        pixel_score_sd=np.sqrt(np.maximum(s2 / nu - mean_s ** 2, 0)),
        **{f"frac_{n.lower()}": votes[:, k] / nu for k, n in enumerate(CLASS_NAMES)},
        REP=rep_, PRI=pri_, class_code=cls_c, class_name=[CLASS_NAMES[c - 1] for c in cls_c]))
    table.to_csv(out / "crown_condition.csv", index=False)

    # ---- second pass: crown class raster, and crown pixels without a valid spectrum take their crown's class ----
    lut = np.zeros(C, np.uint8); lut[ids] = cls_c
    lut_s = np.full(C, -9999, np.float32); lut_s[ids] = cum.sum(1)
    lut_h = np.full(C, -9999, np.float32); lut_h[ids] = cum[:, 0]
    lut_m = np.full(C, -9999, np.float32); lut_m[ids] = cum[:, 1]
    lut_v = np.full(C, -9999, np.float32); lut_v[ids] = vac
    with rasterio.open(p["crown_raster"]) as csrc, rasterio.open(f_cls) as tc, rasterio.open(f_lay) as tl, \
         rasterio.open(out / "crown_class.tif", "w", **base, count=1, dtype="uint8", nodata=0) as d, \
         rasterio.open(out / "pixel_class.tif", "w", **base, count=1, dtype="uint8", nodata=0) as dc, \
         rasterio.open(out / "pixel_layers.tif", "w", **base, count=len(names), dtype="float32", nodata=-9999) as dl:
        for r0 in range(0, H, 512):
            w_ = Window(0, r0, W, min(512, H - r0))
            cr = csrc.read(1, window=w_)
            cc = lut[cr]
            d.write(cc[None], window=w_)
            pc = tc.read(1, window=w_); ly = tl.read(window=w_)
            pc[cc == 0] = 0; ly[:, cc == 0] = -9999                       # only classified crowns are written
            fill = (cc > 0) & (pc == 0)                                   # crown pixels without a valid spectrum
            if fill.any():
                pc[fill] = cc[fill]
                for b_, lut_ in enumerate((lut_s, lut_h, lut_m, lut_v)):
                    ly[b_][fill] = lut_[cr[fill]]
                ly[4][fill] = 0; ly[5][fill] = 4
            dc.write(pc[None], window=w_); dl.write(ly, window=w_)
        d.set_band_description(1, "crown class: " + ",".join(f"{k+1} {n}" for k, n in enumerate(CLASS_NAMES))
                               + " (0 = outside crowns)")
        dc.set_band_description(1, "pixel class: " + ",".join(f"{k+1} {n}" for k, n in enumerate(CLASS_NAMES))
                                + " (0 = outside crowns)")
        for i, n in enumerate(names, 1):
            dl.set_band_description(i, n if n != "pixel_type" else
                                    "pixel_type: 1 sunlit leaf, 2 shaded leaf, 3 non-leaf, 4 no valid spectrum (crown class)")
    for f_ in (f_cls, f_lay):
        try:
            f_.unlink()
        except OSError:
            pass
    return table


SHP_SHORT = {"n_pixels": "n_px", "n_sunlit": "n_sun", "frac_assessed": "f_assess", "frac_low_confidence": "f_lowconf",
             "evidence_source": "evidence", "n_pixels_used": "n_px_used", "p_gt_healthy": "p_gt_H", "p_gt_moderate": "p_gt_M",
             "crown_stress_score": "stress", "crown_vacuity": "vacuity", "reliability": "reliab",
             "pixel_score_mean": "px_sc_mean", "pixel_score_sd": "px_sc_sd", "class_code": "cls_code", "class_name": "cls_name",
             "n_valid": "n_valid", "n_shaded": "n_shade", "n_non_leaf": "n_nonleaf", "n_filled": "n_filled",
             **{f"frac_{n.lower()}": f"f_{n.lower()[:8]}" for n in CLASS_NAMES}}


def write_vectors(table, gdf, stem):
    """GeoPackage (all attributes) and shapefile (<= 10-character field names) of a crown table."""
    t = table.drop(columns=["plot", "crown_id"], errors="ignore")
    gdf.merge(t, on="uid", how="inner").to_file(stem.with_suffix(".gpkg"), driver="GPKG")     # classified crowns only
    shp = gdf[["uid", "plot", "crown_id", "geometry"]].merge(t, on="uid", how="inner")
    if "class_code" in shp:
        shp["class_code"] = shp["class_code"].astype(int)
    shp.rename(columns=SHP_SHORT).to_file(stem.with_suffix(".shp"))


def maps_dir(tag, seed):
    return run_dir(tag, seed) / "maps"


def map_stamp(tag, seed):
    """Identifies the model and data a run's maps were made with (maps are redone if either changes)."""
    f = run_dir(tag, seed) / "teacher.pt"
    return dict(tag=tag, seed=seed, data_sig=DATA_SIG, model_mtime=int(f.stat().st_mtime) if f.exists() else None,
                plots=[p["name"] for p in PLOT_INFO])


def maps_done(tag, seed):
    root = maps_dir(tag, seed)
    try:
        stamp = json.loads((root / "map_info.json").read_text())
    except Exception:
        return False
    return stamp == map_stamp(tag, seed) and (root / "crown_condition_all_plots.shp").exists() and all(
        (root / p["name"] / f).exists() for p in PLOT_INFO for f in ("pixel_class.tif", "crown_class.tif", "crown_condition.shp"))


def map_run(tag, seed, cfg=None, abl=None, skip_done=True):
    """All plots with the model of one run -> runs/<tag>/seed_<s>/maps/; returns the crown table of all plots."""
    root = maps_dir(tag, seed); root.mkdir(parents=True, exist_ok=True)
    if skip_done and maps_done(tag, seed):
        print(f"  {tag} / seed {seed}: maps already complete -> skipped")
        return pd.read_csv(root / "crown_condition_all_plots.csv")
    r_ = load_run(tag, seed, cfg, abl)
    if r_ is None:
        print(f"  {tag} / seed {seed}: no trained model for the current data/settings (train it first) -> skipped")
        return None
    c = r_["cal"]
    mdl = dict(teacher=r_["teacher"], vac_thr=c["vac_thr"], mu=c["mu"], prec=c["prec"], maha_thr=c["maha_thr"])
    print(f"  {tag} / seed {seed}: mapping (low confidence = vacuity > {mdl['vac_thr']:.3f} or Mahalanobis > "
          f"{mdl['maha_thr']:.2f})", flush=True)
    tabs = []
    for p in PLOT_INFO:
        out = root / p["name"]; out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        t = map_uav(p, out, mdl)
        write_vectors(t, crowns_gdf[crowns_gdf["plot"] == p["name"]], out / "crown_condition")
        print(f"    {p['name']}: " + ", ".join(f"{k} {v}" for k, v in t["class_name"].value_counts().items())
              + f" ({time.time() - t0:.0f} s)", flush=True)
        tabs.append(t)
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(root / "crown_condition_all_plots.csv", index=False)
    write_vectors(tab, crowns_gdf, root / "crown_condition_all_plots")
    (root / "map_info.json").write_text(json.dumps(map_stamp(tag, seed), indent=2))
    del mdl, r_
    gc.collect()
    if str(DEVICE).startswith("cuda"):
        torch.cuda.empty_cache()
    return tab


def crown_plausibility(table):
    """Spearman rho of crown stress score with REP and PRI (crowns with sunlit pixels, reliability high/medium).
    Expected negative. Label-free evidence, not accuracy."""
    chk = table.dropna(subset=["crown_stress_score", "REP", "PRI"])
    chk = chk[chk["reliability"] != "low"]
    out = dict(n_crowns_rho=len(chk))
    for col in ("REP", "PRI"):
        if len(chk) > 5 and chk["crown_stress_score"].std() > 0:
            rho, pv = spearmanr(chk["crown_stress_score"], chk[col])
            out[f"rho_{col}"], out[f"p_{col}"] = float(rho), float(pv)
        else:
            out[f"rho_{col}"], out[f"p_{col}"] = np.nan, np.nan
    cls = table[table["class_code"].between(1, K)]
    out.update({f"crown_frac_{n}": float((cls["class_code"] == k + 1).mean()) if len(cls) else np.nan
                for k, n in enumerate(CLASS_NAMES)})
    out["crown_frac_high_reliability"] = float((table["reliability"] == "high").mean())
    return out


def consensus(codes):
    """codes: (n_crowns_or_pixels, n_seeds), 1..K valid, other values ignored -> majority class (ties: class nearest
    to the median of the seeds), number of valid seeds, fraction of valid seeds that agree with the majority."""
    valid = (codes >= 1) & (codes <= K)
    cnt = np.stack([((codes == k) & valid).sum(1) for k in range(1, K + 1)], 1)
    n_valid = valid.sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                     # rows without any valid seed
        med = np.nan_to_num(np.nanmedian(np.where(valid, codes, np.nan).astype(float), 1)) if codes.shape[1] else np.zeros(len(codes))
    best = cnt.max(1, keepdims=True)
    tie_dist = np.where(cnt == best, np.abs(np.arange(1, K + 1)[None] - med[:, None]), np.inf)
    cons = np.where(n_valid > 0, tie_dist.argmin(1) + 1, 0)
    agree = np.where(n_valid > 0, cnt.max(1) / np.maximum(n_valid, 1), np.nan)
    return cons, n_valid, agree



def seed_agreement(tables, map_dir_of, out_dir, label="seeds"):
    """Crown and pixel agreement between several runs (tables: seed -> crown table; map_dir_of(seed) -> maps folder)."""
    AGREE_DIR = Path(out_dir); AGREE_DIR.mkdir(parents=True, exist_ok=True)
    seed_tables, mapped = tables, list(tables)
    seed_map_dir = map_dir_of
    # crown level
    wide = CROWN_INFO.reset_index()[["uid", "plot", "crown_id"]]
    for s in mapped:
        st = seed_tables[s][["uid", "class_code", "crown_stress_score", "crown_vacuity"]]
        wide = wide.merge(st.rename(columns={"class_code": f"cls_s{s}", "crown_stress_score": f"stress_s{s}",
                                             "crown_vacuity": f"vac_s{s}"}), on="uid", how="left")
    codes = wide[[f"cls_s{s}" for s in mapped]].fillna(0).to_numpy().astype(int)
    cons, n_valid, agree = consensus(codes)
    wide["n_seeds"] = n_valid
    wide["cons_code"] = cons
    wide["cons_name"] = [(["No data"] + CLASS_NAMES)[c] for c in cons]
    wide["agreement"] = agree
    wide["unanimous"] = (agree == 1.0) & (n_valid == len(mapped))
    stress = wide[[f"stress_s{s}" for s in mapped]]
    wide["stress_mean"], wide["stress_sd"] = stress.mean(1), stress.std(1, ddof=1)
    wide["vac_mean"] = wide[[f"vac_s{s}" for s in mapped]].mean(1)
    wide.to_csv(AGREE_DIR / "crown_condition_seed_agreement.csv", index=False)
    g = crowns_gdf[["uid", "plot", "crown_id", "geometry"]].merge(wide.drop(columns=["plot", "crown_id"]), on="uid", how="left")
    g.to_file(AGREE_DIR / "crown_condition_seed_agreement.gpkg", driver="GPKG")
    shp_names = {"agreement": "agree", "unanimous": "unanim", "stress_mean": "stress_mn",
                 **{f"stress_s{s}": f"st_s{s}" for s in mapped}}            # shapefile: <= 10 characters
    g = g.rename(columns=shp_names)
    g["unanim"] = g["unanim"].fillna(False).astype(int)
    g["cons_code"] = g["cons_code"].fillna(0).astype(int); g["cons_name"] = g["cons_name"].fillna("No data")
    g.to_file(AGREE_DIR / "crown_condition_seed_agreement.shp")

    # pixel level: majority class and % of seeds agreeing, per plot
    for p in PLOT_INFO:
        out = AGREE_DIR / p["name"]; out.mkdir(exist_ok=True)
        srcs = [rasterio.open(seed_map_dir(s) / p["name"] / "pixel_class.tif") for s in mapped]
        try:
            prof = srcs[0].profile.copy()
            prof.update(count=2, dtype="uint8", nodata=255)
            H, W = srcs[0].height, srcs[0].width
            with rasterio.open(out / "pixel_class_consensus.tif", "w", **prof) as dst:
                for r0 in range(0, H, 512):
                    win = Window(0, r0, W, min(512, H - r0)); h = int(win.height)
                    stack = np.stack([s_.read(1, window=win).ravel() for s_ in srcs], 1).astype(int)
                    c, nv, a = consensus(stack)
                    band1 = np.where(nv > 0, c, stack[:, 0]).astype(np.uint8)          # 0 outside / 4 not assessed kept
                    band2 = np.where(nv > 0, np.round(100 * np.nan_to_num(a)), 255).astype(np.uint8)
                    dst.write(np.stack([band1, band2]).reshape(2, h, W), window=win)
                dst.set_band_description(1, "majority class over seeds: 0 outside crowns," +
                                         ",".join(f"{k+1} {n}" for k, n in enumerate(CLASS_NAMES)) + f",{K+1} Not assessed")
                dst.set_band_description(2, "percent of seeds agreeing with the majority class (255 = not classified)")
        finally:
            for s_ in srcs:
                s_.close()

    # class counts per plot and seed, mean ± SD over seeds
    cnt = pd.concat([seed_tables[s].assign(seed=s) for s in mapped])
    cnt = cnt[cnt["class_code"].between(1, K)].groupby(["plot", "seed", "class_name"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(columns=CLASS_NAMES, fill_value=0)
    tot = cnt.groupby(level="seed").sum().assign(plot="All plots").set_index("plot", append=True).swaplevel()
    cnt = pd.concat([cnt, tot])
    cnt.to_csv(AGREE_DIR / "crown_class_counts_by_seed.csv")
    summ = cnt.groupby(level="plot").agg(["mean", "std"])
    summ_txt = pd.DataFrame({n: [f"{m:.1f} ± {sd:.1f}" for m, sd in zip(summ[(n, "mean")], summ[(n, "std")].fillna(0))]
                             for n in CLASS_NAMES}, index=summ.index)
    summ_txt.to_csv(AGREE_DIR / "crown_class_counts_mean_sd.csv")
    print(f"\nCrowns per class, mean ± SD over {len(mapped)} seeds:\n{summ_txt.to_string()}")

    v = wide[wide["n_seeds"] > 0]
    agr = v.groupby("plot").agg(n_crowns=("uid", "size"), unanimous_pct=("unanimous", "mean"),
                                mean_agreement=("agreement", "mean"), stress_sd_mean=("stress_sd", "mean"))
    agr.loc["All plots"] = [len(v), v["unanimous"].mean(), v["agreement"].mean(), v["stress_sd"].mean()]
    agr["unanimous_pct"] *= 100
    agr.to_csv(AGREE_DIR / "seed_agreement_by_plot.csv")
    print(f"\nAgreement between {label} (unanimous = same class from all {len(mapped)} seeds):\n" + agr.round(3).to_string())

    print(f"\nSeed agreement written to {AGREE_DIR}")
    return wide, agr

print("Mapping functions ready.")

# %%
# ==========================================================
# SECTION 7b — MAP EVERY SEED OF THE FINAL MODEL
# ==========================================================
MAP_SEEDS = SEEDS               # seeds to map; e.g. [MAP_SEED] for the best seed only
MAP_SKIP_DONE = True            # skip a seed whose maps are already complete (resume after an interruption)
MAP_COPY_MAIN = True            # copy the best-seed maps to OUT_DIR/<plot>/ for Sections 8-9 and the figure notebook

if MAP_SEED not in MAP_SEEDS:
    MAP_SEEDS = [MAP_SEED] + list(MAP_SEEDS)
seed_tables = {}
for seed in MAP_SEEDS:
    print(f"\n==================== mapping, seed {seed} ====================", flush=True)
    t = map_run(FINAL_TAG, seed, skip_done=MAP_SKIP_DONE)
    if t is not None:
        seed_tables[seed] = t
if MAP_SEED not in seed_tables:
    raise RuntimeError(f"The best seed {MAP_SEED} could not be mapped (see the messages above).")

crown_table = seed_tables[MAP_SEED]
if MAP_COPY_MAIN:
    for p in PLOT_INFO:
        shutil.copytree(maps_dir(FINAL_TAG, MAP_SEED) / p["name"], OUT_DIR / p["name"], dirs_exist_ok=True)
    for f in maps_dir(FINAL_TAG, MAP_SEED).glob("crown_condition_all_plots.*"):
        shutil.copy(f, OUT_DIR / f.name)
    print(f"\nMain maps (best seed {MAP_SEED}) copied to {OUT_DIR}\\<plot> and {OUT_DIR}\\crown_condition_all_plots.*")

if len(seed_tables) > 1:
    AGREE_WIDE, AGREE_BY_PLOT = seed_agreement(seed_tables, lambda s: maps_dir(FINAL_TAG, s), OUT_DIR / "seed_agreement")
else:
    print("Only one seed mapped: no seed-agreement outputs.")

print(f"\nMain maps (best seed {MAP_SEED}): crown classes per plot:\n"
      + crown_table.groupby("plot")["class_name"].value_counts().unstack(fill_value=0).to_string())
print("\nEvidence used for the crown class:\n" + crown_table["evidence_source"].value_counts().to_string())
print("\nReliability:\n" + crown_table["reliability"].value_counts().to_string())
print("Per-seed maps in", SEED_DIR)

# %%
# ==========================================================
# SECTION 8 — PLAUSIBILITY (every seed)
# ==========================================================
_pl = pd.DataFrame([dict(seed=s_, **crown_plausibility(t_)) for s_, t_ in seed_tables.items()])
SEED_RESULTS = SEED_RESULTS.drop(columns=[c for c in _pl.columns if c != "seed" and c in SEED_RESULTS]).merge(_pl, on="seed", how="left")
SEED_RESULTS.to_csv(OUT_DIR / "seed_results.csv", index=False)
cols = metric_cols + [c for c in _pl.columns if c != "seed" and not c.startswith("p_")]
summ = pd.DataFrame(dict(mean=SEED_RESULTS[cols].mean(), sd=SEED_RESULTS[cols].std(ddof=1),
                         min=SEED_RESULTS[cols].min(), max=SEED_RESULTS[cols].max(),
                         best_seed=SEED_RESULTS.set_index("seed").loc[MAP_SEED, cols]))
summ["mean ± SD"] = [f"{a:.3f} ± {b:.3f}" if np.isfinite(b) else f"{a:.3f}" for a, b in zip(summ["mean"], summ["sd"])]
summ.to_csv(OUT_DIR / "seed_results_summary.csv")
print("Plausibility per seed (expected: rho < 0):")
print(SEED_RESULTS[["seed", "best_seed", "n_crowns_rho", "rho_REP", "p_REP", "rho_PRI", "p_PRI"]
                   + [f"crown_frac_{n}" for n in CLASS_NAMES]].round(3).to_string(index=False))
print(f"\nAll label-free results, mean ± SD over {len(SEED_RESULTS)} seeds (best seed = {MAP_SEED}):")
print(summ[["mean ± SD", "min", "max", "best_seed"]].round(3).to_string())

# %%
# ==========================================================
# SECTION 9 — EXPORT DATA FOR THE FIGURE NOTEBOOK
# ==========================================================
_r9 = np.random.default_rng(CFG["seed"] + 9)

np.savez_compressed(FIG_DATA / "leaf_spectra.npz", wl=WL, R=R_leaf, R_fit=R_fit, y=y_leaf, val=leaf_val,
                    groups=np.asarray(leaf_groups).astype(str))
pd.DataFrame(TRAITS, columns=TRAIT_NAMES).assign(cls=y_leaf, val=leaf_val, fit_rmse=fit_rmse).to_csv(
    FIG_DATA / "leaf_traits.csv", index=False)

_si = _r9.choice(len(sim_R), min(6000, len(sim_R)), replace=False) if len(sim_R) else np.zeros(0, int)
np.savez_compressed(FIG_DATA / "sail_library_sample.npz", R=sim_R[_si], y=y_leaf[sim_leaf[_si]] if len(_si) else np.zeros(0, int),
                    lai=sim_meta["lai"].to_numpy()[_si] if len(_si) else np.zeros(0),
                    ala=sim_meta["ala"].to_numpy()[_si] if len(_si) else np.zeros(0), bg=BG)

_ui = _r9.choice(len(R_tgt), min(20000, len(R_tgt)), replace=False)
_ou = predict(teacher, R_tgt[_ui], TGT)
np.savez_compressed(FIG_DATA / "uav_pixels_sample.npz", R=R_tgt[_ui].astype(np.float32), uid=tgt_crown[_ui],
                    probs=_ou["probs"], cum=_ou["cum"], vac=_ou["vac"],
                    R_screened=R_screened[:5000].astype(np.float32), R_bg=R_bg[:5000].astype(np.float32))

pd.DataFrame(dict(source=np.where(src_is_sim[val_idx], "4SAIL", "leaf"), y=src_y[val_idx],
                  pred=(ov["cum"] > 0.5).sum(1), lai=src_lai[val_idx], vacuity=ov["vac"],
                  **{f"p_{n.lower()}": ov["probs"][:, k] for k, n in enumerate(CLASS_NAMES)})).to_csv(
    FIG_DATA / "heldout_predictions.csv", index=False)

np.savez_compressed(FIG_DATA / "vacuity_check.npz", sunlit=ok_v, screened=scr_v, vac_thr=VAC_THR, maha_thr=MAHA_THR)

_lv = np.flatnonzero(~src_is_sim[val_idx])
_sv = np.flatnonzero(src_is_sim[val_idx]); _sv = _r9.choice(_sv, min(2000, len(_sv)), replace=False) if len(_sv) else _sv
_tu = _r9.choice(len(_ui), min(3000, len(_ui)), replace=False)
np.savez_compressed(FIG_DATA / "features_sample.npz",
                    f=np.vstack([ov["f"][_lv], ov["f"][_sv], _ou["f"][_tu]]),
                    domain=np.array(["Leaf"] * len(_lv) + ["4SAIL canopy"] * len(_sv) + ["UAV pixel"] * len(_tu)),
                    cls=np.r_[src_y[val_idx][_lv], src_y[val_idx][_sv], np.full(len(_tu), -1)],
                    pred=np.r_[(ov["cum"][_lv] > 0.5).sum(1), (ov["cum"][_sv] > 0.5).sum(1), (_ou["cum"][_tu] > 0.5).sum(1)])

hist.to_csv(FIG_DATA / "training_history.csv", index=False)
import shutil
for _src, _dst in ((OUT_DIR / "seed_results.csv", "seed_results.csv"),
                   (OUT_DIR / "seed_results_summary.csv", "seed_results_summary.csv"),
                   (TUNE_DIR / "stage1_one_at_a_time.csv", "tuning_stage1.csv"),
                   (TUNE_DIR / "stage2_summary.csv", "tuning_stage2.csv")):
    if _src.exists():
        shutil.copy(_src, FIG_DATA / _dst)
for _f in ("field_points.csv", "field_crowns.csv", "field_assessment_by_seed.csv", "field_confusion_by_seed.csv",
           "field_crown_predictions_by_seed.csv"):
    if (FIG_DATA / _f).exists():                 # left over from an earlier version: no field results are exported now
        (FIG_DATA / _f).unlink()


def _jsonable(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.ndarray, tuple)): return [_jsonable(x) for x in v]
    if isinstance(v, (Path,)): return str(v)
    return v


manifest = dict(
    class_names=CLASS_NAMES, wl_model=WL.tolist(), wl_full=wl_full.tolist(), band_idx=band_idx.tolist(),
    wl_selected=None if CFG.get("band_sel") is None else WL[CFG["band_sel"]].tolist(),
    reflectance_scale=float(MASK["scale"]), selected_epoch=int(SELECTED_EPOCH), seeds=SEEDS, map_seed=MAP_SEED, warmup_epochs=CFG["warmup_epochs"],
    vac_thr=VAC_THR, maha_thr=MAHA_THR, pi_t=[float(v) for v in PI_T], crs=str(crowns_gdf.crs), out_dir=str(OUT_DIR),
    crown_table=str(OUT_DIR / "crown_condition_all_plots.csv"), field=False, final_setting=BEST_SETTING,
    plots=[dict(name=p["name"], hsi=str(p["hsi"]), crown_raster=str(p["crown_raster"]), map_dir=str(OUT_DIR / p["name"]),
                n_crowns=int((crowns_gdf["plot"] == p["name"]).sum())) for p in PLOT_INFO],
    cfg={k: _jsonable(v) for k, v in CFG.items()})
(FIG_DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("Figure data written to", FIG_DATA)
for f in sorted(FIG_DATA.iterdir()):
    print(f"  {f.name:28s} {f.stat().st_size / 1e6:7.2f} MB")
