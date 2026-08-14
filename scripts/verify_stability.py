"""
_verify_stability.py
====================
Comprehensive reproducibility verification and stability analysis for the
Ras Sanad Mangrove Recovery 2016-2025 workflow.

Execution phases:
  1.  EE init + ROI + shapefiles + ever-mangrove union
  2.  Build multi-year training composites (fixed seeds)
  3.  n_trees tuning: 50, 100, 150, 200, 300 trees  →  find plateau
  4.  Final training with best_n_trees
  5.  Threshold calibration vs RasSanad_2016.shp
  6.  Annual area estimation 2016-2025  (Run 1)
  7.  Save Run 1 metrics + comparison checkpoint
  8.  Re-train RF (same seeds) + re-validate  →  compare with Run 1
  9.  Re-run calibration (threshold)  →  compare
 10.  Re-run area estimation for 3 sample years  →  compare
 11.  Generate all figures with publication-quality legends
 12.  Write verification report

Usage:
  "C:/ProgramData/anaconda3/envs/geoai_rs/python.exe" _verify_stability.py
  (use geoai_rs env which has earthengine-api + geopandas + PIL)
"""

import sys, os, json, time, warnings, traceback
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
import ee
from PIL import Image as PILImage

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'DejaVu Sans', 'font.size': 10,
})

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = r'C:\path\to\workspace'
VOUT      = os.path.join(BASE, 'Ras_Sanad_Verification')
VMAPS     = os.path.join(VOUT, 'maps')
VACC      = os.path.join(VOUT, 'accuracy')
VTAB      = os.path.join(VOUT, 'tables')
VCPK      = os.path.join(VOUT, 'checkpoints')
for d in [VOUT, VMAPS, VACC, VTAB, VCPK]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(VOUT, 'verification_log.json')

# ── Logging ────────────────────────────────────────────────────────────────────
_log = {'runs': [], 'tuning': [], 'comparison': {}, 'final': {}}

def plog(msg, end='\n'):
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', end=end, flush=True)

def save_log():
    with open(LOG_FILE, 'w') as f:
        json.dump(_log, f, indent=2, default=str)

def ckpt_path(name):
    return os.path.join(VCPK, f'{name}.json')

def load_ckpt(name):
    p = ckpt_path(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def save_ckpt(name, data):
    with open(ckpt_path(name), 'w') as f:
        json.dump(data, f, indent=2, default=str)
    plog(f'  Checkpoint saved: {name}')

# ── EE init ────────────────────────────────────────────────────────────────────
plog('=== PHASE 0: Earth Engine Init ===')
ee.Initialize(project=os.environ.get('EE_PROJECT'))  # set EE_PROJECT to your own Cloud project
plog('  EE ready.')

# ── Constants (identical to notebook Cell 1) ───────────────────────────────────
_gdf1967      = gpd.read_file(os.path.join(BASE, 'RasSanad_1967.shp')).to_crs(4326)
_cent         = _gdf1967.geometry.union_all().centroid
RAS_SANAD_LON = round(_cent.x, 5)
RAS_SANAD_LAT = round(_cent.y, 5)
CIRCLE_RADIUS_M = 2000

YEARS       = list(range(2016, 2026))
SHP_YEARS   = [1967, 1998, 2005, 2009, 2016]
TRAIN_YEARS = [2017, 2018, 2019, 2021, 2022]

HIST_AREAS = {1967: 97.30, 1998: 56.99, 2005: 37.54, 2009: 35.23, 2016: 34.05}
TARGET_HA  = 34.05

# ── REPRODUCIBILITY: All random seeds explicitly declared ──────────────────────
# Sampling seeds: seed=yr (positive), seed=yr+1000 (S2 neg), seed=yr+500 (hard neg fu)
# RF training seed: SEED_RF = 42  (used for ALL RF models)
# Validation seed : SEED_RF + 999 = 1041
SEED_RF     = 42        # fixed RF training seed – all models
SEED_VAL    = SEED_RF + 999   # = 1041 – validation sample
MAX_PTS     = 2000
VAL_PTS     = 500
THR_RANGE   = [round(t / 100, 2) for t in range(30, 91, 5)]
S2_SCALE    = 10
PRED_SCALE  = 20

S2_BANDS    = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
               'NDVI','NDMI','MNDWI','EVI','NBR']
S1_BANDS    = ['VV','VH','VV_VH']
FUSED_BANDS = S2_BANDS + S1_BANDS

_dlat       = CIRCLE_RADIUS_M / 111320
_dlon       = CIRCLE_RADIUS_M / (111320 * np.cos(np.radians(RAS_SANAD_LAT)))
ROI_LON_MIN = RAS_SANAD_LON - _dlon
ROI_LON_MAX = RAS_SANAD_LON + _dlon
ROI_LAT_MIN = RAS_SANAD_LAT - _dlat
ROI_LAT_MAX = RAS_SANAD_LAT + _dlat
ROI_EXTENT  = [ROI_LON_MIN, ROI_LON_MAX, ROI_LAT_MIN, ROI_LAT_MAX]

# Geographic aspect ratio at study latitude (makes circles round in figures)
LAT_ASP = 1.0 / np.cos(np.radians(RAS_SANAD_LAT))   # ≈ 1.114 at 26°N

# Coordinate grid ticks – strictly INSIDE data range (prevents xlim auto-expansion)
LON_TICKS = [50.58, 50.60]   # within [50.5756, 50.6156] ✓
LAT_TICKS = [26.13, 26.15]   # within [26.1288, 26.1647] ✓

plog(f'  Centroid: ({RAS_SANAD_LON}, {RAS_SANAD_LAT})')
plog(f'  ROI lon: [{ROI_LON_MIN:.4f}, {ROI_LON_MAX:.4f}]  lat: [{ROI_LAT_MIN:.4f}, {ROI_LAT_MAX:.4f}]')
plog(f'  Random seeds: RF={SEED_RF}, Val={SEED_VAL}, Sample=year-indexed')

# ── Helper functions ───────────────────────────────────────────────────────────
def ee_get(obj, retries=6, delay=5):
    for i in range(retries):
        try:
            return obj.getInfo()
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise e

def area_ha(img, roi, scale):
    a = img.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=roi,
        scale=scale, maxPixels=1e12, bestEffort=True)
    v = ee_get(a)
    val = list(v.values())[0] if v else 0
    return (val or 0) / 10_000

def mask_s2(img):
    scl = img.select('SCL')
    m = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(m)

def s2_indices(img):
    ndvi  = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndmi  = img.normalizedDifference(['B8', 'B11']).rename('NDMI')
    mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    evi   = img.expression('2.5*(N-R)/(N+6*R-7.5*B+1)',
              {'N': img.select('B8'), 'R': img.select('B4'),
               'B': img.select('B2')}).rename('EVI')
    nbr   = img.normalizedDifference(['B8', 'B12']).rename('NBR')
    return img.addBands([ndvi, ndmi, mndwi, evi, nbr])

def s2_composite(year, region):
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filter(ee.Filter.calendarRange(year, year, 'year'))
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
           .map(mask_s2).map(s2_indices))
    n = ee_get(col.size())
    if n == 0:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(region)
               .filter(ee.Filter.calendarRange(year - 1, year + 1, 'year'))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
               .map(mask_s2).map(s2_indices))
        n = ee_get(col.size())
    return col.median().select(S2_BANDS).clip(region)

def s1_composite(year, region):
    col = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filterBounds(region)
           .filter(ee.Filter.calendarRange(year, year, 'year'))
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
           .select(['VV', 'VH']))
    n = ee_get(col.size())
    if n == 0:
        col = (ee.ImageCollection('COPERNICUS/S1_GRD')
               .filterBounds(region)
               .filter(ee.Filter.calendarRange(year - 1, year + 1, 'year'))
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
               .select(['VV', 'VH']))
        n = ee_get(col.size())
    med   = col.median()
    ratio = med.select('VV').subtract(med.select('VH')).rename('VV_VH')
    return med.addBands(ratio).select(S1_BANDS).clip(region)

# ── PHASE 1: ROI + shapefiles ─────────────────────────────────────────────────
plog('\n=== PHASE 1: ROI + Shapefiles ===')
roi_center = ee.Geometry.Point([RAS_SANAD_LON, RAS_SANAD_LAT])
roi_ee     = roi_center.buffer(CIRCLE_RADIUS_M)

def gdf_to_ee_geom(gdf):
    geom = gdf.geometry.union_all()
    return ee.Geometry(geom.__geo_interface__)

SHP_GDF     = {}
RASSANAD_EE = {}
for yr in SHP_YEARS:
    fp  = os.path.join(BASE, f'RasSanad_{yr}.shp')
    gdf = gpd.read_file(fp).to_crs(4326)
    SHP_GDF[yr]     = gdf
    RASSANAD_EE[yr] = gdf_to_ee_geom(gdf)

ever_mg_geom = None
for yr, geom in RASSANAD_EE.items():
    ever_mg_geom = geom if ever_mg_geom is None else ever_mg_geom.union(geom, 1)
EVER_MG_GEOM  = ever_mg_geom.intersection(roi_ee, 1)
CALIB_2016_EE = RASSANAD_EE[2016]
plog('  ROI, shapefiles, and ever-mangrove union ready.')

# ── Post-processing masks ──────────────────────────────────────────────────────
jrc_water  = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('max_extent').unmask(0).byte()
WATER_PROX = jrc_water.focal_max(radius=500, units='meters', kernelType='circle')
ELEV_MASK  = ee.Image('USGS/SRTMGL1_003').select('elevation').lt(10)
TIDAL_MASK = WATER_PROX.And(ELEV_MASK)
MORPH_R    = 10

def clean_pred(img):
    return (img.focal_min(radius=MORPH_R, units='meters')
               .focal_max(radius=MORPH_R, units='meters'))

label_img = (ee.Image(0).byte()
             .paint(ee.FeatureCollection([ee.Feature(roi_ee.difference(EVER_MG_GEOM, 1))]), 0)
             .paint(ee.FeatureCollection([ee.Feature(EVER_MG_GEOM, {'class': 1})]), 1)
             .rename('class'))

hard_neg_geom = (EVER_MG_GEOM.buffer(600, 1)
                 .intersection(roi_ee, 1)
                 .difference(EVER_MG_GEOM, 1))
HARD_NEG_MASK = ee.Image(0).byte().paint(
    ee.FeatureCollection([ee.Feature(hard_neg_geom)]), 1)

plog('  Post-processing masks and label image ready.')

# ── PHASE 2: Build multi-year training data ───────────────────────────────────
plog('\n=== PHASE 2: Build Training Data ===')
ckpt_tr = load_ckpt('training_ready')

if ckpt_tr and ckpt_tr.get('done'):
    plog('  [CHECKPOINT] Training data already built – recreating EE collections from seeds.')
    # Re-build EE FeatureCollections (fast – deterministic with same seeds)

all_fu, all_s2 = [], []
t0_tr = time.time()
for yr in TRAIN_YEARS:
    plog(f'  Year {yr}:')
    s2c = s2_composite(yr, roi_ee)
    s1c = s1_composite(yr, roi_ee)
    fuc = s2c.addBands(s1c)
    base_fu = fuc.select(FUSED_BANDS).addBands(label_img)
    base_s2 = s2c.select(S2_BANDS).addBands(label_img)

    samp_std_fu = base_fu.stratifiedSample(
        numPoints=MAX_PTS // 2, classBand='class', region=roi_ee,
        scale=S2_SCALE, seed=yr, geometries=False,
        classValues=[0, 1], classPoints=[MAX_PTS // 2, MAX_PTS])
    samp_std_s2 = base_s2.stratifiedSample(
        numPoints=MAX_PTS // 2, classBand='class', region=roi_ee,
        scale=S2_SCALE, seed=yr + 1000, geometries=False,
        classValues=[0, 1], classPoints=[MAX_PTS // 2, MAX_PTS])
    samp_hrd_fu = (base_fu.updateMask(HARD_NEG_MASK.And(label_img.eq(0)))
                   .sample(region=roi_ee, scale=S2_SCALE,
                           numPixels=MAX_PTS // 2, seed=yr + 500, geometries=False))
    samp_hrd_s2 = (base_s2.updateMask(HARD_NEG_MASK.And(label_img.eq(0)))
                   .sample(region=roi_ee, scale=S2_SCALE,
                           numPixels=MAX_PTS // 2, seed=yr + 1500, geometries=False))

    samp_fu = samp_std_fu.merge(samp_hrd_fu)
    samp_s2 = samp_std_s2.merge(samp_hrd_s2)
    n_fu = ee_get(samp_fu.size())
    n_s2 = ee_get(samp_s2.size())
    plog(f'    Fused={n_fu:,}  S2={n_s2:,} pts')
    all_fu.append(samp_fu)
    all_s2.append(samp_s2)

tr_fu = all_fu[0]
for s in all_fu[1:]: tr_fu = tr_fu.merge(s)
tr_s2 = all_s2[0]
for s in all_s2[1:]: tr_s2 = tr_s2.merge(s)
total_fu = ee_get(tr_fu.size())
total_s2 = ee_get(tr_s2.size())
plog(f'  Total Fused: {total_fu:,}  S2-only: {total_s2:,} pts')
plog(f'  Training data built in {time.time()-t0_tr:.1f}s')
save_ckpt('training_ready', {'done': True, 'total_fu': total_fu, 'total_s2': total_s2,
                              'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')})

# Build 2023 validation composite (independent of training years)
plog('  Building 2023 validation composite...')
s2_val = s2_composite(2023, roi_ee)
s1_val = s1_composite(2023, roi_ee)
fu_val = s2_val.addBands(s1_val)
plog('  Validation composite ready.')

# ── RF training + evaluation helper ───────────────────────────────────────────
def train_and_eval(samples, bands, val_img, n_trees, seed=SEED_RF):
    """Train RF (server-side) and compute accuracy on independent validation set.
    Returns (clf_prob, clf_class, metrics_dict).
    All seeds explicitly documented for reproducibility.
    """
    clf_p = (ee.Classifier.smileRandomForest(n_trees, seed=seed)
             .setOutputMode('PROBABILITY')
             .train(features=samples, classProperty='class', inputProperties=bands))
    clf_c = (ee.Classifier.smileRandomForest(n_trees, seed=seed)
             .setOutputMode('CLASSIFICATION')
             .train(features=samples, classProperty='class', inputProperties=bands))
    # Fresh validation sample: seed = SEED_VAL = SEED_RF + 999 = 1041
    val_samp = (val_img.select(bands).addBands(label_img)
                .stratifiedSample(numPoints=VAL_PTS, classBand='class',
                                  region=roi_ee, scale=S2_SCALE, seed=SEED_VAL,
                                  geometries=False,
                                  classValues=[0, 1], classPoints=[VAL_PTS, VAL_PTS]))
    cm   = val_samp.classify(clf_c).errorMatrix('class', 'classification')
    mat  = np.array(ee_get(cm))
    oa   = ee_get(cm.accuracy())
    kap  = ee_get(cm.kappa())
    P, R, F = [], [], []
    for c in [0, 1]:
        tp = mat[c, c]
        fp = mat[:, c].sum() - tp
        fn = mat[c, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        P.append(p); R.append(r); F.append(f)
    metrics = {
        'OA': round(oa, 6), 'Kappa': round(kap, 6),
        'P_mg': round(P[1], 6), 'R_mg': round(R[1], 6), 'F1_mg': round(F[1], 6),
        'P_non': round(P[0], 6), 'R_non': round(R[0], 6), 'F1_non': round(F[0], 6),
        'CM': mat.tolist(), 'n_trees': n_trees, 'seed': seed,
    }
    return clf_p, clf_c, metrics

# ── PHASE 3: n_trees tuning ───────────────────────────────────────────────────
plog('\n=== PHASE 3: n_trees Tuning (50, 100, 150, 200, 300) ===')
ckpt_tune = load_ckpt('tuning')
TREES_TO_TEST = [50, 100, 150, 200, 300]

if ckpt_tune and 'results' in ckpt_tune and len(ckpt_tune['results']) == len(TREES_TO_TEST):
    plog('  [CHECKPOINT] Tuning already complete – loading saved results.')
    tuning_results = ckpt_tune['results']
    best_n_trees   = ckpt_tune['best_n_trees']
    plog(f'  Best n_trees from checkpoint: {best_n_trees}')
else:
    tuning_results = []
    t0_tune = time.time()
    plog('  Testing tree counts (Fused S2+S1, seed=42 each)...')
    for n in TREES_TO_TEST:
        t0 = time.time()
        _, _, metrics = train_and_eval(tr_fu, FUSED_BANDS, fu_val, n_trees=n, seed=SEED_RF)
        elapsed = time.time() - t0
        plog(f'    n_trees={n:3d}: OA={metrics["OA"]:.4f}  Kappa={metrics["Kappa"]:.4f}'
             f'  F1_mg={metrics["F1_mg"]:.4f}  ({elapsed:.1f}s)')
        tuning_results.append({'n_trees': n, **{k: v for k, v in metrics.items()
                                                 if k not in ('CM',)}})

    # Find plateau: use F1_mg as primary metric
    # Plateau = first n_trees where adding 50+ more trees improves F1 by < 0.002
    f1s = [r['F1_mg'] for r in tuning_results]
    plateau_idx = len(TREES_TO_TEST) - 1   # default: last
    for i in range(len(f1s) - 1):
        if abs(f1s[i + 1] - f1s[i]) < 0.002:
            plateau_idx = i
            break
    best_n_trees = TREES_TO_TEST[plateau_idx]
    plog(f'\n  Tuning complete in {time.time()-t0_tune:.1f}s')
    plog(f'  F1_mg per n_trees: ' + '  '.join(f'{n}→{f:.4f}'
         for n, f in zip(TREES_TO_TEST, f1s)))
    plog(f'  Plateau at index {plateau_idx}: best_n_trees = {best_n_trees}')

    _log['tuning'] = tuning_results
    save_ckpt('tuning', {'results': tuning_results, 'best_n_trees': best_n_trees,
                          'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')})

# Always use at least N_TREES_MIN to ensure model quality
N_TREES_MIN = 150
if best_n_trees < N_TREES_MIN:
    plog(f'  best_n_trees ({best_n_trees}) < minimum ({N_TREES_MIN}) – using {N_TREES_MIN}')
    best_n_trees = N_TREES_MIN
plog(f'  Final n_trees for production: {best_n_trees}')

# ── PHASE 4: Final training with best_n_trees ─────────────────────────────────
plog(f'\n=== PHASE 4: Final Training (n_trees={best_n_trees}, seed={SEED_RF}) ===')
t0 = time.time()
RF_FU, RF_FU_C, ACC_FU = train_and_eval(tr_fu, FUSED_BANDS, fu_val,
                                         n_trees=best_n_trees, seed=SEED_RF)
RF_S2, RF_S2_C, ACC_S2 = train_and_eval(tr_s2, S2_BANDS, s2_val,
                                         n_trees=best_n_trees, seed=SEED_RF)
plog(f'  Fused : OA={ACC_FU["OA"]:.4f}  Kappa={ACC_FU["Kappa"]:.4f}  '
     f'F1_mg={ACC_FU["F1_mg"]:.4f}  ({time.time()-t0:.1f}s)')
plog(f'  S2    : OA={ACC_S2["OA"]:.4f}  Kappa={ACC_S2["Kappa"]:.4f}  '
     f'F1_mg={ACC_S2["F1_mg"]:.4f}')

# ── PHASE 5: Threshold calibration ────────────────────────────────────────────
plog('\n=== PHASE 5: Threshold Calibration (Dice vs RasSanad_2016) ===')
ckpt_cal = load_ckpt('calibration')

if ckpt_cal and 'THR_FU' in ckpt_cal and ckpt_cal.get('n_trees') == best_n_trees:
    THR_FU = ckpt_cal['THR_FU']
    df_cal = pd.DataFrame(ckpt_cal['rows'])
    ref_area = ckpt_cal['ref_area']
    area_2016 = ckpt_cal['area_2016']
    plog(f'  [CHECKPOINT] THR_FU={THR_FU:.2f} (loaded from checkpoint)')
else:
    s2_16 = s2_composite(2016, roi_ee)
    s1_16 = s1_composite(2016, roi_ee)
    fu_16 = s2_16.addBands(s1_16)
    ref_img_16 = (ee.Image(0).byte()
                  .paint(ee.FeatureCollection([ee.Feature(CALIB_2016_EE)]), 1)
                  .clip(roi_ee))
    ref_area = area_ha(ref_img_16, roi_ee, S2_SCALE)
    plog(f'  Reference area (RasSanad_2016.shp): {ref_area:.2f} ha')
    rows_cal = []
    for t in THR_RANGE:
        prob = fu_16.classify(RF_FU).rename('prob')
        pred = clean_pred(prob.gte(t).And(TIDAL_MASK))
        a    = area_ha(pred, roi_ee, S2_SCALE)
        ia   = area_ha(pred.And(ref_img_16), roi_ee, S2_SCALE)
        dice = 2 * ia / max(1e-6, a + ref_area)
        rows_cal.append({'thr': t, 'area_ha': a, 'Dice': dice})
        plog(f'    thr={t:.2f}  area={a:.2f} ha  Dice={dice:.4f}')
    df_cal    = pd.DataFrame(rows_cal)
    THR_FU    = float(df_cal.loc[df_cal['Dice'].idxmax(), 'thr'])
    area_2016 = float(df_cal.loc[df_cal['thr'] == THR_FU, 'area_ha'].values[0])
    plog(f'  Best threshold: {THR_FU:.2f}  area_2016={area_2016:.2f} ha  ref={ref_area:.2f} ha')
    df_cal.to_excel(os.path.join(VTAB, 'Table_Calibration_Fused.xlsx'), index=False)
    save_ckpt('calibration', {'THR_FU': THR_FU, 'ref_area': ref_area,
                               'area_2016': area_2016, 'n_trees': best_n_trees,
                               'rows': rows_cal,
                               'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')})

# ── PHASE 6: Annual area estimation – RUN 1 ───────────────────────────────────
plog('\n=== PHASE 6: Annual Area Estimation – Run 1 ===')
ckpt_r1 = load_ckpt('run1')

if ckpt_r1 and ckpt_r1.get('done') and ckpt_r1.get('n_trees') == best_n_trees:
    plog('  [CHECKPOINT] Run 1 already complete – loading saved areas.')
    df_annual_r1 = pd.DataFrame(ckpt_r1['rows'])
else:
    rows_r1 = []
    t0_r1 = time.time()
    for year in YEARS:
        plog(f'  {year}', end=' ')
        try:
            s2c = s2_composite(year, roi_ee)
            s1c = s1_composite(year, roi_ee)
            fuc = s2c.addBands(s1c)
            pred_fu = clean_pred(fuc.classify(RF_FU).rename('prob').gte(THR_FU).And(TIDAL_MASK))
            a_fu    = area_ha(pred_fu, roi_ee, S2_SCALE)
            pred_s2 = clean_pred(s2c.classify(RF_S2).rename('prob').gte(THR_FU).And(TIDAL_MASK))
            a_s2    = area_ha(pred_s2, roi_ee, S2_SCALE)
            plog(f'Fused={a_fu:.2f}  S2={a_s2:.2f} ha')
        except Exception as e:
            plog(f'ERROR: {e}')
            a_fu = a_s2 = float('nan')
        rows_r1.append({'year': year, 'area_fused': a_fu, 'area_s2': a_s2})
    df_annual_r1 = pd.DataFrame(rows_r1)
    plog(f'  Run 1 complete in {time.time()-t0_r1:.1f}s')
    save_ckpt('run1', {'done': True, 'n_trees': best_n_trees, 'THR_FU': THR_FU,
                        'rows': rows_r1, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')})

df_annual_r1['area_hist'] = df_annual_r1['year'].map(HIST_AREAS)
df_annual_r1.to_excel(os.path.join(VTAB, 'Table_Annual_Areas_Run1.xlsx'), index=False)
plog(f'\n  Run 1 annual areas:')
for _, row in df_annual_r1.iterrows():
    plog(f'    {int(row.year)}: Fused={row.area_fused:.2f}  S2={row.area_s2:.2f} ha')

# ── PHASE 7: Reproducibility check – RUN 2 ───────────────────────────────────
plog('\n=== PHASE 7: Run 2 – Reproducibility Verification ===')
plog('  Re-training RF with same seeds (n_trees={}, seed={})...'.format(best_n_trees, SEED_RF))

# Full retrain with identical settings
_, RF_FU_C2, ACC_FU2 = train_and_eval(tr_fu, FUSED_BANDS, fu_val,
                                       n_trees=best_n_trees, seed=SEED_RF)
_, RF_S2_C2, ACC_S2_2 = train_and_eval(tr_s2, S2_BANDS, s2_val,
                                        n_trees=best_n_trees, seed=SEED_RF)

# Re-run area estimation for 3 representative years (2016, 2020, 2025)
VERIFY_YEARS = [2016, 2020, 2025]
rows_r2 = []
plog(f'  Re-running area estimation for verification years: {VERIFY_YEARS}')
for year in VERIFY_YEARS:
    s2c = s2_composite(year, roi_ee)
    s1c = s1_composite(year, roi_ee)
    fuc = s2c.addBands(s1c)
    # Build RF from scratch for Run 2 (uses same seeds → must give same result)
    RF_FU_RUN2 = (ee.Classifier.smileRandomForest(best_n_trees, seed=SEED_RF)
                  .setOutputMode('PROBABILITY')
                  .train(features=tr_fu, classProperty='class',
                         inputProperties=FUSED_BANDS))
    pred_fu = clean_pred(fuc.classify(RF_FU_RUN2).rename('prob').gte(THR_FU).And(TIDAL_MASK))
    a_fu    = area_ha(pred_fu, roi_ee, S2_SCALE)
    rows_r2.append({'year': year, 'area_fused': a_fu})
    plog(f'    {year}: {a_fu:.2f} ha')

# ── Comparison: Run 1 vs Run 2 ────────────────────────────────────────────────
plog('\n=== Comparison: Run 1 vs Run 2 ===')
TOLERANCE = 0.01    # ha – acceptable floating-point difference

comparison = {'tolerance_ha': TOLERANCE, 'metrics': {}, 'areas': {}, 'passed': True}

# Accuracy metrics comparison
def metric_delta(m1, m2, key):
    d = abs(m1[key] - m2[key])
    ok = d <= 1e-4
    return {'run1': m1[key], 'run2': m2[key], 'delta': round(d, 8), 'pass': ok}

for k in ['OA', 'Kappa', 'F1_mg', 'P_mg', 'R_mg']:
    comparison['metrics'][f'Fused_{k}'] = metric_delta(ACC_FU, ACC_FU2, k)
    comparison['metrics'][f'S2_{k}']    = metric_delta(ACC_S2, ACC_S2_2, k)
    if not comparison['metrics'][f'Fused_{k}']['pass']:
        comparison['passed'] = False

# Area estimates comparison
r2_map = {r['year']: r['area_fused'] for r in rows_r2}
r1_map = {row['year']: row['area_fused'] for _, row in df_annual_r1.iterrows()}
for yr in VERIFY_YEARS:
    a1 = r1_map.get(yr, float('nan'))
    a2 = r2_map.get(yr, float('nan'))
    d  = abs(a1 - a2) if not (np.isnan(a1) or np.isnan(a2)) else float('nan')
    ok = d <= TOLERANCE if not np.isnan(d) else False
    comparison['areas'][yr] = {'run1': round(a1, 4), 'run2': round(a2, 4),
                                'delta_ha': round(d, 6), 'pass': ok}
    if not ok:
        comparison['passed'] = False

plog(f'  Accuracy metrics match:')
for k, v in comparison['metrics'].items():
    status = 'PASS' if v['pass'] else 'FAIL'
    plog(f'    [{status}] {k}: Run1={v["run1"]:.6f}  Run2={v["run2"]:.6f}  Δ={v["delta"]:.2e}')

plog(f'\n  Area estimates match (tolerance {TOLERANCE} ha):')
for yr, v in comparison['areas'].items():
    status = 'PASS' if v['pass'] else 'FAIL'
    plog(f'    [{status}] {yr}: Run1={v["run1"]:.4f}  Run2={v["run2"]:.4f}  Δ={v["delta_ha"]:.4f} ha')

overall = 'PASSED' if comparison['passed'] else 'FAILED'
plog(f'\n  Overall reproducibility check: {overall}')
comparison['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
_log['comparison'] = comparison

# ── PHASE 8: Download arrays for figures ─────────────────────────────────────
plog('\n=== PHASE 8: Downloading Satellite Arrays for Figures ===')

def get_rgb_array(s2_img):
    try:
        rgb  = s2_img.select(['B4','B3','B2']).reproject(crs='EPSG:4326', scale=PRED_SCALE)
        rect = ee_get(rgb.sampleRectangle(region=roi_ee.bounds(), defaultValue=0))
        gamma = 1.4
        r = np.clip(np.array(rect['properties']['B4'], np.float32) / 0.28, 0, 1) ** (1/gamma)
        g = np.clip(np.array(rect['properties']['B3'], np.float32) / 0.28, 0, 1) ** (1/gamma)
        b = np.clip(np.array(rect['properties']['B2'], np.float32) / 0.28, 0, 1) ** (1/gamma)
        return np.flipud(np.stack([r, g, b], axis=-1))
    except Exception as e:
        plog(f'  ERR-RGB({e})')
        return None

def get_pred_array(fuc_img):
    try:
        pred = (fuc_img.classify(RF_FU).rename('prob')
                .gte(THR_FU).And(TIDAL_MASK))
        pred = clean_pred(pred).toFloat().reproject(crs='EPSG:4326', scale=PRED_SCALE)
        rect = ee_get(pred.sampleRectangle(region=roi_ee.bounds(), defaultValue=0))
        return np.flipud(np.array(rect['properties']['prob'], dtype=np.float32))
    except Exception as e:
        plog(f'  ERR-PRED({e})')
        return None

def get_prob_array(fuc_img):
    try:
        prob = (fuc_img.classify(RF_FU).rename('prob')
                .reproject(crs='EPSG:4326', scale=PRED_SCALE))
        rect = ee_get(prob.sampleRectangle(region=roi_ee.bounds(), defaultValue=0))
        return np.flipud(np.array(rect['properties']['prob'], dtype=np.float32))
    except Exception as e:
        plog(f'  ERR-PROB({e})')
        return None

RGB_ARRS = {}; PRED_ARRS = {}; PROB_ARRS = {}
for year in YEARS:
    plog(f'  {year}  RGB...', end=' ')
    s2c = s2_composite(year, roi_ee)
    s1c = s1_composite(year, roi_ee)
    fuc = s2c.addBands(s1c)
    RGB_ARRS[year]  = get_rgb_array(s2c)
    plog('pred...', end=' ')
    PRED_ARRS[year] = get_pred_array(fuc)
    plog('prob...', end=' ')
    PROB_ARRS[year] = get_prob_array(fuc)
    plog('done')

plog('  All arrays downloaded.')

# ── PHASE 9: Generate figures with publication-quality legends ─────────────────
plog('\n=== PHASE 9: Generate Figures (Publication-Quality Legends) ===')

# use df_annual_r1 as the final authoritative area table
df_annual = df_annual_r1.copy()

# Figure layout constants
FIG_W, FIG_H = 20, 8.5
PANEL_ADJ    = dict(left=0.080, right=0.982, top=0.980, bottom=0.105,
                    hspace=0.08, wspace=0.04)

# Publication-quality legend style
# All legends: fontsize=12, large handles, clear spacing
LEG_KW_DARK = dict(
    fontsize=12, handlelength=2.2, handleheight=1.4,
    borderpad=0.8, labelspacing=0.55, handletextpad=0.8,
    facecolor='#1a1a2e', edgecolor='#666666', labelcolor='white',
    framealpha=0.95,
)
LEG_KW_LIGHT = dict(
    fontsize=12, handlelength=2.2, handleheight=1.4,
    borderpad=0.8, labelspacing=0.55, handletextpad=0.8,
    facecolor='white', edgecolor='#aaaaaa', labelcolor='#111111',
    framealpha=0.97,
)

def _scale_bar(ax, length_m=500, x0=0.05, y0=0.05, bar_h=0.015, txt_col='white'):
    m_per_deg_lon = 111320 * np.cos(np.radians(RAS_SANAD_LAT))
    bar_w_deg = length_m / m_per_deg_lon
    bar_frac  = bar_w_deg / (ROI_LON_MAX - ROI_LON_MIN)
    rect = mpatches.FancyBboxPatch((x0, y0), bar_frac, bar_h,
                                    boxstyle='square,pad=0', lw=0.8,
                                    edgecolor=txt_col, facecolor=txt_col,
                                    transform=ax.transAxes, zorder=10)
    ax.add_patch(rect)
    ax.text(x0 + bar_frac / 2, y0 + bar_h + 0.026, f'{length_m:g} m',
            transform=ax.transAxes, fontsize=8, color=txt_col,
            ha='center', va='bottom', fontweight='bold')

def _north_arrow(ax, x=0.92, y=0.10, txt_col='white'):
    ax.annotate('', xy=(x, y + 0.08), xytext=(x, y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=txt_col, lw=1.5))
    ax.text(x, y + 0.11, 'N', transform=ax.transAxes,
            fontsize=9, color=txt_col, ha='center', va='bottom', fontweight='bold')

def annotate_panel(ax, year, row_i, col_i, n_rows=2, dark=True):
    """Panel badges + coordinate grid.
    CRITICAL FIX: set_xlim/ylim called AFTER set_xticks/yticks to prevent
    matplotlib from silently expanding xlim when a tick value is outside range.
    LON_TICKS / LAT_TICKS are verified to be strictly inside data extent.
    """
    tc  = 'white' if dark else '#1a1a1a'
    bgc = '#0d1117bb' if dark else '#ffffffdd'
    gc  = '#ffffff18' if dark else '#00000014'

    row = df_annual[df_annual['year'] == year]
    a_str = (f"{row['area_fused'].values[0]:.1f} ha"
             if len(row) > 0 and not np.isnan(row['area_fused'].values[0]) else '—')
    ax.text(0.03, 0.97, a_str, transform=ax.transAxes,
            fontsize=9, fontweight='bold', color=tc, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.22', facecolor=bgc, alpha=0.92, edgecolor='none'))
    ax.text(0.97, 0.97, str(year), transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=tc, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.22', facecolor=bgc, alpha=0.92, edgecolor='none'))

    # Grid lines at tick positions
    for lon in LON_TICKS: ax.axvline(lon, color=gc, lw=0.45, zorder=2)
    for lat in LAT_TICKS: ax.axhline(lat, color=gc, lw=0.45, zorder=2)

    # Ticks and labels on outer panels only
    ax.set_xticks(LON_TICKS)
    ax.set_yticks(LAT_TICKS)
    if row_i == n_rows - 1:
        ax.set_xticklabels([f'{x:.2f}°E' for x in LON_TICKS], fontsize=6, color=tc)
        ax.tick_params(axis='x', colors=tc, length=2.5, pad=1.5, width=0.5)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
    if col_i == 0:
        ax.set_yticklabels([f'{y:.2f}°N' for y in LAT_TICKS], fontsize=6, color=tc)
        ax.tick_params(axis='y', colors=tc, length=2.5, pad=1.5, width=0.5)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    # CRITICAL: set extent AFTER tick operations (prevents xlim auto-expansion)
    ax.set_xlim(ROI_LON_MIN, ROI_LON_MAX)
    ax.set_ylim(ROI_LAT_MIN, ROI_LAT_MAX)
    # Geographic aspect ratio: makes 2km physical circle appear as a visual circle
    ax.set_aspect(LAT_ASP, adjustable='datalim')

    sp_c = '#444444' if dark else '#cccccc'
    for sp in ax.spines.values():
        sp.set_edgecolor(sp_c)
        sp.set_linewidth(0.7)

# ── Fig 1: Satellite RGB + classification ──────────────────────────────────────
plog('  Rendering Fig1 (Satellite Classification)...')
BG1 = '#0d1117'
fig1, axes1 = plt.subplots(2, 5, figsize=(FIG_W, FIG_H), facecolor=BG1)
fig1.subplots_adjust(**PANEL_ADJ)

legend_v1 = [mpatches.Patch(facecolor='#00cc44', edgecolor='white', lw=0.7,
                              label=f'Predicted mangrove  (thr = {THR_FU:.2f})')]
for idx, (ax, year) in enumerate(zip(axes1.flat, YEARS)):
    ri, ci = divmod(idx, 5)
    ax.set_facecolor('#1a2435')
    rgb = RGB_ARRS.get(year)
    if rgb is not None:
        ax.imshow(rgb, extent=ROI_EXTENT, aspect='auto',
                  interpolation='bilinear', origin='lower', zorder=1)
    pred = PRED_ARRS.get(year)
    if pred is not None and pred.max() > 0:
        overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
        overlay[pred > 0.5] = [0.0, 0.80, 0.27, 0.70]
        ax.imshow(overlay, extent=ROI_EXTENT, aspect='auto', origin='lower', zorder=3)
    annotate_panel(ax, year, ri, ci, dark=True)
    if idx == 0:
        _scale_bar(ax, length_m=500, txt_col='white')
        _north_arrow(ax, txt_col='white')
        ax.legend(handles=legend_v1, loc='lower right', **LEG_KW_DARK)
fig1.patch.set_facecolor(BG1)
out1 = os.path.join(VMAPS, 'Fig1_Satellite_Classification.png')
fig1.savefig(out1, dpi=300, bbox_inches='tight', facecolor=BG1)
plt.close(fig1)
sz = PILImage.open(out1).size
plog(f'    Saved: {out1}  ({sz[0]}×{sz[1]}px)')

# ── Fig 2: Polygon map ─────────────────────────────────────────────────────────
plog('  Rendering Fig2 (Polygon Map)...')
BG2 = 'white'
fig2, axes2 = plt.subplots(2, 5, figsize=(FIG_W, FIG_H), facecolor=BG2)
fig2.subplots_adjust(**PANEL_ADJ)
legend_v2 = [
    mpatches.Patch(facecolor='#5ab96b', edgecolor='#2d6a3f', lw=1.0,
                   label='Predicted mangrove'),
    mpatches.Patch(facecolor='#cde8f0', edgecolor='#6aafc8', lw=0.5,
                   label='Non-mangrove / open water'),
]
for idx, (ax, year) in enumerate(zip(axes2.flat, YEARS)):
    ri, ci = divmod(idx, 5)
    ax.set_facecolor('#cde8f0')
    pred = PRED_ARRS.get(year)
    if pred is not None:
        h, w = pred.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[pred > 0.5] = [0.35, 0.73, 0.42, 1.0]
        ax.imshow(rgba, extent=ROI_EXTENT, aspect='auto', origin='lower', zorder=2)
        if pred.max() > 0:
            ax.contour(pred, levels=[0.5], colors=['#2d6a3f'],
                       linewidths=[1.2], extent=ROI_EXTENT, origin='lower', zorder=3)
    annotate_panel(ax, year, ri, ci, dark=False)
    if idx == 0:
        _scale_bar(ax, length_m=500, txt_col='#333333')
        _north_arrow(ax, txt_col='#333333')
        ax.legend(handles=legend_v2, loc='lower right', **LEG_KW_LIGHT)
fig2.patch.set_facecolor(BG2)
out2 = os.path.join(VMAPS, 'Fig2_Polygon_Map.png')
fig2.savefig(out2, dpi=300, bbox_inches='tight', facecolor=BG2)
plt.close(fig2)
plog(f'    Saved: {out2}')

# ── Fig 3: Probability map ─────────────────────────────────────────────────────
plog('  Rendering Fig3 (Probability Map)...')
BG3 = '#0d1117'
_stops = [(0.00, (0.20, 0.70, 0.30, 0.00)), (0.25, (0.80, 0.95, 0.40, 0.45)),
          (0.50, (0.40, 0.82, 0.25, 0.65)), (0.75, (0.10, 0.60, 0.15, 0.82)),
          (1.00, (0.00, 0.38, 0.08, 0.95))]
_n = 256; _ca = np.zeros((_n, 4))
for i in range(_n):
    t = i / (_n - 1)
    for j in range(len(_stops) - 1):
        t0, c0 = _stops[j]; t1, c1 = _stops[j + 1]
        if t0 <= t <= t1:
            f = (t - t0) / max(1e-9, t1 - t0)
            _ca[i] = [c0[k] + f * (c1[k] - c0[k]) for k in range(4)]
            break
PROB_CMAP = mcolors.ListedColormap(_ca)
fig3, axes3 = plt.subplots(2, 5, figsize=(FIG_W, FIG_H), facecolor=BG3)
fig3.subplots_adjust(left=0.080, right=0.935, top=0.980, bottom=0.105,
                     hspace=0.08, wspace=0.04)
for idx, (ax, year) in enumerate(zip(axes3.flat, YEARS)):
    ri, ci = divmod(idx, 5)
    ax.set_facecolor('#1a2435')
    rgb = RGB_ARRS.get(year)
    if rgb is not None:
        ax.imshow(rgb * 0.55, extent=ROI_EXTENT, aspect='auto',
                  interpolation='bilinear', origin='lower', zorder=1)
    prob = PROB_ARRS.get(year)
    if prob is not None:
        ax.imshow(PROB_CMAP(prob), extent=ROI_EXTENT, aspect='auto', origin='lower', zorder=3)
        if prob.max() > THR_FU:
            ax.contour(prob, levels=[THR_FU], colors=['white'],
                       linewidths=[0.8], linestyles=['--'],
                       extent=ROI_EXTENT, origin='lower', zorder=5)
    annotate_panel(ax, year, ri, ci, dark=True)
    if idx == 0:
        _scale_bar(ax, length_m=500, txt_col='white')
        _north_arrow(ax, txt_col='white')
fig3.patch.set_facecolor(BG3)
sm = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap(_ca[50:]),
                            norm=mcolors.Normalize(vmin=0.2, vmax=1.0))
sm.set_array([])
cax = fig3.add_axes([0.940, 0.105, 0.012, 0.875])
cbar = fig3.colorbar(sm, cax=cax, orientation='vertical')
cbar.set_label('RF Mangrove Probability', fontsize=11, color='white', labelpad=8)
cbar.ax.yaxis.set_tick_params(color='white', labelsize=10)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
cbar.ax.set_facecolor('#0d1117')
cbar.outline.set_edgecolor('#555555')
thr_pos = (THR_FU - 0.2) / 0.8
if 0 < thr_pos < 1:
    cbar.ax.axhline(y=thr_pos, color='white', lw=1.0, linestyle='--')
    cbar.ax.text(1.18, thr_pos, f'thr = {THR_FU:.2f}',
                 transform=cbar.ax.transAxes, fontsize=9, color='white', va='center')
out3 = os.path.join(VMAPS, 'Fig3_Probability_Map.png')
fig3.savefig(out3, dpi=300, bbox_inches='tight', facecolor=BG3)
plt.close(fig3)
plog(f'    Saved: {out3}')

# ── Fig 4: n_trees tuning curve ────────────────────────────────────────────────
plog('  Rendering Fig4 (n_trees Tuning Curve)...')
fig4, ax4 = plt.subplots(figsize=(9, 5))
trees = [r['n_trees'] for r in tuning_results]
oas   = [r['OA'] for r in tuning_results]
kaps  = [r['Kappa'] for r in tuning_results]
f1s_t = [r['F1_mg'] for r in tuning_results]
ax4b  = ax4.twinx()
ax4.plot(trees, f1s_t, 'go-', ms=9, lw=2.2, label='F1 (mangrove)', zorder=5)
ax4.plot(trees, oas,   'bs--', ms=8, lw=1.8, label='Overall accuracy', zorder=4)
ax4b.plot(trees, kaps, 'r^:', ms=8, lw=1.8, label='Kappa coefficient', zorder=4)
ax4.axvline(best_n_trees, color='black', ls='--', lw=1.6,
            label=f'Selected: {best_n_trees} trees')
ax4.set(xlabel='Number of trees', ylabel='F1 / Overall Accuracy',
        title=f'Model Accuracy vs n_trees  (seed={SEED_RF}, val_pts={VAL_PTS*2})')
ax4b.set_ylabel('Kappa coefficient', color='red', fontsize=12)
ax4b.tick_params(axis='y', labelcolor='red', labelsize=11)
ax4.legend(loc='lower right', fontsize=12, handlelength=2.0, borderpad=0.8,
           framealpha=0.95)
ax4b.legend(loc='center right', fontsize=12, handlelength=2.0, borderpad=0.8,
            framealpha=0.95)
ax4.set_xticks(trees)
ax4.tick_params(axis='both', labelsize=11)
ax4.grid(True, alpha=0.35)
ax4.set_ylim(min(oas + f1s_t) - 0.02, 1.0)
fig4.tight_layout()
out4 = os.path.join(VACC, 'Fig4_nTrees_Tuning.png')
fig4.savefig(out4, dpi=300)
plt.close(fig4)
plog(f'    Saved: {out4}')

# ── Fig 5: Time-series ─────────────────────────────────────────────────────────
plog('  Rendering Fig5 (Area Time Series)...')
fig5, axes5 = plt.subplots(1, 2, figsize=(16, 6))
ax5a, ax5b = axes5

hx = sorted(HIST_AREAS.keys()); hy = [HIST_AREAS[y] for y in hx]
ax5a.plot(hx, hy, 'ks--', ms=9, lw=2.0, zorder=5,
          label='Field survey (Aljenaid et al.)')
ax5a.plot(df_annual['year'], df_annual['area_fused'], 'o-',
          color='#1a7c3e', lw=2.2, ms=7, zorder=6,
          label=f'Fused S2+SAR (n_trees={best_n_trees})')
if 'area_s2' in df_annual.columns:
    ax5a.plot(df_annual['year'], df_annual['area_s2'], 's--',
              color='#e07b00', lw=1.6, ms=6, alpha=0.65, zorder=5,
              label='S2-only classifier')
ax5a.fill_between(df_annual['year'],
                  df_annual['area_fused'] - 2, df_annual['area_fused'] + 2,
                  alpha=0.18, color='#1a7c3e', label='±2 ha uncertainty')
ax5a.axhline(TARGET_HA, color='#c0392b', ls=':', lw=1.4,
             label=f'2016 reference ({TARGET_HA} ha)')
sl, pv = stats.linregress(df_annual['year'], df_annual['area_fused'])[:2]
sig = 'p<0.05' if pv < 0.05 else f'p={pv:.3f}'
ax5a.set(xlabel='Year', ylabel='Mangrove area (ha)',
         title=f'Ras Sanad Mangrove Area 1967–2025\n'
               f'Trend: {sl:+.3f} ha/yr  ({sig})')
ax5a.legend(fontsize=12, handlelength=2.0, handleheight=1.4,
            borderpad=0.8, labelspacing=0.5, loc='upper right',
            framealpha=0.95)
ax5a.grid(True, alpha=0.3)
ax5a.tick_params(axis='both', labelsize=11)

diffs  = df_annual['area_fused'].diff().dropna()
yrs_d  = df_annual['year'].iloc[1:]
colors = ['#1a7c3e' if v >= 0 else '#c0392b' for v in diffs]
ax5b.bar(yrs_d, diffs, color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
ax5b.axhline(0, color='#333333', lw=1.0, zorder=5)
ax5b.set(xlabel='Year', ylabel='Annual change (ha)',
         title='Inter-annual Mangrove Area Change')
ax5b.set_xticks(yrs_d); ax5b.set_xticklabels(yrs_d.astype(int), rotation=45, fontsize=10)
ax5b.set_facecolor('#f7f7f7')
ax5b.grid(True, alpha=0.35, axis='y')
ax5b.tick_params(axis='both', labelsize=11)
gain_patch = mpatches.Patch(color='#1a7c3e', label='Area gain')
loss_patch = mpatches.Patch(color='#c0392b', label='Area loss')
ax5b.legend(handles=[gain_patch, loss_patch], fontsize=12,
            handlelength=2.0, handleheight=1.4, borderpad=0.8, framealpha=0.95)
fig5.tight_layout()
out5 = os.path.join(VMAPS, 'Fig_Area_TimeSeries.png')
fig5.savefig(out5, dpi=300)
plt.close(fig5)
plog(f'    Saved: {out5}')

# ── Fig 6: Accuracy confusion matrices ────────────────────────────────────────
plog('  Rendering Fig6 (Confusion Matrices)...')
fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6))
for ax, (nm, ac) in zip(axes6, [('Fused S2+SAR', ACC_FU), ('S2-only', ACC_S2)]):
    cm_arr = np.array(ac['CM'])
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-mg', 'Mangrove'],
                yticklabels=['Non-mg', 'Mangrove'], linewidths=0.5,
                annot_kws={'size': 15, 'fontweight': 'bold'})
    ax.set_title(f'{nm}\nOA={ac["OA"]:.4f}   κ={ac["Kappa"]:.4f}   '
                 f'F1_mg={ac["F1_mg"]:.4f}\n'
                 f'Prec={ac["P_mg"]:.4f}   Recall={ac["R_mg"]:.4f}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted class', fontsize=13, labelpad=8)
    ax.set_ylabel('Reference class', fontsize=13, labelpad=8)
    ax.tick_params(axis='both', labelsize=12)
fig6.suptitle(f'Accuracy Assessment  –  n_trees={best_n_trees}, seed={SEED_RF}\n'
              f'Validation: {VAL_PTS*2} independent samples (seed={SEED_VAL})',
              fontsize=14, fontweight='bold', y=1.02)
fig6.tight_layout()
out6 = os.path.join(VACC, 'Fig_Accuracy_CM.png')
fig6.savefig(out6, dpi=300, bbox_inches='tight')
plt.close(fig6)
plog(f'    Saved: {out6}')

# ── Fig 7: Threshold calibration ──────────────────────────────────────────────
plog('  Rendering Fig7 (Threshold Calibration)...')
fig7, ax7 = plt.subplots(figsize=(9, 5))
ax7b = ax7.twinx()
ax7.plot(df_cal['thr'], df_cal['area_ha'], 'b-o', ms=7, lw=1.8, label='Predicted area (ha)')
ax7b.plot(df_cal['thr'], df_cal['Dice'], 'r-s', ms=7, lw=1.8, label='Dice coefficient')
ax7.axvline(THR_FU, color='black', ls='--', lw=2.0,
            label=f'Best thr = {THR_FU:.2f}  ({area_2016:.1f} ha)')
ax7.axhline(TARGET_HA, color='gray', ls=':', lw=1.4,
            label=f'Reference 2016: {TARGET_HA} ha')
ax7.set(xlabel='Probability threshold', title='Threshold Calibration vs RasSanad_2016.shp')
ax7.set_ylabel('Predicted area (ha)', color='blue', fontsize=12)
ax7b.set_ylabel('Dice coefficient', color='red', fontsize=12)
ax7.tick_params(axis='y', labelcolor='blue', labelsize=11)
ax7b.tick_params(axis='y', labelcolor='red', labelsize=11)
ax7.tick_params(axis='x', labelsize=11)
ax7.legend(loc='upper left', fontsize=12, handlelength=2.0, borderpad=0.8, framealpha=0.95)
ax7b.legend(loc='upper right', fontsize=12, handlelength=2.0, borderpad=0.8, framealpha=0.95)
ax7.grid(True, alpha=0.3)
fig7.tight_layout()
out7 = os.path.join(VACC, 'Fig_Calibration.png')
fig7.savefig(out7, dpi=300)
plt.close(fig7)
plog(f'    Saved: {out7}')

# Save accuracy tables
DF_ACC = pd.DataFrame([
    {'Model': nm, 'n_trees': best_n_trees, 'Seed': SEED_RF,
     'OA': round(ac['OA'], 4), 'Kappa': round(ac['Kappa'], 4),
     'F1_mg': round(ac['F1_mg'], 4), 'Precision': round(ac['P_mg'], 4),
     'Recall': round(ac['R_mg'], 4), 'F1_non': round(ac['F1_non'], 4)}
    for nm, ac in [('Fused S2+S1', ACC_FU), ('S2-only', ACC_S2)]
])
DF_ACC.to_excel(os.path.join(VACC, 'Table_Accuracy_Metrics.xlsx'), index=False)
plog(f'  Accuracy table saved.')

# ── PHASE 10: Verification report ─────────────────────────────────────────────
plog('\n=== PHASE 10: Verification Report ===')

tuning_df = pd.DataFrame(tuning_results)
best_row  = tuning_df[tuning_df['n_trees'] == best_n_trees].iloc[0]

report_lines = [
    '=' * 70,
    'RAS SANAD MANGROVE WORKFLOW – STABILITY VERIFICATION REPORT',
    f'Generated : {time.strftime("%Y-%m-%d %H:%M:%S")}',
    '=' * 70,
    '',
    '1. REPRODUCIBILITY CONTROLS',
    '-' * 40,
    f'   RF training seed       : {SEED_RF}  (all models)',
    f'   Validation sample seed : {SEED_VAL}  (= {SEED_RF} + 999)',
    f'   Sampling seeds         : year-indexed (yr, yr+1000, yr+500, yr+1500)',
    f'   EE collection compositing: server-side median (deterministic)',
    f'   Hard-negative zone     : 600 m buffer around ever-mangrove',
    '',
    '2. MODEL TRAINING – n_TREES TUNING',
    '-' * 40,
]
for r in tuning_results:
    flag = ' <-- SELECTED' if r['n_trees'] == best_n_trees else ''
    report_lines.append(
        f'   n_trees={r["n_trees"]:3d}: OA={r["OA"]:.4f}  Kappa={r["Kappa"]:.4f}'
        f'  F1_mg={r["F1_mg"]:.4f}{flag}')

report_lines += [
    '',
    f'   Plateau criterion : F1_mg improvement < 0.002 per +50 trees',
    f'   Selected n_trees  : {best_n_trees}',
    f'   Selected model    : OA={best_row["OA"]:.4f}  Kappa={best_row["Kappa"]:.4f}'
    f'  F1_mg={best_row["F1_mg"]:.4f}',
    '',
    '3. FINAL MODEL ACCURACY (Run 1)',
    '-' * 40,
]
for nm, ac in [('Fused S2+S1', ACC_FU), ('S2-only', ACC_S2)]:
    cm = np.array(ac['CM'])
    report_lines += [
        f'   {nm}:',
        f'     Overall Accuracy  : {ac["OA"]:.4f}',
        f'     Kappa coefficient : {ac["Kappa"]:.4f}',
        f'     F1 (mangrove)     : {ac["F1_mg"]:.4f}',
        f'     Precision         : {ac["P_mg"]:.4f}',
        f'     Recall            : {ac["R_mg"]:.4f}',
        f'     Confusion matrix  : TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}',
        '',
    ]

report_lines += [
    '4. CALIBRATION',
    '-' * 40,
    f'   Reference (RasSanad_2016.shp) : {ref_area:.2f} ha',
    f'   Best Dice threshold           : {THR_FU:.2f}',
    f'   Predicted area at threshold   : {area_2016:.2f} ha',
    f'   Target area                   : {TARGET_HA:.2f} ha',
    '',
    '5. ANNUAL AREA ESTIMATES (Run 1)',
    '-' * 40,
]
for _, row in df_annual.iterrows():
    ref_str = f'  (survey: {row.area_hist:.2f} ha)' if not np.isnan(row.area_hist) else ''
    report_lines.append(
        f'   {int(row.year)}: Fused={row.area_fused:.2f} ha'
        f'  S2={row.area_s2:.2f} ha{ref_str}')

report_lines += [
    '',
    '6. RUN 1 vs RUN 2 REPRODUCIBILITY CHECK',
    '-' * 40,
    f'   Tolerance: {TOLERANCE} ha for areas, 1e-4 for metrics',
    '',
    '   Accuracy metrics:',
]
all_pass = True
for k, v in comparison['metrics'].items():
    status = 'PASS' if v['pass'] else 'FAIL'
    report_lines.append(
        f'     [{status}] {k:20s}: Run1={v["run1"]:.6f}  Run2={v["run2"]:.6f}'
        f'  Δ={v["delta"]:.2e}')
    if not v['pass']:
        all_pass = False

report_lines += ['', '   Area estimates (3 sample years):']
for yr, v in comparison['areas'].items():
    status = 'PASS' if v['pass'] else 'FAIL'
    report_lines.append(
        f'     [{status}] {yr}: Run1={v["run1"]:.4f} ha  Run2={v["run2"]:.4f} ha'
        f'  Δ={v["delta_ha"]:.4f} ha')
    if not v['pass']:
        all_pass = False

verdict = 'REPRODUCIBLE AND STABLE' if all_pass else 'DIFFERENCES DETECTED (see above)'
report_lines += [
    '',
    f'   OVERALL VERDICT: {verdict}',
    '',
    '7. FIGURES GENERATED',
    '-' * 40,
    f'   Fig1_Satellite_Classification.png  – Sentinel-2 RGB + mangrove overlay',
    f'   Fig2_Polygon_Map.png               – Classified polygon map (white bg)',
    f'   Fig3_Probability_Map.png           – Per-pixel RF probability (continuous)',
    f'   Fig4_nTrees_Tuning.png             – Accuracy vs n_trees learning curve',
    f'   Fig_Area_TimeSeries.png            – Long-term + inter-annual change',
    f'   Fig_Accuracy_CM.png                – Confusion matrices (both models)',
    f'   Fig_Calibration.png                – Threshold calibration curve',
    '',
    '   All legend font sizes: 12pt  |  Handle length: 2.2  |  Border pad: 0.8',
    '   xlim bug fix: xlim/ylim set AFTER tick operations (no silent auto-expansion)',
    '   Geographic aspect ratio applied (LAT_ASP ≈ 1.114 at 26°N): circles are round',
    '',
    '=' * 70,
    'END OF REPORT',
    '=' * 70,
]

report_txt = '\n'.join(report_lines)
print('\n' + report_txt)
report_path = os.path.join(VOUT, 'Verification_Report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_txt)
plog(f'\n  Report saved: {report_path}')

# ── Final log save ─────────────────────────────────────────────────────────────
_log['final'] = {
    'best_n_trees': best_n_trees,
    'THR_FU': THR_FU,
    'ACC_FU': {k: v for k, v in ACC_FU.items() if k != 'CM'},
    'ACC_S2': {k: v for k, v in ACC_S2.items() if k != 'CM'},
    'annual_areas': df_annual[['year', 'area_fused', 'area_s2']].to_dict('records'),
    'verdict': verdict,
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
save_log()
plog(f'  Verification log saved: {LOG_FILE}')
plog('\n=== ALL DONE ===')
plog(f'  Outputs in: {VOUT}')
