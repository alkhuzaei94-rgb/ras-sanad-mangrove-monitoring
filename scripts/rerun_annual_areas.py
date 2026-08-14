"""
_rerun_annual_areas.py
======================
Regenerates the full results chain on the CURRENT GEE collection state, so the
resubmission can adopt one internally consistent August 2026 run:

  1. Training data + both RF models (fused, S2-only), production seeds.
  2. Validation on the 2023 composite (1000 pts, 500/class, seed 1041):
     confusion matrices + metrics for both models.
  3. Annual areas 2016-2025 for both models at T = 0.90 (published) and
     T = 0.92 (DSC optimum from the extended sweep) - a 2x2 sensitivity grid.

Output: Ras_Sanad_Verification/tables/Rerun_2026-08_Annual_Areas.xlsx + .json

Run:
  & "C:\\ProgramData\\anaconda3\\envs\\geoai_rs\\python.exe" _rerun_annual_areas.py
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import geopandas as gpd
import ee

BASE    = r'C:\path\to\workspace'
SHP_DIR = os.path.join(BASE, '_archive_root_duplicates')
VTAB    = os.path.join(BASE, 'mangrove', 'Ras_Sanad_Verification', 'tables')
os.makedirs(VTAB, exist_ok=True)

SEED_RF, SEED_VAL = 42, 1041
MAX_PTS, VAL_PTS  = 2000, 500
N_TREES   = 150
S2_SCALE  = 10
YEARS       = list(range(2016, 2026))
TRAIN_YEARS = [2017, 2018, 2019, 2021, 2022]
THRESHOLDS  = [0.90, 0.92]

S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
            'NDVI','NDMI','MNDWI','EVI','NBR']
S1_BANDS = ['VV','VH','VV_VH']
FUSED_BANDS = S2_BANDS + S1_BANDS

def plog(msg, end='\n'):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', end=end, flush=True)

def ee_get(obj, retries=8, delay=10):
    for i in range(retries):
        try:
            return obj.getInfo()
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay * (i + 1))
            else:
                raise e

plog('=== EE init ===')
ee.Initialize(project=os.environ.get('EE_PROJECT'))  # set EE_PROJECT to your own Cloud project

gdf1967 = gpd.read_file(os.path.join(SHP_DIR, 'RasSanad_1967.shp')).to_crs(4326)
cent = gdf1967.geometry.union_all().centroid
roi_ee = ee.Geometry.Point([round(cent.x, 5), round(cent.y, 5)]).buffer(2000)

def gdf_to_ee_geom(gdf):
    return ee.Geometry(gdf.geometry.union_all().__geo_interface__)

RASSANAD_EE = {yr: gdf_to_ee_geom(
    gpd.read_file(os.path.join(SHP_DIR, f'RasSanad_{yr}.shp')).to_crs(4326))
    for yr in [1967, 1998, 2005, 2009, 2016]}

ever = None
for yr, g in RASSANAD_EE.items():
    ever = g if ever is None else ever.union(g, 1)
EVER_MG_GEOM = ever.intersection(roi_ee, 1)

jrc_water  = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('max_extent').unmask(0).byte()
WATER_PROX = jrc_water.focal_max(radius=500, units='meters', kernelType='circle')
ELEV_MASK  = ee.Image('USGS/SRTMGL1_003').select('elevation').lt(10)
TIDAL_MASK = WATER_PROX.And(ELEV_MASK)

def clean_pred(img):
    return (img.focal_min(radius=10, units='meters')
               .focal_max(radius=10, units='meters'))

label_img = (ee.Image(0).byte()
             .paint(ee.FeatureCollection([ee.Feature(roi_ee.difference(EVER_MG_GEOM, 1))]), 0)
             .paint(ee.FeatureCollection([ee.Feature(EVER_MG_GEOM, {'class': 1})]), 1)
             .rename('class'))

hard_neg_geom = (EVER_MG_GEOM.buffer(600, 1)
                 .intersection(roi_ee, 1)
                 .difference(EVER_MG_GEOM, 1))
HARD_NEG_MASK = ee.Image(0).byte().paint(
    ee.FeatureCollection([ee.Feature(hard_neg_geom)]), 1)

def mask_s2(img):
    scl = img.select('SCL')
    return img.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))

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
        plog(f'    (S1 fallback fired for {year})')
        col = (ee.ImageCollection('COPERNICUS/S1_GRD')
               .filterBounds(region)
               .filter(ee.Filter.calendarRange(year - 1, year + 1, 'year'))
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
               .select(['VV', 'VH']))
    med   = col.median()
    ratio = med.select('VV').subtract(med.select('VH')).rename('VV_VH')
    return med.addBands(ratio).select(S1_BANDS).clip(region)

def area_ha(img, roi, scale):
    a = img.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=roi,
        scale=scale, maxPixels=1e12, bestEffort=True)
    v = ee_get(a)
    val = list(v.values())[0] if v else 0
    return (val or 0) / 10_000

# ── Training data, both models ────────────────────────────────────────────────
plog('=== Training data (both models, 5 years) ===')
all_fu, all_s2 = [], []
for yr in TRAIN_YEARS:
    plog(f'  Year {yr}...', end=' ')
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
    all_fu.append(samp_std_fu.merge(samp_hrd_fu))
    all_s2.append(samp_std_s2.merge(samp_hrd_s2))
    print('queued', flush=True)

tr_fu = all_fu[0]
for s in all_fu[1:]: tr_fu = tr_fu.merge(s)
tr_s2 = all_s2[0]
for s in all_s2[1:]: tr_s2 = tr_s2.merge(s)
n_fu, n_s2 = ee_get(tr_fu.size()), ee_get(tr_s2.size())
plog(f'  Training points: fused {n_fu:,}  S2-only {n_s2:,}  (April: 17,697 / 17,636)')

def make_rf(samples, bands, mode):
    return (ee.Classifier.smileRandomForest(N_TREES, seed=SEED_RF)
            .setOutputMode(mode)
            .train(features=samples, classProperty='class', inputProperties=bands))

RF_FU   = make_rf(tr_fu, FUSED_BANDS, 'PROBABILITY')
RF_S2   = make_rf(tr_s2, S2_BANDS, 'PROBABILITY')
RF_FU_C = make_rf(tr_fu, FUSED_BANDS, 'CLASSIFICATION')
RF_S2_C = make_rf(tr_s2, S2_BANDS, 'CLASSIFICATION')

# ── Validation on 2023 (production design) ────────────────────────────────────
plog('=== Validation (2023 composite, 500/class, seed 1041) ===')
s2_val = s2_composite(2023, roi_ee)
s1_val = s1_composite(2023, roi_ee)
fu_val = s2_val.addBands(s1_val)

val_results = {}
for name, val_img, bands, clf in [
        ('Fused',   fu_val, FUSED_BANDS, RF_FU_C),
        ('S2-only', s2_val, S2_BANDS,    RF_S2_C)]:
    val_samp = (val_img.select(bands).addBands(label_img)
                .stratifiedSample(numPoints=VAL_PTS, classBand='class',
                                  region=roi_ee, scale=S2_SCALE, seed=SEED_VAL,
                                  geometries=False,
                                  classValues=[0, 1], classPoints=[VAL_PTS, VAL_PTS]))
    cm  = val_samp.classify(clf).errorMatrix('class', 'classification')
    mat = np.array(ee_get(cm))
    oa  = ee_get(cm.accuracy())
    kap = ee_get(cm.kappa())
    tn, fp = int(mat[0, 0]), int(mat[0, 1])
    fn, tp = int(mat[1, 0]), int(mat[1, 1])
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    val_results[name] = {'CM': [[tn, fp], [fn, tp]], 'OA': round(oa, 4),
                         'Kappa': round(kap, 4), 'Precision': round(pr, 4),
                         'Recall': round(rc, 4), 'F1_mg': round(f1, 4)}
    plog(f'  {name}: CM=[[{tn},{fp}],[{fn},{tp}]]  OA={oa:.4f}  '
         f'kappa={kap:.4f}  F1={f1:.4f}')
plog('  (April: Fused [[381,119],[90,410]] OA=0.791; '
     'S2-only [[381,119],[119,381]] OA=0.762)')

# ── Annual areas, both models, both thresholds ────────────────────────────────
plog('=== Annual areas 2016-2025 (2 models x 2 thresholds) ===')
rows = []
for year in YEARS:
    plog(f'  {year}...', end=' ')
    s2c = s2_composite(year, roi_ee)
    s1c = s1_composite(year, roi_ee)
    fuc = s2c.addBands(s1c)
    prob_fu = fuc.classify(RF_FU).rename('prob')
    prob_s2 = s2c.classify(RF_S2).rename('prob')
    row = {'year': year}
    for t in THRESHOLDS:
        a_fu = area_ha(clean_pred(prob_fu.gte(t).And(TIDAL_MASK)), roi_ee, S2_SCALE)
        a_s2 = area_ha(clean_pred(prob_s2.gte(t).And(TIDAL_MASK)), roi_ee, S2_SCALE)
        row[f'fused_T{t:.2f}'] = round(a_fu, 2)
        row[f's2_T{t:.2f}']    = round(a_s2, 2)
    rows.append(row)
    print('  '.join(f'{k}={v}' for k, v in row.items() if k != 'year'), flush=True)

df = pd.DataFrame(rows)
df.to_excel(os.path.join(VTAB, 'Rerun_2026-08_Annual_Areas.xlsx'), index=False)
out = {
    'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'training_points': {'fused': n_fu, 's2': n_s2},
    'validation_2023': val_results,
    'annual_areas': rows,
    'published_apr_fused_T0.90': [35.24, 39.04, 37.00, 34.83, 35.25,
                                  41.03, 38.59, 37.32, 35.86, 36.42],
}
with open(os.path.join(VTAB, 'Rerun_2026-08_Annual_Areas.json'), 'w') as f:
    json.dump(out, f, indent=2)
plog('Saved Rerun_2026-08_Annual_Areas.xlsx / .json')

pub = out['published_apr_fused_T0.90']
plog('=== Comparison with published Table 5 (fused, T=0.90) ===')
print(f'{"year":>6} {"published":>10} {"new_T0.90":>10} {"delta":>7} {"new_T0.92":>10}')
for i, r in enumerate(rows):
    d = r['fused_T0.90'] - pub[i]
    print(f'{r["year"]:>6} {pub[i]:>10.2f} {r["fused_T0.90"]:>10.2f} '
          f'{d:>+7.2f} {r["fused_T0.92"]:>10.2f}')
mean_new90 = float(np.mean([r['fused_T0.90'] for r in rows]))
mean_new92 = float(np.mean([r['fused_T0.92'] for r in rows]))
plog(f'Means: published 37.06 | new T0.90 {mean_new90:.2f} | new T0.92 {mean_new92:.2f}')
plog('DONE.')
