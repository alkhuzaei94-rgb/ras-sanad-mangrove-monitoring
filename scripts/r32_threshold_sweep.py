"""
_r32_threshold_sweep.py
=======================
Reviewer 3, comment 2: extend the DSC threshold sweep across 0.30-1.00, with
fine 0.01 increments in 0.80-1.00, to establish whether T* = 0.90 is a true
optimum or a boundary artefact.

Method: reproduces the production chain of _verify_stability.py exactly
(same ROI, same shapefiles, same label image, same per-year sampling seeds,
same RF: smileRandomForest 150 trees seed 42 PROBABILITY, same tidal mask
and morphological cleaning), then sweeps the extended threshold set on the
2016 fused composite against the RasSanad_2016.shp reference polygon.

REPRODUCTION GATE: before the new thresholds mean anything, the 13 original
thresholds (0.30-0.90 step 0.05) must match the April 2026 run recorded in
Ras_Sanad_Verification/checkpoints/calibration.json. The script prints a
delta table and flags any |dDice| > 0.005 or |darea| > 0.5 ha.

Output: Ras_Sanad_Verification/tables/Table_Calibration_Extended.xlsx + .json

Run:
  & "C:\\ProgramData\\anaconda3\\envs\\geoai_rs\\python.exe" _r32_threshold_sweep.py
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import geopandas as gpd
import ee

BASE    = r'C:\Users\Manaf\Documents\Jupyter'
SHP_DIR = os.path.join(BASE, '_archive_root_duplicates')
VER     = os.path.join(BASE, 'mangrove', 'Ras_Sanad_Verification')
VTAB    = os.path.join(VER, 'tables')
CKPT    = os.path.join(VER, 'checkpoints', 'calibration.json')
os.makedirs(VTAB, exist_ok=True)

SEED_RF   = 42
MAX_PTS   = 2000
N_TREES   = 150
S2_SCALE  = 10
TRAIN_YEARS = [2017, 2018, 2019, 2021, 2022]

S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
            'NDVI','NDMI','MNDWI','EVI','NBR']
S1_BANDS = ['VV','VH','VV_VH']
FUSED_BANDS = S2_BANDS + S1_BANDS

# Coarse legacy steps below 0.80, fine 0.01 steps 0.80-1.00
THRS = [round(t/100, 2) for t in range(30, 80, 5)] + \
       [round(t/100, 2) for t in range(80, 101, 1)]

def plog(msg, end='\n'):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', end=end, flush=True)

def ee_get(obj, retries=6, delay=5):
    for i in range(retries):
        try:
            return obj.getInfo()
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise e

plog('=== EE init ===')
ee.Initialize(project='ee-mkhuzaei94')

# ── ROI + shapefiles (identical to production) ────────────────────────────────
gdf1967 = gpd.read_file(os.path.join(SHP_DIR, 'RasSanad_1967.shp')).to_crs(4326)
cent = gdf1967.geometry.union_all().centroid
RAS_SANAD_LON, RAS_SANAD_LAT = round(cent.x, 5), round(cent.y, 5)
roi_ee = ee.Geometry.Point([RAS_SANAD_LON, RAS_SANAD_LAT]).buffer(2000)
plog(f'ROI centre ({RAS_SANAD_LON}, {RAS_SANAD_LAT})')

def gdf_to_ee_geom(gdf):
    return ee.Geometry(gdf.geometry.union_all().__geo_interface__)

RASSANAD_EE = {}
for yr in [1967, 1998, 2005, 2009, 2016]:
    RASSANAD_EE[yr] = gdf_to_ee_geom(
        gpd.read_file(os.path.join(SHP_DIR, f'RasSanad_{yr}.shp')).to_crs(4326))

ever = None
for yr, g in RASSANAD_EE.items():
    ever = g if ever is None else ever.union(g, 1)
EVER_MG_GEOM  = ever.intersection(roi_ee, 1)
CALIB_2016_EE = RASSANAD_EE[2016]
plog('Shapefiles + ever-mangrove union ready.')

# ── Masks + label image (identical) ───────────────────────────────────────────
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
    return col.median().select(S2_BANDS).clip(region)

def s1_composite(year, region):
    # EXACT replica of production (_verify_stability.py), including the
    # fallback: if the calendar-year descending collection is empty (true for
    # 2016), widen to year-1..year+1 and drop the orbit filter.
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
        plog(f'    (S1 fallback fired for {year}: +/-1 y window, no orbit filter)')
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

# ── Training data (fused only; identical seeds) ───────────────────────────────
plog('=== Training data (fused, 5 years) ===')
all_fu = []
for yr in TRAIN_YEARS:
    plog(f'  Year {yr}...', end=' ')
    s2c = s2_composite(yr, roi_ee)
    s1c = s1_composite(yr, roi_ee)
    fuc = s2c.addBands(s1c)
    base_fu = fuc.select(FUSED_BANDS).addBands(label_img)
    samp_std = base_fu.stratifiedSample(
        numPoints=MAX_PTS // 2, classBand='class', region=roi_ee,
        scale=S2_SCALE, seed=yr, geometries=False,
        classValues=[0, 1], classPoints=[MAX_PTS // 2, MAX_PTS])
    samp_hrd = (base_fu.updateMask(HARD_NEG_MASK.And(label_img.eq(0)))
                .sample(region=roi_ee, scale=S2_SCALE,
                        numPixels=MAX_PTS // 2, seed=yr + 500, geometries=False))
    samp = samp_std.merge(samp_hrd)
    n = ee_get(samp.size())
    print(f'{n:,} pts', flush=True)
    all_fu.append(samp)

tr_fu = all_fu[0]
for s in all_fu[1:]:
    tr_fu = tr_fu.merge(s)
total = ee_get(tr_fu.size())
plog(f'  Total fused training points: {total:,} (April run: 17,697)')

# ── RF training (identical) ───────────────────────────────────────────────────
plog(f'=== RF training (n_trees={N_TREES}, seed={SEED_RF}, PROBABILITY) ===')
RF_FU = (ee.Classifier.smileRandomForest(N_TREES, seed=SEED_RF)
         .setOutputMode('PROBABILITY')
         .train(features=tr_fu, classProperty='class', inputProperties=FUSED_BANDS))

# ── 2016 composite + reference ────────────────────────────────────────────────
plog('=== 2016 composite + reference polygon ===')
s2_16 = s2_composite(2016, roi_ee)
s1_16 = s1_composite(2016, roi_ee)
fu_16 = s2_16.addBands(s1_16)
ref_img_16 = (ee.Image(0).byte()
              .paint(ee.FeatureCollection([ee.Feature(CALIB_2016_EE)]), 1)
              .clip(roi_ee))
ref_area = area_ha(ref_img_16, roi_ee, S2_SCALE)
plog(f'Reference area: {ref_area:.3f} ha (April run: 34.122)')

prob = fu_16.classify(RF_FU).rename('prob')

# ── Extended sweep ────────────────────────────────────────────────────────────
plog(f'=== Sweep: {len(THRS)} thresholds ===')
rows = []
for t in THRS:
    pred = clean_pred(prob.gte(t).And(TIDAL_MASK))
    a    = area_ha(pred, roi_ee, S2_SCALE)
    ia   = area_ha(pred.And(ref_img_16), roi_ee, S2_SCALE)
    dice = 2 * ia / max(1e-6, a + ref_area)
    rows.append({'thr': t, 'area_ha': a, 'intersect_ha': ia, 'Dice': dice})
    plog(f'  thr={t:.2f}  area={a:8.2f} ha  Dice={dice:.4f}')

df = pd.DataFrame(rows)
best = df.loc[df['Dice'].idxmax()]
plog(f'BEST: thr={best.thr:.2f}  Dice={best.Dice:.4f}  area={best.area_ha:.2f} ha')

# ── Reproduction gate vs April checkpoint ─────────────────────────────────────
plog('=== Reproduction check vs April 2026 checkpoint ===')
with open(CKPT) as f:
    old = {r['thr']: r for r in json.load(f)['rows']}
print(f'{"thr":>5} {"Dice_new":>9} {"Dice_apr":>9} {"dDice":>8} {"area_new":>9} {"area_apr":>9} {"darea":>8}')
worst_dd, worst_da = 0.0, 0.0
for t in sorted(old):
    n = df[np.isclose(df['thr'], t)]
    if n.empty:
        continue
    n = n.iloc[0]
    dd = n['Dice'] - old[t]['Dice']
    da = n['area_ha'] - old[t]['area_ha']
    worst_dd = max(worst_dd, abs(dd)); worst_da = max(worst_da, abs(da))
    print(f'{t:5.2f} {n["Dice"]:9.4f} {old[t]["Dice"]:9.4f} {dd:+8.4f} '
          f'{n["area_ha"]:9.2f} {old[t]["area_ha"]:9.2f} {da:+8.2f}')
gate = 'PASS' if (worst_dd <= 0.005 and worst_da <= 0.5) else 'FAIL'
plog(f'GATE {gate}: max |dDice| = {worst_dd:.4f}, max |darea| = {worst_da:.2f} ha')

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_excel(os.path.join(VTAB, 'Table_Calibration_Extended.xlsx'), index=False)
out = {
    'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'ref_area_ha': ref_area,
    'training_points': total,
    'reproduction_gate': gate,
    'max_dDice_vs_april': worst_dd,
    'max_darea_vs_april': worst_da,
    'best_thr': float(best.thr),
    'best_dice': float(best.Dice),
    'best_area_ha': float(best.area_ha),
    'rows': rows,
}
with open(os.path.join(VTAB, 'Table_Calibration_Extended.json'), 'w') as f:
    json.dump(out, f, indent=2)
plog('Saved Table_Calibration_Extended.xlsx / .json')
plog('DONE.')
