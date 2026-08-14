"""
Regenerate map figures (Fig1/Fig2/Fig3) from GEE with corrected layout.
Key fixes:
  - PANEL_ADJ top=0.980 (panels fill to top — no title stripe to over-crop)
  - No suptitle (Word caption handles description)
  - Figures saved WITHOUT any post-processing crop
  - df_annual loaded from saved Excel (skips slow area-estimation loop)
  - THR_FU = 0.90 (from saved calibration table — skips calibration loop)

Also regenerates Fig_Area_TimeSeries.png with thin/transparent lines.
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import ee

warnings.filterwarnings('ignore')
# figsize target = 14" wide → print scale at 13.8 cm = 0.388
# fontsize 22 → 8.5 pt printed;  26 → 10.1 pt;  30 → 11.6 pt  (target 8-12 pt)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 22,
                     'figure.dpi': 300, 'savefig.dpi': 600})

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = r'C:\Users\Manaf\Documents\Jupyter'
SHP_DIR  = os.path.join(BASE, '_archive_root_duplicates')   # historical shapefiles
OUT_DIR  = os.path.join(BASE, 'mangrove', 'Ras_Sanad_Mangrove_Outputs')
OUT_MAPS = os.path.join(OUT_DIR, 'maps')
OUT_TAB  = os.path.join(OUT_DIR, 'tables')
os.makedirs(OUT_MAPS, exist_ok=True)

# ── EE init ────────────────────────────────────────────────────────────────────
print('Initialising Earth Engine...')
ee.Initialize(project='ee-mkhuzaei94')
print('  EE ready.')

# ── Constants (from Cell 1) ───────────────────────────────────────────────────
_gdf1967      = gpd.read_file(os.path.join(SHP_DIR, 'RasSanad_1967.shp')).to_crs(4326)
_cent         = _gdf1967.geometry.union_all().centroid
RAS_SANAD_LON = round(_cent.x, 5)
RAS_SANAD_LAT = round(_cent.y, 5)
CIRCLE_RADIUS_M = 2000
YEARS           = list(range(2016, 2026))
SHP_YEARS       = [1967, 1998, 2005, 2009, 2016]
TRAIN_YEARS     = [2017, 2018, 2019, 2021, 2022]
HIST_AREAS      = {1967: 97.30, 1998: 56.99, 2005: 37.54, 2009: 35.23, 2016: 34.05}
TARGET_HA       = 34.05
N_TREES         = 150
MAX_PTS         = 2000
VAL_PTS         = 500
S2_SCALE        = 10
S2_BANDS        = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
                   'NDVI','NDMI','MNDWI','EVI','NBR']
S1_BANDS        = ['VV','VH','VV_VH']
FUSED_BANDS     = S2_BANDS + S1_BANDS
THR_FU          = 0.90     # from saved calibration table
PRED_SCALE      = 10       # metres — match the 10 m analysis grid; fetching at
                           # 20 m inflates the drawn fringe area by ~45% via
                           # nearest-neighbour aliasing (measured on pred_2025:
                           # 50.6 ha at 20 m vs 34.9 ha at the 10 m analysis)

print(f'Centroid: ({RAS_SANAD_LON}, {RAS_SANAD_LAT})')

# ── Helper functions (Cell 3) ─────────────────────────────────────────────────
def ee_get(obj, retries=6, delay=5):
    for i in range(retries):
        try: return obj.getInfo()
        except Exception as e:
            if i < retries - 1: time.sleep(delay)
            else: raise e

def area_ha(img, roi, scale):
    a   = img.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=roi,
        scale=scale, maxPixels=1e12, bestEffort=True)
    v   = ee_get(a)
    val = list(v.values())[0] if v else 0
    return (val or 0) / 10_000

def mask_s2(img):
    scl = img.select('SCL')
    m   = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(m)

def s2_indices(img):
    ndvi  = img.normalizedDifference(['B8','B4']).rename('NDVI')
    ndmi  = img.normalizedDifference(['B8','B11']).rename('NDMI')
    mndwi = img.normalizedDifference(['B3','B11']).rename('MNDWI')
    evi   = img.expression('2.5*(N-R)/(N+6*R-7.5*B+1)',
              {'N':img.select('B8'),'R':img.select('B4'),'B':img.select('B2')}).rename('EVI')
    nbr   = img.normalizedDifference(['B8','B12']).rename('NBR')
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
               .filter(ee.Filter.calendarRange(year-1, year+1, 'year'))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
               .map(mask_s2).map(s2_indices))
        n = ee_get(col.size())
        print(f'    S2 {year}: expanded window -> {n} imgs', flush=True)
    else:
        print(f'    S2 {year}: {n} imgs', flush=True)
    return col.median().select(S2_BANDS).clip(region)

def s1_composite(year, region):
    col = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filterBounds(region)
           .filter(ee.Filter.calendarRange(year, year, 'year'))
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
           .select(['VV','VH']))
    n   = ee_get(col.size())
    if n == 0:
        col = (ee.ImageCollection('COPERNICUS/S1_GRD')
               .filterBounds(region)
               .filter(ee.Filter.calendarRange(year-1, year+1, 'year'))
               .filter(ee.Filter.eq('instrumentMode', 'IW'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
               .select(['VV','VH']))
        n = ee_get(col.size())
    med   = col.median()
    ratio = med.select('VV').subtract(med.select('VH')).rename('VV_VH')
    print(f'    S1 {year}: {n} imgs', flush=True)
    return med.addBands(ratio).select(S1_BANDS).clip(region)

# ── ROI + shapefiles (Cell 4) ─────────────────────────────────────────────────
print('\nBuilding ROI and ever-mangrove union...')
roi_center = ee.Geometry.Point([RAS_SANAD_LON, RAS_SANAD_LAT])
roi_ee     = roi_center.buffer(CIRCLE_RADIUS_M)

_dlat = CIRCLE_RADIUS_M / 111320
_dlon = CIRCLE_RADIUS_M / (111320 * np.cos(np.radians(RAS_SANAD_LAT)))
ROI_LAT_MIN = RAS_SANAD_LAT - _dlat
ROI_LAT_MAX = RAS_SANAD_LAT + _dlat
ROI_LON_MIN = RAS_SANAD_LON - _dlon
ROI_LON_MAX = RAS_SANAD_LON + _dlon
ROI_EXTENT  = [ROI_LON_MIN, ROI_LON_MAX, ROI_LAT_MIN, ROI_LAT_MAX]

def gdf_to_ee_geom(gdf):
    geom = gdf.geometry.union_all()
    return ee.Geometry(geom.__geo_interface__)

RASSANAD_EE = {}
for yr in SHP_YEARS:
    fp  = os.path.join(SHP_DIR, f'RasSanad_{yr}.shp')
    gdf = gpd.read_file(fp).to_crs(4326)
    RASSANAD_EE[yr] = gdf_to_ee_geom(gdf)

ever_mg_geom = None
for yr, geom in RASSANAD_EE.items():
    ever_mg_geom = geom if ever_mg_geom is None else ever_mg_geom.union(geom, 1)
EVER_MG_GEOM  = ever_mg_geom.intersection(roi_ee, 1)
CALIB_2016_EE = RASSANAD_EE[2016]
print('  ROI and ever-mangrove ready.')

# ── Training labels (Cell 8) ──────────────────────────────────────────────────
print('\nBuilding training samples (please wait ~10-20 min)...')
mg_geom      = EVER_MG_GEOM
non_mg_geom  = roi_ee.difference(EVER_MG_GEOM, 1)
hard_neg_geom = (EVER_MG_GEOM.buffer(600, 1)
                 .intersection(roi_ee, 1)
                 .difference(EVER_MG_GEOM, 1))

mg_fc     = ee.FeatureCollection([ee.Feature(mg_geom,       {'class': 1})])
non_fc    = ee.FeatureCollection([ee.Feature(non_mg_geom,   {'class': 0})])
label_img = (ee.Image(0).byte()
             .paint(non_fc, 0).paint(mg_fc, 1).rename('class'))
HARD_NEG_MASK = (ee.Image(0).byte()
                 .paint(ee.FeatureCollection([ee.Feature(hard_neg_geom)]), 1))

all_fu, all_s2 = [], []
for yr in TRAIN_YEARS:
    print(f'  Training year {yr}:', end=' ', flush=True)
    s2c = s2_composite(yr, roi_ee)
    s1c = s1_composite(yr, roi_ee)
    fuc = s2c.addBands(s1c)
    base_fu = fuc.select(FUSED_BANDS).addBands(label_img)
    base_s2 = s2c.select(S2_BANDS).addBands(label_img)
    samp_std_fu = base_fu.stratifiedSample(
        numPoints=MAX_PTS//2, classBand='class', region=roi_ee,
        scale=S2_SCALE, seed=yr, geometries=False,
        classValues=[0,1], classPoints=[MAX_PTS//2, MAX_PTS])
    samp_std_s2 = base_s2.stratifiedSample(
        numPoints=MAX_PTS//2, classBand='class', region=roi_ee,
        scale=S2_SCALE, seed=yr+1000, geometries=False,
        classValues=[0,1], classPoints=[MAX_PTS//2, MAX_PTS])
    samp_hrd_fu = (base_fu.updateMask(HARD_NEG_MASK.And(label_img.eq(0)))
                   .sample(region=roi_ee, scale=S2_SCALE,
                           numPixels=MAX_PTS//2, seed=yr+500, geometries=False))
    samp_hrd_s2 = (base_s2.updateMask(HARD_NEG_MASK.And(label_img.eq(0)))
                   .sample(region=roi_ee, scale=S2_SCALE,
                           numPixels=MAX_PTS//2, seed=yr+1500, geometries=False))
    samp_fu = samp_std_fu.merge(samp_hrd_fu)
    samp_s2 = samp_std_s2.merge(samp_hrd_s2)
    n_fu = ee_get(samp_fu.size())
    print(f'Fused={n_fu:,} pts', flush=True)
    all_fu.append(samp_fu)
    all_s2.append(samp_s2)

tr_fu = all_fu[0]
for s in all_fu[1:]: tr_fu = tr_fu.merge(s)
tr_s2 = all_s2[0]
for s in all_s2[1:]: tr_s2 = tr_s2.merge(s)
print(f'  Total Fused: {ee_get(tr_fu.size()):,} pts')

# ── Train RF (Cell 9) ─────────────────────────────────────────────────────────
print('\nTraining RF classifiers...')
s2_val = s2_composite(2023, roi_ee)
s1_val = s1_composite(2023, roi_ee)
fu_val = s2_val.addBands(s1_val)

RF_FU = (ee.Classifier.smileRandomForest(N_TREES, seed=42)
         .setOutputMode('PROBABILITY')
         .train(features=tr_fu, classProperty='class', inputProperties=FUSED_BANDS))
RF_S2 = (ee.Classifier.smileRandomForest(N_TREES, seed=42)
         .setOutputMode('PROBABILITY')
         .train(features=tr_s2, classProperty='class', inputProperties=S2_BANDS))
print('  RF_FU and RF_S2 trained.')

# ── Tidal mask (Cell 10, first half only — THR_FU already known = 0.90) ────────
print('\nBuilding tidal mask...')
jrc_water  = (ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
              .select('max_extent').unmask(0).byte())
WATER_PROX = jrc_water.focal_max(radius=500, units='meters', kernelType='circle')
ELEV_MASK  = ee.Image('USGS/SRTMGL1_003').select('elevation').lt(10)
TIDAL_MASK = WATER_PROX.And(ELEV_MASK)
MORPH_R    = 10

def clean_pred(img):
    return (img.focal_min(radius=MORPH_R, units='meters')
               .focal_max(radius=MORPH_R, units='meters'))

print(f'  Tidal mask ready.  THR_FU = {THR_FU} (from saved calibration table)')

# ── Load df_annual from saved table (skip slow area loop) ────────────────────
df_annual = pd.read_excel(os.path.join(OUT_TAB, 'Table_Annual_Areas_Fused.xlsx'))
df_annual = df_annual.sort_values('year').reset_index(drop=True)
print('\nAnnual areas loaded from saved table:')
print(df_annual[['year','area_fused']].round(2).to_string(index=False))

# ── Download arrays (Cell 13) ─────────────────────────────────────────────────
def get_rgb_array(s2_img):
    # S2_SR_HARMONIZED serves surface reflectance scaled by 10^4, so convert to
    # 0-1 reflectance before the 0.28 display stretch. (The pre-2026-08 cache
    # was built by a notebook that divided by 10^4 upstream; fetching raw here
    # without this conversion saturates the panels to white.)
    try:
        rgb  = s2_img.select(['B4','B3','B2']).reproject(crs='EPSG:4326', scale=PRED_SCALE)
        rect = ee_get(rgb.sampleRectangle(region=roi_ee.bounds(), defaultValue=0))
        gamma = 1.4
        def _stretch(band):
            refl = np.array(rect['properties'][band], np.float32) / 10000.0
            return np.clip(refl / 0.28, 0, 1) ** (1 / gamma)
        r, g, b = _stretch('B4'), _stretch('B3'), _stretch('B2')
        return np.flipud(np.stack([r, g, b], axis=-1))
    except Exception as e:
        print(f'  ERR-RGB({e})', flush=True)
        return None

def get_pred_array(fuc_img):
    try:
        pred = (fuc_img.classify(RF_FU).rename('prob')
                .gte(THR_FU).And(TIDAL_MASK))
        pred = clean_pred(pred).toFloat().reproject(crs='EPSG:4326', scale=PRED_SCALE)
        rect = ee_get(pred.sampleRectangle(region=roi_ee.bounds(), defaultValue=0))
        return np.flipud(np.array(rect['properties']['prob'], dtype=np.float32))
    except Exception as e:
        print(f'  ERR-PRED({e})', flush=True)
        return None

def get_prob_array(fuc_img):
    try:
        prob = (fuc_img.classify(RF_FU).rename('prob')
                .reproject(crs='EPSG:4326', scale=PRED_SCALE))
        rect = ee_get(prob.sampleRectangle(region=roi_ee.bounds(), defaultValue=0))
        return np.flipud(np.array(rect['properties']['prob'], dtype=np.float32))
    except Exception as e:
        print(f'  ERR-PROB({e})', flush=True)
        return None

RGB_ARRS  = {}
PRED_ARRS = {}
PROB_ARRS = {}

# ── Array cache — avoid re-downloading on every render ────────────────────────
CACHE_DIR = os.path.join(OUT_DIR, 'array_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(year, kind):
    return os.path.join(CACHE_DIR, f'{kind}_{year}.npy')

def _all_cached():
    return all(
        os.path.exists(_cache_path(yr, k))
        for yr in YEARS for k in ('rgb', 'pred', 'prob')
    )

print('\nLoading satellite arrays (cache per year/kind; missing kinds fetched)...')
for year in YEARS:
    need = [k for k in ('rgb', 'pred', 'prob')
            if not os.path.exists(_cache_path(year, k))]
    if need:
        print(f'  {year}: fetching {need}...', flush=True)
        s2c = s2_composite(year, roi_ee)
        s1c = s1_composite(year, roi_ee)
        fuc = s2c.addBands(s1c)
    for kind, target, fetch in [
            ('rgb',  RGB_ARRS,  lambda: get_rgb_array(s2c)),
            ('pred', PRED_ARRS, lambda: get_pred_array(fuc)),
            ('prob', PROB_ARRS, lambda: get_prob_array(fuc))]:
        p = _cache_path(year, kind)
        if os.path.exists(p):
            target[year] = np.load(p, allow_pickle=True)
        else:
            arr = fetch()
            target[year] = arr
            if arr is not None:
                np.save(p, arr)
    print(f'  {year} ready', flush=True)

print('\nAll arrays ready.')

# ── Figure layout — FIXED (no suptitle, top=0.980, panels fill to top) ────────
# Key fix: top=0.980 means panels start only 2% from top of figure.
# bbox_inches='tight' saves with minimal whitespace above.
# No suptitle => nothing to crop => circles are never clipped.
FIG_W, FIG_H = 14, 6.5   # 14" wide → scale=0.388 at 13.8 cm print width
LAT_ASP = 1 / np.cos(np.radians(RAS_SANAD_LAT))   # ≈ 1.114 at 26°N — makes circles round

# Coordinate ticks — STRICTLY within data range (prevents xlim auto-expansion)
LON_TICKS = [50.58, 50.60]   # ROI lon range [50.5756, 50.6156] → both inside ✓
LAT_TICKS = [26.13, 26.15]   # ROI lat range [26.1288, 26.1647] → both inside ✓

# Expanded margins to accommodate outer tick labels
PANEL_ADJ = dict(left=0.080, right=0.982, top=0.980, bottom=0.105,
                 hspace=0.08, wspace=0.04)

def _scale_bar(ax, length_m=500, x0=0.05, y0=0.05, bar_h=0.015, txt_col='white'):
    m_per_deg_lon = 111320 * np.cos(np.radians(RAS_SANAD_LAT))
    bar_w_deg = length_m / m_per_deg_lon
    bar_frac  = bar_w_deg / (ROI_LON_MAX - ROI_LON_MIN)
    rect = mpatches.FancyBboxPatch(
        (x0, y0), bar_frac, bar_h, boxstyle='square,pad=0', lw=0.8,
        edgecolor=txt_col, facecolor=txt_col, transform=ax.transAxes, zorder=10)
    ax.add_patch(rect)
    # fontsize=11: in a panel ~2.4" wide, 11pt text = 0.15" — fits cleanly
    ax.text(x0 + bar_frac/2, y0 + bar_h + 0.020, f'{length_m:g} m',
            transform=ax.transAxes, fontsize=11, color=txt_col,
            ha='center', va='bottom', fontweight='bold')

def _north_arrow(ax, x=0.92, y=0.10, txt_col='white'):
    ax.annotate('', xy=(x, y+0.07), xytext=(x, y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=txt_col, lw=1.0))
    ax.text(x, y+0.09, 'N', transform=ax.transAxes,
            fontsize=12, color=txt_col, ha='center', va='bottom', fontweight='bold')

def annotate_panel(ax, year, row_i, col_i, n_rows=2, dark=True):
    # Font sizing rationale for a 2×5 grid (each panel ~2.4" wide at figsize=14"):
    #   year label  fontsize=14 → text "2016" ~0.47" wide = 20% of panel ✓
    #   area badge  fontsize=11 → text "35.2 ha" ~0.55" wide = 23% of panel ✓
    #   ticks       fontsize= 9 → "50.58°E" ~0.38" wide — won't merge between panels ✓
    tc  = 'white' if dark else '#1a1a1a'
    bgc = '#0d1117aa' if dark else '#ffffffcc'
    gc  = '#ffffff18' if dark else '#00000014'
    row = df_annual[df_annual['year'] == year]
    a_str = (f"{row['area_fused'].values[0]:.1f} ha"
             if len(row) > 0 and not np.isnan(row['area_fused'].values[0]) else '-')
    # Area badge: top-left, secondary info
    ax.text(0.03, 0.97, a_str, transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=tc, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=bgc, alpha=0.88, edgecolor='none'))
    # Year: top-right, primary panel identifier
    ax.text(0.97, 0.97, str(year), transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=tc, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=bgc, alpha=0.88, edgecolor='none'))
    for lon in LON_TICKS:
        ax.axvline(lon, color=gc, lw=0.45, zorder=2)
    for lat in LAT_TICKS:
        ax.axhline(lat, color=gc, lw=0.45, zorder=2)
    ax.set_xticks(LON_TICKS)
    ax.set_yticks(LAT_TICKS)
    if row_i == n_rows - 1:
        ax.set_xticklabels([f'{x:.2f}E' for x in LON_TICKS], fontsize=9, color=tc)
        ax.tick_params(axis='x', colors=tc, length=2.0, pad=1.0, width=0.5)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
    if col_i == 0:
        ax.set_yticklabels([f'{y:.2f}N' for y in LAT_TICKS], fontsize=9, color=tc,
                           rotation=90, va='center')
        ax.tick_params(axis='y', colors=tc, length=2.0, pad=1.0, width=0.5)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
    ax.set_xlim(ROI_LON_MIN, ROI_LON_MAX)
    ax.set_ylim(ROI_LAT_MIN, ROI_LAT_MAX)
    ax.set_aspect(LAT_ASP, adjustable='datalim')
    sp_c = '#444444' if dark else '#cccccc'
    for sp in ax.spines.values():
        sp.set_edgecolor(sp_c); sp.set_linewidth(0.7)

# ── FIGURE 1: Satellite RGB + classification overlay ─────────────────────────
# Light figure background — publication standard white framing with satellite imagery panels
print('\nRendering Figure 1 (Satellite Classification)...')
BG1 = 'white'
fig1, axes1 = plt.subplots(2, 5, figsize=(FIG_W, FIG_H), facecolor=BG1)
fig1.subplots_adjust(**PANEL_ADJ)
# Muted green overlay — #4caf50 alpha 0.68 is more conservative than neon green
legend_v1 = [mpatches.Patch(facecolor='#4caf50', edgecolor='#1a5c2a', lw=0.6,
                              label=f'Predicted mangrove  (thr = {THR_FU:.2f})')]
for idx, (ax, year) in enumerate(zip(axes1.flat, YEARS)):
    ri, ci = divmod(idx, 5)
    ax.set_facecolor('#d0dde8')   # light blue-grey for any panel edge gaps
    rgb = RGB_ARRS.get(year)
    if rgb is not None:
        ax.imshow(rgb, extent=ROI_EXTENT, aspect='auto',
                  interpolation='bilinear', origin='lower', zorder=1)
    pred = PRED_ARRS.get(year)
    if pred is not None and pred.max() > 0:
        overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
        overlay[pred > 0.5] = [0.30, 0.69, 0.31, 0.68]   # muted green, 68% opacity
        ax.imshow(overlay, extent=ROI_EXTENT, aspect='auto', origin='lower', zorder=3)
    annotate_panel(ax, year, ri, ci, dark=True)   # white text still readable on satellite
    if idx == 0:
        _scale_bar(ax, length_m=500, txt_col='white')
        _north_arrow(ax, txt_col='white')
        ax.legend(handles=legend_v1, loc='lower right', fontsize=11,
                  facecolor='white', edgecolor='#aaaaaa', labelcolor='#111111',
                  framealpha=0.95, handlelength=1.6, handleheight=1.2, borderpad=0.5)
fig1.patch.set_facecolor(BG1)
out1 = os.path.join(OUT_MAPS, 'Fig1_Satellite_Classification.png')
fig1.savefig(out1, dpi=600, bbox_inches='tight', facecolor=BG1)
plt.close(fig1)
from PIL import Image as _PIL
_s = _PIL.open(out1).size
print(f'  Saved: {out1}  ({_s[0]}x{_s[1]}px)')

# ── FIGURE 2: Cartographic polygon map ───────────────────────────────────────
print('Rendering Figure 2 (Polygon Map)...')
BG2 = 'white'
fig2, axes2 = plt.subplots(2, 5, figsize=(FIG_W, FIG_H), facecolor=BG2)
fig2.subplots_adjust(**PANEL_ADJ)
legend_v2 = [
    mpatches.Patch(facecolor='#5ab96b', edgecolor='#2d6a3f', lw=1.0, label='Predicted mangrove'),
    mpatches.Patch(facecolor='#cde8f0', edgecolor='#6aafc8', lw=0.5, label='Non-mangrove / open water'),
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
        ax.legend(handles=legend_v2, loc='lower right', fontsize=11,
                  facecolor='white', edgecolor='#aaaaaa', labelcolor='#111111',
                  framealpha=0.96, handlelength=1.6, handleheight=1.2, borderpad=0.5)
fig2.patch.set_facecolor(BG2)
out2 = os.path.join(OUT_MAPS, 'Fig2_Polygon_Map.png')
fig2.savefig(out2, dpi=600, bbox_inches='tight', facecolor=BG2)
plt.close(fig2)
_s = _PIL.open(out2).size
print(f'  Saved: {out2}  ({_s[0]}x{_s[1]}px)')

# ── FIGURE 3: Probability map ─────────────────────────────────────────────────
# Light publication background. Perceptually uniform green ramp (transparent →
# opaque), alpha scaled by probability so low-prob areas show satellite.
# Dashed white contour at decision threshold for scientific clarity.
print('Rendering Figure 3 (Probability Map)...')
BG3 = 'white'

# Build perceptually uniform green RGBA colormap using matplotlib Greens
_greens_base = plt.cm.Greens
_n3 = 256
_ca3 = np.zeros((_n3, 4))
for i in range(_n3):
    t = i / (_n3 - 1)
    rgba = list(_greens_base(0.20 + t * 0.80))   # use 20–100% of Greens range
    rgba[3] = max(0.0, min(1.0, t ** 0.6))        # alpha: 0 at t=0, 1 at t=1
    _ca3[i] = rgba
PROB_CMAP = mcolors.ListedColormap(_ca3)

fig3, axes3 = plt.subplots(2, 5, figsize=(FIG_W, FIG_H), facecolor=BG3)
# Wider right margin to give colorbar more space and reduce crowding
fig3.subplots_adjust(left=0.055, right=0.920, top=0.980, bottom=0.075,
                     hspace=0.08, wspace=0.04)
for idx, (ax, year) in enumerate(zip(axes3.flat, YEARS)):
    ri, ci = divmod(idx, 5)
    ax.set_facecolor('#d0dde8')   # light blue-grey — matches Fig1 panel gaps
    rgb = RGB_ARRS.get(year)
    if rgb is not None:
        ax.imshow(rgb * 0.70, extent=ROI_EXTENT, aspect='auto',
                  interpolation='bilinear', origin='lower', zorder=1)
    prob = PROB_ARRS.get(year)
    if prob is not None:
        ax.imshow(PROB_CMAP(prob), extent=ROI_EXTENT, aspect='auto', origin='lower', zorder=3)
        if prob.max() > THR_FU:
            ax.contour(prob, levels=[THR_FU], colors=['white'],
                       linewidths=[0.9], linestyles=['--'],
                       extent=ROI_EXTENT, origin='lower', zorder=5)
    annotate_panel(ax, year, ri, ci, dark=True)
    if idx == 0:
        _scale_bar(ax, length_m=500, txt_col='white')
        _north_arrow(ax, txt_col='white')
fig3.patch.set_facecolor(BG3)
# Colorbar — slightly wider and better positioned, dark label for white bg
sm3 = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap(_ca3[20:]),
                              norm=mcolors.Normalize(vmin=0.08, vmax=1.0))
sm3.set_array([])
cax3  = fig3.add_axes([0.927, 0.075, 0.014, 0.905])
cbar3 = fig3.colorbar(sm3, cax=cax3, orientation='vertical')
cbar3.set_label('RF Mangrove Probability', fontsize=11, color='#111111', labelpad=6)
cbar3.ax.yaxis.set_tick_params(color='#333333', labelsize=9)
plt.setp(cbar3.ax.yaxis.get_ticklabels(), color='#333333')
cbar3.outline.set_edgecolor('#aaaaaa')
# Threshold tick
cbar3.ax.axhline((THR_FU - 0.08) / (1.0 - 0.08), color='white', lw=1.2,
                  linestyle='--', alpha=0.9)
fig3.patch.set_facecolor(BG3)
out3 = os.path.join(OUT_MAPS, 'Fig3_Probability_Map.png')
fig3.savefig(out3, dpi=600, bbox_inches='tight', facecolor=BG3)
plt.close(fig3)
_s = _PIL.open(out3).size
print(f'  Saved: {out3}  ({_s[0]}x{_s[1]}px)')

# Fig1-3 regenerated from the current-archive arrays. Stop here: the study-area
# figure and time series are produced by _make_study_area_fig.py and
# _make_timeseries_fig.py respectively.
print('\nDone (Fig1/Fig2/Fig3). Study-area and time-series figures are handled '
      'by their dedicated scripts.')
sys.exit(0)
