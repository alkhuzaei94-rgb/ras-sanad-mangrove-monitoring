"""
scene_counts.py
====================
Per-year Sentinel-2 and Sentinel-1 scene counts and
cloud statistics for each annual composite, 2016-2025.

Replicates the exact collection filters of the production workflow
(verify_stability.py s2_composite / s1_composite) and reports, per year:
  - S2: granules before and after the CLOUDY_PIXEL_PERCENTAGE < 50 screen,
        distinct acquisition dates, cloud-percentage mean/median/min/max
  - S1: granules (IW, VV+VH, descending), distinct acquisition dates
  - whether the +/-1 year fallback would have triggered (it must not)

Output: Ras_Sanad_Verification/tables/Table_SceneCounts.xlsx + .json

Run:
  python scene_counts.py
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

def plog(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

plog('EE init...')
ee.Initialize(project=os.environ.get('EE_PROJECT'))  # set EE_PROJECT to your own Cloud project

# ROI identical to production workflow: centroid of the 1967 polygon, 2 km buffer
gdf1967 = gpd.read_file(os.path.join(SHP_DIR, 'RasSanad_1967.shp')).to_crs(4326)
cent    = gdf1967.geometry.union_all().centroid
LON, LAT = round(cent.x, 5), round(cent.y, 5)
roi = ee.Geometry.Point([LON, LAT]).buffer(2000)
plog(f'ROI centre: ({LON}, {LAT}), r = 2000 m')

def distinct_dates(col):
    ts = col.aggregate_array('system:time_start').getInfo()
    return sorted({time.strftime('%Y-%m-%d', time.gmtime(t / 1000)) for t in ts})

rows = []
for year in range(2016, 2026):
    plog(f'Year {year}...')
    s2_all = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(roi)
              .filter(ee.Filter.calendarRange(year, year, 'year')))
    s2_scr = s2_all.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))

    n_all = s2_all.size().getInfo()
    n_scr = s2_scr.size().getInfo()
    cc    = s2_scr.aggregate_array('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    d_s2  = distinct_dates(s2_scr)

    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(roi)
          .filter(ee.Filter.calendarRange(year, year, 'year'))
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')))
    n_s1 = s1.size().getInfo()
    d_s1 = distinct_dates(s1)

    # Production fallback in s1_composite(): if the calendar-year descending
    # collection is empty, it widens to year-1..year+1 AND drops the orbit
    # filter. Quantify what that fallback actually contained.
    s1_fb_n, s1_fb_dates, s1_fb_orbits = None, None, None
    if n_s1 == 0:
        s1_fb = (ee.ImageCollection('COPERNICUS/S1_GRD')
                 .filterBounds(roi)
                 .filter(ee.Filter.calendarRange(year - 1, year + 1, 'year'))
                 .filter(ee.Filter.eq('instrumentMode', 'IW'))
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
        s1_fb_n      = s1_fb.size().getInfo()
        s1_fb_dates  = len(distinct_dates(s1_fb))
        orb          = s1_fb.aggregate_array('orbitProperties_pass').getInfo()
        s1_fb_orbits = {o: orb.count(o) for o in set(orb)}
        plog(f'  S1 FALLBACK for {year}: {s1_fb_n} granules {year-1}-{year+1}, '
             f'orbits {s1_fb_orbits}')

    row = {
        'year': year,
        's2_granules_total': n_all,
        's2_granules_cloudlt50': n_scr,
        's2_distinct_dates': len(d_s2),
        's2_cloud_mean': round(float(np.mean(cc)), 2) if cc else None,
        's2_cloud_median': round(float(np.median(cc)), 2) if cc else None,
        's2_cloud_min': round(float(np.min(cc)), 2) if cc else None,
        's2_cloud_max': round(float(np.max(cc)), 2) if cc else None,
        's1_granules': n_s1,
        's1_distinct_dates': len(d_s1),
        's2_fallback_triggered': (n_scr == 0),
        's1_fallback_triggered': (n_s1 == 0),
        's1_fallback_granules': s1_fb_n,
        's1_fallback_dates': s1_fb_dates,
        's1_fallback_orbits': str(s1_fb_orbits) if s1_fb_orbits else None,
    }
    rows.append(row)
    plog(f"  S2 {n_all} -> {n_scr} granules ({len(d_s2)} dates, "
         f"cloud mean {row['s2_cloud_mean']}%)  |  S1 {n_s1} granules ({len(d_s1)} dates)")

df = pd.DataFrame(rows)
df.to_excel(os.path.join(VTAB, 'Table_SceneCounts.xlsx'), index=False)
with open(os.path.join(VTAB, 'Table_SceneCounts.json'), 'w') as f:
    json.dump(rows, f, indent=2)

plog('Saved Table_SceneCounts.xlsx / .json')
print()
print(df.to_string(index=False))
for sensor in ['s2', 's1']:
    col = f'{sensor}_fallback_triggered'
    yrs = df.loc[df[col], 'year'].tolist()
    if yrs:
        print(f'\nNOTE: {sensor.upper()} fallback window fired for year(s): {yrs}')
    else:
        print(f'\n{sensor.upper()}: all years covered within the calendar year, no fallback.')
