"""
Fig0_Study_Area.png — Publication-quality two-panel study area figure.
MDPI Remote Sensing submission standard, 600 dpi.

Panel (a): Bahrain national context with real web-tile basemap (CartoDB Positron)
Panel (b): Ras Sanad 2 km ROI detail with mangrove extent (2016)
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
import contextily as ctx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from shapely.geometry import Point
from pyproj import Transformer

BASE = r'C:\Users\Manaf\Documents\Jupyter'
MAPS = os.path.join(BASE, 'mangrove', 'Ras_Sanad_Mangrove_Outputs', 'maps')

# ── Load shapefiles ───────────────────────────────────────────────────────────
print('Loading shapefiles...')
bhr_gdf = gpd.read_file(os.path.join(BASE, 'shared_data', 'boundaries', 'Bahrain.shp')).to_crs(3857)
mg16    = gpd.read_file(os.path.join(BASE, '_archive_root_duplicates', 'RasSanad_2016.shp')).to_crs(3857)

# ── Coordinate transformers ───────────────────────────────────────────────────
t_fwd = Transformer.from_crs(4326, 3857, always_xy=True)  # lon/lat → Web Mercator
t_inv = Transformer.from_crs(3857, 4326, always_xy=True)  # Web Mercator → lon/lat

def ll2m(lon, lat):
    return t_fwd.transform(lon, lat)

def m2ll(x, y):
    return t_inv.transform(x, y)

# ── ROI parameters (in Web Mercator) ─────────────────────────────────────────
RAS_LON, RAS_LAT = 50.5956, 26.14677
RADIUS_M = 2000
ras_x, ras_y = ll2m(RAS_LON, RAS_LAT)

# ROI circle in Web Mercator
theta = np.linspace(0, 2 * np.pi, 360)
# Approximate 2 km circle in metres (scale factor at this latitude ≈ 1)
scale = 1.0 / np.cos(np.radians(RAS_LAT))   # Web Mercator x-stretch
roi_x = ras_x + RADIUS_M * scale * np.cos(theta)
roi_y = ras_y + RADIUS_M * np.sin(theta)

# ROI extent for panel (b) in Web Mercator
pad_m = 500
b_xmin = ras_x - RADIUS_M * scale - pad_m
b_xmax = ras_x + RADIUS_M * scale + pad_m
b_ymin = ras_y - RADIUS_M - pad_m
b_ymax = ras_y + RADIUS_M + pad_m

# ── Color palette ─────────────────────────────────────────────────────────────
C_BHR   = '#3a5ea8'    # Bahrain border — blue
C_MNG   = '#3a8a4a'    # mangrove green
C_ROI   = '#c0392b'    # study circle red
C_ANNOT = '#c0392b'
C_BG    = 'white'

# ── Font sizes ────────────────────────────────────────────────────────────────
FS_TITLE  = 14
FS_LABEL  = 12
FS_TICK   = 10
FS_ANNOT  = 11
FS_LEGEND = 11
FS_PANEL  = 13

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 6),
                                  gridspec_kw={'wspace': 0.14})
fig.patch.set_facecolor(C_BG)
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.10)

# =============================================================================
# PANEL (a): Bahrain national context — with real tile basemap
# =============================================================================
# Extent: show Bahrain + Saudi Arabia (west) + Qatar (east)
A_LON_MIN, A_LON_MAX = 49.30, 52.10
A_LAT_MIN, A_LAT_MAX = 24.90, 27.30

a_xmin, a_ymin = ll2m(A_LON_MIN, A_LAT_MIN)
a_xmax, a_ymax = ll2m(A_LON_MAX, A_LAT_MAX)

ax_a.set_xlim(a_xmin, a_xmax)
ax_a.set_ylim(a_ymin, a_ymax)

# Add tile basemap — CartoDB Positron: clean, print-friendly, shows country labels
print('Fetching basemap tiles for panel (a)...')
try:
    ctx.add_basemap(ax_a, crs='EPSG:3857',
                    source=ctx.providers.CartoDB.Positron,
                    zoom=8, attribution=False)
    print('  Basemap loaded: CartoDB Positron')
except Exception as e:
    print(f'  CartoDB Positron failed ({e}), trying OpenStreetMap...')
    try:
        ctx.add_basemap(ax_a, crs='EPSG:3857',
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom=8, attribution=False)
        print('  Basemap loaded: OpenStreetMap')
    except Exception as e2:
        print(f'  Basemap fetch failed: {e2}')

# Bahrain border — prominent blue outline on top of tiles
bhr_gdf.plot(ax=ax_a, color='none', edgecolor=C_BHR, linewidth=1.6, zorder=3)

# ROI circle + centre dot
ax_a.fill(roi_x, roi_y, color=C_ROI, alpha=0.15, zorder=4)
ax_a.plot(roi_x, roi_y, '-', color=C_ROI, lw=1.2, zorder=5)
ax_a.plot(ras_x, ras_y, 'o', color=C_ROI, ms=5,
          markeredgecolor='white', markeredgewidth=0.8, zorder=6)

# Ras Sanad annotation
annot_x, annot_y = ll2m(51.15, 26.65)
ax_a.annotate('Ras Sanad',
              xy=(ras_x, ras_y), xytext=(annot_x, annot_y),
              fontsize=FS_ANNOT, color=C_ANNOT, fontweight='bold',
              arrowprops=dict(arrowstyle='->', color=C_ANNOT, lw=0.9,
                              connectionstyle='arc3,rad=-0.2'),
              bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                        edgecolor=C_ANNOT, linewidth=0.7, alpha=0.92),
              zorder=8)

# Geographic tick labels (lon/lat) on Web Mercator axes
lon_ticks = [49.5, 50.0, 50.5, 51.0, 51.5]
lat_ticks = [25.0, 25.5, 26.0, 26.5, 27.0]
ax_a.set_xticks([ll2m(lon, A_LAT_MIN)[0] for lon in lon_ticks])
ax_a.set_yticks([ll2m(A_LON_MIN, lat)[1] for lat in lat_ticks])
ax_a.set_xticklabels([f'{lon:.1f}°E' for lon in lon_ticks], fontsize=FS_TICK)
ax_a.set_yticklabels([f'{lat:.1f}°N' for lat in lat_ticks], fontsize=FS_TICK,
                     rotation=90, va='center')
ax_a.set_xlabel('Longitude', fontsize=FS_LABEL, labelpad=4)
ax_a.set_ylabel('Latitude', fontsize=FS_LABEL, labelpad=4)
for sp in ax_a.spines.values():
    sp.set_edgecolor('#888888'); sp.set_linewidth(0.7)

# North arrow
ax_a.annotate('', xy=(0.93, 0.91), xytext=(0.93, 0.84),
              xycoords='axes fraction', textcoords='axes fraction',
              arrowprops=dict(arrowstyle='->', color='#222222', lw=1.2))
ax_a.text(0.93, 0.93, 'N', transform=ax_a.transAxes,
          fontsize=FS_ANNOT, ha='center', va='bottom',
          color='#222222', fontweight='bold')

# Scale bar — 50 km in Web Mercator (at lat ~26°)
sc50_m = 50000 * (1.0 / np.cos(np.radians(26.0)))   # x-distance in EPSG:3857
sb_x0, sb_y0 = ll2m(49.40, 25.02)
ax_a.plot([sb_x0, sb_x0 + sc50_m], [sb_y0, sb_y0],
          'k-', lw=2.0, solid_capstyle='butt', zorder=9)
ax_a.text(sb_x0 + sc50_m / 2, sb_y0 + 12000, '50 km',
          fontsize=FS_ANNOT - 1, ha='center', va='bottom', zorder=9,
          bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.7))

# Panel label
ax_a.text(0.02, 0.97, '(a)', transform=ax_a.transAxes,
          fontsize=FS_PANEL, fontweight='bold', va='top', ha='left',
          color='#111111', zorder=10,
          bbox=dict(boxstyle='square,pad=0.20', facecolor='white',
                    edgecolor='#888888', linewidth=0.5, alpha=0.85))
ax_a.set_title('Bahrain: national context and study area location',
               fontsize=FS_TITLE, fontweight='bold', pad=4, color='#111111')

# =============================================================================
# PANEL (b): Ras Sanad ROI detail — with real tile basemap
# =============================================================================
ax_b.set_xlim(b_xmin, b_xmax)
ax_b.set_ylim(b_ymin, b_ymax)

# Add tile basemap — higher zoom for local detail
print('Fetching basemap tiles for panel (b)...')
try:
    ctx.add_basemap(ax_b, crs='EPSG:3857',
                    source=ctx.providers.CartoDB.Positron,
                    zoom=15, attribution=False)
    print('  Basemap loaded: CartoDB Positron (zoom 15)')
except Exception as e:
    print(f'  CartoDB Positron failed ({e}), trying OpenStreetMap...')
    try:
        ctx.add_basemap(ax_b, crs='EPSG:3857',
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom=15, attribution=False)
        print('  Basemap loaded: OpenStreetMap (zoom 15)')
    except Exception as e2:
        print(f'  Basemap fetch failed: {e2}')

# Mangrove extent (2016) — clearly prominent on top of tiles
mg16.plot(ax=ax_b, color=C_MNG, edgecolor='#1e5c2a', linewidth=0.8, zorder=4, alpha=0.85)

# ROI circle
ax_b.fill(roi_x, roi_y, color=C_ROI, alpha=0.05, zorder=3)
ax_b.plot(roi_x, roi_y, '-', color=C_ROI, lw=1.4, alpha=0.85, zorder=5)

# Ras Sanad label
mg_cent = mg16.geometry.union_all().centroid
label_x = mg_cent.x + 350
label_y = mg_cent.y + 500
ax_b.text(label_x, label_y, 'Ras Sanad',
          fontsize=FS_ANNOT, color='#222222', fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                    edgecolor='#888888', linewidth=0.6, alpha=0.90),
          zorder=7)

# Geographic ticks
lon_ticks_b = [50.58, 50.60]
lat_ticks_b = [26.13, 26.15]
ax_b.set_xticks([ll2m(lon, RAS_LAT)[0] for lon in lon_ticks_b])
ax_b.set_yticks([ll2m(RAS_LON, lat)[1] for lat in lat_ticks_b])
ax_b.set_xticklabels([f'{lon:.2f}°E' for lon in lon_ticks_b], fontsize=FS_TICK)
ax_b.set_yticklabels([f'{lat:.2f}°N' for lat in lat_ticks_b], fontsize=FS_TICK,
                     rotation=90, va='center')
ax_b.set_xlabel('Longitude', fontsize=FS_LABEL, labelpad=4)
ax_b.set_ylabel('Latitude', fontsize=FS_LABEL, labelpad=4)
for sp in ax_b.spines.values():
    sp.set_edgecolor('#888888'); sp.set_linewidth(0.7)

# North arrow
ax_b.annotate('', xy=(0.93, 0.91), xytext=(0.93, 0.84),
              xycoords='axes fraction', textcoords='axes fraction',
              arrowprops=dict(arrowstyle='->', color='#222222', lw=1.2))
ax_b.text(0.93, 0.93, 'N', transform=ax_b.transAxes,
          fontsize=FS_ANNOT, ha='center', va='bottom',
          color='#222222', fontweight='bold')

# Scale bar — 500 m
sc500_m = 500 * (1.0 / np.cos(np.radians(RAS_LAT)))
sb2_x0 = b_xmin + (b_xmax - b_xmin) * 0.06
sb2_y0 = b_ymin + (b_ymax - b_ymin) * 0.07
ax_b.plot([sb2_x0, sb2_x0 + sc500_m], [sb2_y0, sb2_y0],
          'k-', lw=2.0, solid_capstyle='butt', zorder=9)
ax_b.text(sb2_x0 + sc500_m / 2, sb2_y0 + (b_ymax - b_ymin) * 0.025, '500 m',
          fontsize=FS_ANNOT - 1, ha='center', va='bottom', zorder=9,
          bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.7))

# Legend
leg_handles = [
    mpatches.Patch(facecolor=C_MNG, edgecolor='#1e5c2a', lw=0.8,
                   label='Mangrove extent (2016)'),
    mlines.Line2D([], [], color=C_ROI, lw=1.4, linestyle='-',
                  label='Study ROI (r = 2 km)'),
]
ax_b.legend(handles=leg_handles, loc='lower right', fontsize=FS_LEGEND,
            frameon=True, facecolor='white', edgecolor='#aaaaaa',
            framealpha=0.92, handlelength=1.6, handleheight=1.2,
            borderpad=0.6, labelspacing=0.4)

# Panel label
ax_b.text(0.02, 0.97, '(b)', transform=ax_b.transAxes,
          fontsize=FS_PANEL, fontweight='bold', va='top', ha='left',
          color='#111111', zorder=10,
          bbox=dict(boxstyle='square,pad=0.20', facecolor='white',
                    edgecolor='#888888', linewidth=0.5, alpha=0.85))
ax_b.set_title('Ras Sanad ROI and 2 km study radius',
               fontsize=FS_TITLE, fontweight='bold', pad=4, color='#111111')

# ── Save at 600 dpi ───────────────────────────────────────────────────────────
out = os.path.join(MAPS, 'Fig0_Study_Area.png')
fig.savefig(out, dpi=600, bbox_inches='tight', facecolor=C_BG)
plt.close(fig)

from PIL import Image
im = Image.open(out)
print(f'Saved: {out}')
print(f'  Size: {im.size[0]}x{im.size[1]} px  ({im.size[0]/600:.1f}" x {im.size[1]/600:.1f}")')
print('Done.')
