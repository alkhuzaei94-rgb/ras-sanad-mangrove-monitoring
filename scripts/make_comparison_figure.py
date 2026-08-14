"""
_make_comparison_figure.py
Fig5_Historical_Comparison.png — Publication-quality 1967 vs 2025 comparison.
MDPI Remote Sensing, 600 dpi.

Design principles:
- Cartographically clean, minimal clutter
- 1967 panel: maximum historical extent, single polygon
- 2025 panel: current estimate, 1967 ghost clearly secondary (not distracting)
- No intermediate historical polygons (remove 1998/2005/2009 lines)
- Compact 4-item legend below, no overlap with map content
- ROI circle: subtle dashed outline, no fill
- North arrows: consistent small arrows, upper-right both panels
- Scale bars: lower-left both panels
- Light publication background throughout
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from shapely.geometry import Point
from pyproj import Transformer

BASE    = r'C:\Users\Manaf\Documents\Jupyter'
SHP_DIR = os.path.join(BASE, '_archive_root_duplicates')
MG      = os.path.join(BASE, 'mangrove', 'Ras_Sanad_Mangrove_Outputs')
MAPS    = os.path.join(MG, 'maps')
CACHE   = os.path.join(MG, 'array_cache')
import matplotlib.colors as mcolors

# ── Load shapefiles ────────────────────────────────────────────────────────────
print('Loading shapefiles...')
gdf67 = gpd.read_file(os.path.join(SHP_DIR, 'RasSanad_1967.shp')).to_crs(4326)
gdf16 = gpd.read_file(os.path.join(SHP_DIR, 'RasSanad_2016.shp')).to_crs(4326)

# 2025 predicted extent from the adopted August 2026 run (fused RF, T*=0.90).
# Display grid is EPSG:4326 at scale 10: cells are 10 m tall by 10*cos(lat) m
# wide, i.e. ~89.8 m2 each at 26.15 N.
pred25 = np.load(os.path.join(CACHE, 'pred_2025.npy'), allow_pickle=True)
_cell_m2 = 100.0 * np.cos(np.radians(26.14677))
raster_area_ha = float((pred25 > 0.5).sum()) * _cell_m2 / 10000.0
print(f'  pred_2025 display-raster area: {raster_area_ha:.2f} ha '
      f'(Table 5 analysis value: 34.91 ha)')

try:
    bhr = gpd.read_file(os.path.join(BASE, 'shared_data', 'boundaries', 'Bahrain.shp')).to_crs(4326)
    has_bhr = True
    print('  Bahrain shapefile loaded.')
except Exception:
    has_bhr = False
    print('  Bahrain shapefile not available.')

# ── ROI circle ─────────────────────────────────────────────────────────────────
RAS_LON, RAS_LAT = 50.5956, 26.14677
RADIUS_M = 1500
tfwd = Transformer.from_crs(4326, 32639, always_xy=True)
tinv = Transformer.from_crs(32639, 4326, always_xy=True)
cx, cy = tfwd.transform(RAS_LON, RAS_LAT)
theta = np.linspace(0, 2*np.pi, 360)
roi_lons = np.array([tinv.transform(cx + RADIUS_M*np.cos(t), cy + RADIUS_M*np.sin(t))[0] for t in theta])
roi_lats = np.array([tinv.transform(cx + RADIUS_M*np.cos(t), cy + RADIUS_M*np.sin(t))[1] for t in theta])

# ── Shared map extent (padded around 1967 polygon) ────────────────────────────
b67  = gdf67.total_bounds
b16  = gdf16.total_bounds
pad  = 0.003
XMIN = min(b67[0], b16[0]) - pad
XMAX = max(b67[2], b16[2]) + pad
YMIN = min(b67[1], b16[1]) - pad
YMAX = max(b67[3], b16[3]) + pad

# ── Color palette ──────────────────────────────────────────────────────────────
C_WATER   = '#c8dde8'   # muted sea blue
C_LAND    = '#ede9df'   # warm sand
C_MNG67   = '#1a6b2e'   # deep forest green — 1967
C_MNG25   = '#5aab6c'   # medium green — 2025 (lighter, clearly different shade)
C_GHOST   = '#1a6b2e'   # same hue, used for ghost outline
C_ROI     = '#555555'   # neutral grey — subtle ROI circle
C_BG      = 'white'

# Font sizes — figure 14×7": print scale at MDPI full-width (7") = 0.5×
# fontsize 20 → 10pt printed;  fontsize 14 → 7pt;  fontsize 11 → 5.5pt
FS_SUPTITLE = 18   # main figure title → 9pt
FS_SUBTITLE = 13   # net-loss subtitle → 6.5pt
FS_PANEL    = 20   # panel axis title (1967/2025) → 10pt
FS_ANNOT    = 16   # in-polygon area annotation → 8pt
FS_TICK     = 12   # geographic tick labels → 6pt
FS_LEG      = 13   # legend text → 6.5pt
FS_NORTH    = 11   # north arrow 'N' label → 5.5pt
FS_SCALE    = 10   # scale bar label → 5pt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': FS_TICK,
                     'figure.dpi': 150, 'savefig.dpi': 600})

# ── Helpers ────────────────────────────────────────────────────────────────────
def add_north_arrow(ax, x=0.93, y=0.88):
    """Compact north arrow — arrow then 'N' label."""
    ax.annotate('', xy=(x, y + 0.055), xytext=(x, y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.0))
    ax.text(x, y + 0.075, 'N', transform=ax.transAxes,
            fontsize=FS_NORTH, ha='center', va='bottom',
            color='#333333', fontweight='bold')

def add_scalebar(ax, xmin, xmax, ymin, ymax, bar_km=0.5):
    """Scale bar — lower-left, clean."""
    lat_mid = (ymin + ymax) / 2
    m_per_deg = 111320 * np.cos(np.radians(lat_mid))
    deg_bar   = (bar_km * 1000) / m_per_deg
    x0 = xmin + (xmax - xmin) * 0.06
    y0 = ymin + (ymax - ymin) * 0.055
    x1 = x0 + deg_bar
    # Bar line with end ticks
    ax.plot([x0, x1], [y0, y0], '-', color='#333333', lw=2.5,
            solid_capstyle='butt', zorder=20)
    tick_h = (ymax - ymin) * 0.004
    for xp in [x0, x1]:
        ax.plot([xp, xp], [y0 - tick_h, y0 + tick_h], '-',
                color='#333333', lw=1.5, zorder=20)
    ax.text((x0 + x1) / 2, y0 + (ymax - ymin) * 0.018,
            f'{int(bar_km * 1000)} m',
            ha='center', va='bottom', fontsize=FS_SCALE,
            fontweight='bold', color='#333333', zorder=20)

# ── Figure layout ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                         gridspec_kw={'wspace': 0.06,
                                      'left': 0.05, 'right': 0.97,
                                      'top': 0.87, 'bottom': 0.13})
fig.patch.set_facecolor(C_BG)

LAT_ASP = 1.0 / np.cos(np.radians(RAS_LAT))

for panel_idx, ax in enumerate(axes):
    # ── Background: water ─────────────────────────────────────────────────────
    ax.set_facecolor(C_WATER)

    # ── Land from shapefile ───────────────────────────────────────────────────
    if has_bhr:
        bhr.plot(ax=ax, color=C_LAND, edgecolor='#b0a880', linewidth=0.5, zorder=1)

    if panel_idx == 0:
        # ── LEFT PANEL: 1967 maximum extent ──────────────────────────────────
        gdf67.plot(ax=ax, color=C_MNG67, edgecolor='#0d4a1f',
                   linewidth=0.9, zorder=5, alpha=0.88)

        # ROI circle — subtle, no fill
        ax.plot(roi_lons, roi_lats, '--', color=C_ROI,
                lw=1.0, alpha=0.60, zorder=3)

        # Area label inside polygon
        cent = gdf67.geometry.union_all().centroid
        ax.text(cent.x, cent.y, '97.30 ha',
                ha='center', va='center', fontsize=FS_ANNOT, fontweight='bold',
                color='white',
                path_effects=[pe.withStroke(linewidth=2.0, foreground='#0d4a1f')],
                zorder=10)

    else:
        # ── RIGHT PANEL: 2025 estimate + 1967 ghost ───────────────────────────
        # 1967 ghost — very low alpha fill, dashed outline (secondary context only)
        gdf67.plot(ax=ax, color=C_GHOST, edgecolor='none',
                   linewidth=0, zorder=3, alpha=0.10)
        gdf67.plot(ax=ax, color='none', edgecolor=C_GHOST,
                   linewidth=1.2, linestyle='--', zorder=4, alpha=0.45)

        # 2025 predicted extent: the actual fused-RF classification raster
        # (August 2026 run, T* = 0.90), not a proxy polygon.
        _dlat25 = 2000.0 / 111320.0
        _dlon25 = 2000.0 / (111320.0 * np.cos(np.radians(RAS_LAT)))
        PRED_EXT = [RAS_LON - _dlon25, RAS_LON + _dlon25,
                    RAS_LAT - _dlat25, RAS_LAT + _dlat25]
        rgba25 = np.zeros((*pred25.shape, 4), dtype=np.float32)
        rgba25[pred25 > 0.5] = mcolors.to_rgba(C_MNG25, alpha=0.90)
        ax.imshow(rgba25, extent=PRED_EXT, origin='lower', aspect='auto', zorder=5)
        ax.contour(pred25, levels=[0.5], colors=['#1a6b2e'], linewidths=[0.7],
                   extent=PRED_EXT, origin='lower', zorder=6)

        # ROI circle — subtle, no fill
        ax.plot(roi_lons, roi_lats, '--', color=C_ROI,
                lw=1.0, alpha=0.60, zorder=3)

        # Area label placed near the main stand (2016 polygon centroid)
        cent = gdf16.geometry.union_all().centroid
        ax.text(cent.x, cent.y, '34.91 ha',
                ha='center', va='center', fontsize=FS_ANNOT, fontweight='bold',
                color='white',
                path_effects=[pe.withStroke(linewidth=2.0, foreground='#1a6b2e')],
                zorder=10)

    # ── Map extent and aspect ─────────────────────────────────────────────────
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect(LAT_ASP, adjustable='datalim')

    # ── Geographic ticks ──────────────────────────────────────────────────────
    lon_step = 0.01
    lat_step = 0.01
    lon_ticks = sorted([t for t in np.arange(round(XMIN, 2), XMAX + 0.001, lon_step)
                        if XMIN < t < XMAX])
    lat_ticks = sorted([t for t in np.arange(round(YMIN, 2), YMAX + 0.001, lat_step)
                        if YMIN < t < YMAX])
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xticklabels([f'{t:.2f}\u00b0E' for t in lon_ticks], fontsize=FS_TICK)
    if panel_idx == 0:
        ax.set_yticklabels([f'{t:.2f}\u00b0N' for t in lat_ticks],
                           fontsize=FS_TICK, rotation=90, va='center')
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', length=2.0, width=0.5, pad=2.0)

    # ── Spines ────────────────────────────────────────────────────────────────
    for sp in ax.spines.values():
        sp.set_edgecolor('#aaaaaa')
        sp.set_linewidth(0.7)

    # ── Panel axis title ──────────────────────────────────────────────────────
    year_str = '1967' if panel_idx == 0 else '2025'
    area_str = '97.30 ha' if panel_idx == 0 else '34.91 ha'
    ax.set_title(f'{year_str}  \u2013  {area_str}',
                 fontsize=FS_PANEL, fontweight='bold', pad=6, color='#111111')

    # ── Map furniture ─────────────────────────────────────────────────────────
    add_north_arrow(ax, x=0.93, y=0.87)
    add_scalebar(ax, XMIN, XMAX, YMIN, YMAX, bar_km=0.5)

# ── Figure-level title and subtitle ────────────────────────────────────────────
fig.text(0.50, 0.98, 'Ras Sanad Mangrove Extent Change, 1967\u20132025',
         ha='center', va='top',
         fontsize=FS_SUPTITLE, fontweight='bold', color='#111111')
fig.text(0.50, 0.945, 'Net difference: 62.39 ha (\u221264.1%)',
         ha='center', va='top',
         fontsize=FS_SUBTITLE, color='#555555', style='italic')

# ── Compact legend — 4 items, single row below panels ──────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=C_MNG67, edgecolor='#0d4a1f', alpha=0.88,
                   label='Mangrove 1967 (97.30 ha)'),
    mpatches.Patch(facecolor=C_MNG25, edgecolor='#1a6b2e', alpha=0.90,
                   label='Mangrove 2025, RF classification (34.91 ha)'),
    mpatches.Patch(facecolor=C_GHOST, edgecolor=C_GHOST, alpha=0.12,
                   label='1967 lost area (ghost)'),
    Line2D([0], [0], color=C_ROI, linewidth=1.0, linestyle='--',
           label='Study ROI (r = 1.5 km)'),
]
fig.legend(handles=legend_elements,
           loc='lower center', bbox_to_anchor=(0.50, 0.01),
           ncol=4, fontsize=FS_LEG, frameon=True,
           facecolor='white', edgecolor='#cccccc', framealpha=0.95,
           handlelength=1.8, handleheight=1.2,
           borderpad=0.6, labelspacing=0.4, columnspacing=1.0)

# ── Save ────────────────────────────────────────────────────────────────────────
out = os.path.join(MAPS, 'Fig4_Historical_Comparison.png')
fig.savefig(out, dpi=600, bbox_inches='tight', facecolor=C_BG)
plt.close(fig)
from PIL import Image
im = Image.open(out)
print(f'Saved: {out}')
print(f'  Size: {im.size[0]}\u00d7{im.size[1]} px  ({im.size[0]/600:.1f}" \u00d7 {im.size[1]/600:.1f}")')
print('Done.')
