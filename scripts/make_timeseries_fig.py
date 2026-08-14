"""
Figure 4 — Mangrove Area Time Series & Inter-annual Change
Publication-quality rebuild for MDPI Remote Sensing. 600 dpi.
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
import matplotlib.ticker  as mticker

BASE   = r'C:\path\to\workspace\mangrove'
MAPS   = os.path.join(BASE, 'Ras_Sanad_Mangrove_Outputs', 'maps')
TABLES = os.path.join(BASE, 'Ras_Sanad_Mangrove_Outputs', 'tables')

TARGET_HA  = 34.05
HIST_AREAS = {1967: 97.30, 1998: 56.99, 2005: 37.54, 2009: 35.23, 2016: 34.05}

df = pd.read_excel(os.path.join(TABLES, 'Table_Annual_Areas_Fused.xlsx'))
df = df.sort_values('year').reset_index(drop=True)
sat_yrs = df['year'].values
fused   = df['area_fused'].values
s2      = df['area_s2'].values

# ── Color palette — muted, print-safe, grayscale-compatible ──────────────────
C_SURVEY  = '#555555'   # dark grey     — field surveys (neutral)
C_FUSED   = '#1a6faf'   # steel blue    — primary fused result
C_S2      = '#e07b39'   # muted amber   — S2-only (warm, clearly secondary)
C_REF     = '#aaaaaa'   # light grey    — 2016 reference line (subtle)
C_UNCERT  = '#1a6faf'   # match fused   — uncertainty band
C_GAIN    = '#4d9e4d'   # muted green   — area gain bars
C_LOSS    = '#c0392b'   # muted red     — area loss bars (conventional)
C_GAINFNT = '#2d6e2d'
C_LOSSFNT = '#922b1f'

# ── Font sizes ────────────────────────────────────────────────────────────────
FS_PL  = 12   # (a)/(b) panel label
FS_T   = 11   # panel title
FS_AX  = 10   # axis labels
FS_TK  =  9   # tick labels
FS_LEG =  9   # legend
FS_AN  =  8   # annotations

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':         FS_TK,
    'axes.labelsize':    FS_AX,
    'axes.titlesize':    FS_T,
    'xtick.labelsize':   FS_TK,
    'ytick.labelsize':   FS_TK,
    'legend.fontsize':   FS_LEG,
    'axes.grid':         True,
    'grid.color':        '#e0e0e0',
    'grid.linewidth':    0.5,
    'axes.axisbelow':    True,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'savefig.dpi':       600,
})

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))
fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.18, wspace=0.38)

# =============================================================================
# LEFT PANEL (a) — Mangrove Area Through Time
# =============================================================================
hx = sorted(HIST_AREAS.keys())
hy = [HIST_AREAS[y] for y in hx]

# Uncertainty band — drawn first, behind all lines
ax_l.fill_between(sat_yrs, fused - 2, fused + 2,
                  color=C_UNCERT, alpha=0.13, zorder=1)

# 2016 reference — subtle grey dashed, drawn behind data
ax_l.axhline(TARGET_HA, color=C_REF, ls='--', lw=1.2, zorder=2)

# S2-only — thin dotted amber, clearly secondary
ax_l.plot(sat_yrs, s2, color=C_S2, lw=1.1, ls=':', marker='o', ms=3,
          alpha=0.80, zorder=5)

# Historical survey — grey squares connected by thin dashes PER SEGMENT
# (avoids implying continuity across multi-decade gaps)
ax_l.plot(hx, hy, color=C_SURVEY, lw=0, marker='s', ms=5,
          mec=C_SURVEY, mfc='white', mew=1.3, zorder=7)
for i in range(len(hx) - 1):
    ax_l.plot([hx[i], hx[i+1]], [hy[i], hy[i+1]],
              color=C_SURVEY, lw=1.1, ls='--', alpha=0.65, zorder=6)

# Fused S2+S1 — solid blue, thickest, most prominent
ax_l.plot(sat_yrs, fused, color=C_FUSED, lw=2.0, marker='o', ms=4.5,
          zorder=8)

# Annotations — compact, arrows thin, text in provably empty regions
peak_yr  = int(sat_yrs[np.argmax(fused)])
peak_val = float(fused.max())
min_yr   = int(sat_yrs[np.argmin(fused)])
min_val  = float(fused.min())

ax_l.annotate(f'Peak {peak_val:.1f} ha',
              xy=(peak_yr, peak_val), xytext=(2011, 62),
              fontsize=FS_AN, color=C_FUSED,
              arrowprops=dict(arrowstyle='->', color=C_FUSED, lw=0.8,
                              connectionstyle='arc3,rad=-0.2'))
ax_l.annotate(f'Min {min_val:.1f} ha',
              xy=(min_yr, min_val), xytext=(2022, 24),
              fontsize=FS_AN, color=C_LOSS,
              arrowprops=dict(arrowstyle='->', color=C_LOSS, lw=0.8,
                              connectionstyle='arc3,rad=0.28'))

ax_l.set_xlim(1960, 2027)
ax_l.set_ylim(0, 110)
ax_l.set_xticks([1967, 1998, 2016, 2021, 2025])
ax_l.set_xticklabels(['1967', '1998', '2016', '2021', '2025'])
ax_l.set_xlabel('Year', labelpad=3)
ax_l.set_ylabel('Mangrove Area (ha)', labelpad=3)
ax_l.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax_l.yaxis.grid(True); ax_l.xaxis.grid(False)

ax_l.text(0.02, 0.97, '(a)', transform=ax_l.transAxes,
          fontsize=FS_PL, fontweight='bold', va='top', ha='left',
          color='#111111', zorder=10,
          bbox=dict(boxstyle='square,pad=0.20', facecolor='white',
                    edgecolor='#888888', linewidth=0.5, alpha=0.85))
ax_l.set_title('Mangrove Area Through Time', fontsize=FS_T,
               fontweight='bold', loc='left', pad=3)

# =============================================================================
# RIGHT PANEL (b) — Inter-annual Change
# =============================================================================
diffs  = np.diff(fused)
yrs_r  = sat_yrs[1:]
bcolors = [C_GAIN if v > 0 else C_LOSS for v in diffs]

bars = ax_r.bar(yrs_r, diffs, color=bcolors, edgecolor='white',
                linewidth=0.5, width=0.72, alpha=0.90, zorder=3)
ax_r.axhline(0, color='#333333', lw=0.8, zorder=4)

for bar, val in zip(bars, diffs):
    if abs(val) < 0.5:
        continue
    pad  = 0.10
    ypos = val + pad if val >= 0 else val - pad
    va   = 'bottom' if val >= 0 else 'top'
    col  = C_GAINFNT if val >= 0 else C_LOSSFNT
    ax_r.text(bar.get_x() + bar.get_width() / 2, ypos,
              f'{val:+.1f}', ha='center', va=va,
              fontsize=FS_AN, fontweight='bold', color=col)

ax_r.set_xlim(yrs_r[0] - 0.8, yrs_r[-1] + 0.8)
ax_r.set_ylim(-3.5, 7.2)
ax_r.set_xticks(yrs_r)
ax_r.set_xticklabels(yrs_r.astype(int), rotation=45, ha='right')
ax_r.set_xlabel('Year', labelpad=3)
ax_r.set_ylabel('Annual Change (ha)', labelpad=3)
ax_r.yaxis.set_major_locator(mticker.MultipleLocator(2))
ax_r.yaxis.grid(True); ax_r.xaxis.grid(False)

ax_r.text(0.02, 0.97, '(b)', transform=ax_r.transAxes,
          fontsize=FS_PL, fontweight='bold', va='top', ha='left',
          color='#111111', zorder=10,
          bbox=dict(boxstyle='square,pad=0.20', facecolor='white',
                    edgecolor='#888888', linewidth=0.5, alpha=0.85))
ax_r.set_title('Inter-annual Change', fontsize=FS_T,
               fontweight='bold', loc='left', pad=3)

# =============================================================================
# COMPACT LEGEND — single row beneath both panels
# =============================================================================
leg_handles = [
    mlines.Line2D([], [], color=C_SURVEY, lw=1.1, ls='--',
                  marker='s', ms=4.5, mec=C_SURVEY, mfc='white', mew=1.2,
                  label='Field survey (1967\u20132016)'),
    mlines.Line2D([], [], color=C_FUSED, lw=2.0, marker='o', ms=4.5,
                  label='Fused S2+S1 classifier'),
    mlines.Line2D([], [], color=C_S2, lw=1.1, ls=':', marker='o', ms=3,
                  alpha=0.80, label='S2-only classifier'),
    mpatches.Patch(facecolor=C_UNCERT, alpha=0.25,
                   label='\u00b12 ha uncertainty (fused)'),
    mlines.Line2D([], [], color=C_REF, lw=1.2, ls='--',
                  label=f'2016 reference ({TARGET_HA} ha)'),
    mpatches.Patch(facecolor=C_GAIN, label='Area gain'),
    mpatches.Patch(facecolor=C_LOSS, label='Area loss'),
]

fig.legend(handles=leg_handles,
           loc='lower center', bbox_to_anchor=(0.5, 0.0),
           ncol=7, fontsize=FS_LEG,
           frameon=True, facecolor='white', edgecolor='#cccccc',
           framealpha=0.95, handlelength=1.6, handleheight=1.0,
           borderpad=0.45, columnspacing=0.8, labelspacing=0.3)

out = os.path.join(MAPS, 'Fig_Area_TimeSeries.png')
fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
from PIL import Image
im = Image.open(out)
print(f'Fig 4 saved: {im.size[0]}x{im.size[1]} px  ({im.size[0]/600:.1f}" x {im.size[1]/600:.1f}")')
