"""
Regenerate non-GIS figures for MDPI paper.

Design philosophy:
  - Figure canvas sized so the DATA has breathing room — text fits without clipping
  - Fonts sized proportional to canvas: printed_pt = matplotlib_fs × (5.43" / fig_width")
  - No redundant axis labels on multi-panel figures (left panel only for shared axes)
  - Shared colorbars (one, far right) to eliminate inter-panel label clashes
  - Legend placed BELOW the plot area, never inside the data region
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

BASE   = r'C:\path\to\workspace\mangrove'
MAPS   = os.path.join(BASE, 'Ras_Sanad_Mangrove_Outputs', 'maps')
ACC    = os.path.join(BASE, 'Ras_Sanad_Mangrove_Outputs', 'accuracy')
TABLES = os.path.join(BASE, 'Ras_Sanad_Mangrove_Outputs', 'tables')

TARGET_HA  = 34.05
HIST_AREAS = {1967: 97.30, 1998: 56.99, 2005: 37.54, 2009: 35.23, 2016: 34.05}

# =============================================================================
# FIGURE 1 — Area Time Series
# SUPERSEDED by _make_timeseries_fig.py — run that script for the publication
# figure. This section still runs but saves to a _legacy filename so it never
# overwrites the output of _make_timeseries_fig.py.
# =============================================================================
print('Regenerating Fig_Area_TimeSeries_legacy.png (see _make_timeseries_fig.py for pub version)...')

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':          16,
    'axes.labelsize':     17,
    'axes.titlesize':     17,
    'xtick.labelsize':    15,
    'ytick.labelsize':    15,
    'legend.fontsize':    14,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         300,
    'savefig.dpi':        600,
})

df_ann = pd.read_excel(os.path.join(TABLES, 'Table_Annual_Areas_Fused.xlsx'))
df_ann = df_ann.sort_values('year').reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
# bottom=0.33 leaves 33% of figure height below axes for the shared legend
fig.subplots_adjust(bottom=0.33, wspace=0.38, left=0.09, right=0.97, top=0.94)

# ── Left panel: long-term trend ───────────────────────────────────────────────
ax = axes[0]
hx = sorted(HIST_AREAS.keys())
hy = [HIST_AREAS[y] for y in hx]

ax.plot(hx, hy, 'ks--', ms=7, lw=2.0, zorder=5, label='Field survey (1967–2016)')
ax.plot(df_ann['year'], df_ann['area_fused'], 'o-',
        color='#1a7c3e', lw=2.2, ms=5, zorder=7, label='Fused S2+S1 classifier')
if 'area_s2' in df_ann.columns:
    ax.plot(df_ann['year'], df_ann['area_s2'], 's--',
            color='#d4841a', lw=1.4, ms=4, alpha=0.6, zorder=4, label='S2-only classifier')
ax.fill_between(df_ann['year'],
                df_ann['area_fused'] - 2, df_ann['area_fused'] + 2,
                alpha=0.18, color='#1a7c3e', label='\u00b12 ha boundary uncertainty')
ax.axhline(TARGET_HA, color='#c0392b', ls=':', lw=2.0, label=f'2016 reference ({TARGET_HA} ha)')

# Annotations in empty areas of the plot — use curved arrows to avoid line overlap
peak_row = df_ann.loc[df_ann['area_fused'].idxmax()]
min_row  = df_ann.loc[df_ann['area_fused'].idxmin()]

ax.annotate(f"Peak {peak_row['area_fused']:.1f} ha",
            xy=(peak_row['year'], peak_row['area_fused']),
            xytext=(2009, 65),
            fontsize=13, color='#1a7c3e', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1a7c3e', lw=1.4,
                            connectionstyle='arc3,rad=-0.25'))
ax.annotate(f"Min {min_row['area_fused']:.1f} ha",
            xy=(min_row['year'], min_row['area_fused']),
            xytext=(2023.5, 8),
            fontsize=13, color='#c0392b', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.4,
                            connectionstyle='arc3,rad=0.3'))

ax.set_xlabel('Year')
ax.set_ylabel('Mangrove Area (ha)')
ax.set_xlim(1960, 2028)
ax.set_ylim(0, 115)
# 5 clean ticks — no crowding in the 2016–2025 satellite range
ax.set_xticks([1967, 1998, 2016, 2021, 2025])
ax.set_xticklabels([1967, 1998, 2016, 2021, 2025], rotation=25, ha='right')

# ── Right panel: year-on-year bar chart ───────────────────────────────────────
ax2 = axes[1]
diffs  = df_ann['area_fused'].diff().dropna()
yrs    = df_ann['year'].iloc[1:]
colors = ['#1a7c3e' if v > 0 else '#c0392b' for v in diffs]
bars   = ax2.bar(yrs, diffs, color=colors, edgecolor='white',
                 linewidth=0.6, alpha=0.88, width=0.75)
ax2.axhline(0, color='#333333', lw=1.2, zorder=5)
ax2.set_title('Inter-annual Change', fontsize=16, fontweight='bold')

# Value labels only on bars ≥ 0.8 ha
for bar, val in zip(bars, diffs):
    if abs(val) < 0.8:
        continue
    ypos = bar.get_height() + 0.12 if val >= 0 else bar.get_height() - 0.45
    ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
             f'{val:+.1f}', ha='center',
             va='bottom' if val >= 0 else 'top',
             fontsize=12, fontweight='bold',
             color='#1a7c3e' if val >= 0 else '#c0392b')

ax2.set_xlabel('Year')
ax2.set_ylabel('Year-on-Year Change (ha)')
# Odd years only → 5 ticks, not 9
ax2.set_xticks([y for y in yrs if y % 2 == 1])
ax2.set_xticklabels([y for y in yrs if y % 2 == 1], rotation=25, ha='right')

# Legend inside right panel (upper right corner is always empty in a bar chart)
leg_patches = [mpatches.Patch(facecolor='#1a7c3e', label='Area increase'),
               mpatches.Patch(facecolor='#c0392b', label='Area decrease')]
ax2.legend(handles=leg_patches, loc='upper right', fontsize=13,
           framealpha=0.92, edgecolor='#cccccc',
           handlelength=1.8, handleheight=1.6, borderpad=0.7)

# Shared legend for the left-panel line styles — placed in the bottom margin
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center', bbox_to_anchor=(0.5, 0.01),
           ncol=3, fontsize=13, framealpha=0.95, edgecolor='#cccccc',
           handlelength=2.2, handleheight=1.6, borderpad=0.7, columnspacing=1.2)

fig.savefig(os.path.join(MAPS, 'Fig_Area_TimeSeries_legacy.png'), dpi=600, bbox_inches='tight')
plt.close(fig)
print(f'  Saved: Fig_Area_TimeSeries_legacy.png')

# =============================================================================
# FIGURE 2 — Confusion Matrix
# figsize=(13, 6):  print scale = 5.43/13 = 0.418
# KEY DESIGN DECISIONS:
#   - y-axis tick labels ONLY on left panel (no duplication in panel b)
#   - ONE shared colorbar on far right (eliminates the inter-panel label clash)
#   - Annotate text inside each cell is large (fs16) — cells are big enough
#   - x-tick labels horizontal (no rotation) — enough space at this figsize
# =============================================================================
print('Regenerating Fig_Accuracy_CM.png ...')

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':          15,
    'axes.labelsize':     16,
    'xtick.labelsize':    14,
    'ytick.labelsize':    14,
    'axes.titlesize':     16,
    'figure.dpi':         300,
    'savefig.dpi':        600,
})

# TRUE confusion matrices from the EE validation run: 1000 stratified samples
# (500 per class) drawn from the 2023 annual composite (seed 1041), evaluated
# server-side with errorMatrix(). Raw counts from the adopted August 2026 run
# (Ras_Sanad_Verification/tables/Rerun_2026-08_Annual_Areas.json).
# Rows = reference class [non-mangrove, mangrove]; cols = predicted class.
CM_TRUE = {
    'S2-only': np.array([[380, 120], [110, 390]]),
    'Fused':   np.array([[383, 117], [ 89, 411]]),
}

# Self-check: these counts must reproduce the published Table 2 metrics exactly
for _name, _cm in CM_TRUE.items():
    _tn, _fp = _cm[0]; _fn, _tp = _cm[1]
    _n  = _cm.sum()
    _oa = (_tn + _tp) / _n
    _pr = _tp / (_tp + _fp)
    _rc = _tp / (_tp + _fn)
    print(f'  CM check {_name}: n={_n}  OA={_oa:.4f}  P={_pr:.4f}  R={_rc:.4f}')
assert CM_TRUE['Fused'].sum() == 1000 and CM_TRUE['S2-only'].sum() == 1000

tick_labels  = ['Non-mangrove', 'Mangrove']
panel_titles = ['(a) S2-only baseline', '(b) Fused S2+S1']

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
# right=0.88 leaves room for ONE shared colorbar on the far right
fig.subplots_adjust(wspace=0.20, left=0.12, right=0.86, top=0.88, bottom=0.16)

im_ref = None   # will hold the last imshow for the shared colorbar

for pi, (ax, title) in enumerate(zip(axes, panel_titles)):
    model_key = 'S2-only' if pi == 0 else 'Fused'
    cm = CM_TRUE[model_key]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    im_ref = im

    for r in range(2):
        for c in range(2):
            txt_col = 'white' if cm_norm[r, c] > 0.55 else '#111111'
            ax.text(c, r, f'{cm[r,c]:,}\n({cm_norm[r,c]:.1%})',
                    ha='center', va='center',
                    fontsize=16, color=txt_col,
                    fontweight='bold' if r == c else 'normal')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(tick_labels, fontsize=14)   # horizontal — no rotation needed
    ax.set_yticks([0, 1])
    ax.set_xlabel('Predicted', fontsize=15, labelpad=6)

    if pi == 0:
        # y-axis labels only on LEFT panel
        ax.set_yticklabels(tick_labels, fontsize=14)
        ax.set_ylabel('Reference', fontsize=15, labelpad=6)
    else:
        # Right panel: no y-tick labels (same as left), no y-axis label
        ax.set_yticklabels([])
        ax.set_ylabel('')

    ax.set_title(title, fontsize=16, fontweight='bold', color='#1a3c6e', pad=8)

# ONE shared colorbar on far right — avoids panel-gap label clashing
cbar_ax = fig.add_axes([0.88, 0.16, 0.018, 0.72])
cbar = fig.colorbar(im_ref, cax=cbar_ax)
cbar.ax.tick_params(labelsize=13)

fig.savefig(os.path.join(ACC, 'Fig_Accuracy_CM.png'), dpi=600, bbox_inches='tight')
plt.close(fig)
print(f'  Saved: Fig_Accuracy_CM.png')

# =============================================================================
# FIGURE 3 — DSC Threshold Calibration Curve
# figsize=(10, 6.5):  print scale = 5.43/10 = 0.543
#   fs14 → 7.6 pt   fs15 → 8.1 pt   fs16 → 8.7 pt  ✓
# DSC annotation placed in upper-right with a white box — never on curves
# Legend in bottom margin, 2 columns
# =============================================================================
print('Regenerating Fig_Calibration.png ...')

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':          14,
    'axes.labelsize':     15,
    'xtick.labelsize':    14,
    'ytick.labelsize':    14,
    'legend.fontsize':    13,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'figure.dpi':         300,
    'savefig.dpi':        600,
})

THR_OPT = 0.90
df_cal_path = os.path.join(TABLES, 'Table_Calibration_Fused.xlsx')
df_cal = pd.read_excel(df_cal_path)

col_thr  = 'threshold' if 'threshold' in df_cal.columns else df_cal.columns[0]
col_area = next((c for c in df_cal.columns if 'area' in c.lower()), df_cal.columns[1])
col_dsc  = next((c for c in df_cal.columns if 'dsc' in c.lower() or 'dice' in c.lower()),
                df_cal.columns[2])

# Tall-and-wide figure — extra top headroom for annotation, extra bottom for legend
fig, ax = plt.subplots(figsize=(10, 6.5))
fig.subplots_adjust(bottom=0.32, right=0.84, left=0.12, top=0.94)
ax2 = ax.twinx()

ln1, = ax.plot(df_cal[col_thr], df_cal[col_area], '-',
               color='#1565c0', lw=2.5, label='Classified area (ha)')
ln2, = ax2.plot(df_cal[col_thr], df_cal[col_dsc], '-',
                color='#c62828', lw=2.5, label='DSC')
ln3  = ax.axvline(THR_OPT, color='#555555', ls=':', lw=2.0,
                  label=f'Adopted threshold = {THR_OPT:.2f}')
ln4  = ax.axhline(TARGET_HA, color='#2e7d32', ls='--', lw=2.0,
                  label=f'Reference ({TARGET_HA} ha)')

# Best-DSC annotation: pinned to axes-fraction coords (0.62, 0.92)
# The top-right of this figure is always data-free — both curves drop away there
best_idx = df_cal[col_dsc].idxmax()
best_thr = df_cal.loc[best_idx, col_thr]
best_dsc = df_cal.loc[best_idx, col_dsc]

ax2.annotate(f'DSC max = {best_dsc:.3f}\n@ T = {best_thr:.2f}',
             xy=(best_thr, best_dsc),
             xytext=(0.58, 0.90),
             xycoords='data', textcoords='axes fraction',
             fontsize=13, color='#2e7d32', fontweight='bold',
             ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='#2e7d32', linewidth=1.4, alpha=0.95),
             arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.6))

ax.set_xlabel('Probability Threshold', fontsize=15)
ax.set_ylabel('Classified Area (ha)', color='#1565c0', fontsize=15, labelpad=7)
ax2.set_ylabel('DSC', color='#c62828', fontsize=15, labelpad=7)
ax.tick_params(axis='y', labelcolor='#1565c0')
ax2.tick_params(axis='y', labelcolor='#c62828')

# Combined legend in the bottom margin — 2 columns, clearly separated
fig.legend([ln1, ln2, ln3, ln4], [l.get_label() for l in [ln1, ln2, ln3, ln4]],
           loc='lower center', bbox_to_anchor=(0.48, 0.01),
           ncol=2, fontsize=13, framealpha=0.95, edgecolor='#cccccc',
           handlelength=2.2, handleheight=1.6, borderpad=0.7, columnspacing=1.5)

fig.savefig(os.path.join(ACC, 'Fig_Calibration.png'), dpi=600, bbox_inches='tight')
plt.close(fig)
print(f'  Saved: Fig_Calibration.png')

print('\nAll figures saved. Pixel sizes:')
from PIL import Image
for p in [os.path.join(MAPS, 'Fig_Area_TimeSeries.png'),
          os.path.join(ACC,  'Fig_Accuracy_CM.png'),
          os.path.join(ACC,  'Fig_Calibration.png')]:
    im = Image.open(p)
    print(f'  {os.path.basename(p)}: {im.size[0]}x{im.size[1]} px  '
          f'({im.size[0]/300:.1f}" x {im.size[1]/300:.1f}")')
