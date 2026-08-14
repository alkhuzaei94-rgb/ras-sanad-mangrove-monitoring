"""
_adopt_august_run.py
====================
Adopts the August 2026 rerun as the single source of truth for every number in
the resubmission. Rewrites the canonical output tables that the figure scripts
and the paper builder read, keeping the April 2026 versions as *_apr2026
backups for provenance.

Sources (committed):
  Ras_Sanad_Verification/tables/Rerun_2026-08_Annual_Areas.json   (areas + CMs)
  Ras_Sanad_Verification/tables/Table_Calibration_Extended.json   (DSC sweep)

Targets:
  Ras_Sanad_Mangrove_Outputs/tables/Table_Annual_Areas_Fused.xlsx
  Ras_Sanad_Mangrove_Outputs/tables/Table_Calibration_Fused.xlsx
  Ras_Sanad_Mangrove_Outputs/accuracy/Table_Accuracy_Metrics.xlsx
  Ras_Sanad_Verification/accuracy/Table_Accuracy_Metrics.xlsx

Adopted threshold: T* = 0.90 (DSC plateau 0.89-0.93 + minimal area bias vs the
2016 reference; see RESUBMISSION_PLAN.md).

Run:
  & "C:\\ProgramData\\anaconda3\\envs\\geoai_rs\\python.exe" _adopt_august_run.py
"""
import sys, os, json, shutil
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd

BASE = r'C:\path\to\workspace\mangrove'
VTAB = os.path.join(BASE, 'Ras_Sanad_Verification', 'tables')
VACC = os.path.join(BASE, 'Ras_Sanad_Verification', 'accuracy')
OTAB = os.path.join(BASE, 'Ras_Sanad_Mangrove_Outputs', 'tables')
OACC = os.path.join(BASE, 'Ras_Sanad_Mangrove_Outputs', 'accuracy')

with open(os.path.join(VTAB, 'Rerun_2026-08_Annual_Areas.json')) as f:
    rerun = json.load(f)
with open(os.path.join(VTAB, 'Table_Calibration_Extended.json')) as f:
    sweep = json.load(f)


def backup(path):
    root, ext = os.path.splitext(path)
    bak = f'{root}_apr2026{ext}'
    if os.path.exists(path) and not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f'  backup: {os.path.basename(bak)}')


def cm_metrics(cm):
    (tn, fp), (fn, tp) = cm
    n = tn + fp + fn + tp
    oa = (tn + tp) / n
    pe = ((tn + fp) / n) * ((tn + fn) / n) + ((fn + tp) / n) * ((fp + tp) / n)
    kap = (oa - pe) / (1 - pe)
    p_mg = tp / (tp + fp)
    r_mg = tp / (tp + fn)
    f_mg = 2 * p_mg * r_mg / (p_mg + r_mg)
    p_no = tn / (tn + fn)
    r_no = tn / (tn + fp)
    f_no = 2 * p_no * r_no / (p_no + r_no)
    return dict(OA=round(oa, 3), Kappa=round(kap, 3),
                F1_mg=round(f_mg, 4), Precision=round(p_mg, 4),
                Recall=round(r_mg, 4), F1_non=round(f_no, 4))


# ── 1. Annual areas ───────────────────────────────────────────────────────────
print('1. Table_Annual_Areas_Fused.xlsx')
tgt = os.path.join(OTAB, 'Table_Annual_Areas_Fused.xlsx')
backup(tgt)
rows = []
for r in rerun['annual_areas']:
    rows.append({'year': r['year'],
                 'area_fused': r['fused_T0.90'],
                 'model': 'Fused S2+S1',
                 'area_s2': r['s2_T0.90'],
                 'area_hist': 34.05 if r['year'] == 2016 else None})
df_ann = pd.DataFrame(rows)
df_ann.to_excel(tgt, index=False)
print(df_ann[['year', 'area_fused', 'area_s2']].to_string(index=False))

# ── 2. Calibration sweep ──────────────────────────────────────────────────────
print('\n2. Table_Calibration_Fused.xlsx (extended, 31 thresholds)')
tgt = os.path.join(OTAB, 'Table_Calibration_Fused.xlsx')
backup(tgt)
df_cal = pd.DataFrame([{'thr': r['thr'], 'area_ha': r['area_ha'], 'Dice': r['Dice']}
                       for r in sweep['rows']])
df_cal.to_excel(tgt, index=False)
print(f'  {len(df_cal)} rows, DSC max {df_cal.Dice.max():.4f} at '
      f'thr={df_cal.loc[df_cal.Dice.idxmax(), "thr"]:.2f}; '
      f'adopted T*=0.90 (Dice {df_cal.loc[df_cal.thr == 0.90, "Dice"].iloc[0]:.4f}, '
      f'area {df_cal.loc[df_cal.thr == 0.90, "area_ha"].iloc[0]:.2f} ha)')

# ── 3. Accuracy metrics ───────────────────────────────────────────────────────
print('\n3. Table_Accuracy_Metrics.xlsx (both copies)')
val = rerun['validation_2023']
recs, recs_ver = [], []
for name, key in [('Fused S2+S1', 'Fused'), ('S2-only', 'S2-only')]:
    m = cm_metrics(val[key]['CM'])
    sanity = {'OA': val[key]['OA'], 'Kappa': val[key]['Kappa'],
              'F1_mg': val[key]['F1_mg']}
    assert abs(m['OA'] - sanity['OA']) < 5e-4, (name, m, sanity)
    assert abs(m['Kappa'] - sanity['Kappa']) < 5e-4, (name, m, sanity)
    recs.append({'Model': name, 'OA': m['OA'], 'Kappa': m['Kappa'],
                 'F1_mg': m['F1_mg'], 'Precision': m['Precision'],
                 'Recall': m['Recall']})
    recs_ver.append({'Model': name, 'n_trees': 150, 'Seed': 42, 'OA': m['OA'],
                     'Kappa': m['Kappa'], 'F1_mg': m['F1_mg'],
                     'Precision': m['Precision'], 'Recall': m['Recall'],
                     'F1_non': m['F1_non']})
    print(f'  {name}: CM={val[key]["CM"]}  ' +
          '  '.join(f'{k}={v}' for k, v in m.items()))

for tgt, data in [(os.path.join(OACC, 'Table_Accuracy_Metrics.xlsx'), recs),
                  (os.path.join(VACC, 'Table_Accuracy_Metrics.xlsx'), recs_ver)]:
    backup(tgt)
    pd.DataFrame(data).to_excel(tgt, index=False)

print('\nDone. April backups sit beside each table as *_apr2026.xlsx.')
