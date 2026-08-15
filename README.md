# Ras Sanad Mangrove Monitoring (2016-2025)

Analysis code for the manuscript:

> Decadal Stability of Ras Sanad Mangroves Under Protected Status:
> Multi-Sensor Satellite Assessment and Conservation Policy Implications,
> Kingdom of Bahrain 2016-2025.
> Submitted to *Remote Sensing* (MDPI), manuscript `remotesensing-4462496`.

The scripts in this repository are the code that produced every number, table,
and figure in the manuscript. They are published as run, with two exceptions
noted in the commit history: a synthetic-fallback branch was removed from the
calibration figure script, and dead code was removed after a deliberate early
exit in the map figure script. Fixed random seeds are declared throughout;
re-running the workflow with the same seeds on the same archive state
reproduces the results exactly. The manuscript's analyses were executed in
August 2026; because the Copernicus archive is occasionally reprocessed,
re-runs at a later date may differ at the level of a few percent.

## What each script does

| Script | Purpose |
|---|---|
| `scripts/verify_stability.py` | Full pipeline: training data, Random Forest training and tuning, threshold calibration, annual areas 2016-2025, validation, reproducibility checks. |
| `scripts/scene_counts.py` | Per-year Sentinel-2 and Sentinel-1 scene counts and cloud statistics (manuscript Table 2). |
| `scripts/threshold_sweep.py` | Extended DSC threshold sweep, 0.30-1.00 with 0.01 steps above 0.80 (manuscript Table 4, Figure 3). |
| `scripts/rerun_annual_areas.py` | Regenerates both classifiers, the 2023 validation matrices, and annual areas at T = 0.90 and 0.92 (manuscript Tables 3 and 6). |
| `scripts/adopt_august_run.py` | Writes the canonical tables that the figure scripts read. |
| `scripts/regen_map_figures.py` | Fetches 10 m display arrays and renders the annual classification, polygon, and probability map figures. |
| `scripts/regen_nongeo_figs.py` | Confusion-matrix and calibration-curve figures. |
| `scripts/make_timeseries_fig.py` | Annual area time-series figure. |
| `scripts/make_comparison_figure.py` | 1967 vs 2025 extent comparison figure. |
| `scripts/make_study_area_fig.py` | Study-area location figure. |

Suggested run order: `scene_counts` → `verify_stability` →
`threshold_sweep` → `rerun_annual_areas` → `adopt_august_run` → figure
scripts.

## Requirements

Python 3.11+ with the packages in `requirements.txt`, and an authenticated
[Google Earth Engine](https://earthengine.google.com/) account. Set the
`EE_PROJECT` environment variable to your own Earth Engine Cloud project, and
point the path constants at the top of each script (`BASE`, `SHP_DIR`,
`OUT_DIR` or equivalents) at your local directories; both carry neutral
placeholders. Apart from these placeholders, the scripts are published
exactly as run.

## Data

Satellite inputs are fetched at run time from the public Earth Engine
catalogue: `COPERNICUS/S2_SR_HARMONIZED`, `COPERNICUS/S1_GRD`,
`JRC/GSW1_4/GlobalSurfaceWater`, and `USGS/SRTMGL1_003`.

The historical Ras Sanad survey polygons (`RasSanad_1967/1998/2005/2009/2016`
shapefiles) originate from independent previous research and are used with
permission; they are not ours to redistribute and are not included here.
Requests concerning these data may be addressed to the corresponding author
of the manuscript, who can refer them to the data owners. All derived
numerical results appear as tables in the manuscript.

## Citation

If you use this code, please cite the manuscript above once published.

## License

MIT (see `LICENSE`). The license covers the code only, not the third-party
survey data described above.
