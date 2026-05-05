# xFP Pipeline Scripts

## Production / Re-runnable

- **xfp_v11_lock.py** — Train V11, save bundle, build 2026 projections, build dashboard.
  Run when retraining is needed (rare: only on new feature additions or major data refresh).
  Usage: `python scripts/xfp/xfp_v11_lock.py`
- **xfp_v8_midseason.py** — V8.1 mid-season blend (sample-weighted 2025+2026 inputs). Re-run every
  2-3 weeks during the season as 2026 stats accumulate.
- **xfp_v8_5_pipeline.py** — V8.5 training pipeline (reference / superseded baseline). Imports
  helper functions used by V11 lock.
- **build_sp_multiyr.py** — Re-aggregates Statcast cache (`data/research/xfp_cache/statcast_*.parquet`)
  into the per-pitcher per-season training panel. Run if Statcast pulls are added/refreshed.
- **pull_fg_undetected.py** — FanGraphs Stuff+/Pitching+/PitchingBot pull via Chrome 147 visible mode.
  ~30 seconds per year. Required for V11 retraining and projection input updates.
  Usage: `python scripts/xfp/pull_fg_undetected.py` (requires Chrome 147, visible browser).

## Modules / Helpers

- **xfp_v7_pipeline.py** — Core eval functions (cross_year_evaluate, ooy_evaluate, derive_features,
  add_ip_resid_lag). Used as a module by all later scripts.
- **xfp_v8_pipeline.py** — V8 pipeline + helper builds (derive_v8_features, build_pitch_type_panel,
  score_fn). Used as a module.
- **v11_spotcheck.py** — `build_blended_inputs()` helper used by V11 lock and full spot-check.

## Analysis / Research

- **v11_full_spotcheck.py** — Projection-level spot check of V11 vs V8.5 on top-25 high-K cohort.
  Re-runnable verification that V11 doesn't over-project elite-stuff pitchers.
- **xfp_rolling_ip.py** — Rolling-last-5-starts IP predictor analysis (re-runnable as season progresses).
- **compare_score_formulas.py** — What-if score-formula comparison (linear coef, tolerance threshold,
  quadratic). Reads hard-coded V11/V12 results.

## Archive (`scripts/xfp/archive/`)
Negative results and superseded experiments. See `data/research/xfp_model_research.md` for context.
- `xfp_v9_pipeline.py` — V9 IP decomposition (failed)
- `xfp_v10_pipeline.py` — V10 three-branch search (Marcel / archetype / BaseRuns; all failed)
- `xfp_v12_residual.py` — V12 residual correction (failed)
- `xfp_v7_finalize.py` — V7 finalize standalone
- `xfp_v8_ensemble.py` — V8 ensemble experiment
- `pull_fg_history.py`, `pull_fg_exotic.py`, `pull_fg_playwright.py`, `pull_2015_2020.py` —
  earlier FG pull experiments before undetected-chromedriver succeeded.

## In-season refresh order

```bash
# 1. Refresh Statcast 2026 (pulls latest week's data)
python scripts/xfp/build_sp_multiyr.py

# 2. Refresh FanGraphs 2026 (Pitching+ updates as new starts pile up)
python scripts/xfp/pull_fg_undetected.py

# 3. Re-blend 2025+2026 inputs and re-project (fast — does not retrain)
python scripts/xfp/xfp_v11_lock.py
```

V11 retraining is NOT needed in-season — only the V8.1 blend changes per pitcher's accumulating 2026 stats.
