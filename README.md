# SP xFP Model — Fantasy Points Projector

Predicts per-start ESPN fantasy points for MLB starting pitchers using
Statcast metrics + FanGraphs Pitching+ features. Built for fantasy baseball
roster management.

**Current model:** V11 (V8.5 + pitching_plus + fp_strike_pct)
Cross-year r = 0.614 | 2026 YTD r = 0.511 | MAE = 3.393 FP/start

## Live Dashboard
[View 2026 SP Projections →](https://Kejjeh.github.io/xfp-model/)

## Model Summary
- ESPN scoring: `K×1 + IP×3.3 − H×1 − ER×2 − BB×1 − HBP×1`
- Non-circular constraint: no per-start K/IP/H/ER/BB/HBP as direct features
- Ridge regression with StandardScaler, 10-fold cross-validated alpha selection
- Trained on 2020-2025 SP-seasons (n=768, limited by Pitching+ availability)
- Validated via expand-window cross-year evaluation (9 transitions, 2015-2025)
- Mid-season blend: 2026 YTD Statcast weighted in by pitch count (V8.1 layer)

## V11 features (14 total)

| Category | Features |
|---|---|
| Plate discipline | swstr_pct, c_plus_swstr, o_swing_pct, z_swing_pct, zone_pct, fp_strike_pct |
| Contact quality | xwoba_per_pa, xwoba_x_swstr |
| Stuff | avg_velo, pitching_plus, bb_pfxz |
| Approach | pitch_entropy |
| History | ip_resid_lag1, k_pct_lag1 |

## Refresh Instructions

```bash
# 1. Update FanGraphs Pitching+ features (Chrome 147 + undetected-chromedriver, ~30s/year)
python scripts/pull_fg_undetected.py

# 2. Re-aggregate Statcast (if statcast_2026.parquet was refreshed)
python scripts/build_sp_multiyr.py

# 3. Re-blend mid-season inputs and re-project (no retraining needed)
python scripts/xfp_v11_lock.py

# 4. Rebuild dashboard, push to trigger GitHub Pages refresh
git add docs/index.html data/outputs/xfp_v11_projections.csv
git commit -m "data: refresh V11 projections through $(date +%F)"
git push
```

## Research Docs
- `data/research/CLAUDE_CODE_HANDOFF.md` — full technical context for AI sessions
- `data/research/xfp_model_research.md` — version history, methodology, results
- `data/research/feature_search_log.csv` — complete experiment log
- `data/research/formula_sensitivity.csv` — score formula variants tested
- `data/research/rolling_ip_predictor_analysis.csv` — IP trend predictor analysis

## Version History
V8 (4-feat core, r=0.558) → V8.5 (+pfxz family, r=0.600) →
**V11 (+pitching_plus + fp_strike_pct, r=0.614)** [CURRENT PRODUCTION]

Negative results documented (and not shipped):
- V9: IP decomposition (compound errors from 6 component models)
- V10: Marcel weighting / archetype submodels / BaseRuns (all underperformed)
- V12: Residual correction (era-subset k_bias regression)

See `data/research/xfp_model_research.md` for the full lineage.

## License
Personal-use research project. Not affiliated with MLB, FanGraphs, or any
fantasy platform.
