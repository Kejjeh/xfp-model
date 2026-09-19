# xFP evaluation / provenance audit (generated)

Generated 2026-09-19T01:06:50Z by `python scripts/xfp_audit.py`. Numbers are recomputed from committed artifacts; nothing is retrained or retuned.

## Cross-year evaluation: legacy leave-one-year-out vs point-in-time ip_resid_lag1

| model | ip_resid mode | r | k_bias_hi | k_bias_lo | rmse | n |
|---|---|---|---|---|---|---|
| V8.5 (2015-2025) | loo | 0.59993 | 0.466 | -1.867 | 3.002 | 390 |
| V11 minus fp_strike (2020-2025) | loo | 0.61287 | 0.774 | -1.861 | 2.897 | 375 |
| V8.5 (2015-2025) | point_in_time | 0.60027 | 0.412 | -1.93 | 3.009 | 389 |
| V11 minus fp_strike (2020-2025) | point_in_time | 0.61332 | 0.739 | -1.912 | 2.906 | 374 |

Reproduction check: stored V8.5 bundle r=0.59993, k_bias_hi=0.466, n=390 vs recomputed (loo) r=0.59993 → match=True.
ip_resid_lag1 corr(loo, point-in-time) = 0.9462 over 1360 rows.

## 2026 YTD metric provenance

Frozen V8 bundle, n=112 (gs>=5): pre-season inputs r=0.3136 MAE=3.94 | blended (contemporaneous) inputs r=0.4913 MAE=3.436.
V11 stored YTD sample: {'n_ytd_eval': 112, 'n_inputs_include_2026_starts': 112, 'share_contemporaneous': 1.0, 'label': 'contemporaneous', 'mean_weight_2026': 0.2139}

## IDs / availability

unique pitcher-year: True; integer ids: True

| FG season | panel rows | matched | name mismatches | ok |
|---|---|---|---|---|
| 2020 | 146 | 146 | 0 | True |
| 2021 | 190 | 170 | 0 | True |
| 2022 | 186 | 170 | 0 | True |
| 2023 | 188 | 169 | 0 | True |
| 2024 | 180 | 166 | 0 | True |
| 2025 | 185 | 171 | 0 | True |
| 2026 | 161 | 159 | 0 | True |

| feature | declared first season | observed first season | violation |
|---|---|---|---|
| avg_velo | 2015 | 2015 | False |
| zone_pct | 2015 | 2015 | False |
| o_swing_pct | 2015 | 2015 | False |
| swstr_pct | 2015 | 2015 | False |
| c_plus_swstr | 2015 | 2015 | False |
| xwoba_per_pa | 2015 | 2015 | False |
| z_swing_pct | 2015 | 2015 | False |
| xwoba_x_swstr | 2015 | 2015 | False |
| ip_resid_lag1 | 2017 | 2017 | False |
| k_pct_lag1 | 2016 | 2016 | False |
| pitch_entropy | 2015 | 2015 | False |
| bb_pfxz | 2015 | 2015 | False |
| pitching_plus | 2020 | 2020 | False |

## Blend: latest-available (non-weighted) Pitching+

{'blended_pitchers_with_both_pp': 117, 'abs_delta_pp_gt_5': 52, 'abs_delta_pp_gt_10': 14, 'min_2026_ip': 16.0, 'mean_weight_2026_if_pitch_weighted': 0.212}

## Model bundle

{'sklearn_runtime': '1.9.1', 'unpickle_version_warnings': ['Trying to unpickle estimator StandardScaler from version 1.8.0 when using version 1.9.1. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\nhttps', 'Trying to unpickle estimator RidgeCV from version 1.8.0 when using version 1.9.1. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\nhttps://scik', 'Trying to unpickle estimator Pipeline from version 1.8.0 when using version 1.9.1. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\nhttps://sci'], 'features': ['avg_velo', 'zone_pct', 'o_swing_pct', 'swstr_pct', 'c_plus_swstr', 'xwoba_per_pa', 'z_swing_pct', 'xwoba_x_swstr', 'ip_resid_lag1', 'k_pct_lag1', 'pitch_entropy', 'bb_pfxz', 'pitching_plus', 'fp_strike_pct'], 'cross_year_r': 0.61381, 'k_bias_hi': 0.773, 'score_current': 1.45493, 'score_tolerance_T1': 1.84143, 'formula': 'cross_year_r * 3 - max(0, |k_bias_hi| - T) * 0.5 (T=1.0 production)', 'trained_date': '2026-05-05', 'n_train': 768.0, 'version': 'v11', 'training_years': '2020-2025', 'ytd_mae_2026': 3.393, 'ytd_r_2026': 0.51075, 'note': 'V11 is V8.5 + pitching_plus + fp_strike_pct. Use V8.1 mid-season blend for projections.'}

## docs/index.html cross-check (read-only)

{'generator': 'external (not in this repo); NOT edited by this audit', 'meta_dataThru': '2026-08-31', 'meta_trainedDate': '2026-05-05', 'meta_ytdR': 0.511, 'repo_inputs_as_of': '2026-05-04', 'v11_values_identical_to_repo_csv': 182, 'repo_csv_rows': 182, 'web_rows': 376, 'web_rows_without_repo_v11': 194, 'has_provenance_panel': False}

## Missing sources

- `data/research/xfp_cache/fp_strike_2015_2026.csv` present=False: fp_strike_pct feature for V11 (14th feature); V11 cannot be retrained from the repo without it
- `data/research/xfp_cache/statcast_2025.parquet` present=False: raw Statcast cache (any year); needed by build_sp_multiyr / pitch-type panels
- `data/outputs/xfp_v8_projections.csv` present=False: V8 baseline projections read by xfp_v8_midseason.py
- `data/outputs/xfp_v11_dashboard.html` present=False: dashboard written by xfp_v11_lock.build_dashboard (docs/index.html is generated elsewhere)
- `data/outputs/plv_rolling_2026.parquet` present=False: PLV rolling file read by xfp_rolling_ip.py
