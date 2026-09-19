# xFP historical-evaluation and dashboard-provenance audit

Audit date: 2026-09-19. Scope: the V11 production projection path
(`scripts/xfp_v11_lock.py` and the modules it imports), the cross-year / OOY / YTD evaluations,
the V8.1 mid-season blend, feature availability, ID alignment, and the provenance shown on the
dashboard. Method: code reading, synthetic point-in-time fixtures (`tests/`), and re-running the
frozen evaluations on the committed panel (`scripts/xfp_audit.py`).

Nothing in this audit retrains or retunes the production model, searches features, changes a
held-out boundary, or claims a metric improved. Numbers below are recomputed from committed
artifacts; the generated companion report is `data/research/audit/xfp_eval_audit.md` (+ `.json`).

## Summary of findings

| # | Area | Finding | Severity | Status |
|---|---|---|---|---|
| D1 | Reproducibility | Every script resolved the repo root as `parents[2]` (a `scripts/xfp/` layout) and put `scripts/xfp` on `sys.path`; in this repo `scripts/` sits directly under the root, so no script could run or import as committed. | High | Repaired (root detection by `data/research`). |
| D2 | Leakage | `ip_resid_lag1` was built leave-one-year-out: the IP model scoring season Y was fit on every other season, including future seasons and the partial-2026 slice. `ip_per_start` is a component of the FP target. | Structural leak, immaterial effect | Repaired: point-in-time is now the default; legacy `mode='loo'` kept to reproduce archived numbers. Effect measured (below). |
| D3 | Misleading provenance | "2026 YTD r 0.511 / MAE 3.393" was presented as a live-deployment reality check. It is contemporaneous: all 112 evaluated pitchers had their own 2026 starts (through 2026-05-04) folded into the model inputs by the V8.1 blend, and the actuals are those same starts. | High | Labelled contemporaneous on the dashboard, bundle labels, provenance JSON; frozen-V8 gap quantified. |
| D4 | Blend semantics | Dashboard text says "rate metrics weighted by 2025+2026 pitches", but non-rate features (`pitching_plus`, `pitch_entropy`, `bb_pfxz`, `fp_strike_pct`) take the 2026 YTD value outright when present (no weighting). 14 of 117 blended pitchers had Pitching+ move by >10 on 16-42 IP. | Medium | Documented per feature (`BLEND_SEMANTICS`), pinned by tests, shown on the provenance panel. Behaviour NOT changed (would retune projections). |
| D5 | As-of mismatch | `docs/index.html` shows `dataThru 2026-08-31`, but the repo's V11 projections (identical values for all 182 shared pitchers) were built from inputs through 2026-05-04 (`SEASON_DATES[2026]`, max 7 GS) with a model trained 2026-05-05. | High | Provenance JSON/HTML now states inputs as-of 2026-05-04. `docs/index.html` is generated outside this repo (see Scope limits). |
| D6 | Claim mismatch | README: "expand-window cross-year evaluation (9 transitions, 2015-2025)". Code: 4 transitions (2021→2022 … 2024→2025); V11 is evaluated on the 2020-2025 slice, so its first transition trains on 2020-2021 only. | Medium | README corrected; transitions exposed as `CROSS_YEAR_TRANSITIONS`, pinned by test. |
| D7 | Data availability | `fp_strike_2015_2026.csv` (14th V11 feature) and every Statcast parquet are absent; V11 cannot be retrained or re-projected from the repo. `xfp_v8_projections.csv`, `plv_rolling_2026.parquet`, `xfp_v11_dashboard.html` are also referenced but absent. | High (limits) | Listed explicitly in `missing_sources()`, provenance panel, and below. |
| D8 | Evaluation label | OOY r trains on seasons after the holdout (same-season diagnostic). Documented in code as not point-in-time. | Low | Docstring + test. |
| D9 | Stale dependency | YTD columns in `xfp_v11_projections.csv` (`gs_2026`, `fp_per_start_actual_2026`, `k_pct_2026`) are copied from `xfp_v8_5_projections.csv`, not recomputed. Verified identical to the panel's 2026 slice (124/124 GS, 168 FP values). | Low | Noted in provenance (`ytd_columns_source`). |
| D10 | ID alignment | FanGraphs (`mlb_id`) ↔ Statcast (`pitcher`) joins are integer, unique per season, and surname-consistent for every matched row 2020-2026 (146-171 of 146-190 SP rows per season). | None found | Regression checks added. |
| D11 | Bundle portability | The V11 bundle unpickles under the sandbox's scikit-learn with a version warning (recorded in the audit JSON). Predictions were not re-verified against a same-version environment. | Low | Recorded. |

## Reproduction (frozen evaluations, committed panel)

`python scripts/xfp_audit.py` recomputes these; the regression guards in
`tests/test_real_data_frozen.py` pin them.

| model | ip_resid mode | cross-year r | k_bias_hi | n |
|---|---|---|---|---|
| V8.5 (2015-2025) | loo (archived) | 0.59993 | 0.466 | 390 |
| V8.5 (2015-2025) | point-in-time | 0.60027 | 0.412 | 389 |
| V11 minus fp_strike (2020-2025) | loo (archived) | 0.61287 | 0.774 | 375 |
| V11 minus fp_strike (2020-2025) | point-in-time | 0.61332 | 0.739 | 374 |

The loo row for V8.5 matches the stored bundle metrics (r 0.59993, k_bias_hi 0.466, n 390) to the
last digit, which validates the harness. The leak is structural, not material: r moves by <0.001,
k_bias_hi by about 0.04, and the point-in-time lag correlates 0.946 with the leaky one. The full
V11 number (0.61381) cannot be recomputed because `fp_strike_pct` is missing (D7).

YTD provenance, same frozen V8 weights, same 112 pitchers (gs ≥ 5, data through 2026-05-04):

| inputs | YTD r | YTD MAE |
|---|---|---|
| pre-season (2025 only) | 0.3136 | 3.940 |
| V8.1 blend (includes the scored 2026 starts) | 0.4913 | 3.436 |

These two values are also the stored `xfp_v8` and `xfp_v8_1` columns of `xfp_v8_5_projections.csv`.
The V11 "YTD r 0.511" is of the second kind. A forward V11 number does not exist in the repo.

## What is point-in-time and what is not

| metric | point-in-time | notes |
|---|---|---|
| cross-year r | yes (after D2) | features from season T for pitchers in T and T+1; scaler + RidgeCV fit on rows with year < T+1; k-bias cohorts use test-year K% (post-hoc diagnostic) |
| OOY r | no | trains on seasons after the holdout |
| 2026 YTD r / MAE | no | contemporaneous inputs (D3) |
| 2026 projections | as-of 2026-05-04 | rate features pitch-weighted 2025+2026; non-rate features latest-available (D4) |

## Feature availability (first usable season as an input)

| feature | source | first season | note |
|---|---|---|---|
| Statcast rates (velo, zone, o-swing, swstr, c+swstr, z-swing, xwoba, entropy, bb_pfxz) | Statcast | 2015 | |
| k_pct_lag1 | prior season | 2016 | |
| ip_resid_lag1 | prior-season residual | 2017 (point-in-time) | needs ≥50 earlier rows for the IP model; 2016 residuals are the first |
| pitching_plus | FanGraphs | 2020 | 146-171 of the SP rows per season match |
| fp_strike_pct | Statcast | 2015 | source file not in repo |

Training slice for V11 is 2020-2025 (limited by Pitching+), n=768 per the bundle.

## Historical-data availability limits (explicit)

- 2026 inputs end 2026-05-04 (`SEASON_DATES[2026]`), max 7 GS per pitcher. Anything later on the
  public dashboard comes from a generator outside this repository.
- Raw Statcast caches (`statcast_<year>.parquet`) are not committed; the pitch-type and pfxz panels
  are the only Statcast derivatives available, so those features cannot be rebuilt or extended.
- `fp_strike_2015_2026.csv` is absent: V11 cannot be retrained, re-projected, or fully re-evaluated.
  V8.5 and V11-minus-fp_strike can.
- `xfp_v8_projections.csv` is absent, so `xfp_v8_midseason.py` cannot run end to end.
- Pitching+ history starts 2020; the V11 cross-year evaluation therefore has 4 transitions with
  a first training window of two seasons (2020-2021).
- The panel file is named `sp_multiyr_2015_2025.csv` but carries the partial 2026 slice.

## Scope limits of this change set

- `docs/` (index.html, xfp_board, matchup, live_dashboard, player_profiles, triangulate) is
  regenerated by an external nightly job (league audit, hitters, relievers, "V12" values that do
  not exist in this repo). It was read, cross-checked, and NOT edited. The provenance panel is
  delivered as `data/outputs/xfp_v11_provenance.{json,html}` and embedded in the dashboard that
  this repo builds (`xfp_v11_lock.build_dashboard`); the external generator can embed the same
  JSON/HTML fragment.
- The production bundle and projection csv are untouched. Adopting the point-in-time `ip_resid`
  in production requires a retrain, which is blocked by the missing `fp_strike` source.

## Verification commands

```bash
pip install pandas numpy scikit-learn joblib pytest      # existing project dependencies only
python -m pytest tests -q -m "not slow"                  # synthetic point-in-time suite (~1.5 min)
python -m pytest tests -q -m slow                        # frozen real-data guards (~2-3 min)
python scripts/xfp_audit.py                              # regenerates data/research/audit/xfp_eval_audit.{json,md}
python scripts/xfp_provenance.py                         # regenerates data/outputs/xfp_v11_provenance.{json,html}
```
