# xFP evaluation / provenance audit (generated)

Generated 2026-09-19T21:20:45Z by `python scripts/xfp_audit.py`. Numbers are recomputed from committed artifacts; nothing is retrained or retuned.

## Cross-year evaluation: legacy leave-one-year-out vs point-in-time ip_resid_lag1

| model | ip_resid mode | r | k_bias_hi | k_bias_lo | rmse | n |
|---|---|---|---|---|---|---|
| V8.5 (2015-2025) | loo | 0.59993 | 0.466 | -1.867 | 3.002 | 390 |
| V11 minus fp_strike (2020-2025) | loo | 0.61287 | 0.774 | -1.861 | 2.897 | 375 |
| V8.5 (2015-2025) | point_in_time | 0.60027 | 0.412 | -1.93 | 3.009 | 389 |
| V11 minus fp_strike (2020-2025) | point_in_time | 0.61332 | 0.739 | -1.912 | 2.906 | 374 |

Stored bundle cross_year_r=0.61381: status **archived_legacy_loo (ip_resid built with future seasons; not point-in-time)** (ip_resid_mode=loo (archived, inferred: bundle predates audit)). Only the point_in_time rows above are point-in-time.

Reproduction check: stored V8.5 bundle r=0.59993, k_bias_hi=0.466, n=390 vs recomputed (loo) r=0.59993 → match=True.
ip_resid_lag1 corr(loo, point-in-time) = 0.9462 over 1360 rows.

## 2026 YTD metric provenance

Frozen V8 bundle, n=112 (gs>=5): pre-season inputs r=0.3136 MAE=3.94 | blended (contemporaneous) inputs r=0.4913 MAE=3.436.
V11 stored YTD sample: {'n_ytd_eval': 112, 'n_inputs_include_2026_starts': 112, 'share_contemporaneous': 1.0, 'label': 'contemporaneous', 'forward_status': 'not_forward', 'mean_weight_2026': 0.2139}

## IDs / availability

unique pitcher-year: True; integer ids: True

Name status is evidence of consistency, never verified identity (identity_verified_by_name=False). ambiguous = given-name / suffix conflict or same-surname pool; resolve by id.

| FG season | panel rows | matched | consistent | ambiguous | mismatch | ok |
|---|---|---|---|---|---|---|
| 2020 | 146 | 146 | 117 | 29 | 0 | True |
| 2021 | 190 | 170 | 138 | 32 | 0 | True |
| 2022 | 186 | 170 | 144 | 26 | 0 | True |
| 2023 | 188 | 169 | 141 | 28 | 0 | True |
| 2024 | 180 | 166 | 133 | 33 | 0 | True |
| 2025 | 185 | 171 | 136 | 35 | 0 | True |
| 2026 | 161 | 159 | 137 | 22 | 0 | True |

Ambiguous rows (id, panel name, FG name):
- 2020: (434671, 'Sánchez, Aníbal', 'Anibal Sanchez')
- 2020: (448179, 'Hill, Rich', 'Rich Hill')
- 2020: (474463, 'Anderson, Brett', 'Brett Anderson')
- 2020: (502624, 'Anderson, Chase', 'Chase Anderson')
- 2020: (527048, 'Pérez, Martín', 'Martin Perez')
- 2020: (542881, 'Anderson, Tyler', 'Tyler Anderson')
- 2020: (543037, 'Cole, Gerrit', 'Gerrit Cole')
- 2020: (543243, 'Gray, Sonny', 'Sonny Gray')
- 2020: (547942, 'Kim, Kwang Hyun', 'Kwang-hyun Kim')
- 2020: (547943, 'Ryu, Hyun Jin', 'Hyun-Jin Ryu')
- 2020: (572070, 'Richards, Garrett', 'Garrett Richards')
- 2020: (592351, 'Gray, Jon', 'Jon Gray')
- 2020: (592866, 'Williams, Trevor', 'Trevor Williams')
- 2020: (605242, 'Fulmer, Michael', 'Michael Fulmer')
- 2020: (605347, 'López, Jorge', 'Jorge Lopez')
- 2020: (622491, 'Castillo, Luis', 'Luis Castillo')
- 2020: (622694, 'Hernández, Elieser', 'Elieser Hernandez')
- 2020: (625643, 'López, Reynaldo', 'Reynaldo Lopez')
- 2020: (641154, 'López, Pablo', 'Pablo Lopez')
- 2020: (641712, 'Hudson, Dakota', 'Dakota Hudson')
- 2021: (474463, 'Anderson, Brett', 'Brett Anderson')
- 2021: (503449, 'Peralta, Wily', 'Wily Peralta')
- 2021: (518876, 'Kelly, Merrill', 'Merrill Kelly')
- 2021: (542881, 'Anderson, Tyler', 'Tyler Anderson')
- 2021: (543243, 'Gray, Sonny', 'Sonny Gray')
- 2021: (547942, 'Kim, Kwang Hyun', 'Kwang-hyun Kim')
- 2021: (547943, 'Ryu, Hyun Jin', 'Hyun-Jin Ryu')
- 2021: (572070, 'Richards, Garrett', 'Garrett Richards')
- 2021: (592346, 'González, Chi Chi', 'Chi Chi Gonzalez')
- 2021: (592351, 'Gray, Jon', 'Jon Gray')
- 2021: (592866, 'Williams, Trevor', 'Trevor Williams')
- 2021: (593958, 'Rodriguez, Eduardo', 'Eduardo Rodriguez')
- 2021: (605347, 'López, Jorge', 'Jorge Lopez')
- 2021: (605507, 'Thompson, Zach', 'Zach Thompson')
- 2021: (621219, 'Mills, Alec', 'Alec Mills')
- 2021: (622491, 'Castillo, Luis', 'Luis Castillo')
- 2021: (622694, 'Hernández, Elieser', 'Elieser Hernandez')
- 2021: (624133, 'Suarez, Ranger', 'Ranger Suarez')
- 2021: (641154, 'López, Pablo', 'Pablo Lopez')
- 2021: (641302, 'Alexander, Tyler', 'Tyler Alexander')
- 2022: (448179, 'Hill, Rich', 'Rich Hill')
- 2022: (518876, 'Kelly, Merrill', 'Merrill Kelly')
- 2022: (527048, 'Pérez, Martín', 'Martin Perez')
- 2022: (542881, 'Anderson, Tyler', 'Tyler Anderson')
- 2022: (543243, 'Gray, Sonny', 'Sonny Gray')
- 2022: (592351, 'Gray, Jon', 'Jon Gray')
- 2022: (592717, 'Sanchez, Aaron', 'Aaron Sanchez')
- 2022: (592866, 'Williams, Trevor', 'Trevor Williams')
- 2022: (593958, 'Rodriguez, Eduardo', 'Eduardo Rodriguez')
- 2022: (605507, 'Thompson, Zach', 'Zach Thompson')
- 2022: (607259, 'Martinez, Nick', 'Nick Martinez')
- 2022: (622491, 'Castillo, Luis', 'Luis Castillo')
- 2022: (624133, 'Suarez, Ranger', 'Ranger Suarez')
- 2022: (624522, 'Thompson, Keegan', 'Keegan Thompson')
- 2022: (641154, 'López, Pablo', 'Pablo Lopez')
- 2022: (641302, 'Alexander, Tyler', 'Tyler Alexander')
- 2022: (641712, 'Hudson, Dakota', 'Dakota Hudson')
- 2022: (641745, 'Keller, Brad', 'Brad Keller')
- 2022: (642547, 'Peralta, Freddy', 'Freddy Peralta')
- 2022: (656605, 'Keller, Mitch', 'Mitch Keller')
- 2023: (502624, 'Anderson, Chase', 'Chase Anderson')
- 2023: (518876, 'Kelly, Merrill', 'Merrill Kelly')
- 2023: (527048, 'Pérez, Martín', 'Martin Perez')
- 2023: (542881, 'Anderson, Tyler', 'Tyler Anderson')
- 2023: (543243, 'Gray, Sonny', 'Sonny Gray')
- 2023: (547943, 'Ryu, Hyun Jin', 'Hyun-Jin Ryu')
- 2023: (592351, 'Gray, Jon', 'Jon Gray')
- 2023: (592836, 'Walker, Taijuan', 'Taijuan Walker')
- 2023: (593958, 'Rodriguez, Eduardo', 'Eduardo Rodriguez')
- 2023: (608344, 'Irvin, Cole', 'Cole Irvin')
- 2023: (624133, 'Suarez, Ranger', 'Ranger Suarez')
- 2023: (641154, 'López, Pablo', 'Pablo Lopez')
- 2023: (642547, 'Peralta, Freddy', 'Freddy Peralta')
- 2023: (656731, 'Megill, Tylor', 'Tylor Megill')
- 2023: (657277, 'Webb, Logan', 'Logan Webb')
- 2023: (663623, 'Irvin, Jake', 'Jake Irvin')
- 2023: (665795, 'Cabrera, Edward', 'Edward Cabrera')
- 2023: (666129, 'Garrett, Braxton', 'Braxton Garrett')
- 2023: (668909, 'Williams, Gavin', 'Gavin Williams')
- 2023: (669194, 'Nelson, Ryne', 'Ryne Nelson')
- 2024: (518876, 'Kelly, Merrill', 'Merrill Kelly')
- 2024: (527048, 'Pérez, Martín', 'Martin Perez')
- 2024: (542881, 'Anderson, Tyler', 'Tyler Anderson')
- 2024: (543243, 'Gray, Sonny', 'Sonny Gray')
- 2024: (544150, 'Suárez, Albert', 'Albert Suarez')
- 2024: (592351, 'Gray, Jon', 'Jon Gray')
- 2024: (592866, 'Williams, Trevor', 'Trevor Williams')
- 2024: (593958, 'Rodriguez, Eduardo', 'Eduardo Rodriguez')
- 2024: (607259, 'Martinez, Nick', 'Nick Martinez')
- 2024: (608344, 'Irvin, Cole', 'Cole Irvin')
- 2024: (622491, 'Castillo, Luis', 'Luis Castillo')
- 2024: (624133, 'Suarez, Ranger', 'Ranger Suarez')
- 2024: (625643, 'López, Reynaldo', 'Reynaldo Lopez')
- 2024: (641154, 'López, Pablo', 'Pablo Lopez')
- 2024: (641712, 'Hudson, Dakota', 'Dakota Hudson')
- 2024: (642547, 'Peralta, Freddy', 'Freddy Peralta')
- 2024: (650633, 'King, Michael', 'Michael King')
- 2024: (650911, 'Sánchez, Cristopher', 'Cristopher Sanchez')
- 2024: (656731, 'Megill, Tylor', 'Tylor Megill')
- 2024: (657277, 'Webb, Logan', 'Logan Webb')
- 2025: (518876, 'Kelly, Merrill', 'Merrill Kelly')
- 2025: (527048, 'Pérez, Martín', 'Martin Perez')
- 2025: (542881, 'Anderson, Tyler', 'Tyler Anderson')
- 2025: (592836, 'Walker, Taijuan', 'Taijuan Walker')
- 2025: (592866, 'Williams, Trevor', 'Trevor Williams')
- 2025: (593958, 'Rodriguez, Eduardo', 'Eduardo Rodriguez')
- 2025: (605280, 'Holmes, Clay', 'Clay Holmes')
- 2025: (607259, 'Martinez, Nick', 'Nick Martinez')
- 2025: (624133, 'Suarez, Ranger', 'Ranger Suarez')
- 2025: (641154, 'López, Pablo', 'Pablo Lopez')
- 2025: (642547, 'Peralta, Freddy', 'Freddy Peralta')
- 2025: (650633, 'King, Michael', 'Michael King')
- 2025: (656550, 'Holmes, Grant', 'Grant Holmes')
- 2025: (656605, 'Keller, Mitch', 'Mitch Keller')
- 2025: (656731, 'Megill, Tylor', 'Tylor Megill')
- 2025: (657277, 'Webb, Logan', 'Logan Webb')
- 2025: (663436, 'Martin, Davis', 'Davis Martin')
- 2025: (668909, 'Williams, Gavin', 'Gavin Williams')
- 2025: (669302, 'Gilbert, Logan', 'Logan Gilbert')
- 2025: (669432, 'Rogers, Trevor', 'Trevor Rogers')
- 2026: (518876, 'Kelly, Merrill', 'Merrill Kelly')
- 2026: (527048, 'Pérez, Martín', 'Martin Perez')
- 2026: (592836, 'Walker, Taijuan', 'Taijuan Walker')
- 2026: (593958, 'Rodriguez, Eduardo', 'Eduardo Rodriguez')
- 2026: (605280, 'Holmes, Clay', 'Clay Holmes')
- 2026: (624133, 'Suarez, Ranger', 'Ranger Suarez')
- 2026: (625643, 'López, Reynaldo', 'Reynaldo Lopez')
- 2026: (642547, 'Peralta, Freddy', 'Freddy Peralta')
- 2026: (650633, 'King, Michael', 'Michael King')
- 2026: (656550, 'Holmes, Grant', 'Grant Holmes')
- 2026: (656605, 'Keller, Mitch', 'Mitch Keller')
- 2026: (657277, 'Webb, Logan', 'Logan Webb')
- 2026: (667755, 'Soriano, José', 'Jose Soriano')
- 2026: (668909, 'Williams, Gavin', 'Gavin Williams')
- 2026: (669432, 'Rogers, Trevor', 'Trevor Rogers')
- 2026: (677952, 'Ashcraft, Braxton', 'Braxton Ashcraft')
- 2026: (680732, 'Burke, Sean', 'Sean Burke')
- 2026: (682052, 'Lopez, Jacob', 'Jacob Lopez')
- 2026: (683004, 'Leiter, Jack', 'Jack Leiter')
- 2026: (691587, 'Pérez, Eury', 'Eury Perez')

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
