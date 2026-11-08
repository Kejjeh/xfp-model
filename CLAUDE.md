# Agent guide

## What this is

SP xFP projects per-start ESPN fantasy points for MLB starting pitchers: a Ridge regression (V11, 14 Statcast + FanGraphs Pitching+ features) trained on 2020-2025 SP-seasons, plus a mid-season blend of 2025 and 2026 inputs. It publishes a GitHub Pages dashboard (`docs/`). Python, no database; models are committed pickles, inputs are committed CSVs. **Current state: frozen.** Tests pass. The 2026-09-19 audit (PR #1) is the last code change, and the owner closed the repo out with "no further xfp-model work planned". V11 **cannot be retrained or re-projected from this repo** because the inputs are missing (see gotchas). Read `data/research/EVAL_PROVENANCE_AUDIT.md` (findings D1-D11) and `data/research/audit/HANDOFF_2026-09-19.md` before any change.

This repo sits inside the `plv_clone` checkout, which ignores it. The parent's CLAUDE.md, skills and commands are about plv_clone and do not apply here.

## Commands (all verified from repo root)

```bash
pip install pandas numpy scikit-learn joblib pytest   # existing deps; no requirements file
python -m pytest tests -q                  # ~45s. Expect: 43 passed
python -m pytest tests -q -m "not slow"    # ~30s. Expect: 38 passed, 5 deselected (synthetic point-in-time suite)
python -m pytest tests -q -m slow          # ~17s. Expect: 5 passed, 38 deselected (frozen real-data guards)
python scripts/xfp_audit.py                # ~20s. Rewrites data/research/audit/xfp_eval_audit.{json,md}
python scripts/xfp_provenance.py           # ~2s.  Rewrites data/outputs/xfp_v11_provenance.{json,html}
```

- Verified offline with the `nonet` pytest plugin: no test touches the network.
- After the two generators, only the timestamp (`generated_utc` / "Generated ..."), `git_head` and, in the audit, the `sklearn_runtime` / unpickle-warning lines should differ. The committed audit was made on scikit-learn 1.9.1; this machine has 1.8.0 (the version the bundles were pickled with), so the warnings vanish here. If only those lines moved, discard the regeneration with `git checkout -- data`.
- **Not runnable here** (per the audit, D7): `scripts/xfp_v11_lock.py` and `scripts/build_sp_multiyr.py` need `fp_strike_2015_2026.csv` and the Statcast `*.parquet` caches, which are not in the repo. `scripts/pull_fg_undetected.py` scrapes FanGraphs through a visible Chrome 147; never run it without the owner.

## Architecture map

| File | Role |
|---|---|
| `scripts/xfp_v7_pipeline.py` | Core eval: `derive_features`, `add_ip_resid_lag(mode=)`, `cross_year_evaluate`, `ooy_evaluate`, `CROSS_YEAR_TRANSITIONS` |
| `scripts/xfp_v8_pipeline.py` / `xfp_v8_5_pipeline.py` | V8 / V8.5 pipelines; helper builds imported by V11 |
| `scripts/xfp_v8_midseason.py`, `v11_spotcheck.py` | V8.1 mid-season blend (`build_blended_inputs`) |
| `scripts/xfp_v11_lock.py` | Train V11, project 2026, `build_dashboard` (embeds the provenance panel) |
| `scripts/xfp_checks.py` | Pure helpers: `FEATURE_AVAILABILITY`, `BLEND_SEMANTICS`, ID alignment, `missing_sources()` |
| `scripts/xfp_audit.py`, `xfp_provenance.py` | Audit report and provenance record/panel generators |
| `tests/` | `conftest.py` synthetic panel; `test_point_in_time`, `test_blend_and_ids`, `test_provenance`; `test_real_data_frozen` (`-m slow`) |
| `data/models/*.pkl` | V8, V8.5, V11 bundles (V11 is production) |
| `data/outputs/` | Projections CSVs, FanGraphs pulls 2020-2026, provenance json/html |
| `data/research/` | Audit, research history, experiment logs, `xfp_cache/` training panels |
| `docs/` | GitHub Pages site, **generated outside this repo** |

## Conventions and gotchas

- **`docs/` is written by an external nightly job** (`plv_clone/scripts/xfp/refresh_dashboards.py`, which commits and pushes here; 447 of this repo's commits are its `live:`/`refresh:` commits, the last on 2026-09-02). Never edit `docs/` from this repo.
- **Never hand-edit generated files**: `data/outputs/xfp_v11_projections.csv`, `xfp_v8_5_projections.csv`, `xfp_v11_provenance.{json,html}`, `data/research/audit/xfp_eval_audit.{json,md}`, `data/models/*.pkl`.
- **The headline YTD r 0.511 / MAE 3.393 is contemporaneous**, not a forward test: the scored 2026 starts are in the blend inputs. Never present it as out-of-sample. The cross-year r 0.614 is archived and was built with the legacy leaky `ip_resid` (`mode='loo'`).
- **`add_ip_resid_lag` defaults to `mode='point_in_time'`**; `mode='loo'` exists only to reproduce archived numbers. New work uses point-in-time.
- **Non-circular constraint:** no per-start K/IP/H/ER/BB/HBP as direct features.
- **Blend semantics (D4):** rate features are pitch-weighted 2025+2026; `pitching_plus`, `pitch_entropy`, `bb_pfxz`, `fp_strike_pct` take the 2026 YTD value outright. This is documented, not endorsed; changing it retunes projections.
- **IDs:** FanGraphs `mlb_id` joins Statcast `pitcher` as strict integers. Name agreement never verifies identity.
- `scripts/README.md` lists a `scripts/archive/` folder that is not in this repo.
- **Main checkout:** it carries a deliberate `git stash`. Never pop, drop or apply it; work in a worktree.
- No secrets in this repo.
- **Do not read into context**: `docs/player_profiles_data.js` (42 MB), `docs/triangulate.html` (4.6 MB), `docs/index.html` (1.7 MB) and the other `docs/*.html`, `data/models/*.pkl`, `data/research/xfp_cache/*.csv`, `data/outputs/fangraphs_pitchers_*.csv`. `CLAUDE_CODE_HANDOFF.md` (~800 lines) and `xfp_model_research.md` (~900) are history: read the section you need.

## Skills for this repo

Load these with the Skill tool when the situation comes up; the user does not need to ask.

| Situation in this repo | Skill |
|---|---|
| Touching a feature build, the mid-season blend, `ip_resid_lag1` or an evaluation split, where future seasons could leak into the past | `sop-fail-loud` |
| A cross-year r, YTD figure or projection looks wrong, or a frozen guard in `test_real_data_frozen.py` fails | `diagnosing-bugs` |
| Fixing an evaluation or blend bug: add a synthetic point-in-time case to `tests/` first | `tdd` |
| A change would move projections, the audit report, the provenance panel or a model bundle | `sop-regenerate-outputs` |
| Whether to change blend semantics, retrain, ship a new model version or relabel a headline metric | `sop-owner-decision` |
| Judging a correlation, MAE, CV alpha or "contemporaneous vs point-in-time" claim | `data:statistical-analysis` |
| Committing | `safe-commit` |
| Opening or merging a PR | `sop-ship-pr` |
| Editing this CLAUDE.md, or a change moves a number in it | `sop-project-guide` |

## Before you finish any task

1. `python -m pytest tests -q` → `43 passed` (more is fine if you added tests).
2. If you touched `scripts/`: run `python scripts/xfp_audit.py` and `python scripts/xfp_provenance.py`, then `git diff data/`. Only timestamp, `git_head` and sklearn-runtime lines may change. Any other change is a moved number: stop and use `sop-regenerate-outputs`. Otherwise `git checkout -- data`.
3. `git diff --stat` → only files you meant to change; nothing under `docs/`.

## Model routing

Safe for Sonnet: doc edits; adding tests for existing behaviour; running the suite and the two generators and confirming only timestamps moved.

Escalate to Opus: anything in `xfp_v7_pipeline.py` evaluation functions, the blend, feature construction, `ip_resid` modes, model bundles or score formulas; any change to how a metric is labelled; anything that contradicts `EVAL_PROVENANCE_AUDIT.md`.
