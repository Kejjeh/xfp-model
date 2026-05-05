# xFP Model Optimization — Claude Code Handoff

## **CURRENT PRODUCTION MODEL: V11**
*Last updated 2026-05-05*

**V11 = V8.5 + pitching_plus + fp_strike_pct.** Cross-year r 0.614, YTD r 0.511, YTD MAE 3.393.
Trained on 2020-2025 SP-seasons (n=768). Production bundle at `data/models/xfp_v11_pipeline.pkl`.

| Phase | Status | Outcome |
|---|---|---|
| Phase 7-9 | COMPLETE | V8 + V8.5 baseline established |
| Phase 10 | COMPLETE | Marcel / archetype / BaseRuns negative results documented |
| Phase 11 | COMPLETE | FG Stuff+/Pitching+ history pulled (undetected-chromedriver), V11 ships |
| Phase 12 | COMPLETE (negative) | Residual correction architecture rejected |

### Benchmark table (final)

| Metric | V8 (frozen) | V8.5 (superseded) | **V11 (production)** |
|---|---|---|---|
| Cross-year r | 0.558 | 0.600 | **0.614** |
| k_bias_hi | 0.241 | 0.466 | 0.773 |
| Score (0.5 coef) | 1.555 | 1.567 | 1.455 |
| Score (T=1.0 tolerance) | 1.800 | 1.800 | **1.841** |
| OOY r | – | – | 0.836 |
| 2026 YTD r | – | 0.475 | **0.511** |
| 2026 YTD MAE | – | 3.484 | **3.393** |

### Next Phase Candidates

- **Phase 13: Injury history → IP predictor.** Returning-from-IL pitchers face managed pitch counts
  → fewer IP/start; chronically injured pitchers face precautionary pulls → suppressed IP; durable
  workhorses earn elevated IP premium. Data sources: Baseball Reference transaction logs, MLB
  transactions feed, or Roster Resource. Specifically targets the Woodruff/Ragans-archetype
  overprojection that no process model has been able to fix.
- **Phase 14: fp_strike_pct deeper analysis.** Stabilization curve (currently treated as one feature)
  vs. interaction with swstr_pct (does it amplify whiff signal differently for command-heavy pitchers?).
  May surface subgroups where fp_strike_pct deserves higher weight.
- **Phase 15: Score-formula calibration audit.** Use `scripts/xfp/compare_score_formulas.py` to
  document how V11 vs V8.5 ranks under tolerance T=0.5/0.7/1.0. Tolerance T=1.0 is the recommended
  scoring formula for V11+ evaluation; this is the basis V11 ships under.

A `formula_sensitivity.csv` is created in Phase 12 and documents scoring-formula variants tested.
T=1.0 tolerance formula is recommended for V11+ evaluation.

---

## Mission (original, retained for context)
Build and run an **automated feature search loop** for the xFP fantasy baseball SP model.
Systematically test feature combinations, record every result, keep what works, discard what doesn't.
The goal is to find the highest cross-year r + lowest high-K bias with fully non-circular features.

**Primary validation metric is cross-year r, not same-year OOY r.** See validation section below.

---

## Project Context

**What this is:** A Ridge regression model that predicts fantasy points per start (FP/start) for
MLB starting pitchers. Used for ESPN standard scoring: K×1 + IP×3.3 − H×1 − ER×2 − BB×1 − HBP×1.

**Non-circular constraint:** Features may NOT directly contain components of the FP formula
(K, IP, H, ER, BB, HBP per start). Everything must be *process* metrics: pitch behavior,
batted ball quality, contact rates from Statcast.

**Key insight:** High-K pitchers (>30% K rate) are systematically underpredicted. V6 reduces
this bias from +0.57 → +0.21 FP/start through: xwoba×swstr interaction, ip_resid_lag1, and
Bayesian xwoba shrinkage. But there is likely more to squeeze.

---

## File Locations (all persistent on disk)

| File | Description |
|------|-------------|
| `/tmp/sp_multiyr.csv` | 636 SP-seasons 2021-2025, core training data |
| `/tmp/sp_extra_metrics.csv` | xwoba_contact, avg_ev per pitcher-season |
| `/tmp/xfp_v6_final.csv` | 2026 projections with V6 model applied |
| `/tmp/ip_resid_prior.csv` | Prior-year IP depth residuals per pitcher |
| `data/research/xfp_model_research.md` | Full research history & findings |
| `data/outputs/xfp_v6_dashboard.html` | Current V6 dashboard |

**Note:** `/tmp/` files are session-local. If they're gone, re-derive from the workspace:
`/sessions/upbeat-dazzling-bardeen/mnt/plv_clone/` is the persistent workspace.

---

## Current Best Model (V6)

```python
V6_FEATS = [
    'avg_velo',        # fastball velocity
    'abs_pfxz',        # |vertical movement| (abs value of avg_pfxz)
    'avg_ext',         # extension toward plate
    'zone_pct',        # strike zone %, z_swing/z_zone proxy
    'o_swing_pct',     # chase rate (o_swing_num/o_zone_tot)
    'swstr_pct',       # whiff rate (whiffs/total_pitches)
    'c_plus_swstr',    # called_strikes+whiffs / total_pitches
    'xwoba_contact',   # xwOBA on contact (Bayes-shrunken for projections)
    'z_swing_pct',     # in-zone swing commitment (z_swing/in_zone)
    'xwoba_x_swstr',   # xwoba_contact × swstr_pct (interaction)
    'ip_resid_lag1',   # prior-year IP depth residual
]
```

**Performance (OOY validation, 2022-2025 holdouts):**
- OOY r = 0.9081 ← same-year: year T metrics → year T FP
- Cross-year r = NOT YET COMPUTED ← prior-year: year T metrics → year T+1 FP (deployment target)
- High-K bias (>30% K rate): +0.21 FP/start
- Training N: 363 pitcher-seasons (those with prior-year data)

**Why cross-year r matters more than OOY r for 2026 deployment:**
OOY validation tests: "given 2025 Statcast metrics, can the model correctly score 2025 FP?"
That's same-season prediction. But in actual use, we feed 2025 metrics to predict 2026 FP.
Year-to-year metric stability is the bottleneck OOY doesn't test. A feature with high
same-year predictive power but low year-to-year stability looks great in OOY but fails in
real deployment. Cross-year r surfaces this directly.

**Derived columns needed:**
```python
df['abs_pfxz']      = df['avg_pfxz'].abs()
df['z_swing_pct']   = df['z_swing'] / df['in_zone']
df['xwoba_x_swstr'] = df['xwoba_contact'] * df['swstr_pct']
# ip_resid_lag1: requires fitting an IP model on V5 first, then lagging residuals
```

---

## Task: Build Automated Feature Search Script

### Script: `scripts/xfp_feature_search.py`

Create a script that:

1. **Loads all data** and builds every candidate feature
2. **Runs an exhaustive (or greedy) search** over feature combinations
3. **Evaluates each combination** with full OOY validation
4. **Logs every result** to `data/research/feature_search_log.csv`
5. **Prints a leaderboard** at the end sorted by OOY r

### Evaluation Functions — TWO ARE REQUIRED

Both functions must be run for every model variant. Log both. Primary ranking metric is
cross-year r. OOY r is secondary context.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import numpy as np

def ooy_evaluate(df, feats, label=""):
    """
    Same-year OOY: train on years != holdout, test on holdout year's own metrics.
    Tests: "can the model correctly score a pitcher given their own season's metrics?"
    Secondary metric — useful for model sanity but NOT the deployment scenario.
    """
    preds_all, acts_all, res_rows = [], [], []
    for yr in [2022, 2023, 2024, 2025]:
        tr = df[df['year'] != yr]
        te = df[df['year'] == yr].copy()
        tr_clean = tr.dropna(subset=feats + ['fp_per_start_actual'])
        te_clean = te.dropna(subset=feats + ['fp_per_start_actual'])
        if len(te_clean) < 10: continue
        pipe = Pipeline([('sc', StandardScaler()),
                         ('r', RidgeCV(alphas=np.logspace(-1, 5, 80), cv=5))])
        pipe.fit(tr_clean[feats], tr_clean['fp_per_start_actual'])
        te_clean = te_clean.copy()
        te_clean['pred'] = pipe.predict(te_clean[feats])
        preds_all.extend(te_clean['pred'])
        acts_all.extend(te_clean['fp_per_start_actual'])
        res_rows.append(te_clean)
    res = pd.concat(res_rows)
    res['resid'] = res['fp_per_start_actual'] - res['pred']
    r = np.corrcoef(preds_all, acts_all)[0, 1]
    k_bias_hi = res[res['k_pct'] > 0.30]['resid'].mean()
    k_bias_lo = res[res['k_pct'] < 0.18]['resid'].mean()
    rmse = np.sqrt(np.mean(res['resid']**2))
    return {'type': 'ooy', 'r': round(r, 5), 'k_bias_hi': round(k_bias_hi, 3),
            'k_bias_lo': round(k_bias_lo, 3), 'rmse': round(rmse, 3),
            'n': len(res), 'feats': label or str(feats)}


def cross_year_evaluate(df, feats, label=""):
    """
    Cross-year: use year T metrics to predict year T+1 FP.
    Tests: "given what a pitcher did last season, can we predict next season's FP?"
    PRIMARY metric — this is exactly the 2026 deployment scenario.
    Transitions tested: 2021→2022, 2022→2023, 2023→2024, 2024→2025.
    Only pitchers appearing in BOTH years are included (requires continuity).
    """
    preds_all, acts_all, res_rows = [], [], []
    transitions = [(2021,2022), (2022,2023), (2023,2024), (2024,2025)]
    for yr_train, yr_test in transitions:
        # Pitchers appearing in both years
        pitchers_train = set(df[df['year']==yr_train]['pitcher'])
        pitchers_test  = set(df[df['year']==yr_test]['pitcher'])
        shared = pitchers_train & pitchers_test

        train_rows = df[df['year']==yr_train][df[df['year']==yr_train]['pitcher'].isin(shared)]
        test_rows  = df[df['year']==yr_test ][df[df['year']==yr_test ]['pitcher'].isin(shared)].copy()

        # Features come from TRAIN year, target comes from TEST year
        merged = test_rows[['pitcher','fp_per_start_actual','k_pct']].merge(
            train_rows[['pitcher'] + feats], on='pitcher', how='inner')
        merged = merged.dropna(subset=feats + ['fp_per_start_actual'])
        if len(merged) < 10: continue

        # Fit model on ALL prior data (not just yr_train) excluding test year
        prior = df[(df['year'] < yr_test)].dropna(subset=feats + ['fp_per_start_actual'])
        pipe = Pipeline([('sc', StandardScaler()),
                         ('r', RidgeCV(alphas=np.logspace(-1, 5, 80), cv=5))])
        pipe.fit(prior[feats], prior['fp_per_start_actual'])

        merged['pred'] = pipe.predict(merged[feats])
        preds_all.extend(merged['pred'])
        acts_all.extend(merged['fp_per_start_actual'])
        res_rows.append(merged)

    res = pd.concat(res_rows)
    res['resid'] = res['fp_per_start_actual'] - res['pred']
    r = np.corrcoef(preds_all, acts_all)[0, 1]
    k_bias_hi = res[res['k_pct'] > 0.30]['resid'].mean()
    k_bias_lo = res[res['k_pct'] < 0.18]['resid'].mean()
    rmse = np.sqrt(np.mean(res['resid']**2))
    n_transitions = len(transitions)
    return {'type': 'cross_year', 'r': round(r, 5), 'k_bias_hi': round(k_bias_hi, 3),
            'k_bias_lo': round(k_bias_lo, 3), 'rmse': round(rmse, 3),
            'n': len(res), 'n_transitions': n_transitions, 'feats': label or str(feats)}


def ytd_tracking_evaluate(proj_df):
    """
    2026 YTD reality check: correlate pre-season xFP predictions with 2026 actual
    FP/start through current date. Noisy (only ~7 GS per pitcher) but a real-world
    sanity check that the cross-year model is tracking correctly in live deployment.
    proj_df must have columns: xfp_v7 (or whichever model), fp_actual (2026 YTD), gs
    Only include pitchers with gs >= 5 to reduce noise.
    """
    valid = proj_df[proj_df['gs'] >= 5].dropna(subset=['xfp_v7','fp_actual'])
    if len(valid) < 10:
        return {'type': 'ytd_2026', 'r': None, 'n': len(valid), 'note': 'insufficient data'}
    r = np.corrcoef(valid['xfp_v7'], valid['fp_actual'])[0, 1]
    bias = (valid['fp_actual'] - valid['xfp_v7']).mean()
    k_bias = valid[valid['k_pct'] > 0.30]['fp_actual'].sub(
             valid[valid['k_pct'] > 0.30]['xfp_v7']).mean()
    return {'type': 'ytd_2026', 'r': round(r, 5), 'bias': round(bias, 3),
            'k_bias': round(k_bias, 3), 'n': len(valid),
            'note': f'2026 YTD through ~{valid.gs.mean():.0f} GS/pitcher'}
```

**Interpretation guide:**

| Scenario | What it means | Action |
|----------|---------------|--------|
| cross_year r ≈ ooy_r | Metrics are stable year-to-year — model is genuinely predictive | Feature is clean, keep it |
| cross_year r << ooy_r by >0.02 | Feature captures within-season patterns that don't persist — model will overfit | Drop the feature |
| cross_year r > ooy_r | Feature predicts future better than present — strong forward signal | Prioritize this feature |
| ytd_2026 r near cross_year r | Model is tracking as expected in live deployment | ✓ |
| ytd_2026 r << cross_year r | Something is structurally different about 2026 — investigate | Flag for manual review |

### Search Space — Candidate Features to Add/Swap

**Available in sp_multiyr.csv (already computed):**
- `z_contact_pct` = z_contact / z_swing (in-zone contact rate, flip of in-zone whiff)
- `swing_pct` (overall swing rate)
- `contact_pct` (overall contact rate on swings)
- `avg_pfxx` (horizontal movement — may separate from abs_pfxz)
- `gb_pct`, `hard_hit_pct`, `barrel_pct` (batted ball — note: slightly downstream, check circularity)

**Derivable from existing columns:**
- `bip_pct` = bip / tbf (BIP rate — how often PA ends in contact)
- `xwoba_nc_pa` = xwoba_contact × (1 - swstr_pct) (per-PA damage proxy)
- `xwoba_x_cplus` = xwoba_contact × c_plus_swstr (wider K proxy interaction)
- `swstr_sq` = swstr_pct² (nonlinearity for elite K pitchers)
- `cplus_sq` = c_plus_swstr² (nonlinearity)
- `velo_x_swstr` = avg_velo × swstr_pct (velocity-amplified whiff)
- `velo_x_cplus` = avg_velo × c_plus_swstr
- `o_swing_x_swstr` = o_swing_pct × swstr_pct (double threat: chases AND misses)
- `zone_x_oswing` = zone_pct × o_swing_pct
- `abs_pfxz_x_velo` = abs_pfxz × avg_velo
- Log transforms: `log_swstr` = log(swstr_pct), `log_cplus` = log(c_plus_swstr)
- `k_bb_proxy` = c_plus_swstr - bb_pct (not in training data, bb_pct is available)
- `bb_pct` (walk rate — check: not in FP formula directly, so non-circular!)
- `hard_hit_neg` = 1 - hard_hit_pct (flipped hard hit rate for contact quality)

**ip_resid variants:**
- `ip_resid_career` = career average IP residual (from ip_resid_prior.csv)
- `ip_resid_2yr_avg` = mean of last 2 years ip_resid

**From sp_extra_metrics.csv:**
- `avg_ev` (average exit velocity — check collinearity with xwoba_contact)

### Search Strategy

**Phase 1: Single-feature additions to V6**
For each candidate feature X, test V6 + [X]. Log result. Mark as "winner" if OOY r > 0.9081
OR k_bias_hi < 0.21.

**Phase 2: Best additions combined**
Take top 3-5 features from Phase 1. Test all 2-combinations added to V6 simultaneously.

**Phase 3: Replacements**
For key V6 features (esp. xwoba_contact, z_swing_pct), try replacing with variants.
E.g., xwoba_contact → xwoba_nc_pa; or xwoba_contact → (xwoba_contact + xwoba_nc_pa both).

**Phase 4: Kitchen sink then backward elimination**
Start with all non-collinear candidates. Drop one at a time (smallest Ridge coef), stop
when OOY r starts dropping.

### Search with ip_resid_lag1 (sample=363)

For any feature involving ip_resid_lag1, training sample shrinks to 363 obs (2022-2025 only,
pitchers with prior year). For features that don't require lag, use all 636 obs (2021-2025).
Report both: `r_full` (no lag, N=636) and `r_lag` (with lag, N=363).

---

## Output Format

### `data/research/feature_search_log.csv`
```
run_id, timestamp, feats, n_feats, n_obs, ooy_r, k_bias_hi, k_bias_lo, rmse, notes
```

### `data/research/feature_search_report.md`
Auto-generated after each run. Sections:
1. Current champion (best OOY r)
2. Best K-bias reduction
3. Top 10 feature additions
4. Features that consistently hurt performance
5. Recommended V7 features

---

## Circularity Audit Function

Before testing any feature, run this check:
```python
CIRCULAR = ['k', 'bb', 'hbp', 'h', 'hr', 'er', 'outs', 'ip',
            'fp_per_start', 'k_per_start', 'h_per_start', 'bb_per_start',
            'hbp_per_start', 'hr_per_start', 'er_per_start']

def is_circular(feat_name):
    """Returns True if feature is directly derived from FP formula components."""
    for c in CIRCULAR:
        if feat_name.startswith(c + '_') or feat_name == c:
            return True
    return False
# Note: k_pct = k/tbf is semi-circular (skip), bip_pct = bip/tbf is OK (bip not in formula)
# bb_pct = bb/tbf is OK (bb is in formula as bb/start, but rate is one step removed — flag as semi-circular)
```

---

## After Finding V7 Features

Once you have the best feature set:

1. **Retrain final V7 model** on all applicable data
2. **Apply to 2026 projections** at `/tmp/xfp_v6_final.csv` (update xwoba shrinkage if needed)
3. **Update dashboard** at `data/outputs/xfp_v6_dashboard.html` → rename to `xfp_v7_dashboard.html`
4. **Append to research doc** `data/research/xfp_model_research.md` with V7 findings
5. **Save final model pipeline** using joblib: `data/models/xfp_v7_pipeline.pkl`

---

## Quick Validation — Does Your V7 Beat V6?

V6 benchmarks to beat (compute all three for V6 first as baseline, then compare V7):

| Metric | V6 Baseline | Target |
|--------|-------------|--------|
| OOY r (same-year) | 0.9081 | Higher |
| Cross-year r (deployment) | **UNKNOWN — compute first** | Higher than OOY gap allows |
| High-K bias >30% | +0.21 | Closer to 0 |
| Low-K bias <18% | -0.36 | Closer to 0 |
| 2026 YTD r | **UNKNOWN — compute first** | Tracks cross-year r |

**The OOY → cross-year gap is the key diagnostic.** If V6 cross-year r = 0.88 but OOY r = 0.91,
the model is 0.03 points of r inflated by within-season signal. V7 features should close that gap,
not just push OOY r higher. A feature that raises OOY r but widens the cross-year gap is making
the model worse for actual 2026 deployment even though the OOY number looks better.

---

## Notes on Token Efficiency

- **Run the script headless** — don't print every row, just log to CSV and print summary
- **Timeout guard** — add `signal.alarm(30)` or use `multiprocessing` with timeout per model fit
- **Cache the data load** — load once at top, derive all features, then loop over feature lists
- **Use CV r not OOY r for quick phase 1 screening** — CV is 5x faster, then do full OOY on top-20
- **Seed everything** — `np.random.seed(42)` before cross_val_score for reproducibility

---

## Phase 5: Nonlinear Ceiling Check

**This is critical.** Before declaring V7 "done", run nonlinear models on the same feature set.
If they significantly outperform Ridge, there is exploitable nonlinearity we're leaving on the table
and we should hunt for the right feature engineering to close the gap.

```python
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb  # pip install xgboost

def ooy_evaluate_nonlinear(df, feats, model_name='xgb'):
    """Same OOY protocol but with nonlinear model."""
    preds_all, acts_all, res_rows = [], [], []
    for yr in [2022, 2023, 2024, 2025]:
        tr = df[df['year'] != yr].dropna(subset=feats + ['fp_per_start_actual'])
        te = df[df['year'] == yr].copy().dropna(subset=feats + ['fp_per_start_actual'])
        if len(te) < 10: continue

        sc = StandardScaler()
        X_tr = sc.fit_transform(tr[feats])
        X_te = sc.transform(te[feats])
        y_tr = tr['fp_per_start_actual'].values

        if model_name == 'xgb':
            m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=0)
        elif model_name == 'rf':
            m = RandomForestRegressor(n_estimators=500, max_depth=5,
                                      min_samples_leaf=5, random_state=42)
        elif model_name == 'gbm':
            m = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                          learning_rate=0.05, random_state=42)

        m.fit(X_tr, y_tr)
        te = te.copy(); te['pred'] = m.predict(X_te)
        preds_all.extend(te['pred']); acts_all.extend(te['fp_per_start_actual'])
        res_rows.append(te)

    res = pd.concat(res_rows)
    res['resid'] = res['fp_per_start_actual'] - res['pred']
    r = np.corrcoef(preds_all, acts_all)[0, 1]
    k_bias_hi = res[res['k_pct'] > 0.30]['resid'].mean()
    return {'model': model_name, 'r': round(r, 5),
            'k_bias_hi': round(k_bias_hi, 3), 'n': len(res)}
```

**Run this matrix:**
- XGBoost on V6 features
- XGBoost on V6 + all candidate features (kitchen sink)
- Random Forest on V6 features
- Ridge on V6 features (control — should match 0.9081)

**Interpret results:**

| Gap (XGB vs Ridge) | Meaning | Action |
|---|---|---|
| < 0.003 | Ridge is at signal ceiling | V6/V7 is probably optimal. Done. |
| 0.003–0.010 | Mild nonlinearity exists | Hunt for 1-2 interaction/polynomial terms that capture it |
| > 0.010 | Significant nonlinearity | Consider stacked model or polynomial feature expansion |

**If XGBoost wins by >0.005**, run SHAP to identify which features have nonlinear relationships:
```python
import shap
# After fitting XGBoost on all data:
explainer = shap.TreeExplainer(m)
shap_values = explainer.shap_values(X_all)
shap.summary_plot(shap_values, pd.DataFrame(X_all, columns=feats))
# Look for features with curved SHAP dependence plots — those are candidates
# for log/sqrt/squared transformations or threshold-based splits
```

---

## Phase 6: Alternative Model Architectures

Test these if Phase 5 shows meaningful nonlinear signal:

### Stacked Ensemble
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# Level 1 estimators
estimators = [
    ('ridge', Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-1,5,80)))])),
    ('gbm',   GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)),
    ('rf',    RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42)),
]
# Level 2 meta-learner
stack = StackingRegressor(estimators=estimators,
                          final_estimator=Ridge(alpha=1.0),
                          cv=5, passthrough=False)
# Evaluate with OOY protocol — watch for overfitting (N=363 is small for stacking)
```

### Lasso + ElasticNet (automatic feature selection)
```python
from sklearn.linear_model import LassoCV, ElasticNetCV

# LassoCV will zero out irrelevant features automatically
lasso_pipe = Pipeline([('sc', StandardScaler()),
                       ('l', LassoCV(alphas=np.logspace(-3, 2, 60), cv=10, max_iter=5000))])
# After fitting: lasso_pipe.named_steps['l'].coef_ — zeros = dropped features
# Use surviving features as a refined feature set for final Ridge
```

### Polynomial Ridge (degree-2 interactions only)
```python
from sklearn.preprocessing import PolynomialFeatures

# Only interactions (no squared terms) to control dimensionality
poly_pipe = Pipeline([
    ('sc',   StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('r',    RidgeCV(alphas=np.logspace(0, 6, 80), cv=10))
])
# WARNING: 11 features → 66 interaction terms. Ridge will regularize most away.
# Run OOY eval — if it beats 0.9081 cleanly, there are interaction terms worth isolating
# Then extract which poly features have largest coefs and add them explicitly to V7
```

---

## Phase 7: Pull Fresh Statcast Data (If Needed)

If the search exhausts current features and the ceiling check shows remaining nonlinear signal,
pull additional raw Statcast data that isn't in sp_multiyr.csv:

```python
import pybaseball as pb

# Spin rate by pitch type (fastball spin may predict swing-and-miss independently of velocity)
# Note: population-level r<0.15 was found in earlier research, but by PITCH TYPE it may differ
def pull_spin_by_type(year):
    raw = pb.statcast(f'{year}-03-01', f'{year}-10-31')
    raw = raw[raw['game_type']=='R']
    # Filter to SPs (pitcher_id in our known SP list)
    # Group by pitcher × pitch_type, compute mean spin_rate, release_speed
    # Pivot to wide: pitcher, year, FF_spin, SL_spin, CH_spin, etc.
    pass

# Pitch mix / tunneling metrics
# - pitch_type diversity (entropy of pitch usage)
# - arm-side run on breaking balls
# - vertical approach angle (VAA) — not in sp_multiyr.csv but in raw Statcast

# Zone profile
# - top-of-zone vs bottom-of-zone split (command to different locations)
# - FUTURE: release point consistency (std dev of release_pos_x/z)
```

**New Statcast pulls to consider:**

| Metric | Rationale | Expected signal |
|--------|-----------|-----------------|
| Fastball spin rate (per pitcher) | Independent of velocity; predicts swing-and-miss on heaters | Moderate (r≈0.25-0.35 with swstr?) |
| Vertical Approach Angle (VAA) | Steeper = harder to barrel; not correlated with velo | Unknown — test it |
| Pitch entropy / mix diversity | Keeps batters guessing; correlates with lower contact | Low-moderate |
| Release point consistency (std) | Lower std = more deceptive tunneling | Unknown |
| First-pitch strike rate | Counts not in FP formula; predicts count leverage | Moderate |
| Strikeout-to-walk ratio proxy | c_plus_swstr - bb_pct interaction | Likely already captured |

For each new metric, run the same Phase 1 single-feature addition test before anything else.

---

## Phase 8: Pitcher Aging / Role Stability Features

These require multi-year pitcher history (already have 2021-2025):

```python
# Age (from player birth years — requires an external lookup or scrape)
# Years as SP (consecutive SP seasons in data)
df_tenure = df.groupby('pitcher').agg(
    n_seasons=('year','count'),
    first_year=('year','min'),
    career_fp_mean=('fp_per_start_actual','mean'),
    career_fp_std=('fp_per_start_actual','std'),
).reset_index()
# career_fp_mean: career average FP/start (SEMI-CIRCULAR — contains FP, flag it)
# n_seasons: years of experience as SP — non-circular, may capture durability/consistency

# Rolling form: prior-year actual FP vs career mean (regression-to-mean signal)
df = df.sort_values(['pitcher','year'])
df['fp_lag1'] = df.groupby('pitcher')['fp_per_start_actual'].shift(1)
df['fp_lag2'] = df.groupby('pitcher')['fp_per_start_actual'].shift(2)
df['fp_career_mean_lag'] = df.groupby('pitcher')['fp_per_start_actual'].transform(
    lambda x: x.expanding().mean().shift(1)
)
# fp_lag1 IS semi-circular (it's a past FP value) — use with caution / flag in circularity audit
# BUT: it's a valid predictive feature for in-season stability analysis
# Decision: test it, label results clearly as "uses prior FP (semi-circular)"
```

---

## Phase 9: Target-Encode Pitcher Identity

For pitchers with 3+ seasons of history, their identity itself is a signal above and beyond
their metrics in any given year. Test a simplified version:

```python
# Pitcher FP percentile rank across career (non-circular: rank, not raw FP)
df['pitcher_career_rank'] = df.groupby('pitcher')['fp_per_start_actual'].transform(
    lambda x: x.expanding().mean().shift(1).rank(pct=True)
)
# This is semi-circular but captures "elite pitchers stay elite" beyond what stuff metrics show
# Test with and without, label clearly
```

---

## Phase 10: Build the V7 Dashboard with Model Comparison Panel

Once V7 features are locked, the dashboard should add:

1. **Model comparison tab** — show V5, V6, V7 xFP side by side for each pitcher
2. **Nonlinear ceiling** — if XGB OOY r was tested, show the "theoretical ceiling" xFP
3. **SHAP contribution bars** — for XGB model, show per-pitcher feature contributions
4. **Residual explorer** — scatter plot of actual vs predicted, highlight systematic outliers
5. **K-rate bias chart** — bar chart of mean residual by K-rate decile (show improvement V5→V7)

Dashboard output: `data/outputs/xfp_v7_dashboard.html`

---

## Tool Choice: Terminal Claude Code, NOT Cowork

Run this in **Claude Code via terminal or VS Code** — not the Cowork desktop app.
Cowork has a 45-second bash timeout per call. This pipeline runs 10-30 minutes.
Use `claude` in your terminal, or the Claude Code extension in VS Code.

---

## Opus 1M Context Window — Full One-Shot Prompt

This prompt is designed for **claude-opus-4-5** (or later Opus model with 1M+ context).
The strategy: load EVERYTHING into context upfront, reason across it all at once,
never lose track of prior phase results, validate obsessively at each step.

```
CONTEXT LOAD — DO THIS FIRST BEFORE ANY CODE:

Read the following files completely into your context window. Do not summarize.
Read every line. You will need all of it.

1. data/research/CLAUDE_CODE_HANDOFF.md          ← you're reading this
2. data/research/xfp_model_research.md           ← full research history
3. /tmp/sp_multiyr.csv                           ← 636 training rows (or rederive if missing)
4. /tmp/sp_extra_metrics.csv                     ← xwoba_contact per pitcher-season
5. /tmp/ip_resid_prior.csv                       ← IP residuals for 2026 projection
6. /tmp/xfp_v6_final.csv                        ← current 2026 projections

After reading all files, print a one-paragraph summary of what you understand
the project to be, what V6 currently achieves, and what the ceiling check will
tell us. This is your validation that context loaded correctly.

---

YOUR MISSION: Run all 10 phases of the xFP optimization pipeline in a single
autonomous session. Do not stop. Do not ask for permission between phases.
Validate results at every checkpoint before proceeding. Write all findings
to disk so nothing is lost if context fills.

---

PHASE 1 — CV SCREENING (fast, ~2 min)
Install requirements: pip install pybaseball xgboost shap scikit-learn pandas numpy joblib
Build ALL candidate features listed in CLAUDE_CODE_HANDOFF.md.
For each candidate feature, run 10-fold CV on V6+[feature]. Log to feature_search_log.csv.
VALIDATION CHECKPOINT: Before moving on, verify:
  - feature_search_log.csv exists and has at least 20 rows
  - V6 baseline CV r is within 0.002 of 0.9022 (sanity check)
  - No NaN in results
Print: "PHASE 1 COMPLETE — top 5 features: [list]"

PHASE 2 — OOY VALIDATION (medium, ~5 min)
Take every feature from Phase 1 with CV r > V6 baseline.
Run full OOY holdout (2022/23/24/25) on each.
VALIDATION CHECKPOINT:
  - OOY r for V6 baseline is within 0.003 of 0.9081
  - High-K bias for V6 baseline is within 0.05 of +0.21
  - At least 1 variant was tested
Write interim report section "Phase 2 winners" to feature_search_report.md.
Print: "PHASE 2 COMPLETE — OOY winners: [list with r values]"

PHASE 3 — FEATURE REPLACEMENTS (~3 min)
For each Phase 2 winner, also test replacing xwoba_contact with it (not just adding).
Also test: xwoba_contact replaced by xwoba_nc_pa, xwoba_per_pa.
Log all results.
Print: "PHASE 3 COMPLETE"

PHASE 4 — BACKWARD ELIMINATION (~5 min)
Build kitchen-sink feature set: V6 + all Phase 2 winners + all candidates.
Fit Ridge. Drop the feature with smallest |standardized coef|. Re-evaluate OOY.
Repeat until OOY r drops below Phase 2 best. Record the optimal trim point.
VALIDATION CHECKPOINT: final set must not include circular features (run circularity check).
Print: "PHASE 4 COMPLETE — optimal feature set: [list], OOY r: [value]"

PHASE 5 — NONLINEAR CEILING CHECK (~5 min)
Run XGBoost, RandomForest, GBM with same OOY protocol on best feature set from Phase 4.
Also run on V6 features as control.
VALIDATION CHECKPOINT:
  - Ridge on V6 features gives r ≈ 0.9081 (±0.003). If not, something is wrong — STOP and debug.
  - XGB on V6 features: record r
  - XGB on Phase 4 features: record r
Compute gap = XGB_r - Ridge_r.
If gap < 0.003: "Ridge is at ceiling — proceed with Ridge V7"
If gap 0.003-0.010: "Mild nonlinearity — hunt for 2-3 polynomial terms"
If gap > 0.010: "Strong nonlinearity — run SHAP, consider stacking"
Print gap, decision, and reasoning.

PHASE 6 — FOLLOW-ON BASED ON PHASE 5 RESULT
If gap < 0.003: Skip to Phase 9.
If gap >= 0.003:
  - Run SHAP on XGB to find features with nonlinear dependence
  - Add polynomial/log transforms for top 3 SHAP nonlinear features
  - Re-run OOY evaluation
  - If stacking warranted: build StackingRegressor and OOY-evaluate
VALIDATION CHECKPOINT: stacked model must be evaluated with strict OOY (no leakage).
Print: "PHASE 6 COMPLETE — best architecture: [Ridge/Stack/XGB], r: [value]"

PHASE 7 — FRESH STATCAST PULL (if gap still > 0.005 after Phase 6)
Pull 2021-2025 SP-level: spin_rate by pitch type, vertical approach angle (VAA),
pitch entropy (usage diversity), release point consistency (std_dev).
Use pybaseball.statcast() monthly to avoid timeout.
Merge to training data on pitcher × year. Test each new metric with Phase 1 protocol.
VALIDATION CHECKPOINT: new data rows must match existing pitcher-season count (636 ±5%).
Print: "PHASE 7 COMPLETE — new features that help: [list]"

PHASE 8 — TENURE + AGING FEATURES
Compute: n_seasons, career_fp_mean (labeled semi-circular), ip_resid_career.
Test each with same OOY protocol. Label semi-circular results clearly in log.
Print: "PHASE 8 COMPLETE"

PHASE 9 — LOCK V7 FEATURE SET
From all phases, pick the combination that maximizes:
  score = OOY_r × 2 + (0.21 - |k_bias_hi|) × 0.5
  (reward accuracy, reward K-bias correction)
Must be fully non-circular (or semi-circular with clear label).
Train final V7 Ridge on ALL available data (636 or 363 obs depending on lag).
Save: joblib.dump(pipe_v7, 'data/models/xfp_v7_pipeline.pkl')
If nonlinear won: also save data/models/xfp_v7_xgb.pkl
VALIDATION CHECKPOINT:
  - Reload pipeline from disk, predict on training data, verify r > 0.85 (sanity)
  - Confirm no circular features
  - Print feature list and standardized coefficients

PHASE 10 — REBUILD 2026 PROJECTIONS + DASHBOARD
Apply V7 to /tmp/xfp_v6_final.csv (same projection inputs, new model).
Apply Bayesian xwoba shrinkage (PRIOR_N=40) to projections as in V6.
Save projections to data/outputs/xfp_v7_projections.csv.
Build new dashboard data/outputs/xfp_v7_dashboard.html based on V6 dashboard structure.
Dashboard must include:
  - V5 / V6 / V7 xFP columns side by side
  - Delta vs V6 highlighted for big movers
  - K-rate bias chart (bar chart: mean residual by K decile, V5 vs V6 vs V7)
  - Schlittler rank and xFP in all three model versions
  - If nonlinear model was used: show "ceiling xFP" column
VALIDATION CHECKPOINT:
  - Open dashboard HTML, verify it loads without JS errors (run: python3 -c "open file, check structure")
  - Confirm Schlittler rank improved from V5 (#74) and V6 (#48)
  - Confirm top 10 contains at least 5 pitchers with gs >= 10 (not purely small-sample)

FINAL — WRITE RESEARCH NOTES
Append complete V7 findings to data/research/xfp_model_research.md.
Include: feature set, coefficients, OOY r, K-bias comparison table V5/V6/V7,
nonlinear ceiling gap, and the 3 biggest individual mover explanations.

---

CONTINUOUS VALIDATION RULES (apply throughout):
1. After EVERY model fit, assert OOY r is in [0.70, 0.99]. Outside that range = bug.
2. After EVERY data merge, assert row count hasn't dropped by more than 15%.
3. After EVERY feature derivation, assert no infinite or NaN values.
4. Keep feature_search_log.csv append-only — never overwrite prior results.
5. If any phase produces unexpected results (r drops >0.01 unexpectedly, k_bias
   moves wrong direction), STOP, write a "ANOMALY:" entry to the log with details,
   debug before continuing.
6. Every 30 minutes of wall time: write current state to feature_search_report.md
   even if mid-phase, so nothing is lost.

---

BENCHMARKS TO BEAT:
- V6 OOY r: 0.9081
- V6 high-K bias (k_pct > 0.30): +0.21
- V6 low-K bias (k_pct < 0.18): -0.36
- Non-circular constraint: HARD (no exceptions without explicit semi-circular label)
```

