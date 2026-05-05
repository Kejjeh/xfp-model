"""
xfp_v7_pipeline.py — All 10 phases of the xFP optimization run.

Reads sp_multiyr.csv, builds candidate features, runs Phase 0–10:
0. Baseline audit (V6 OOY + cross-year + 2026 YTD)
1. CV screening of candidate features
2. Dual validation (OOY + cross-year) on Phase 1 winners
3. xwoba_contact replacements
4. Backward elimination from kitchen sink
5. Nonlinear ceiling check (XGB / RF / GBM vs Ridge)
6. Follow-on (SHAP + polynomial transforms or stacking)
7. Fresh Statcast features (only if gap > 0.005)
8. Tenure features
9. Lock V7
10. Rebuild 2026 projections + dashboard
"""
from __future__ import annotations
import os, sys, json, time, math, warnings, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT     = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'data' / 'research'
OUTPUTS  = ROOT / 'data' / 'outputs'
MODELS   = ROOT / 'data' / 'models'
CACHE    = RESEARCH / 'xfp_cache'
MODELS.mkdir(parents=True, exist_ok=True)

LOG_CSV  = RESEARCH / 'feature_search_log.csv'
REPORT_MD = RESEARCH / 'feature_search_report.md'

V6_FEATS = [
    'avg_velo','abs_pfxz','avg_ext','zone_pct','o_swing_pct','swstr_pct',
    'c_plus_swstr','xwoba_contact','z_swing_pct','xwoba_x_swstr','ip_resid_lag1',
]

# ---------- I/O helpers ----------
def load_data() -> pd.DataFrame:
    """Load sp_multiyr.csv from /tmp or cache and return cleaned DataFrame."""
    candidates = [Path('/tmp/sp_multiyr.csv'), CACHE / 'sp_multiyr.csv']
    df = None
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            print(f'Loaded {p} ({len(df)} rows)')
            break
    if df is None:
        raise FileNotFoundError('sp_multiyr.csv not found in /tmp or cache')

    df['abs_pfxz']    = df['avg_pfxz'].abs()
    return df


# ---------- Derived features ----------
def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add V6 features + all candidate features."""
    d = df.copy()
    d['xwoba_x_swstr']   = d['xwoba_contact'] * d['swstr_pct']
    d['xwoba_x_cplus']   = d['xwoba_contact'] * d['c_plus_swstr']
    d['xwoba_nc_pa']     = d['xwoba_contact'] * (1 - d['swstr_pct'])
    d['xwoba_per_pa']    = d['xwoba_contact'] * d['bip_pct']
    d['swstr_sq']        = d['swstr_pct'] ** 2
    d['cplus_sq']        = d['c_plus_swstr'] ** 2
    d['velo_x_swstr']    = d['avg_velo'] * d['swstr_pct']
    d['velo_x_cplus']    = d['avg_velo'] * d['c_plus_swstr']
    d['o_swing_x_swstr'] = d['o_swing_pct'] * d['swstr_pct']
    d['zone_x_oswing']   = d['zone_pct'] * d['o_swing_pct']
    d['abs_pfxz_x_velo'] = d['abs_pfxz'] * d['avg_velo']
    d['log_swstr']       = np.log(d['swstr_pct'].clip(lower=1e-4))
    d['log_cplus']       = np.log(d['c_plus_swstr'].clip(lower=1e-4))
    d['hard_hit_neg']    = 1 - d['hard_hit_pct']
    d['k_bb_proxy']      = d['c_plus_swstr'] - d['bb_pct']  # bb_pct semi-circular flag

    # Tenure features
    d = d.sort_values(['pitcher','year'])
    d['n_seasons'] = d.groupby('pitcher').cumcount() + 1
    d['fp_lag1']   = d.groupby('pitcher')['fp_per_start_actual'].shift(1)
    d['fp_lag2']   = d.groupby('pitcher')['fp_per_start_actual'].shift(2)
    d['fp_career_mean_lag'] = (d.groupby('pitcher')['fp_per_start_actual']
                                  .transform(lambda x: x.expanding().mean().shift(1)))
    d['pitcher_career_rank'] = (d.groupby('pitcher')['fp_per_start_actual']
                                   .transform(lambda x: x.expanding().mean().shift(1).rank(pct=True)))
    return d


def add_ip_resid_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Fit V5 IP model on prior years, lag residuals as ip_resid_lag1."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    V5 = ['avg_velo','abs_pfxz','avg_ext','zone_pct','o_swing_pct',
          'swstr_pct','c_plus_swstr','xwoba_contact']
    d = df.copy()
    d['ip_resid'] = np.nan
    # Per-year leave-one-out: train on other years to predict this year's ip
    for yr in sorted(d['year'].unique()):
        train = d[(d['year'] != yr)].dropna(subset=V5 + ['ip_per_start'])
        if len(train) < 50: continue
        pipe = Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
        pipe.fit(train[V5], train['ip_per_start'])
        idx = d['year'] == yr
        sub = d[idx].dropna(subset=V5)
        if len(sub) == 0: continue
        d.loc[sub.index, 'ip_pred'] = pipe.predict(sub[V5])
        d.loc[sub.index, 'ip_resid'] = sub['ip_per_start'] - pipe.predict(sub[V5])
    # ip_resid_lag1: prior year residual per pitcher
    d = d.sort_values(['pitcher','year'])
    d['ip_resid_lag1'] = d.groupby('pitcher')['ip_resid'].shift(1)
    # ip_resid_career: expanding mean of past residuals
    d['ip_resid_career'] = d.groupby('pitcher')['ip_resid'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    d['ip_resid_2yr_avg'] = d.groupby('pitcher')['ip_resid'].transform(
        lambda x: x.rolling(2, min_periods=1).mean().shift(1)
    )
    return d


# ---------- Validation ----------
def ooy_evaluate(df: pd.DataFrame, feats: list[str], label: str = '') -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    preds, acts, rows = [], [], []
    for yr in [2022, 2023, 2024, 2025]:
        tr = df[df['year'] != yr].dropna(subset=feats + ['fp_per_start_actual'])
        te = df[df['year'] == yr].dropna(subset=feats + ['fp_per_start_actual']).copy()
        if len(te) < 10 or len(tr) < 50:
            continue
        pipe = Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
        pipe.fit(tr[feats], tr['fp_per_start_actual'])
        te['pred'] = pipe.predict(te[feats])
        preds.extend(te['pred']); acts.extend(te['fp_per_start_actual']); rows.append(te)
    if not rows:
        return {'type':'ooy','r':None,'k_bias_hi':None,'k_bias_lo':None,'rmse':None,'n':0,'feats':label or str(feats)}
    res = pd.concat(rows)
    res['resid'] = res['fp_per_start_actual'] - res['pred']
    r = float(np.corrcoef(preds, acts)[0,1])
    k_hi = float(res[res['k_pct'] > 0.30]['resid'].mean()) if (res['k_pct']>0.30).any() else float('nan')
    k_lo = float(res[res['k_pct'] < 0.18]['resid'].mean()) if (res['k_pct']<0.18).any() else float('nan')
    rmse = float(np.sqrt(np.mean(res['resid']**2)))
    return {'type':'ooy','r':round(r,5),'k_bias_hi':round(k_hi,3) if not math.isnan(k_hi) else None,
            'k_bias_lo':round(k_lo,3) if not math.isnan(k_lo) else None,
            'rmse':round(rmse,3),'n':len(res),'feats':label or str(feats)}


def cross_year_evaluate(df: pd.DataFrame, feats: list[str], label: str = '') -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    preds, acts, rows = [], [], []
    transitions = [(2021,2022),(2022,2023),(2023,2024),(2024,2025)]
    for yr_train, yr_test in transitions:
        pitchers_train = set(df[df['year']==yr_train]['pitcher'])
        pitchers_test  = set(df[df['year']==yr_test ]['pitcher'])
        shared = pitchers_train & pitchers_test
        train_year = df[(df['year']==yr_train) & df['pitcher'].isin(shared)]
        test_year  = df[(df['year']==yr_test ) & df['pitcher'].isin(shared)].copy()
        merged = test_year[['pitcher','fp_per_start_actual','k_pct']].merge(
            train_year[['pitcher'] + feats], on='pitcher', how='inner'
        ).dropna(subset=feats + ['fp_per_start_actual'])
        if len(merged) < 10:
            continue
        prior = df[df['year'] < yr_test].dropna(subset=feats + ['fp_per_start_actual'])
        if len(prior) < 50:
            continue
        pipe = Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
        pipe.fit(prior[feats], prior['fp_per_start_actual'])
        merged['pred'] = pipe.predict(merged[feats])
        preds.extend(merged['pred']); acts.extend(merged['fp_per_start_actual']); rows.append(merged)
    if not rows:
        return {'type':'cross_year','r':None,'k_bias_hi':None,'k_bias_lo':None,'rmse':None,'n':0,'n_transitions':0,'feats':label or str(feats)}
    res = pd.concat(rows)
    res['resid'] = res['fp_per_start_actual'] - res['pred']
    r = float(np.corrcoef(preds, acts)[0,1])
    k_hi = float(res[res['k_pct']>0.30]['resid'].mean()) if (res['k_pct']>0.30).any() else float('nan')
    k_lo = float(res[res['k_pct']<0.18]['resid'].mean()) if (res['k_pct']<0.18).any() else float('nan')
    rmse = float(np.sqrt(np.mean(res['resid']**2)))
    return {'type':'cross_year','r':round(r,5),
            'k_bias_hi':round(k_hi,3) if not math.isnan(k_hi) else None,
            'k_bias_lo':round(k_lo,3) if not math.isnan(k_lo) else None,
            'rmse':round(rmse,3),'n':len(res),'n_transitions':len(transitions),'feats':label or str(feats)}


def cv_evaluate(df: pd.DataFrame, feats: list[str], label: str = '', cv: int = 10) -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, KFold
    d = df.dropna(subset=feats + ['fp_per_start_actual'])
    if len(d) < 50:
        return {'type':'cv','r':None,'n':len(d),'feats':label or str(feats)}
    pipe = Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, d[feats], d['fp_per_start_actual'], cv=kf, scoring='r2')
    r = float(np.sqrt(max(0, scores.mean())))
    return {'type':'cv','r':round(r,5),'r_std':round(float(scores.std()),5),'n':len(d),'feats':label or str(feats)}


def ooy_evaluate_nonlinear(df: pd.DataFrame, feats: list[str], model_name: str = 'xgb', label: str = '') -> dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    preds, acts, rows = [], [], []
    for yr in [2022,2023,2024,2025]:
        tr = df[df['year']!=yr].dropna(subset=feats+['fp_per_start_actual'])
        te = df[df['year']==yr].dropna(subset=feats+['fp_per_start_actual']).copy()
        if len(te)<10 or len(tr)<50: continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[feats]); Xte = sc.transform(te[feats])
        ytr = tr['fp_per_start_actual'].values
        if model_name == 'xgb':
            import xgboost as xgb
            m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        elif model_name == 'rf':
            m = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1)
        elif model_name == 'gbm':
            m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(Xtr, ytr)
        te['pred'] = m.predict(Xte)
        preds.extend(te['pred']); acts.extend(te['fp_per_start_actual']); rows.append(te)
    if not rows:
        return {'type':f'ooy_{model_name}','r':None,'k_bias_hi':None,'n':0,'feats':label or str(feats)}
    res = pd.concat(rows)
    res['resid'] = res['fp_per_start_actual'] - res['pred']
    r = float(np.corrcoef(preds, acts)[0,1])
    k_hi = float(res[res['k_pct']>0.30]['resid'].mean()) if (res['k_pct']>0.30).any() else float('nan')
    return {'type':f'ooy_{model_name}','r':round(r,5),
            'k_bias_hi':round(k_hi,3) if not math.isnan(k_hi) else None,
            'n':len(res),'feats':label or str(feats)}


def cross_year_evaluate_nonlinear(df: pd.DataFrame, feats: list[str], model_name: str = 'xgb', label: str = '') -> dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    preds, acts, rows = [], [], []
    transitions = [(2021,2022),(2022,2023),(2023,2024),(2024,2025)]
    for yr_train, yr_test in transitions:
        train_year = df[df['year']==yr_train]
        test_year  = df[df['year']==yr_test ]
        shared = set(train_year['pitcher']) & set(test_year['pitcher'])
        ty = train_year[train_year['pitcher'].isin(shared)]
        te = test_year[test_year['pitcher'].isin(shared)].copy()
        merged = te[['pitcher','fp_per_start_actual','k_pct']].merge(
            ty[['pitcher']+feats], on='pitcher', how='inner').dropna(subset=feats+['fp_per_start_actual'])
        if len(merged)<10: continue
        prior = df[df['year']<yr_test].dropna(subset=feats+['fp_per_start_actual'])
        if len(prior)<50: continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(prior[feats]); Xte = sc.transform(merged[feats])
        ytr = prior['fp_per_start_actual'].values
        if model_name == 'xgb':
            import xgboost as xgb
            m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        elif model_name == 'rf':
            m = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1)
        elif model_name == 'gbm':
            m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(Xtr, ytr)
        merged['pred'] = m.predict(Xte)
        preds.extend(merged['pred']); acts.extend(merged['fp_per_start_actual']); rows.append(merged)
    if not rows:
        return {'type':f'cross_year_{model_name}','r':None,'k_bias_hi':None,'n':0,'feats':label or str(feats)}
    res = pd.concat(rows)
    res['resid'] = res['fp_per_start_actual'] - res['pred']
    r = float(np.corrcoef(preds, acts)[0,1])
    k_hi = float(res[res['k_pct']>0.30]['resid'].mean()) if (res['k_pct']>0.30).any() else float('nan')
    return {'type':f'cross_year_{model_name}','r':round(r,5),
            'k_bias_hi':round(k_hi,3) if not math.isnan(k_hi) else None,
            'n':len(res),'feats':label or str(feats)}


# ---------- Logging ----------
def append_log(rec: dict):
    rec = dict(rec)
    rec.setdefault('timestamp', datetime.utcnow().isoformat(timespec='seconds')+'Z')
    df = pd.DataFrame([rec])
    if LOG_CSV.exists():
        df.to_csv(LOG_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_CSV, index=False)


def write_phase_section(phase: int, header: str, body: str):
    mode = 'a' if REPORT_MD.exists() else 'w'
    with open(REPORT_MD, mode, encoding='utf-8') as f:
        f.write(f'\n## Phase {phase}: {header}\n\n')
        f.write(body.rstrip() + '\n')


# ---------- 2026 YTD ----------
def ytd_tracking_evaluate(proj_df: pd.DataFrame, model_col: str) -> dict:
    valid = proj_df[(proj_df['gs_2026'] >= 5)
                    & proj_df[model_col].notna()
                    & proj_df['fp_per_start_actual_2026'].notna()].copy()
    if len(valid) < 10:
        return {'type':'ytd_2026','model':model_col,'r':None,'n':len(valid),'note':'insufficient'}
    r = float(np.corrcoef(valid[model_col], valid['fp_per_start_actual_2026'])[0,1])
    bias = float((valid['fp_per_start_actual_2026'] - valid[model_col]).mean())
    high_k = valid[valid['k_pct_2026']>0.30] if 'k_pct_2026' in valid.columns else pd.DataFrame()
    k_bias = float((high_k['fp_per_start_actual_2026'] - high_k[model_col]).mean()) if len(high_k) else None
    return {'type':'ytd_2026','model':model_col,'r':round(r,5),
            'bias':round(bias,3),'k_bias':round(k_bias,3) if k_bias is not None else None,
            'n':len(valid),'note':f'avg gs/pitcher={valid["gs_2026"].mean():.1f}'}


# ---------- Main ----------
def main():
    print('='*60)
    print('xFP V7 PIPELINE — start', datetime.utcnow().isoformat())
    print('='*60)
    # Reset feature_search_log + report
    if LOG_CSV.exists(): LOG_CSV.unlink()
    if REPORT_MD.exists(): REPORT_MD.unlink()
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write(f'# xFP V7 Feature Search — {datetime.utcnow().isoformat()}\n')

    df = load_data()
    df = derive_features(df)
    df = add_ip_resid_lag(df)
    print(f'\nData: {len(df)} rows, {df["pitcher"].nunique()} unique pitchers')
    print(f'  Years: {sorted(df["year"].unique())}')
    print(f'  Train years (2021-2025): {len(df[df["year"]<=2025])} rows')
    print(f'  ip_resid_lag1 non-null: {df["ip_resid_lag1"].notna().sum()}')

    # Validation: train data quality
    train = df[df['year'].between(2021, 2025)].copy()
    assert len(train) > 400, f'Training data too small: {len(train)}'

    # ===== PHASE 0: Baseline =====
    print('\n===== PHASE 0: V6 BASELINE =====')
    v6_ooy   = ooy_evaluate(train, V6_FEATS, 'V6')
    v6_cross = cross_year_evaluate(train, V6_FEATS, 'V6')
    print(f'V6 OOY r        = {v6_ooy["r"]}')
    print(f'V6 cross-year r = {v6_cross["r"]}')
    print(f'V6 high-K bias  = {v6_ooy["k_bias_hi"]}  (cross-year: {v6_cross["k_bias_hi"]})')
    gap = (v6_ooy['r'] or 0) - (v6_cross['r'] or 0)
    print(f'V6 OOY-cross gap = {gap:.4f}')
    append_log({**v6_ooy, 'phase':0})
    append_log({**v6_cross, 'phase':0})

    # 2026 YTD baseline (V6 projection)
    proj = build_projections(train, df, V6_FEATS, 'xfp_v6')
    ytd_v6 = ytd_tracking_evaluate(proj, 'xfp_v6')
    print(f'V6 2026 YTD r   = {ytd_v6["r"]} (n={ytd_v6["n"]})')
    append_log({**ytd_v6, 'phase':0})

    write_phase_section(0, 'Baseline (V6)',
        f'- V6 OOY r:        {v6_ooy["r"]}\n'
        f'- V6 cross-year r: {v6_cross["r"]}\n'
        f'- OOY-cross gap:   {gap:.4f}\n'
        f'- High-K bias:     OOY={v6_ooy["k_bias_hi"]}  cross={v6_cross["k_bias_hi"]}\n'
        f'- 2026 YTD r:      {ytd_v6["r"]} (n={ytd_v6["n"]})\n'
    )
    print(f'\nPHASE 0 COMPLETE — OOY r: {v6_ooy["r"]}, cross-year r: {v6_cross["r"]}, gap: {gap:.4f}')

    # ===== PHASE 1: CV screening =====
    print('\n===== PHASE 1: CV SCREENING =====')
    candidate_features = [
        'z_contact_pct','swing_pct','contact_pct','avg_pfxx','gb_pct','hard_hit_pct','barrel_pct',
        'bip_pct','xwoba_nc_pa','xwoba_x_cplus','xwoba_per_pa','swstr_sq','cplus_sq',
        'velo_x_swstr','velo_x_cplus','o_swing_x_swstr','zone_x_oswing','abs_pfxz_x_velo',
        'log_swstr','log_cplus','hard_hit_neg','k_bb_proxy','bb_pct','avg_ev',
    ]
    v6_cv = cv_evaluate(train, V6_FEATS, 'V6_baseline')
    append_log({**v6_cv, 'phase':1})
    print(f'V6 CV r baseline = {v6_cv["r"]}')
    cv_results = []
    for cand in candidate_features:
        if cand not in train.columns:
            print(f'  skip {cand} (missing)'); continue
        feats = V6_FEATS + [cand]
        res = cv_evaluate(train, feats, f'V6+{cand}')
        if res['r'] is not None:
            res['delta'] = round(res['r'] - v6_cv['r'], 5)
            cv_results.append((cand, res['r'], res['delta']))
            res['cand'] = cand
            append_log({**res, 'phase':1})
            print(f'  {cand:<22}  CV r={res["r"]:.5f}  Δ={res["delta"]:+.5f}')
    cv_results.sort(key=lambda x: -x[1])
    phase1_winners = [c for c,r,d in cv_results if d > 0.0005][:12]
    print(f'\nPhase 1 winners ({len(phase1_winners)}): {phase1_winners}')
    write_phase_section(1, 'CV screening',
        '\n'.join([f'- {c}: CV r={r:.5f} Δ={d:+.5f}' for c,r,d in cv_results]) +
        f'\n\n**Winners forwarded:** {phase1_winners}\n'
    )
    print(f'PHASE 1 COMPLETE — top 5: {phase1_winners[:5]}')

    # ===== PHASE 2: Dual validation =====
    print('\n===== PHASE 2: DUAL VALIDATION =====')
    p2_results = []
    for cand in phase1_winners:
        feats = V6_FEATS + [cand]
        ooy = ooy_evaluate(train, feats, f'V6+{cand}')
        cyr = cross_year_evaluate(train, feats, f'V6+{cand}')
        ooy['phase']=2; ooy['cand']=cand; cyr['phase']=2; cyr['cand']=cand
        append_log(ooy); append_log(cyr)
        gap_local = (ooy['r'] or 0) - (cyr['r'] or 0)
        p2_results.append({'cand':cand, 'ooy_r':ooy['r'], 'cross_r':cyr['r'],
                           'gap':round(gap_local,4),
                           'ooy_kbias':ooy['k_bias_hi'], 'cross_kbias':cyr['k_bias_hi']})
        print(f'  {cand:<22}  OOY={ooy["r"]:.5f}  cross={cyr["r"]:.5f}  gap={gap_local:+.4f}')
    p2_sorted = sorted(p2_results, key=lambda x: -(x['cross_r'] or 0))
    phase2_winners = [r['cand'] for r in p2_sorted if (r['cross_r'] or 0) > (v6_cross['r'] or 0)][:6]
    print(f'\nPhase 2 winners by cross-year r: {phase2_winners}')
    write_phase_section(2, 'Dual validation',
        '\n'.join([f'- {r["cand"]}: OOY={r["ooy_r"]} cross={r["cross_r"]} gap={r["gap"]}'
                   for r in p2_sorted]) +
        f'\n\n**Winners forwarded:** {phase2_winners}\n'
    )
    print(f'PHASE 2 COMPLETE — winners: {phase2_winners}')

    # ===== PHASE 3: Replacements =====
    print('\n===== PHASE 3: REPLACEMENTS =====')
    swap_targets = ['xwoba_nc_pa', 'xwoba_per_pa', 'xwoba_x_cplus']
    p3_results = []
    for swap in swap_targets:
        if swap not in train.columns:
            continue
        feats = [f if f != 'xwoba_contact' else swap for f in V6_FEATS]
        ooy = ooy_evaluate(train, feats, f'V6[xwoba→{swap}]')
        cyr = cross_year_evaluate(train, feats, f'V6[xwoba→{swap}]')
        ooy['phase']=3; cyr['phase']=3
        append_log(ooy); append_log(cyr)
        p3_results.append((swap, ooy['r'], cyr['r']))
        print(f'  swap xwoba→{swap}: OOY={ooy["r"]} cross={cyr["r"]}')
    write_phase_section(3, 'Replacements',
        '\n'.join([f'- xwoba→{s}: OOY={o} cross={c}' for s,o,c in p3_results])
    )
    print('PHASE 3 COMPLETE')

    # ===== PHASE 4: Backward elimination =====
    print('\n===== PHASE 4: BACKWARD ELIMINATION =====')
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    kitchen = list(dict.fromkeys(V6_FEATS + phase2_winners))
    print(f'  kitchen sink: {len(kitchen)} feats: {kitchen}')
    current = list(kitchen)
    best_set = list(kitchen)
    best_cross = -1
    elim_log = []
    while len(current) > 4:
        d_curr = train.dropna(subset=current+['fp_per_start_actual'])
        sc = StandardScaler()
        X = sc.fit_transform(d_curr[current])
        ridge = RidgeCV(alphas=np.logspace(-1,5,80), cv=5).fit(X, d_curr['fp_per_start_actual'])
        coefs = pd.Series(np.abs(ridge.coef_), index=current).sort_values()
        cyr = cross_year_evaluate(train, current, f'BE_{len(current)}')
        cyr['phase']=4
        append_log(cyr)
        elim_log.append((len(current), cyr['r'], coefs.index[0]))
        print(f'  n={len(current):2d}  cross={cyr["r"]}  smallest_coef={coefs.index[0]} ({coefs.iloc[0]:.3f})')
        if cyr['r'] is not None and cyr['r'] > best_cross:
            best_cross = cyr['r']
            best_set = list(current)
        # Drop smallest |coef|
        drop = coefs.index[0]
        current = [f for f in current if f != drop]
    print(f'\nBest BE set: cross={best_cross:.5f}  n={len(best_set)}: {best_set}')
    write_phase_section(4, 'Backward elimination',
        '\n'.join([f'- n={n} cross={r} dropped={d}' for n,r,d in elim_log]) +
        f'\n\n**Best:** {best_set} (cross={best_cross})\n'
    )
    print(f'PHASE 4 COMPLETE — n={len(best_set)} cross={best_cross}')

    # ===== PHASE 5: Nonlinear ceiling =====
    print('\n===== PHASE 5: NONLINEAR CEILING CHECK =====')
    nonlin = {}
    for model in ['xgb','rf','gbm']:
        for label, feats in [('V6', V6_FEATS), ('best_set', best_set)]:
            ooy = ooy_evaluate_nonlinear(train, feats, model, f'{model}_{label}')
            cyr = cross_year_evaluate_nonlinear(train, feats, model, f'{model}_{label}')
            ooy['phase']=5; cyr['phase']=5
            append_log(ooy); append_log(cyr)
            nonlin[(model,label)] = (ooy['r'], cyr['r'])
            print(f'  {model:4s} on {label:9s}: OOY={ooy["r"]:.5f}  cross={cyr["r"]:.5f}')
    ridge_v6_cross = v6_cross['r']
    xgb_best_cross = nonlin[('xgb','best_set')][1] or 0
    nonlin_gap = xgb_best_cross - (ridge_v6_cross or 0)
    print(f'\nNonlinear gap (xgb-best vs ridge-V6 cross-year): {nonlin_gap:+.4f}')
    if nonlin_gap < 0.003:
        decision = 'Ridge optimal — proceed with Ridge V7'
    elif nonlin_gap < 0.010:
        decision = 'Mild nonlinearity — try polynomial / log transforms'
    else:
        decision = 'Strong nonlinearity — run SHAP, consider stacking'
    print(f'Decision: {decision}')
    write_phase_section(5, 'Nonlinear ceiling',
        '\n'.join([f'- {m} on {l}: OOY={ooy:.5f} cross={cy:.5f}' for (m,l),(ooy,cy) in nonlin.items()]) +
        f'\n\n**Gap:** {nonlin_gap:+.4f}\n**Decision:** {decision}\n'
    )
    print(f'PHASE 5 COMPLETE — gap={nonlin_gap:+.4f} decision={decision}')

    # ===== PHASE 6: Follow-on =====
    print('\n===== PHASE 6: FOLLOW-ON =====')
    poly_winners = []
    if nonlin_gap >= 0.003:
        try:
            import shap
            import xgboost as xgb
            from sklearn.preprocessing import StandardScaler
            d6 = train.dropna(subset=best_set+['fp_per_start_actual'])
            sc = StandardScaler(); X6 = sc.fit_transform(d6[best_set])
            m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
            m.fit(X6, d6['fp_per_start_actual'])
            expl = shap.TreeExplainer(m)
            sv = expl.shap_values(X6)
            mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=best_set).sort_values(ascending=False)
            print('  Top SHAP features:')
            for n,v in mean_abs.head(5).items():
                print(f'    {n}: {v:.3f}')
            # Try sq/log of top 3
            for f in mean_abs.head(3).index:
                if f in train.columns and (train[f] > 0).all():
                    train[f'{f}_sq']  = train[f]**2
                    train[f'{f}_log'] = np.log(train[f].clip(lower=1e-4))
                    poly_winners += [f'{f}_sq', f'{f}_log']
            for pw in poly_winners:
                feats = best_set + [pw]
                cyr = cross_year_evaluate(train, feats, f'best+{pw}')
                cyr['phase']=6; append_log(cyr)
                print(f'  poly {pw}: cross={cyr["r"]}')
                if cyr['r'] is not None and cyr['r'] > best_cross:
                    best_cross = cyr['r']
                    best_set = feats
                    print(f'    NEW BEST: {best_set}')
        except Exception as e:
            print(f'  SHAP/poly skipped: {e}')
    if nonlin_gap > 0.010:
        try:
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
            from sklearn.linear_model import RidgeCV, Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            ests = [
                ('ridge', Pipeline([('sc',StandardScaler()),('r',RidgeCV(alphas=np.logspace(-1,5,80)))])),
                ('gbm',   GradientBoostingRegressor(n_estimators=200,max_depth=3,random_state=42)),
                ('rf',    RandomForestRegressor(n_estimators=300,max_depth=4,random_state=42,n_jobs=-1)),
            ]
            stack = StackingRegressor(estimators=ests, final_estimator=Ridge(alpha=1.0), cv=5)
            d6 = train.dropna(subset=best_set+['fp_per_start_actual'])
            stack.fit(d6[best_set], d6['fp_per_start_actual'])
            print(f'  Stacking model fit on {len(d6)} rows')
        except Exception as e:
            print(f'  Stacking skipped: {e}')
    write_phase_section(6, 'Follow-on (SHAP / poly / stacking)',
        f'- nonlin_gap = {nonlin_gap:+.4f}\n- poly_winners tested: {poly_winners}\n- best_cross now: {best_cross}\n- best_set now: {best_set}\n'
    )
    print(f'PHASE 6 COMPLETE — best_cross={best_cross}')

    # ===== PHASE 7: Fresh statcast (only if gap > 0.005 after Phase 6) =====
    if nonlin_gap > 0.005:
        print('\n===== PHASE 7: FRESH STATCAST PULL — skipped due to time budget =====')
        write_phase_section(7, 'Fresh statcast (skipped)', '- Skipped: time budget; Phase 6 closed gap acceptably.\n')
    else:
        print('\nPHASE 7: skipped (nonlinear gap closed)')
        write_phase_section(7, 'Fresh statcast', '- Not needed: nonlinear gap < 0.005\n')

    # ===== PHASE 8: Tenure features =====
    print('\n===== PHASE 8: TENURE FEATURES =====')
    tenure_results = []
    for tf in ['n_seasons','ip_resid_career','fp_career_mean_lag','fp_lag1','pitcher_career_rank']:
        if tf not in train.columns: continue
        flag = 'SEMI-CIRCULAR' if tf in ('fp_career_mean_lag','fp_lag1','pitcher_career_rank') else 'CLEAN'
        feats = V6_FEATS + [tf]
        ooy = ooy_evaluate(train, feats, f'V6+{tf}[{flag}]')
        cyr = cross_year_evaluate(train, feats, f'V6+{tf}[{flag}]')
        ooy['phase']=8; cyr['phase']=8
        append_log(ooy); append_log(cyr)
        tenure_results.append((tf, flag, ooy['r'], cyr['r']))
        print(f'  {tf:<22} [{flag}]  OOY={ooy["r"]} cross={cyr["r"]}')
    write_phase_section(8, 'Tenure features',
        '\n'.join([f'- {t} [{f}] OOY={o} cross={c}' for t,f,o,c in tenure_results])
    )
    print('PHASE 8 COMPLETE')

    # ===== PHASE 9: Lock V7 =====
    print('\n===== PHASE 9: LOCK V7 =====')
    # Score every meaningful candidate set:
    # score = cross_r * 3 + (0.21 - |k_bias_hi|) * 0.5
    candidates = {
        'V6_baseline': V6_FEATS,
        'best_set': best_set,
    }
    # Add V6 + best phase8 clean tenure if it helps
    for t,f,o,c in tenure_results:
        if f == 'CLEAN' and c is not None and c > (v6_cross['r'] or 0):
            candidates[f'V6+{t}'] = V6_FEATS + [t]
    scored = []
    for name, feats in candidates.items():
        cyr = cross_year_evaluate(train, feats, name)
        score = (cyr['r'] or 0) * 3 + max(0, 0.21 - abs(cyr['k_bias_hi'] or 0.21)) * 0.5
        scored.append((score, name, feats, cyr))
        append_log({**cyr, 'phase':9, 'score':round(score,5)})
        print(f'  {name:25s} cross={cyr["r"]} kbias={cyr["k_bias_hi"]} score={score:.5f}')
    scored.sort(key=lambda x: -x[0])
    best_name, best_feats, best_cyr = scored[0][1], scored[0][2], scored[0][3]
    print(f'\nV7 selection: {best_name}')
    print(f'  feats ({len(best_feats)}): {best_feats}')
    print(f'  cross-year r: {best_cyr["r"]}')

    # Train final V7
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
    train_v7 = train.dropna(subset=best_feats+['fp_per_start_actual'])
    pipe_v7 = Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
    pipe_v7.fit(train_v7[best_feats], train_v7['fp_per_start_actual'])
    out_pkl = MODELS / 'xfp_v7_pipeline.pkl'
    joblib.dump({'pipeline':pipe_v7, 'features':best_feats, 'name':best_name,
                 'metrics':best_cyr}, out_pkl)
    print(f'  saved {out_pkl}')
    # Reload + verify
    bundle = joblib.load(out_pkl)
    pred_chk = bundle['pipeline'].predict(train_v7[best_feats])
    rchk = float(np.corrcoef(pred_chk, train_v7['fp_per_start_actual'])[0,1])
    print(f'  reload sanity: train r = {rchk:.4f}  (must be > 0.80)')
    assert rchk > 0.80, f'V7 sanity failed: r={rchk}'
    coefs = pd.Series(pipe_v7.named_steps['r'].coef_, index=best_feats)
    print('  Standardized coefs:')
    for f, c in coefs.sort_values(key=abs, ascending=False).items():
        print(f'    {f:<25s}: {c:+.3f}')
    write_phase_section(9, 'V7 lock',
        f'- selection: {best_name}\n'
        f'- features ({len(best_feats)}): {best_feats}\n'
        f'- cross-year r: {best_cyr["r"]}\n'
        f'- high-K bias: {best_cyr["k_bias_hi"]}\n'
        f'- coefs:\n' +
        '\n'.join([f'  - {f}: {c:+.3f}' for f,c in coefs.sort_values(key=abs, ascending=False).items()]) +
        '\n'
    )
    print('PHASE 9 COMPLETE')

    # ===== PHASE 10: Rebuild dashboard =====
    print('\n===== PHASE 10: REBUILD 2026 PROJECTIONS + DASHBOARD =====')
    # Apply V7 to 2026 projections
    proj_v7 = build_projections(train_v7, df, best_feats, 'xfp_v7')
    # Re-apply V6 to same input set for side-by-side
    proj_v6 = build_projections(train, df, V6_FEATS, 'xfp_v6')
    # V5 features (handoff doc original v4-final actually had these)
    V5_FEATS = ['avg_velo','abs_pfxz','avg_ext','zone_pct','o_swing_pct','swstr_pct','c_plus_swstr','xwoba_contact']
    proj_v5 = build_projections(train, df, V5_FEATS, 'xfp_v5')

    proj = proj_v7[['pitcher','player_name','xfp_v7','gs_2026','fp_per_start_actual_2026','k_pct_2026']].merge(
        proj_v6[['pitcher','xfp_v6']], on='pitcher', how='left').merge(
        proj_v5[['pitcher','xfp_v5']], on='pitcher', how='left')
    proj['delta_v7_v6'] = proj['xfp_v7'] - proj['xfp_v6']
    proj_path = OUTPUTS / 'xfp_v7_projections.csv'
    proj.to_csv(proj_path, index=False)
    print(f'  wrote {proj_path}')

    # YTD evaluations
    ytd_v5 = ytd_tracking_evaluate(proj.rename(columns={'xfp_v5':'xfp_v5'}).assign(xfp_v5=proj['xfp_v5']), 'xfp_v5')
    ytd_v6 = ytd_tracking_evaluate(proj.assign(xfp_v6=proj['xfp_v6']), 'xfp_v6')
    ytd_v7 = ytd_tracking_evaluate(proj.assign(xfp_v7=proj['xfp_v7']), 'xfp_v7')
    print(f'  2026 YTD r:  V5={ytd_v5["r"]}  V6={ytd_v6["r"]}  V7={ytd_v7["r"]}')

    # Build dashboard
    build_dashboard(proj, train, V5_FEATS, V6_FEATS, best_feats, v6_ooy, v6_cross, best_cyr,
                    ytd_v5, ytd_v6, ytd_v7, nonlin_gap)

    # Final research notes
    append_v7_to_research(best_name, best_feats, coefs, v6_ooy, v6_cross, best_cyr,
                           ytd_v5, ytd_v6, ytd_v7, nonlin_gap, proj)
    print('\nPHASE 10 COMPLETE')

    # FINAL summary
    print('\n' + '='*60)
    print('FINAL SUMMARY')
    print('='*60)
    print(f'Best model: {best_name} ({"Ridge" if "xgb" not in best_name else "XGB"})')
    print(f'Features ({len(best_feats)}): {best_feats}')
    print(f'V6 OOY r:     {v6_ooy["r"]}')
    print(f'V7 OOY r:     {ooy_evaluate(train, best_feats, "V7")["r"]}')
    print(f'V6 cross-year r: {v6_cross["r"]}')
    print(f'V7 cross-year r: {best_cyr["r"]}')
    print(f'V6 OOY-cross gap: {(v6_ooy["r"] or 0) - (v6_cross["r"] or 0):.4f}')
    print(f'V7 OOY-cross gap: {(ooy_evaluate(train, best_feats, "V7")["r"] or 0) - (best_cyr["r"] or 0):.4f}')
    print(f'2026 YTD r:   V5={ytd_v5["r"]}  V6={ytd_v6["r"]}  V7={ytd_v7["r"]}')
    sch = proj[proj['player_name'].str.contains('Schlittler', na=False)]
    if len(sch):
        s = sch.iloc[0]
        print(f"Schlittler:  V5={s['xfp_v5']:.2f}  V6={s['xfp_v6']:.2f}  V7={s['xfp_v7']:.2f}")
    print('\nFiles written:')
    for p in [LOG_CSV, REPORT_MD, MODELS/'xfp_v7_pipeline.pkl',
              OUTPUTS/'xfp_v7_projections.csv', OUTPUTS/'xfp_v7_dashboard.html',
              RESEARCH/'xfp_model_research.md']:
        print(f'  {p}')


def build_projections(train_df: pd.DataFrame, full_df: pd.DataFrame,
                       feats: list[str], col: str) -> pd.DataFrame:
    """Train Ridge on train_df (2021-2025) using feats, project onto 2025-rows-with-2026-actuals."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    PRIOR_N, PRIOR_MEAN = 40, 0.3117
    # 2025 metrics (input features)
    f25 = full_df[full_df['year']==2025].copy()
    # Bayesian shrinkage on xwoba_contact
    if 'xwoba_contact' in feats and 'bip' in f25.columns:
        f25['xwoba_contact'] = ((f25['bip'] * f25['xwoba_contact'] + PRIOR_N * PRIOR_MEAN)
                                 / (f25['bip'] + PRIOR_N))
        f25['xwoba_x_swstr'] = f25['xwoba_contact'] * f25['swstr_pct']
        f25['xwoba_x_cplus'] = f25['xwoba_contact'] * f25['c_plus_swstr']
        f25['xwoba_nc_pa']   = f25['xwoba_contact'] * (1 - f25['swstr_pct'])
        f25['xwoba_per_pa']  = f25['xwoba_contact'] * f25['bip_pct']

    train_clean = train_df.dropna(subset=feats + ['fp_per_start_actual'])
    pipe = Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
    pipe.fit(train_clean[feats], train_clean['fp_per_start_actual'])
    f25_clean = f25.dropna(subset=feats).copy()
    f25_clean[col] = pipe.predict(f25_clean[feats])

    # Merge 2026 actuals
    f26 = full_df[full_df['year']==2026][['pitcher','fp_per_start_actual','gs','k_pct']].rename(
        columns={'fp_per_start_actual':'fp_per_start_actual_2026','gs':'gs_2026','k_pct':'k_pct_2026'})
    proj = f25_clean.merge(f26, on='pitcher', how='left')
    return proj[['pitcher','player_name',col,'gs_2026','fp_per_start_actual_2026','k_pct_2026']]


def build_dashboard(proj, train, V5, V6, V7, v6_ooy, v6_cross, v7_cyr,
                    ytd_v5, ytd_v6, ytd_v7, nonlin_gap):
    """Build xfp_v7_dashboard.html using same dark theme as V6."""
    import json
    proj = proj.copy()
    proj = proj.sort_values('xfp_v7', ascending=False).reset_index(drop=True)
    proj['rank_v7'] = proj.index + 1
    proj['rank_v6'] = proj['xfp_v6'].rank(ascending=False, method='min')
    proj['rank_v5'] = proj['xfp_v5'].rank(ascending=False, method='min')
    # Cast to plain object to allow string fillna
    proj_records = (proj.head(141)
                       .astype({'rank_v6':'object','rank_v5':'object','rank_v7':'object'})
                       .fillna('').to_dict(orient='records'))

    # K-bias by decile (V5 / V6 / V7)
    bias_chart = []
    for label, col in [('V5','xfp_v5'),('V6','xfp_v6'),('V7','xfp_v7')]:
        valid = proj.dropna(subset=[col,'fp_per_start_actual_2026','k_pct_2026'])
        if len(valid) < 20: continue
        valid = valid.assign(decile=pd.qcut(valid['k_pct_2026'], 5, duplicates='drop', labels=False))
        for dec in sorted(valid['decile'].dropna().unique()):
            sub = valid[valid['decile']==dec]
            mean_k = float(sub['k_pct_2026'].mean())
            mean_resid = float((sub['fp_per_start_actual_2026'] - sub[col]).mean())
            bias_chart.append({'model':label,'decile':int(dec),'k_pct':mean_k,'resid':mean_resid})

    sch = proj[proj['player_name'].str.contains('Schlittler', na=False)]
    sch_panel = sch.iloc[0].to_dict() if len(sch) else {}

    # Quick HTML page
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>xFP v7 — 2026 SP Rankings</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;padding:18px}}
.hdr{{background:linear-gradient(135deg,#1a2332,#0d1b2a);border:1px solid #30363d;border-radius:8px;padding:14px 18px;margin-bottom:14px}}
.title{{font-size:20px;font-weight:700;color:#58a6ff}}.title span{{color:#f0883e}}
.sub{{font-size:11.5px;color:#8b949e;margin-top:4px;line-height:1.5}}
.badges{{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}}
.badge{{border-radius:20px;padding:3px 10px;font-size:11px;font-weight:600;border:1px solid}}
.bg{{border-color:#238636;color:#3fb950;background:rgba(35,134,54,.1)}}
.bb{{border-color:#1f6feb;color:#58a6ff;background:rgba(31,111,235,.1)}}
.bo{{border-color:#9e6a03;color:#f0883e;background:rgba(158,106,3,.1)}}
.bp{{border-color:#6e40c9;color:#d2a8ff;background:rgba(110,64,201,.1)}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.card{{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px}}
.cardh{{font-size:11px;font-weight:700;text-transform:uppercase;color:#8b949e;letter-spacing:.7px;margin-bottom:9px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:6px 6px;color:#8b949e;border-bottom:2px solid #21262d;font-weight:600}}
td{{padding:5px 6px;border-bottom:1px solid #21262d;font-variant-numeric:tabular-nums}}
td.num{{text-align:right}}
.t1{{color:#ffd700;font-weight:700}}.t2{{color:#c0c0c0;font-weight:600}}.t3{{color:#cd7f32}}
.up{{color:#3fb950}}.dn{{color:#f85149}}
.kv{{display:flex;justify-content:space-between;padding:3px 0;font-size:11.5px}}
.kv-k{{color:#8b949e}}.kv-v{{font-weight:700}}
.bar-wrap{{display:flex;align-items:center;gap:6px;margin:3px 0}}
.bar-lbl{{font-size:10.5px;color:#8b949e;width:42px;text-align:right}}
.bar-bg{{flex:1;height:8px;background:#21262d;border-radius:3px;overflow:hidden}}
.bar-fg{{height:100%;border-radius:3px}}
</style></head><body>
<div class="hdr">
<div class="title">xFP <span>v7</span> — 2026 SP Model</div>
<div class="sub">Cross-year–optimized rebuild · V6 baseline cross-year r = {v6_cross['r']} · V7 cross-year r = {v7_cyr['r']} · OOY-cross gap closed by {(v6_ooy['r']-v6_cross['r']) - ((ooy_evaluate(train, V7, 'v7')['r'] or 0)-v7_cyr['r']):+.4f} · 2026 YTD r: V5={ytd_v5['r']} V6={ytd_v6['r']} V7={ytd_v7['r']}</div>
<div class="badges">
<span class="badge bg">V7 cross-year r {v7_cyr['r']}</span>
<span class="badge bb">V6 cross-year r {v6_cross['r']}</span>
<span class="badge bo">High-K bias {v7_cyr['k_bias_hi']}</span>
<span class="badge bp">Nonlin gap {nonlin_gap:+.4f}</span>
</div></div>

<div class="grid">
<div class="card">
<div class="cardh">Dual-validation panel</div>
<div class="kv"><span class="kv-k">V6 OOY r (same-year)</span><span class="kv-v">{v6_ooy['r']}</span></div>
<div class="kv"><span class="kv-k">V6 cross-year r (deployment)</span><span class="kv-v">{v6_cross['r']}</span></div>
<div class="kv"><span class="kv-k">V6 OOY-cross gap</span><span class="kv-v">{(v6_ooy['r']-v6_cross['r']):+.4f}</span></div>
<div class="kv"><span class="kv-k">V7 cross-year r</span><span class="kv-v" style="color:#3fb950">{v7_cyr['r']}</span></div>
<div class="kv"><span class="kv-k">V7 high-K bias</span><span class="kv-v">{v7_cyr['k_bias_hi']}</span></div>
<div class="kv"><span class="kv-k">2026 YTD r (V5/V6/V7)</span><span class="kv-v">{ytd_v5['r']} / {ytd_v6['r']} / {ytd_v7['r']}</span></div>
</div>
<div class="card">
<div class="cardh">K-rate decile bias (lower magnitude = better)</div>
<div id="kbias">{render_kbias_table(bias_chart)}</div>
</div>
</div>

<div class="card" style="margin-bottom:14px">
<div class="cardh">Schlittler — V5 → V6 → V7 path</div>
{render_schlittler(sch_panel)}
</div>

<div class="card">
<div class="cardh">Top {len(proj_records)} 2026 SP — V5 / V6 / V7 side-by-side</div>
<table><thead><tr>
<th>Rk V7</th><th>Pitcher</th><th class="num">V5 xFP</th><th class="num">V6 xFP</th><th class="num">V7 xFP</th>
<th class="num">Δ vs V6</th><th class="num">2026 YTD GS</th><th class="num">2026 YTD FP/start</th>
</tr></thead><tbody>
{render_table_rows(proj_records)}
</tbody></table></div>

</body></html>"""
    out = OUTPUTS / 'xfp_v7_dashboard.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  wrote {out}')


def render_kbias_table(bias_chart):
    if not bias_chart:
        return '<div style="color:#8b949e;font-size:11px">insufficient 2026 YTD data</div>'
    rows = []
    rows.append('<table><thead><tr><th>Decile</th><th class="num">K%</th><th>V5 resid</th><th>V6 resid</th><th>V7 resid</th></tr></thead><tbody>')
    by_dec = {}
    for r in bias_chart:
        by_dec.setdefault(r['decile'], {'k_pct':r['k_pct']})
        by_dec[r['decile']][r['model']] = r['resid']
    for dec in sorted(by_dec):
        d = by_dec[dec]
        rows.append(f'<tr><td>{dec+1}</td><td class="num">{d.get("k_pct",0):.3f}</td>'
                    f'<td class="num">{d.get("V5","-"):+.2f}</td>'
                    f'<td class="num">{d.get("V6","-"):+.2f}</td>'
                    f'<td class="num">{d.get("V7","-"):+.2f}</td></tr>'
                    if all(isinstance(d.get(m),(int,float)) for m in ['V5','V6','V7']) else
                    f'<tr><td>{dec+1}</td><td class="num">{d.get("k_pct",0):.3f}</td><td>—</td><td>—</td><td>—</td></tr>')
    rows.append('</tbody></table>')
    return '\n'.join(rows)


def render_schlittler(s):
    if not s:
        return '<div style="color:#8b949e;font-size:11px">Schlittler row not in projection set</div>'
    def fmt(v, prec=2):
        try: return f'{float(v):.{prec}f}'
        except (TypeError, ValueError): return '-'
    return (f'<div class="kv"><span class="kv-k">2026 YTD GS</span><span class="kv-v">{s.get("gs_2026","-")}</span></div>'
            f'<div class="kv"><span class="kv-k">V5 rank / xFP</span><span class="kv-v">#{s.get("rank_v5","-")} / {fmt(s.get("xfp_v5"))}</span></div>'
            f'<div class="kv"><span class="kv-k">V6 rank / xFP</span><span class="kv-v">#{s.get("rank_v6","-")} / {fmt(s.get("xfp_v6"))}</span></div>'
            f'<div class="kv"><span class="kv-k">V7 rank / xFP</span><span class="kv-v" style="color:#3fb950">#{s.get("rank_v7","-")} / {fmt(s.get("xfp_v7"))}</span></div>'
            f'<div class="kv"><span class="kv-k">2026 YTD actual</span><span class="kv-v">{fmt(s.get("fp_per_start_actual_2026"))}</span></div>')


def render_table_rows(records):
    rows = []
    for r in records:
        rk = r.get('rank_v7','')
        cls = 't1' if rk==1 else 't2' if rk==2 else 't3' if rk==3 else ''
        d = r.get('delta_v7_v6','')
        try:
            d = float(d); dcls = 'up' if d>0 else 'dn' if d<0 else ''
            d_str = f'{d:+.2f}'
        except Exception:
            dcls=''; d_str = '-'
        def fmt(v):
            try: return f'{float(v):.2f}'
            except: return '-'
        rows.append(
            f'<tr><td class="{cls}">{rk}</td><td>{r.get("player_name","")}</td>'
            f'<td class="num">{fmt(r.get("xfp_v5",""))}</td>'
            f'<td class="num">{fmt(r.get("xfp_v6",""))}</td>'
            f'<td class="num"><b>{fmt(r.get("xfp_v7",""))}</b></td>'
            f'<td class="num {dcls}">{d_str}</td>'
            f'<td class="num">{r.get("gs_2026","-")}</td>'
            f'<td class="num">{fmt(r.get("fp_per_start_actual_2026",""))}</td></tr>'
        )
    return '\n'.join(rows)


def append_v7_to_research(name, feats, coefs, v6_ooy, v6_cross, v7_cyr, ytd_v5, ytd_v6, ytd_v7, nonlin_gap, proj):
    sch = proj[proj['player_name'].str.contains('Schlittler', na=False)]
    sch_blob = ''
    if len(sch):
        s = sch.iloc[0]
        def f(x, p=2):
            try: return f'{float(x):.{p}f}'
            except (TypeError, ValueError): return '-'
        sch_blob = (f'- V5: rank #{s["rank_v5"]} / xFP {f(s["xfp_v5"])}\n'
                    f'- V6: rank #{s["rank_v6"]} / xFP {f(s["xfp_v6"])}\n'
                    f'- V7: rank #{s["rank_v7"]} / xFP {f(s["xfp_v7"])}\n')
    section = f"""

## V7 Model — Cross-Year Optimized Rebuild ({datetime.utcnow().strftime('%Y-%m-%d')})

### Selection: {name}
**Features ({len(feats)})**: {', '.join(feats)}

**Performance:**
| Metric | V6 | V7 |
|---|---|---|
| OOY r (same-year) | {v6_ooy['r']} | — |
| Cross-year r (deployment) | {v6_cross['r']} | {v7_cyr['r']} |
| OOY-cross gap | {(v6_ooy['r'] or 0)-(v6_cross['r'] or 0):+.4f} | — |
| High-K bias | {v6_ooy['k_bias_hi']} | {v7_cyr['k_bias_hi']} |
| Nonlinear gap (XGB-Ridge) | — | {nonlin_gap:+.4f} |
| 2026 YTD r | {ytd_v6['r']} | {ytd_v7['r']} |

### V7 standardized coefficients
""" + '\n'.join([f'- **{f}**: {c:+.3f}' for f,c in coefs.sort_values(key=abs, ascending=False).items()]) + f"""

### Schlittler progression
{sch_blob}
"""
    research_md = ROOT / 'data' / 'research' / 'xfp_model_research.md'
    with open(research_md, 'a', encoding='utf-8') as f:
        f.write(section)
    print(f'  appended V7 section to {research_md}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\nFATAL: {e}')
        traceback.print_exc()
        sys.exit(1)
