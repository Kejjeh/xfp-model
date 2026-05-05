"""
xfp_v8_pipeline.py - V8 model: fix Phase 9 score bug + xwoba_per_pa base + k_pct_lag1.

Key changes from V7:
1. Scoring formula: cross_year_r * 3 - abs(k_bias_hi) * 0.5  (no max() floor)
2. Base feature set uses xwoba_per_pa (= xwoba_contact * bip_pct) instead of xwoba_contact.
   xwoba_per_pa naturally penalizes K pitchers' high contact quality across PAs that don't
   reach contact - addresses the high-K bias that V7 worsened.
3. Adds k_pct_lag1 and bb_pct_lag1 (both SEMI-CIRCULAR) as candidate features.
4. Adds pitch-type Statcast features when cached parquet has the raw data:
   FF_spin, breaking_spin, offspeed_spin, vaa_ff, velo_diff, pitch_entropy.
5. Optional two-tier model for k_pct_lag1 > 0.28 cohort (Phase 11D).
"""
from __future__ import annotations
import os, sys, json, time, math, warnings, traceback
from pathlib import Path
from datetime import datetime, timezone
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

LOG_CSV   = RESEARCH / 'feature_search_log.csv'
REPORT_MD = RESEARCH / 'feature_search_report.md'

V6_FEATS = [
    'avg_velo','abs_pfxz','avg_ext','zone_pct','o_swing_pct','swstr_pct',
    'c_plus_swstr','xwoba_contact','z_swing_pct','xwoba_x_swstr','ip_resid_lag1',
]
V7_FEATS = ['avg_velo','o_swing_pct','swstr_pct','c_plus_swstr','xwoba_contact','z_swing_pct']
V5_FEATS = ['avg_velo','abs_pfxz','avg_ext','zone_pct','o_swing_pct',
            'swstr_pct','c_plus_swstr','xwoba_contact']

# V8 base: V6 with xwoba_contact replaced by xwoba_per_pa
V8_BASE = ['avg_velo','abs_pfxz','avg_ext','zone_pct','o_swing_pct','swstr_pct',
           'c_plus_swstr','xwoba_per_pa','z_swing_pct','xwoba_x_swstr','ip_resid_lag1']


def score_fn(cross_r: float | None, k_bias_hi: float | None) -> float:
    """V8 composite score: rewards cross_year r, penalizes |k_bias_hi|."""
    if cross_r is None or k_bias_hi is None:
        return float('-inf')
    return cross_r * 3 - abs(k_bias_hi) * 0.5


# ---------- Reuse V7 core functions ----------
sys.path.insert(0, str(ROOT / 'scripts' / 'xfp'))
from xfp_v7_pipeline import (load_data, derive_features, add_ip_resid_lag,
                              ooy_evaluate, cross_year_evaluate, cv_evaluate,
                              ooy_evaluate_nonlinear, cross_year_evaluate_nonlinear,
                              build_projections, render_kbias_table)


def load_extended_data() -> pd.DataFrame:
    """Try to load 2015-2025 extended dataset; fall back to 2021-2025."""
    extended_path = CACHE / 'sp_multiyr_2015_2025.csv'
    if extended_path.exists():
        df = pd.read_csv(extended_path)
        print(f'Loaded extended {extended_path} ({len(df)} rows, years {sorted(df["year"].unique())})')
        return df
    print('Extended dataset not built yet; falling back to 2021-2025')
    return load_data()


def derive_v8_features(df: pd.DataFrame) -> pd.DataFrame:
    """V8-specific derived features. Builds on derive_features() output."""
    d = df.copy()
    # xwoba_per_pa already in derive_features
    if 'xwoba_per_pa' not in d.columns:
        d['xwoba_per_pa'] = d['xwoba_contact'] * d['bip_pct']
    # Lag features
    d = d.sort_values(['pitcher','year'])
    d['k_pct_lag1']  = d.groupby('pitcher')['k_pct'].shift(1)
    d['bb_pct_lag1'] = d.groupby('pitcher')['bb_pct'].shift(1)
    # xwoba_per_pa interaction with swstr (compounded high-K signal)
    d['xwoba_per_pa_x_swstr'] = d['xwoba_per_pa'] * d['swstr_pct']
    return d


# ---------- Pitch-type Statcast features (Phase 11B) ----------
PITCH_FAMILIES = {
    'FF': {'FF','SI','FC','FT'},   # fastballs
    'BR': {'SL','CU','KC','SV','ST'}, # breaking
    'CH': {'CH','FS','FO','SC'},   # offspeed/changeup
}

def derive_pitch_type_features(year: int) -> pd.DataFrame | None:
    """Extract per-(pitcher) pitch-type features from cached Statcast parquet."""
    cache_path = CACHE / f'statcast_{year}.parquet'
    if not cache_path.exists():
        return None
    df = pd.read_parquet(cache_path)
    if df.empty:
        return None
    # Map pitch_type to family
    df['family'] = pd.NA
    for fam, types in PITCH_FAMILIES.items():
        df.loc[df['pitch_type'].isin(types), 'family'] = fam
    df = df.dropna(subset=['family','pitcher'])

    # Spin rate by family (mean per pitcher × family)
    spin = df.groupby(['pitcher','family'])['release_spin_rate'].agg(['mean','count']).reset_index()
    spin = spin[spin['count'] >= 20]
    spin_pivot = spin.pivot(index='pitcher', columns='family', values='mean')
    spin_pivot.columns = [f'{c}_spin' for c in spin_pivot.columns]
    spin_pivot = spin_pivot.rename(columns={'FF_spin':'FF_spin','BR_spin':'breaking_spin','CH_spin':'offspeed_spin'})

    # Velocity by family
    velo = df.groupby(['pitcher','family'])['release_speed'].agg(['mean','count']).reset_index()
    velo = velo[velo['count'] >= 30]
    velo_pivot = velo.pivot(index='pitcher', columns='family', values='mean')
    velo_pivot.columns = [f'{c}_velo' for c in velo_pivot.columns]
    velo_pivot = velo_pivot.rename(columns={'FF_velo':'FF_velo','BR_velo':'breaking_velo','CH_velo':'offspeed_velo'})

    # Pitch mix entropy
    mix = df.groupby(['pitcher','pitch_type']).size().reset_index(name='n')
    mix['pct'] = mix.groupby('pitcher')['n'].transform(lambda x: x / x.sum())
    mix = mix[mix['pct'] >= 0.02]  # only count types used >2%
    ent = mix.groupby('pitcher')['pct'].apply(lambda p: -(p * np.log(p)).sum()).rename('pitch_entropy')

    # Vertical Approach Angle for FF family
    ff = df[df['family']=='FF'].copy()
    if {'vy0','vz0','ay','az','release_extension'}.issubset(ff.columns):
        for c in ['vy0','vz0','ay','az','release_extension']:
            ff[c] = pd.to_numeric(ff[c], errors='coerce')
        ff = ff.dropna(subset=['vy0','vz0','ay','az','release_extension'])
        # t to plate (from release point to plate, ~60.5 - extension feet)
        # Solve quadratic: y(t) = y0 + vy0*t + 0.5*ay*t^2 = -1.417 (back of plate)
        # Use simplified: t = (vy0 - sqrt(vy0^2 - 2*ay*(distance)))/ay  (negative roots; signs)
        # Approx: t ≈ 50 / abs(vy0)  (typical 0.4s flight)
        with np.errstate(invalid='ignore'):
            distance = 60.5 - ff['release_extension'] - 1.417  # to back of plate
            disc = ff['vy0']**2 - 2*ff['ay']*distance
            disc = np.where(disc < 0, np.nan, disc)
            t_plate = (-ff['vy0'] - np.sqrt(disc)) / ff['ay']
        ff['vz_plate'] = ff['vz0'] + ff['az'] * t_plate
        ff['vy_plate'] = ff['vy0'] + ff['ay'] * t_plate
        ff['vaa'] = np.degrees(np.arctan(ff['vz_plate'] / ff['vy_plate'].abs()))
        # VAA is typically -3 to -10 degrees (negative = downward); clip outliers
        ff = ff[ff['vaa'].between(-15, 0)]
        vaa_ff = ff.groupby('pitcher')['vaa'].mean().rename('vaa_ff')
    else:
        vaa_ff = pd.Series(dtype=float, name='vaa_ff')

    # Combine
    out = (spin_pivot
            .join(velo_pivot, how='outer')
            .join(ent, how='outer')
            .join(vaa_ff, how='outer'))
    out['velo_diff'] = out.get('FF_velo', np.nan) - out.get('offspeed_velo', np.nan)
    out['year'] = year
    out = out.reset_index()
    return out


def build_pitch_type_panel(years: list[int]) -> pd.DataFrame:
    cache_csv = CACHE / 'sp_statcast_features_2015_2025.csv'
    if cache_csv.exists():
        df = pd.read_csv(cache_csv)
        # only use if it covers requested years
        if set(years).issubset(set(df['year'].unique())):
            print(f'Loaded cached pitch-type features ({len(df)} rows)')
            return df
    frames = []
    for yr in years:
        f = derive_pitch_type_features(yr)
        if f is not None:
            print(f'  pitch-type features {yr}: {len(f)} pitchers')
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(cache_csv, index=False)
    print(f'Saved pitch-type features to {cache_csv}')
    return df


# ---------- Logging ----------
def append_log(rec: dict):
    rec = dict(rec)
    rec.setdefault('timestamp', datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z'))
    df = pd.DataFrame([rec])
    if LOG_CSV.exists():
        df.to_csv(LOG_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_CSV, index=False)


def write_phase_section(phase_label: str, body: str):
    mode = 'a' if REPORT_MD.exists() else 'w'
    with open(REPORT_MD, mode, encoding='utf-8') as f:
        f.write(f'\n## {phase_label}\n\n')
        f.write(body.rstrip() + '\n')


def evaluate_with_score(df: pd.DataFrame, feats: list[str], label: str, phase: str) -> dict:
    cyr = cross_year_evaluate(df, feats, label)
    score = score_fn(cyr['r'], cyr['k_bias_hi'])
    cyr['score'] = round(score, 5) if score != float('-inf') else None
    cyr['phase'] = phase
    cyr['label'] = label
    append_log(cyr)
    return cyr


# ---------- Main ----------
def main():
    print('='*60)
    print(f'xFP V8 PIPELINE - start {datetime.now(timezone.utc).isoformat()}')
    print('='*60)

    df = load_extended_data()
    df = derive_features(df)
    df = add_ip_resid_lag(df)
    df = derive_v8_features(df)

    # Try to merge pitch-type features
    train_years = sorted(df['year'].unique())
    train_years = [y for y in train_years if y <= 2025]
    pt = build_pitch_type_panel(train_years + [2026])
    if not pt.empty:
        df = df.merge(pt, on=['pitcher','year'], how='left')
        print(f'After pitch-type merge: {len(df)} rows. Coverage:')
        for c in ['FF_spin','breaking_spin','offspeed_spin','velo_diff','vaa_ff','pitch_entropy']:
            if c in df.columns:
                print(f'  {c}: {df[c].notna().sum()}/{len(df)} non-null')

    train = df[df['year'].between(min(train_years), 2025)].copy()
    print(f'\nTraining set: {len(train)} rows, years {sorted(train["year"].unique())}')
    print(f'Lag features non-null:')
    for c in ['ip_resid_lag1','k_pct_lag1','bb_pct_lag1']:
        if c in train.columns:
            print(f'  {c}: {train[c].notna().sum()}')

    # Validation
    assert len(train) > 400, f'Training data too small: {len(train)}'

    # ===== PHASE 11.0: Baselines =====
    print('\n===== PHASE 11.0: Baselines (V6/V7 with NEW scoring formula) =====')
    v6_eval = evaluate_with_score(train, V6_FEATS, 'V6_baseline', '11.0')
    v7_eval = evaluate_with_score(train, V7_FEATS, 'V7_baseline', '11.0')
    v6_per_pa = [f if f != 'xwoba_contact' else 'xwoba_per_pa' for f in V6_FEATS]
    v6_per_pa_eval = evaluate_with_score(train, v6_per_pa, 'V6[xwoba_per_pa]', '11.0')
    v8_base_eval = evaluate_with_score(train, V8_BASE, 'V8_BASE', '11.0')

    print(f'V6:                cross={v6_eval["r"]} k_bias_hi={v6_eval["k_bias_hi"]} score={v6_eval["score"]}')
    print(f'V7:                cross={v7_eval["r"]} k_bias_hi={v7_eval["k_bias_hi"]} score={v7_eval["score"]}')
    print(f'V6[xwoba_per_pa]:  cross={v6_per_pa_eval["r"]} k_bias_hi={v6_per_pa_eval["k_bias_hi"]} score={v6_per_pa_eval["score"]}')
    print(f'V8_BASE:           cross={v8_base_eval["r"]} k_bias_hi={v8_base_eval["k_bias_hi"]} score={v8_base_eval["score"]}')

    write_phase_section('Phase 11.0: Baselines (NEW scoring)',
        f'- V6:                cross={v6_eval["r"]} kbias={v6_eval["k_bias_hi"]} **score={v6_eval["score"]}**\n'
        f'- V7:                cross={v7_eval["r"]} kbias={v7_eval["k_bias_hi"]} **score={v7_eval["score"]}**\n'
        f'- V6[xwoba_per_pa]:  cross={v6_per_pa_eval["r"]} kbias={v6_per_pa_eval["k_bias_hi"]} **score={v6_per_pa_eval["score"]}**\n'
        f'- V8_BASE (V6 ints + xwoba_per_pa): cross={v8_base_eval["r"]} kbias={v8_base_eval["k_bias_hi"]} **score={v8_base_eval["score"]}**\n')

    # ===== PHASE 11C: k_pct_lag1 / bb_pct_lag1 tests =====
    print('\n===== PHASE 11C: k_pct_lag1 / bb_pct_lag1 SEMI-CIRCULAR features =====')
    sc_results = []
    test_combos = [
        ('V7+k_pct_lag1', V7_FEATS + ['k_pct_lag1']),
        ('V6+k_pct_lag1', V6_FEATS + ['k_pct_lag1']),
        ('V6[per_pa]+k_pct_lag1', v6_per_pa + ['k_pct_lag1']),
        ('V8_BASE+k_pct_lag1', V8_BASE + ['k_pct_lag1']),
        ('V7+bb_pct_lag1', V7_FEATS + ['bb_pct_lag1']),
        ('V8_BASE+bb_pct_lag1', V8_BASE + ['bb_pct_lag1']),
        ('V8_BASE+k_pct_lag1+bb_pct_lag1', V8_BASE + ['k_pct_lag1','bb_pct_lag1']),
        ('k_pct_lag1_alone', ['k_pct_lag1']),
    ]
    for label, feats in test_combos:
        if not all(f in train.columns for f in feats):
            print(f'  skip {label}: missing features')
            continue
        e = evaluate_with_score(train, feats, label + ' [SEMI-CIRCULAR]', '11C')
        print(f'  {label:<35s} cross={e["r"]} kbias={e["k_bias_hi"]} score={e["score"]}')
        sc_results.append((label, feats, e))

    # ===== PHASE 11B-test: pitch-type features =====
    print('\n===== PHASE 11B test: pitch-type Statcast features =====')
    pt_results = []
    pt_features = ['FF_spin','breaking_spin','offspeed_spin','vaa_ff','velo_diff','pitch_entropy']
    available_pt = [f for f in pt_features if f in train.columns and train[f].notna().sum() > 100]
    print(f'  Available pitch-type features: {available_pt}')
    for cand in available_pt:
        feats = V8_BASE + [cand]
        e = evaluate_with_score(train, feats, f'V8_BASE+{cand}', '11B')
        print(f'  V8_BASE+{cand:<18s} cross={e["r"]} kbias={e["k_bias_hi"]} score={e["score"]}')
        pt_results.append((cand, feats, e))

    # ===== PHASE 11E: Backward elimination from kitchen sink =====
    print('\n===== PHASE 11E: Backward elimination =====')
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Kitchen sink: V8_BASE + best k_pct_lag1 combo + best pitch-type features
    kitchen = list(V8_BASE)
    if 'k_pct_lag1' in train.columns:
        kitchen.append('k_pct_lag1')
    if 'bb_pct_lag1' in train.columns:
        kitchen.append('bb_pct_lag1')
    for cand in available_pt:
        kitchen.append(cand)
    kitchen = list(dict.fromkeys(kitchen))
    print(f'  kitchen sink ({len(kitchen)}): {kitchen}')

    current = list(kitchen)
    best_score, best_set, best_eval = -float('inf'), list(kitchen), None
    elim_log = []
    while len(current) >= 4:
        d_curr = train.dropna(subset=current+['fp_per_start_actual'])
        if len(d_curr) < 100:
            print(f'  n={len(current)}: too few rows ({len(d_curr)}); stopping')
            break
        sc = StandardScaler()
        X = sc.fit_transform(d_curr[current])
        ridge = RidgeCV(alphas=np.logspace(-1,5,80), cv=5).fit(X, d_curr['fp_per_start_actual'])
        coefs = pd.Series(np.abs(ridge.coef_), index=current).sort_values()
        e = evaluate_with_score(train, current, f'BE_{len(current)}', '11E')
        elim_log.append((len(current), e['r'], e['k_bias_hi'], e['score'], coefs.index[0]))
        print(f'  n={len(current):2d}  cross={e["r"]} kbias={e["k_bias_hi"]} score={e["score"]} drop={coefs.index[0]} ({coefs.iloc[0]:.3f})')
        if e['score'] is not None and e['score'] > best_score:
            best_score = e['score']; best_set = list(current); best_eval = e
        current = [f for f in current if f != coefs.index[0]]

    print(f'\nBest BE set: score={best_score} cross={best_eval["r"]} kbias={best_eval["k_bias_hi"]}')
    print(f'  features ({len(best_set)}): {best_set}')

    write_phase_section('Phase 11E: Backward elimination',
        '\n'.join([f'- n={n} cross={r} kbias={k} score={s} dropped={d}' for n,r,k,s,d in elim_log]) +
        f'\n\n**Best BE set:** {best_set} score={best_score}\n')

    # ===== PHASE 11.5: Final V8 selection =====
    print('\n===== PHASE 11.5: V8 SELECTION =====')
    # Compare candidates: V8_BASE, BE-best, V8_BASE+k_pct_lag1 (if it scored well)
    candidates = {
        'V6_baseline':   V6_FEATS,
        'V7_baseline':   V7_FEATS,
        'V8_BASE':       V8_BASE,
        'V6[per_pa]':    v6_per_pa,
        'BE_best':       best_set,
    }
    # Add top 2 sc_results
    for label, feats, e in sorted(sc_results, key=lambda x: -(x[2]['score'] or -1))[:3]:
        candidates[label + ' [SEMI-CIRCULAR]'] = feats

    leaderboard = []
    for name, feats in candidates.items():
        e = evaluate_with_score(train, feats, f'V8_lock:{name}', '11.5')
        leaderboard.append((name, feats, e))
        print(f'  {name:<35s} score={e["score"]} cross={e["r"]} kbias={e["k_bias_hi"]}')

    leaderboard.sort(key=lambda x: -(x[2]['score'] or -1))
    v8_name, v8_feats, v8_eval = leaderboard[0]
    print(f'\n>>> V8 SELECTION: {v8_name} score={v8_eval["score"]}')
    print(f'    feats ({len(v8_feats)}): {v8_feats}')

    # Train final V8
    train_v8 = train.dropna(subset=v8_feats + ['fp_per_start_actual'])
    pipe_v8 = Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
    pipe_v8.fit(train_v8[v8_feats], train_v8['fp_per_start_actual'])
    pipe_v8_path = MODELS / 'xfp_v8_pipeline.pkl'
    import joblib
    joblib.dump({'pipeline': pipe_v8, 'features': v8_feats, 'name': v8_name,
                 'metrics': v8_eval, 'all_candidates': leaderboard,
                 'score_formula': 'cross_year_r * 3 - abs(k_bias_hi) * 0.5'},
                pipe_v8_path)
    print(f'  saved {pipe_v8_path}')
    coefs = pd.Series(pipe_v8.named_steps['r'].coef_, index=v8_feats)
    print('  Standardized coefs:')
    for f, c in coefs.sort_values(key=abs, ascending=False).items():
        print(f'    {f:<25s}: {c:+.3f}')

    write_phase_section('Phase 11.5: V8 lock',
        f'- selection: {v8_name}\n'
        f'- features ({len(v8_feats)}): {v8_feats}\n'
        f'- score: {v8_eval["score"]}\n'
        f'- cross-year r: {v8_eval["r"]}\n'
        f'- k_bias_hi: {v8_eval["k_bias_hi"]}\n'
        f'- coefs:\n' + '\n'.join([f'  - {f}: {c:+.3f}' for f,c in coefs.sort_values(key=abs, ascending=False).items()]) + '\n')

    # ===== PHASE 11F: Projections + Dashboard =====
    print('\n===== PHASE 11F: PROJECTIONS + DASHBOARD =====')
    proj_v8 = build_projections(train, df, v8_feats, 'xfp_v8')
    proj_v7 = build_projections(train, df, V7_FEATS, 'xfp_v7')
    proj_v6 = build_projections(train, df, V6_FEATS, 'xfp_v6')
    proj_v5 = build_projections(train, df, V5_FEATS, 'xfp_v5')

    proj = (proj_v8[['pitcher','player_name','xfp_v8','gs_2026','fp_per_start_actual_2026','k_pct_2026']]
            .merge(proj_v7[['pitcher','xfp_v7']], on='pitcher', how='left')
            .merge(proj_v6[['pitcher','xfp_v6']], on='pitcher', how='left')
            .merge(proj_v5[['pitcher','xfp_v5']], on='pitcher', how='left'))
    proj['delta_v8_v7'] = proj['xfp_v8'] - proj['xfp_v7']
    proj['delta_v8_v6'] = proj['xfp_v8'] - proj['xfp_v6']
    proj_path = OUTPUTS / 'xfp_v8_projections.csv'
    proj.to_csv(proj_path, index=False)
    print(f'  wrote {proj_path}')

    # YTD evaluations
    def ytd_for(col):
        valid = proj[(proj['gs_2026'] >= 5) & proj[col].notna() & proj['fp_per_start_actual_2026'].notna()].copy()
        if len(valid) < 10:
            return {'r': None, 'n': len(valid)}
        r = float(np.corrcoef(valid[col], valid['fp_per_start_actual_2026'])[0,1])
        bias = float((valid['fp_per_start_actual_2026'] - valid[col]).mean())
        high_k = valid[valid['k_pct_2026']>0.30]
        k_bias = float((high_k['fp_per_start_actual_2026'] - high_k[col]).mean()) if len(high_k) else None
        return {'r': round(r,5), 'bias': round(bias,3),
                'k_bias': round(k_bias,3) if k_bias is not None else None,
                'n': len(valid)}

    ytd_v5 = ytd_for('xfp_v5'); ytd_v6 = ytd_for('xfp_v6')
    ytd_v7 = ytd_for('xfp_v7'); ytd_v8 = ytd_for('xfp_v8')
    print(f'  2026 YTD r: V5={ytd_v5["r"]} V6={ytd_v6["r"]} V7={ytd_v7["r"]} V8={ytd_v8["r"]}')

    # Schlittler check
    sch = proj[proj['player_name'].str.contains('Schlittler', na=False)]
    if len(sch):
        s = sch.iloc[0]
        print(f"  Schlittler: V5={s['xfp_v5']:.2f}  V6={s['xfp_v6']}  V7={s['xfp_v7']:.2f}  V8={s['xfp_v8']:.2f}")
    print(f'  Schlittler 2026 actual: {s.get("fp_per_start_actual_2026", "n/a") if len(sch) else "no row"}')

    build_v8_dashboard(proj, v6_eval, v7_eval, v8_eval, ytd_v5, ytd_v6, ytd_v7, ytd_v8,
                       v8_name, v8_feats, coefs)

    append_v8_to_research(v8_name, v8_feats, coefs, v6_eval, v7_eval, v8_eval,
                           ytd_v5, ytd_v6, ytd_v7, ytd_v8, sc_results, pt_results, available_pt,
                           proj, leaderboard)

    print('\n' + '='*60)
    print('FINAL V8 SUMMARY')
    print('='*60)
    print(f'V8 selection: {v8_name}')
    print(f'V8 features ({len(v8_feats)}): {v8_feats}')
    print()
    print(f'                  score    cross     k_bias_hi')
    print(f'V6 baseline:      {v6_eval["score"]}    {v6_eval["r"]}   {v6_eval["k_bias_hi"]}')
    print(f'V7 (prior):       {v7_eval["score"]}    {v7_eval["r"]}   {v7_eval["k_bias_hi"]}')
    print(f'V8 (this run):    {v8_eval["score"]}    {v8_eval["r"]}   {v8_eval["k_bias_hi"]}')
    print()
    print(f'2026 YTD r: V5={ytd_v5["r"]} V6={ytd_v6["r"]} V7={ytd_v7["r"]} V8={ytd_v8["r"]}')


def build_v8_dashboard(proj, v6_eval, v7_eval, v8_eval, ytd_v5, ytd_v6, ytd_v7, ytd_v8,
                        v8_name, v8_feats, coefs):
    proj_d = proj.copy().sort_values('xfp_v8', ascending=False).reset_index(drop=True)
    proj_d['rank_v8'] = proj_d.index + 1
    proj_d['rank_v7'] = proj_d['xfp_v7'].rank(ascending=False, method='min')
    proj_d['rank_v6'] = proj_d['xfp_v6'].rank(ascending=False, method='min')
    proj_d['rank_v5'] = proj_d['xfp_v5'].rank(ascending=False, method='min')
    rec = (proj_d.head(160)
              .astype({'rank_v8':'object','rank_v7':'object','rank_v6':'object','rank_v5':'object'})
              .where(lambda d: d.notna(), '')
              .to_dict(orient='records'))

    sch = proj_d[proj_d['player_name'].str.contains('Schlittler', na=False)]
    sch_panel = sch.iloc[0].to_dict() if len(sch) else {}

    # K-bias by decile
    bias_chart = []
    for label, col in [('V5','xfp_v5'),('V6','xfp_v6'),('V7','xfp_v7'),('V8','xfp_v8')]:
        valid = proj_d.dropna(subset=[col,'fp_per_start_actual_2026','k_pct_2026'])
        if len(valid) < 20: continue
        valid = valid.assign(decile=pd.qcut(valid['k_pct_2026'], 5, duplicates='drop', labels=False))
        for dec in sorted(valid['decile'].dropna().unique()):
            sub = valid[valid['decile']==dec]
            bias_chart.append({'model': label, 'decile': int(dec),
                                'k_pct': float(sub['k_pct_2026'].mean()),
                                'resid': float((sub['fp_per_start_actual_2026'] - sub[col]).mean())})

    def fmt(v, p=2):
        try: return f'{float(v):.{p}f}'
        except (TypeError, ValueError): return '-'

    # K-bias bar chart values for header
    k_biases = {'V5':None,'V6':v6_eval['k_bias_hi'],'V7':v7_eval['k_bias_hi'],'V8':v8_eval['k_bias_hi']}
    scores   = {'V5':None,'V6':v6_eval['score'],'V7':v7_eval['score'],'V8':v8_eval['score']}

    sub_text = (f"V8 = backward-elimination from V8_BASE (V6 with xwoba_per_pa) + k_pct_lag1 / pitch-type features. "
                f"NEW score formula: cross_year_r*3 - |k_bias_hi|*0.5. "
                f"V6 score {v6_eval['score']} -> V7 {v7_eval['score']} (regression) -> V8 {v8_eval['score']}. "
                f"Cross-year r {v6_eval['r']} -> {v7_eval['r']} -> {v8_eval['r']}. "
                f"K-bias {v6_eval['k_bias_hi']} -> {v7_eval['k_bias_hi']} -> {v8_eval['k_bias_hi']}.")

    feat_list_html = ', '.join(v8_feats)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>xFP v8 - 2026 SP Rankings</title>
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
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}}
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
.bar-row{{display:flex;align-items:center;gap:6px;margin:4px 0}}
.bar-lbl{{font-size:10.5px;color:#8b949e;width:32px}}
.bar-bg{{flex:1;height:14px;background:#21262d;border-radius:3px;overflow:hidden;position:relative}}
.bar-fg{{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#f85149,#f0883e)}}
.bar-val{{font-size:10.5px;color:#e6edf3;width:50px;text-align:right;font-variant-numeric:tabular-nums}}
.feat-pill{{display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:11px;padding:2px 9px;font-size:10.5px;color:#e6edf3;margin:2px}}
</style></head><body>
<div class="hdr">
<div class="title">xFP <span>v8</span> - 2026 SP Model</div>
<div class="sub">{sub_text}</div>
<div class="badges">
<span class="badge bg">V8 score {v8_eval['score']}</span>
<span class="badge bb">cross-year r {v8_eval['r']}</span>
<span class="badge bo">k_bias_hi {v8_eval['k_bias_hi']}</span>
<span class="badge bp">{len(v8_feats)} features</span>
</div></div>

<div class="grid3">

<div class="card">
<div class="cardh">Composite score V5 -&gt; V6 -&gt; V7 -&gt; V8</div>
<div class="kv"><span class="kv-k">V6 score</span><span class="kv-v">{fmt(v6_eval['score'],3)}</span></div>
<div class="kv"><span class="kv-k">V7 score</span><span class="kv-v" style="color:#f85149">{fmt(v7_eval['score'],3)}</span></div>
<div class="kv"><span class="kv-k">V8 score</span><span class="kv-v" style="color:#3fb950">{fmt(v8_eval['score'],3)}</span></div>
<div style="margin-top:8px;font-size:10.5px;color:#8b949e;line-height:1.4">Score = cross_year_r * 3 - |k_bias_hi| * 0.5. V7 lost ground because dropping ip_resid_lag1 / xwoba_x_swstr made k_bias worse with no offsetting cross-year gain. V8 base uses xwoba_per_pa which natively penalizes high-K pitchers' contact quality less than xwoba_contact does.</div>
</div>

<div class="card">
<div class="cardh">High-K bias (FP/start) — lower magnitude is better</div>
{render_bias_bars(k_biases)}
</div>

<div class="card">
<div class="cardh">2026 YTD r (n={ytd_v8['n'] or 0})</div>
<div class="kv"><span class="kv-k">V5 YTD r</span><span class="kv-v">{ytd_v5['r']}</span></div>
<div class="kv"><span class="kv-k">V6 YTD r</span><span class="kv-v">{ytd_v6['r']}</span></div>
<div class="kv"><span class="kv-k">V7 YTD r</span><span class="kv-v">{ytd_v7['r']}</span></div>
<div class="kv"><span class="kv-k">V8 YTD r</span><span class="kv-v" style="color:#3fb950">{ytd_v8['r']}</span></div>
<div class="kv"><span class="kv-k">V8 high-K YTD bias</span><span class="kv-v">{ytd_v8['k_bias']}</span></div>
</div>

</div>

<div class="grid">

<div class="card">
<div class="cardh">V8 feature set ({len(v8_feats)})</div>
<div>{''.join(f'<span class="feat-pill">{f}</span>' for f in v8_feats)}</div>
<div style="margin-top:11px"><span class="cardh" style="margin:0">Standardized coefs</span></div>
{''.join(f'<div class="kv"><span class="kv-k">{f}</span><span class="kv-v">{c:+.3f}</span></div>' for f,c in coefs.sort_values(key=abs, ascending=False).items())}
</div>

<div class="card">
<div class="cardh">Cameron Schlittler (V5 -&gt; V6 -&gt; V7 -&gt; V8)</div>
{render_schlittler_v8(sch_panel)}
</div>

</div>

<div class="card" style="margin-bottom:14px">
<div class="cardh">K-rate decile residual (mean 2026 actual minus xFP). Closer to 0 = less bias.</div>
{render_kbias_table_v8(bias_chart)}
</div>

<div class="card">
<div class="cardh">Top {len(rec)} 2026 SP — V5 / V6 / V7 / V8 side-by-side</div>
<table><thead><tr>
<th>Rk V8</th><th>Pitcher</th>
<th class="num">V5 xFP</th><th class="num">V6 xFP</th><th class="num">V7 xFP</th><th class="num">V8 xFP</th>
<th class="num">D(V8-V7)</th><th class="num">D(V8-V6)</th>
<th class="num">2026 GS</th><th class="num">2026 actual FP</th>
</tr></thead><tbody>
{render_table_v8(rec)}
</tbody></table></div>
</body></html>"""
    out = OUTPUTS / 'xfp_v8_dashboard.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  wrote {out}')


def render_bias_bars(k_biases):
    rows = []
    max_abs = max(abs(v) for v in k_biases.values() if v is not None) or 1.0
    for lbl, v in k_biases.items():
        if v is None:
            rows.append(f'<div class="bar-row"><span class="bar-lbl">{lbl}</span><span class="bar-bg"></span><span class="bar-val">-</span></div>')
        else:
            pct = min(99, abs(v) / max_abs * 100)
            rows.append(f'<div class="bar-row"><span class="bar-lbl">{lbl}</span><span class="bar-bg"><span class="bar-fg" style="width:{pct:.1f}%"></span></span><span class="bar-val">{v:+.3f}</span></div>')
    return '\n'.join(rows)


def render_kbias_table_v8(bias_chart):
    if not bias_chart:
        return '<div style="color:#8b949e;font-size:11px">insufficient 2026 YTD data</div>'
    by_dec = {}
    for r in bias_chart:
        by_dec.setdefault(r['decile'], {'k_pct': r['k_pct']})
        by_dec[r['decile']][r['model']] = r['resid']
    rows = ['<table><thead><tr><th>Decile</th><th class="num">K%</th><th class="num">V5 resid</th><th class="num">V6 resid</th><th class="num">V7 resid</th><th class="num">V8 resid</th></tr></thead><tbody>']
    def f(x):
        try: return f'{float(x):+.2f}'
        except (TypeError, ValueError): return '-'
    for dec in sorted(by_dec):
        d = by_dec[dec]
        rows.append(f'<tr><td>{dec+1}</td><td class="num">{d.get("k_pct",0):.3f}</td><td class="num">{f(d.get("V5"))}</td><td class="num">{f(d.get("V6"))}</td><td class="num">{f(d.get("V7"))}</td><td class="num"><b>{f(d.get("V8"))}</b></td></tr>')
    rows.append('</tbody></table>')
    return '\n'.join(rows)


def render_schlittler_v8(s):
    if not s:
        return '<div style="color:#8b949e;font-size:11px">Schlittler not in projection set</div>'
    def f(x, p=2):
        try: return f'{float(x):.{p}f}'
        except (TypeError, ValueError): return '-'
    return ('<div class="kv"><span class="kv-k">2026 YTD GS</span><span class="kv-v">{}</span></div>'
            '<div class="kv"><span class="kv-k">V5 rank / xFP</span><span class="kv-v">#{} / {}</span></div>'
            '<div class="kv"><span class="kv-k">V6 rank / xFP</span><span class="kv-v">#{} / {}</span></div>'
            '<div class="kv"><span class="kv-k">V7 rank / xFP</span><span class="kv-v">#{} / {}</span></div>'
            '<div class="kv"><span class="kv-k">V8 rank / xFP</span><span class="kv-v" style="color:#3fb950">#{} / {}</span></div>'
            '<div class="kv"><span class="kv-k">2026 YTD actual FP/start</span><span class="kv-v">{}</span></div>'
            ).format(s.get('gs_2026','-'),
                     s.get('rank_v5','-'), f(s.get('xfp_v5')),
                     s.get('rank_v6','-'), f(s.get('xfp_v6')),
                     s.get('rank_v7','-'), f(s.get('xfp_v7')),
                     s.get('rank_v8','-'), f(s.get('xfp_v8')),
                     f(s.get('fp_per_start_actual_2026')))


def render_table_v8(records):
    rows = []
    for r in records:
        rk = r.get('rank_v8','')
        cls = 't1' if rk == 1 else 't2' if rk == 2 else 't3' if rk == 3 else ''
        def fmt(v):
            try: return f'{float(v):.2f}'
            except (TypeError, ValueError): return '-'
        d_v7 = r.get('delta_v8_v7','')
        d_v6 = r.get('delta_v8_v6','')
        try:
            d_v7f = float(d_v7); cls7 = 'up' if d_v7f > 0 else 'dn' if d_v7f < 0 else ''
            d_v7s = f'{d_v7f:+.2f}'
        except (TypeError, ValueError):
            cls7 = ''; d_v7s = '-'
        try:
            d_v6f = float(d_v6); cls6 = 'up' if d_v6f > 0 else 'dn' if d_v6f < 0 else ''
            d_v6s = f'{d_v6f:+.2f}'
        except (TypeError, ValueError):
            cls6 = ''; d_v6s = '-'
        rows.append(
            f'<tr><td class="{cls}">{rk}</td><td>{r.get("player_name","")}</td>'
            f'<td class="num">{fmt(r.get("xfp_v5"))}</td>'
            f'<td class="num">{fmt(r.get("xfp_v6"))}</td>'
            f'<td class="num">{fmt(r.get("xfp_v7"))}</td>'
            f'<td class="num"><b>{fmt(r.get("xfp_v8"))}</b></td>'
            f'<td class="num {cls7}">{d_v7s}</td>'
            f'<td class="num {cls6}">{d_v6s}</td>'
            f'<td class="num">{r.get("gs_2026","-")}</td>'
            f'<td class="num">{fmt(r.get("fp_per_start_actual_2026"))}</td></tr>')
    return '\n'.join(rows)


def append_v8_to_research(name, feats, coefs, v6_eval, v7_eval, v8_eval,
                            ytd_v5, ytd_v6, ytd_v7, ytd_v8, sc_results, pt_results, available_pt,
                            proj, leaderboard):
    research_md = ROOT / 'data' / 'research' / 'xfp_model_research.md'

    sch = proj[proj['player_name'].str.contains('Schlittler', na=False)]
    sch_blob = ''
    if len(sch):
        s = sch.iloc[0]
        def f(x,p=2):
            try: return f'{float(x):.{p}f}'
            except (TypeError, ValueError): return '-'
        sch_blob = (f"- V5 xFP: {f(s.get('xfp_v5'))}\n"
                    f"- V6 xFP: {f(s.get('xfp_v6'))}\n"
                    f"- V7 xFP: {f(s.get('xfp_v7'))}\n"
                    f"- V8 xFP: {f(s.get('xfp_v8'))}\n"
                    f"- 2026 actual FP/start: {f(s.get('fp_per_start_actual_2026'))} (gs={s.get('gs_2026')})\n")

    sc_blob = '\n'.join([f'- {lbl} cross={e["r"]} kbias={e["k_bias_hi"]} score={e["score"]}'
                          for lbl, _, e in sc_results])
    pt_blob = '\n'.join([f'- V8_BASE+{c} cross={e["r"]} kbias={e["k_bias_hi"]} score={e["score"]}'
                          for c, _, e in pt_results]) if pt_results else '- (no pitch-type features available)'
    leader_blob = '\n'.join([f'- {n} score={e["score"]} cross={e["r"]} kbias={e["k_bias_hi"]}'
                              for n, _, e in leaderboard])

    section = f"""

## V8 Model - Phase 11 (Score-Fix + xwoba_per_pa Base) ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})

### Phase 9 Bug Fixed
The V7 selection used scoring formula `cross_year_r * 3 + max(0, 0.21 - abs(k_bias_hi)) * 0.5`.
With every variant having k_bias_hi >> 0.21, the second term floored at 0 and only cross_year_r drove
the selection. V7 dropped ip_resid_lag1 and xwoba_x_swstr (each helped k_bias) for a +0.002 gain in
cross_year_r at the cost of k_bias going from 1.014 to 1.183.

NEW formula (V8 onward): `score = cross_year_r * 3 - abs(k_bias_hi) * 0.5`. No max() floor.
Direct penalty for k_bias.

### V8 Selection: {name}

**Features ({len(feats)})**: {', '.join(feats)}

### Performance (under NEW scoring)
| Metric | V6 | V7 | V8 | V8-V7 |
|---|---|---|---|---|
| Cross-year r | {v6_eval['r']} | {v7_eval['r']} | {v8_eval['r']} | {(v8_eval['r'] or 0)-(v7_eval['r'] or 0):+.5f} |
| k_bias_hi | {v6_eval['k_bias_hi']} | {v7_eval['k_bias_hi']} | {v8_eval['k_bias_hi']} | {(v8_eval['k_bias_hi'] or 0)-(v7_eval['k_bias_hi'] or 0):+.3f} |
| Score | {v6_eval['score']} | {v7_eval['score']} | {v8_eval['score']} | {(v8_eval['score'] or 0)-(v7_eval['score'] or 0):+.4f} |
| 2026 YTD r | {ytd_v6['r']} | {ytd_v7['r']} | {ytd_v8['r']} | - |

### V8 Standardized Coefficients
""" + '\n'.join([f'- **{f}**: {c:+.3f}' for f,c in coefs.sort_values(key=abs, ascending=False).items()]) + f"""

### Phase 11C: k_pct_lag1 / bb_pct_lag1 (semi-circular)
{sc_blob}

### Phase 11B: pitch-type Statcast features tested ({len(available_pt)} available)
{pt_blob}

### Phase 11.5 V8 Selection Leaderboard
{leader_blob}

### Schlittler progression V5 -> V6 -> V7 -> V8
{sch_blob}

### Notes
- xwoba_per_pa = xwoba_contact * bip_pct. For a high-K pitcher (k_pct=0.32), bip_pct ~ 0.65, so
  xwoba_per_pa ~ 0.65 * xwoba_contact - it captures both contact quality AND low-contact-rate.
- k_pct_lag1 is semi-circular (K is in FP formula). Even so, prior-year K rate is a strong
  forward-looking signal that survives the cross-year test if YoY stability is high.
- ip_resid_lag1 was dropped by V7 BE because the V7 score formula didn't penalize k_bias loss.
  Under V8 scoring, ip_resid_lag1 is back in V6_BASE and may survive into V8 depending on BE.

### Files written
- `data/models/xfp_v8_pipeline.pkl`
- `data/outputs/xfp_v8_projections.csv`
- `data/outputs/xfp_v8_dashboard.html`
"""

    with open(research_md, 'a', encoding='utf-8') as f:
        f.write(section)
    print(f'  appended V8 section to {research_md}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\nFATAL: {e}')
        traceback.print_exc()
        sys.exit(1)
