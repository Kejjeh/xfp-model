"""
v11_spotcheck.py - Compare V8.5 vs V11 (V8.5 + pitching_plus) on actual 2026 projections.
Specifically: top 25 pitchers by k_pct_2026 — what's the absolute FP/start lift?

V11 uses V8.5_FEATS + pitching_plus, trained on 2020-2025 (where pitching_plus is non-null).
Both models use V8.1 mid-season blend for 2026 input features.
"""
from __future__ import annotations
import sys, joblib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'xfp'))

from xfp_v7_pipeline import derive_features, add_ip_resid_lag
from xfp_v8_pipeline import derive_v8_features, build_pitch_type_panel
from xfp_v8_5_pipeline import build_pfxz_panel
from xfp_v8_midseason import blend_pitcher

CACHE   = ROOT / 'data' / 'research' / 'xfp_cache'
OUTPUTS = ROOT / 'data' / 'outputs'

V85_FEATS = ['avg_velo','zone_pct','o_swing_pct','swstr_pct','c_plus_swstr','xwoba_per_pa',
              'z_swing_pct','xwoba_x_swstr','ip_resid_lag1','k_pct_lag1','pitch_entropy','bb_pfxz']
V11_FEATS = V85_FEATS + ['pitching_plus']


def load_data():
    df = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    df = derive_features(df)
    df = add_ip_resid_lag(df)
    df = derive_v8_features(df)
    pt = build_pitch_type_panel(sorted(df['year'].unique()))
    if not pt.empty: df = df.merge(pt, on=['pitcher','year'], how='left')
    pfxz = build_pfxz_panel(sorted(df['year'].unique()))
    if not pfxz.empty: df = df.merge(pfxz, on=['pitcher','year'], how='left')

    fg_rows = []
    for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
        path = OUTPUTS / f'fangraphs_pitchers_{yr}.csv'
        if not path.exists(): continue
        f = pd.read_csv(path).rename(columns={'mlb_id': 'pitcher'})
        f['year'] = yr
        fg_rows.append(f[['pitcher','year','pitching_plus','stuff_plus','location_plus']])
    if fg_rows:
        fg = pd.concat(fg_rows, ignore_index=True)
        df = df.merge(fg, on=['pitcher','year'], how='left')
    return df


def build_blended_inputs(df, feats_needed):
    """V8.1 mid-season blend for 2026 projection."""
    df_25 = df[df['year']==2025].set_index('pitcher')
    df_26 = df[df['year']==2026].set_index('pitcher')
    rows = []
    for p in sorted(set(df_25.index) | set(df_26.index)):
        r25 = df_25.loc[p].to_dict() if p in df_25.index else None
        r26 = df_26.loc[p].to_dict() if p in df_26.index else None
        if r25: r25 = pd.Series({**r25, 'pitcher':p})
        if r26: r26 = pd.Series({**r26, 'pitcher':p})
        b = blend_pitcher(r25, r26)
        if b is None: continue
        for f in feats_needed:
            if f in b: continue
            if r26 is not None and pd.notna(r26.get(f)):
                b[f] = float(r26.get(f))
            elif r25 is not None and pd.notna(r25.get(f)):
                b[f] = float(r25.get(f))
            else:
                b[f] = np.nan
        rows.append(b)
    blended = pd.DataFrame(rows)
    blended['xwoba_per_pa']  = blended['xwoba_contact'] * blended['bip_pct']
    blended['xwoba_x_swstr'] = blended['xwoba_contact'] * blended['swstr_pct']
    return blended


def main():
    print('=' * 80)
    print('V11 vs V8.5 — projection spot check on top 25 by k_pct_2026')
    print('=' * 80)

    df = load_data()

    # Train V8.5 on full 2015-2025
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    train_v85 = df[df['year'].between(2015, 2025)].dropna(subset=V85_FEATS + ['fp_per_start_actual'])
    pipe_v85 = Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
    pipe_v85.fit(train_v85[V85_FEATS], train_v85['fp_per_start_actual'])
    print(f'V8.5 trained on {len(train_v85)} rows')

    # Train V11 on 2020-2025 (where pitching_plus exists)
    train_v11 = df[df['year'].between(2020, 2025)].dropna(subset=V11_FEATS + ['fp_per_start_actual'])
    pipe_v11 = Pipeline([('sc', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
    pipe_v11.fit(train_v11[V11_FEATS], train_v11['fp_per_start_actual'])
    print(f'V11 trained on {len(train_v11)} rows (2020-2025, pitching_plus non-null)')
    coefs_v11 = pd.Series(pipe_v11.named_steps['r'].coef_, index=V11_FEATS)
    print(f'V11 standardized coefs:')
    for f, c in coefs_v11.sort_values(key=abs, ascending=False).items():
        print(f'  {f:<22s}: {c:+.3f}')

    # Build blended 2026 inputs (covering both V8.5 and V11 features)
    blended = build_blended_inputs(df, V11_FEATS)

    # V8.5 projections
    v85_valid = blended.dropna(subset=V85_FEATS).copy()
    v85_valid['xfp_v8_5'] = pipe_v85.predict(v85_valid[V85_FEATS])
    print(f'V8.5 projections: {len(v85_valid)} pitchers')

    # V11 projections (need pitching_plus; for those without, fall back to V8.5)
    v11_valid = blended.dropna(subset=V11_FEATS).copy()
    v11_valid['xfp_v11'] = pipe_v11.predict(v11_valid[V11_FEATS])
    print(f'V11 projections: {len(v11_valid)} pitchers (with pitching_plus)')

    # Merge and compare
    out = v85_valid[['pitcher','player_name','xfp_v8_5']].merge(
        v11_valid[['pitcher','xfp_v11']], on='pitcher', how='left')
    # Fall back: if V11 NaN, use V8.5
    out['xfp_v11'] = out['xfp_v11'].fillna(out['xfp_v8_5'])
    out['delta'] = out['xfp_v11'] - out['xfp_v8_5']

    # Merge 2026 actual k_pct from prior projections file
    v85_proj = pd.read_csv(OUTPUTS / 'xfp_v8_5_projections.csv')
    out = out.merge(v85_proj[['pitcher','k_pct_2026','gs_2026','fp_per_start_actual_2026']],
                     on='pitcher', how='left')

    # Top 25 by k_pct_2026 (active SPs only, gs_2026 >= 5 for relevance)
    eligible = out[out['k_pct_2026'].notna() & (out['gs_2026'] >= 5)].copy()
    print(f'\nEligible (gs_2026 >= 5, k_pct_2026 non-null): {len(eligible)}')
    top25 = eligible.nlargest(25, 'k_pct_2026').copy()

    print('\n' + '=' * 80)
    print('TOP 25 BY 2026 K_PCT — V8.5 vs V11 projections')
    print('=' * 80)
    print(f'{"Pitcher":<24s} {"k%":>5s} {"V8.5":>7s} {"V11":>7s} {"Δ":>6s} {"actual":>8s}')
    for _, r in top25.iterrows():
        print(f'{r["player_name"]:<24s} {r["k_pct_2026"]:.3f} {r["xfp_v8_5"]:>7.2f} {r["xfp_v11"]:>7.2f} '
              f'{r["delta"]:>+6.2f} {r["fp_per_start_actual_2026"]:>8.2f}')

    # Summary stats on delta for top-25
    delta_stats = top25['delta']
    print(f'\nTop-25 high-K delta stats:')
    print(f'  mean delta:   {delta_stats.mean():+.3f}')
    print(f'  median delta: {delta_stats.median():+.3f}')
    print(f'  max lift:     {delta_stats.max():+.3f}')
    print(f'  min lift:     {delta_stats.min():+.3f}')
    print(f'  std:          {delta_stats.std():.3f}')

    # Compare to overall (all 2026 SPs)
    print(f'\nAll-pitcher delta stats (n={len(eligible)}):')
    print(f'  mean delta:   {eligible["delta"].mean():+.3f}')
    print(f'  median delta: {eligible["delta"].median():+.3f}')

    # Overall correlation between delta and k_pct (does V11 systematically lift high-K guys more?)
    delta_corr = float(np.corrcoef(eligible['delta'], eligible['k_pct_2026'])[0,1])
    print(f'\ncorr(delta, k_pct_2026) = {delta_corr:+.4f}')
    print('  (positive = V11 systematically lifts high-K pitchers more than low-K)')

    # YTD delta — is V11 closer to actual?
    ytd_eligible = eligible.dropna(subset=['fp_per_start_actual_2026'])
    if len(ytd_eligible) > 5:
        v85_err = (ytd_eligible['fp_per_start_actual_2026'] - ytd_eligible['xfp_v8_5']).abs().mean()
        v11_err = (ytd_eligible['fp_per_start_actual_2026'] - ytd_eligible['xfp_v11']).abs().mean()
        v85_r = float(np.corrcoef(ytd_eligible['xfp_v8_5'], ytd_eligible['fp_per_start_actual_2026'])[0,1])
        v11_r = float(np.corrcoef(ytd_eligible['xfp_v11'], ytd_eligible['fp_per_start_actual_2026'])[0,1])
        print(f'\n2026 YTD MAE (gs>=5, n={len(ytd_eligible)}):')
        print(f'  V8.5 MAE: {v85_err:.3f}  YTD r: {v85_r:.4f}')
        print(f'  V11 MAE:  {v11_err:.3f}  YTD r: {v11_r:.4f}')

        # Top-25 high-K specific
        high_k_ytd = top25.dropna(subset=['fp_per_start_actual_2026'])
        if len(high_k_ytd) > 5:
            v85_err_hk = (high_k_ytd['fp_per_start_actual_2026'] - high_k_ytd['xfp_v8_5']).abs().mean()
            v11_err_hk = (high_k_ytd['fp_per_start_actual_2026'] - high_k_ytd['xfp_v11']).abs().mean()
            print(f'\n2026 YTD MAE on top-25 high-K (n={len(high_k_ytd)}):')
            print(f'  V8.5 MAE: {v85_err_hk:.3f}')
            print(f'  V11 MAE:  {v11_err_hk:.3f}')


if __name__ == '__main__':
    main()
