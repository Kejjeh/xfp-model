"""
v11_full_spotcheck.py - Full V11 spot-check (V8.5 + pitching_plus + fp_strike_pct).
Compare V11_full vs V11 (pitching_plus only) vs V8.5 on 2026 projections.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'xfp'))
from xfp_v7_pipeline import derive_features, add_ip_resid_lag
from xfp_v8_pipeline import derive_v8_features, build_pitch_type_panel
from xfp_v8_5_pipeline import build_pfxz_panel
from xfp_v8_midseason import blend_pitcher
from v11_spotcheck import build_blended_inputs

CACHE   = ROOT / 'data' / 'research' / 'xfp_cache'
OUTPUTS = ROOT / 'data' / 'outputs'

V85_FEATS = ['avg_velo','zone_pct','o_swing_pct','swstr_pct','c_plus_swstr','xwoba_per_pa',
              'z_swing_pct','xwoba_x_swstr','ip_resid_lag1','k_pct_lag1','pitch_entropy','bb_pfxz']
V11_PP    = V85_FEATS + ['pitching_plus']
V11_FULL  = V85_FEATS + ['pitching_plus','fp_strike_pct']


def load_data():
    df = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    df = derive_features(df); df = add_ip_resid_lag(df); df = derive_v8_features(df)
    pt = build_pitch_type_panel(sorted(df['year'].unique()))
    if not pt.empty: df = df.merge(pt, on=['pitcher','year'], how='left')
    pfxz = build_pfxz_panel(sorted(df['year'].unique()))
    if not pfxz.empty: df = df.merge(pfxz, on=['pitcher','year'], how='left')

    fg_rows = []
    for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
        path = OUTPUTS / f'fangraphs_pitchers_{yr}.csv'
        if not path.exists(): continue
        f = pd.read_csv(path).rename(columns={'mlb_id':'pitcher'})
        f['year'] = yr
        fg_rows.append(f[['pitcher','year','pitching_plus','stuff_plus']])
    if fg_rows:
        df = df.merge(pd.concat(fg_rows, ignore_index=True), on=['pitcher','year'], how='left')

    fp_strike_path = CACHE / 'fp_strike_2015_2026.csv'
    if fp_strike_path.exists():
        fp = pd.read_csv(fp_strike_path)
        df = df.merge(fp, on=['pitcher','year'], how='left')
    return df


def main():
    print('=' * 80)
    print('V11_FULL spot-check (V8.5 + pitching_plus + fp_strike_pct)')
    print('=' * 80)

    df = load_data()

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    # Train V8.5 on full 2015-2025
    train_v85 = df[df['year'].between(2015, 2025)].dropna(subset=V85_FEATS + ['fp_per_start_actual'])
    pipe_v85 = Pipeline([('sc', StandardScaler()),
                          ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=10))])
    pipe_v85.fit(train_v85[V85_FEATS], train_v85['fp_per_start_actual'])
    print(f'V8.5 trained on {len(train_v85)} rows')

    # Train V11_pp (pitching_plus only) on 2020-2025
    train_v11pp = df[df['year'].between(2020, 2025)].dropna(subset=V11_PP + ['fp_per_start_actual'])
    pipe_v11pp = Pipeline([('sc', StandardScaler()),
                            ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=10))])
    pipe_v11pp.fit(train_v11pp[V11_PP], train_v11pp['fp_per_start_actual'])
    print(f'V11_pp trained on {len(train_v11pp)} rows')

    # Train V11_full (pitching_plus + fp_strike_pct) on 2020-2025
    train_v11full = df[df['year'].between(2020, 2025)].dropna(subset=V11_FULL + ['fp_per_start_actual'])
    pipe_v11full = Pipeline([('sc', StandardScaler()),
                              ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=10))])
    pipe_v11full.fit(train_v11full[V11_FULL], train_v11full['fp_per_start_actual'])
    print(f'V11_full trained on {len(train_v11full)} rows')

    coefs = pd.Series(pipe_v11full.named_steps['r'].coef_, index=V11_FULL)
    print(f'\nV11_full standardized coefs:')
    for f, c in coefs.sort_values(key=abs, ascending=False).items():
        print(f'  {f:<22s}: {c:+.3f}')

    # 2026 projections via V8.1 blend
    blended = build_blended_inputs(df, V11_FULL)
    v85_valid = blended.dropna(subset=V85_FEATS).copy()
    v85_valid['xfp_v8_5'] = pipe_v85.predict(v85_valid[V85_FEATS])

    v11pp_valid = blended.dropna(subset=V11_PP).copy()
    v11pp_valid['xfp_v11_pp'] = pipe_v11pp.predict(v11pp_valid[V11_PP])

    v11full_valid = blended.dropna(subset=V11_FULL).copy()
    v11full_valid['xfp_v11_full'] = pipe_v11full.predict(v11full_valid[V11_FULL])

    out = (v85_valid[['pitcher','player_name','xfp_v8_5']]
            .merge(v11pp_valid[['pitcher','xfp_v11_pp']], on='pitcher', how='left')
            .merge(v11full_valid[['pitcher','xfp_v11_full']], on='pitcher', how='left'))
    # Fall back: if V11 NaN, use V8.5
    out['xfp_v11_pp']   = out['xfp_v11_pp'].fillna(out['xfp_v8_5'])
    out['xfp_v11_full'] = out['xfp_v11_full'].fillna(out['xfp_v8_5'])
    out['delta_pp_v85']    = out['xfp_v11_pp']   - out['xfp_v8_5']
    out['delta_full_v85']  = out['xfp_v11_full'] - out['xfp_v8_5']
    out['delta_full_pp']   = out['xfp_v11_full'] - out['xfp_v11_pp']

    v85_proj = pd.read_csv(OUTPUTS / 'xfp_v8_5_projections.csv')
    out = out.merge(v85_proj[['pitcher','k_pct_2026','gs_2026','fp_per_start_actual_2026']],
                     on='pitcher', how='left')

    eligible = out[out['k_pct_2026'].notna() & (out['gs_2026']>=5)].copy()
    top25 = eligible.nlargest(25, 'k_pct_2026').copy()

    print('\n' + '=' * 80)
    print('TOP 25 BY 2026 K_PCT — V8.5 vs V11_pp vs V11_full')
    print('=' * 80)
    print(f'{"Pitcher":<24s} {"k%":>5s} {"V8.5":>7s} {"V11pp":>7s} {"V11full":>8s} {"Δ_full":>7s} {"actual":>8s}')
    for _, r in top25.iterrows():
        print(f'{r["player_name"]:<24s} {r["k_pct_2026"]:.3f} {r["xfp_v8_5"]:>7.2f} {r["xfp_v11_pp"]:>7.2f} '
              f'{r["xfp_v11_full"]:>8.2f} {r["delta_full_v85"]:>+7.2f} {r["fp_per_start_actual_2026"]:>8.2f}')

    print(f'\nTop-25 Δ stats (V11_full vs V8.5):')
    d = top25['delta_full_v85']
    print(f'  mean: {d.mean():+.3f}  median: {d.median():+.3f}  max: {d.max():+.3f}  min: {d.min():+.3f}  std: {d.std():.3f}')
    print(f'  corr(Δ, k_pct_2026) = {float(np.corrcoef(eligible["delta_full_v85"], eligible["k_pct_2026"])[0,1]):+.4f}')

    # Top-25 Δ stats fp_strike_pct contribution (V11_full - V11_pp)
    print(f'\nfp_strike_pct contribution (V11_full - V11_pp) on top-25:')
    d_inc = top25['delta_full_pp']
    print(f'  mean: {d_inc.mean():+.3f}  median: {d_inc.median():+.3f}  max: {d_inc.max():+.3f}  min: {d_inc.min():+.3f}')

    # Marquee guys
    print('\nMarquee high-K pitchers:')
    for n in ['Skubal','Glasnow','Schlittler','Sale','Crochet','deGrom','Imanaga','Fried','Wheeler','Woodruff','Ragans']:
        r = out[out['player_name'].fillna('').str.contains(n, na=False)]
        if len(r):
            s = r.iloc[0]
            actual = s.get('fp_per_start_actual_2026')
            actual_str = f'{actual:.2f}' if pd.notna(actual) else 'n/a'
            print(f'  {n:<13s} V8.5={s["xfp_v8_5"]:>6.2f}  V11_full={s["xfp_v11_full"]:>6.2f}  '
                  f'Δ={s["delta_full_v85"]:>+5.2f}  actual={actual_str}')

    # YTD MAE/r comparison
    ytd = eligible.dropna(subset=['fp_per_start_actual_2026'])
    if len(ytd) >= 10:
        v85_mae   = (ytd['fp_per_start_actual_2026'] - ytd['xfp_v8_5']).abs().mean()
        v11pp_mae = (ytd['fp_per_start_actual_2026'] - ytd['xfp_v11_pp']).abs().mean()
        v11f_mae  = (ytd['fp_per_start_actual_2026'] - ytd['xfp_v11_full']).abs().mean()
        v85_r = float(np.corrcoef(ytd['xfp_v8_5'], ytd['fp_per_start_actual_2026'])[0,1])
        v11pp_r = float(np.corrcoef(ytd['xfp_v11_pp'], ytd['fp_per_start_actual_2026'])[0,1])
        v11f_r  = float(np.corrcoef(ytd['xfp_v11_full'], ytd['fp_per_start_actual_2026'])[0,1])
        print(f'\n2026 YTD MAE / r (n={len(ytd)}):')
        print(f'  V8.5      MAE={v85_mae:.3f}  r={v85_r:.4f}')
        print(f'  V11_pp    MAE={v11pp_mae:.3f}  r={v11pp_r:.4f}')
        print(f'  V11_full  MAE={v11f_mae:.3f}  r={v11f_r:.4f}')

    # Check |Δ_full vs V8.5| > 0.5 for marquee guys
    big_movers = []
    for n in ['Skubal','Glasnow','Schlittler','Sale','Crochet','deGrom']:
        r = out[out['player_name'].fillna('').str.contains(n, na=False)]
        if len(r) and abs(r.iloc[0]['delta_full_v85']) > 0.5:
            big_movers.append((n, r.iloc[0]['delta_full_v85']))
    if big_movers:
        print(f'\n*** MARQUEE PITCHERS WITH |Δ| > 0.5 ***')
        for n, d in big_movers:
            print(f'  {n}: {d:+.2f}')
    else:
        print(f'\n*** All marquee high-K pitchers Δ within ±0.5 FP/start ***')

if __name__ == '__main__':
    main()
