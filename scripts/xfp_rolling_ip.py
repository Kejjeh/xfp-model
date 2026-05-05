"""
xfp_rolling_ip.py - Two-score projection (full xFP + stuff xFP) + rolling IP analysis.

Task 1: stuff_xfp = V9-stuff prediction + LEAGUE_AVG_IP * 3.3
        ip_premium = full_xfp - stuff_xfp = (projected_ip - LEAGUE_AVG_IP) * 3.3 implicitly

Task 2: Investigate which rolling stats predict ip_per_start. Build composite ip_trend score.

Task 3: Update V8.5 dashboard with new columns + panels + Schlittler card.
"""
from __future__ import annotations
import sys, joblib, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'xfp'))

from xfp_v7_pipeline import derive_features, add_ip_resid_lag
from xfp_v8_pipeline import V6_FEATS, V7_FEATS, V8_BASE, derive_v8_features, build_pitch_type_panel
from xfp_v8_5_pipeline import build_pfxz_panel
from xfp_v8_midseason import blend_pitcher

CACHE   = ROOT / 'data' / 'research' / 'xfp_cache'
OUTPUTS = ROOT / 'data' / 'outputs'
MODELS  = ROOT / 'data' / 'models'
RESEARCH= ROOT / 'data' / 'research'

V5_FEATS = ['avg_velo','abs_pfxz','avg_ext','zone_pct','o_swing_pct',
             'swstr_pct','c_plus_swstr','xwoba_contact']

# V9 BE-best stuff feature set (from previous run, reported as best n=16 cross_r=0.5407
# but stuff-only r_no_ip=0.59177 — the relevant one for projection here)
V9_STUFF_FEATS = ['abs_pfxz','avg_ext','zone_pct','swstr_pct','c_plus_swstr',
                   'xwoba_per_pa','z_swing_pct','xwoba_x_swstr','ip_resid_lag1','k_pct_lag1',
                   'offspeed_spin','vaa_ff','velo_diff','pitch_entropy','fb_pfxz','pfxz_spread']


# ============================================================
# TASK 1: full xFP + stuff xFP
# ============================================================
def build_v9_stuff_model(df: pd.DataFrame):
    """Train Ridge on (fp_per_start_actual - ip_per_start*3.3) using V9 stuff feats."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    train = df[df['year'].between(2015, 2025)].copy()
    train['fp_no_ip'] = train['fp_per_start_actual'] - train['ip_per_start'] * 3.3
    train_clean = train.dropna(subset=V9_STUFF_FEATS + ['fp_no_ip'])
    pipe = Pipeline([('sc', StandardScaler()),
                      ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
    pipe.fit(train_clean[V9_STUFF_FEATS], train_clean['fp_no_ip'])
    train_r = float(np.corrcoef(pipe.predict(train_clean[V9_STUFF_FEATS]), train_clean['fp_no_ip'])[0,1])
    print(f'  V9 stuff model trained on {len(train_clean)} rows; train r={train_r:.4f}')
    return pipe


def build_v85_blended_inputs(df: pd.DataFrame, feats_needed: list[str]) -> pd.DataFrame:
    """Build the V8.1 mid-season blended inputs for projection. Mirrors xfp_v8_midseason logic."""
    df_25 = df[df['year']==2025].set_index('pitcher')
    df_26 = df[df['year']==2026].set_index('pitcher')
    pitchers_union = sorted(set(df_25.index) | set(df_26.index))
    rows = []
    for p in pitchers_union:
        r25 = df_25.loc[p].to_dict() if p in df_25.index else None
        r26 = df_26.loc[p].to_dict() if p in df_26.index else None
        if r25: r25 = pd.Series({**r25, 'pitcher': p})
        if r26: r26 = pd.Series({**r26, 'pitcher': p})
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
    out = pd.DataFrame(rows)
    out['xwoba_per_pa']  = out['xwoba_contact'] * out['bip_pct']
    out['xwoba_x_swstr'] = out['xwoba_contact'] * out['swstr_pct']
    return out


def task1_stuff_xfp():
    print('=' * 60)
    print('TASK 1: full xFP + stuff xFP')
    print('=' * 60)
    df = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    df = derive_features(df)
    df = add_ip_resid_lag(df)
    df = derive_v8_features(df)
    pt = build_pitch_type_panel(sorted(df['year'].unique()))
    if not pt.empty:
        df = df.merge(pt, on=['pitcher','year'], how='left')
    pfxz = build_pfxz_panel(sorted(df['year'].unique()))
    if not pfxz.empty:
        df = df.merge(pfxz, on=['pitcher','year'], how='left')

    # League avg IP (2021-2025 SPs)
    train_recent = df[df['year'].isin([2021,2022,2023,2024,2025])]
    LEAGUE_AVG_IP = float(train_recent['ip_per_start'].mean())
    print(f'\nLeague avg IP/start (2021-2025): {LEAGUE_AVG_IP:.4f}')

    # Train V9 stuff model
    pipe_stuff = build_v9_stuff_model(df)
    joblib.dump({'pipeline': pipe_stuff, 'features': V9_STUFF_FEATS,
                  'name': 'V9_stuff_recovered', 'target': 'fp_no_ip'},
                 MODELS / 'xfp_v9_no_ip_pipeline.pkl')
    print(f'  saved {MODELS / "xfp_v9_no_ip_pipeline.pkl"}')

    # Build blended projection inputs
    proj_inputs = build_v85_blended_inputs(df, V9_STUFF_FEATS)

    # Impute league median for features that may be missing for fastball-only pitchers
    # (e.g., offspeed_spin and velo_diff are NaN for pitchers without changeup-family pitches)
    impute_feats = ['offspeed_spin','velo_diff','ip_resid_lag1','k_pct_lag1']
    train_pool = df[df['year'].between(2021,2025)]
    medians = {f: float(train_pool[f].median()) for f in impute_feats if f in proj_inputs.columns}
    print(f'  Imputing medians for missing-data pitchers: {medians}')
    for f, med in medians.items():
        proj_inputs[f] = proj_inputs[f].fillna(med)

    valid = proj_inputs.dropna(subset=V9_STUFF_FEATS).copy()
    valid['stuff_pred_no_ip'] = pipe_stuff.predict(valid[V9_STUFF_FEATS])
    valid['stuff_xfp'] = valid['stuff_pred_no_ip'] + LEAGUE_AVG_IP * 3.3
    print(f'  {len(valid)} pitchers got stuff_xfp predictions (out of {len(proj_inputs)} blended)')

    # Load existing V8.5 projections, merge
    v85 = pd.read_csv(OUTPUTS / 'xfp_v8_5_projections.csv')
    # Drop any stale columns from previous runs to avoid _x/_y duplication
    for stale in ['stuff_pred_no_ip','stuff_xfp','ip_premium','rolling_ip_last5','ip_trend_score','ip_trend']:
        if stale in v85.columns:
            v85 = v85.drop(columns=[stale])
    out = v85.merge(valid[['pitcher','stuff_pred_no_ip','stuff_xfp']], on='pitcher', how='left')
    out['ip_premium'] = out['xfp_v8_5'] - out['stuff_xfp']

    # Stats on ip_premium
    ip_p = out['ip_premium'].dropna()
    print(f'\nip_premium: mean={ip_p.mean():+.3f}  std={ip_p.std():.3f}  '
          f'min={ip_p.min():+.3f}  max={ip_p.max():+.3f}')
    print('\nTop 5 highest ip_premium (workhorses):')
    print(out.nlargest(5, 'ip_premium')[['player_name','xfp_v8_5','stuff_xfp','ip_premium']].to_string(index=False))
    print('\nTop 5 lowest ip_premium (short-leash):')
    print(out.nsmallest(5, 'ip_premium')[['player_name','xfp_v8_5','stuff_xfp','ip_premium']].to_string(index=False))

    # Spot check
    print('\nSPOT CHECK:')
    for n in ['Schlittler','Glasnow','Imanaga','Fried','Wheeler','Woodruff']:
        r = out[out['player_name'].fillna('').str.contains(n, na=False)]
        if len(r):
            s = r.iloc[0]
            print(f"  {n:<13s} full={s['xfp_v8_5']:.2f}  stuff={s['stuff_xfp']:.2f}  "
                  f"ip_premium={s['ip_premium']:+.2f}  actual={s['fp_per_start_actual_2026'] if pd.notna(s['fp_per_start_actual_2026']) else 'n/a'}")

    out.to_csv(OUTPUTS / 'xfp_v8_5_projections.csv', index=False)
    print(f'  wrote {OUTPUTS / "xfp_v8_5_projections.csv"}')
    return out, LEAGUE_AVG_IP


# ============================================================
# TASK 2: rolling IP investigation
# ============================================================
def build_per_start_panel():
    """Build per-start panel from cached statcast_2026 + 2025 for stability."""
    print('\n  Building per-start panel from cached statcast (2025 + 2026)')
    frames = []
    for yr in [2025, 2026]:
        cache_path = CACHE / f'statcast_{yr}.parquet'
        if not cache_path.exists(): continue
        df = pd.read_parquet(cache_path, columns=['pitcher','player_name','game_pk','game_date',
                                                    'inning','inning_topbot','events','description',
                                                    'bat_score','post_bat_score'])
        df = df.dropna(subset=['pitcher','game_pk'])
        # Identify starter (first pitch in inning 1 by side)
        df['inning'] = pd.to_numeric(df['inning'], errors='coerce')
        starts = (df[df['inning']==1].groupby(['game_pk','inning_topbot'])['pitcher']
                  .first().reset_index().rename(columns={'pitcher':'starter_id'}))
        df = df.merge(starts, on=['game_pk','inning_topbot'], how='left')
        sp = df[df['pitcher']==df['starter_id']].copy()
        # Per-PA-end events
        ev = sp['events'].fillna('')
        sp['is_pa_end'] = ev != ''
        sp['is_k']  = ev == 'strikeout'
        sp['is_bb'] = ev == 'walk'
        sp['is_h']  = ev.isin(['single','double','triple','home_run'])
        out_events = {'strikeout','field_out','grounded_into_double_play','sac_fly',
                       'sac_bunt','force_out','double_play','triple_play','fielders_choice_out',
                       'caught_stealing_2b','caught_stealing_3b','caught_stealing_home','other_out'}
        sp['outs_made'] = ev.isin(out_events).astype(int)
        sp.loc[ev.isin(['grounded_into_double_play','double_play']),'outs_made'] = 2
        sp.loc[ev=='triple_play','outs_made'] = 3
        sp['runs_on_play'] = (pd.to_numeric(sp['post_bat_score'], errors='coerce')
                                - pd.to_numeric(sp['bat_score'], errors='coerce')).clip(lower=0)
        sp.loc[~sp['is_pa_end'], 'runs_on_play'] = 0
        # Strike count from description
        desc = sp['description'].fillna('')
        sp['is_strike'] = desc.isin(['called_strike','swinging_strike','swinging_strike_blocked',
                                       'foul','foul_tip','hit_into_play','foul_bunt','missed_bunt'])

        g = sp.groupby(['pitcher','game_date','game_pk'])
        per_start = g.agg(
            pitches=('pitcher','size'),
            outs=('outs_made','sum'),
            tbf=('is_pa_end','sum'),
            k=('is_k','sum'),
            bb=('is_bb','sum'),
            h=('is_h','sum'),
            er_est=('runs_on_play','sum'),
            strikes=('is_strike','sum'),
            player_name=('player_name','first'),
        ).reset_index()
        per_start['ip_this_start'] = per_start['outs'] / 3.0
        per_start['pitches_per_ip'] = per_start['pitches'] / per_start['ip_this_start'].replace(0, np.nan)
        per_start['strike_pct']    = per_start['strikes'] / per_start['pitches'].replace(0, np.nan)
        per_start['bb_per_ip']     = per_start['bb'] / per_start['ip_this_start'].replace(0, np.nan)
        per_start['k_per_ip']      = per_start['k']  / per_start['ip_this_start'].replace(0, np.nan)
        per_start['er_per_ip']     = per_start['er_est'] / per_start['ip_this_start'].replace(0, np.nan)
        per_start['year'] = yr
        per_start = per_start[per_start['ip_this_start'] >= 1.0]  # filter incomplete starts
        frames.append(per_start)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(['pitcher','game_date']).reset_index(drop=True)
    print(f'  per-start panel: {len(panel)} starts ({panel["year"].value_counts().to_dict()})')
    return panel


def add_rolling_features(panel: pd.DataFrame, window: int = 5):
    """Add rolling-last-N-starts metrics per pitcher."""
    panel = panel.sort_values(['pitcher','game_date']).reset_index(drop=True)
    metrics = ['ip_this_start','pitches_per_ip','bb_per_ip','k_per_ip','er_per_ip','strike_pct']
    for m in metrics:
        panel[f'rolling_{m}_last{window}'] = (panel.groupby('pitcher')[m]
                                               .shift(1)
                                               .rolling(window, min_periods=2)
                                               .mean()
                                               .reset_index(0, drop=True))
    return panel


def task2_rolling_analysis(LEAGUE_AVG_IP):
    print('=' * 60)
    print('TASK 2: ROLLING IP INVESTIGATION')
    print('=' * 60)
    panel = build_per_start_panel()
    panel = add_rolling_features(panel, window=5)

    # PLV rolling 2026 — per-day rolling, take latest per pitcher
    plv = pd.read_parquet(OUTPUTS / 'plv_rolling_2026.parquet')
    plv = plv.sort_values(['pitcher','date']).groupby('pitcher').tail(1)[
        ['pitcher','plv','whiff_rate','called_strike_rate','swing_rate',
         'rolling_k_ip','rolling_bb_ip','rolling_h_ip','rolling_er_ip']
    ].rename(columns={'plv':'plv_current','whiff_rate':'whiff_current',
                       'rolling_k_ip':'plv_rolling_k_ip','rolling_bb_ip':'plv_rolling_bb_ip',
                       'rolling_er_ip':'plv_rolling_er_ip','rolling_h_ip':'plv_rolling_h_ip'})

    # ===== STEP 2: correlation analysis on 2026 starts =====
    p26 = panel[panel['year']==2026].copy()
    print(f'\n2026 starts: {len(p26)} rows; pitchers: {p26["pitcher"].nunique()}')

    metrics_to_test = ['rolling_ip_this_start_last5', 'rolling_pitches_per_ip_last5',
                        'rolling_bb_per_ip_last5', 'rolling_k_per_ip_last5',
                        'rolling_er_per_ip_last5', 'rolling_strike_pct_last5']
    rows = []
    for m in metrics_to_test:
        valid = p26.dropna(subset=[m, 'ip_this_start'])
        if len(valid) < 30: continue
        r = float(np.corrcoef(valid[m], valid['ip_this_start'])[0,1])
        rows.append({'metric': m, 'r_vs_ip_same_start': round(r, 4), 'n_obs': len(valid)})

    # Cross-year stability: 2025 metric → 2026 ip_per_start (per-pitcher means)
    p25 = panel[panel['year']==2025].copy()
    pa = p25.groupby('pitcher').agg(
        ip25=('ip_this_start','mean'),
        pitches_per_ip25=('pitches_per_ip','mean'),
        bb_per_ip25=('bb_per_ip','mean'),
        k_per_ip25=('k_per_ip','mean'),
        er_per_ip25=('er_per_ip','mean'),
        strike_pct25=('strike_pct','mean')).reset_index()
    pb = p26.groupby('pitcher').agg(ip26=('ip_this_start','mean')).reset_index()
    cross = pa.merge(pb, on='pitcher', how='inner')

    cross_results = []
    for col25, m25 in [('ip25','rolling_ip_this_start_last5'),
                         ('pitches_per_ip25','rolling_pitches_per_ip_last5'),
                         ('bb_per_ip25','rolling_bb_per_ip_last5'),
                         ('k_per_ip25','rolling_k_per_ip_last5'),
                         ('er_per_ip25','rolling_er_per_ip_last5'),
                         ('strike_pct25','rolling_strike_pct_last5')]:
        valid = cross.dropna(subset=[col25,'ip26'])
        if len(valid) < 30: continue
        r = float(np.corrcoef(valid[col25], valid['ip26'])[0,1])
        cross_results.append({'metric': m25, 'r_vs_ip_next_yr': round(r,4), 'n_cross_year': len(valid)})

    df_corr = pd.DataFrame(rows).merge(pd.DataFrame(cross_results), on='metric', how='outer')
    interp_map = {
        'rolling_ip_this_start_last5':  'prior-IP autocorrelation (manager trust signal)',
        'rolling_pitches_per_ip_last5': 'efficiency (lower = deeper outings)',
        'rolling_bb_per_ip_last5':      'command -> efficiency',
        'rolling_k_per_ip_last5':       'stuff -> manager confidence',
        'rolling_er_per_ip_last5':      'results -> manager trust',
        'rolling_strike_pct_last5':     'overall strike-throwing efficiency',
    }
    df_corr['interpretation'] = df_corr['metric'].map(interp_map)
    df_corr = df_corr.sort_values('r_vs_ip_same_start', ascending=False, na_position='last').reset_index(drop=True)
    df_corr['rank'] = df_corr.index + 1

    print('\nRanked rolling-stat predictors of ip_this_start:')
    print(df_corr.to_string(index=False))
    df_corr.to_csv(RESEARCH / 'rolling_ip_predictor_analysis.csv', index=False)
    print(f'  wrote {RESEARCH / "rolling_ip_predictor_analysis.csv"}')

    # ===== STEP 3: composite ip_trend_score =====
    # Use top 3 by same-start r (ignore sign - we'll standardize signs)
    valid_same = df_corr.dropna(subset=['r_vs_ip_same_start']).head(3)
    print(f'\nTop 3 metrics for composite: {valid_same["metric"].tolist()}')

    # Per-pitcher latest rolling values from panel (2026)
    latest_2026 = (p26.dropna(subset=valid_same['metric'].tolist())
                       .sort_values(['pitcher','game_date'])
                       .groupby('pitcher').tail(1))
    print(f'  pitchers with all 3 metrics: {len(latest_2026)}')

    if len(latest_2026) > 0:
        # Standardize each metric, then sign by correlation direction
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X = sc.fit_transform(latest_2026[valid_same['metric'].tolist()])
        signs = np.sign(valid_same['r_vs_ip_same_start'].values)
        z = X * signs
        latest_2026 = latest_2026.copy()
        latest_2026['ip_trend_score'] = z.mean(axis=1)
        mean_, std_ = latest_2026['ip_trend_score'].mean(), latest_2026['ip_trend_score'].std()
        latest_2026['ip_trend'] = 'NORMAL'
        latest_2026.loc[latest_2026['ip_trend_score'] > mean_ + 0.75*std_, 'ip_trend'] = 'HIGH'
        latest_2026.loc[latest_2026['ip_trend_score'] < mean_ - 0.75*std_, 'ip_trend'] = 'LOW'

        print('\nTop 10 HIGH ip_trend (going deeper):')
        print(latest_2026[latest_2026['ip_trend']=='HIGH']
                .nlargest(10, 'ip_trend_score')
                [['player_name','rolling_ip_this_start_last5','ip_trend_score','ip_trend']]
                .to_string(index=False))
        print('\nTop 10 LOW ip_trend (being pulled early):')
        print(latest_2026[latest_2026['ip_trend']=='LOW']
                .nsmallest(10, 'ip_trend_score')
                [['player_name','rolling_ip_this_start_last5','ip_trend_score','ip_trend']]
                .to_string(index=False))

    return df_corr, latest_2026


def task3_dashboard_update(proj, df_corr, latest_2026, LEAGUE_AVG_IP):
    print('=' * 60)
    print('TASK 3: DASHBOARD UPDATE')
    print('=' * 60)

    # Merge ip_trend into projections
    proj = proj.merge(latest_2026[['pitcher','rolling_ip_this_start_last5','ip_trend_score','ip_trend']]
                       .rename(columns={'rolling_ip_this_start_last5':'rolling_ip_last5'}),
                       on='pitcher', how='left')
    proj['ip_trend'] = proj['ip_trend'].fillna('NORMAL')
    proj.to_csv(OUTPUTS / 'xfp_v8_5_projections.csv', index=False)
    print(f'  re-wrote {OUTPUTS / "xfp_v8_5_projections.csv"} with ip_trend cols')

    # Build dashboard
    proj_d = proj.sort_values('xfp_v8_5', ascending=False, na_position='last').reset_index(drop=True)
    proj_d['rank_v8_5'] = proj_d.index + 1

    def fmt(x, p=2):
        try:
            f = float(x)
            if not np.isfinite(f): return '-'
            return f'{f:.{p}f}'
        except (TypeError, ValueError):
            return '-'

    def trend_icon(t):
        return {'HIGH':'<span style="color:#3fb950">↑ HIGH</span>',
                 'LOW':'<span style="color:#f85149">↓ LOW</span>',
                 'NORMAL':'<span style="color:#8b949e">→</span>'}.get(t, '-')

    # Main table rows
    main_rows = []
    for _, s in proj_d.head(80).iterrows():
        cls = 't1' if s['rank_v8_5']==1 else 't2' if s['rank_v8_5']==2 else 't3' if s['rank_v8_5']==3 else ''
        gold = ' style="color:#ffd700"' if 'Schlittler' in str(s['player_name']) else ''
        ip_p = s.get('ip_premium')
        try:
            ip_pf = float(ip_p) if pd.notna(ip_p) else None
            ip_pcls = 'up' if (ip_pf or 0) > 0 else 'dn' if (ip_pf or 0) < 0 else ''
            ip_pstr = f'{ip_pf:+.2f}' if ip_pf is not None else '-'
        except (TypeError, ValueError):
            ip_pcls=''; ip_pstr='-'
        # Decomposition mini-bar
        full = float(s['xfp_v8_5']) if pd.notna(s['xfp_v8_5']) else 0
        stuff = float(s['stuff_xfp']) if pd.notna(s.get('stuff_xfp')) else 0
        ip_amt = full - stuff
        scale = max(abs(full), 1)
        stuff_w = abs(stuff) / scale * 100
        ip_w = abs(ip_amt) / scale * 100
        ip_color = '#1f6feb' if ip_amt >= 0 else '#f85149'
        decomp = (f'<div style="display:flex;height:10px;border-radius:2px;overflow:hidden;background:#21262d">'
                   f'<div style="width:{stuff_w:.1f}%;background:#f0883e" title="stuff"></div>'
                   f'<div style="width:{ip_w:.1f}%;background:{ip_color}" title="ip premium"></div>'
                   f'</div>')
        main_rows.append(
            f'<tr><td class="{cls}">{int(s["rank_v8_5"])}</td><td{gold}>{s["player_name"]}</td>'
            f'<td class="num"><b>{fmt(s["xfp_v8_5"])}</b></td>'
            f'<td class="num">{fmt(s.get("stuff_xfp"))}</td>'
            f'<td class="num {ip_pcls}">{ip_pstr}</td>'
            f'<td>{decomp}</td>'
            f'<td>{trend_icon(s.get("ip_trend","NORMAL"))}</td>'
            f'<td class="num">{fmt(s.get("rolling_ip_last5"),1)}</td>'
            f'<td class="num">{fmt(s.get("gs_2026"),0)}</td>'
            f'<td class="num">{fmt(s.get("fp_per_start_actual_2026"))}</td></tr>')

    # Top 10 HIGH ip_trend panel
    high_rows = []
    for _, s in proj_d[proj_d['ip_trend']=='HIGH'].nlargest(10, 'ip_trend_score').iterrows():
        try: ip_pstr = f'{float(s["ip_premium"]):+.2f}' if pd.notna(s.get('ip_premium')) else '-'
        except (TypeError, ValueError): ip_pstr = '-'
        high_rows.append(
            f'<tr><td>{s["player_name"]}</td>'
            f'<td class="num">{fmt(s.get("rolling_ip_last5"),2)}</td>'
            f'<td class="num">{fmt(s.get("ip_trend_score"),3)}</td>'
            f'<td class="num"><b>{fmt(s["xfp_v8_5"])}</b></td>'
            f'<td class="num">{fmt(s.get("stuff_xfp"))}</td>'
            f'<td class="num up">{ip_pstr}</td></tr>')

    # Schlittler card
    sch = proj_d[proj_d['player_name'].fillna('').str.contains('Schlittler', na=False)]
    if len(sch):
        s = sch.iloc[0]
        try: ip_pstr = f'{float(s["ip_premium"]):+.2f}' if pd.notna(s.get('ip_premium')) else '-'
        except (TypeError, ValueError): ip_pstr = '-'
        sch_card = (f'<div class="kv"><span class="kv-k">Full xFP</span><span class="kv-v">{fmt(s["xfp_v8_5"])}</span></div>'
                    f'<div class="kv"><span class="kv-k">Stuff xFP</span><span class="kv-v">{fmt(s.get("stuff_xfp"))}</span></div>'
                    f'<div class="kv"><span class="kv-k">IP Premium</span><span class="kv-v">{ip_pstr}</span></div>'
                    f'<div class="kv"><span class="kv-k">IP Trend</span><span class="kv-v">{trend_icon(s.get("ip_trend","NORMAL"))}</span></div>'
                    f'<div class="kv"><span class="kv-k">2026 actual</span><span class="kv-v" style="color:#ffd700">{fmt(s.get("fp_per_start_actual_2026"))}</span></div>'
                    f'<div style="margin-top:6px;font-size:10.5px;color:#8b949e">If 2026 IP/start holds at ~6+, full xFP should converge toward actual.</div>')
    else:
        sch_card = '<div>Schlittler not in projection set</div>'

    # Top 3 predictor metrics for the guide
    top3 = df_corr.dropna(subset=['r_vs_ip_same_start']).head(3)['metric'].tolist()

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>xFP v8.5 + IP Decomposition</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;padding:18px}}
.hdr{{background:linear-gradient(135deg,#1a2332,#0d1b2a);border:1px solid #30363d;border-radius:8px;padding:14px 18px;margin-bottom:14px}}
.title{{font-size:20px;font-weight:700;color:#58a6ff}}.title span{{color:#f0883e}}
.sub{{font-size:11.5px;color:#8b949e;margin-top:4px;line-height:1.5}}
.card{{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px}}
.cardh{{font-size:11px;font-weight:700;text-transform:uppercase;color:#8b949e;letter-spacing:.7px;margin-bottom:9px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:6px 6px;color:#8b949e;border-bottom:2px solid #21262d;font-weight:600}}
td{{padding:5px 6px;border-bottom:1px solid #21262d;font-variant-numeric:tabular-nums}}
td.num{{text-align:right}}
.t1{{color:#ffd700;font-weight:700}}.t2{{color:#c0c0c0;font-weight:600}}.t3{{color:#cd7f32}}
.up{{color:#3fb950}}.dn{{color:#f85149}}
.kv{{display:flex;justify-content:space-between;padding:3px 0;font-size:11.5px}}
.kv-k{{color:#8b949e}}.kv-v{{font-weight:700}}
.guide{{background:rgba(31,111,235,.05);border:1px solid rgba(31,111,235,.3);border-radius:8px;padding:12px;font-size:11.5px;line-height:1.6}}
</style></head><body>
<div class="hdr">
<div class="title">xFP <span>v8.5</span> + IP Decomposition</div>
<div class="sub">V8.5 score 1.567 (cross-year r 0.600). Stuff xFP recovers V9 stuff-only signal (r_no_ip 0.591) and adds league-avg IP.
LEAGUE_AVG_IP={LEAGUE_AVG_IP:.3f} innings/start (2021-2025).
ip_premium = full_xfp - stuff_xfp = (projected IP - league avg) × 3.3 implicit.
ip_trend uses rolling-last-5-starts metrics from 2026 game logs.</div>
</div>

<div class="grid3">
<div class="card"><div class="cardh">Schlittler — full vs stuff vs trend</div>{sch_card}</div>

<div class="card guide">
<b>Interpretation guide</b><br>
<b>Full xFP</b>: total projected FP/start including IP depth (use for roster decisions)<br>
<b>Stuff xFP</b>: projected FP if everyone got league-avg IP ({LEAGUE_AVG_IP:.2f} IP) — pure skill score<br>
<b>IP Premium</b>: extra FP/start from going deeper than average (workhorse bonus)<br>
<b>IP Trend ↑</b>: rolling stats suggest pitcher is currently pitching deep into games<br>
<b>Top predictors (rolling-last-5)</b>: {', '.join(top3)}
</div>

<div class="card"><div class="cardh">Currently going deeper (top 10 HIGH ip_trend)</div>
<table><thead><tr><th>Pitcher</th><th class="num">L5 IP</th><th class="num">trend</th><th class="num">Full</th><th class="num">Stuff</th><th class="num">IP Prem</th></tr></thead>
<tbody>{''.join(high_rows) or '<tr><td colspan=6>No HIGH ip_trend pitchers yet</td></tr>'}</tbody></table>
<div style="margin-top:6px;font-size:10.5px;color:#8b949e">These pitchers are currently trending toward deeper outings — expect their IP-driven FP to hold or improve.</div>
</div>
</div>

<div class="card">
<div class="cardh">Top 80 SP — V8.5 with stuff/IP decomposition (Schlittler in gold)</div>
<table><thead><tr>
<th>Rk</th><th>Pitcher</th>
<th class="num">Full xFP</th><th class="num">Stuff xFP</th><th class="num">IP Prem</th>
<th>Decomp</th><th>Trend</th><th class="num">L5 IP</th>
<th class="num">2026 GS</th><th class="num">2026 actual</th>
</tr></thead><tbody>{''.join(main_rows)}</tbody></table></div>
</body></html>'''
    out_path = OUTPUTS / 'xfp_v8_5_dashboard.html'
    out_path.write_text(html, encoding='utf-8')
    print(f'  wrote {out_path}')


def append_research(LEAGUE_AVG_IP, df_corr, latest_2026):
    high = latest_2026[latest_2026['ip_trend']=='HIGH'].nlargest(10,'ip_trend_score')
    low  = latest_2026[latest_2026['ip_trend']=='LOW' ].nsmallest(10,'ip_trend_score')
    section = f"""

## Rolling IP Predictor Analysis — May 2026

LEAGUE_AVG_IP/start (2021-2025): {LEAGUE_AVG_IP:.4f}

### Two-score projection

- **Full xFP** = V8.5 prediction (V8.5 model trained on full FP target)
- **Stuff xFP** = V9-stuff prediction (target = `FP - IP×3.3`) + LEAGUE_AVG_IP × 3.3
- **IP Premium** = full_xfp − stuff_xfp = projected workhorse bonus

### Top rolling-stat predictors of ip_per_start

{df_corr.to_markdown(index=False)}

### Composite ip_trend methodology

- Take top 3 metrics by same-start r against ip_this_start
- Standardize each, sign by correlation direction
- ip_trend_score = mean of standardized values
- Label: HIGH if > mean + 0.75 std; LOW if < mean - 0.75 std

### Top 10 HIGH ip_trend (currently going deeper)

{high[['player_name','rolling_ip_this_start_last5','ip_trend_score']].to_markdown(index=False)}

### Top 10 LOW ip_trend (being pulled early)

{low[['player_name','rolling_ip_this_start_last5','ip_trend_score']].to_markdown(index=False)}

### Files
- `data/research/rolling_ip_predictor_analysis.csv`
- `data/outputs/xfp_v8_5_projections.csv` (added: stuff_xfp, ip_premium, rolling_ip_last5, ip_trend_score, ip_trend)
- `data/outputs/xfp_v8_5_dashboard.html` (added: stuff/IP decomposition columns + ip_trend panels)
- `data/models/xfp_v9_no_ip_pipeline.pkl` (V9 stuff model recovered for projection use)
"""
    research_md = RESEARCH / 'xfp_model_research.md'
    with open(research_md, 'a', encoding='utf-8') as f:
        f.write(section)
    print(f'  appended Rolling IP Analysis section to {research_md}')


def main():
    proj, LEAGUE_AVG_IP = task1_stuff_xfp()
    df_corr, latest_2026 = task2_rolling_analysis(LEAGUE_AVG_IP)
    proj = pd.read_csv(OUTPUTS / 'xfp_v8_5_projections.csv')
    task3_dashboard_update(proj, df_corr, latest_2026, LEAGUE_AVG_IP)
    append_research(LEAGUE_AVG_IP, df_corr, latest_2026)

    # Final summary
    print('\n' + '=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)
    proj = pd.read_csv(OUTPUTS / 'xfp_v8_5_projections.csv')
    print(f'LEAGUE_AVG_IP: {LEAGUE_AVG_IP:.4f}')
    top_metric = df_corr.dropna(subset=['r_vs_ip_same_start']).iloc[0]
    print(f'Top IP predictor: {top_metric["metric"]} (r={top_metric["r_vs_ip_same_start"]})')
    high5 = proj[proj['ip_trend']=='HIGH'].nlargest(5, 'ip_trend_score')
    print(f'Top 5 HIGH ip_trend: {", ".join(high5["player_name"].tolist())}')
    low5 = proj[proj['ip_trend']=='LOW'].nsmallest(5, 'ip_trend_score')
    print(f'Top 5 LOW ip_trend: {", ".join(low5["player_name"].tolist())}')
    for n in ['Schlittler','Fried','Glasnow']:
        r = proj[proj['player_name'].fillna('').str.contains(n, na=False)]
        if len(r):
            s = r.iloc[0]
            ip_p = f'{float(s["ip_premium"]):+.2f}' if pd.notna(s.get('ip_premium')) else '-'
            print(f"{n}: full={s['xfp_v8_5']:.2f} | stuff={s['stuff_xfp']:.2f} | "
                  f"ip_premium={ip_p} | trend={s.get('ip_trend','-')}")


if __name__ == '__main__':
    main()
