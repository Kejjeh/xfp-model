"""
build_sp_multiyr.py - Reconstruct sp_multiyr.csv from Statcast pitch-by-pitch.

Pulls 2021-2026 Statcast data (regular season), aggregates to per-(pitcher, year)
SP-season level, and writes sp_multiyr.csv with the columns the xFP V6 pipeline
expects.

Output schema (per pitcher-season):
    pitcher (mlb_id), player_name, year,
    pitches, tbf, bip, in_zone, swing, contact, swstr,
    z_swing, z_zone, z_contact, o_swing_num, o_zone_tot,
    avg_velo, avg_ext, avg_pfxz, abs_pfxz, avg_pfxx,
    swing_pct, contact_pct, swstr_pct, c_plus_swstr,
    zone_pct, o_swing_pct, z_swing_pct, z_contact_pct,
    xwoba_contact, hard_hit_pct, barrel_pct, gb_pct, avg_ev,
    k, bb, hbp, h, hr, er_est, ip_outs, gs,
    k_pct, bb_pct, fp_per_start_actual, ip_per_start, k_per_start,
    bb_per_start, h_per_start, hr_per_start, hbp_per_start, er_per_start

Caches per-year parquet under data/research/xfp_cache/statcast_{year}.parquet
"""
from __future__ import annotations
import os, sys, time, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'data' / 'research' / 'xfp_cache'
CACHE.mkdir(parents=True, exist_ok=True)
TMP = Path('/tmp')
try:
    TMP.mkdir(parents=True, exist_ok=True)
except Exception:
    TMP = ROOT / 'data' / 'research' / 'xfp_cache'

# Year range for training + 2026 for YTD
import os
_ENV_YEARS = os.environ.get('XFP_YEARS', '')
if _ENV_YEARS:
    YEARS = [int(y) for y in _ENV_YEARS.split(',')]
else:
    YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
SEASON_DATES = {
    2015: ('2015-04-05', '2015-10-15'),
    2016: ('2016-04-03', '2016-10-15'),
    2017: ('2017-04-02', '2017-10-15'),
    2018: ('2018-03-29', '2018-10-15'),
    2019: ('2019-03-20', '2019-10-15'),
    2020: ('2020-07-23', '2020-09-30'),
    2021: ('2021-04-01', '2021-10-03'),
    2022: ('2022-04-07', '2022-10-05'),
    2023: ('2023-03-30', '2023-10-01'),
    2024: ('2024-03-28', '2024-09-29'),
    2025: ('2025-03-27', '2025-10-05'),
    2026: ('2026-03-26', '2026-05-04'),  # YTD only
}

def pull_year(year: int) -> pd.DataFrame:
    """Pull one season of pitch-by-pitch Statcast (regular season only)."""
    cache_path = CACHE / f'statcast_{year}.parquet'
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f'  [{year}] cached: {len(df):,} pitches')
        return df
    import pybaseball as pb
    pb.cache.enable()
    start, end = SEASON_DATES[year]
    print(f'  [{year}] pulling Statcast {start} -> {end} ...', flush=True)
    t0 = time.time()
    df = pb.statcast(start_dt=start, end_dt=end, verbose=False)
    df = df[df['game_type'] == 'R'].copy()
    df.to_parquet(cache_path, index=False)
    print(f'  [{year}] {len(df):,} pitches, {time.time()-t0:.0f}s', flush=True)
    return df


# Statcast description → categorize
SWING_DESC = {
    'swinging_strike','swinging_strike_blocked','foul','foul_tip','hit_into_play',
    'foul_bunt','missed_bunt'
}
SWSTR_DESC = {'swinging_strike','swinging_strike_blocked','foul_tip','missed_bunt'}
CONTACT_DESC = {'foul','hit_into_play','foul_bunt'}  # +foul_tip is technically swstr
CALLED_STRIKE_DESC = {'called_strike'}

def aggregate_pitcher_season(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compute per-(pitcher) aggregates for one season."""
    d = df.copy()
    d['year'] = year
    # Zone: Statcast 'zone' 1-9 = in zone; 11-14 = outside zone
    d['in_zone'] = (d['zone'] >= 1) & (d['zone'] <= 9)
    d['out_zone'] = (d['zone'] >= 11) & (d['zone'] <= 14)
    desc = d['description'].fillna('')
    d['is_swing'] = desc.isin(SWING_DESC)
    d['is_swstr'] = desc.isin(SWSTR_DESC)
    d['is_contact'] = d['is_swing'] & ~d['is_swstr']
    d['is_called_strike'] = desc == 'called_strike'
    d['z_swing']   = d['is_swing']   & d['in_zone']
    d['z_contact'] = d['is_contact'] & d['in_zone']
    d['o_swing']   = d['is_swing']   & d['out_zone']
    # Batted ball flags
    bb_type = d['bb_type'].fillna('')
    d['is_gb'] = bb_type == 'ground_ball'
    d['is_pop'] = bb_type == 'popup'
    # PA-ending events
    ev = d['events'].fillna('')
    d['is_pa_end'] = ev != ''
    d['is_k'] = ev == 'strikeout'
    d['is_bb'] = ev == 'walk'
    d['is_hbp'] = ev == 'hit_by_pitch'
    d['is_h'] = ev.isin(['single','double','triple','home_run'])
    d['is_hr'] = ev == 'home_run'
    d['is_bip'] = d['is_pa_end'] & ~d['is_k'] & ~d['is_bb'] & ~d['is_hbp']
    # launch_speed / launch_angle for hard_hit, barrels, ev, gb
    ls = pd.to_numeric(d.get('launch_speed'), errors='coerce')
    la = pd.to_numeric(d.get('launch_angle'), errors='coerce')
    d['hard_hit'] = (ls >= 95) & d['is_bip']
    # Statcast-style barrel approximation: EV >= 98 and LA in 26-30 (simplified)
    # Use Statcast's launch_speed_angle column if present (1=weak, 6=barrel)
    if 'launch_speed_angle' in d.columns:
        d['barrel'] = (pd.to_numeric(d['launch_speed_angle'], errors='coerce') == 6) & d['is_bip']
    else:
        d['barrel'] = (ls >= 98) & la.between(26, 30) & d['is_bip']
    # xwOBA — TWO variants:
    # xwoba_bip:    mean over BIP only (raw contact quality on contact)
    # xwoba_pa:     per-PA xwoba using woba_value col (K=0, BB=0.7, hits use estimated_woba)
    #               This is the V6-research-doc "xwoba_contact" (~0.31 league mean).
    xwoba_con = pd.to_numeric(d.get('estimated_woba_using_speedangle'), errors='coerce')
    d['xwoba_con_val'] = xwoba_con.where(d['is_bip'])
    woba_v = pd.to_numeric(d.get('woba_value'), errors='coerce')
    woba_d = pd.to_numeric(d.get('woba_denom'), errors='coerce')
    # Replace woba value of BIP with estimated_woba_using_speedangle for xwOBA approach
    d['woba_v_pa'] = woba_v
    d.loc[d['is_bip'] & xwoba_con.notna(), 'woba_v_pa'] = xwoba_con[d['is_bip'] & xwoba_con.notna()]
    d['woba_d_pa'] = woba_d

    g = d.groupby('pitcher')
    agg = g.agg(
        pitches=('pitcher','size'),
        tbf=('is_pa_end','sum'),
        bip=('is_bip','sum'),
        in_zone=('in_zone','sum'),
        out_zone=('out_zone','sum'),
        swing=('is_swing','sum'),
        contact=('is_contact','sum'),
        swstr=('is_swstr','sum'),
        called_strike=('is_called_strike','sum'),
        z_swing=('z_swing','sum'),
        z_contact=('z_contact','sum'),
        o_swing_num=('o_swing','sum'),
        avg_velo=('release_speed','mean'),
        avg_ext=('release_extension','mean'),
        avg_pfxz=('pfx_z','mean'),
        avg_pfxx=('pfx_x','mean'),
        avg_ev=('launch_speed','mean'),
        hard_hit_n=('hard_hit','sum'),
        barrel_n=('barrel','sum'),
        gb_n=('is_gb','sum'),
        k=('is_k','sum'),
        bb=('is_bb','sum'),
        hbp=('is_hbp','sum'),
        h=('is_h','sum'),
        hr=('is_hr','sum'),
        xwoba_bip=('xwoba_con_val','mean'),
        woba_v_sum=('woba_v_pa','sum'),
        woba_d_sum=('woba_d_pa','sum'),
    ).reset_index()
    # xwoba_contact (per-PA, league mean ~0.31): research-doc definition.
    agg['xwoba_contact'] = agg['woba_v_sum'] / agg['woba_d_sum'].replace(0, np.nan)
    # Player names from this dataset
    name_map = d.dropna(subset=['player_name']).groupby('pitcher')['player_name'].first()
    agg['player_name'] = agg['pitcher'].map(name_map)
    agg['year'] = year
    # Derived rate columns
    agg['z_zone']      = agg['in_zone']  # alias
    agg['o_zone_tot']  = agg['out_zone']
    agg['swing_pct']    = agg['swing'] / agg['pitches']
    agg['contact_pct']  = agg['contact'] / agg['swing'].replace(0, np.nan)
    agg['swstr_pct']    = agg['swstr'] / agg['pitches']
    agg['c_plus_swstr'] = (agg['called_strike'] + agg['swstr']) / agg['pitches']
    agg['zone_pct']     = agg['in_zone'] / agg['pitches']
    agg['o_swing_pct']  = agg['o_swing_num'] / agg['o_zone_tot'].replace(0, np.nan)
    agg['z_swing_pct']  = agg['z_swing'] / agg['in_zone'].replace(0, np.nan)
    agg['z_contact_pct']= agg['z_contact'] / agg['z_swing'].replace(0, np.nan)
    agg['hard_hit_pct'] = agg['hard_hit_n'] / agg['bip'].replace(0, np.nan)
    agg['barrel_pct']   = agg['barrel_n'] / agg['bip'].replace(0, np.nan)
    agg['gb_pct']       = agg['gb_n'] / agg['bip'].replace(0, np.nan)
    agg['k_pct']        = agg['k'] / agg['tbf'].replace(0, np.nan)
    agg['bb_pct']       = agg['bb'] / agg['tbf'].replace(0, np.nan)
    agg['bip_pct']      = agg['bip'] / agg['tbf'].replace(0, np.nan)
    agg['abs_pfxz']     = agg['avg_pfxz'].abs()
    return agg


def compute_per_start(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate per-(pitcher, game_pk) for starting pitcher only. Then mean per pitcher."""
    d = df.copy()
    # Identify starting pitcher of each (game_pk, inning_topbot side)
    # The starter throws the first pitch of the half-inning he pitches in
    # Easiest: pitcher is starter if first inning-1 pitch in game_pk for that side
    # In Statcast: for "Top" of inning, home team's pitcher; "Bot" → away pitcher
    # Use min(pitch_id) by (game_pk, pitcher) and check inning==1
    d['inning'] = pd.to_numeric(d['inning'], errors='coerce')
    starts = (d[d['inning'] == 1]
              .groupby(['game_pk','inning_topbot'])['pitcher']
              .first()
              .reset_index()
              .rename(columns={'pitcher':'starter_id'}))
    # Mark each pitch as belonging-to-start or not
    d = d.merge(starts, on=['game_pk','inning_topbot'], how='left')
    d['is_start_pitch'] = d['pitcher'] == d['starter_id']
    starter_pitches = d[d['is_start_pitch']].copy()

    # Per-game per-starter aggregation
    ev = starter_pitches['events'].fillna('')
    starter_pitches['is_k']   = ev == 'strikeout'
    starter_pitches['is_bb']  = ev == 'walk'
    starter_pitches['is_hbp'] = ev == 'hit_by_pitch'
    starter_pitches['is_hr']  = ev == 'home_run'
    starter_pitches['is_h']   = ev.isin(['single','double','triple','home_run'])
    starter_pitches['is_pa_end'] = ev != ''
    # Outs from this pitcher's PAs (ignore inherited runners; we're only counting
    # outs when he's still in)
    # Use 'outs_when_up' is at start of PA; for outs during PA: count via events that produce outs
    out_events = {'strikeout','field_out','grounded_into_double_play','sac_fly',
                  'sac_bunt','force_out','double_play','triple_play','fielders_choice_out',
                  'caught_stealing_2b','caught_stealing_3b','caught_stealing_home','other_out'}
    starter_pitches['outs_made'] = ev.isin(out_events).astype(int)
    # Ground-into-double-play counts as 2 outs
    starter_pitches.loc[ev.isin(['grounded_into_double_play','double_play']), 'outs_made'] = 2
    starter_pitches.loc[ev == 'triple_play', 'outs_made'] = 3
    # Sac fly inherits +1 (already in field_out via event), but sac_fly is separate event
    # Earned runs: cannot be inferred from pitch-level. Approximate: runs scored during their tenure.
    # We'll use bat_score / fld_score progression to estimate runs allowed during their PAs.

    g = starter_pitches.groupby(['game_pk','pitcher'])
    per_start = g.agg(
        k=('is_k','sum'),
        bb=('is_bb','sum'),
        hbp=('is_hbp','sum'),
        h=('is_h','sum'),
        hr=('is_hr','sum'),
        outs=('outs_made','sum'),
        tbf=('is_pa_end','sum'),
    ).reset_index()
    per_start['ip'] = per_start['outs'] / 3.0

    # Earned runs: approximate from runs_scored events in their PAs.
    # post_bat_score - bat_score on PA-ending pitches gives runs scored on the play.
    sp = starter_pitches.copy()
    sp['runs_on_play'] = (pd.to_numeric(sp['post_bat_score'], errors='coerce')
                          - pd.to_numeric(sp['bat_score'], errors='coerce')).clip(lower=0)
    sp.loc[~sp['is_pa_end'], 'runs_on_play'] = 0
    runs = sp.groupby(['game_pk','pitcher'])['runs_on_play'].sum().reset_index()
    per_start = per_start.merge(runs, on=['game_pk','pitcher'], how='left')
    per_start['er_est'] = per_start['runs_on_play'].fillna(0)  # approximation

    # ESPN FP: K + IP*3.3 - H - 2*ER - BB - HBP
    per_start['fp'] = (per_start['k']
                       + per_start['ip'] * 3.3
                       - per_start['h']
                       - 2 * per_start['er_est']
                       - per_start['bb']
                       - per_start['hbp'])

    # Aggregate per-pitcher (across all starts)
    pg = per_start.groupby('pitcher').agg(
        gs=('game_pk','count'),
        ip_per_start=('ip','mean'),
        k_per_start=('k','mean'),
        bb_per_start=('bb','mean'),
        h_per_start=('h','mean'),
        hr_per_start=('hr','mean'),
        hbp_per_start=('hbp','mean'),
        er_per_start=('er_est','mean'),
        fp_per_start_actual=('fp','mean'),
        fp_total=('fp','sum'),
    ).reset_index()
    pg['year'] = year
    return pg


def build():
    print('=== build_sp_multiyr ===', flush=True)
    season_frames = []
    per_start_frames = []
    for yr in YEARS:
        print(f'\n--- year {yr} ---', flush=True)
        try:
            raw = pull_year(yr)
        except Exception as e:
            print(f'  [{yr}] pull failed: {e}', flush=True)
            continue
        season = aggregate_pitcher_season(raw, yr)
        starts = compute_per_start(raw, yr)
        merged = season.merge(starts, on=['pitcher','year'], how='inner')
        # SP filter: gs >= 10 AND >= 500 pitches as starter (matches V6's 636 SP-seasons)
        # For 2026 YTD year only, allow gs>=3.
        # 2020 was 60-game season; lower cutoff to gs>=6 to retain SPs.
        if yr == 2026:
            sp = merged[(merged['gs'] >= 3) & (merged['pitches'] >= 100)].copy()
        elif yr == 2020:
            sp = merged[(merged['gs'] >= 6) & (merged['pitches'] >= 200)].copy()
        else:
            sp = merged[(merged['gs'] >= 10) & (merged['pitches'] >= 500)].copy()
        print(f'  [{yr}] {len(merged)} pitchers -> {len(sp)} SPs', flush=True)
        season_frames.append(sp)
        per_start_frames.append(starts)

    sp_all = pd.concat(season_frames, ignore_index=True)
    print(f'\nTotal SP-seasons: {len(sp_all)}', flush=True)
    # If extended (>= 2020 inclusion), also write a separate extended file
    if 2015 in YEARS or 2016 in YEARS or 2017 in YEARS or 2018 in YEARS or 2019 in YEARS or 2020 in YEARS:
        ext_out = CACHE / 'sp_multiyr_2015_2025.csv'
        sp_all.to_csv(ext_out, index=False)
        print(f'Wrote extended {ext_out} ({len(sp_all)} rows)', flush=True)
    out_path = TMP / 'sp_multiyr.csv'
    sp_all.to_csv(out_path, index=False)
    print(f'Wrote {out_path} ({len(sp_all)} rows, {len(sp_all.columns)} cols)', flush=True)

    # Also write extra metrics file
    extra = sp_all[['pitcher','year','xwoba_contact','xwoba_bip','avg_ev','barrel_pct','hard_hit_pct']].copy()
    extra_path = TMP / 'sp_extra_metrics.csv'
    extra.to_csv(extra_path, index=False)
    print(f'Wrote {extra_path}', flush=True)

    # Mirror to local cache for safety (in case /tmp is wiped)
    sp_all.to_csv(CACHE / 'sp_multiyr.csv', index=False)
    extra.to_csv(CACHE / 'sp_extra_metrics.csv', index=False)
    return sp_all


if __name__ == '__main__':
    build()
