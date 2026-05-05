"""
xfp_v8_5_pipeline.py - V8.5 contact-manager features.

Adds fb_pfxz, bb_pfxz, pfxz_spread (per-pitcher per-year) from cached Statcast.
Re-runs Phase 11E backward elimination with these new features in the candidate pool.
Targets the Fried/Suarez/Steele archetype that V8's swstr-heavy formula misses.

Decision rule: V8.5 ships only if score >= V8 (1.555) + 0.010.
"""
from __future__ import annotations
import sys, joblib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'xfp'))

from xfp_v7_pipeline import derive_features, add_ip_resid_lag, cross_year_evaluate
from xfp_v8_pipeline import V6_FEATS, V7_FEATS, V8_BASE, derive_v8_features, build_pitch_type_panel, score_fn

CACHE   = ROOT / 'data' / 'research' / 'xfp_cache'
OUTPUTS = ROOT / 'data' / 'outputs'
MODELS  = ROOT / 'data' / 'models'
RESEARCH= ROOT / 'data' / 'research'

LOG_CSV   = RESEARCH / 'feature_search_log.csv'

FB_TYPES = {'FF','SI','FC','FT'}
BB_TYPES = {'SL','CU','KC','SV','ST'}

V8_FEATS = ['swstr_pct','c_plus_swstr','xwoba_per_pa','xwoba_x_swstr']
V8_SCORE_BASE = 1.555  # benchmark to beat by >= 0.010


def derive_pitch_type_pfxz(year: int) -> pd.DataFrame | None:
    """Compute fb_pfxz, bb_pfxz, pfxz_spread per pitcher for given year from cached statcast."""
    cache_path = CACHE / f'statcast_{year}.parquet'
    if not cache_path.exists():
        return None
    df = pd.read_parquet(cache_path, columns=['pitcher','pitch_type','pfx_z'])
    df = df.dropna(subset=['pitcher','pitch_type','pfx_z'])

    df['family'] = pd.NA
    df.loc[df['pitch_type'].isin(FB_TYPES), 'family'] = 'FB'
    df.loc[df['pitch_type'].isin(BB_TYPES), 'family'] = 'BB'
    df = df.dropna(subset=['family'])

    # min 30 pitches per family
    g = df.groupby(['pitcher','family'])['pfx_z'].agg(['mean','count']).reset_index()
    g = g[g['count'] >= 30]
    pivot = g.pivot(index='pitcher', columns='family', values='mean')
    pivot.columns = [f'{c.lower()}_pfxz' for c in pivot.columns]  # 'fb_pfxz' / 'bb_pfxz'
    pivot['pfxz_spread'] = pivot.get('fb_pfxz', np.nan) - pivot.get('bb_pfxz', np.nan)
    pivot['year'] = year
    return pivot.reset_index()


def build_pfxz_panel(years: list[int]) -> pd.DataFrame:
    cache_csv = CACHE / 'sp_pitch_type_pfxz_2015_2026.csv'
    if cache_csv.exists():
        df = pd.read_csv(cache_csv)
        if set(years).issubset(set(df['year'].unique())):
            print(f'Loaded cached pfxz panel ({len(df)} rows)')
            return df
    frames = []
    for yr in years:
        f = derive_pitch_type_pfxz(yr)
        if f is not None:
            print(f'  pfxz {yr}: {len(f)} pitchers (fb_pfxz nn={f["fb_pfxz"].notna().sum()}, bb_pfxz nn={f["bb_pfxz"].notna().sum()})')
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(cache_csv, index=False)
    print(f'Saved {cache_csv}')
    return df


def append_log(rec: dict):
    rec = dict(rec)
    rec.setdefault('timestamp', datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z'))
    df = pd.DataFrame([rec])
    if LOG_CSV.exists():
        df.to_csv(LOG_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_CSV, index=False)


def evaluate_with_score(df, feats, label, phase):
    cyr = cross_year_evaluate(df, feats, label)
    score = score_fn(cyr['r'], cyr['k_bias_hi'])
    cyr['score'] = round(score, 5) if score != float('-inf') else None
    cyr['phase'] = phase
    cyr['label'] = label
    append_log(cyr)
    return cyr


def main():
    print('=' * 60)
    print(f'xFP V8.5 CONTACT-MANAGER FEATURES | {datetime.now(timezone.utc).isoformat()}')
    print('=' * 60)

    # Load training data
    df = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    df = derive_features(df)
    df = add_ip_resid_lag(df)
    df = derive_v8_features(df)

    # Load other pitch-type features (FF_spin etc)
    pt = build_pitch_type_panel(sorted(df['year'].unique()))
    if not pt.empty:
        df = df.merge(pt, on=['pitcher','year'], how='left')

    # Load NEW pfxz panel
    pfxz = build_pfxz_panel(sorted(df['year'].unique()))
    if not pfxz.empty:
        df = df.merge(pfxz, on=['pitcher','year'], how='left')
        print(f'Coverage: fb_pfxz={df["fb_pfxz"].notna().sum()}/{len(df)} '
              f'bb_pfxz={df["bb_pfxz"].notna().sum()}/{len(df)} '
              f'pfxz_spread={df["pfxz_spread"].notna().sum()}/{len(df)}')

    # FRIED VALIDATION CHECK
    print('\nFried 2025 pfxz check (should have fb_pfxz~+0.6 to +1.0, bb_pfxz~-0.6 to -1.0):')
    fried = df[(df['year']==2025) & df['player_name'].fillna('').str.contains('Fried', na=False)]
    if len(fried):
        f = fried.iloc[0]
        print(f"  Fried 2025: fb_pfxz={f.get('fb_pfxz', 'n/a')} bb_pfxz={f.get('bb_pfxz', 'n/a')} pfxz_spread={f.get('pfxz_spread', 'n/a')}")
    if not (len(fried) and pd.notna(fried.iloc[0].get('fb_pfxz')) and pd.notna(fried.iloc[0].get('bb_pfxz'))):
        print('  WARNING: Fried pfxz values missing or zero. Family grouping may be wrong.')

    train = df[df['year'].between(2015, 2025)].copy()
    print(f'\nTraining set: {len(train)} rows, years {sorted(train["year"].unique())}')

    # ===== STEP 2/3: Single-feature additions + BE =====
    print('\n===== Phase 1: single-feature additions =====')
    for cand in ['fb_pfxz','bb_pfxz','pfxz_spread']:
        if cand not in train.columns:
            print(f'  skip {cand}: missing'); continue
        feats = V8_FEATS + [cand]
        e = evaluate_with_score(train, feats, f'V8+{cand}', '11.5C')
        print(f'  V8+{cand:<14s} cross={e["r"]:.5f} kbias={e["k_bias_hi"]:+.3f} score={e["score"]:.5f}')

    print('\n===== Phase 11E (V8.5 BE) =====')
    pfxz_feats = [c for c in ['fb_pfxz','bb_pfxz','pfxz_spread'] if c in train.columns]
    other_pt   = [c for c in ['FF_spin','breaking_spin','offspeed_spin','vaa_ff','velo_diff','pitch_entropy']
                   if c in train.columns and train[c].notna().sum() > 100]
    kitchen = list(dict.fromkeys(V8_BASE + ['k_pct_lag1','bb_pct_lag1'] + other_pt + pfxz_feats))
    print(f'  kitchen sink ({len(kitchen)}): {kitchen}')

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    current = list(kitchen)
    best_score, best_set, best_eval = -float('inf'), list(kitchen), None
    while len(current) >= 4:
        d_curr = train.dropna(subset=current+['fp_per_start_actual'])
        if len(d_curr) < 100:
            print(f'  n={len(current)}: only {len(d_curr)} rows, stopping'); break
        sc = StandardScaler()
        X = sc.fit_transform(d_curr[current])
        ridge = RidgeCV(alphas=np.logspace(-1,5,80), cv=5).fit(X, d_curr['fp_per_start_actual'])
        coefs = pd.Series(np.abs(ridge.coef_), index=current).sort_values()
        e = evaluate_with_score(train, current, f'V8.5_BE_{len(current)}', '11.5E')
        print(f'  n={len(current):2d} cross={e["r"]:.5f} kbias={e["k_bias_hi"]:+.3f} score={e["score"]:.5f} drop={coefs.index[0]} ({coefs.iloc[0]:.3f})')
        if e['score'] is not None and e['score'] > best_score:
            best_score = e['score']; best_set = list(current); best_eval = e
        current = [f for f in current if f != coefs.index[0]]

    print(f'\nV8.5 best BE: score={best_score:.5f} cross={best_eval["r"]:.5f} kbias={best_eval["k_bias_hi"]:+.3f}')
    print(f'  features ({len(best_set)}): {best_set}')

    pfxz_in_best = [f for f in best_set if f in pfxz_feats]
    has_pfxz = len(pfxz_in_best) > 0
    score_delta = best_score - V8_SCORE_BASE
    print(f'\nV8.5 score delta vs V8 ({V8_SCORE_BASE}): {score_delta:+.5f}')
    print(f'pfxz features in best set: {pfxz_in_best}')

    # ===== Decision rule =====
    SHIPS = (score_delta >= 0.010)
    print(f'\nDECISION: V8.5 {"SHIPS" if SHIPS else "DOES NOT SHIP"} '
          f'(threshold delta >= 0.010, actual {score_delta:+.5f})')

    # Contact-manager subset bias
    cm_mask = (train['gb_pct'] > 0.50) & (train['swstr_pct'] < 0.12)
    n_cm = cm_mask.sum()
    print(f'\nContact-manager subset (gb_pct>0.50 & swstr_pct<0.12): n={n_cm}')

    if SHIPS:
        # Train final V8.5
        train_v85 = train.dropna(subset=best_set + ['fp_per_start_actual'])
        pipe_v85 = Pipeline([('sc', StandardScaler()),
                             ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=5))])
        pipe_v85.fit(train_v85[best_set], train_v85['fp_per_start_actual'])
        pkl_path = MODELS / 'xfp_v8_5_pipeline.pkl'
        joblib.dump({'pipeline':pipe_v85, 'features':best_set, 'name':'V8.5_BE_best',
                     'metrics':best_eval,
                     'score_formula':'cross_year_r * 3 - abs(k_bias_hi) * 0.5'}, pkl_path)
        print(f'  saved {pkl_path}')

        # Print coefs
        coefs = pd.Series(pipe_v85.named_steps['r'].coef_, index=best_set)
        print('  Standardized coefs:')
        for f, c in coefs.sort_values(key=abs, ascending=False).items():
            print(f'    {f:<25s}: {c:+.3f}')

        # Build projections
        proj = build_v85_projections(df, train, best_set, pipe_v85)
        proj_path = OUTPUTS / 'xfp_v8_5_projections.csv'
        proj.to_csv(proj_path, index=False)
        print(f'  wrote {proj_path}')

        # Build dashboard
        build_v85_dashboard(proj, best_eval, best_set, coefs)
        print(f'  wrote {OUTPUTS / "xfp_v8_5_dashboard.html"}')

    # Append research notes
    append_v85_research(SHIPS, best_set, best_eval, score_delta, pfxz_in_best, n_cm)
    print('\nWORKSTREAM 2 COMPLETE — V8.5 ' + ('SHIPPED' if SHIPS else 'NOT SHIPPED'))
    return SHIPS, best_set, best_eval


def build_v85_projections(df, train, feats, pipe_v85):
    """Build V8.5 2026 projections using the V8.1 mid-season blend approach."""
    # Reuse V8.1 blend logic but with V8.5 features
    from xfp_v8_midseason import blend_pitcher, RATE_FEATS, PRIOR_N, PRIOR_MEAN
    df_25 = df[df['year']==2025].set_index('pitcher')
    df_26 = df[df['year']==2026].set_index('pitcher')
    pitchers_union = sorted(set(df_25.index) | set(df_26.index))
    blended_rows = []
    for p in pitchers_union:
        r25 = df_25.loc[p].to_dict() if p in df_25.index else None
        r26 = df_26.loc[p].to_dict() if p in df_26.index else None
        if r25: r25 = pd.Series({**r25, 'pitcher': p})
        if r26: r26 = pd.Series({**r26, 'pitcher': p})
        b = blend_pitcher(r25, r26)
        if b is None: continue
        # For non-rate features (lag, pfxz, pitch type) take from 2026 if available else 2025
        for f in feats:
            if f in b: continue
            if r26 is not None and pd.notna(r26.get(f)):
                b[f] = float(r26.get(f))
            elif r25 is not None and pd.notna(r25.get(f)):
                b[f] = float(r25.get(f))
            else:
                b[f] = np.nan
        blended_rows.append(b)
    blended_df = pd.DataFrame(blended_rows)
    # Recompute interactions
    blended_df['xwoba_per_pa']  = blended_df['xwoba_contact'] * blended_df['bip_pct']
    blended_df['xwoba_x_swstr'] = blended_df['xwoba_contact'] * blended_df['swstr_pct']
    valid = blended_df.dropna(subset=feats).copy()
    valid['xfp_v8_5'] = pipe_v85.predict(valid[feats])

    # Merge with V8.1 baseline
    v81 = pd.read_csv(OUTPUTS / 'xfp_v8_1_projections.csv')
    out = valid[['pitcher','player_name','xfp_v8_5','cohort','weight_2026']].merge(
        v81[['pitcher','xfp_v8','xfp_v8_1','xfp_v7','xfp_v6','xfp_v5','gs_2026','fp_per_start_actual_2026','k_pct_2026']],
        on='pitcher', how='left')
    out['delta_v8_5_v8_1'] = out['xfp_v8_5'] - out['xfp_v8_1']
    return out


def build_v85_dashboard(proj, best_eval, feats, coefs):
    """Render xfp_v8_5_dashboard.html."""
    proj_d = proj.sort_values('xfp_v8_5', ascending=False, na_position='last').reset_index(drop=True)
    proj_d['rank_v8_5'] = proj_d.index + 1
    proj_d['rank_v8_1'] = proj_d['xfp_v8_1'].rank(ascending=False, method='min')

    def fmt(x, p=2):
        try:
            f = float(x)
            if not np.isfinite(f): return '-'
            return f'{f:.{p}f}'
        except (TypeError, ValueError):
            return '-'

    def archetype_card(name):
        r = proj_d[proj_d['player_name'].fillna('').str.contains(name, na=False)]
        if not len(r):
            return f'<div class="card"><div class="cardh">{name}</div><div>not in set</div></div>'
        s = r.iloc[0]
        return (f'<div class="card"><div class="cardh">{s["player_name"]}</div>'
                f'<div class="kv"><span class="kv-k">V8.1 xFP</span><span class="kv-v">{fmt(s["xfp_v8_1"])}</span></div>'
                f'<div class="kv"><span class="kv-k">V8.5 xFP</span><span class="kv-v" style="color:#3fb950">{fmt(s["xfp_v8_5"])}</span></div>'
                f'<div class="kv"><span class="kv-k">delta</span><span class="kv-v">{s["delta_v8_5_v8_1"]:+.2f}</span></div>'
                f'<div class="kv"><span class="kv-k">2026 actual</span><span class="kv-v">{fmt(s["fp_per_start_actual_2026"])}</span></div>'
                f'</div>')

    archetype_html = ''.join(archetype_card(n) for n in
        ['Schlittler','Glasnow','Imanaga','Fried','Suarez, Ranger','Steele, Justin','Woodruff'])

    # Top 80
    main_rows = []
    for _, s in proj_d.head(80).iterrows():
        try: dv = float(s['delta_v8_5_v8_1']); dvc = 'up' if dv>0 else 'dn' if dv<0 else ''
        except: dvc=''; dv=0
        cls = 't1' if s['rank_v8_5']==1 else 't2' if s['rank_v8_5']==2 else 't3' if s['rank_v8_5']==3 else ''
        gold = ' style="color:#ffd700"' if 'Schlittler' in str(s['player_name']) else ''
        main_rows.append(
            f'<tr><td class="{cls}">{int(s["rank_v8_5"])}</td><td{gold}>{s["player_name"]}</td>'
            f'<td class="num">{fmt(s["xfp_v8"])}</td>'
            f'<td class="num">{fmt(s["xfp_v8_1"])}</td>'
            f'<td class="num"><b>{fmt(s["xfp_v8_5"])}</b></td>'
            f'<td class="num {dvc}">{dv:+.2f}</td>'
            f'<td class="num">{fmt(s["gs_2026"],0)}</td>'
            f'<td class="num">{fmt(s["fp_per_start_actual_2026"])}</td></tr>')

    coef_html = ''.join(f'<div class="kv"><span class="kv-k">{f}</span><span class="kv-v">{c:+.3f}</span></div>'
                          for f,c in coefs.sort_values(key=abs, ascending=False).items())

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>xFP v8.5 contact-manager features</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;padding:18px}}
.hdr{{background:linear-gradient(135deg,#1a2332,#0d1b2a);border:1px solid #30363d;border-radius:8px;padding:14px 18px;margin-bottom:14px}}
.title{{font-size:20px;font-weight:700;color:#58a6ff}}.title span{{color:#f0883e}}
.sub{{font-size:11.5px;color:#8b949e;margin-top:4px;line-height:1.5}}
.card{{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px}}
.cardh{{font-size:11px;font-weight:700;text-transform:uppercase;color:#8b949e;letter-spacing:.7px;margin-bottom:9px}}
.grid7{{display:grid;grid-template-columns:repeat(7, minmax(0, 1fr));gap:8px;margin-bottom:14px}}
.grid2{{display:grid;grid-template-columns:1fr 2fr;gap:14px;margin-bottom:14px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:6px 6px;color:#8b949e;border-bottom:2px solid #21262d;font-weight:600}}
td{{padding:5px 6px;border-bottom:1px solid #21262d;font-variant-numeric:tabular-nums}}
td.num{{text-align:right}}
.t1{{color:#ffd700;font-weight:700}}.t2{{color:#c0c0c0;font-weight:600}}.t3{{color:#cd7f32}}
.up{{color:#3fb950}}.dn{{color:#f85149}}
.kv{{display:flex;justify-content:space-between;padding:3px 0;font-size:11.5px}}
.kv-k{{color:#8b949e}}.kv-v{{font-weight:700}}
</style></head><body>
<div class="hdr">
<div class="title">xFP <span>v8.5</span> Contact-Manager Features</div>
<div class="sub">Adds fb_pfxz, bb_pfxz, pfxz_spread to V8 candidate pool. Backward elimination from kitchen sink. Targets the Fried/Suarez/Steele archetype that V8's swstr-heavy formula underweights.</div>
</div>

<div class="card" style="margin-bottom:14px"><div class="cardh">V8.5 metrics</div>
<div class="kv"><span class="kv-k">Cross-year r</span><span class="kv-v">{best_eval['r']}</span></div>
<div class="kv"><span class="kv-k">k_bias_hi</span><span class="kv-v">{best_eval['k_bias_hi']}</span></div>
<div class="kv"><span class="kv-k">Score</span><span class="kv-v" style="color:#3fb950">{best_eval['score']}</span></div>
<div class="kv"><span class="kv-k">vs V8 (1.555)</span><span class="kv-v">{best_eval['score']-1.555:+.5f}</span></div>
<div class="kv"><span class="kv-k">Features ({len(feats)})</span><span class="kv-v">{', '.join(feats)}</span></div>
</div>

<div class="grid2">
<div class="card"><div class="cardh">Standardized coefs</div>{coef_html}</div>
<div class="card"><div class="cardh">Archetype callouts (Fried/Suarez/Steele are key)</div>
<div class="grid7">{archetype_html}</div></div>
</div>

<div class="card"><div class="cardh">Top 80 SP — V8 / V8.1 / V8.5 (Schlittler in gold)</div>
<table><thead><tr><th>Rk V8.5</th><th>Pitcher</th><th class="num">V8 xFP</th><th class="num">V8.1 xFP</th><th class="num">V8.5 xFP</th><th class="num">Δ vs V8.1</th><th class="num">2026 GS</th><th class="num">Actual</th></tr></thead>
<tbody>{''.join(main_rows)}</tbody></table></div>
</body></html>'''
    out_path = OUTPUTS / 'xfp_v8_5_dashboard.html'
    out_path.write_text(html, encoding='utf-8')


def append_v85_research(SHIPS, best_set, best_eval, score_delta, pfxz_in_best, n_cm):
    section = f"""

## V8.5 — Contact-Manager Features ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})

Added per-pitcher per-year fb_pfxz (FB family pfx_z), bb_pfxz (BB family pfx_z), pfxz_spread (FB-BB).
Re-ran Phase 11E backward elimination with these in the candidate pool.

### Result: V8.5 {"SHIPPED" if SHIPS else "NOT SHIPPED"} (decision rule: score delta >= 0.010)

| | V8 | V8.5 | Δ |
|---|---|---|---|
| Cross-year r | 0.55839 | {best_eval['r']} | {(best_eval['r'] or 0)-0.55839:+.5f} |
| k_bias_hi | 0.241 | {best_eval['k_bias_hi']} | {(best_eval['k_bias_hi'] or 0)-0.241:+.3f} |
| Score | 1.555 | {best_eval['score']} | {score_delta:+.5f} |

### V8.5 best feature set ({len(best_set)})
{', '.join(best_set)}

pfxz features that survived BE: {pfxz_in_best if pfxz_in_best else 'NONE — pfxz dropped during elimination'}

Contact-manager subset (gb_pct > 0.50 AND swstr_pct < 0.12): n={n_cm} pitcher-seasons.

### Files
{('- `data/models/xfp_v8_5_pipeline.pkl`' + chr(10) + '- `data/outputs/xfp_v8_5_projections.csv`' + chr(10) + '- `data/outputs/xfp_v8_5_dashboard.html`') if SHIPS else 'No model artifacts saved (decision rule failed).'}
"""
    research_md = RESEARCH / 'xfp_model_research.md'
    with open(research_md, 'a', encoding='utf-8') as f:
        f.write(section)
    print(f'  appended V8.5 section to {research_md}')


if __name__ == '__main__':
    main()
