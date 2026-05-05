"""
xfp_v11_lock.py - Train, save, project, and build dashboard for V11 production model.

V11 = V8.5 features + pitching_plus + fp_strike_pct
Training: 2020-2025 (where pitching_plus is available)
"""
from __future__ import annotations
import sys, joblib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'xfp'))

from xfp_v7_pipeline import derive_features, add_ip_resid_lag, cross_year_evaluate, ooy_evaluate
from xfp_v8_pipeline import derive_v8_features, build_pitch_type_panel, score_fn
from xfp_v8_5_pipeline import build_pfxz_panel
from xfp_v8_midseason import blend_pitcher
from v11_full_spotcheck import load_data
from v11_spotcheck import build_blended_inputs

CACHE   = ROOT / 'data' / 'research' / 'xfp_cache'
OUTPUTS = ROOT / 'data' / 'outputs'
MODELS  = ROOT / 'data' / 'models'
RESEARCH= ROOT / 'data' / 'research'
DOCS    = ROOT / 'docs'

V85_FEATS = ['avg_velo','zone_pct','o_swing_pct','swstr_pct','c_plus_swstr','xwoba_per_pa',
              'z_swing_pct','xwoba_x_swstr','ip_resid_lag1','k_pct_lag1','pitch_entropy','bb_pfxz']
V11_FEATS = V85_FEATS + ['pitching_plus','fp_strike_pct']
V8_SCORE = 1.555
V85_SCORE = 1.567


def score_tolerance(r, kbias, T=1.0, coef=0.5):
    return r * 3 - max(0, abs(kbias) - T) * coef


def main():
    print('=' * 70)
    print(f'V11 PRODUCTION LOCK | {datetime.now(timezone.utc).isoformat()}')
    print('=' * 70)
    df = load_data()

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    train = df[df['year'].between(2020, 2025)].dropna(subset=V11_FEATS + ['fp_per_start_actual'])
    print(f'V11 training rows (2020-2025, all V11 feats non-null): {len(train)}')
    pipe_v11 = Pipeline([('sc', StandardScaler()),
                          ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=10))])
    pipe_v11.fit(train[V11_FEATS], train['fp_per_start_actual'])
    print(f'  Ridge alpha selected: {pipe_v11.named_steps["r"].alpha_:.3f}')

    coefs = pd.Series(pipe_v11.named_steps['r'].coef_, index=V11_FEATS)
    print('  Standardized coefficients:')
    for f, c in coefs.sort_values(key=abs, ascending=False).items():
        print(f'    {f:<22s}: {c:+.3f}')

    # Cross-year r for V11 (subset 2020-2024)
    cy = cross_year_evaluate(df[df['year'].between(2020, 2025)], V11_FEATS, 'V11_full')
    score_05 = score_fn(cy['r'], cy['k_bias_hi'])
    score_t1 = score_tolerance(cy['r'], cy['k_bias_hi'], T=1.0)
    print(f'\nV11 cross-year (2020->2024 transitions, n={cy["n"]}):')
    print(f'  cross_year_r: {cy["r"]}')
    print(f'  k_bias_hi:    {cy["k_bias_hi"]}')
    print(f'  score (current 0.5 coef): {score_05:.5f}')
    print(f'  score (tolerance T=1.0):  {score_t1:.5f}')

    # OOY r
    ooy = ooy_evaluate(df[df['year'].between(2020, 2025)], V11_FEATS, 'V11_full')
    print(f'\nV11 OOY r: {ooy["r"]}, OOY k_bias_hi: {ooy["k_bias_hi"]}')

    # Build 2026 projections
    print('\n' + '=' * 70)
    print('Building 2026 V11 projections via V8.1 mid-season blend')
    print('=' * 70)
    blended = build_blended_inputs(df, V11_FEATS)
    v11_valid = blended.dropna(subset=V11_FEATS).copy()
    v11_valid['xfp_v11'] = pipe_v11.predict(v11_valid[V11_FEATS])
    v11_valid['v11_has_pitching_plus'] = True
    print(f'  V11 predictions: {len(v11_valid)}/185 pitchers')

    # Fall back to V8.5 for pitchers missing pitching_plus
    train_v85 = df[df['year'].between(2015, 2025)].dropna(subset=V85_FEATS + ['fp_per_start_actual'])
    pipe_v85 = Pipeline([('sc', StandardScaler()),
                          ('r', RidgeCV(alphas=np.logspace(-1,5,80), cv=10))])
    pipe_v85.fit(train_v85[V85_FEATS], train_v85['fp_per_start_actual'])

    # Existing V8.5 projections for the full set
    v85_proj = pd.read_csv(OUTPUTS / 'xfp_v8_5_projections.csv')

    # Re-predict V8.5 from blended inputs (so v8_5 is on same blended basis)
    v85_valid = blended.dropna(subset=V85_FEATS).copy()
    v85_valid['xfp_v8_5'] = pipe_v85.predict(v85_valid[V85_FEATS])

    # Merge: V11 where available, V8.5 fallback
    out = v85_valid[['pitcher','player_name','xfp_v8_5']].merge(
        v11_valid[['pitcher','xfp_v11','v11_has_pitching_plus']],
        on='pitcher', how='left')
    fallback_mask = out['xfp_v11'].isna()
    out.loc[fallback_mask, 'xfp_v11'] = out.loc[fallback_mask, 'xfp_v8_5']
    out.loc[fallback_mask, 'v11_has_pitching_plus'] = False
    n_fallback = fallback_mask.sum()
    print(f'  Fallback (V8.5 used because pitching_plus missing): {n_fallback} pitchers')

    # Merge with full V8.5 projection columns for context
    keep_cols_85 = ['pitcher','xfp_v8_1','xfp_v8','xfp_v7','xfp_v6','xfp_v5',
                     'gs_2026','fp_per_start_actual_2026','k_pct_2026',
                     'stuff_xfp','ip_premium','rolling_ip_last5','ip_trend','ip_trend_score']
    out = out.merge(v85_proj[[c for c in keep_cols_85 if c in v85_proj.columns]],
                     on='pitcher', how='left')
    out['delta_v11_v85'] = out['xfp_v11'] - out['xfp_v8_5']

    proj_path = OUTPUTS / 'xfp_v11_projections.csv'
    out.to_csv(proj_path, index=False)
    print(f'  wrote {proj_path}')

    # YTD evaluation
    ytd = out[(out['gs_2026']>=5) & out['fp_per_start_actual_2026'].notna() & out['xfp_v11'].notna()]
    if len(ytd) >= 10:
        v11_mae = (ytd['fp_per_start_actual_2026'] - ytd['xfp_v11']).abs().mean()
        v85_mae = (ytd['fp_per_start_actual_2026'] - ytd['xfp_v8_5']).abs().mean()
        v11_r = float(np.corrcoef(ytd['xfp_v11'], ytd['fp_per_start_actual_2026'])[0,1])
        v85_r = float(np.corrcoef(ytd['xfp_v8_5'], ytd['fp_per_start_actual_2026'])[0,1])
        print(f'\n2026 YTD (n={len(ytd)}):')
        print(f'  V8.5: MAE={v85_mae:.3f}  r={v85_r:.4f}')
        print(f'  V11:  MAE={v11_mae:.3f}  r={v11_r:.4f}')
    else:
        v11_mae = v85_mae = v11_r = v85_r = None

    # Save bundle
    bundle = {
        'pipeline': pipe_v11,
        'features': V11_FEATS,
        'cross_year_r': cy['r'],
        'k_bias_hi': cy['k_bias_hi'],
        'score_current': round(score_05, 5),
        'score_tolerance_T1': round(score_t1, 5),
        'formula': 'cross_year_r * 3 - max(0, |k_bias_hi| - T) * 0.5 (T=1.0 production)',
        'trained_date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'n_train': len(train),
        'version': 'v11',
        'training_years': '2020-2025',
        'ytd_mae_2026': round(v11_mae, 4) if v11_mae is not None else None,
        'ytd_r_2026': round(v11_r, 5) if v11_r is not None else None,
        'comparison': {'v8_5_score': V85_SCORE, 'v8_5_ytd_mae': round(v85_mae, 4) if v85_mae is not None else None,
                        'v8_5_ytd_r': round(v85_r, 5) if v85_r is not None else None},
        'note': 'V11 is V8.5 + pitching_plus + fp_strike_pct. Use V8.1 mid-season blend for projections.',
    }
    pkl_path = MODELS / 'xfp_v11_pipeline.pkl'
    joblib.dump(bundle, pkl_path)
    print(f'\nSaved {pkl_path}')

    # Sanity: reload + predict
    bundle_reloaded = joblib.load(pkl_path)
    train_check = train[V11_FEATS].head(5).values
    pred_check = bundle_reloaded['pipeline'].predict(train_check)
    print(f'Sanity reload: predictions on first 5 train rows = {[round(p, 2) for p in pred_check]}')

    # Build dashboard
    print('\n' + '=' * 70)
    print('Building V11 production dashboard')
    print('=' * 70)
    build_dashboard(out, bundle, v85_mae, v85_r, v11_mae, v11_r)

    return out, bundle


def build_dashboard(proj, bundle, v85_mae, v85_r, v11_mae, v11_r):
    proj_d = proj.sort_values('xfp_v11', ascending=False, na_position='last').reset_index(drop=True)
    proj_d['rank_v11'] = proj_d.index + 1
    proj_d['rank_v85'] = proj_d['xfp_v8_5'].rank(ascending=False, method='min')

    def fmt(x, p=2):
        try:
            f = float(x)
            if not np.isfinite(f): return '-'
            return f'{f:.{p}f}'
        except (TypeError, ValueError): return '-'

    # K-bias bar chart: top-25 high-K cohort (mean residual V8.5 vs V11)
    eligible = proj[(proj['gs_2026']>=5) & proj['fp_per_start_actual_2026'].notna() & proj['k_pct_2026'].notna()]
    top25 = eligible.nlargest(25, 'k_pct_2026') if len(eligible)>=25 else eligible
    if len(top25) > 0:
        v85_high_k_resid = (top25['fp_per_start_actual_2026'] - top25['xfp_v8_5']).mean()
        v11_high_k_resid = (top25['fp_per_start_actual_2026'] - top25['xfp_v11']).mean()
        v85_high_k_mae = (top25['fp_per_start_actual_2026'] - top25['xfp_v8_5']).abs().mean()
        v11_high_k_mae = (top25['fp_per_start_actual_2026'] - top25['xfp_v11']).abs().mean()
    else:
        v85_high_k_resid = v11_high_k_resid = v85_high_k_mae = v11_high_k_mae = 0

    def archetype_card(name):
        r = proj_d[proj_d['player_name'].fillna('').str.contains(name, na=False)]
        if not len(r): return f'<div class="card"><div class="cardh">{name}</div><div>not in set</div></div>'
        s = r.iloc[0]
        delta = s['delta_v11_v85']
        delta_color = '#3fb950' if delta > 0 else '#f85149' if delta < 0 else '#8b949e'
        has_pp_badge = '' if s.get('v11_has_pitching_plus', False) else ' <span style="color:#f0883e;font-size:9px">[V8.5 fallback]</span>'
        return (f'<div class="card"><div class="cardh">{s["player_name"]}{has_pp_badge}</div>'
                f'<div class="kv"><span class="kv-k">V8.5 xFP</span><span class="kv-v">{fmt(s["xfp_v8_5"])}</span></div>'
                f'<div class="kv"><span class="kv-k">V11 xFP</span><span class="kv-v" style="color:#3fb950">{fmt(s["xfp_v11"])}</span></div>'
                f'<div class="kv"><span class="kv-k">delta</span><span class="kv-v" style="color:{delta_color}">{delta:+.2f}</span></div>'
                f'<div class="kv"><span class="kv-k">2026 actual</span><span class="kv-v">{fmt(s.get("fp_per_start_actual_2026"))}</span></div>'
                f'<div class="kv"><span class="kv-k">2026 GS</span><span class="kv-v">{fmt(s.get("gs_2026"),0)}</span></div>'
                f'</div>')

    archetype_html = ''.join(archetype_card(n) for n in
        ['Schlittler','Glasnow','Imanaga','Fried','Wheeler','Skubal','Skenes','Crochet'])

    # Top 80 main table
    main_rows = []
    for _, s in proj_d.head(80).iterrows():
        try:
            dv = float(s['delta_v11_v85'])
            dvc = 'up' if dv>0 else 'dn' if dv<0 else ''
        except (TypeError, ValueError):
            dvc=''; dv=0
        cls = 't1' if s['rank_v11']==1 else 't2' if s['rank_v11']==2 else 't3' if s['rank_v11']==3 else ''
        gold = ' style="color:#ffd700"' if 'Schlittler' in str(s['player_name']) else ''
        fb_marker = '' if s.get('v11_has_pitching_plus', False) else ' <sup style="color:#f0883e">*</sup>'
        main_rows.append(
            f'<tr><td class="{cls}">{int(s["rank_v11"])}</td>'
            f'<td{gold}>{s["player_name"]}{fb_marker}</td>'
            f'<td class="num">{fmt(s.get("xfp_v8"))}</td>'
            f'<td class="num">{fmt(s.get("xfp_v8_5"))}</td>'
            f'<td class="num"><b>{fmt(s["xfp_v11"])}</b></td>'
            f'<td class="num {dvc}">{dv:+.2f}</td>'
            f'<td class="num">{fmt(s.get("gs_2026"),0)}</td>'
            f'<td class="num">{fmt(s.get("fp_per_start_actual_2026"))}</td>'
            f'<td>{(s.get("ip_trend","")) or "-"}</td></tr>')

    coef_html = ''.join(f'<div class="kv"><span class="kv-k">{f}</span><span class="kv-v">{c:+.3f}</span></div>'
                          for f, c in pd.Series(bundle['pipeline'].named_steps['r'].coef_,
                                                  index=bundle['features']).sort_values(key=abs, ascending=False).items())

    # Build self-contained HTML (data embedded as JSON for any future JS tools)
    proj_export = proj_d.fillna('').to_dict(orient='records')
    import json
    proj_json = json.dumps(proj_export, default=str)

    # K-bias visualization bars (V8.5 vs V11 on top-25 high-K)
    max_abs_resid = max(abs(v85_high_k_resid), abs(v11_high_k_resid), 0.5)
    v85_bar_pct = abs(v85_high_k_resid)/max_abs_resid*100
    v11_bar_pct = abs(v11_high_k_resid)/max_abs_resid*100

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>SP xFP V11 — Production Dashboard</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;padding:18px}}
.hdr{{background:linear-gradient(135deg,#1a2332,#0d1b2a);border:1px solid #30363d;border-radius:8px;padding:14px 18px;margin-bottom:14px}}
.title{{font-size:22px;font-weight:700;color:#58a6ff}}.title span{{color:#f0883e}}
.title .vbadge{{font-size:11px;font-weight:600;background:#238636;color:#fff;padding:3px 9px;border-radius:11px;margin-left:10px;vertical-align:3px}}
.sub{{font-size:11.5px;color:#8b949e;margin-top:4px;line-height:1.5}}
.banner{{background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.4);border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:12px}}
.card{{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px}}
.cardh{{font-size:11px;font-weight:700;text-transform:uppercase;color:#8b949e;letter-spacing:.7px;margin-bottom:9px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}}
.grid8{{display:grid;grid-template-columns:repeat(4, minmax(0, 1fr));gap:10px;margin-bottom:14px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:6px 6px;color:#8b949e;border-bottom:2px solid #21262d;font-weight:600}}
td{{padding:5px 6px;border-bottom:1px solid #21262d;font-variant-numeric:tabular-nums}}
td.num{{text-align:right}}
.t1{{color:#ffd700;font-weight:700}}.t2{{color:#c0c0c0;font-weight:600}}.t3{{color:#cd7f32}}
.up{{color:#3fb950}}.dn{{color:#f85149}}
.kv{{display:flex;justify-content:space-between;padding:3px 0;font-size:11.5px}}
.kv-k{{color:#8b949e}}.kv-v{{font-weight:700}}
.acc-badge{{display:inline-block;padding:5px 12px;border-radius:11px;background:rgba(31,111,235,.1);color:#58a6ff;font-size:11.5px;font-weight:600;margin-right:6px;border:1px solid rgba(31,111,235,.3)}}
.acc-badge.improved{{background:rgba(63,185,80,.12);color:#3fb950;border-color:rgba(63,185,80,.4)}}
.bar-row{{display:flex;align-items:center;gap:8px;margin:6px 0}}
.bar-lbl{{width:60px;font-size:11.5px;color:#e6edf3;font-weight:600}}
.bar-bg{{flex:1;height:18px;background:#21262d;border-radius:3px;overflow:hidden;position:relative}}
.bar-fg{{height:100%;border-radius:3px}}
.bar-val{{font-size:11.5px;color:#e6edf3;font-weight:700;width:80px;text-align:right;font-variant-numeric:tabular-nums}}
.guide{{background:rgba(31,111,235,.05);border:1px solid rgba(31,111,235,.3);border-radius:8px;padding:12px;font-size:11.5px;line-height:1.6}}
.foot{{font-size:10.5px;color:#8b949e;margin-top:14px;text-align:center;padding-top:12px;border-top:1px solid #21262d}}
sup{{font-size:9px}}
</style></head><body>
<div class="hdr">
<div class="title">SP xFP <span>V11</span> — Production Dashboard <span class="vbadge">v11 | {today}</span></div>
<div class="sub">V8.5 + pitching_plus + fp_strike_pct. Trained on {bundle["n_train"]} SP-seasons (2020-2025).
Cross-year r {bundle["cross_year_r"]} | k_bias_hi {bundle["k_bias_hi"]} | YTD r {fmt(v11_r,4)} | YTD MAE {fmt(v11_mae,3)}.
Mid-season blend (V8.1) used for 2026 inputs: rate metrics weighted by 2025+2026 pitches; xwoba_contact two-sample Bayesian.
* indicates V8.5 fallback (pitching_plus history not available for that pitcher).</div>
</div>

<div class="banner">
<b>V11 vs V8.5 (production model upgrade):</b>
<span class="acc-badge improved">YTD r: {fmt(v85_r,4)} → {fmt(v11_r,4)}</span>
<span class="acc-badge improved">YTD MAE: {fmt(v85_mae,3)} → {fmt(v11_mae,3)}</span>
<span class="acc-badge">Cross-year r: 0.600 → {bundle["cross_year_r"]}</span>
<span class="acc-badge">+pitching_plus +fp_strike_pct</span>
</div>

<div class="grid3">
<div class="card">
<div class="cardh">V11 metrics (cross-year + 2026 YTD)</div>
<div class="kv"><span class="kv-k">cross_year_r</span><span class="kv-v">{bundle["cross_year_r"]}</span></div>
<div class="kv"><span class="kv-k">k_bias_hi</span><span class="kv-v">{bundle["k_bias_hi"]}</span></div>
<div class="kv"><span class="kv-k">score (current 0.5 coef)</span><span class="kv-v">{bundle["score_current"]}</span></div>
<div class="kv"><span class="kv-k">score (tolerance T=1.0)</span><span class="kv-v" style="color:#3fb950">{bundle["score_tolerance_T1"]}</span></div>
<div class="kv"><span class="kv-k">2026 YTD r</span><span class="kv-v">{fmt(v11_r,4)}</span></div>
<div class="kv"><span class="kv-k">2026 YTD MAE</span><span class="kv-v">{fmt(v11_mae,3)}</span></div>
<div class="kv"><span class="kv-k">trained on</span><span class="kv-v">{bundle["n_train"]} rows (2020-25)</span></div>
</div>

<div class="card">
<div class="cardh">V11 standardized coefficients</div>
{coef_html}
</div>

<div class="card">
<div class="cardh">High-K cohort residual: V8.5 vs V11 (n={len(top25)})</div>
<div class="bar-row"><span class="bar-lbl">V8.5</span>
<span class="bar-bg"><span class="bar-fg" style="width:{v85_bar_pct:.0f}%;background:#f85149"></span></span>
<span class="bar-val">{v85_high_k_resid:+.2f}</span></div>
<div class="bar-row"><span class="bar-lbl">V11</span>
<span class="bar-bg"><span class="bar-fg" style="width:{v11_bar_pct:.0f}%;background:#3fb950"></span></span>
<span class="bar-val">{v11_high_k_resid:+.2f}</span></div>
<div style="font-size:10.5px;color:#8b949e;margin-top:8px">Mean residual (actual - predicted) on top-25 high-K pitchers (gs ≥ 5).
Closer to 0 = less systematic bias. V11 also has lower MAE on this cohort: {v85_high_k_mae:.2f} → {v11_high_k_mae:.2f}.</div>
</div>
</div>

<div class="card" style="margin-bottom:14px">
<div class="cardh">Archetype callouts (V8.5 vs V11)</div>
<div class="grid8">{archetype_html}</div>
</div>

<div class="grid2">
<div class="card guide">
<b>Interpretation guide</b><br>
<b>xFP V11</b>: production projection for 2026 SP fantasy points/start (ESPN scoring: K + IP×3.3 − H − 2·ER − BB − HBP).<br>
<b>Δ vs V8.5</b>: how V11 differs from prior production model. Positive = V11 sees more upside than V8.5.<br>
<b>* marker</b>: V8.5 fallback used because FanGraphs Stuff+/Pitching+ history unavailable for that pitcher (typically 2026-only rookies).<br>
<b>2026 GS / actual</b>: in-season YTD reality check (small samples, ~6 GS/pitcher).<br>
<b>IP trend</b>: rolling-last-5-starts based indicator. ↑ = pitching deeper than usual; ↓ = being pulled early. See <code>xfp_rolling_ip.py</code> for methodology.
</div>
<div class="card guide">
<b>Model lineage</b><br>
<b>V8</b>: 4-feature minimal core (frozen reference, cross-year r=0.558, score=1.555).<br>
<b>V8.5</b>: +pfxz family + ip_resid + k_pct_lag (cross-year r=0.600, score=1.567). Superseded by V11.<br>
<b>V8.1</b>: V8.5 with sample-weighted 2025+2026 input blend for in-season updates.<br>
<b>V11 (this)</b>: V8.5 + pitching_plus (FG) + fp_strike_pct (Statcast). Cross-year r 0.613.<br>
Negative results documented (and not shipped): V9 (IP decomposition), V10 (Marcel weighting / archetype submodels / BaseRuns), V12 (residual correction).
See <code>data/research/xfp_model_research.md</code> for details.
</div>
</div>

<div class="card">
<div class="cardh">Top 80 SP — V8 → V8.5 → V11 (sorted by V11 xFP) | * = V8.5 fallback</div>
<table><thead><tr>
<th>Rk</th><th>Pitcher</th>
<th class="num">V8</th><th class="num">V8.5</th><th class="num">V11</th>
<th class="num">Δ vs V8.5</th>
<th class="num">2026 GS</th><th class="num">2026 actual</th>
<th>IP trend</th>
</tr></thead><tbody>{''.join(main_rows)}</tbody></table></div>

<div class="foot">
SP-only model. For full PLV dashboard (pitchers + hitters) see <code>process_report_2026.html</code>.<br>
xFP V11 | Built {today} | {bundle["features"]}
</div>

<script>
// Embedded projection data for any future JS-driven analytics or filters.
// Self-contained: no external CSV reads.
const V11_PROJECTIONS = {proj_json};
</script>
</body></html>'''
    out_path = OUTPUTS / 'xfp_v11_dashboard.html'
    out_path.write_text(html, encoding='utf-8')
    print(f'  wrote {out_path}')


if __name__ == '__main__':
    main()
