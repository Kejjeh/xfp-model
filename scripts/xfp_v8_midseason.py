"""
xfp_v8_midseason.py - V8.1 mid-season update.

Sample-weighted blend of 2025 + 2026 input features per pitcher, fed into the FROZEN
V8 pipeline. Saves data/outputs/xfp_v8_1_projections.csv + dashboard.

Cohorts:
  Blended (has 2025 + 2026): pitch-count weighted blend
  2026-only (rookies/returners): use 2026 alone if pitches >= 200, else fallback
  2025-only: keep V8 prediction
"""
from __future__ import annotations
import sys, joblib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'xfp'))

CACHE   = ROOT / 'data' / 'research' / 'xfp_cache'
OUTPUTS = ROOT / 'data' / 'outputs'
MODELS  = ROOT / 'data' / 'models'
RESEARCH= ROOT / 'data' / 'research'

PRIOR_N, PRIOR_MEAN = 40, 0.3117
ROOKIE_MIN_PITCHES = 200

RATE_FEATS = ['swstr_pct','c_plus_swstr','zone_pct','o_swing_pct','z_swing_pct',
               'avg_velo','avg_ext','abs_pfxz','bip_pct']

def blend_pitcher(row25: pd.Series | None, row26: pd.Series | None) -> dict | None:
    """Blend a pitcher's 2025 and 2026 metrics. Returns dict of blended values."""
    has25 = row25 is not None
    has26 = row26 is not None
    if not has25 and not has26:
        return None
    if has26 and not has25:
        if row26['pitches'] < ROOKIE_MIN_PITCHES:
            return None  # too small a 2026 sample for rookie projection
        out = row26.to_dict()
        out['cohort']  = '2026_only'
        out['n_pitches_2025'] = 0
        out['n_pitches_2026'] = float(row26['pitches'])
        out['weight_2026']    = 1.0
        return out
    if has25 and not has26:
        out = row25.to_dict()
        out['cohort']  = '2025_only'
        out['n_pitches_2025'] = float(row25['pitches'])
        out['n_pitches_2026'] = 0
        out['weight_2026']    = 0.0
        return out
    # Both present — blend
    n25 = float(row25['pitches']); n26 = float(row26['pitches'])
    out = {'pitcher': row25['pitcher'], 'player_name': row25.get('player_name', row26.get('player_name'))}
    for f in RATE_FEATS:
        v25 = row25.get(f); v26 = row26.get(f)
        if pd.isna(v25) and pd.isna(v26):
            out[f] = np.nan
        elif pd.isna(v25):
            out[f] = float(v26)
        elif pd.isna(v26):
            out[f] = float(v25)
        else:
            out[f] = (n25 * v25 + n26 * v26) / (n25 + n26)
    # Bayesian xwoba_contact (two-sample with prior)
    bip25 = float(row25.get('bip', 0)); bip26 = float(row26.get('bip', 0))
    x25 = float(row25.get('xwoba_contact', PRIOR_MEAN)); x26 = float(row26.get('xwoba_contact', PRIOR_MEAN))
    if pd.isna(x25): x25 = PRIOR_MEAN
    if pd.isna(x26): x26 = PRIOR_MEAN
    out['xwoba_contact'] = ((bip25 * x25) + (bip26 * x26) + (PRIOR_N * PRIOR_MEAN)) / (bip25 + bip26 + PRIOR_N)
    out['bip'] = bip25 + bip26
    out['cohort'] = 'blended'
    out['n_pitches_2025'] = n25
    out['n_pitches_2026'] = n26
    out['weight_2026']    = n26 / (n25 + n26) if (n25 + n26) > 0 else 0
    return out


def main():
    print('=' * 60)
    print(f'xFP V8.1 MID-SEASON UPDATE | {datetime.now(timezone.utc).isoformat()}')
    print('=' * 60)

    df = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    df_25 = df[df['year']==2025].set_index('pitcher')
    df_26 = df[df['year']==2026].set_index('pitcher')
    print(f'2025 SPs: {len(df_25)}, 2026 SPs: {len(df_26)}')

    pitchers_union = sorted(set(df_25.index) | set(df_26.index))
    print(f'Union: {len(pitchers_union)} unique pitchers')

    # Build blended rows
    blended_rows = []
    cohort_counts = {'blended': 0, '2026_only': 0, '2025_only': 0, 'fallback': 0}
    for p in pitchers_union:
        r25 = df_25.loc[p].to_dict() if p in df_25.index else None
        r26 = df_26.loc[p].to_dict() if p in df_26.index else None
        if r25 is not None:
            r25 = pd.Series({**r25, 'pitcher': p})
        if r26 is not None:
            r26 = pd.Series({**r26, 'pitcher': p})
        blended = blend_pitcher(r25, r26)
        if blended is None:
            cohort_counts['fallback'] += 1
            continue
        cohort_counts[blended['cohort']] += 1
        blended_rows.append(blended)

    print(f'\nCohort counts: Blended={cohort_counts["blended"]} | '
          f'2026-only={cohort_counts["2026_only"]} | '
          f'2025-only={cohort_counts["2025_only"]} | '
          f'fallback={cohort_counts["fallback"]}')
    blended_df = pd.DataFrame(blended_rows)

    # Recompute V8 interactions from blended primitives
    blended_df['xwoba_per_pa']  = blended_df['xwoba_contact'] * blended_df['bip_pct']
    blended_df['xwoba_x_swstr'] = blended_df['xwoba_contact'] * blended_df['swstr_pct']

    # Predict with frozen V8 pipeline
    bundle = joblib.load(MODELS / 'xfp_v8_pipeline.pkl')
    V8_FEATS = bundle['features']
    print(f'V8 features: {V8_FEATS}')
    valid = blended_df.dropna(subset=V8_FEATS).copy()
    valid['xfp_v8_1'] = bundle['pipeline'].predict(valid[V8_FEATS])
    print(f'V8.1 predictions: {len(valid)} of {len(blended_df)} rows')

    # Merge with V8 baseline projections
    v8_proj = pd.read_csv(OUTPUTS / 'xfp_v8_projections.csv')
    out = valid[['pitcher','player_name','xfp_v8_1','cohort','n_pitches_2025','n_pitches_2026','weight_2026']].merge(
        v8_proj[['pitcher','xfp_v8','xfp_v7','xfp_v6','xfp_v5','gs_2026','fp_per_start_actual_2026','k_pct_2026']],
        on='pitcher', how='left')
    out['delta_v8_1_v8'] = out['xfp_v8_1'] - out['xfp_v8']

    # For pitchers in V8 but not in V8.1 (e.g., low-pitches rookies who fell back), keep V8
    fallback_pitchers = set(v8_proj['pitcher']) - set(out['pitcher'])
    if fallback_pitchers:
        fallback_rows = v8_proj[v8_proj['pitcher'].isin(fallback_pitchers)].copy()
        fallback_rows['xfp_v8_1']         = fallback_rows['xfp_v8']
        fallback_rows['cohort']           = 'v8_fallback'
        fallback_rows['n_pitches_2025']   = np.nan
        fallback_rows['n_pitches_2026']   = np.nan
        fallback_rows['weight_2026']      = 0.0
        fallback_rows['delta_v8_1_v8']    = 0.0
        out = pd.concat([out, fallback_rows], ignore_index=True, sort=False)
        print(f'  fallback (kept V8): {len(fallback_pitchers)}')

    out_path = OUTPUTS / 'xfp_v8_1_projections.csv'
    cols = ['pitcher','player_name','cohort','xfp_v5','xfp_v6','xfp_v7','xfp_v8','xfp_v8_1','delta_v8_1_v8',
            'n_pitches_2025','n_pitches_2026','weight_2026','gs_2026','fp_per_start_actual_2026','k_pct_2026']
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(out_path, index=False)
    print(f'  wrote {out_path}')

    # Spot check
    print('\nSPOT CHECK:')
    for n in ['Schlittler','Glasnow','Imanaga','Fried','Woodruff','Ragans']:
        r = out[out['player_name'].fillna('').str.contains(n, na=False)]
        if len(r):
            row = r.iloc[0]
            print(f'  {n:<13s} V8={row["xfp_v8"]:.2f}  V8.1={row["xfp_v8_1"]:.2f}  '
                  f'delta={row["delta_v8_1_v8"]:+.2f}  w26={row["weight_2026"]:.3f}  '
                  f'actual={row["fp_per_start_actual_2026"] if pd.notna(row["fp_per_start_actual_2026"]) else "-":}'
                  f'  cohort={row["cohort"]}')

    # YTD evaluation (gs >= 5)
    ytd = out[(out['gs_2026'] >= 5) & out['fp_per_start_actual_2026'].notna()
              & out['xfp_v8'].notna() & out['xfp_v8_1'].notna()].copy()
    r_v8   = float(np.corrcoef(ytd['xfp_v8'],   ytd['fp_per_start_actual_2026'])[0,1]) if len(ytd) >= 10 else None
    r_v8_1 = float(np.corrcoef(ytd['xfp_v8_1'], ytd['fp_per_start_actual_2026'])[0,1]) if len(ytd) >= 10 else None
    high_k = ytd[ytd['k_pct_2026'] > 0.30]
    kb_v8   = float((high_k['fp_per_start_actual_2026'] - high_k['xfp_v8']).mean())   if len(high_k) else None
    kb_v8_1 = float((high_k['fp_per_start_actual_2026'] - high_k['xfp_v8_1']).mean()) if len(high_k) else None
    print(f'\nYTD r (n={len(ytd)}): V8={r_v8:.5f}  V8.1={r_v8_1:.5f}  delta={r_v8_1-r_v8:+.5f}')
    print(f'YTD k_bias (n_high_k={len(high_k)}): V8={kb_v8:+.3f}  V8.1={kb_v8_1:+.3f}')

    # Decision
    sch_row = out[out['player_name'].fillna('').str.contains('Schlittler', na=False)]
    gla_row = out[out['player_name'].fillna('').str.contains('Glasnow', na=False)]
    sch_v81 = float(sch_row.iloc[0]['xfp_v8_1']) if len(sch_row) else None
    gla_v81 = float(gla_row.iloc[0]['xfp_v8_1']) if len(gla_row) else None
    pass_r = (r_v8_1 is not None) and (r_v8_1 >= r_v8 - 0.005)  # tolerate tiny noise
    pass_sch = sch_v81 is not None and sch_v81 > 12.0
    pass_gla = gla_v81 is not None and gla_v81 > 14.5
    decision = 'PASS' if (pass_r and pass_sch and pass_gla) else 'PARTIAL/FAIL'
    print(f'\nDECISION: {decision}')
    print(f'  YTD r >= V8: {pass_r}  ({r_v8_1:.5f} vs {r_v8:.5f})')
    print(f'  Schlittler > 12.0: {pass_sch}  ({sch_v81:.2f})')
    print(f'  Glasnow > 14.5: {pass_gla}  ({gla_v81:.2f})')

    # Build dashboard
    build_dashboard(out, r_v8, r_v8_1, kb_v8, kb_v8_1, decision)

    # Append to research notes
    append_research(out, r_v8, r_v8_1, kb_v8, kb_v8_1, decision)

    print('\nWORKSTREAM 1 COMPLETE')
    return out, r_v8, r_v8_1, decision


def build_dashboard(out, r_v8, r_v8_1, kb_v8, kb_v8_1, decision):
    """Render xfp_v8_1_dashboard.html."""
    out_d = out.sort_values('xfp_v8_1', ascending=False, na_position='last').reset_index(drop=True)
    out_d['rank_v8_1'] = out_d.index + 1
    out_d['rank_v8']   = out_d['xfp_v8'].rank(ascending=False, method='min')

    # biggest movers
    moved = out_d.dropna(subset=['delta_v8_1_v8']).copy()
    top_up = moved.nlargest(10, 'delta_v8_1_v8')
    top_dn = moved.nsmallest(5, 'delta_v8_1_v8')

    def fmt(x, p=2):
        try:
            f = float(x)
            if not np.isfinite(f): return '-'
            return f'{f:.{p}f}'
        except (TypeError, ValueError):
            return '-'

    def archetype_card(name):
        r = out_d[out_d['player_name'].fillna('').str.contains(name, na=False)]
        if not len(r):
            return f'<div class="card"><div class="cardh">{name}</div><div>Not in projection set</div></div>'
        s = r.iloc[0]
        delta_color = '#3fb950' if s['delta_v8_1_v8']>0 else '#f85149' if s['delta_v8_1_v8']<0 else '#8b949e'
        return (f'<div class="card"><div class="cardh">{s["player_name"]}</div>'
                f'<div class="kv"><span class="kv-k">V8 xFP</span><span class="kv-v">{fmt(s["xfp_v8"])}</span></div>'
                f'<div class="kv"><span class="kv-k">V8.1 xFP</span><span class="kv-v" style="color:#3fb950">{fmt(s["xfp_v8_1"])}</span></div>'
                f'<div class="kv"><span class="kv-k">delta</span><span class="kv-v" style="color:{delta_color}">{s["delta_v8_1_v8"]:+.2f}</span></div>'
                f'<div class="kv"><span class="kv-k">2026 actual</span><span class="kv-v">{fmt(s["fp_per_start_actual_2026"])}</span></div>'
                f'<div class="kv"><span class="kv-k">w_2026</span><span class="kv-v">{fmt(s["weight_2026"],3)}</span></div>'
                f'</div>')

    archetype_html = ''.join(archetype_card(n) for n in
        ['Schlittler','Glasnow','Imanaga','Fried','Woodruff','Ragans'])

    def mover_row(s):
        cl = 'up' if s['delta_v8_1_v8']>0 else 'dn'
        return (f'<tr><td>{s["player_name"]}</td>'
                f'<td class="num">{int(s["rank_v8"]) if pd.notna(s["rank_v8"]) else "-"}</td>'
                f'<td class="num">{int(s["rank_v8_1"])}</td>'
                f'<td class="num">{fmt(s["xfp_v8"])}</td>'
                f'<td class="num"><b>{fmt(s["xfp_v8_1"])}</b></td>'
                f'<td class="num {cl}">{s["delta_v8_1_v8"]:+.2f}</td>'
                f'<td class="num">{fmt(s["weight_2026"],3)}</td>'
                f'<td class="num">{fmt(s["fp_per_start_actual_2026"])}</td></tr>')
    movers_up = '\n'.join(mover_row(s) for _, s in top_up.iterrows())
    movers_dn = '\n'.join(mover_row(s) for _, s in top_dn.iterrows())

    # Top 60 main table
    main_rows = []
    for _, s in out_d.head(80).iterrows():
        try: dv = float(s['delta_v8_1_v8']); dvc = 'up' if dv>0 else 'dn' if dv<0 else ''
        except: dvc=''; dv=0
        cls = 't1' if s['rank_v8_1']==1 else 't2' if s['rank_v8_1']==2 else 't3' if s['rank_v8_1']==3 else ''
        gold = ' style="color:#ffd700"' if 'Schlittler' in str(s['player_name']) else ''
        main_rows.append(
            f'<tr><td class="{cls}">{int(s["rank_v8_1"])}</td><td{gold}>{s["player_name"]}</td>'
            f'<td class="num">{fmt(s["xfp_v8"])}</td>'
            f'<td class="num"><b>{fmt(s["xfp_v8_1"])}</b></td>'
            f'<td class="num {dvc}">{dv:+.2f}</td>'
            f'<td class="num">{fmt(s["weight_2026"],3)}</td>'
            f'<td class="num">{fmt(s["gs_2026"],0)}</td>'
            f'<td class="num">{fmt(s["fp_per_start_actual_2026"])}</td></tr>')
    main_html = '\n'.join(main_rows)

    decision_color = '#3fb950' if decision=='PASS' else '#f0883e'

    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>xFP v8.1 mid-season update</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;padding:18px}}
.hdr{{background:linear-gradient(135deg,#1a2332,#0d1b2a);border:1px solid #30363d;border-radius:8px;padding:14px 18px;margin-bottom:14px}}
.title{{font-size:20px;font-weight:700;color:#58a6ff}}.title span{{color:#f0883e}}
.sub{{font-size:11.5px;color:#8b949e;margin-top:4px;line-height:1.5}}
.banner{{background:rgba(63,185,80,.08);border:1px solid rgba(63,185,80,.4);border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:12px}}
.card{{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px}}
.cardh{{font-size:11px;font-weight:700;text-transform:uppercase;color:#8b949e;letter-spacing:.7px;margin-bottom:9px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}}
.grid6{{display:grid;grid-template-columns:repeat(6, minmax(0, 1fr));gap:10px;margin-bottom:14px}}
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
<div class="title">xFP <span>v8.1</span> Mid-Season Update</div>
<div class="sub">Sample-weighted blend of 2025 + 2026 input metrics fed into the frozen V8 pipeline.
V8 cross-year r=0.558, k_bias=0.241, score=1.555. V8.1 retains the V8 model coefficients but updates each pitcher's
input features with their 2026 YTD data (sample-weighted by pitch count). Schlittler/Glasnow/Imanaga targeted.</div>
</div>

<div class="banner">
<b>Validation panel:</b> YTD r V8 = {fmt(r_v8,5)} → V8.1 = {fmt(r_v8_1,5)} (Δ {(r_v8_1 or 0)-(r_v8 or 0):+.5f}) ·
YTD high-K bias V8 = {fmt(kb_v8,3)} → V8.1 = {fmt(kb_v8_1,3)} ·
<span style="color:{decision_color};font-weight:700">DECISION: {decision}</span>
</div>

<div class="card" style="margin-bottom:14px"><div class="cardh">Six archetype callouts (Breakdown 1)</div>
<div class="grid6">{archetype_html}</div></div>

<div class="grid">
<div class="card"><div class="cardh">Biggest movers UP (V8 → V8.1)</div>
<table><thead><tr><th>Pitcher</th><th class="num">V8 rk</th><th class="num">V8.1 rk</th><th class="num">V8</th><th class="num">V8.1</th><th class="num">Δ</th><th class="num">w26</th><th class="num">Actual</th></tr></thead><tbody>{movers_up}</tbody></table></div>
<div class="card"><div class="cardh">Biggest movers DOWN</div>
<table><thead><tr><th>Pitcher</th><th class="num">V8 rk</th><th class="num">V8.1 rk</th><th class="num">V8</th><th class="num">V8.1</th><th class="num">Δ</th><th class="num">w26</th><th class="num">Actual</th></tr></thead><tbody>{movers_dn}</tbody></table></div>
</div>

<div class="card">
<div class="cardh">Top 80 SP — V8 vs V8.1 side by side (Schlittler row in gold)</div>
<table><thead><tr><th>Rk V8.1</th><th>Pitcher</th><th class="num">V8 xFP</th><th class="num">V8.1 xFP</th><th class="num">Δ</th><th class="num">w_2026</th><th class="num">2026 GS</th><th class="num">2026 actual</th></tr></thead>
<tbody>{main_html}</tbody></table></div>
</body></html>'''
    out_path = OUTPUTS / 'xfp_v8_1_dashboard.html'
    out_path.write_text(html, encoding='utf-8')
    print(f'  wrote {out_path}')


def append_research(out, r_v8, r_v8_1, kb_v8, kb_v8_1, decision):
    sch = out[out['player_name'].fillna('').str.contains('Schlittler', na=False)]
    six_lines = []
    for n in ['Schlittler','Glasnow','Imanaga','Fried','Woodruff','Ragans']:
        r = out[out['player_name'].fillna('').str.contains(n, na=False)]
        if len(r):
            s = r.iloc[0]
            six_lines.append(f"- {s['player_name']:<22} V8={s['xfp_v8']:.2f}  V8.1={s['xfp_v8_1']:.2f}  "
                              f"Δ={s['delta_v8_1_v8']:+.2f}  w_2026={s['weight_2026']:.3f}  "
                              f"actual_2026={s['fp_per_start_actual_2026'] if pd.notna(s['fp_per_start_actual_2026']) else 'n/a'}")

    section = f"""

## V8.1 — Mid-Season Update ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})

V8 model frozen. Re-projected with sample-weighted blends of 2025 + 2026 input metrics per pitcher.
Rate metrics blend by total pitches; xwoba_contact uses two-sample Bayesian shrinkage.

### Cohorts
- Blended (has both 2025 & 2026 SP rows): pitch-count weighted blend
- 2026-only: use 2026 alone if pitches >= {ROOKIE_MIN_PITCHES}, else fall back to V8
- 2025-only: keep V8 prediction unchanged

### Validation
| Metric | V8 | V8.1 | Δ |
|---|---|---|---|
| 2026 YTD r (gs>=5) | {r_v8:.5f} | {r_v8_1:.5f} | {(r_v8_1 or 0)-(r_v8 or 0):+.5f} |
| YTD k_bias_hi | {kb_v8 if kb_v8 is not None else 'n/a':.3f} | {kb_v8_1 if kb_v8_1 is not None else 'n/a':.3f} | {(kb_v8_1 or 0)-(kb_v8 or 0):+.3f} |

**Decision: {decision}**

### Six target archetype callouts
""" + '\n'.join(six_lines) + """

### Files
- `data/outputs/xfp_v8_1_projections.csv`
- `data/outputs/xfp_v8_1_dashboard.html`
"""
    research_md = RESEARCH / 'xfp_model_research.md'
    with open(research_md, 'a', encoding='utf-8') as f:
        f.write(section)
    print(f'  appended V8.1 section to {research_md}')


if __name__ == '__main__':
    main()
