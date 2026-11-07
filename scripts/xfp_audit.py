"""
xfp_audit.py - Reproducible evaluation/provenance audit of the V11 production projection path.

Reads only committed artifacts (training panel, pitch-type panels, FanGraphs csvs, model bundles,
projection csvs). Fits nothing that ships: the cross-year evaluations below re-fit the evaluation
Ridge exactly as the archived research did, once in the legacy leave-one-year-out ip_resid mode
(to reproduce the archived numbers) and once in point-in-time mode (the repaired default).

Writes:
    data/research/audit/xfp_eval_audit.json   machine-readable results
    data/research/audit/xfp_eval_audit.md     human-readable summary

Usage:  python scripts/xfp_audit.py          (~2 min, CPU only, no network)
"""
from __future__ import annotations
import json, sys, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import xfp_v7_pipeline as v7
import xfp_v8_pipeline as v8
import xfp_v8_5_pipeline as v85
import xfp_v8_midseason as ms
import xfp_checks as xc
from xfp_provenance import build_provenance

ROOT = xc.ROOT
CACHE, OUT, MODELS = ROOT / 'data/research/xfp_cache', ROOT / 'data/outputs', ROOT / 'data/models'
AUDIT_DIR = ROOT / 'data/research/audit'
V85_FEATS = ['avg_velo', 'zone_pct', 'o_swing_pct', 'swstr_pct', 'c_plus_swstr', 'xwoba_per_pa',
             'z_swing_pct', 'xwoba_x_swstr', 'ip_resid_lag1', 'k_pct_lag1', 'pitch_entropy', 'bb_pfxz']
V11_PP = V85_FEATS + ['pitching_plus']          # V11 minus fp_strike_pct (source file not in repo)


def load_panel(mode: str) -> pd.DataFrame:
    raw = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    d = v8.derive_v8_features(v7.add_ip_resid_lag(v7.derive_features(raw), mode=mode))
    yrs = sorted(raw['year'].unique())
    d = d.merge(v8.build_pitch_type_panel(yrs), on=['pitcher', 'year'], how='left')
    d = d.merge(v85.build_pfxz_panel(yrs), on=['pitcher', 'year'], how='left')
    fg = pd.concat([pd.read_csv(OUT / f'fangraphs_pitchers_{y}.csv').rename(columns={'mlb_id': 'pitcher'})
                    .assign(year=y)[['pitcher', 'year', 'pitching_plus']]
                    for y in range(2020, 2027) if (OUT / f'fangraphs_pitchers_{y}.csv').exists()])
    return d.merge(fg, on=['pitcher', 'year'], how='left')


def frozen_v8_ytd_gap() -> dict:
    import joblib
    b = joblib.load(MODELS / 'xfp_v8_pipeline.pkl'); F = b['features']
    raw = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    d25 = raw[raw.year == 2025].set_index('pitcher'); d26 = raw[raw.year == 2026].set_index('pitcher')
    rows = []
    for p in sorted(set(d25.index) & set(d26.index)):
        r25 = pd.Series({**d25.loc[p].to_dict(), 'pitcher': p}); r26 = pd.Series({**d26.loc[p].to_dict(), 'pitcher': p})
        pre, bl = r25.to_dict(), ms.blend_pitcher(r25, r26)
        for x in (pre, bl):
            x['xwoba_per_pa'] = x['xwoba_contact'] * x['bip_pct']; x['xwoba_x_swstr'] = x['xwoba_contact'] * x['swstr_pct']
        rows.append({'gs': r26['gs'], 'actual': r26['fp_per_start_actual'],
                     'pre': b['pipeline'].predict(pd.DataFrame([pre])[F])[0],
                     'blend': b['pipeline'].predict(pd.DataFrame([bl])[F])[0]})
    o = pd.DataFrame(rows); o = o[o.gs >= 5]
    return {'model': 'V8 frozen bundle (same weights for both rows)', 'n': int(len(o)),
            'preseason_inputs': {'r': round(float(np.corrcoef(o.pre, o.actual)[0, 1]), 4),
                                 'mae': round(float((o.actual - o.pre).abs().mean()), 3)},
            'blended_inputs_contemporaneous': {'r': round(float(np.corrcoef(o.blend, o.actual)[0, 1]), 4),
                                               'mae': round(float((o.actual - o.blend).abs().mean()), 3)}}


def main():
    t0 = datetime.now(timezone.utc)
    print('=' * 72); print(f'xFP EVALUATION / PROVENANCE AUDIT | {t0.isoformat(timespec="seconds")}'); print('=' * 72)
    res: dict = {'generated_utc': t0.strftime('%Y-%m-%dT%H:%M:%SZ')}

    # 1. sklearn version used to unpickle bundles
    import sklearn, joblib
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        b11 = joblib.load(MODELS / 'xfp_v11_pipeline.pkl')
        vers_warn = [str(x.message)[:200] for x in w if 'version' in str(x.message).lower()]
    res['bundle'] = {'sklearn_runtime': sklearn.__version__, 'unpickle_version_warnings': vers_warn,
                     **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in b11.items() if k not in ('pipeline', 'comparison')}}

    # 2. leaky vs point-in-time ip_resid
    panels = {m: load_panel(m) for m in ('loo', 'point_in_time')}
    m = panels['loo'][['pitcher', 'year', 'ip_resid_lag1']].merge(
        panels['point_in_time'][['pitcher', 'year', 'ip_resid_lag1']], on=['pitcher', 'year'], suffixes=('_loo', '_pit')).dropna()
    res['ip_resid_lag1'] = {'defect': 'IP model for season Y was fit on every OTHER season incl. future seasons and partial 2026',
                            'rows_compared': int(len(m)),
                            'corr_loo_vs_pit': round(float(np.corrcoef(m.ip_resid_lag1_loo, m.ip_resid_lag1_pit)[0, 1]), 4)}
    evals = {}
    for mode, d in panels.items():
        for lbl, feats, yrs in [('V8.5 (2015-2025)', V85_FEATS, (2015, 2025)), ('V11 minus fp_strike (2020-2025)', V11_PP, (2020, 2025))]:
            cy = v7.cross_year_evaluate(d[d['year'].between(*yrs)], feats, lbl)
            evals[f'{lbl} | ip_resid={mode}'] = {k: cy[k] for k in ('r', 'k_bias_hi', 'k_bias_lo', 'rmse', 'n', 'n_transitions')}
            print(f'  {lbl:32s} ip_resid={mode:14s} r={cy["r"]} k_bias_hi={cy["k_bias_hi"]} n={cy["n"]}')
    res['cross_year'] = evals
    stored85 = joblib.load(MODELS / 'xfp_v8_5_pipeline.pkl')['metrics']
    res['reproduction_check'] = {'stored_v85': {k: stored85[k] for k in ('r', 'k_bias_hi', 'n')},
                                 'recomputed_v85_loo': evals['V8.5 (2015-2025) | ip_resid=loo'],
                                 'match': abs(stored85['r'] - evals['V8.5 (2015-2025) | ip_resid=loo']['r']) < 5e-4}

    # 3. YTD provenance
    res['ytd_frozen_v8_gap'] = frozen_v8_ytd_gap()
    proj = pd.read_csv(OUT / 'xfp_v11_projections.csv'); raw = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    p85 = pd.read_csv(OUT / 'xfp_v8_5_projections.csv')
    res['ytd_v11_overlap'] = xc.ytd_overlap_report(proj.merge(p85[['pitcher', 'weight_2026']], on='pitcher', how='left'),
                                                   raw[raw.year == 2026])

    # 4. IDs + availability
    res['ids'] = {'unique_pitcher_year': xc.check_unique_pitcher_year(raw), 'integer_ids': xc.check_integer_ids(raw),
                  'fangraphs_alignment': [xc.check_fg_id_alignment(raw, pd.read_csv(OUT / f'fangraphs_pitchers_{y}.csv'), y)
                                          for y in range(2020, 2027)]}
    res['feature_availability'] = xc.check_feature_availability(panels['point_in_time'], V11_PP).to_dict(orient='records')

    # 5. blend semantics: how far 'latest_available' Pitching+ moves vs 2025 for blended pitchers
    fg25 = pd.read_csv(OUT / 'fangraphs_pitchers_2025.csv').rename(columns={'mlb_id': 'pitcher'})
    fg26 = pd.read_csv(OUT / 'fangraphs_pitchers_2026.csv').rename(columns={'mlb_id': 'pitcher'})
    both = (raw[raw.year == 2026][['pitcher', 'pitches']].merge(raw[raw.year == 2025][['pitcher', 'pitches']], on='pitcher', suffixes=('26', '25'))
            .merge(fg25[['pitcher', 'pitching_plus']], on='pitcher').merge(fg26[['pitcher', 'pitching_plus', 'ip']], on='pitcher', suffixes=('25', '26')))
    dpp = (both.pitching_plus26 - both.pitching_plus25).abs()
    res['blend_latest_available'] = {'blended_pitchers_with_both_pp': int(len(both)), 'abs_delta_pp_gt_5': int((dpp > 5).sum()),
                                     'abs_delta_pp_gt_10': int((dpp > 10).sum()), 'min_2026_ip': float(both.ip.min()),
                                     'mean_weight_2026_if_pitch_weighted': round(float((both.pitches26 / (both.pitches26 + both.pitches25)).mean()), 3)}

    # 6. provenance + docs dashboard cross-check (read-only; docs/index.html is generated elsewhere)
    res['provenance'] = build_provenance(ROOT, bundle=b11, proj=proj, panel=raw, blend_proj=p85)
    idx = ROOT / 'docs/index.html'
    if idx.exists():
        import re
        s = idx.read_text(encoding='utf-8')
        meta = json.loads(re.search(r'window\.XFP_META = (\{.*?\});\n', s, re.S).group(1))
        web = pd.DataFrame(json.loads(re.search(r'window\.XFP_PROJECTIONS = (\[.*?\]);\n', s, re.S).group(1)))
        mm = proj.merge(web[['mlbId', 'xfpV11']].rename(columns={'mlbId': 'pitcher'}), on='pitcher', how='left')
        res['docs_dashboard'] = {'generator': 'external (not in this repo); NOT edited by this audit',
                                 'meta_dataThru': meta.get('dataThru'), 'meta_trainedDate': meta.get('trainedDate'), 'meta_ytdR': meta.get('ytdR'),
                                 'repo_inputs_as_of': res['provenance']['inputs']['inputs_as_of'],
                                 'v11_values_identical_to_repo_csv': int(np.isclose(mm.xfp_v11.round(2), mm.xfpV11, atol=0.011).sum()),
                                 'repo_csv_rows': int(len(proj)), 'web_rows': int(len(web)),
                                 'web_rows_without_repo_v11': int((~web.mlbId.isin(proj.pitcher)).sum()),
                                 'has_provenance_panel': 'xfp-provenance' in s}

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / 'xfp_eval_audit.json').write_text(json.dumps(res, indent=2, default=str), encoding='utf-8')
    (AUDIT_DIR / 'xfp_eval_audit.md').write_text(render_md(res), encoding='utf-8')
    print(f'\nwrote {AUDIT_DIR / "xfp_eval_audit.json"}\nwrote {AUDIT_DIR / "xfp_eval_audit.md"}')
    print(f'elapsed {(datetime.now(timezone.utc) - t0).total_seconds():.0f}s')
    return res


def render_md(res: dict) -> str:
    L = ['# xFP evaluation / provenance audit (generated)', '',
         f'Generated {res["generated_utc"]} by `python scripts/xfp_audit.py`. Numbers are recomputed from committed artifacts; nothing is retrained or retuned.', '',
         '## Cross-year evaluation: legacy leave-one-year-out vs point-in-time ip_resid_lag1', '',
         '| model | ip_resid mode | r | k_bias_hi | k_bias_lo | rmse | n |', '|---|---|---|---|---|---|---|']
    for k, v in res['cross_year'].items():
        lbl, mode = k.split(' | ip_resid=')
        L.append(f'| {lbl} | {mode} | {v["r"]} | {v["k_bias_hi"]} | {v["k_bias_lo"]} | {v["rmse"]} | {v["n"]} |')
    rc = res['reproduction_check']
    mp = res['provenance']['metrics_provenance']['cross_year_r']
    L += ['', f'Stored bundle cross_year_r={mp["value"]}: status **{mp["status"]}** (ip_resid_mode={mp["ip_resid_mode"]}). Only the point_in_time rows above are point-in-time.']
    L += ['', f'Reproduction check: stored V8.5 bundle r={rc["stored_v85"]["r"]}, k_bias_hi={rc["stored_v85"]["k_bias_hi"]}, n={rc["stored_v85"]["n"]} '
              f'vs recomputed (loo) r={rc["recomputed_v85_loo"]["r"]} → match={rc["match"]}.',
          f'ip_resid_lag1 corr(loo, point-in-time) = {res["ip_resid_lag1"]["corr_loo_vs_pit"]} over {res["ip_resid_lag1"]["rows_compared"]} rows.', '',
          '## 2026 YTD metric provenance', '']
    g = res['ytd_frozen_v8_gap']
    L += [f'Frozen V8 bundle, n={g["n"]} (gs>=5): pre-season inputs r={g["preseason_inputs"]["r"]} MAE={g["preseason_inputs"]["mae"]} | '
          f'blended (contemporaneous) inputs r={g["blended_inputs_contemporaneous"]["r"]} MAE={g["blended_inputs_contemporaneous"]["mae"]}.',
          f'V11 stored YTD sample: {res["ytd_v11_overlap"]}', '',
          '## IDs / availability', '',
          f'unique pitcher-year: {res["ids"]["unique_pitcher_year"]["ok"]}; integer ids: {res["ids"]["integer_ids"]["ok"]}', '',
          'Name status is evidence of consistency, never verified identity (identity_verified_by_name=False). ambiguous = given-name / suffix conflict or same-surname pool; resolve by id.', '',
          '| FG season | panel rows | matched | consistent | ambiguous | mismatch | ok |', '|---|---|---|---|---|---|---|']
    for a in res['ids']['fangraphs_alignment']:
        L.append(f'| {a["year"]} | {a.get("n_panel")} | {a.get("n_matched")} | {a.get("n_consistent")} | {a.get("n_ambiguous")} | {a.get("n_name_mismatch")} | {a["ok"]} |')
    amb = [(a['year'], x) for a in res['ids']['fangraphs_alignment'] for x in a.get('ambiguous', [])]
    if amb:
        L += ['', 'Ambiguous rows (id, panel name, FG name):'] + [f'- {y}: {x}' for y, x in amb]
    L += ['', '| feature | declared first season | observed first season | violation |', '|---|---|---|---|']
    for f in res['feature_availability']:
        L.append(f'| {f["feature"]} | {f["declared_first_year"]} | {f["observed_first_year"]} | {f["violation"]} |')
    L += ['', '## Blend: latest-available (non-weighted) Pitching+', '', f'{res["blend_latest_available"]}', '',
          '## Model bundle', '', f'{res["bundle"]}', '']
    if 'docs_dashboard' in res:
        L += ['## docs/index.html cross-check (read-only)', '', f'{res["docs_dashboard"]}', '']
    L += ['## Missing sources', ''] + [f'- `{s["path"]}` present={s["present"]}: {s["role"]}' for s in res['provenance']['inputs']['missing_sources']]
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
