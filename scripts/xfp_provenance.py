"""
xfp_provenance.py - Source / model / as-of provenance for the V11 projection path.

Builds a machine-readable provenance record from the artifacts that are actually in the repo
(model bundle metadata, projection csv, training panel coverage, FanGraphs files, git HEAD) and
renders it as an HTML panel. The panel is embedded by xfp_v11_lock.build_dashboard and is also
written standalone so that any downstream dashboard generator can embed it.

Usage:
    python scripts/xfp_provenance.py            # writes data/outputs/xfp_v11_provenance.{json,html}

No network, no model fitting, no mutation of any other file.
"""
from __future__ import annotations
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules
from xfp_checks import (ROOT, FEATURE_AVAILABILITY, BLEND_SEMANTICS, evaluation_labels,
                        missing_sources, ytd_overlap_report)

CACHE   = ROOT / 'data' / 'research' / 'xfp_cache'
OUTPUTS = ROOT / 'data' / 'outputs'
MODELS  = ROOT / 'data' / 'models'


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _season_dates() -> dict:
    try:
        import build_sp_multiyr as b
        return {int(k): list(v) for k, v in b.SEASON_DATES.items()}
    except Exception:
        return {}


def build_provenance(root: Path = ROOT, bundle: dict | None = None,
                     proj: pd.DataFrame | None = None, panel: pd.DataFrame | None = None,
                     blend_proj: pd.DataFrame | None = None) -> dict:
    """Assemble the provenance record. Every argument can be injected (tests) or read from disk."""
    cache, outputs, models = root / 'data/research/xfp_cache', root / 'data/outputs', root / 'data/models'
    if bundle is None:
        import joblib
        bundle = joblib.load(models / 'xfp_v11_pipeline.pkl')
    if proj is None:
        proj = pd.read_csv(outputs / 'xfp_v11_projections.csv')
    if panel is None:
        panel = pd.read_csv(cache / 'sp_multiyr_2015_2025.csv')
    if blend_proj is None and (outputs / 'xfp_v8_5_projections.csv').exists():
        blend_proj = pd.read_csv(outputs / 'xfp_v8_5_projections.csv')

    feats = list(bundle.get('features', []))
    season_dates = _season_dates()
    p26 = panel[panel['year'] == 2026]
    yrs = sorted(int(y) for y in panel['year'].unique())

    fg_years = sorted(int(p.stem.rsplit('_', 1)[1]) for p in outputs.glob('fangraphs_pitchers_*.csv'))

    ytd_src = proj.copy()
    if blend_proj is not None and 'weight_2026' in blend_proj.columns:
        ytd_src = ytd_src.merge(blend_proj[['pitcher', 'cohort', 'weight_2026']], on='pitcher', how='left')
    ytd = ytd_overlap_report(ytd_src, p26) if len(p26) else {'n_ytd_eval': 0, 'label': 'n/a'}

    cohorts = (blend_proj['cohort'].value_counts().to_dict()
               if blend_proj is not None and 'cohort' in blend_proj.columns else {})

    inputs_as_of = season_dates.get(2026, [None, None])[1]
    metrics = {k: bundle.get(k) for k in ('cross_year_r', 'k_bias_hi', 'score_current',
                                          'score_tolerance_T1', 'ytd_r_2026', 'ytd_mae_2026', 'n_train')}
    metrics = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()}

    return {
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'git_head': _git_head(root),
        'model': {
            'version': bundle.get('version', 'v11'),
            'bundle_path': 'data/models/xfp_v11_pipeline.pkl',
            'trained_date': bundle.get('trained_date'),
            'training_years': bundle.get('training_years'),
            'n_train': bundle.get('n_train'),
            'features': feats,
            'estimator': 'StandardScaler -> RidgeCV(alphas=logspace(-1,5,80), cv=10), scaler fit on training rows only',
            'fallback': 'V8.5 (12 features, 2015-2025) when pitching_plus history is missing (v11_has_pitching_plus=False)',
        },
        'inputs': {
            'training_panel': {'path': 'data/research/xfp_cache/sp_multiyr_2015_2025.csv',
                               'years': yrs, 'rows': int(len(panel)),
                               'note': 'file name says 2015_2025 but it also carries the partial-2026 slice'},
            'season_windows': season_dates,
            'inputs_as_of': inputs_as_of,
            'season_2026_partial': {'rows': int(len(p26)),
                                    'max_gs': int(p26['gs'].max()) if len(p26) else None,
                                    'window_end': inputs_as_of},
            'fangraphs_years': fg_years,
            'blend': {'method': 'V8.1 mid-season blend (xfp_v8_midseason.blend_pitcher)',
                      'cohorts': {str(k): int(v) for k, v in cohorts.items()},
                      'per_feature_semantics': {f: BLEND_SEMANTICS.get(f, 'unknown') for f in feats}},
            'feature_availability': {f: FEATURE_AVAILABILITY.get(f) for f in feats},
            'missing_sources': missing_sources(root),
        },
        'projections': {'path': 'data/outputs/xfp_v11_projections.csv', 'rows': int(len(proj)),
                        'n_v11': int(proj['v11_has_pitching_plus'].sum()) if 'v11_has_pitching_plus' in proj else None,
                        'n_v85_fallback': int((~proj['v11_has_pitching_plus'].astype(bool)).sum()) if 'v11_has_pitching_plus' in proj else None,
                        'ytd_columns_source': 'data/outputs/xfp_v8_5_projections.csv (gs_2026, fp_per_start_actual_2026, k_pct_2026)'},
        'metrics': metrics,
        'evaluation_labels': evaluation_labels(),
        'ytd_overlap': ytd,
    }


def render_provenance_html(prov: dict) -> str:
    """Compact HTML fragment (self-contained styles) to embed in a dashboard."""
    m, i, mt, yo = prov['model'], prov['inputs'], prov['metrics'], prov['ytd_overlap']
    def kv(k, v): return f'<div class="pv-kv"><span class="pv-k">{k}</span><span class="pv-v">{v}</span></div>'
    def fmt(v, p=3):
        try: return f'{float(v):.{p}f}'
        except (TypeError, ValueError): return '-'
    sem_rows = ''.join(f'<tr><td>{f}</td><td>{(i["feature_availability"].get(f) or {}).get("source","?")}</td>'
                       f'<td>{(i["feature_availability"].get(f) or {}).get("first_year","?")}</td>'
                       f'<td>{i["blend"]["per_feature_semantics"].get(f,"?")}</td></tr>' for f in m['features'])
    missing = ''.join(f'<li>{s["path"]} — {s["role"]}</li>' for s in i['missing_sources'] if not s['present'])
    labels = prov['evaluation_labels']
    return f'''<div class="pv-panel" id="xfp-provenance">
<style>
.pv-panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin:14px 0;font-size:11.5px;color:#e6edf3}}
.pv-h{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:#8b949e;margin:0 0 8px}}
.pv-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.pv-kv{{display:flex;justify-content:space-between;gap:8px;padding:2px 0;border-bottom:1px solid #21262d}}
.pv-k{{color:#8b949e}}.pv-v{{font-weight:600;text-align:right}}
.pv-warn{{background:rgba(240,136,62,.1);border:1px solid rgba(240,136,62,.5);border-radius:6px;padding:8px 10px;margin-top:10px}}
.pv-panel table{{width:100%;border-collapse:collapse;font-size:11px;margin-top:8px}}
.pv-panel th,.pv-panel td{{text-align:left;padding:3px 5px;border-bottom:1px solid #21262d}}
.pv-panel ul{{margin:4px 0 0 16px;padding:0}}
</style>
<div class="pv-h">Provenance — source / model / as-of</div>
<div class="pv-grid">
<div>{kv('model', m['version'])}{kv('trained', m['trained_date'])}{kv('training seasons', m['training_years'])}
{kv('n_train', m['n_train'])}{kv('bundle', m['bundle_path'])}{kv('git HEAD', prov.get('git_head') or '-')}
{kv('provenance generated', prov['generated_utc'])}</div>
<div>{kv('inputs as-of (2026 window end)', i['inputs_as_of'] or '-')}{kv('2026 rows / max GS', f"{i['season_2026_partial']['rows']} / {i['season_2026_partial']['max_gs']}")}
{kv('training panel', i['training_panel']['path'])}{kv('panel seasons', f"{i['training_panel']['years'][0]}–{i['training_panel']['years'][-1]}")}
{kv('FanGraphs seasons', f"{i['fangraphs_years'][0]}–{i['fangraphs_years'][-1]}" if i['fangraphs_years'] else '-')}
{kv('blend cohorts', ', '.join(f'{k}={v}' for k, v in i['blend']['cohorts'].items()) or '-')}</div>
<div>{kv('cross-year r (point-in-time, 4 transitions)', fmt(mt.get('cross_year_r')))}{kv('k_bias_hi', fmt(mt.get('k_bias_hi')))}
{kv('2026 YTD r (CONTEMPORANEOUS)', fmt(mt.get('ytd_r_2026'), 3))}{kv('2026 YTD MAE (CONTEMPORANEOUS)', fmt(mt.get('ytd_mae_2026'), 3))}
{kv('YTD sample with own 2026 starts in inputs', f"{yo.get('n_inputs_include_2026_starts','-')}/{yo.get('n_ytd_eval','-')}")}</div>
</div>
<div class="pv-warn"><b>Read the YTD numbers as a fit check, not a forecast score.</b> {labels['ytd_r_2026']['caveat']}
<br><b>Cross-year r:</b> {labels['cross_year_r']['transitions']}. {labels['cross_year_r']['caveat']}</div>
<table><thead><tr><th>feature</th><th>source</th><th>first usable season</th><th>2026 input semantics (V8.1 blend)</th></tr></thead>
<tbody>{sem_rows}</tbody></table>
{('<div class="pv-warn"><b>Not reproducible from this repository (missing inputs):</b><ul>' + missing + '</ul></div>') if missing else ''}
</div>'''


def main():
    prov = build_provenance()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / 'xfp_v11_provenance.json').write_text(json.dumps(prov, indent=2, default=str), encoding='utf-8')
    html = ('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>xFP V11 provenance</title></head>'
            '<body style="background:#0d1117;margin:0;padding:18px;font-family:system-ui,sans-serif">'
            + render_provenance_html(prov) + '</body></html>')
    (OUTPUTS / 'xfp_v11_provenance.html').write_text(html, encoding='utf-8')
    print(f'wrote {OUTPUTS / "xfp_v11_provenance.json"}')
    print(f'wrote {OUTPUTS / "xfp_v11_provenance.html"}')
    print(json.dumps({'model': prov['model']['version'], 'trained': prov['model']['trained_date'],
                      'inputs_as_of': prov['inputs']['inputs_as_of'], 'ytd_overlap': prov['ytd_overlap']}, indent=1))


if __name__ == '__main__':
    main()
