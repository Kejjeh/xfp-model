"""Provenance record and dashboard rendering (synthetic bundle/projections; no model fitting)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest

from conftest import make_panel, REPO
import xfp_provenance as xp
import xfp_v11_lock as lock


class _FakePipe:
    """Stands in for the sklearn Pipeline: only .named_steps['r'].coef_ is read by the dashboard."""
    class _R:
        coef_ = np.linspace(-1, 1, 14)
    named_steps = {'r': _R()}


def _bundle():
    return {'pipeline': _FakePipe(), 'features': list(xp.FEATURE_AVAILABILITY), 'cross_year_r': 0.61381,
            'k_bias_hi': 0.773, 'score_current': 1.455, 'score_tolerance_T1': 1.841, 'trained_date': '2026-05-05',
            'n_train': 768, 'version': 'v11', 'training_years': '2020-2025', 'ytd_mae_2026': 3.393,
            'ytd_r_2026': 0.511, 'formula': 'x'}


def _proj(n=30):
    rng = np.random.default_rng(1)
    return pd.DataFrame({'pitcher': 500000 + np.arange(n), 'player_name': [f'Last{i}, First' for i in range(n)],
                         'xfp_v8_5': rng.normal(11, 2, n), 'xfp_v11': rng.normal(11, 2, n),
                         'v11_has_pitching_plus': [True] * (n - 3) + [False] * 3,
                         'xfp_v8': rng.normal(11, 2, n), 'gs_2026': rng.integers(0, 8, n).astype(float),
                         'fp_per_start_actual_2026': rng.normal(10, 4, n), 'k_pct_2026': rng.uniform(0.1, 0.4, n),
                         'ip_trend': 'NORMAL'}).assign(delta_v11_v85=lambda d: d.xfp_v11 - d.xfp_v8_5)


def test_provenance_record_has_source_model_asof(tmp_path):
    panel = make_panel()
    prov = xp.build_provenance(REPO, bundle=_bundle(), proj=_proj(), panel=panel,
                               blend_proj=pd.DataFrame({'pitcher': [], 'cohort': [], 'weight_2026': []}))
    assert prov['model']['trained_date'] == '2026-05-05'
    assert prov['model']['training_years'] == '2020-2025'
    assert prov['inputs']['training_panel']['years'][0] == 2015 and prov['inputs']['training_panel']['years'][-1] == 2026
    assert prov['inputs']['inputs_as_of'] == '2026-05-04'           # SEASON_DATES[2026] window end
    assert prov['evaluation_labels']['ytd_r_2026']['point_in_time'] is False
    # the synthetic bundle has no ip_resid_mode -> archived legacy construction, NOT point-in-time
    assert prov['evaluation_labels']['cross_year_r']['point_in_time'] is False
    assert prov['metrics_provenance']['cross_year_r']['status'].startswith('archived_legacy_loo')
    prov_pit = xp.build_provenance(REPO, bundle={**_bundle(), 'ip_resid_mode': 'point_in_time'}, proj=_proj(), panel=panel, blend_proj=pd.DataFrame())
    assert prov_pit['metrics_provenance']['cross_year_r']['point_in_time'] is True
    assert prov['inputs']['blend']['per_feature_semantics']['pitching_plus'].startswith('latest_available')
    assert any(s['path'].endswith('fp_strike_2015_2026.csv') and not s['present'] for s in prov['inputs']['missing_sources'])
    json.dumps(prov, default=str)                                   # serialisable


def test_provenance_html_states_contemporaneous_and_asof():
    prov = xp.build_provenance(REPO, bundle=_bundle(), proj=_proj(), panel=make_panel(), blend_proj=pd.DataFrame())
    html = xp.render_provenance_html(prov)
    for needle in ['CONTEMPORANEOUS', '2026-05-04', '2026-05-05', 'xfp_v11_pipeline.pkl', 'latest_available',
                   'fp_strike_2015_2026.csv', '4; NOT 9', 'NOT point-in-time', 'archived_legacy_loo']:
        assert needle in html, needle
    assert 'cross-year r, 4 transitions (point-in-time)' not in html   # archived bundle score is never headlined as PIT


def test_dashboard_embeds_provenance_panel(tmp_path):
    prov = xp.build_provenance(REPO, bundle=_bundle(), proj=_proj(), panel=make_panel(), blend_proj=pd.DataFrame())
    out = tmp_path / 'dash.html'
    lock.build_dashboard(_proj(), _bundle(), 3.48, 0.475, 3.39, 0.511, provenance=prov, out_path=out)
    html = out.read_text(encoding='utf-8')
    assert 'id="xfp-provenance"' in html
    assert 'YTD r (contemporaneous)' in html
    assert '2026 inputs as-of 2026-05-04' in html
    assert 'Model trained 2026-05-05' in html
    assert 'Cross-year r 0.61381 (archived_legacy_loo' in html


def test_missing_sources_lists_fp_strike():
    ms = {s['path']: s['present'] for s in xp.missing_sources(REPO)}
    assert 'data/research/xfp_cache/fp_strike_2015_2026.csv' in ms


def test_missing_sources_paths_are_posix_on_every_platform(monkeypatch):
    """Control for the Windows portability failure: serialized report paths must use '/' regardless of
    the host separator, so the documented output (and the tests that read it) are identical everywhere."""
    import pathlib
    for s in xp.missing_sources(REPO):
        assert '\\' not in s['path'] and s['path'] == pathlib.PurePosixPath(s['path']).as_posix()
    # simulate a Windows-style root: PureWindowsPath arithmetic must still serialize as POSIX
    win_root = pathlib.PureWindowsPath('C:/repo')
    rel = (win_root / 'data' / 'research' / 'xfp_cache' / 'fp_strike_2015_2026.csv').relative_to(win_root)
    assert str(rel) == 'data\\research\\xfp_cache\\fp_strike_2015_2026.csv'   # what str() gives on Windows
    assert rel.as_posix() == 'data/research/xfp_cache/fp_strike_2015_2026.csv'    # what the report must emit

