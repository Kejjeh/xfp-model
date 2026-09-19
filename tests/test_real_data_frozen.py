"""Frozen-evaluation regression guards on the REAL cached panel (skipped when the data is absent).

These pin the archived numbers so that any change to the evaluation path is visible. They do not
retrain the production model and do not claim any improvement. Run time: ~1-2 minutes.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest

from conftest import requires_real_data, REPO, CACHE
import xfp_v7_pipeline as v7
import xfp_v8_pipeline as v8
import xfp_v8_5_pipeline as v85
import xfp_v8_midseason as ms
import xfp_checks as xc

warnings.filterwarnings('ignore')
pytestmark = pytest.mark.slow
V85_FEATS = ['avg_velo', 'zone_pct', 'o_swing_pct', 'swstr_pct', 'c_plus_swstr', 'xwoba_per_pa',
             'z_swing_pct', 'xwoba_x_swstr', 'ip_resid_lag1', 'k_pct_lag1', 'pitch_entropy', 'bb_pfxz']
OUT = REPO / 'data' / 'outputs'


def _load(mode):
    raw = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    d = v8.derive_v8_features(v7.add_ip_resid_lag(v7.derive_features(raw), mode=mode))
    yrs = sorted(raw['year'].unique())
    d = d.merge(v8.build_pitch_type_panel(yrs), on=['pitcher', 'year'], how='left')
    d = d.merge(v85.build_pfxz_panel(yrs), on=['pitcher', 'year'], how='left')
    return d


@requires_real_data
def test_archived_v85_metrics_reproduce_under_legacy_loo_mode():
    """The stored V8.5 bundle metrics (r 0.59993, k_bias 0.466, n 390) were computed with the leaky
    leave-one-year-out ip_resid. Reproducing them exactly proves the harness is faithful."""
    import joblib
    stored = joblib.load(REPO / 'data/models/xfp_v8_5_pipeline.pkl')['metrics']
    res = v7.cross_year_evaluate(_load('loo'), V85_FEATS)
    assert res['n'] == stored['n'] == 390
    assert abs(res['r'] - stored['r']) < 5e-4
    assert abs(res['k_bias_hi'] - stored['k_bias_hi']) < 5e-3


@requires_real_data
def test_point_in_time_v85_metrics_are_pinned():
    """Point-in-time rebuild of ip_resid_lag1 (the repaired default). Pinned, not 'improved'."""
    res = v7.cross_year_evaluate(_load('point_in_time'), V85_FEATS)
    assert res['n'] == 389
    assert abs(res['r'] - 0.60027) < 5e-4
    assert abs(res['k_bias_hi'] - 0.412) < 5e-3


@requires_real_data
def test_real_panel_ids_and_availability():
    raw = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    assert xc.check_unique_pitcher_year(raw)['ok']
    assert xc.check_integer_ids(raw)['ok']
    for y in range(2020, 2027):
        fg = pd.read_csv(OUT / f'fangraphs_pitchers_{y}.csv')
        res = xc.check_fg_id_alignment(raw, fg, y)
        assert res['ok'], res                                   # no surname mismatch under a strict integer join
        assert res['n_matched'] >= 0.85 * res['n_panel'], res
        assert res['identity_verified_by_name'] is False
        assert res['n_consistent'] + res['n_ambiguous'] == res['n_matched']
    d = _load('point_in_time')
    fg_all = pd.concat([pd.read_csv(OUT / f'fangraphs_pitchers_{y}.csv').rename(columns={'mlb_id': 'pitcher'})
                        .assign(year=y)[['pitcher', 'year', 'pitching_plus']] for y in range(2020, 2027)])
    d = d.merge(fg_all, on=['pitcher', 'year'], how='left')
    rep = xc.check_feature_availability(d, V85_FEATS + ['pitching_plus'])
    assert not rep['violation'].any(), rep


@requires_real_data
def test_frozen_v8_preseason_vs_contemporaneous_ytd_gap_is_pinned():
    """Same frozen V8 bundle, same 112 pitchers: pre-season (2025-only) inputs vs the V8.1 blend
    that folds the scored 2026 starts into the inputs. Pins the size of the provenance gap."""
    import joblib
    b = joblib.load(REPO / 'data/models/xfp_v8_pipeline.pkl'); F = b['features']
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
    r_pre = np.corrcoef(o.pre, o.actual)[0, 1]; r_bl = np.corrcoef(o.blend, o.actual)[0, 1]
    assert len(o) == 112
    assert abs(r_pre - 0.3136) < 2e-3
    assert abs(r_bl - 0.4913) < 2e-3
    # the stored projection csv carries exactly these two columns
    p85 = pd.read_csv(OUT / 'xfp_v8_5_projections.csv'); y = p85[(p85.gs_2026 >= 5) & p85.fp_per_start_actual_2026.notna()]
    assert abs(np.corrcoef(y.xfp_v8, y.fp_per_start_actual_2026)[0, 1] - r_pre) < 1e-3
    assert abs(np.corrcoef(y.xfp_v8_1, y.fp_per_start_actual_2026)[0, 1] - r_bl) < 1e-3


@requires_real_data
def test_stored_v11_ytd_metrics_are_contemporaneous():
    proj = pd.read_csv(OUT / 'xfp_v11_projections.csv'); raw = pd.read_csv(CACHE / 'sp_multiyr_2015_2025.csv')
    rep = xc.ytd_overlap_report(proj, raw[raw.year == 2026])
    assert rep['label'] == 'contemporaneous' and rep['share_contemporaneous'] == 1.0 and rep['n_ytd_eval'] == 112
    y = proj[(proj.gs_2026 >= 5) & proj.fp_per_start_actual_2026.notna()]
    assert abs(np.corrcoef(y.xfp_v11, y.fp_per_start_actual_2026)[0, 1] - 0.51075) < 1e-3
