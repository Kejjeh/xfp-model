"""Point-in-time invariants for feature construction and cross-year evaluation (synthetic data)."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest

from conftest import make_panel
import xfp_v7_pipeline as v7
import xfp_v8_pipeline as v8

warnings.filterwarnings('ignore')
V85 = ['avg_velo', 'zone_pct', 'o_swing_pct', 'swstr_pct', 'c_plus_swstr', 'xwoba_per_pa',
       'z_swing_pct', 'xwoba_x_swstr', 'ip_resid_lag1', 'k_pct_lag1', 'pitch_entropy', 'bb_pfxz']


_CACHE: dict = {}


def _prep(panel, mode, key=None):
    """Derive features (+ip_resid in the requested mode). Cached per (key, mode) to keep the suite fast."""
    if key is not None and (key, mode) in _CACHE:
        return _CACHE[(key, mode)].copy()
    d = v7.derive_features(panel)
    d = v7.add_ip_resid_lag(d, mode=mode)
    d = v8.derive_v8_features(d)
    if key is not None:
        _CACHE[(key, mode)] = d.copy()
    return d


# ---------------------------------------------------------------- ip_resid_lag1
def test_ip_resid_point_in_time_is_invariant_to_future_seasons():
    """Residuals for seasons < 2023 must not change when 2023+ IP values are shifted."""
    a = _prep(make_panel(), 'point_in_time', key='base')
    b = _prep(make_panel(future_ip_shift=2.0), 'point_in_time', key='shift')
    past_a = a[a.year < 2023].set_index(['pitcher', 'year'])['ip_resid']
    past_b = b[b.year < 2023].set_index(['pitcher', 'year'])['ip_resid']
    assert np.allclose(past_a.values, past_b.values, equal_nan=True)
    # and the lag carried into 2023 (a 2022 residual) is unchanged too
    lag_a = a[a.year == 2023].set_index('pitcher')['ip_resid_lag1']
    lag_b = b[b.year == 2023].set_index('pitcher')['ip_resid_lag1']
    assert np.allclose(lag_a.values, lag_b.values, equal_nan=True)


def test_ip_resid_legacy_loo_mode_reproduces_the_leak():
    """Documents the defect: in 'loo' mode the 2016-2022 residuals move when 2023+ IP changes."""
    a = _prep(make_panel(), 'loo', key='base')
    b = _prep(make_panel(future_ip_shift=2.0), 'loo', key='shift')
    past_a = a[a.year.between(2016, 2022)]['ip_resid'].values
    past_b = b[b.year.between(2016, 2022)]['ip_resid'].values
    assert not np.allclose(past_a, past_b, equal_nan=True)


def test_ip_resid_point_in_time_first_seasons_are_nan():
    d = _prep(make_panel(), 'point_in_time', key='base')
    assert d.loc[d.year == 2015, 'ip_resid'].isna().all()          # no earlier seasons at all
    assert d.loc[d.year == 2016, 'ip_resid_lag1'].isna().all()     # lag of a NaN residual


def test_ip_resid_default_mode_is_point_in_time():
    d = v7.derive_features(make_panel())
    assert v7.add_ip_resid_lag(d).equals(v7.add_ip_resid_lag(d, mode='point_in_time'))


def test_ip_resid_rejects_unknown_mode():
    with pytest.raises(ValueError):
        v7.add_ip_resid_lag(v7.derive_features(make_panel()), mode='future')


# ---------------------------------------------------------------- lag features
def test_lag_features_only_look_backwards():
    d = _prep(make_panel(), 'point_in_time', key='base').sort_values(['pitcher', 'year'])
    for pid, g in d.groupby('pitcher'):
        k = g['k_pct'].values; lag = g['k_pct_lag1'].values
        assert np.isnan(lag[0])
        assert np.allclose(lag[1:], k[:-1])


# ---------------------------------------------------------------- cross-year evaluation
def test_cross_year_predictions_ignore_test_year_targets():
    """Scrambling the 2025 targets must not move a single 2024->2025 prediction."""
    base = _prep(make_panel(), 'point_in_time', key='base')
    scr = base.copy()
    m = scr.year == 2025
    scr.loc[m, 'fp_per_start_actual'] = scr.loc[m, 'fp_per_start_actual'].values[::-1] + 7.0
    r1 = v7.cross_year_evaluate(base, V85, return_rows=True, transitions=[(2024, 2025)])['rows']
    r2 = v7.cross_year_evaluate(scr, V85, return_rows=True, transitions=[(2024, 2025)])['rows']
    assert len(r1) > 10
    assert np.allclose(r1.sort_values('pitcher')['pred'].values, r2.sort_values('pitcher')['pred'].values)


def test_cross_year_predictions_ignore_seasons_after_test_year():
    """Dropping or distorting 2026 rows must not change any 2021->2025 prediction (expanding window)."""
    base = _prep(make_panel(), 'point_in_time', key='base')
    no26 = base[base.year < 2026]
    r1 = v7.cross_year_evaluate(base, V85, return_rows=True)['rows']
    r2 = v7.cross_year_evaluate(no26, V85, return_rows=True)['rows']
    key = ['pitcher', 'yr_test']
    m = r1.merge(r2, on=key, suffixes=('_a', '_b'))
    assert len(m) == len(r1) == len(r2)
    assert np.allclose(m['pred_a'], m['pred_b'])


def test_cross_year_scaler_is_fit_on_prior_seasons_only():
    """An extreme-valued pitcher who exists ONLY in the test season cannot influence the scaler."""
    base = _prep(make_panel(), 'point_in_time', key='base')
    extra = base[base.year == 2025].head(1).copy()
    extra['pitcher'] = 999999
    for f in V85:
        extra[f] = extra[f] * 50 + 1000
    with_outlier = pd.concat([base, extra], ignore_index=True)
    r1 = v7.cross_year_evaluate(base, V85, return_rows=True, transitions=[(2024, 2025)])['rows']
    r2 = v7.cross_year_evaluate(with_outlier, V85, return_rows=True, transitions=[(2024, 2025)])['rows']
    assert np.allclose(r1.sort_values('pitcher')['pred'].values, r2.sort_values('pitcher')['pred'].values)


def test_cross_year_training_window_is_expanding_and_strictly_prior():
    base = _prep(make_panel(), 'point_in_time', key='base')
    rows = v7.cross_year_evaluate(base, V85, return_rows=True)['rows']
    n_prior = rows.groupby('yr_test')['n_prior'].first()
    assert list(n_prior.index) == [2022, 2023, 2024, 2025]
    assert n_prior.is_monotonic_increasing
    for yr_test in n_prior.index:
        expected = base[base.year < yr_test].dropna(subset=V85 + ['fp_per_start_actual'])
        assert n_prior[yr_test] == len(expected)


def test_cross_year_default_transitions_are_four_not_nine():
    assert v7.CROSS_YEAR_TRANSITIONS == [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
    res = v7.cross_year_evaluate(_prep(make_panel(), 'point_in_time', key='base'), V85)
    assert res['n_transitions'] == 4


def test_ooy_is_documented_as_not_point_in_time():
    """OOY trains on seasons AFTER the holdout: distorting 2025 targets changes the 2022 holdout score."""
    base = _prep(make_panel(), 'point_in_time', key='base')
    scr = base.copy(); scr.loc[scr.year == 2025, 'fp_per_start_actual'] += 25.0
    only22 = lambda d: d[d.year.isin([2019, 2020, 2021, 2022, 2025])]
    a = v7.ooy_evaluate(only22(base), V85); b = v7.ooy_evaluate(only22(scr), V85)
    assert a['r'] != b['r']
    assert 'NOT point-in-time' in v7.ooy_evaluate.__doc__
