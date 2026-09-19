"""Mid-season blend semantics, ID alignment and feature-availability checks (synthetic data)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from conftest import make_panel, surname
import xfp_v8_midseason as ms
import xfp_checks as xc
from v11_spotcheck import build_blended_inputs


def _row(year, pitcher=1, pitches=1000, bip=300, **kw):
    base = {'pitcher': pitcher, 'player_name': 'A, B', 'year': year, 'pitches': pitches, 'bip': bip,
            'swstr_pct': 0.10, 'c_plus_swstr': 0.27, 'zone_pct': 0.48, 'o_swing_pct': 0.30,
            'z_swing_pct': 0.66, 'avg_velo': 93.0, 'avg_ext': 6.4, 'abs_pfxz': 0.6, 'bip_pct': 0.68,
            'xwoba_contact': 0.36, 'pitching_plus': 100.0, 'pitch_entropy': 1.2, 'bb_pfxz': -0.5,
            'ip_resid_lag1': 0.1, 'k_pct_lag1': 0.22, 'gs': 10, 'fp_per_start_actual': 10.0}
    base.update(kw)
    return pd.Series(base)


# ---------------------------------------------------------------- blend_pitcher
def test_blend_rate_features_are_pitch_weighted():
    r25 = _row(2025, pitches=3000, swstr_pct=0.10); r26 = _row(2026, pitches=1000, swstr_pct=0.14)
    b = ms.blend_pitcher(r25, r26)
    assert b['cohort'] == 'blended'
    assert np.isclose(b['weight_2026'], 0.25)
    assert np.isclose(b['swstr_pct'], (3000 * 0.10 + 1000 * 0.14) / 4000)


def test_blend_xwoba_is_two_sample_bayes():
    r25 = _row(2025, bip=400, xwoba_contact=0.30); r26 = _row(2026, bip=100, xwoba_contact=0.40)
    b = ms.blend_pitcher(r25, r26)
    expect = (400 * 0.30 + 100 * 0.40 + ms.PRIOR_N * ms.PRIOR_MEAN) / (500 + ms.PRIOR_N)
    assert np.isclose(b['xwoba_contact'], expect)


def test_blend_2026_only_requires_min_pitches():
    assert ms.blend_pitcher(None, _row(2026, pitches=ms.ROOKIE_MIN_PITCHES - 1)) is None
    b = ms.blend_pitcher(None, _row(2026, pitches=ms.ROOKIE_MIN_PITCHES))
    assert b['cohort'] == '2026_only' and b['weight_2026'] == 1.0


def test_blend_2025_only_keeps_prior_season():
    b = ms.blend_pitcher(_row(2025), None)
    assert b['cohort'] == '2025_only' and b['weight_2026'] == 0.0


def test_blended_row_carries_as_of_sample_sizes():
    b = ms.blend_pitcher(_row(2025, pitches=2500), _row(2026, pitches=500))
    assert b['n_pitches_2025'] == 2500 and b['n_pitches_2026'] == 500


def test_non_rate_features_use_latest_available_value_outright():
    """Pins the documented (and dashboard-labelled) behaviour: pitching_plus / pitch_entropy / bb_pfxz
    are NOT pitch-weighted -- the 2026 YTD value is used outright when present."""
    df = pd.DataFrame([_row(2025, pitches=3000, pitching_plus=90.0, pitch_entropy=1.0, bb_pfxz=-0.2),
                       _row(2026, pitches=300, pitching_plus=120.0, pitch_entropy=1.5, bb_pfxz=-0.9)])
    feats = ['swstr_pct', 'pitching_plus', 'pitch_entropy', 'bb_pfxz', 'k_pct_lag1', 'ip_resid_lag1']
    b = build_blended_inputs(df, feats).iloc[0]
    assert np.isclose(b['pitching_plus'], 120.0)
    assert np.isclose(b['pitch_entropy'], 1.5)
    assert np.isclose(b['bb_pfxz'], -0.9)
    assert xc.BLEND_SEMANTICS['pitching_plus'].startswith('latest_available')
    assert xc.BLEND_SEMANTICS['swstr_pct'] == 'pitch_weighted'


def test_blended_inputs_fall_back_to_2025_when_2026_value_missing():
    df = pd.DataFrame([_row(2025, pitching_plus=95.0), _row(2026, pitching_plus=np.nan)])
    b = build_blended_inputs(df, ['pitching_plus']).iloc[0]
    assert np.isclose(b['pitching_plus'], 95.0)


def test_blended_interactions_recomputed_from_blended_primitives():
    df = pd.DataFrame([_row(2025), _row(2026, swstr_pct=0.2, bip_pct=0.6)])
    b = build_blended_inputs(df, ['xwoba_per_pa', 'xwoba_x_swstr']).iloc[0]
    assert np.isclose(b['xwoba_per_pa'], b['xwoba_contact'] * b['bip_pct'])
    assert np.isclose(b['xwoba_x_swstr'], b['xwoba_contact'] * b['swstr_pct'])


# ---------------------------------------------------------------- ID alignment
def test_unique_pitcher_year_detects_duplicates():
    p = make_panel(n_pitchers=10)
    assert xc.check_unique_pitcher_year(p)['ok']
    dup = pd.concat([p, p.head(1)])
    res = xc.check_unique_pitcher_year(dup)
    assert not res['ok'] and res['n_duplicates'] == 2


def test_integer_ids_required():
    p = make_panel(n_pitchers=5)
    assert xc.check_integer_ids(p)['ok']
    assert not xc.check_integer_ids(p.assign(pitcher=p['pitcher'].astype(str)))['ok']
    assert not xc.check_integer_ids(p.assign(pitcher=p['pitcher'].astype(float).where(p.index > 0)))['ok']


def test_fg_alignment_passes_on_consistent_ids_and_name_formats():
    p = make_panel(n_pitchers=8, years=[2024])
    fg = pd.DataFrame({'mlb_id': p['pitcher'], 'player_name_fg': [f'First {surname(i)}' for i in range(8)]})
    fg.loc[0, 'player_name_fg'] = f'First {surname(0)} Jr.'          # suffix variant
    fg.loc[1, 'player_name_fg'] = f'First X. {surname(1)}'            # middle-initial variant
    res = xc.check_fg_id_alignment(p, fg, 2024)
    assert res['ok'] and res['n_matched'] == 8 and res['n_consistent'] == 8 and res['n_ambiguous'] == 0
    assert res['identity_verified_by_name'] is False       # name agreement is never proof of identity


def test_fg_alignment_swapped_siblings_are_ambiguous_not_consistent():
    """Control: two pitchers share a surname (Rogers, Taylor / Rogers, Tyler) and their ids are swapped.
    Surname-only checking passes this silently; the status must be 'ambiguous', never 'consistent'."""
    p = make_panel(n_pitchers=2, years=[2024])
    p['player_name'] = ['Rogers, Taylor', 'Rogers, Tyler']
    fg = pd.DataFrame({'mlb_id': p['pitcher'].values[::-1], 'player_name_fg': ['Taylor Rogers', 'Tyler Rogers']})
    res = xc.check_fg_id_alignment(p, fg, 2024)
    assert res['n_consistent'] == 0 and res['n_ambiguous'] == 2 and res['n_name_mismatch'] == 0
    assert res['identity_verified_by_name'] is False
    # even with ids in the right order a same-surname pool stays ambiguous (names cannot separate them)
    fg_ok = pd.DataFrame({'mlb_id': p['pitcher'].values, 'player_name_fg': ['Taylor Rogers', 'Tyler Rogers']})
    res_ok = xc.check_fg_id_alignment(p, fg_ok, 2024)
    assert res_ok['n_ambiguous'] == 2 and res_ok['n_consistent'] == 0


def test_fg_alignment_first_name_conflict_is_ambiguous():
    p = make_panel(n_pitchers=1, years=[2024]); p['player_name'] = ['Rogers, Taylor']
    fg = pd.DataFrame({'mlb_id': p['pitcher'], 'player_name_fg': ['Tyler Rogers']})
    res = xc.check_fg_id_alignment(p, fg, 2024)
    assert res['ok'] and res['n_ambiguous'] == 1 and res['n_consistent'] == 0
    assert xc.name_status('Rogers, Taylor', 'Tyler Rogers') == 'ambiguous'
    assert xc.names_match('Rogers, Taylor', 'Tyler Rogers')          # surname-only helper is (by design) weaker


def test_name_status_preserves_accents_and_suffix_semantics():
    assert xc.name_status('Peña, Jeremy', 'Jeremy Pena') == 'consistent'          # accent variants tolerated
    assert xc.name_status('Otto Jr., Glenn', 'Glenn Otto') == 'consistent'        # suffix on one side only
    assert xc.name_status('Acuña Jr., Ronald', 'Ronald Acuna Sr.') == 'ambiguous' # conflicting suffixes
    assert xc.name_status('Ortiz, Luis', 'Luis L. Ortiz') == 'consistent'
    assert xc.name_status('Woods Richardson, Simeon', 'Simeon Woods Richardson') == 'consistent'
    assert xc.name_status('Smith, John', 'John Jones') == 'mismatch'


def test_fg_alignment_refuses_fractional_ids_instead_of_truncating():
    p = make_panel(n_pitchers=2, years=[2024])
    fg = pd.DataFrame({'mlb_id': [float(p['pitcher'][0]) + 0.5, float(p['pitcher'][1])],
                       'player_name_fg': ['First ' + surname(0), 'First ' + surname(1)]})
    res = xc.check_fg_id_alignment(p, fg, 2024)
    assert not res['ok'] and 'fractional' in res['error']
    ids, err = xc.strict_int_ids(pd.Series([1.0, 2.0]))
    assert err is None and ids.tolist() == [1, 2]
    assert xc.strict_int_ids(pd.Series(['1', 'x']))[1] is not None


def test_fg_alignment_flags_swapped_ids():
    p = make_panel(n_pitchers=4, years=[2024])
    fg = pd.DataFrame({'mlb_id': p['pitcher'].values[::-1], 'player_name_fg': [f'First {surname(i)}' for i in range(4)]})
    res = xc.check_fg_id_alignment(p, fg, 2024)
    assert not res['ok'] and res['n_name_mismatch'] == 4


def test_fg_alignment_flags_null_or_duplicate_ids():
    p = make_panel(n_pitchers=3, years=[2024])
    fg = pd.DataFrame({'mlb_id': [p['pitcher'][0], p['pitcher'][0], None], 'player_name_fg': ['a', 'b', 'c']})
    assert not xc.check_fg_id_alignment(p, fg, 2024)['ok']


# ---------------------------------------------------------------- feature availability
def test_feature_availability_table_matches_synthetic_panel():
    import xfp_v7_pipeline as v7, xfp_v8_pipeline as v8
    d = v8.derive_v8_features(v7.add_ip_resid_lag(v7.derive_features(make_panel())))
    feats = list(xc.FEATURE_AVAILABILITY)
    rep = xc.check_feature_availability(d, feats)
    assert not rep['violation'].any()
    obs = rep.set_index('feature')['observed_first_year']
    assert obs['pitching_plus'] == 2020 and obs['k_pct_lag1'] == 2016
    # synthetic 2015-2016 have < 50 rows, so the first point-in-time residual season is later than in real data
    assert obs['ip_resid_lag1'] >= 2017


def test_feature_availability_detects_impossible_history():
    d = make_panel(n_pitchers=5, years=[2018, 2019])
    rep = xc.check_feature_availability(d.assign(pitching_plus=100.0), ['pitching_plus'])
    assert rep['violation'].iloc[0]


def test_ytd_overlap_report_labels_contemporaneous_sample():
    proj = pd.DataFrame({'pitcher': [1, 2, 3], 'gs_2026': [7, 6, 2], 'fp_per_start_actual_2026': [10., 12., 5.]})
    panel26 = pd.DataFrame({'pitcher': [1, 2]})
    rep = xc.ytd_overlap_report(proj, panel26)
    assert rep['n_ytd_eval'] == 2 and rep['n_inputs_include_2026_starts'] == 2
    assert rep['label'] == 'contemporaneous' and rep['forward_status'] == 'not_forward'
    # no overlap is NOT evidence of a forward evaluation without frozen input/cutoff provenance
    none = xc.ytd_overlap_report(proj, pd.DataFrame({'pitcher': [99]}))
    assert none['label'] == 'no_overlap_detected' and none['forward_status'].startswith('unverified')
    assert 'forward' not in none['label']


def test_metric_provenance_is_tied_to_the_artifact():
    legacy = xc.metric_provenance({'cross_year_r': 0.61381, 'trained_date': '2026-05-05'})['cross_year_r']
    assert legacy['point_in_time'] is False and legacy['status'].startswith('archived_legacy_loo')
    pit = xc.metric_provenance({'cross_year_r': 0.6, 'ip_resid_mode': 'point_in_time'})['cross_year_r']
    assert pit['point_in_time'] is True and pit['status'] == 'point_in_time'
    unk = xc.metric_provenance({'cross_year_r': 0.6, 'ip_resid_mode': 'something_else'})['cross_year_r']
    assert unk['point_in_time'] is None and unk['status'] == 'unverified'
    assert xc.metric_provenance(None)['cross_year_r']['status'] == 'unverified'
    assert xc.evaluation_labels()['cross_year_r']['point_in_time'] is None
    assert xc.evaluation_labels({'cross_year_r': 0.6})['cross_year_r']['point_in_time'] is False
