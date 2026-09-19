"""
xfp_checks.py - Data-integrity, ID-alignment, feature-availability and provenance helpers.

Pure functions (no side effects) used by the regression suite (tests/) and the audit runner
(scripts/xfp_audit.py). Everything here is deterministic and works on synthetic fixtures.
"""
from __future__ import annotations
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

# Repo root = nearest ancestor holding data/research (works for scripts/ and scripts/xfp/ layouts).
ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'data' / 'research').is_dir()),
            Path(__file__).resolve().parents[1])

# ---------------------------------------------------------------------------
# Feature availability (first season a feature can be a point-in-time input)
# ---------------------------------------------------------------------------
# source: where the raw values come from
# first_year: earliest season for which the feature is a legitimate *input* for that season
# reason: why the limit exists
FEATURE_AVAILABILITY = {
    'avg_velo':       {'source': 'statcast',   'first_year': 2015, 'reason': 'Statcast pitch tracking starts 2015'},
    'zone_pct':       {'source': 'statcast',   'first_year': 2015, 'reason': 'Statcast pitch tracking starts 2015'},
    'o_swing_pct':    {'source': 'statcast',   'first_year': 2015, 'reason': 'Statcast pitch tracking starts 2015'},
    'swstr_pct':      {'source': 'statcast',   'first_year': 2015, 'reason': 'Statcast pitch tracking starts 2015'},
    'c_plus_swstr':   {'source': 'statcast',   'first_year': 2015, 'reason': 'Statcast pitch tracking starts 2015'},
    'z_swing_pct':    {'source': 'statcast',   'first_year': 2015, 'reason': 'Statcast pitch tracking starts 2015'},
    'xwoba_per_pa':   {'source': 'statcast',   'first_year': 2015, 'reason': 'xwOBA (estimated_woba_using_speedangle) from 2015'},
    'xwoba_x_swstr':  {'source': 'statcast',   'first_year': 2015, 'reason': 'derived from xwoba_contact x swstr_pct'},
    'pitch_entropy':  {'source': 'statcast',   'first_year': 2015, 'reason': 'pitch_type mix from Statcast'},
    'bb_pfxz':        {'source': 'statcast',   'first_year': 2015, 'reason': 'pfx_z by pitch family, >=30 breaking balls'},
    'k_pct_lag1':     {'source': 'statcast_lag','first_year': 2016, 'reason': 'needs the pitcher''s PRIOR season (2015 is the first)'},
    'ip_resid_lag1':  {'source': 'statcast_lag','first_year': 2017,
                       'reason': 'needs prior-season residual from an IP model fit on >=50 rows of EARLIER seasons '
                                 '(point-in-time: 2015 has no earlier seasons, so 2016 residuals are the first)'},
    'pitching_plus':  {'source': 'fangraphs',  'first_year': 2020, 'reason': 'FanGraphs Stuff+/Location+/Pitching+ published from 2020'},
    'fp_strike_pct':  {'source': 'statcast',   'first_year': 2015,
                       'reason': 'first-pitch strike rate from Statcast; SOURCE FILE (fp_strike_2015_2026.csv) NOT IN REPO'},
}

# Which V8.1-blend semantics each production feature gets (see xfp_v8_midseason.blend_pitcher and
# v11_spotcheck.build_blended_inputs). 'pitch_weighted' = (n25*v25 + n26*v26)/(n25+n26);
# 'bayes_two_sample' = shrunk toward PRIOR_MEAN with PRIOR_N; 'latest_available' = the 2026 YTD value
# is used OUTRIGHT when present (no weighting), else the 2025 value; 'derived' = recomputed from blended primitives.
BLEND_SEMANTICS = {
    'avg_velo': 'pitch_weighted', 'zone_pct': 'pitch_weighted', 'o_swing_pct': 'pitch_weighted',
    'swstr_pct': 'pitch_weighted', 'c_plus_swstr': 'pitch_weighted', 'z_swing_pct': 'pitch_weighted',
    'xwoba_per_pa': 'derived (bayes xwoba_contact x pitch-weighted bip_pct)',
    'xwoba_x_swstr': 'derived (bayes xwoba_contact x pitch-weighted swstr_pct)',
    'ip_resid_lag1': 'latest_available (= 2025 residual; 2026 row carries the 2025 lag)',
    'k_pct_lag1': 'latest_available (= 2025 k_pct; 2026 row carries the 2025 lag)',
    'pitch_entropy': 'latest_available (2026 YTD outright)',
    'bb_pfxz': 'latest_available (2026 YTD outright)',
    'pitching_plus': 'latest_available (2026 YTD outright)',
    'fp_strike_pct': 'latest_available (2026 YTD outright)',
}


def feature_first_year(feat: str) -> int | None:
    rec = FEATURE_AVAILABILITY.get(feat)
    return rec['first_year'] if rec else None


def check_feature_availability(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Return one row per feature: declared first_year vs first year with any non-null value in df.
    `violation` is True when data claims the feature EARLIER than it can exist."""
    rows = []
    for f in feats:
        declared = feature_first_year(f)
        if f in df.columns and df[f].notna().any():
            observed = int(df.loc[df[f].notna(), 'year'].min())
        else:
            observed = None
        rows.append({'feature': f, 'declared_first_year': declared, 'observed_first_year': observed,
                     'violation': (declared is not None and observed is not None and observed < declared)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ID alignment
# ---------------------------------------------------------------------------
_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv'}


def _clean(s) -> str:
    s = '' if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return s.lower()


def _alpha(s: str) -> str:
    return ''.join(ch for ch in s if ch.isalpha())


def _strip_suffix_tokens(s: str) -> list[str]:
    return [t for t in s.replace('.', ' ').split() if t not in _SUFFIXES]


def _surname_key(s) -> str:
    """Surname (alpha, accent-stripped, suffix-free). 'Last, First' -> whole part before the comma
    (multi-word surnames kept); 'First M. Last' -> last token."""
    s = _clean(s)
    if ',' in s:
        return _alpha(''.join(_strip_suffix_tokens(s.split(',', 1)[0])))
    toks = _strip_suffix_tokens(s)
    return _alpha(toks[-1]) if toks else ''


def names_match(panel_name, fg_name) -> bool:
    """True when the Statcast 'Last, First' name and the FanGraphs 'First Last' name plausibly denote
    the same person: the comma-form surname must end the FanGraphs name (handles 'Woods Richardson',
    'Ponce de Leon', 'Luis L. Ortiz', 'Glenn Otto' vs 'Otto Jr., Glenn', accents)."""
    a, b = _clean(panel_name), _clean(fg_name)
    if ',' in a and ',' not in b:
        surname = _surname_key(a)
        full_b = _alpha(''.join(_strip_suffix_tokens(b)))
        return bool(surname) and full_b.endswith(surname)
    return _surname_key(a) == _surname_key(b)


def check_unique_pitcher_year(df: pd.DataFrame) -> dict:
    dup = df.duplicated(['pitcher', 'year'], keep=False)
    return {'ok': not dup.any(), 'n_duplicates': int(dup.sum()),
            'examples': df.loc[dup, ['pitcher', 'year']].head(5).values.tolist()}


def check_integer_ids(df: pd.DataFrame, col: str = 'pitcher') -> dict:
    s = df[col]
    is_int = pd.api.types.is_integer_dtype(s)
    nulls = int(s.isna().sum())
    return {'ok': is_int and nulls == 0, 'dtype': str(s.dtype), 'n_null': nulls}


def check_fg_id_alignment(panel: pd.DataFrame, fg: pd.DataFrame, year: int) -> dict:
    """Join FanGraphs (mlb_id) to the Statcast panel (pitcher) for one season and verify that the
    names on both sides agree by surname. A mismatch means an MLBAM id points at a different person."""
    f = fg.rename(columns={'mlb_id': 'pitcher'})
    if 'pitcher' not in f.columns:
        return {'ok': False, 'year': year, 'error': 'fangraphs file has no mlb_id column'}
    if f['pitcher'].isna().any():
        return {'ok': False, 'year': year, 'error': f'{int(f["pitcher"].isna().sum())} null mlb_id rows'}
    if f['pitcher'].duplicated().any():
        return {'ok': False, 'year': year, 'error': f'{int(f["pitcher"].duplicated().sum())} duplicate mlb_id rows'}
    f = f.assign(pitcher=f['pitcher'].astype('int64'))
    sub = panel[panel['year'] == year]
    m = sub.merge(f[['pitcher', 'player_name_fg']], on='pitcher', how='left')
    matched = m['player_name_fg'].notna()
    mism = [(a, b) for a, b in zip(m.loc[matched, 'player_name'], m.loc[matched, 'player_name_fg'])
            if not names_match(a, b)]
    return {'ok': len(mism) == 0, 'year': year, 'n_panel': int(len(sub)), 'n_matched': int(matched.sum()),
            'n_name_mismatch': len(mism), 'mismatches': mism[:10]}


# ---------------------------------------------------------------------------
# Evaluation-provenance labels
# ---------------------------------------------------------------------------
def ytd_overlap_report(proj: pd.DataFrame, panel_2026: pd.DataFrame, gs_min: int = 5) -> dict:
    """How much of the 'YTD' evaluation sample had its own 2026 starts folded into the model inputs.

    proj must carry gs_2026 / fp_per_start_actual_2026 (from the projection csv) and, if available,
    cohort / weight_2026 (from the V8.1 blend). panel_2026 is the 2026 slice of the training panel
    (the same rows whose stats were blended into the inputs)."""
    y = proj[(proj['gs_2026'] >= gs_min) & proj['fp_per_start_actual_2026'].notna()].copy()
    in_panel = y['pitcher'].isin(panel_2026['pitcher'])
    out = {'n_ytd_eval': int(len(y)), 'n_inputs_include_2026_starts': int(in_panel.sum()),
           'share_contemporaneous': round(float(in_panel.mean()), 4) if len(y) else None,
           'label': 'contemporaneous' if in_panel.any() else 'forward'}
    if 'weight_2026' in y.columns:
        out['mean_weight_2026'] = round(float(y['weight_2026'].fillna(0).mean()), 4)
    return out


def evaluation_labels() -> dict:
    """Machine-readable description of what each reported metric actually measures."""
    return {
        'cross_year_r': {
            'point_in_time': True,
            'definition': 'features from season T (pitchers present in T and T+1) -> FP/start in T+1; '
                          'scaler+RidgeCV fit on rows with year < T+1 only',
            'transitions': '2021->2022, 2022->2023, 2023->2024, 2024->2025 (4; NOT 9)',
            'caveat': 'ip_resid_lag1 in archived numbers was built in leave-one-year-out mode '
                      '(future seasons incl. partial 2026 in the IP model). Point-in-time rebuild changes '
                      'r by <0.001 (see audit). k_bias cohorts use TEST-year k_pct (post-hoc diagnostic).',
        },
        'ooy_r': {
            'point_in_time': False,
            'definition': 'same-season holdout: model for season Y trained on all OTHER seasons, '
                          'including seasons after Y',
            'caveat': 'secondary diagnostic; never a deployment metric',
        },
        'ytd_r_2026': {
            'point_in_time': False,
            'definition': 'corr(projection, 2026 FP/start through data cutoff) for gs_2026 >= 5',
            'caveat': 'CONTEMPORANEOUS, not forward: the projection inputs are the V8.1 blend, which folds '
                      'the SAME 2026 starts (swstr, xwoba, Pitching+ ...) into the features. '
                      'Frozen-V8 check: pre-season inputs r=0.31 vs blended inputs r=0.49 on the same 112 pitchers.',
        },
    }


def missing_sources(root: Path = ROOT) -> list[dict]:
    """Inputs the production path reads that are NOT in the repository."""
    cache = root / 'data' / 'research' / 'xfp_cache'
    checks = [
        (cache / 'fp_strike_2015_2026.csv', 'fp_strike_pct feature for V11 (14th feature); V11 cannot be retrained from the repo without it'),
        (cache / 'statcast_2025.parquet', 'raw Statcast cache (any year); needed by build_sp_multiyr / pitch-type panels'),
        (root / 'data' / 'outputs' / 'xfp_v8_projections.csv', 'V8 baseline projections read by xfp_v8_midseason.py'),
        (root / 'data' / 'outputs' / 'xfp_v11_dashboard.html', 'dashboard written by xfp_v11_lock.build_dashboard (docs/index.html is generated elsewhere)'),
        (root / 'data' / 'outputs' / 'plv_rolling_2026.parquet', 'PLV rolling file read by xfp_rolling_ip.py'),
    ]
    return [{'path': str(p.relative_to(root)), 'present': p.exists(), 'role': role} for p, role in checks]
