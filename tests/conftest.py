"""Synthetic point-in-time fixtures for the xFP regression suite.

Nothing here touches the real data or the network. The panel mimics the schema of
data/research/xfp_cache/sp_multiyr_2015_2025.csv closely enough for derive_features(),
add_ip_resid_lag(), derive_v8_features(), cross_year_evaluate(), ooy_evaluate() and
blend_pitcher() to run end-to-end.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

CACHE = REPO / 'data' / 'research' / 'xfp_cache'
REAL_PANEL = CACHE / 'sp_multiyr_2015_2025.csv'


def surname(i: int) -> str:
    """Distinct alphabetic surnames (no digits, so name-normalisation keeps them apart)."""
    a, b = chr(97 + (i // 26) % 26), chr(97 + i % 26)
    return f'{a.upper()}{b}rrington'


def make_panel(n_pitchers: int = 60, years=range(2015, 2027), seed: int = 7,
               future_ip_shift: float = 0.0, future_from_year: int = 2023) -> pd.DataFrame:
    """Deterministic pitcher-season panel with a latent skill so features correlate with FP.

    future_ip_shift adds a constant to ip_per_start for every season >= future_from_year. Used to
    prove that point-in-time constructions for earlier seasons are invariant to future data.
    """
    rng = np.random.default_rng(seed)
    years = list(years)
    skill = rng.normal(0, 1, n_pitchers)
    rows = []
    for i in range(n_pitchers):
        pid = 500000 + i
        # each pitcher is present for a contiguous block of seasons (creates real gaps/entries)
        start = int(rng.integers(0, 4)) if len(years) > 6 else 0
        stop = len(years) - (int(rng.integers(0, 3)) if len(years) > 6 else 0)
        for y in years[start:stop]:
            s = skill[i] + rng.normal(0, 0.5)
            swstr = np.clip(0.10 + 0.02 * s + rng.normal(0, 0.01), 0.04, 0.22)
            cs = np.clip(0.16 + rng.normal(0, 0.02), 0.08, 0.25)
            xw = np.clip(0.37 - 0.02 * s + rng.normal(0, 0.02), 0.25, 0.50)
            bip_pct = np.clip(0.68 - 0.1 * swstr + rng.normal(0, 0.02), 0.5, 0.85)
            pitches = int(rng.integers(600, 3200)) if y != 2026 else int(rng.integers(120, 720))
            tbf = int(pitches / 3.9)
            gs = max(1, int(pitches / 95))
            ip = np.clip(5.3 + 0.4 * s + rng.normal(0, 0.4), 3.0, 7.5)
            if y >= future_from_year:
                ip = ip + future_ip_shift
            k_pct = np.clip(0.22 + 0.04 * s + rng.normal(0, 0.02), 0.08, 0.40)
            fp = 3.3 * ip + 10 * swstr - 15 * (xw - 0.37) + 2.5 * s + rng.normal(0, 2.0)
            rows.append({
                'pitcher': pid, 'player_name': f'{surname(i)}, First', 'year': y,
                'pitches': pitches, 'tbf': tbf, 'bip': int(tbf * bip_pct), 'gs': gs,
                'avg_velo': 93 + 1.5 * s + rng.normal(0, 0.5), 'avg_ext': 6.4 + rng.normal(0, 0.2),
                'avg_pfxz': rng.normal(0.6, 0.4), 'abs_pfxz': abs(rng.normal(0.6, 0.4)),
                'zone_pct': np.clip(0.48 + rng.normal(0, 0.02), 0.35, 0.6),
                'o_swing_pct': np.clip(0.30 + 0.02 * s + rng.normal(0, 0.02), 0.15, 0.45),
                'z_swing_pct': np.clip(0.66 + rng.normal(0, 0.02), 0.5, 0.8),
                'swstr_pct': swstr, 'c_plus_swstr': swstr + cs, 'xwoba_contact': xw,
                'bip_pct': bip_pct, 'k_pct': k_pct, 'bb_pct': np.clip(0.08 + rng.normal(0, 0.015), 0.02, 0.15),
                'hard_hit_pct': np.clip(0.38 + rng.normal(0, 0.04), 0.2, 0.55),
                'ip_per_start': ip, 'fp_per_start_actual': fp,
                'pitch_entropy': np.clip(1.2 + rng.normal(0, 0.2), 0.5, 2.0),
                'bb_pfxz': rng.normal(-0.5, 0.3),
                'pitching_plus': (100 + 8 * s + rng.normal(0, 4)) if y >= 2020 else np.nan,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()


@pytest.fixture
def real_panel() -> pd.DataFrame:
    if not REAL_PANEL.exists():
        pytest.skip('real training panel not present')
    return pd.read_csv(REAL_PANEL)


def requires_real_data(fn):
    return pytest.mark.skipif(not REAL_PANEL.exists(), reason='real training panel not present')(fn)
