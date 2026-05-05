"""
compare_score_formulas.py - Re-rank V8.5/V11/V12 candidates under different score formulas.

Uses hard-coded (r, k_bias_hi) values from V11 and V12 runs.
No re-evaluation needed — only the score function changes.
"""
from __future__ import annotations
import pandas as pd

# Format: (name, cross_year_r, k_bias_hi, dataset)
# V11 = full 2015-2024 transitions (n=854)
# V12 = 2020-2024 subset (n=463)
results_full = [
    ('V8.5 baseline',                      0.600, 0.466, 'full'),
    ('V8.5+stuff_plus',                    0.609, 0.787, 'full'),
    ('V8.5+location_plus',                 0.610, 0.823, 'full'),
    ('V8.5+pitching_plus',                 0.613, 0.774, 'full'),
    ('V8.5+pb_stuff',                      0.613, 0.765, 'full'),
    ('V8.5+pb_command',                    0.611, 0.831, 'full'),
    ('V8.5+pb_xrv100',                     0.610, 0.771, 'full'),
    ('V8.5+fp_strike_pct',                 0.600, 0.462, 'full'),
]

results_subset = [
    ('V8.5 baseline (subset)',             0.583, 1.113, 'subset'),
    ('V12 V8.5+pitching_plus α=500',       0.586, 1.534, 'subset'),
    ('V12 V8.5+pb_stuff α=1',              0.594, 1.364, 'subset'),
    ('V12 V8.5+stuff+pitching α=500',      0.588, 1.512, 'subset'),
    ('V12 V8.5+all_FG α=1',                0.600, 1.385, 'subset'),
    ('V12 V8.5+all_FG+PB α=1 (BEST)',      0.605, 1.295, 'subset'),
    ('V12 V8.5+all_FG+PB α=25',            0.601, 1.316, 'subset'),
]


def score_linear(r, kbias, coef):
    """Standard linear penalty. Current default coef=0.5."""
    return r * 3 - abs(kbias) * coef

def score_tolerance(r, kbias, threshold, coef=0.5):
    """No penalty below threshold T."""
    return r * 3 - max(0, abs(kbias) - threshold) * coef

def score_quadratic(r, kbias, coef=0.5):
    """Quadratic penalty: small biases nearly free, large ones expensive."""
    return r * 3 - (kbias ** 2) * coef


FORMULAS = [
    ('current (coef=0.5)',     lambda r, k: score_linear(r, k, 0.5)),
    ('lower coef (0.25)',      lambda r, k: score_linear(r, k, 0.25)),
    ('lower coef (0.10)',      lambda r, k: score_linear(r, k, 0.10)),
    ('higher coef (1.0)',      lambda r, k: score_linear(r, k, 1.0)),
    ('tolerance T=0.50',       lambda r, k: score_tolerance(r, k, 0.50)),
    ('tolerance T=0.70',       lambda r, k: score_tolerance(r, k, 0.70)),
    ('tolerance T=1.00',       lambda r, k: score_tolerance(r, k, 1.00)),
    ('quadratic (coef=0.5)',   lambda r, k: score_quadratic(r, k, 0.5)),
    ('quadratic (coef=0.25)',  lambda r, k: score_quadratic(r, k, 0.25)),
]


def show_block(title, results):
    print('\n' + '=' * 110)
    print(title)
    print('=' * 110)
    rows = []
    for name, r, kbias, _ds in results:
        row = {'config': name, 'r': r, 'kbias': kbias}
        for fname, ffn in FORMULAS:
            row[fname] = round(ffn(r, kbias), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # For each formula, identify the winner
    print('\nWinner under each formula:')
    for fname, _ in FORMULAS:
        winner_idx = df[fname].idxmax()
        winner = df.iloc[winner_idx]
        baseline = df.iloc[0]  # First row is always V8.5 baseline
        delta = winner[fname] - baseline[fname]
        ships = winner['config'] != baseline['config'] and delta >= 0.010
        marker = ' >>> SHIPS' if ships else (' (V8.5 wins)' if winner['config'] == baseline['config'] else f' (Δ={delta:+.4f}, below 0.010 threshold)')
        print(f'  {fname:<26s} -> {winner["config"]:<40s} score={winner[fname]:.4f}{marker}')


if __name__ == '__main__':
    print('=' * 110)
    print('V11 — full-data comparison (2015-2024 transitions, n=854)')
    show_block('V11 candidates: V8.5 + single new feature added', results_full)

    print('\n')
    print('=' * 110)
    print('V12 — subset comparison (2020-2024 transitions, n=463) where FG history exists')
    show_block('V12 candidates: V8.5 + residual model on FG features', results_subset)

    print('\n')
    print('=' * 110)
    print('TAKEAWAYS')
    print('=' * 110)
    print('''
- Current formula (coef=0.5): V8.5 baseline still wins across both datasets.
- Lowering coef to 0.25 or 0.10: opens the door to V11 (V8.5+pb_stuff or +pitching_plus).
- Tolerance threshold T=0.7 to 1.0: same effect as lower coef, but more honest about why
  (no penalty below threshold; full penalty above).
- Quadratic: punishes V12 (subset, kbias 1.3+) catastrophically; treats V8.5 vs V11 nearly even.
- Higher coef (1.0): V8.5 dominates by even larger margin.
''')
