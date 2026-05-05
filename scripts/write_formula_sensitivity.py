"""Write formula_sensitivity.csv documenting score-formula variants tested for V11 evaluation."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'data' / 'research'

# (config_name, cross_year_r, k_bias_hi, dataset)
results = [
    # V11 search results (full data, 2015-2024 transitions, n=854)
    ('V8.5 baseline (full data)',           0.600, 0.466, 'full'),
    ('V8.5+stuff_plus',                     0.609, 0.787, 'full'),
    ('V8.5+location_plus',                  0.610, 0.823, 'full'),
    ('V8.5+pitching_plus',                  0.613, 0.774, 'full'),
    ('V8.5+pb_stuff',                       0.613, 0.765, 'full'),
    ('V8.5+pb_command',                     0.611, 0.831, 'full'),
    ('V8.5+pb_xrv100',                      0.610, 0.771, 'full'),
    ('V8.5+fp_strike_pct',                  0.600, 0.462, 'full'),
    ('V11 (V8.5+pitching_plus+fp_strike)',  0.614, 0.773, 'full'),
    # V12 results (subset 2020-2024, n=463)
    ('V8.5 baseline (subset)',              0.583, 1.113, 'subset'),
    ('V12 V8.5+pitching_plus a=500',        0.586, 1.534, 'subset'),
    ('V12 V8.5+pb_stuff a=1',               0.594, 1.364, 'subset'),
    ('V12 V8.5+all_FG a=1',                 0.600, 1.385, 'subset'),
    ('V12 V8.5+all_FG+PB a=1 (best)',       0.605, 1.295, 'subset'),
    ('V12 V8.5+all_FG+PB a=25',             0.601, 1.316, 'subset'),
]

formulas = [
    ('current (linear coef=0.5)',  lambda r, k: r * 3 - abs(k) * 0.5),
    ('linear coef=0.25',           lambda r, k: r * 3 - abs(k) * 0.25),
    ('linear coef=0.10',           lambda r, k: r * 3 - abs(k) * 0.10),
    ('linear coef=1.0',            lambda r, k: r * 3 - abs(k) * 1.0),
    ('tolerance T=0.50, coef=0.5', lambda r, k: r * 3 - max(0, abs(k) - 0.50) * 0.5),
    ('tolerance T=0.70, coef=0.5', lambda r, k: r * 3 - max(0, abs(k) - 0.70) * 0.5),
    ('tolerance T=1.00, coef=0.5', lambda r, k: r * 3 - max(0, abs(k) - 1.00) * 0.5),
    ('quadratic coef=0.5',         lambda r, k: r * 3 - (k ** 2) * 0.5),
    ('quadratic coef=0.25',        lambda r, k: r * 3 - (k ** 2) * 0.25),
]

rows = []
for name, r, kbias, ds in results:
    row = {'config': name, 'dataset': ds, 'cross_year_r': r, 'k_bias_hi': kbias}
    for fname, ffn in formulas:
        row[fname] = round(ffn(r, kbias), 5)
    rows.append(row)
df = pd.DataFrame(rows)

# Add ranking under each formula (within each dataset)
for fname, _ in formulas:
    df[f'{fname}_rank'] = df.groupby('dataset')[fname].rank(ascending=False, method='min').astype(int)

out_path = RESEARCH / 'formula_sensitivity.csv'
df.to_csv(out_path, index=False)
print(f'wrote {out_path} ({len(df)} configs x {len(formulas)} formulas)')

# Print which config wins under each formula on full-data subset
print('\nWinners per formula (full-data subset):')
full = df[df['dataset']=='full']
for fname, _ in formulas:
    winner = full.loc[full[fname].idxmax()]
    print(f'  {fname:<35s} -> {winner["config"]:<40s} score={winner[fname]}')
