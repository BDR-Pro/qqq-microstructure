# Part of qqq-microstructure.
#
# Signal #6: cross-sectional intraday continuation. At 10:30, rank the universe
# by its own first hour (log p60/open) and hold top-minus-bottom quintile from
# 10:30 to the close. One pass, no sweeps.
#
# Pre-declared direction: CONTINUATION. The time-series version -- sign of QQQ's
# first hour, held to the close -- is this repo's oldest surviving signal
# (RESULTS 8/11), and Heston-Korajczyk-Sadka (2010) document cross-sectional
# intraday periodicity. But short-horizon cross-sections often REVERSE, so the
# sign is genuinely at risk here; whatever prints is recorded, and a negative
# result is a finding, not a license to flip the sign after seeing it.
#
# Why this leg matters for the stack: the deployed intraday leg is QQQ-only, and
# the RESULTS 12 holdout failed on QQQ specifically while SPY was fine --
# single-name regime risk. This spreads the same 10:30->close window across the
# top-150 with the same capital (the overnight basket is flat by 09:30), which
# is the diversification a single instrument cannot provide.
#
# Mechanics shared with xsec_backtest.py: universe lagged one month (trade month
# T on file T-1's list), ETFs and test symbols excluded, and both returns are
# same-day same-scale so splits cannot contaminate them; a +/-25% guard drops
# data-error rows and is counted. Costs: the 10:30 entry CROSSES the spread (it
# is not an auction), the exit is MOC; the L/S pays 4 crossings/day.
#
#   python src/xsec_intraday.py            # needs data/xsec/ from xsec_extract.py

import os, argparse
import numpy as np, pandas as pd
from xsec_backtest import load_panel, stats, era_table, ETF, MIN_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    argparse.ArgumentParser().parse_args()
    df = load_panel()
    df = df[df.p60 > 0].copy()
    df['fh_bps'] = np.log(df.p60 / df.open) * 1e4
    df['rest_bps'] = np.log(df.close / df.p60) * 1e4
    bad = (df.fh_bps.abs() > 2500) | (df.rest_bps.abs() > 2500)
    print(f'first-hour rows: {len(df):,}   +/-25% guard dropped {int(bad.sum())}')
    df = df[~bad]

    months = sorted(df.month.unique())
    uni = {m: set(g) for m, g in df.groupby('month')['ticker']}
    rows = []
    for i in range(1, len(months)):
        T = months[i]
        base = (uni[months[i - 1]] & uni[T]) - ETF
        sub = df[(df.month == T) & df.ticker.isin(base)]
        for day, g in sub.groupby('day'):
            if len(g) < MIN_NAMES:
                continue
            k = len(g) // 5
            o = g.fh_bps.values.argsort()
            r = g.rest_bps.values
            rows.append((day, r[o[-k:]].mean() - r[o[:k]].mean(),
                         r[o[-k:]].mean() - r.mean(), r.mean()))
    A = pd.DataFrame(rows, columns=['day', 'ls', 'tilt', 'ew']).set_index('day')

    print(f'\nSignal #6: cross-sectional intraday continuation '
          f'(rank first hour, hold 10:30 -> close)\n  {len(A)} days')
    stats(A.ls.values, 'L/S Q5-Q1')
    stats(A.tilt.values, 'long tilt Q5-EW')
    stats(A.ew.values, 'EW rest-of-day')
    m = A.ls.mean()
    print(f'  costs: L/S pays 4 crossings/day (10:30 entries cross the spread) '
          f'-> break-even one-way {m/4:.2f} bps;')
    print(f'         net at c=2.5/5: {m-10:+.2f}  {m-20:+.2f} bps/day')
    print('  by 4-year era (L/S):')
    era_table(A.index, A.ls.values)

    p = os.path.join(ROOT, 'data', 'xsec_intraday.csv')
    A.rename(columns={'ls': 'xid_ls', 'tilt': 'xid_tilt', 'ew': 'xid_ew'}) \
     .to_csv(p, float_format='%.3f')
    print(f'daily series -> {p}')


if __name__ == '__main__':
    main()
