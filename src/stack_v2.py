# Part of qqq-microstructure.
#
# Stack v2: add up the book. Every input series here was produced by its own
# script under its own discipline; this file only aligns and sums them, which is
# the one operation that cannot overfit. Near-zero correlations are the entire
# argument -- RESULTS 13's two-leg stack beat both of its legs, and the legs
# below share one pot of capital because they never hold at the same time
# (overnight basket is flat by 09:30, the intraday legs run 10:30 -> close).
#
# The QQQ legs (overnight and the sign-rule intraday momentum) are derived from
# the xsec panel itself -- QQQ/QQQQ are in every month's top-150, and the panel
# carries open/p60/close -- NOT from data/daily_hf_QQQ.parquet. The first run
# used the daily_hf spine and silently got 1999-2005 only, because the committed
# hf_bars are a partial subset; a hot-era MOM leg then inflated every combined
# row. Hence two rules baked in here: QQQ legs come from the same panel as every
# other leg, and EVERY row prints its own date window and day count.
#
# Legs, each included only if its input exists (missing ones are reported):
#
#   ON    overnight basket, ML-Q5 close -> open          data/xsec_ml_daily.csv
#   NEU   market-neutral overnight, ML-Q5 minus QQQ      (panel QQQ overnight)
#   QQQ_ON  reference: QQQ overnight, RESULTS 13's leg   (panel)
#   MOM   QQQ intraday momentum, sign rule 10:30->close  (panel)
#   XID   cross-sectional intraday continuation L/S      data/xsec_intraday.csv
#
# Costs stated, not hidden: basket overnight legs pay 2 single-name auction
# crossings/day at --c bps one-way (default 1.0 per RESULTS 15b's MOO/MOC
# finding); NEU adds a QQQ pair at the house 0.34 bps round trip; MOM pays the
# house 0.34 (overnight_study.py convention); XID's 10:30 entries cross the
# spread, so it pays 4 x --c-intra (default 2.5).
#
#   python src/stack_v2.py [--c 1.0] [--c-intra 2.5]

import os, argparse
import numpy as np, pandas as pd
from xsec_backtest import load_panel, stats, era_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QQQ_RT = 0.34                       # house round-trip cost for QQQ legs


def win(s):
    return f'{s.index[0]}..{s.index[-1]} ({len(s):,}d)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c', type=float, default=1.0,
                    help='one-way bps per single-name auction crossing')
    ap.add_argument('--c-intra', type=float, default=2.5,
                    help='one-way bps per 10:30 spread crossing (XID)')
    a = ap.parse_args()
    d = os.path.join(ROOT, 'data')
    legs = {}

    df = load_panel()
    q = df[df.ticker.isin({'QQQ', 'QQQQ'})].sort_values(['day', 'ticker']) \
          .groupby('day').first()
    legs['QQQ_ON'] = q.on_bps.dropna() - QQQ_RT
    qq = q[q.p60 > 0]
    legs['MOM'] = pd.Series(
        np.sign(np.log(qq.p60 / qq.open)) * np.log(qq.close / qq.p60) * 1e4
        - QQQ_RT, index=qq.index).dropna()

    p = os.path.join(d, 'xsec_ml_daily.csv')
    if os.path.exists(p):
        ml = pd.read_csv(p, dtype={'day': str}).set_index('day')
        if 'mlon_q5' in ml:
            legs['ON'] = (ml.mlon_q5 - 2 * a.c).dropna()
            j = pd.concat([ml.mlon_q5, q.on_bps], axis=1).dropna()
            legs['NEU'] = j.mlon_q5 - j.on_bps - 2 * a.c - QQQ_RT
        else:
            print('xsec_ml_daily.csv predates the q5 column -- re-run xsec_ml.py')
    else:
        print('missing data/xsec_ml_daily.csv -- run xsec_ml.py (skipping ON/NEU)')

    p = os.path.join(d, 'xsec_intraday.csv')
    if os.path.exists(p):
        xi = pd.read_csv(p, dtype={'day': str}).set_index('day')
        legs['XID'] = (xi.xid_ls - 4 * a.c_intra).dropna()
    else:
        print('missing data/xsec_intraday.csv -- run xsec_intraday.py (skipping XID)')

    L = pd.DataFrame(legs).sort_index()
    print(f'\nlegs at c={a.c} / c_intra={a.c_intra} bps one-way, net bps/day:')
    for k in L:
        s = L[k].dropna()
        stats(s.values, k)
        print(f'                     {win(s)}')

    both = L.dropna()
    if len(both) > 500 and len(L.columns) > 1:
        print(f'correlations ({len(both):,} common days):')
        cm = both.corr()
        print('        ' + ''.join(f'{k:>8}' for k in cm))
        for k in cm.index:
            print(f'{k:>8}' + ''.join(f'{cm.loc[k, j]:>8.2f}' for j in cm))

    combos = [('v1  (QQQ_ON+MOM)', ['QQQ_ON', 'MOM']),
              ('v2  (ON+MOM)', ['ON', 'MOM']),
              ('v2n (NEU+MOM)', ['NEU', 'MOM']),
              ('v2x (ON+MOM+XID)', ['ON', 'MOM', 'XID']),
              ('v2nx(NEU+MOM+XID)', ['NEU', 'MOM', 'XID'])]
    print('\ncombined (legs summed on common days -- one pot of capital, '
          'non-overlapping hours; XID is an L/S overlay;\n v1 is RESULTS 13 at '
          'panel prices, on each combo row\'s own window):')
    shown = {}
    for lab, ks in combos:
        miss = [k for k in ks if k not in L]
        if miss:
            print(f'  {lab:<18} skipped (missing {", ".join(miss)})')
            continue
        r = L[ks].dropna().sum(axis=1)
        stats(r.values, lab)
        print(f'                     {win(r)}')
        shown[lab.split()[0]] = r
    # the upgrade comparison must be same-window: v1 restricted to v2's days
    if 'v1' in shown and 'v2' in shown:
        r = shown['v1'].loc[shown['v1'].index.intersection(shown['v2'].index)]
        stats(r.values, 'v1 @ v2 window')
        shown['v1@v2'] = r
    for lab in [k for k in ('v1@v2', 'v2', 'v2x') if k in shown]:
        r = shown[lab]
        print(f'\n  by 4-year era ({lab}):')
        era_table(r.index, r.values)


if __name__ == '__main__':
    main()
