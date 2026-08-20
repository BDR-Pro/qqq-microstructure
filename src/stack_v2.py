# Part of qqq-microstructure.
#
# Stack v2: add up the book. Every input series here was produced by its own
# script under its own discipline; this file only aligns and sums them, which is
# the one operation that cannot overfit. Near-zero correlations are the entire
# argument -- RESULTS 13's two-leg stack beat both of its legs, and the legs
# below share one pot of capital because they never hold at the same time
# (overnight basket is flat by 09:30, the intraday legs run 10:30 -> close).
#
# Legs, each included only if its input file exists (missing ones are reported):
#
#   ON    overnight basket, ML-Q5 close -> open          data/xsec_ml_daily.csv
#   NEU   market-neutral overnight, ML-Q5 minus QQQ      + data/xsec_daily.csv
#   MOM   QQQ intraday momentum, sign rule 10:30 -> close  data/daily_hf_QQQ.parquet
#   XID   cross-sectional intraday continuation L/S      data/xsec_intraday.csv
#
# Costs stated, not hidden: basket overnight legs pay 2 single-name auction
# crossings/day at --c bps one-way (default 1.0 per RESULTS 15b's MOO/MOC
# finding); NEU adds a QQQ pair at the house 0.34 bps round trip; MOM pays the
# house 0.34 (overnight_study.py convention); XID's 10:30 entries cross the
# spread, so it pays 4 x --c-intra (default 2.5). Change the assumptions on the
# command line, not in the code.
#
#   python src/stack_v2.py [--c 1.0] [--c-intra 2.5]

import os, argparse
import numpy as np, pandas as pd
from xsec_backtest import stats, era_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QQQ_RT = 0.34                       # house round-trip cost for QQQ legs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c', type=float, default=1.0,
                    help='one-way bps per single-name auction crossing')
    ap.add_argument('--c-intra', type=float, default=2.5,
                    help='one-way bps per 10:30 spread crossing (XID)')
    a = ap.parse_args()
    d = os.path.join(ROOT, 'data')
    legs = {}

    p = os.path.join(d, 'xsec_ml_daily.csv')
    if os.path.exists(p):
        ml = pd.read_csv(p, dtype={'day': str}).set_index('day')
        if 'mlon_q5' in ml:
            legs['ON'] = ml.mlon_q5 - 2 * a.c
        else:
            print('xsec_ml_daily.csv predates the q5 column -- re-run xsec_ml.py')
    else:
        print('missing data/xsec_ml_daily.csv -- run xsec_ml.py')

    p = os.path.join(d, 'xsec_daily.csv')
    if os.path.exists(p) and 'ON' in legs:
        bt = pd.read_csv(p, dtype={'day': str}).set_index('day')
        j = pd.concat([ml.mlon_q5, bt.qqq_on], axis=1).dropna()
        legs['NEU'] = j.mlon_q5 - j.qqq_on - 2 * a.c - QQQ_RT
        legs['QQQ_ON'] = bt.qqq_on.dropna() - QQQ_RT   # reference leg, RESULTS 13
    elif 'ON' in legs:
        print('missing data/xsec_daily.csv -- run xsec_backtest.py (skipping NEU)')

    p = os.path.join(d, 'daily_hf_QQQ.parquet')
    if os.path.exists(p):
        q = pd.read_parquet(p).dropna(subset=['p60', 'ret60_bps'])
        mom = (np.sign(q.ret60_bps.values)
               * np.log(q.close.values / q.p60.values) * 1e4 - QQQ_RT)
        legs['MOM'] = pd.Series(mom, index=q.day.astype(str))
    else:
        print('missing data/daily_hf_QQQ.parquet (hf_history.py) -- skipping MOM')

    p = os.path.join(d, 'xsec_intraday.csv')
    if os.path.exists(p):
        xi = pd.read_csv(p, dtype={'day': str}).set_index('day')
        legs['XID'] = xi.xid_ls - 4 * a.c_intra
    else:
        print('missing data/xsec_intraday.csv -- run xsec_intraday.py (skipping XID)')

    if not legs:
        raise SystemExit('no legs available')
    L = pd.DataFrame(legs).sort_index()
    print(f'\nlegs at c={a.c} / c_intra={a.c_intra} bps one-way, net bps/day:')
    for k in L:
        stats(L[k].dropna().values, k)

    both = L.dropna()
    if len(both) > 500 and len(L.columns) > 1:
        print(f'\ncorrelations ({len(both)} common days):')
        cm = both.corr()
        print('        ' + ''.join(f'{k:>8}' for k in cm))
        for k in cm.index:
            print(f'{k:>8}' + ''.join(f'{cm.loc[k, j]:>8.2f}' for j in cm))

    combos = [('v1  (QQQ_ON+MOM)  = RESULTS 13, panel prices', ['QQQ_ON', 'MOM']),
              ('v2  (ON+MOM)', ['ON', 'MOM']),
              ('v2n (NEU+MOM)', ['NEU', 'MOM']),
              ('v2x (ON+MOM+XID)', ['ON', 'MOM', 'XID']),
              ('v2nx(NEU+MOM+XID)', ['NEU', 'MOM', 'XID'])]
    print('\ncombined (legs summed on common days -- one pot of capital, '
          'non-overlapping hours; XID is an L/S overlay):')
    best = None
    for lab, ks in combos:
        if not all(k in L for k in ks):
            continue
        s = L[ks].dropna()
        r = s.sum(axis=1)
        stats(r.values, lab)
        best = (lab, r)
    if best is not None:
        lab, r = best
        print(f'\n  by 4-year era ({lab.split()[0]}):')
        era_table(r.index, r.values)


if __name__ == '__main__':
    main()
