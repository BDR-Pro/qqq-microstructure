# Part of qqq-microstructure.
#
# The out-of-universe replication -- the strongest external check the overnight
# edge can get from historical data. RESULTS 22 proved the edge is broad WITHIN
# the top-150; this tests it on names that were NEVER in the top-150: the
# liquidity tier below, ranks 151..N, extracted from the same minute source
# with the same per-month-universe rule, so it is survivorship-free the same
# way the core panel is (dead names are present in the months they were alive).
#
# Needs the wide panel first (heavy, one-time, resumable -- re-downloads each
# month's minute file and keeps the top N by that month's dollar volume):
#
#   python src/xsec_extract.py --top 1000 --out data/xsec1000
#
# Then this runs the SAME persistence rule (xsec_replicate's signal_rows /
# ls_series, unchanged) on three disjoint tiers, each ranked only against
# itself:
#
#   CORE       names ranked <=150 that month (should reproduce RESULTS 22)
#   MID        monthly rank 151-400, and STRICTLY never <=150 in ANY month
#   DEEP       monthly rank 401+,   same strict never-core condition
#
# The strict never-core condition means MID/DEEP names share NOTHING with the
# panel the models were built on -- no month, no name. If the edge shows up
# there, it is a property of the US equity market's overnight session, not of
# the mega-cap tier; if it does not, the edge is tier-specific (still
# tradeable in its tier, but the mechanism is tied to index/mega-cap
# structure and should not be extrapolated).
#
# The rank is recomputed inside the wide panel by each month's own dollar
# volume -- the same rule the extraction used to pick names, so tier
# membership uses only information from that month. Same cleaning as the core
# panel (load_panel's split exclusion etc., pointed at the wide dir).
#
# Validated on planted truth: on a synthetic wide panel with persistence
# planted in every name, CORE, MID and DEEP all replicate and the tier report
# reads BROAD; the never-core condition is verified to exclude every name
# that ever ranks <=150 (see RESULTS).
#
#   python src/xsec_wide.py [--dir data/xsec1000] [--signal on_12m]

import os, argparse
import numpy as np, pandas as pd
from xsec_backtest import load_panel
from xsec_replicate import signal_rows, ls_series, report_subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_N = 150


def monthly_rank(df):
    """Each name's rank by ITS month's total dollar volume within the wide
    panel -- the extraction's own selection rule, month-local information."""
    dv = df.groupby(['month', 'ticker']).dollar_vol.sum().reset_index()
    dv['rank'] = dv.groupby('month').dollar_vol.rank(ascending=False,
                                                     method='first')
    return dv.set_index(['month', 'ticker'])['rank']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/xsec1000')
    ap.add_argument('--signal', choices=['on_12m', 'on_1m'], default='on_12m')
    a = ap.parse_args()
    path = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    df = load_panel(path)
    rk = monthly_rank(df)

    # tiers by month-local rank; never-core = never ranked <=CORE_N anywhere
    best = rk.groupby('ticker').min()
    never_core = set(best[best > CORE_N].index)
    med = rk.groupby('ticker').median()

    rows = signal_rows(df, a.signal)
    names = set(rows.ticker.unique())
    core = {tk for tk in names if best.get(tk, 1e9) <= CORE_N}
    mid = {tk for tk in names if tk in never_core and med[tk] <= 400}
    deep = {tk for tk in names if tk in never_core and med[tk] > 400}
    assert not (mid | deep) & core, 'never-core violated -- tiering bug'

    full_n = rows.tmonth.nunique()
    print(f'\nOUT-OF-UNIVERSE replication -- signal {a.signal}, wide panel '
          f'{df.month.nunique()} months, {df.ticker.nunique()} names')
    print(f'tiers: CORE {len(core)} (ever rank<={CORE_N})   '
          f'MID {len(mid)} (never core, median rank<=400)   '
          f'DEEP {len(deep)} (never core, deeper)')

    res = {}
    for lab, tier in (('CORE', core), ('MID', mid), ('DEEP', deep)):
        sub = rows[rows.ticker.isin(tier)]
        print(f'\n{lab} ({"the RESULTS 22 reference" if lab == "CORE" else "shares NO name with the core panel, ever"}):')
        res[lab] = report_subset(lab, sub, full_n)

    oou = [r for k, r in res.items() if k != 'CORE' and r and r['powered']]
    print()
    if not res.get('CORE'):
        print('CORE tier failed to report -- check the wide panel')
    elif not oou:
        print('VERDICT: out-of-universe tiers too thin to judge -- extract '
              'deeper (--top 1000) or more months')
    else:
        weakest = min(oou, key=lambda r: r['t'])
        if all(r['mean'] > 0 for r in oou) and weakest['t'] > 2:
            print(f'VERDICT: REPLICATES OUT OF UNIVERSE -- weakest never-core '
                  f'tier {weakest["mean"]:+.2f} bps/day (t={weakest["t"]:+.2f}). '
                  f'The edge is a market-wide overnight phenomenon, not a '
                  f'mega-cap quirk.')
        elif all(r['mean'] > 0 for r in oou):
            print(f'VERDICT: WEAK REPLICATION -- right sign everywhere but '
                  f'weakest t only {weakest["t"]:+.2f}; suggestive, not '
                  f'settled.')
        else:
            print('VERDICT: DOES NOT REPLICATE out of universe -- the edge is '
                  'specific to the mega-cap tier. Still tradeable there '
                  '(that IS the traded universe), but do not extrapolate the '
                  'mechanism.')
    print('\n(the traded book is unchanged either way -- this is evidence '
          'about the mechanism,\n not a new strategy; universe changes remain '
          'a deliberate re-freeze per RESULTS 20)')


if __name__ == '__main__':
    main()
