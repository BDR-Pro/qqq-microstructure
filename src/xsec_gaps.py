# Part of qqq-microstructure.
#
# Audit 2026-09, finding B1#1: load_panel NaN's overnight gaps that look like
# split ratios (|on|>25% within 3.5% log of a ratio) and everything beyond the
# +/-40% backstop (which in log space fires at -28.6% on the downside). For an
# L/S book that is conservative; for the DEPLOYED long-only Q5 / tilt it is a
# direction-known optimistic bias -- a Q5 name's -25% earnings night vanishes
# from every backtest, replay and Monte-Carlo series while xsec_live's grade
# and fills.py book it in full. The exclusion count was never recorded.
#
# This is a MEASUREMENT, not a fix: the frozen models were trained on the
# masked series and must keep scoring on it (RESULTS 19 clock). It prints:
#   1. the exclusion counts and every censored night that is NOT a verified
#      known split, with ticker/day/size/nearest ratio, so real crashes can be
#      told from splits by eye;
#   2. the rule-B monthly Q5 / Q1 / L/S / tilt-vs-QQQ under the default mask
#      versus with the BACKSTOP nights re-inserted (the band nights are NOT
#      re-inserted: most are genuine splits and re-inserting one 2:1 would
#      inject a -6,931 bps 'night' -- see the listing instead), and the delta,
#      which is a LOWER BOUND on the long-only bias.
#
#   python src/xsec_gaps.py [--signal on_1m|on_12m] [--list 60]

import argparse
import numpy as np, pandas as pd
from xsec_backtest import load_panel, KNOWN_SPLITS, SPLIT_LOGR, ETF
from xsec_replicate import signal_rows


def monthly_legs(df, signal):
    rows = signal_rows(df, signal)
    q = df[df.ticker.isin({'QQQ', 'QQQQ'})].sort_values(['day', 'ticker']) \
          .groupby('day').first()
    qm = q.groupby('month').on_bps.mean()
    out = {}
    for T, g in rows.groupby('tmonth'):
        n = len(g)
        if n < 20:
            continue
        k = max(4, n // 5)
        s = g.sort_values('sig')
        out[T] = dict(q5=s.realized.iloc[-k:].mean(),
                      q1=s.realized.iloc[:k].mean(), qqq=qm.get(T, np.nan))
    o = pd.DataFrame(out).T
    o['ls'] = o.q5 - o.q1
    o['tilt'] = o.q5 - o.qqq
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signal', choices=['on_1m', 'on_12m'], default='on_1m')
    ap.add_argument('--list', type=int, default=60,
                    help='print the N largest censored nights')
    a = ap.parse_args()

    base = load_panel()
    kept = load_panel(keep_backstop=True)
    known = {(t, d) for t, d, _ in KNOWN_SPLITS}
    on_raw = np.log(base.open / base.prev_close) * 1e4
    censored = base[base.on_bps.isna() & on_raw.notna()
                    & (base.dpos - base.groupby('ticker').dpos.shift() == 1)].copy()
    censored['on_raw'] = on_raw[censored.index]
    censored['near_ratio'] = np.exp(SPLIT_LOGR[np.abs(
        censored.on_raw.values[:, None] / 1e4 - SPLIT_LOGR[None, :]).argmin(1)])
    censored['known'] = [(t, d) in known for t, d in
                         zip(censored.ticker, censored.day)]
    band = censored[np.abs(censored.on_raw) <= np.log(1.40) * 1e4]
    back = censored[np.abs(censored.on_raw) > np.log(1.40) * 1e4]
    print(f'\ncensored nights: {len(censored):,} total = {len(band):,} split-'
          f'band + {len(back):,} backstop; {int(censored.known.sum())} are '
          f'verified known splits')
    print(f'  backstop nights re-inserted below: {len(back):,} '
          f'({int((back.on_raw < 0).sum())} negative, '
          f'{int((back.on_raw > 0).sum())} positive)')
    show = censored[~censored.known].reindex(
        censored[~censored.known].on_raw.abs().sort_values(ascending=False).index)
    print(f'\n  largest {min(a.list, len(show))} censored nights that are NOT '
          f'verified splits (sign tells crash from split; a real split sits '
          f'within 3.5% of its ratio):')
    for r in show.head(a.list).itertuples():
        kind = 'backstop' if abs(r.on_raw) > np.log(1.40) * 1e4 else 'band'
        print(f'    {r.ticker:<7} {r.day}  {r.on_raw/100:+8.1f}%  '
              f'nearest ratio {r.near_ratio:6.2f}  [{kind}]')

    m0 = monthly_legs(base, a.signal)
    m1 = monthly_legs(kept, a.signal)
    j = m0.join(m1, lsuffix='_mask', rsuffix='_keep').dropna()
    print(f'\nrule-B ({a.signal}) monthly legs, {len(j)} months, bps/day: '
          f'default mask vs backstop nights re-inserted')
    print(f'{"":8}{"masked":>9}{"kept":>9}{"delta":>8}')
    for c in ('q5', 'q1', 'ls', 'tilt'):
        d = j[f'{c}_keep'] - j[f'{c}_mask']
        tt = d.mean() / (d.std() / np.sqrt(len(d))) if d.std() > 0 else float('nan')
        print(f'  {c:<6}{j[f"{c}_mask"].mean():>+9.2f}{j[f"{c}_keep"].mean():>+9.2f}'
              f'{d.mean():>+8.2f}   (t of delta '
              + (f'{tt:+.1f}' if np.isfinite(tt) else 'n/a: no re-inserted night landed in a leg') + ')')
    print('\n  delta on q5/tilt is the LOWER BOUND on the long-only censoring '
          'bias (band nights\n  not re-inserted). Paste this block into RESULTS '
          '15; the frozen path is unchanged.')


if __name__ == '__main__':
    main()
