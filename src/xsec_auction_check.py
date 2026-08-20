# Part of qqq-microstructure.
#
# RESULTS 15 ends on a range: the overnight-persistence spread is +10.0 bps/day if
# you can sell the 09:30 print and +4.8 if you sell at 09:45, and minute-bar trade
# data cannot say which one a market-on-open order would actually collect. This
# script brings in the independent price source that can: Yahoo's official daily
# open and close, which for US equities are the primary-exchange auction prints.
#
# Method: compare OVERNIGHT RETURNS, not price levels. Yahoo back-adjusts all OHLC
# for splits, so levels differ by cumulative split factors -- but the one-day ratio
# open_t / close_{t-1} is unaffected away from split days, and the panel's split
# days are already excluded. Dividends are in neither source's overnight, so the
# convention matches. For the most-frequent Q5 and Q1 members of signal B since
# --start, the script reports:
#
#   1. agreement: |panel overnight - yahoo overnight| per matched name-day;
#   2. the bounce loading: the SIGNED gap (panel minus yahoo) for Q5-frequent vs
#      Q1-frequent names -- a positive Q5-minus-Q1 gap is the opening print
#      inflating the measured premium;
#   3. the verdict: B's monthly Q5-Q1 overnight recomputed on the IDENTICAL
#      matched name-days with panel prices and with yahoo official prices. The
#      difference between those two lines is the part of the spread that an
#      auction order does not collect.
#
# Coverage caveat: Yahoo serves listed names; delisted members drop out, so run
# this on a recent window (default 2012-01) and read it as a price-source test,
# not a full-history replication. The panel numbers on the matched sample will
# differ from RESULTS 15's for that reason alone.
#
#   pip install yfinance
#   python src/xsec_auction_check.py [--start 2012-01] [--names 120]

import os, argparse
from collections import Counter
import numpy as np, pandas as pd
from xsec_backtest import load_panel, stats, ETF, MIN_NAMES, MIN_ON_DAYS


def b_memberships(df, start):
    months = sorted(df.month.unique())
    uni = {m: set(g) for m, g in df.groupby('month')['ticker']}
    agg = df.groupby(['ticker', 'month']).agg(
        on_mean=('on_bps', 'mean'), on_n=('on_bps', 'count')).to_dict('index')
    mem = {}
    for i in range(1, len(months)):
        T = months[i]
        if T < start:
            continue
        base = (uni[months[i - 1]] & uni[T]) - ETF
        elig, sig = [], []
        for tk in sorted(base):
            r = agg.get((tk, months[i - 1]))
            if r is None or r['on_n'] < MIN_ON_DAYS or np.isnan(r['on_mean']):
                continue
            elig.append(tk)
            sig.append(r['on_mean'])
        if len(elig) < MIN_NAMES:
            continue
        o = np.argsort(sig)
        k = len(elig) // 5
        mem[T] = ({elig[j] for j in o[-k:]}, {elig[j] for j in o[:k]})
    return mem


def fetch_yahoo(tickers, d0, d1):
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit('pip install yfinance')
    ym = {tk: tk.replace('.', '-') for tk in tickers}
    data = yf.download(list(ym.values()), start=d0, end=d1, auto_adjust=False,
                       actions=False, group_by='ticker', progress=False,
                       threads=True)
    yon, ok = {}, []
    for tk, ytk in ym.items():
        try:
            s = data[ytk][['Open', 'Close']].dropna()
        except KeyError:
            continue
        if len(s) < 100:
            continue
        ok.append(tk)
        r = np.log(s.Open / s.Close.shift(1)) * 1e4
        for day, v in zip(s.index.strftime('%Y%m%d'), r.values):
            if np.isfinite(v):
                yon[(tk, day)] = v
    return yon, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2012-01')
    ap.add_argument('--names', type=int, default=120)
    a = ap.parse_args()
    df = load_panel()
    mem = b_memberships(df, a.start)
    if not mem:
        raise SystemExit(f'no trade months >= {a.start}')
    f5, f1 = Counter(), Counter()
    for q5, q1 in mem.values():
        f5.update(q5)
        f1.update(q1)
    half = a.names // 2
    chosen = {tk: 'q5' for tk, _ in f5.most_common(half)}
    for tk, _ in f1.most_common(half):
        chosen.setdefault(tk, 'q1')
    print(f'\n{len(mem)} trade months >= {a.start}; fetching '
          f'{len(chosen)} names (most-frequent Q5/Q1 members)')

    p = df[df.ticker.isin(chosen) & (df.month >= a.start) & df.on_bps.notna()]
    d0 = pd.to_datetime(p.day.min()) - pd.Timedelta(days=7)
    d1 = pd.to_datetime(p.day.max()) + pd.Timedelta(days=2)
    yon, ok = fetch_yahoo(sorted(chosen), d0.date(), d1.date())
    print(f'yahoo resolved {len(ok)}/{len(chosen)} names')

    rows = p[p.ticker.isin(ok)].copy()
    rows['yon'] = [yon.get((tk, day), np.nan)
                   for tk, day in zip(rows.ticker, rows.day)]
    rows = rows.dropna(subset=['yon'])
    rows = rows[rows.yon.abs() < 2500]
    diff = rows.on_bps - rows.yon
    print(f'\nmatched name-days: {len(rows):,}   '
          f'corr(panel, yahoo overnight) {np.corrcoef(rows.on_bps, rows.yon)[0,1]:.4f}')
    print(f'|panel - yahoo|: mean {diff.abs().mean():.2f} bps   '
          f'median {diff.abs().median():.2f} bps')

    grp = rows.ticker.map(chosen)
    g5, g1 = diff[grp == 'q5'], diff[grp == 'q1']
    print(f'signed gap (panel - yahoo): Q5-frequent {g5.mean():+.2f} bps   '
          f'Q1-frequent {g1.mean():+.2f} bps   '
          f'bounce loading (Q5-Q1) {g5.mean()-g1.mean():+.2f} bps/day')

    pan, yah = [], []
    for T, g in rows.groupby('month'):
        if T not in mem:
            continue
        q5, q1 = mem[T]
        m5, m1 = g[g.ticker.isin(q5)], g[g.ticker.isin(q1)]
        for day, gg in g.groupby('day'):
            a5, a1 = m5[m5.day == day], m1[m1.day == day]
            if len(a5) < 5 or len(a1) < 5:
                continue
            pan.append(a5.on_bps.mean() - a1.on_bps.mean())
            yah.append(a5.yon.mean() - a1.yon.mean())
    print(f'\nB on the identical matched name-days ({len(pan)} days):')
    stats(np.array(pan), 'L/S panel prices')
    stats(np.array(yah), 'L/S yahoo official')
    gap = np.mean(pan) - np.mean(yah)
    print(f'  the print component at official prices: {gap:+.2f} bps/day')
    print('  yahoo ~= panel  -> the ceiling is real: a market-on-open order '
          'collects it\n  yahoo << panel  -> the premium is the first-bar print; '
          'underwrite the floor')


if __name__ == '__main__':
    main()
