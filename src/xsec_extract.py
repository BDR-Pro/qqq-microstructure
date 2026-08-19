# Part of qqq-microstructure.
#
# Cross-sectional extraction: for every month, keep the TOP-N US tickers by that
# month's dollar volume, collapsed to daily rows during extraction.
#
# Two design decisions that carry the statistics:
#
#   1. The universe is chosen PER MONTH by that month's own dollar volume, so dead and
#      delisted names are in the sample for the months they were alive. Choosing
#      today's liquid names would be survivorship bias. A backtest must still LAG the
#      universe by one month (trade month M+1 on month M's list) because month M's
#      volume ranking is only fully known at M's end.
#
#   2. Minute bars are collapsed to daily rows immediately: open/close/high/low,
#      snapshot prices at 15/30/60 minutes after the open, first-hour range, overnight
#      is computable via prev close downstream. 150 tickers x 27 years fits in ~60 MB
#      where minute bars would be ~50 GB.
#
#   python src/xsec_extract.py                  # 1999-03 -> 2026-03, resumable

import os, sys, time, argparse, urllib.request, datetime as dt
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'xsec')
TMP = os.path.join(ROOT, 'data', '_work_hf')
URL = ('https://huggingface.co/datasets/mito0o852/OHLCV-1m/resolve/main/'
       'data/ohlcv_{m}.parquet')
TOP_N = 150
SNAPS = [15, 30, 60]


def months(start, end):
    a = dt.date(int(start[:4]), int(start[5:7]), 1)
    b = dt.date(int(end[:4]), int(end[5:7]), 1)
    out = []
    while a <= b:
        out.append(f'{a.year:04d}-{a.month:02d}')
        a = dt.date(a.year + (a.month == 12), a.month % 12 + 1, 1)
    return out


def one_month(m):
    dest = os.path.join(OUT, f'{m}.parquet')
    if os.path.exists(dest):
        return -1
    tmp = os.path.join(TMP, f'x{m}.{os.getpid()}.parquet')
    req = urllib.request.Request(URL.format(m=m), headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, 'wb') as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        t = pd.read_parquet(tmp, columns=['timestamp', 'open', 'high', 'low',
                                          'close', 'volume', 'ticker'])
        t['et'] = t.timestamp.dt.tz_convert('America/New_York')
        t = t[(t.et.dt.time >= dt.time(9, 30)) & (t.et.dt.time < dt.time(16, 0))]
        t['dollar'] = t.volume * t.close
        top = (t.groupby('ticker')['dollar'].sum()
                .sort_values(ascending=False).head(TOP_N).index)
        t = t[t.ticker.isin(top)].copy()
        t['day'] = t.et.dt.strftime('%Y%m%d')
        t['mfo'] = ((t.et.dt.hour - 9) * 60 + t.et.dt.minute - 30).astype(int)
        t = t.sort_values(['ticker', 'day', 'mfo'])
        rows = []
        for (tk, day), g in t.groupby(['ticker', 'day'], sort=False):
            if len(g) < 150:                   # need a reasonably full session
                continue
            r_ = {'ticker': tk, 'day': day, 'month': m,
                  'open': float(g['open'].iloc[0]), 'close': float(g['close'].iloc[-1]),
                  'high': float(g.high.max()), 'low': float(g.low.min()),
                  'dollar_vol': float(g.dollar.sum()), 'bars': len(g)}
            for k in SNAPS:
                w = g[g.mfo <= k]
                r_[f'p{k}'] = float(w['open'].iloc[-1]) if len(w) else r_['open']
            rows.append(r_)
        pd.DataFrame(rows).to_parquet(dest, index=False)
        return len(rows)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='1999-03')
    ap.add_argument('--end', default='2026-03')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    ms = months(a.start, a.end)
    todo = [m for m in ms if not os.path.exists(os.path.join(OUT, f'{m}.parquet'))]
    print(f'{len(ms)} months, {len(todo)} to fetch | top {TOP_N} by monthly $ volume',
          flush=True)
    t0 = time.time()
    for i, m in enumerate(todo, 1):
        try:
            n = one_month(m)
        except Exception as ex:
            print(f'  {m}: {type(ex).__name__}: {ex}', flush=True)
            continue
        el = time.time() - t0
        print(f'  [{i}/{len(todo)}] {m}: {n:,} ticker-days   '
              f'elapsed {el/60:.0f}m  eta {el/i*(len(todo)-i)/60:.0f}m', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
