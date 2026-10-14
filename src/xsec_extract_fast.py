# Part of qqq-microstructure.
#
# xsec_extract.py's month collapse, vectorised, for the WIDE panel
# (xsec_wide.py's out-of-universe test needs the top 1000 names every month,
# 1999-03..2026-03). The reference loops over every (ticker, day) group in
# Python: ~150 s a month at 1000 names, ~15 hours for the archive. This does
# the same thing with whole-column operations, several months at a time.
#
# It is not a new definition of the panel: one_month() must return exactly
# the rows xsec_extract.one_month() would -- same universe rule, same session
# filter, same 150-bar floor, same snapshot rule, same column order and
# types -- and tests/test_xsec_extract_fast.py checks that against the
# reference on real and synthetic minute data. xsec_extract.py is unchanged
# and stays the panel's definition; the core top-150 panel keeps using it.
#
#   python src/xsec_extract_fast.py --top 1000 --out data/xsec1000 --jobs 3

import os, sys, argparse, datetime as dt, urllib.request
from multiprocessing import Pool
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xsec_extract as xe

COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ticker']


def collapse(t, m, top_n):
    """The reference's per-(ticker, day) loop, column-wise. t: the month's
    minute bars as read from the source file."""
    # integer clock arithmetic instead of the reference's datetime.time and
    # strftime objects (24M rows a month): the same session filter, day and
    # minute-from-open, a sixth of the time
    et = t.timestamp.dt.tz_convert('America/New_York')
    mins = (et.dt.hour * 60 + et.dt.minute).to_numpy()
    keep = (mins >= 570) & (mins < 960)              # 09:30 <= t < 16:00
    t = t[keep].copy()
    et = et[keep]
    t['dollar'] = t.volume * t.close
    # the reference's exact calls, so ties at the 1000th name fall the same
    top = (t.groupby('ticker')['dollar'].sum()
           .sort_values(ascending=False).head(top_n).index)
    sel = t.ticker.isin(top).to_numpy()
    t = t[sel].copy()
    et = et[sel]
    t['day'] = (et.dt.year * 10000 + et.dt.month * 100 + et.dt.day).to_numpy()
    t['mfo'] = (mins[keep][sel] - 570).astype(int)
    # YYYYMMDD integers sort exactly as the reference's strings do
    t = t.sort_values(['ticker', 'day', 'mfo']).reset_index(drop=True)
    key = ['ticker', 'day']
    # positional first/last rows, like the reference's iloc[0] / iloc[-1]
    # (groupby first()/last() would skip NaNs, iloc does not)
    first = t[~t.duplicated(key, keep='first')].set_index(key)
    last = t[~t.duplicated(key, keep='last')].set_index(key)
    g = t.groupby(key, sort=False)
    out = pd.DataFrame({
        'open': first['open'], 'close': last['close'],
        'high': g.high.max(), 'low': g.low.min(),
        'dollar_vol': g.dollar.sum(), 'bars': g.size()})
    for k in xe.SNAPS:
        w = t[t.mfo <= k]
        snap = w[~w.duplicated(key, keep='last')].set_index(key)['open']
        out[f'p{k}'] = snap.reindex(out.index)
        out[f'p{k}'] = out[f'p{k}'].fillna(out['open']) \
            if out[f'p{k}'].isna().any() else out[f'p{k}']
    out = out[out.bars >= 150].reset_index()
    out['day'] = out['day'].astype(str)
    out.insert(2, 'month', m)
    out = out[['ticker', 'day', 'month', 'open', 'close', 'high', 'low',
               'dollar_vol', 'bars'] + [f'p{k}' for k in xe.SNAPS]]
    for c in ['open', 'close', 'high', 'low', 'dollar_vol'] + \
            [f'p{k}' for k in xe.SNAPS]:
        out[c] = out[c].astype(float)
    out['bars'] = out['bars'].astype(int)
    return out


def download(url, dest, tries=4, timeout=60):
    """The month file, whole or not at all. A connection the far side cuts
    early ends like a normal EOF, so the bytes received are checked against
    Content-Length, and a stalled read times out instead of hanging a
    worker (both happened on the first full run: 32 truncated months and
    three workers stuck for 15 minutes)."""
    import time
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':
                                                       'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r, \
                    open(dest, 'wb') as f:
                want = int(r.headers.get('Content-Length') or -1)
                got = 0
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
            if want < 0 or got == want:
                return got
            err = f'truncated: {got:,} of {want:,} bytes'
        except Exception as ex:
            err = f'{type(ex).__name__}: {ex}'
        if i < tries - 1:
            time.sleep(5 * 2 ** i)
    raise IOError(f'{url}: {err} after {tries} tries')


def one_month(args):
    """Download one month, collapse it, write it; resumable like the
    reference (an existing HF-built month is never redone)."""
    m, out_dir, top_n = args
    dest = os.path.join(out_dir, f'{m}.parquet')
    if os.path.exists(dest):
        try:
            if 'src' not in pd.read_parquet(dest).columns:
                return m, -1
        except Exception:
            return m, -1
    os.makedirs(xe.TMP, exist_ok=True)
    tmp = os.path.join(xe.TMP, f'f{m}.{os.getpid()}.parquet')
    try:
        download(xe.URL.format(m=m), tmp)
        rows = collapse(pd.read_parquet(tmp, columns=COLS), m, top_n)
        part = dest + f'.part{os.getpid()}'
        rows.to_parquet(part, index=False)
        os.replace(part, dest)          # never a half-written month on disk
        return m, len(rows)
    except Exception as ex:
        return m, f'{type(ex).__name__}: {ex}'
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='1999-03')
    ap.add_argument('--end', default='2026-03')
    ap.add_argument('--top', type=int, default=1000)
    ap.add_argument('--out', default=os.path.join(xe.ROOT, 'data',
                                                  'xsec1000'))
    ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    todo = [(m, a.out, a.top) for m in xe.months(a.start, a.end)]
    failed = []
    with Pool(a.jobs) as pool:
        for m, n in pool.imap_unordered(one_month, todo):
            if isinstance(n, str):
                failed.append(m)
            print(f'{m} {n}', flush=True)
    print(f'done: {len(todo) - len(failed)} of {len(todo)} months'
          + (f'; failed (re-run to retry): {" ".join(sorted(failed))}'
             if failed else ''), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
