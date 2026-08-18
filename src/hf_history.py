# Part of qqq-microstructure.
#
# Pulls QQQ minute bars from the free HuggingFace dataset `mito0o852/OHLCV-1m`, which
# was validated against the paid Databento tick archive to sub-tick precision (RESULTS
# §10: |mean| price error 0.15-0.39 bps, signal agreement 41/41 days).
#
# The dataset is one Parquet file per month containing EVERY US ticker, so QQQ is ~1 MB
# of a ~330 MB file. Column-pruned remote reads were tested and are slower than a bulk
# download (66s vs 33s per file) because QQQ appears in all 25 row groups and nothing
# can be skipped. So: download, keep the QQQ slice, delete the file, repeat.
#
#   python src/hf_history.py                    # 1999-03 -> 2025-10, resumable
#   python src/hf_history.py --start 2010-01
#   python src/hf_history.py --build            # minute bars -> daily spine
#
# Why this matters: 515 days of tick data cannot resolve a Sharpe-0.9 signal (minimum
# detectable is 1.40). ~6,700 days takes that to ~0.51, and covers 2000, 2008, 2018,
# 2020 and 2022 -- regimes entirely absent from the tick archive.

import os, sys, time, argparse, urllib.request, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'hf_bars')
# Extracting several ETFs costs nothing extra -- the monthly file is downloaded either
# way and holds every US ticker. SPY is the instrument the published intraday-momentum
# result used (Gao, Han, Li & Zhou, JFE 2018), so it is the direct replication target;
# IWM and DIA test whether the effect is a market phenomenon or a QQQ artefact.
TICKERS = ['QQQ', 'SPY', 'IWM', 'DIA']
TMP = os.path.join(ROOT, 'data', '_work_hf')
DAILY = os.path.join(ROOT, 'data', 'daily_hf_{tk}.parquet')
URL = ('https://huggingface.co/datasets/mito0o852/OHLCV-1m/resolve/main/'
       'data/ohlcv_{m}.parquet')
SNAPS = [1, 5, 10, 15, 30, 60, 120]


def months(start, end):
    a = dt.date(int(start[:4]), int(start[5:7]), 1)
    b = dt.date(int(end[:4]), int(end[5:7]), 1)
    out = []
    while a <= b:
        out.append(f'{a.year:04d}-{a.month:02d}')
        a = dt.date(a.year + (a.month == 12), a.month % 12 + 1, 1)
    return out


def fetch_month(m):
    """Download one monthly file, keep the wanted tickers, delete it. Returns counts."""
    dests = {tk: os.path.join(OUT, tk, f'{m}.parquet') for tk in TICKERS}
    if all(os.path.exists(d) for d in dests.values()):
        return -1                              # already done
    tmp = os.path.join(TMP, f'{m}.parquet')
    req = urllib.request.Request(URL.format(m=m), headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, 'wb') as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        t = pd.read_parquet(tmp, columns=['timestamp', 'open', 'high', 'low', 'close',
                                          'volume', 'ticker'],
                            filters=[('ticker', 'in', TICKERS)])
        out = {}
        for tk, g in t.groupby('ticker'):
            os.makedirs(os.path.join(OUT, str(tk)), exist_ok=True)
            g.drop(columns=['ticker']).to_parquet(dests[str(tk)], index=False)
            out[str(tk)] = len(g)
        return out
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def build_daily(tk='QQQ'):
    """Collapse the minute bars into the same daily schema as daily.parquet."""
    import glob
    fs = sorted(glob.glob(os.path.join(OUT, tk, '*.parquet')))
    if not fs:
        raise SystemExit('nothing downloaded yet')
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df['et'] = df.timestamp.dt.tz_convert('America/New_York')
    df = df[(df.et.dt.time >= dt.time(9, 30)) & (df.et.dt.time < dt.time(16, 0))]
    df['day'] = df.et.dt.strftime('%Y%m%d')
    df['mfo'] = ((df.et.dt.hour - 9) * 60 + df.et.dt.minute - 30).astype(int)
    df = df.sort_values(['day', 'mfo'])
    rows = []
    for day, g in df.groupby('day', sort=True):
        if len(g) < 300:                       # half day / bad day
            continue
        px = g.set_index('mfo')['open']
        r = {'day': day, 'open': float(g.iloc[0]['open']), 'close': float(g.iloc[-1]['close']),
             'high': float(g.high.max()), 'low': float(g.low.min()),
             'volume': float(g.volume.sum()), 'bars': len(g)}
        r['oc_bps'] = np.log(r['close'] / r['open']) * 1e4
        r['range_bps'] = np.log(r['high'] / r['low']) * 1e4
        for k in SNAPS:
            idx = px.index[px.index <= k]
            v = float(px.loc[idx[-1]]) if len(idx) else r['open']
            r[f'p{k}'] = v
            r[f'ret{k}_bps'] = np.log(v / r['open']) * 1e4
            w = g[g.mfo <= k]
            r[f'range{k}_bps'] = (np.log(w.high.max() / w.low.min()) * 1e4) if len(w) else 0.0
        mr = np.diff(np.log(g['close'].values)) * 1e4
        r['rv_1m_bps'] = float(np.std(mr)) if len(mr) > 10 else np.nan
        rows.append(r)
    out = pd.DataFrame(rows).sort_values('day').reset_index(drop=True)
    out['prev_close'] = out['close'].shift(1)
    out['gap_bps'] = np.log(out['open'] / out['prev_close']) * 1e4
    out['prev_oc_bps'] = out['oc_bps'].shift(1)
    out['prev_rv'] = out['rv_1m_bps'].shift(1)
    out['dow'] = pd.to_datetime(out.day, format='%Y%m%d').dt.dayofweek
    path = DAILY.format(tk=tk)
    out.to_parquet(path, index=False)
    print(f'{tk}: {len(out)} trading days  {out.day.iloc[0]} .. {out.day.iloc[-1]}'
          f'  -> {os.path.relpath(path, ROOT)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='1999-03')     # QQQ inception
    ap.add_argument('--end', default='2025-10')
    ap.add_argument('--build', action='store_true', help='skip download, build the spine')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    if a.build:
        for tk in TICKERS:
            try: build_daily(tk)
            except Exception as ex: print(f'  {tk}: {ex}')
        return
    ms = months(a.start, a.end)
    todo = [m for m in ms
            if not all(os.path.exists(os.path.join(OUT, tk, f'{m}.parquet'))
                       for tk in TICKERS)]
    print(f'{len(ms)} months requested, {len(todo)} still to fetch', flush=True)
    t0 = time.time()
    for i, m in enumerate(todo, 1):
        try:
            n = fetch_month(m)
        except Exception as ex:
            print(f'  {m}: {type(ex).__name__}: {ex}', flush=True)
            continue
        el = time.time() - t0
        eta = el / i * (len(todo) - i)
        got = ' '.join(f'{k}:{v:,}' for k, v in sorted(n.items())) if isinstance(n, dict) else 'cached'
        print(f'  [{i}/{len(todo)}] {m}  {got}   '
              f'elapsed {el/60:.0f}m  eta {eta/60:.0f}m', flush=True)
    print('\ndownload done; building the daily spines', flush=True)
    for tk in TICKERS:
        try:
            build_daily(tk)
        except Exception as ex:
            print(f'  {tk}: {ex}')


if __name__ == '__main__':
    main()
