# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
#
# Daily spine for the 0DTE work: one row per trading day, built from the tick data.
#
# A 0DTE trade opened at the bell and closed at the bell is fully described by the
# open-to-close path of the underlying, so this collapses each day to the handful of
# numbers that decide such a trade: where it opened, where it closed, how far it ranged,
# how much it actually moved, and what the first minutes looked like (the only
# information a strategy could condition on without peeking).
#
# 515 days is a small sample. Everything downstream of this file has to be judged with
# that in mind -- it is four orders of magnitude less data than the microstructure work.

import os, sys, zipfile
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, databento as db
from concurrent.futures import ProcessPoolExecutor

np.seterr(all='ignore')

def session_start_ns(first_ts_ns):
    """UTC nanoseconds of 09:30 America/New_York on the day containing `first_ts_ns`.

    Deriving this from event counts is unsafe and was wrong here for 169 of 515 days.
    Under EST a 13:30-20:00 UTC window still clears any activity threshold while
    actually spanning 08:30-15:00 ET -- silently trading an hour of pre-market for the
    closing hour. Compute it from the calendar instead.
    """
    d = dt.datetime.fromtimestamp(first_ts_ns / 1e9, dt.timezone.utc).date()
    et = dt.datetime.combine(d, dt.time(9, 30), ZoneInfo('America/New_York'))
    return int(et.astimezone(dt.timezone.utc).timestamp()) * 10**9


ZIP = os.environ.get("QQQ_DBN_ZIP",
    os.path.expanduser("~/Downloads/XNAS-20251009-UVUB86RLRM.zip"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'data', 'daily.parquet')
WORK = os.path.join(ROOT, 'data', '_work_daily')

# minutes after the open at which we snapshot the price (opening-range features)
SNAPS = [1, 5, 10, 15, 30, 60, 120]


def one_day(name):
    day = name.split('-')[2].split('.')[0]
    tmp = os.path.join(WORK, f'{day}.zst')
    try:
        with zipfile.ZipFile(ZIP) as z, open(tmp, 'wb') as fo:
            fo.write(z.read(name))
        d = db.DBNStore.from_file(tmp).to_df().reset_index()
        d = d[(d['flags'].values.astype(np.uint8) & 128) != 0]
        ts = d.ts_recv.values.astype('datetime64[ns]').astype(np.int64)
        d0 = (ts[0] // 86_400_000_000_000) * 86_400_000_000_000
        s_ns = session_start_ns(ts[0])
        m = (ts >= s_ns) & (ts < s_ns + 390 * 60 * 10**9)
        if m.sum() < 10_000:
            return None
        # pre-market, for the overnight gap as it was observable at the bell
        pre = ts < s_ns
        m = (ts >= s_ns) & (ts < s_ns + 390 * 60 * 10**9)

        bp, ap = d.bid_px_00.values, d.ask_px_00.values
        bs, as_ = d.bid_sz_00.values.astype(float), d.ask_sz_00.values.astype(float)
        ok = np.isfinite(bp) & np.isfinite(ap) & (ap > bp) & (bs > 0) & (as_ > 0)
        mid_all, spr_all = (bp + ap) / 2.0, ap - bp

        r = {'day': day}
        pm = ok & pre
        r['premkt_last'] = float(mid_all[pm][-1]) if pm.sum() > 50 else np.nan
        r['premkt_vol'] = float(d['size'].values[(d.action.values == 'T') & pm].sum()) \
            if pm.sum() else 0.0

        sel = ok & m
        tsr, mid, spr = ts[sel], mid_all[sel], spr_all[sel]
        act, sz = d.action.values[sel], d['size'].values[sel].astype(float)
        n = len(tsr)
        if n < 10_000:
            return None

        r['open'] = float(mid[0])
        r['close'] = float(mid[-1])
        r['high'] = float(mid.max())
        r['low'] = float(mid.min())
        r['oc_bps'] = float(np.log(mid[-1] / mid[0]) * 1e4)
        r['range_bps'] = float(np.log(mid.max() / mid.min()) * 1e4)
        isT = act == 'T'
        r['volume'] = float(sz[isT].sum())
        r['notional'] = float((sz[isT] * mid[isT]).sum())
        r['spread_bps'] = float(np.mean(spr / mid) * 1e4)

        for k in SNAPS:
            j = np.searchsorted(tsr, s_ns + k * 60 * 10**9, 'left')
            j = min(j, n - 1)
            r[f'p{k}'] = float(mid[j])
            r[f'ret{k}_bps'] = float(np.log(mid[j] / mid[0]) * 1e4)
            hi = mid[:j + 1].max() if j > 0 else mid[0]
            lo = mid[:j + 1].min() if j > 0 else mid[0]
            r[f'range{k}_bps'] = float(np.log(hi / lo) * 1e4)

        # realised vol from 1-minute mid returns, annualised-free (bps per minute)
        grid = np.arange(s_ns, s_ns + 390 * 60 * 10**9, 60 * 10**9)
        gi = np.minimum(np.searchsorted(tsr, grid, 'right') - 1, n - 1)
        gi = np.maximum(gi, 0)
        mr = np.diff(np.log(mid[gi])) * 1e4
        mr = mr[np.isfinite(mr)]
        r['rv_1m_bps'] = float(np.std(mr)) if len(mr) > 10 else np.nan
        r['rv_day_bps'] = float(np.std(mr) * np.sqrt(390)) if len(mr) > 10 else np.nan
        # how much of the day's range the close gave back -- trend vs chop
        r['close_loc'] = float((mid[-1] - mid.min()) / max(mid.max() - mid.min(), 1e-9))
        return r
    except Exception as ex:
        print(f'  {day}: {type(ex).__name__}: {ex}', flush=True)
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == '__main__':
    os.makedirs(WORK, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.dbn.zst'))
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sel = names[::stride]
    print(f'building daily spine over {len(sel)} days', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=5) as ex:
        for i, r in enumerate(ex.map(one_day, sel), 1):
            if r:
                rows.append(r)
            if i % 50 == 0:
                print(f'  {i}/{len(sel)}', flush=True)
    df = pd.DataFrame(rows).sort_values('day').reset_index(drop=True)
    # features that need the previous day
    df['prev_close'] = df['close'].shift(1)
    df['gap_bps'] = np.log(df['open'] / df['prev_close']) * 1e4
    df['prev_oc_bps'] = df['oc_bps'].shift(1)
    df['prev_rv'] = df['rv_1m_bps'].shift(1)
    df['dow'] = pd.to_datetime(df.day, format='%Y%m%d').dt.dayofweek
    df.to_parquet(OUT, index=False)
    print(f'\n{len(df)} days -> {os.path.relpath(OUT, ROOT)}')
    print(df[['day', 'open', 'close', 'oc_bps', 'range_bps', 'rv_1m_bps', 'gap_bps']]
          .tail(5).to_string(index=False))
