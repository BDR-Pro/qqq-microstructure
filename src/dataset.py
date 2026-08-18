# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
#
# Builds the labelled training set for the adverse-selection model.
#
# One row per passive fill. Features describe the book from the perspective of the
# side the maker was resting on ("own" vs "opp"), so the model is side-agnostic and a
# buy quote and a sell quote share the same parameters.
#
# Label is the maker's realised markout at 1s in bps -- the P&L of having been filled,
# assuming the position is unwound at the mid one second later. Everything is computed
# from the book state at t-1, the event *before* the trade, so there is no lookahead.
#
# features_from_dbn() is the single feature code path, used by both training extraction
# and evaluation of new files, so eval can never drift from training.

import os, sys, glob, zipfile
import numpy as np, pandas as pd, databento as db
from concurrent.futures import ProcessPoolExecutor

np.seterr(all='ignore')

ZIP = os.environ.get("QQQ_DBN_ZIP",
    os.path.expanduser("~/Downloads/XNAS-20251009-UVUB86RLRM.zip"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'data', 'fills')
WORK = os.path.join(ROOT, 'data', '_work_dataset')

FEATURES = [
    'spread_ticks', 'spread_bps',
    'own_sz_log', 'opp_sz_log', 'own_ct', 'opp_ct', 'own_avg_order',
    'imb_own',                      # (own - opp) / (own + opp); the `fav` signal
    'micro_off',                    # microprice offset from mid, signed toward own side
    'ofi_100ms', 'ofi_1s', 'ofi_10s',
    'tflow_1s',
    'wide_age_log',                 # ms the current >=2-tick state has persisted
    'since_trade_log', 'trade_rate_1s',
    'rv_1s', 'minutes',
]
LABEL = 'markout_bps'
CARRY = ['ts', 'price', 'size', 'maker_side', 'day']   # for the portfolio sim, not inputs


def roll_sum(x, ts, win_ns):
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.searchsorted(ts, ts - win_ns, 'left')
    return c[np.arange(len(x)) + 1] - c[lo]


def features_from_dbn(path, day=None):
    """Feature/label table for one day, from a .dbn.zst file on disk.

    Returns None if the file has too little regular-hours activity to be usable.
    """
    d = db.DBNStore.from_file(path).to_df().reset_index()
    d = d[(d['flags'].values.astype(np.uint8) & 128) != 0]
    if len(d) < 10_000:
        return None
    ts = d.ts_recv.values.astype('datetime64[ns]').astype(np.int64)
    if day is None:
        day = str(np.datetime64(ts[0], 'D')).replace('-', '')
    d0 = (ts[0] // 86_400_000_000_000) * 86_400_000_000_000
    m = None
    for off in (13, 14):                       # 09:30 ET under EDT / EST
        s_ns = d0 + (off * 3600 + 1800) * 10**9
        cand = (ts >= s_ns) & (ts < s_ns + 390 * 60 * 10**9)
        if cand.sum() > 100_000:
            m = cand
            break
    if m is None:
        return None
    d, ts = d[m], ts[m]

    bp, ap = d.bid_px_00.values, d.ask_px_00.values
    bs, as_ = d.bid_sz_00.values.astype(float), d.ask_sz_00.values.astype(float)
    bc, ac = d.bid_ct_00.values.astype(float), d.ask_ct_00.values.astype(float)
    ok = np.isfinite(bp) & np.isfinite(ap) & (ap > bp) & (bs > 0) & (as_ > 0)
    d, ts, bp, ap, bs, as_, bc, ac = (d[ok], ts[ok], bp[ok], ap[ok],
                                      bs[ok], as_[ok], bc[ok], ac[ok])
    n = len(ts)
    if n < 10_000:
        return None

    mid = (bp + ap) / 2.0
    spr = ap - bp
    tick = np.round(spr * 100).astype(int)
    micro = (bp * as_ + ap * bs) / (bs + as_)      # size-weighted microprice

    # order-flow imbalance per Cont-Kukanov-Stoikov, rolled over windows
    dbp, dap = np.diff(bp, prepend=bp[0]), np.diff(ap, prepend=ap[0])
    pbs, pas = np.roll(bs, 1), np.roll(as_, 1)
    pbs[0], pas[0] = bs[0], as_[0]
    e = (np.where(dbp > 0, bs, np.where(dbp == 0, bs - pbs, 0))
         - np.where(dap < 0, as_, np.where(dap == 0, as_ - pas, 0)))
    ofi = {w: roll_sum(e, ts, w) for w in (10**8, 10**9, 10**10)}

    isT = (d.action.values == 'T')
    side = d.side.values
    sgn_agg = np.where(side == 'B', 1.0, -1.0)     # +1 = buy aggressor lifted the ask
    tsz = np.where(isT, d['size'].values.astype(float), 0.0)
    tflow = roll_sum(sgn_agg * tsz, ts, 10**9)
    trate = roll_sum(isT.astype(float), ts, 10**9)

    # age of the current >=2-tick run at every event
    w = tick >= 2
    prev = np.concatenate([[False], w[:-1]])
    idx = np.where(w & ~prev, np.arange(n), 0)
    wide_age = np.where(w, ts - ts[np.maximum.accumulate(idx)], 0)

    # time since the previous trade
    tix = np.flatnonzero(isT)
    since_trade = np.zeros(n, dtype=np.int64)
    if len(tix):
        pos = np.searchsorted(tix, np.arange(n), 'right') - 1
        last_t = np.where(pos >= 0, ts[tix[np.maximum(pos, 0)]], ts[0])
        since_trade = np.maximum(ts - last_t, 0)

    lm = np.log(mid)
    rv = np.sqrt(np.maximum(roll_sum(np.diff(lm, prepend=lm[0])**2, ts, 10**9), 0)) * 1e4

    # ---- one row per passive fill ----
    ti = np.flatnonzero(isT & ((side == 'B') | (side == 'A')))
    ti = ti[ti > 0]
    if len(ti) < 500:
        return None
    pre = ti - 1                                   # book state BEFORE the trade
    sg = sgn_agg[ti]
    # the maker is opposite the aggressor: a buy aggressor means the maker sold
    own_sz = np.where(sg > 0, as_[pre], bs[pre])
    opp_sz = np.where(sg > 0, bs[pre], as_[pre])
    own_ct = np.where(sg > 0, ac[pre], bc[pre])
    opp_ct = np.where(sg > 0, bc[pre], ac[pre])

    j = np.minimum(np.searchsorted(ts, ts[ti] + 10**9, 'left'), n - 1)
    px = d.price.values[ti]
    markout = sg * (px - mid[j]) / mid[ti] * 1e4   # maker P&L in bps

    return pd.DataFrame({
        'spread_ticks':    tick[pre].astype(np.float32),
        'spread_bps':      (spr[pre] / mid[pre] * 1e4).astype(np.float32),
        'own_sz_log':      np.log1p(own_sz).astype(np.float32),
        'opp_sz_log':      np.log1p(opp_sz).astype(np.float32),
        'own_ct':          own_ct.astype(np.float32),
        'opp_ct':          opp_ct.astype(np.float32),
        'own_avg_order':   (own_sz / np.maximum(own_ct, 1)).astype(np.float32),
        'imb_own':         ((own_sz - opp_sz) / (own_sz + opp_sz)).astype(np.float32),
        'micro_off':       (-sg * (micro[pre] - mid[pre]) / mid[pre] * 1e4).astype(np.float32),
        'ofi_100ms':       (-sg * ofi[10**8][pre] / np.maximum(own_sz + opp_sz, 1)).astype(np.float32),
        'ofi_1s':          (-sg * ofi[10**9][pre] / np.maximum(own_sz + opp_sz, 1)).astype(np.float32),
        'ofi_10s':         (-sg * ofi[10**10][pre] / np.maximum(own_sz + opp_sz, 1)).astype(np.float32),
        'tflow_1s':        (-sg * tflow[pre] / 1000.0).astype(np.float32),
        'wide_age_log':    np.log1p(wide_age[pre] / 1e6).astype(np.float32),
        'since_trade_log': np.log1p(since_trade[pre] / 1e6).astype(np.float32),
        'trade_rate_1s':   trate[pre].astype(np.float32),
        'rv_1s':           rv[pre].astype(np.float32),
        'minutes':         ((ts[ti] - s_ns) / 6e10).astype(np.float32),
        LABEL:             markout.astype(np.float32),
        'ts':              ts[ti],
        'price':           px.astype(np.float32),
        'size':            d['size'].values.astype(np.float32)[ti],
        'maker_side':      np.where(sg > 0, 'S', 'B'),
        'day':             day,
    })


def build_day(name):
    """Extract one archive day to parquet. Streams from the zip, deletes the temp file."""
    day = name.split('-')[2].split('.')[0]
    out = os.path.join(OUT, f'{day}.parquet')
    if os.path.exists(out):
        return day, 'cached'
    tmp = os.path.join(WORK, f'{day}.zst')
    try:
        with zipfile.ZipFile(ZIP) as z, open(tmp, 'wb') as fo:
            fo.write(z.read(name))
        t = features_from_dbn(tmp, day)
        if t is None:
            return day, 'thin'
        os.makedirs(OUT, exist_ok=True)
        t.to_parquet(out, index=False)
        return day, f'{len(t):,} fills'
    except Exception as ex:
        return day, f'ERR {type(ex).__name__}: {ex}'
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def day_names():
    with zipfile.ZipFile(ZIP) as z:
        return sorted(n for n in z.namelist() if n.endswith('.dbn.zst'))


def extracted_days():
    return sorted(os.path.basename(f)[:8] for f in glob.glob(os.path.join(OUT, '*.parquet')))


def load(days=None):
    """Load extracted fill tables. `days` filters to a list of YYYYMMDD strings."""
    files = sorted(glob.glob(os.path.join(OUT, '*.parquet')))
    if days is not None:
        keep = set(days)
        files = [f for f in files if os.path.basename(f)[:8] in keep]
    if not files:
        raise SystemExit(f'no extracted fills in {OUT} -- run: python src/dataset.py')
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    names = day_names()
    # hold the most recent HOLDOUT_DAYS out of the training pool entirely; they are the
    # stand-in for "future data" until real post-archive files arrive in eval_data/
    holdout = int(os.environ.get('HOLDOUT_DAYS', '12'))
    pool = names[:-holdout] if holdout else names
    sel = pool[::stride]
    print(f'{len(names)} days in archive | {holdout} held out for eval '
          f'| extracting {len(sel)} for training (stride {stride})', flush=True)
    with ProcessPoolExecutor(max_workers=5) as ex:
        for i, (day, st) in enumerate(ex.map(build_day, sel), 1):
            print(f'[{i}/{len(sel)}] {day} {st}', flush=True)
