# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
#
# Can a directional QQQ strategy hit a 75% win rate, and does it matter?
#
# Triple-barrier trades: enter aggressively (crossing the spread), exit on a profit
# target of T ticks, a stop of S ticks, or a time limit H -- whichever comes first.
# Direction comes from queue imbalance, the strongest short-horizon signal in this
# dataset (IC 0.236 @100ms).
#
# Two rates are reported and they are not the same thing:
#   hit_rate  -- fraction of trades where the target barrier was reached before the stop
#   win_rate  -- fraction of trades that made money AFTER paying the spread and fees
#
# The gap between them is the entire lesson. A 1-tick target on a 1-tick spread is hit
# constantly and nets nothing, because you paid a tick to get in and out.

import os, sys, zipfile, argparse
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



ZIP = os.environ.get("QQQ_DBN_ZIP", os.path.join(
    os.path.expanduser("~"), "OneDrive", "المستندات",
    "Trading", "XNAS-20251009-UVUB86RLRM.zip"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, 'data', '_work_winrate')

TAKER_FEE_PER_SHARE = 0.0030      # Nasdaq remove-liquidity fee, $/share
TICK = 0.01
N_ENTRIES = 12_000                # sampled entries per day
QI_GATE = 0.30                    # only trade when imbalance is at least this strong

GRID_T = [1, 2, 3, 5, 10]         # profit target, ticks
GRID_S = [1, 2, 3, 5, 10, 20]     # stop, ticks
GRID_H = [10, 60, 300]            # time limit, seconds


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
        d, ts = d[m], ts[m]
        bp, ap = d.bid_px_00.values, d.ask_px_00.values
        bs, as_ = d.bid_sz_00.values.astype(float), d.ask_sz_00.values.astype(float)
        ok = np.isfinite(bp) & np.isfinite(ap) & (ap > bp) & (bs > 0) & (as_ > 0)
        ts, bp, ap, bs, as_ = ts[ok], bp[ok], ap[ok], bs[ok], as_[ok]
        n = len(ts)
        if n < 50_000:
            return None
        mid = (bp + ap) / 2.0
        spr = ap - bp
        qi = (bs - as_) / (bs + as_)

        # entries: evenly spaced, then gated on a meaningful imbalance
        cand = np.linspace(0, n - 1, N_ENTRIES * 3).astype(int)
        cand = cand[np.abs(qi[cand]) >= QI_GATE][:N_ENTRIES]
        if len(cand) < 500:
            return None
        direction = np.sign(qi[cand])          # imbalance leans -> trade that way

        rows = []
        for H in GRID_H:
            end = np.minimum(np.searchsorted(ts, ts[cand] + H * 10**9, 'left'), n - 1)
            for T in GRID_T:
                for S in GRID_S:
                    hit = np.zeros(len(cand), dtype=np.int8)   # +1 target, -1 stop, 0 timeout
                    exit_mid = np.empty(len(cand))
                    for k, (i, j, dr) in enumerate(zip(cand, end, direction)):
                        if j <= i:
                            exit_mid[k] = mid[i]
                            continue
                        path = mid[i:j + 1]
                        up = mid[i] + dr * T * TICK
                        dn = mid[i] - dr * S * TICK
                        a = (path >= up) if dr > 0 else (path <= up)
                        b = (path <= dn) if dr > 0 else (path >= dn)
                        ia = a.argmax() if a.any() else len(path)
                        ib = b.argmax() if b.any() else len(path)
                        if ia == ib == len(path):
                            exit_mid[k] = path[-1]
                        elif ia < ib:
                            hit[k] = 1
                            exit_mid[k] = path[ia]
                        else:
                            hit[k] = -1
                            exit_mid[k] = path[ib]
                    gross = direction * (exit_mid - mid[cand]) / mid[cand] * 1e4
                    # cross the spread both ways, plus the taker fee both ways
                    cost = (spr[cand] / mid[cand] * 1e4
                            + 2 * TAKER_FEE_PER_SHARE / mid[cand] * 1e4)
                    net = gross - cost
                    rows.append({
                        'day': day, 'T': T, 'S': S, 'H': H, 'n': len(cand),
                        'hit_rate': float((hit == 1).mean()),
                        'win_rate': float((net > 0).mean()),
                        'expectancy_bps': float(net.mean()),
                        'gross_bps': float(gross.mean()),
                        'cost_bps': float(cost.mean()),
                    })
        return pd.DataFrame(rows)
    except Exception as ex:
        print(f'  {day}: {type(ex).__name__}: {ex}', flush=True)
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.dbn.zst'))
    sel = names[::max(1, len(names) // a.days)][:a.days]
    print(f'triple-barrier study on {len(sel)} days, {N_ENTRIES:,} entries/day, '
          f'|imbalance| >= {QI_GATE}', flush=True)
    with ProcessPoolExecutor(max_workers=5) as ex:
        parts = [p for p in ex.map(one_day, sel) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    g = df.groupby(['H', 'T', 'S']).agg(
        hit_rate=('hit_rate', 'mean'), win_rate=('win_rate', 'mean'),
        expectancy_bps=('expectancy_bps', 'mean'), cost_bps=('cost_bps', 'mean')).reset_index()
    out = os.path.join(ROOT, 'data', 'win_rate_grid.csv')
    g.to_csv(out, index=False)

    print(f'\naverage round-trip cost: {g.cost_bps.mean():.3f} bps '
          f'(full spread + {TAKER_FEE_PER_SHARE*2:.4f}/share fees)\n')
    for H in GRID_H:
        sub = g[g.H == H]
        print(f'=== time limit {H}s ===')
        print(f'{"T/S":>7}' + ''.join(f'{f"S={s}":>16}' for s in GRID_S))
        for T in GRID_T:
            cells = []
            for S in GRID_S:
                r = sub[(sub['T'] == T) & (sub['S'] == S)]   # sub.T is transpose, not a column
                if r.empty:
                    cells.append(f'{"-":>16}')
                else:
                    r = r.iloc[0]
                    cells.append(f'{r.win_rate*100:>6.1f}%{r.expectancy_bps:>+9.3f}')
            print(f'{f"T={T}":>7}' + ''.join(cells))
        print('        (cell = net win rate | expectancy in bps)\n')

    hi = g[g.win_rate >= 0.75]
    print(f'--- cells reaching a 75%+ win rate: {len(hi)} of {len(g)} ---')
    if len(hi):
        hi = hi.sort_values('expectancy_bps', ascending=False)
        print(hi.head(8).to_string(index=False,
              float_format=lambda x: f'{x:.4f}'))
        pos = hi[hi.expectancy_bps > 0]
        print(f'\nof those, profitable after costs: {len(pos)}')
        if len(pos):
            print(pos.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    best = g.sort_values('expectancy_bps', ascending=False).head(5)
    print(f'\n--- highest expectancy regardless of win rate ---')
    print(best.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\ngrid -> {os.path.relpath(out, ROOT)}')


if __name__ == '__main__':
    main()
