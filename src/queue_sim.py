# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
#
# PHASE 4 -- the honest fill simulation.
#
# Every P&L figure elsewhere in this repo assumes you were filled on each trade you
# wanted. You would not have been. You join the back of a queue you cannot see, and the
# fills that reach you are the ones the faster participants ahead of you declined.
#
# This simulator models that. It walks the event stream maintaining an explicit queue
# position for a resting order on each side:
#
#   post      -> queue_ahead = the displayed size you are joining behind
#   trade     -> queue_ahead -= traded size; you fill only once it goes negative
#   cancel    -> queue_ahead -= cancels * (queue_ahead / level_size), i.e. cancels are
#                assumed uniformly distributed through the queue. --cancel-ahead 0
#                switches to the pessimistic assumption that they were all behind you.
#   price move-> your order is no longer at the touch; cancel and re-post at the back
#                of the new level, losing the position you had built.
#
# Latency enters twice: your decision is made on a view of the book `latency` old, and
# an order posted at t joins the queue as it stood at t + latency.
#
# Fees are per rebate tier, not the top tier -- a new entrant does not have the top tier.

import os, sys, json, argparse, datetime as dt
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, model as M

ROOT = dataset.ROOT
WORK = os.path.join(ROOT, 'data', '_work_queue')
REPORT_DIR = os.path.join(ROOT, 'reports')

# Nasdaq add-liquidity rebates, $/share. A new entrant earns the bottom of this range.
REBATE_TIERS = {
    'top (>0.9% of consolidated volume)': 0.00305,
    'mid': 0.00250,
    'entry': 0.00200,
    'base (no tier)': 0.00150,
    'none': 0.0,
}


def simulate_day(A, pred_bid, pred_ask, our_size, latency_ns, threshold,
                 rebate, cancel_ahead, unwind_ns=10**9):
    """Walk one day maintaining real queue positions. Returns per-fill records."""
    n = A['n']
    ts, bp, ap, bs, as_ = A['ts'], A['bp'], A['ap'], A['bs'], A['as_']
    mid, price, size, isT, side = A['mid'], A['price'], A['size'], A['isT'], A['side']

    # decision view is `latency` old; a post joins the queue as of now + latency
    lag = np.searchsorted(ts, ts - latency_ns, 'left')
    arr = np.minimum(np.searchsorted(ts, ts + latency_ns, 'left'), n - 1)
    unwind = np.minimum(np.searchsorted(ts, ts + unwind_ns, 'left'), n - 1)

    # python lists are markedly faster than numpy scalar indexing in a tight loop
    L_ts, L_bp, L_ap = ts.tolist(), bp.tolist(), ap.tolist()
    L_bs, L_as = bs.tolist(), as_.tolist()
    L_mid, L_px, L_sz = mid.tolist(), price.tolist(), size.tolist()
    L_isT, L_side = isT.tolist(), side.tolist()
    L_lag, L_arr, L_unw = lag.tolist(), arr.tolist(), unwind.tolist()
    L_pb, L_pa = pred_bid.tolist(), pred_ask.tolist()

    # resting order state per side: 0 = bid (we buy), 1 = ask (we sell)
    act = [False, False]
    qpx = [0.0, 0.0]
    qah = [0.0, 0.0]          # shares ahead of us
    posted = [0, 0]

    fills = []
    prev_bs, prev_as = L_bs[0], L_as[0]
    prev_bp, prev_ap = L_bp[0], L_ap[0]

    for i in range(1, n):
        b_px, a_px = L_bp[i], L_ap[i]
        b_sz, a_sz = L_bs[i], L_as[i]
        traded = L_sz[i] if L_isT[i] else 0.0
        tside = L_side[i] if L_isT[i] else ''
        m_i = L_mid[i]

        for s in (0, 1):
            lvl_px = b_px if s == 0 else a_px
            lvl_sz = b_sz if s == 0 else a_sz
            prev_sz = prev_bs if s == 0 else prev_as
            prev_px = prev_bp if s == 0 else prev_ap

            if act[s]:
                if qpx[s] != lvl_px:
                    act[s] = False          # market moved; our level is gone
                else:
                    # a sell aggressor ('A') hits the bid; a buy aggressor ('B') lifts the ask
                    mine = traded if ((s == 0 and tside == 'A') or
                                      (s == 1 and tside == 'B')) and L_px[i] == qpx[s] else 0.0
                    if mine > 0.0:
                        if mine > qah[s]:
                            got = min(mine - qah[s], our_size)
                            j = L_unw[i]
                            sgn = 1.0 if s == 1 else -1.0    # +1 we sold, -1 we bought
                            mk = sgn * (qpx[s] - L_mid[j]) / m_i * 1e4
                            fills.append((L_ts[i], s, qpx[s], got, mk,
                                          L_ts[i] - posted[s]))
                            act[s] = False
                            qah[s] = 0.0
                        else:
                            qah[s] -= mine
                    if act[s]:
                        # size lost beyond what trades explain is cancellation
                        drop = prev_sz - lvl_sz
                        canc = drop - mine
                        if canc > 0.0 and prev_sz > 0.0:
                            qah[s] -= canc * (qah[s] / prev_sz) * cancel_ahead
                            if qah[s] < 0.0:
                                qah[s] = 0.0

            if not act[s]:
                k = L_lag[i]
                ev = (L_pb[k] if s == 0 else L_pa[k]) + rebate / m_i * 1e4
                if ev > threshold:
                    # You can only aim at the price you could see, `latency` ago. Taking
                    # the price at arrival instead would hand the strategy foresight, and
                    # the more latency the more foresight -- which is exactly backwards.
                    tgt = L_bp[k] if s == 0 else L_ap[k]
                    j = L_arr[i]
                    now = L_bp[j] if s == 0 else L_ap[j]
                    better = (tgt > now) if s == 0 else (tgt < now)
                    worse = (tgt < now) if s == 0 else (tgt > now)
                    if not worse:
                        act[s] = True
                        qpx[s] = tgt
                        # improving on the touch means an empty queue in front of you --
                        # and the adverse selection of being alone at a stale price
                        qah[s] = 0.0 if better else (L_bs[j] if s == 0 else L_as[j])
                        posted[s] = L_ts[i]
                    # `worse` = the market left us behind; the order rests off the touch
                    # and is treated as a miss. That is the real cost of being slow.

        prev_bs, prev_as, prev_bp, prev_ap = b_sz, a_sz, b_px, a_px

    return fills


def run(day, path, booster, man, cfg):
    A = dataset.book_arrays(path)
    if A is None:
        return None
    idx = np.arange(A['n'])
    pb = booster.predict(dataset.features_for_side(A, idx, -1.0))   # we rest on the bid
    pa = booster.predict(dataset.features_for_side(A, idx, +1.0))   # we rest on the ask
    out = []
    for lat_ms in cfg['latencies_ms']:
      for thr in cfg['thresholds']:
        for tier, reb in cfg['tiers'].items():
            f = simulate_day(A, pb, pa, cfg['size'], int(lat_ms * 1e6),
                             thr, reb, cfg['cancel_ahead'])
            if not f:
                out.append(dict(day=day, latency_ms=lat_ms, threshold=thr, tier=tier, rebate=reb,
                                fills=0, shares=0.0, pnl_usd=0.0, bps_per_share=0.0,
                                venue_pct=0.0, win_rate=0.0, peak_inventory_usd=0.0))
                continue
            arr = np.array([(x[2], x[3], x[4]) for x in f])
            px, sh, mk = arr[:, 0], arr[:, 1], arr[:, 2]
            net = mk + reb / px * 1e4
            pnl = sh * px * net / 1e4
            sides = np.array([x[1] for x in f])
            signed = np.where(sides == 1, -sh, sh)     # ask fill = short, bid fill = long
            inv = np.cumsum(signed)
            out.append(dict(
                day=day, latency_ms=lat_ms, threshold=thr, tier=tier, rebate=reb,
                fills=len(f), shares=float(sh.sum()), pnl_usd=float(pnl.sum()),
                bps_per_share=float(np.average(net, weights=sh)),
                venue_pct=float(sh.sum() / A['size'][A['isT']].sum() * 100),
                win_rate=float((net > 0).mean()),
                peak_inventory_usd=float(np.abs(inv).max() * px.mean()),
                median_queue_wait_ms=float(np.median([x[5] for x in f]) / 1e6),
            ))
    return out


_BOOSTER = None


def _worker(job):
    """One day, all latency x tier combinations. Model is loaded once per process."""
    global _BOOSTER
    name, cfg = job
    if _BOOSTER is None:
        _BOOSTER = M.load()
    booster, man = _BOOSTER
    day, p = dataset.extract_to(name, WORK)
    try:
        return run(day, p, booster, man, cfg)
    finally:
        if os.path.exists(p):
            os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=8, help='most recent archive days to simulate')
    ap.add_argument('--size', type=float, default=100.0, help='our quote size, shares')
    ap.add_argument('--cancel-ahead', type=float, default=1.0,
                    help='1.0 = cancels uniform through the queue, 0.0 = all behind us')
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--capital', type=float, default=100_000.0)
    ap.add_argument('--latencies', default='0.5,5,50', help='ms, comma separated')
    ap.add_argument('--thresholds', default=None,
                    help='sweep the quoting threshold in bps, comma separated')
    a = ap.parse_args()

    booster, man = M.load()
    if booster is None:
        raise SystemExit('no checkpoint -- run: python src/train.py')
    thr = a.threshold if a.threshold is not None else (man.get('decision_threshold_bps') or 0.0)
    thrs = [float(x) for x in a.thresholds.split(',')] if a.thresholds else [thr]
    cfg = dict(size=a.size, threshold=thr, thresholds=thrs, cancel_ahead=a.cancel_ahead,
               latencies_ms=[float(x) for x in a.latencies.split(',')], tiers=REBATE_TIERS)

    names = dataset.day_names()[-a.days:]
    print(f'phase 4 queue simulation | {len(names)} days | quote {a.size:.0f} sh '
          f'| threshold {thr:+.3f} bps | cancels {"uniform" if a.cancel_ahead else "behind us"}')
    print(f'latencies {cfg["latencies_ms"]} ms x {len(REBATE_TIERS)} rebate tiers\n')

    rows = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for r in ex.map(_worker, [(nm, cfg) for nm in names]):
            if not r:
                continue
            rows.extend(r)
            top = [x for x in r if x['latency_ms'] == cfg['latencies_ms'][0]
                   and x['tier'] == 'entry' and x['threshold'] == cfg['thresholds'][0]][0]
            print(f'  {top["day"]}: {top["fills"]:>6,} fills  {top["shares"]:>9,.0f} sh  '
                  f'{top["venue_pct"]:>5.2f}% of venue  '
                  f'${top["pnl_usd"]:>9,.2f} @ entry tier / {cfg["latencies_ms"][0]}ms', flush=True)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, 'data', 'queue_sim_grid.csv')
    df.to_csv(out_csv, index=False)

    g = df.groupby(['latency_ms', 'threshold', 'tier']).agg(
        days=('day', 'nunique'), fills=('fills', 'sum'), shares=('shares', 'sum'),
        pnl_usd=('pnl_usd', 'sum'), bps=('bps_per_share', 'mean'),
        venue_pct=('venue_pct', 'mean'), win_rate=('win_rate', 'mean'),
        peak_inv=('peak_inventory_usd', 'max')).reset_index()
    g['pnl_per_day'] = g.pnl_usd / g.days
    g['roi_annual_pct'] = g.pnl_per_day * 252 / a.capital * 100

    print(f'\n{"":=<92}')
    print('  P&L PER DAY BY LATENCY AND REBATE TIER')
    print(f'{"":=<92}')
    print(f'{"lat/thr":>12}' + ''.join(f'{t.split(" ")[0]:>15}' for t in REBATE_TIERS))
    for lat in cfg['latencies_ms']:
        for th in cfg['thresholds']:
            cells = []
            for t in REBATE_TIERS:
                r = g[(g.latency_ms == lat) & (g.threshold == th) & (g.tier == t)]
                cells.append(f'{r.pnl_per_day.iloc[0]:>15,.2f}' if len(r) else f'{"-":>15}')
            print(f'{lat:>5.1f}ms {th:>+5.2f}' + ''.join(cells))

    print(f'\n{"":=<92}')
    print('  DETAIL')
    print(f'{"":=<92}')
    show = g[['latency_ms', 'threshold', 'tier', 'fills', 'venue_pct', 'bps',
              'win_rate', 'pnl_per_day', 'roi_annual_pct']].copy()
    show.columns = ['lat_ms', 'thr', 'tier', 'fills', 'venue%', 'bps/sh',
                    'win%', '$/day', 'ROI%/yr']
    print(show.to_string(index=False, float_format=lambda x: f'{x:,.3f}'))

    pos = g[g.pnl_per_day > 0]
    print(f'\nprofitable cells: {len(pos)} of {len(g)}')
    if len(pos):
        b = pos.sort_values('pnl_per_day', ascending=False).iloc[0]
        print(f'  best: {b.tier} @ {b.latency_ms}ms thr {b.threshold:+.2f} -> ${b.pnl_per_day:,.2f}/day, '
              f'{b.venue_pct:.2f}% of venue, ROI {b.roi_annual_pct:.2f}%/yr on ${a.capital:,.0f}')
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, 'phase4_queue_sim.json'), 'w') as f:
        json.dump({'generated_at': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
                   'config': {k: v for k, v in cfg.items() if k != 'tiers'},
                   'capital_usd': a.capital, 'grid': g.to_dict('records')}, f, indent=2)
    print(f'\ngrid -> {os.path.relpath(out_csv, ROOT)}')
    print(f'report -> reports/phase4_queue_sim.json')


if __name__ == '__main__':
    main()
