# Part of qqq-microstructure.
#
# The ROI test for the rehearsal: what did the paper account ACTUALLY make,
# against what the graded record says the strategy made at official prints?
# Three numbers per night, nothing hidden:
#
#   realized   qty-weighted from your fills: buy MOC at D's close, sell MOO
#              at D+1's open, commissions from the broker's own reports
#   graded     reports/xsec_paper_daily.csv's q5_on_bps -- the equal-weight
#              basket at official auction prints, friction-free (xsec_live.py)
#   gap        realized gross minus graded: auction slippage + integer-share
#              tracking error + names the capital couldn't buy, combined
#
# Since 2026-10-14 the book is hedged (orders.py's HEDGE: short QQQ 1:1 at
# the close, covered at the open). The hedge is a SHORT round trip -- sold
# first, bought back the next morning -- so pair_hedge() pairs it, never
# pair_legs(). A night with a hedge leg is scored as the hedged book: the
# basket's realized return plus the short's, against the graded tilt_bps
# (Q5 - QQQ overnight), which is that book's grade.
#
# One command, run any time TWS is up -- after the 16:00 close to capture the
# buys, after the 09:30 open to capture the sells (the API only serves the
# CURRENT day's executions, so capture same-day or the fills are gone from
# the API -- the CSV keeps everything ever captured):
#
#   python src/fills.py            # capture new fills if TWS is reachable,
#                                  # then reconcile and print (works offline
#                                  # from the CSV alone if TWS is down)
#
# reports/fills.csv is append-only, deduped on the broker's execution id;
# commit it with reports/ -- realized numbers belong in the paper trail.

import os, csv, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from notify import send
from orders import HEDGE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILLS = os.path.join(ROOT, 'reports', 'fills.csv')
DAILY = os.path.join(ROOT, 'reports', 'xsec_paper_daily.csv')
ET = ZoneInfo('America/New_York')
COLS = ['exec_id', 'time_et', 'date', 'symbol', 'side', 'qty', 'price',
        'commission']


def capture():
    """Pull today's executions from TWS into reports/fills.csv (deduped)."""
    from orders import connect_paper
    ib = connect_paper(client_offset=1)
    ib.sleep(1.0)                       # let commission reports arrive
    rows = []
    for f in ib.reqExecutions():
        e = f.execution
        t = e.time.astimezone(ET) if e.time.tzinfo else e.time
        rows.append(dict(
            exec_id=e.execId, time_et=t.strftime('%Y-%m-%d %H:%M:%S'),
            date=t.strftime('%Y%m%d'), symbol=f.contract.symbol,
            side=e.side, qty=int(e.shares), price=float(e.price),
            commission=float(f.commissionReport.commission)
            if f.commissionReport else 0.0))
    ib.disconnect()
    have = set()
    if os.path.exists(FILLS):
        have = set(pd.read_csv(FILLS, dtype=str).exec_id)
    new = [r for r in rows if r['exec_id'] not in have]
    if new:
        first = not os.path.exists(FILLS)
        with open(FILLS, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            if first:
                w.writeheader()
            w.writerows(new)
    print(f'captured {len(new)} new fill(s) ({len(rows)} served by the API)')
    return len(new)


def _agg(f):
    """Partial executions summed per (date, symbol, side), at the
    quantity-weighted price."""
    f = f.copy()
    f['notl'] = f.qty * f.price
    agg = f.groupby(['date', 'symbol', 'side'], as_index=False).agg(
        qty=('qty', 'sum'), notl=('notl', 'sum'),
        comm=('commission', 'sum'))
    agg['px'] = agg.notl / agg.qty
    return agg


def pair_legs(f, quiet=False):
    """Pair each sell with the latest prior buy of the same symbol, after
    summing partial executions per (date, symbol, side). One dict per
    round trip: night (the buy date), sym, qty, buy_px, sell_px,
    sell_date, notional (qty x buy_px), bps (log return x 1e4), comm (both
    legs). Shared by reconcile() and paper_bench.py so the two can never
    pair the same fills differently. The hedge symbol is left to
    pair_hedge(): read buy-then-sell, a night's short would pair with the
    previous morning's cover."""
    agg = _agg(f[f.symbol != HEDGE[0]])
    buys = {(r.date, r.symbol): r for r in
            agg[agg.side == 'BOT'].itertuples()}
    legs = []
    for s in agg[agg.side == 'SLD'].itertuples():
        prior = [d for d, sym in buys if sym == s.symbol and d < s.date]
        if not prior:
            if not quiet:
                print(f'  {s.symbol} sold {s.date} with no prior buy on file '
                      f'-- skipped')
            continue
        b = buys[(max(prior), s.symbol)]
        q = min(b.qty, s.qty)
        if b.qty != s.qty and not quiet:
            print(f'  {s.symbol} {b.date}: buy {b.qty} vs sell {s.qty} -- '
                  f'using {q}')
        legs.append(dict(night=b.date, sym=s.symbol, qty=q, buy_px=b.px,
                         sell_px=s.px, sell_date=s.date, notional=q * b.px,
                         bps=float(np.log(s.px / b.px) * 1e4),
                         comm=b.comm + s.comm))
    return legs


def pair_hedge(f, quiet=False):
    """pair_legs' mirror for the hedge: each cover (a BOT of the hedge
    symbol) paired with the latest prior short (SLD). One dict per round
    trip: night (the short's date), sym, qty, short_px, cover_px,
    cover_date, notional (qty x short_px), bps (the SHORT's log return x
    1e4, + when the hedge fell), comm (both legs)."""
    agg = _agg(f[f.symbol == HEDGE[0]])
    shorts = {(r.date, r.symbol): r for r in
              agg[agg.side == 'SLD'].itertuples()}
    legs = []
    for b in agg[agg.side == 'BOT'].itertuples():
        prior = [d for d, sym in shorts if sym == b.symbol and d < b.date]
        if not prior:
            if not quiet:
                print(f'  {b.symbol} covered {b.date} with no prior short on '
                      f'file -- skipped')
            continue
        s = shorts[(max(prior), b.symbol)]
        q = min(s.qty, b.qty)
        if s.qty != b.qty and not quiet:
            print(f'  {b.symbol} {s.date}: short {s.qty} vs cover {b.qty} -- '
                  f'using {q}')
        legs.append(dict(night=s.date, sym=b.symbol, qty=q, short_px=s.px,
                         cover_px=b.px, cover_date=b.date, notional=q * s.px,
                         bps=float(np.log(s.px / b.px) * 1e4),
                         comm=s.comm + b.comm))
    return legs


def reconcile():
    """Pair each night's buys with the next morning's sells, per symbol,
    qty-weighted; print realized vs graded vs gap."""
    if not os.path.exists(FILLS):
        print('no reports/fills.csv yet -- nothing to reconcile')
        return None
    nights = {}
    f = pd.read_csv(FILLS, dtype={'date': str})
    for leg in pair_legs(f):
        nights.setdefault(leg['night'], []).append(dict(
            sym=leg['sym'], notional=leg['notional'], bps=leg['bps'],
            comm=leg['comm']))
    hedge = {h['night']: h for h in pair_hedge(f)}
    for D in sorted(set(hedge) - set(nights)):
        print(f'  {D}: a {hedge[D]["sym"]} short with no long leg -- a naked '
              f'short night, left out of the scorecard')
    if not nights:
        print('no completed buy->sell nights yet')
        return None

    graded, tilt = {}, {}
    if os.path.exists(DAILY):
        d = pd.read_csv(DAILY, dtype={'date': str})
        graded = dict(zip(d.date, d.q5_on_bps.astype(float)))
        if 'tilt_bps' in d:
            tilt = dict(zip(d.date, d.tilt_bps.astype(float)))

    print(f'\n{"night":>10}{"names":>7}{"realized":>10}{"hedge":>8}'
          f'{"comm":>8}{"net":>8}{"graded":>8}{"gap":>8}   (bps of deployed '
          f'long notional; H = hedged night, graded on tilt)')
    tot_not, tot_net, tot_g, tot_gn, rows = 0.0, 0.0, 0.0, 0.0, 0
    for D in sorted(nights):
        legs = nights[D]
        notional = sum(x['notional'] for x in legs)
        gross = sum(x['bps'] * x['notional'] for x in legs) / notional
        comm = sum(x['comm'] for x in legs) / notional * 1e4
        h = hedge.get(D)
        hb = h['bps'] * h['notional'] / notional if h else 0.0
        if h:
            comm += h['comm'] / notional * 1e4
        gross += hb
        g = (tilt if h else graded).get(D)
        print(f'{D:>10}{len(legs):>7}{gross - hb:>+10.1f}'
              + (f'{hb:>+8.1f}' if h else f'{"--":>8}')
              + f'{-comm:>8.1f}{gross - comm:>+8.1f}'
              + (f'{g:>+8.1f}{gross - g:>+8.1f}' if g is not None
                 else f'{"--":>8}{"--":>8}   (not graded yet -- monthly.py '
                      f'grades it)')
              + (' H' if h else ''))
        tot_not += notional
        tot_net += (gross - comm) * notional
        if g is not None:
            tot_g += g * notional
            tot_gn += notional
            rows += 1
    net = tot_net / tot_not
    line = (f'{len(nights)} night(s): realized net {net:+.1f} bps/night '
            f'on ${tot_not / len(nights):,.0f} avg'
            + (f'   graded {tot_g / tot_gn:+.1f} over the {rows} graded'
               if rows else ''))
    print('\n  ' + line)
    print('  gap = slippage + integer-share tracking + unbought names (+ the '
          'hedge ratio and the short\'s fills on H nights); commissions shown '
          'separately.\n  ROI check: net bps/night x nights/mo / 100 = %/mo '
          'on deployed capital.')
    return net, line


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true',
                    help='no capture: reconcile the committed CSV and '
                         'telegram the scorecard (what the CI job runs)')
    a = ap.parse_args()
    n = 0
    if not a.report:
        try:
            n = capture()
        except SystemExit as ex:
            print(f'{ex}\n(TWS unreachable -- reconciling from the CSV alone)')
    out = reconcile()
    if out is not None and (n or a.report):
        send(f'qqq ROI: {out[1]}' + (f' ({n} new fills)' if n else ''))
    elif out is None and a.report:
        # no paper round trip at all while the grader keeps grading: the VM
        # legs are not running. This used to print to a CI log nobody reads
        # and send nothing, which is how seven weeks of forward test went by
        # with zero paper fills (audit 2026-10).
        g = len(pd.read_csv(DAILY)) if os.path.exists(DAILY) else 0
        if g:
            msg = (f'qqq ROI: paper leg SILENT -- {g} graded nights, 0 paper '
                   f'round trips in reports/fills.csv. The VM order legs '
                   f'(roi.yml crontab) are not running; the forward test is '
                   f'grading a book nobody is trading.')
            print(msg)
            send(msg)
    if not a.report:
        print('\ncommit reports/fills.csv with the rest of reports/')


if __name__ == '__main__':
    main()
