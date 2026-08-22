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


def reconcile():
    """Pair each night's buys with the next morning's sells, per symbol,
    qty-weighted; print realized vs graded vs gap."""
    if not os.path.exists(FILLS):
        print('no reports/fills.csv yet -- nothing to reconcile')
        return None
    f = pd.read_csv(FILLS, dtype={'date': str})
    f['notl'] = f.qty * f.price
    agg = f.groupby(['date', 'symbol', 'side'], as_index=False).agg(
        qty=('qty', 'sum'), notl=('notl', 'sum'),
        comm=('commission', 'sum'))
    agg['px'] = agg.notl / agg.qty
    buys = {(r.date, r.symbol): r for r in
            agg[agg.side == 'BOT'].itertuples()}
    nights = {}
    for s in agg[agg.side == 'SLD'].itertuples():
        prior = [d for d, sym in buys if sym == s.symbol and d < s.date]
        if not prior:
            print(f'  {s.symbol} sold {s.date} with no prior buy on file -- '
                  f'skipped')
            continue
        b = buys[(max(prior), s.symbol)]
        q = min(b.qty, s.qty)
        if b.qty != s.qty:
            print(f'  {s.symbol} {b.date}: buy {b.qty} vs sell {s.qty} -- '
                  f'using {q}')
        nights.setdefault(b.date, []).append(dict(
            sym=s.symbol, notional=q * b.px,
            bps=float(np.log(s.px / b.px) * 1e4),
            comm=b.comm + s.comm))
    if not nights:
        print('no completed buy->sell nights yet')
        return None

    graded = {}
    if os.path.exists(DAILY):
        d = pd.read_csv(DAILY, dtype={'date': str})
        graded = dict(zip(d.date, d.q5_on_bps.astype(float)))

    print(f'\n{"night":>10}{"names":>7}{"realized":>10}{"comm":>8}'
          f'{"net":>8}{"graded":>8}{"gap":>8}   (bps of deployed notional)')
    tot_not, tot_net, tot_g, tot_gn, rows = 0.0, 0.0, 0.0, 0.0, 0
    for D in sorted(nights):
        legs = nights[D]
        notional = sum(x['notional'] for x in legs)
        gross = sum(x['bps'] * x['notional'] for x in legs) / notional
        comm = sum(x['comm'] for x in legs) / notional * 1e4
        g = graded.get(D)
        print(f'{D:>10}{len(legs):>7}{gross:>+10.1f}{-comm:>8.1f}'
              f'{gross - comm:>+8.1f}'
              + (f'{g:>+8.1f}{gross - g:>+8.1f}' if g is not None
                 else f'{"--":>8}{"--":>8}   (not graded yet -- monthly.py '
                      f'grades it)'))
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
    print('  gap = slippage + integer-share tracking + unbought names; '
          'commissions shown separately.\n  ROI check: net bps/night x '
          'nights/mo / 100 = %/mo on deployed capital.')
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
    if not a.report:
        print('\ncommit reports/fills.csv with the rest of reports/')


if __name__ == '__main__':
    main()
