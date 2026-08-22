# Part of qqq-microstructure.
#
# The go-live pipe, built during probation so verdict day is not day one of
# plumbing. It turns the month's PRE-REGISTERED basket (reports/xsec_paper.csv
# -- the long Q5, exactly as xsec_live.py emitted and git recorded it) into
# the two daily order lists the strategy consists of:
#
#   close leg   BUY  MOC  every name      (submit before ~15:50 ET; Nasdaq
#                                          accepts MOC until 15:55)
#   open  leg   SELL MOO  the positions   (submit before 09:28 ET)
#
# Three modes, in order of ceremony:
#   DRY RUN (default)  compute, print, Telegram, log. Submits nothing,
#                      needs no broker running. This is the monthly rehearsal.
#   --submit           place the orders in IBKR PAPER. Requires TWS or IB
#                      Gateway running with the API enabled, and refuses any
#                      account not prefixed 'D' (IBKR paper accounts).
#   live               intentionally NOT implemented. RESULTS 19/20's
#                      probation says paper until the forward verdict; when
#                      that day comes, enabling it is a small, deliberate,
#                      dated commit -- not a flag someone fat-fingers.
#
# Environment:
#   IBKR_API       host:port:clientId of TWS/Gateway (default 127.0.0.1:7497:7
#                  -- the TWS paper port; Gateway paper is 4002). IBKR has no
#                  API key: the "API" is a local socket you enable in TWS via
#                  File > Global Configuration > API > Settings > Enable
#                  ActiveX and Socket Clients (and add 127.0.0.1 to trusted).
#   TELEGRAM_API   bot token; TELEGRAM_CHAT pinned chat (see notify.py).
#
# Discipline rails: a stale registration is refused, not traded (if the
# current month has no committed basket, the answer is run monthly.py, not
# trade last month's list); sizing splits --capital equally over ALL basket
# names and leaves an unpriced name's slice in cash rather than silently
# re-concentrating; every run appends one row to reports/orders_log.csv --
# commit it with the rest of reports/, it is the rehearsal's paper trail.
# Scope: the overnight basket only -- the MOM and OPT legs are separate
# machinery and are not traded here.
#
#   python src/orders.py --capital 5000                  # dry run, leg by clock
#   python src/orders.py --capital 5000 --leg close      # explicit leg
#   python src/orders.py --capital 5000 --leg close --submit    # IBKR paper

import os, csv, argparse, datetime as dt
from zoneinfo import ZoneInfo
import pandas as pd
from notify import send

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, 'reports', 'xsec_paper.csv')
OLOG = os.path.join(ROOT, 'reports', 'orders_log.csv')
ET = ZoneInfo('America/New_York')


def basket(now):
    if not os.path.exists(LOG):
        raise SystemExit('no reports/xsec_paper.csv -- run monthly.py first')
    log = pd.read_csv(LOG, dtype=str)
    cur = now.strftime('%Y-%m')
    row = log[log.month == cur]
    if row.empty:
        raise SystemExit(f'no pre-registered basket for {cur} (latest is '
                         f'{log.month.iloc[-1]}) -- run monthly.py first; a '
                         f'stale basket is not tradeable')
    r = row.iloc[0]
    return r.month, r.q5.split(';'), r.emitted_on


def prices(names):
    import yfinance as yf
    v = yf.download([n.replace('.', '-') for n in names], period='5d',
                    interval='1d', auto_adjust=False, actions=False,
                    group_by='ticker', progress=False)
    px = {}
    for n in names:
        try:
            s = v[n.replace('.', '-')]['Close'].dropna()
            if len(s):
                px[n] = float(s.iloc[-1])
        except KeyError:
            continue
    return px


def size(names, px, capital):
    per = capital / len(names)
    rows = []
    for n in names:
        p = px.get(n)
        q = int(per // p) if p else 0
        rows.append((n, p, q, (q * p) if p else 0.0))
    return rows, per


def submit_paper(rows, leg, now):
    try:
        from ib_insync import IB, Stock, Order
    except ImportError:
        raise SystemExit('pip install ib_insync   (needed only for --submit)')
    host, port, cid = (os.environ.get('IBKR_API') or
                       '127.0.0.1:7497:7').split(':')
    ib = IB()
    try:
        ib.connect(host, int(port), clientId=int(cid), timeout=8)
    except Exception as ex:
        raise SystemExit(
            f'cannot reach TWS/Gateway at {host}:{port} ({type(ex).__name__})'
            f' -- is it running, is the API enabled (File > Global '
            f'Configuration > API), and is the port right? (TWS paper 7497, '
            f'Gateway paper 4002; set IBKR_API=host:port:clientId)')
    acct = (ib.managedAccounts() or ['?'])[0]
    if not acct.startswith('D'):
        ib.disconnect()
        raise SystemExit(
            f'account {acct} is not an IBKR PAPER account (paper accounts '
            f'start with D). Live submission is intentionally not implemented '
            f'until the forward verdict (RESULTS 19/20) -- log into TWS in '
            f'paper mode.')
    print(f'connected: paper account {acct}')
    working = {t.contract.symbol for t in ib.openTrades()
               if t.order.orderType in ('MOC', 'MKT')}
    placed = 0
    if leg == 'open':
        pos = {p.contract.symbol: int(p.position) for p in ib.positions()
               if p.position > 0}
    for n, p, q, _ in rows:
        sym = n.replace('.', ' ')
        if leg == 'open':
            q = pos.get(sym, 0)
        if q <= 0:
            print(f'  {n:<6} skipped ({"no position" if leg == "open" else "zero shares"})')
            continue
        if sym in working:
            print(f'  {n:<6} skipped (order already working)')
            continue
        c = ib.qualifyContracts(Stock(sym, 'SMART', 'USD'))
        if not c:
            print(f'  {n:<6} skipped (contract not found)')
            continue
        o = Order(action='BUY', orderType='MOC', totalQuantity=q) \
            if leg == 'close' else \
            Order(action='SELL', orderType='MKT', tif='OPG', totalQuantity=q)
        tr = ib.placeOrder(c[0], o)
        ib.sleep(0.3)
        print(f'  {n:<6} {o.action} {q} {o.orderType}{"/OPG" if leg == "open" else ""} '
              f'-> {tr.orderStatus.status}')
        placed += 1
    ib.disconnect()
    return placed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capital', type=float, required=True,
                    help='dollars for the whole basket, split equally')
    ap.add_argument('--leg', choices=['close', 'open'], default=None,
                    help='close = BUY MOC tonight; open = SELL MOO. Default: '
                         'inferred from the ET clock (dry run only)')
    ap.add_argument('--submit', action='store_true',
                    help='place the orders in IBKR paper (default: dry run)')
    a = ap.parse_args()
    now = dt.datetime.now(ET)
    leg = a.leg
    if leg is None:
        if a.submit:
            raise SystemExit('--submit requires an explicit --leg')
        leg = 'open' if now.hour < 12 else 'close'
        print(f'leg not given -- inferred "{leg}" from ET clock '
              f'({now:%H:%M} ET); pass --leg to override')

    month, names, emitted = basket(now)
    px = prices(names)
    rows, per = size(names, px, a.capital)
    mode = 'SUBMIT (paper)' if a.submit else 'DRY RUN'

    print(f'\n{month} basket (pre-registered {emitted}), {len(names)} names, '
          f'${a.capital:,.0f} -> ${per:,.0f}/name   [{mode}]')
    act = 'BUY  MOC' if leg == 'close' else 'SELL MOO'
    total = 0.0
    for n, p, q, notional in rows:
        if p is None:
            print(f'  {act}  {n:<6}     ?? no price -- slice stays in cash')
        elif q == 0:
            print(f'  {act}  {n:<6}      0 @ {p:>9.2f}  (price > slice; '
                  f'raise --capital)')
        else:
            print(f'  {act}  {n:<6} {q:>6} @ {p:>9.2f}  = {q * p:>10,.2f}')
            total += notional
    print(f'  total {total:>10,.2f} of {a.capital:,.2f} '
          f'({total / a.capital * 100:.0f}% deployed)')
    if leg == 'close' and (now.hour, now.minute) >= (15, 45):
        print('  WARNING: past ~15:45 ET -- MOC cutoff is 15:50 NYSE / '
              '15:55 Nasdaq')
    if leg == 'open' and now.hour < 12 and (now.hour, now.minute) >= (9, 25):
        print('  WARNING: past 09:25 ET -- MOO/OPG cutoff is 09:28 '
              '(evening pre-placement for tomorrow is fine)')

    placed = None
    if a.submit:
        print()
        placed = submit_paper(rows, leg, now)
        print(f'{placed} order(s) placed in paper')

    os.makedirs(os.path.dirname(OLOG), exist_ok=True)
    new = not os.path.exists(OLOG)
    with open(OLOG, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['run_at_et', 'month', 'leg', 'mode', 'names',
                        'notional', 'capital', 'placed'])
        w.writerow([now.strftime('%Y-%m-%d %H:%M'), month, leg,
                    'paper' if a.submit else 'dry',
                    ';'.join(n for n, _, q, _ in rows if q),
                    f'{total:.2f}', f'{a.capital:.2f}',
                    '' if placed is None else placed])
    print(f'logged -> reports/orders_log.csv (commit with reports/)')

    msg = (f'qqq {month} {leg} leg [{mode}]\n'
           + '\n'.join(f'{act} {n} x{q} @ {p:.2f}' for n, p, q, _ in rows
                       if q) +
           f'\ntotal ${total:,.0f} / ${a.capital:,.0f}'
           + (f'\n{placed} placed in paper' if placed is not None else ''))
    if send(msg):
        print('telegram: sent')


if __name__ == '__main__':
    main()
