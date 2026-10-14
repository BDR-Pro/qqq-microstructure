# Part of qqq-microstructure.
#
# The go-live pipe, built during probation so verdict day is not day one of
# plumbing. It turns the month's PRE-REGISTERED basket (reports/xsec_paper.csv
# -- the long Q5, exactly as xsec_live.py emitted and git recorded it) into
# the two daily order lists the strategy consists of.
#
# THE TRADED OBJECT, since 2026-10-14 (the owner's decision; RESULTS 29): the
# QQQ-HEDGED book. Long the Q5 basket overnight AND short QQQ overnight, 1:1
# in dollars against the long notional actually bought. This is RESULTS 21's
# NEU object: alpha +3.79 bps/day net (t 5.25), total overnight beta 0.42
# against the long-only book's 1.42. Its nightly return is exactly the
# forward test's paper-log metric, tilt_bps = Q5 - QQQ overnight
# (xsec_live.py). The forward metrics do not depend on the configuration
# (RESULTS 20), so this is a sizing choice, not new evidence, and the
# pre-registered test (RESULTS 19, xsec_replay.py) is untouched. The 1:1
# dollar hedge was chosen from the backtest, not from the 34 live nights. It
# is the constant HEDGE below: declared and dated, not a flag.
#
#   close leg   BUY  MOC  every basket name   (submit before ~15:50 ET:
#               SELL MOC  QQQ, the short        Nasdaq takes MOC until 15:55
#                                               and freezes it at 15:50)
#   open  leg   SELL MOO  every long held     (submit before 09:28 ET)
#               BUY  MOO  every short held, i.e. the cover
#
# Hedge sizing: round(long notional bought x ratio / QQQ price) whole shares,
# priced from the same snapshot as the basket. "Bought" means the slices that
# got shares: a name priced above its slice, or unpriced, is out of the hedge
# base as it is out of the book. In paper it means the long orders IBKR
# ACCEPTED. The basket goes in first and the short is sized from what was
# accepted, so a failure can leave the old unhedged book but never a naked
# short. Shares are rounded to the nearest, not down: always rounding down
# would be a lasting under-hedge (-2.4% at $25k).
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
# What the short needs from the account (IBKR margin pages, read 2026-10-14):
#   - A non-IRA MARGIN account. Cash and IRA accounts cannot short; the
#     rejection shows as the SELL's status, and the run then says loudly that
#     the long book is unhedged tonight.
#   - Reg T: 50% of EACH leg at the end-of-day SMA check, with no offset
#     between a stock basket and a QQQ short (12 CFR 220.12). The overnight
#     book therefore needs long + |short| <= 2 x equity, half the long-only
#     ceiling. --submit reads the account's NetLiquidation and refuses unless
#     long + |short| <= 2 x 0.95 x equity. The 5% covers a cross printing
#     above the sizing prices, the half-share rounding, fees, and equity
#     shrinking after losing nights. A 1:1 book fully deployed is about
#     2 x --capital gross, so fund the account about 5% above --capital, or
#     pass --capital <= 0.95 x equity. The dry run prints the equity needed.
#   - The API action is a plain SELL: IBKR marks it short and does the Reg SHO
#     locate itself (SSHORT is for institutional accounts only). QQQ is easy
#     to borrow: 8-10M shares available in Sep-Oct 2026, against a book that
#     needs tens to hundreds.
#   - The account to itself. QQQ is now the hedge AND live_momentum.py's only
#     symbol (the MOM forward test: +-10 QQQ from 10:31, flattened at 15:58,
#     on the same paper ports). This close leg runs at 15:25, inside MOM's
#     holding window, and reads any QQQ position as the hedge: a MOM short
#     makes it send a smaller short (or none) that MOM's 15:58 buy then
#     removes, a MOM long makes it refuse the night, and MOM's flatten covers
#     a hedge left open by a failed morning cover. Never run live_momentum.py
#     in the account this pipe trades.
#
# Known and NOT handled here:
#   - Rule 201 (SSR). After QQQ falls 10% from the prior close, for the rest
#     of that day and all of the next, a short MOC does not take part in a
#     cross priced at or below the national best bid. The long then fills
#     unhedged, on the most volatile nights. QQQ has been under SSR on 2 days
#     on record (last 2020-03-17). A stand-aside rule is the owner's to
#     declare; until then this is a known gap.
#   - QQQ dividends. A short held over an ex-date (quarterly, Mondays) owes the
#     dividend, about 10 bps on that night. tilt_bps uses unadjusted prices,
#     which credit the short with the ex-date drop, so live trails tilt by
#     about 0.17 bps/night on average, partly offset by the basket's own
#     dividends. IBKR paper processes no dividends. The QQQ borrow fee (general
#     collateral, ~0.3-0.4%/yr) is not priced here either; paper_bench says so.
#
# Environment:
#   IBKR_API       host:port:clientId of TWS/Gateway (default 127.0.0.1:7497:7
#                  -- the TWS paper port; Gateway paper is 4002). IBKR has no
#                  API key: the "API" is a local socket you enable in TWS via
#                  File > Global Configuration > API > Settings > Enable
#                  ActiveX and Socket Clients (and add 127.0.0.1 to trusted).
#   TELEGRAM_API   bot token; TELEGRAM_CHAT pinned chat (see notify.py).
#   cron reads no profile: install_ibkr.sh exports IBKR_API in ~/.profile, so
#   a crontab line that does not source it dials 7497 while the Gateway
#   listens on 4002, and finds no Telegram token for the alerts below.
#
# Discipline rails:
#   - A stale registration is refused, not traded. If the current month has
#     no committed basket, the answer is to run monthly.py, not to trade last
#     month's list.
#   - Sizing splits --capital equally over ALL basket names and leaves an
#     unpriced name's slice in cash rather than silently re-concentrating.
#   - No hedge price, or the hedge symbol inside the basket: the close leg
#     stands aside.
#   - Only the declared hedge symbol is ever sold short.
#   - A leg that stands aside or is refused (no hedge price, Reg T, the
#     account long the hedge, TWS unreachable) is logged and pushed to
#     Telegram before the run exits non-zero, with any short still open: a
#     failed morning cover left behind by a refused close leg is a naked
#     short overnight, and under cron nobody reads qqq.log.
#   - Every run appends one row to reports/orders_log.csv and its hedge
#     order(s) to reports/orders_hedge_log.csv. Commit both with the rest of
#     reports/; they are the rehearsal's paper trail.
# Scope: the overnight book only -- the MOM and OPT legs are separate
# machinery and are not traded here, and MOM must not trade in this account
# (above: both trade QQQ).
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
# The hedge leg has its own log rather than new columns in orders_log.csv:
# that file's header is history, and its 'names' and 'notional' columns keep
# meaning the long basket.
HLOG = os.path.join(ROOT, 'reports', 'orders_hedge_log.csv')
ET = ZoneInfo('America/New_York')

HEDGE = ('QQQ', 1.0)    # declared 2026-10-14, RESULTS 29: short QQQ, 1:1 dollars
REGT_GROSS = 2.0        # Reg T end-of-day SMA check: long + |short| <= 2 x equity
REGT_USE = 0.95         # ...and this pipe uses at most 95% of that line
REJECTED = ('Cancelled', 'ApiCancelled', 'Inactive', 'contract not found')


class StandAside(SystemExit):
    """A leg that refuses to trade. main() logs and pushes it before the run
    exits non-zero; `shorts` is what the account still holds short, if the
    run got as far as reading it."""

    def __init__(self, msg, shorts=None):
        super().__init__(msg)
        self.shorts = shorts


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


def hedge_shares(long_notional, price, ratio=HEDGE[1]):
    """Whole shares of the hedge to short against `long_notional` dollars of
    long actually bought. None bought, or no hedge price: 0, never a naked
    short."""
    if not price or long_notional <= 0:
        return 0
    return int(round(long_notional * ratio / price))


def hedge_leg(rows, px):
    """(symbol, price, shares, long notional) of the short for a sized long
    book: the base is the rows that got shares, not --capital."""
    sym = HEDGE[0]
    long_ = sum(v for _, p, q, v in rows if p and q)
    return sym, px.get(sym), hedge_shares(long_, px.get(sym)), long_


def check_basket(names):
    """The close leg refuses a basket that contains the hedge symbol: the same
    symbol long and short nets to nothing at IBKR and is not the book."""
    if HEDGE[0] in names:
        raise SystemExit(f'the hedge symbol {HEDGE[0]} is in the basket '
                         f'({";".join(names)}) -- refusing; the hedged book '
                         f'cannot be long and short the same symbol')


def regt_check(long_notional, short_notional, equity):
    """None if the overnight book fits Reg T's end-of-day check with this
    pipe's 5% margin, else the refusal message."""
    gross = long_notional + abs(short_notional)
    if equity is None:
        return ('REG T: cannot read the account\'s NetLiquidation (USD), so '
                'the overnight book cannot be checked against it. Nothing '
                'was placed.')
    line = REGT_GROSS * REGT_USE * equity
    if gross <= line:
        return None
    return (f'REG T: long ${long_notional:,.0f} + short '
            f'${abs(short_notional):,.0f} = ${gross:,.0f} gross overnight, '
            f'above {REGT_GROSS:.0f} x {REGT_USE:.0%} x equity '
            f'${equity:,.0f} = ${line:,.0f}. Reg T charges 50% on each '
            f'leg at IBKR\'s end-of-day SMA check, with no offset between the '
            f'basket and the QQQ short. Fund the account to at least '
            f'${gross / (REGT_GROSS * REGT_USE):,.0f} or lower --capital. '
            f'Nothing was placed.')


def connect_paper(client_offset=0):
    """Connect to TWS/Gateway and REFUSE anything but a paper account.
    The paper-only rule lives here and only here."""
    try:
        from ib_insync import IB
    except ImportError:
        raise SystemExit('pip install ib_insync   (needed only for --submit)')
    host, port, cid = (os.environ.get('IBKR_API') or
                       '127.0.0.1:7497:7').split(':')
    ib = IB()
    try:
        ib.connect(host, int(port), clientId=int(cid) + client_offset,
                   timeout=8)
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
    return ib


def net_liquidation(ib):
    """The account's equity in USD as IBKR reports it, or None."""
    for v in ib.accountValues():
        if v.tag == 'NetLiquidation' and v.currency == 'USD':
            try:
                return float(v.value)
            except ValueError:
                return None
    return None


def close_quantities(rows, held):
    """Shares to BUY at the close: the target minus what the account already
    holds, never below zero. If last night's open leg failed, buying the full
    target again would double a 2.4-beta, one-theme basket overnight (review,
    2026-10-12); this tops up to the target and no further."""
    return [(n, p, max(0, q - held.get(n.replace('.', ' '), 0)), v)
            for n, p, q, v in rows]


def short_quantity(target, held_short):
    """Shares to SELL short at the close: the hedge target minus the short
    already open, never below zero -- close_quantities' mirror. A failed
    morning cover must not double the short."""
    return max(0, target - held_short)


def submit_paper(rows, leg, now, hedge_px=None):
    """Place the leg in IBKR paper. Returns (orders placed, hedge lines),
    a hedge line being (symbol, signed shares sent, price, status). A hedge
    order not sent is a line of 0 shares whose status says why."""
    from ib_insync import Stock, Order
    ib = connect_paper()
    working = {}
    for t in ib.openTrades():
        # openTrades() drops only Filled/Cancelled/ApiCancelled, so an order
        # IBKR rejected ('Inactive') in an earlier run is still listed. It is
        # not working: counting it would block the retry and put a long that
        # will not exist into the hedge base.
        if t.order.orderType in ('MOC', 'MKT') and \
                t.orderStatus.status not in REJECTED:
            working.setdefault(t.contract.symbol, []).append(t)
    signed = {}
    avg = {}
    for p in ib.positions():
        if p.contract.secType == 'STK' and int(p.position):
            signed[p.contract.symbol] = int(p.position)
            avg[p.contract.symbol] = float(getattr(p, 'avgCost', 0) or 0)
    pos = {s: q for s, q in signed.items() if q > 0}
    shorts = {s: -q for s, q in signed.items() if q < 0}
    hsym = HEDGE[0]
    placed, hedge_lines = 0, []

    def working_side(sym, action):
        return any(t.order.action == action for t in working.get(sym, []))

    def place(sym, action, q, close):
        c = ib.qualifyContracts(Stock(sym, 'SMART', 'USD'))
        if not c:
            return None
        o = Order(action=action, orderType='MOC', totalQuantity=q) if close \
            else Order(action=action, orderType='MKT', tif='OPG',
                       totalQuantity=q)
        tr = ib.placeOrder(c[0], o)
        ib.sleep(0.3)
        return tr

    if leg == 'close':
        if pos:
            print(f'  WARNING: {len(pos)} position(s) still open at the close '
                  f'-- did the open leg fail? Buying only up to target: '
                  + ' '.join(f'{k} {v}' for k, v in sorted(pos.items())))
        if shorts:
            print(f'  WARNING: short(s) still open at the close -- did the '
                  f'cover fail, or is live_momentum.py trading {hsym} in this '
                  f'account (it must not)? Shorting only up to target: '
                  + ' '.join(f'{k} -{v}' for k, v in sorted(shorts.items())))
        if pos.get(hsym, 0) > 0:
            ib.disconnect()
            raise StandAside(f'the account is LONG {pos[hsym]} {hsym}, the '
                             f'hedge symbol -- a SELL would close that, not '
                             f'short (live_momentum.py in this account?). '
                             f'Refusing; flatten it first.', shorts)
        # Reg T, on the book the account would hold overnight, before any
        # order: each basket name at max(target, held), longs outside the
        # basket at cost, and the hedge at max(target, held short).
        basket_syms = {n.replace('.', ' ') for n, *_ in rows}
        long_on = sum(max(q, pos.get(n.replace('.', ' '), 0)) * p
                      for n, p, q, _ in rows if p)
        long_on += sum(q * avg.get(s, 0.0) for s, q in pos.items()
                       if s not in basket_syms)
        _, _, h_target, _ = hedge_leg(rows, {hsym: hedge_px})
        short_on = max(h_target, shorts.get(hsym, 0)) * (hedge_px or 0.0)
        msg = regt_check(long_on, short_on, net_liquidation(ib))
        if msg:
            ib.disconnect()
            raise StandAside(msg, shorts)
        rows = close_quantities(rows, pos)
    if leg == 'open':
        # the open leg SELLS WHATEVER THE ACCOUNT HOLDS -- every long stock
        # position, whether or not it is in the current basket. The first
        # version sold only current-basket names, so a month rollover (Aug-31
        # buys, September basket emitted days later) could leave positions
        # unsold for a week (audit 2026-09, critic #4). This paper account is
        # dedicated to the strategy by construction.
        held = set(pos) - {n.replace('.', ' ') for n, *_ in rows}
        if held:
            print(f'  liquidating {len(held)} position(s) not in the basket: '
                  + ' '.join(sorted(held)))
        rows = list(rows) + [(h.replace(' ', '.'), None, pos[h], 0.0)
                             for h in sorted(held)]
    trades = {}
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
        tr = place(sym, 'BUY' if leg == 'close' else 'SELL', q,
                   close=leg == 'close')
        if tr is None:
            print(f'  {n:<6} skipped (contract not found)')
            continue
        trades[sym] = (tr, q)
        print(f'  {n:<6} {tr.order.action} {q} {tr.order.orderType}'
              f'{"/OPG" if leg == "open" else ""} -> {tr.orderStatus.status}')
        placed += 1

    if leg == 'close':
        # the hedge base is what will be held long overnight: shares already
        # held, plus buys IBKR accepted (statuses re-read after a pause, since
        # a rejection can arrive late), plus BUYs already working from an
        # earlier run today
        if trades:
            ib.sleep(1.0)
        base = 0.0
        for n, p, _, _ in rows:
            sym = n.replace('.', ' ')
            if not p:
                continue
            tr, q = trades.get(sym, (None, 0))
            ok = q if tr is not None and tr.orderStatus.status not in REJECTED \
                else 0
            wq = sum(int(t.order.totalQuantity) for t in working.get(sym, [])
                     if t.order.action == 'BUY')
            base += (pos.get(sym, 0) + ok + wq) * p
        target = hedge_shares(base, hedge_px)
        held = shorts.get(hsym, 0)
        q = short_quantity(target, held)
        why = None
        if target == 0:
            why = (f'target 0 (no long accepted, or no {hsym} price; '
                   f'{held} held' + (' -- NAKED tonight' if held else '')
                   + ')')
            print(f'  {hsym:<6} no short (no long accepted, or no {hsym} '
                  f'price) -- never a naked short' +
                  (f'; but {held} is still held short from an earlier run: '
                   f'NAKED tonight' if held else ''))
        elif working_side(hsym, 'SELL'):
            why = f'a SELL already working ({held} held, target {target})'
            print(f'  {hsym:<6} skipped (order already working)')
        elif q == 0:
            why = f'already at target ({held} held, target {target})'
            print(f'  {hsym:<6} short already at target ({held}'
                  f' held, target {target})')
        if why:
            hedge_lines.append((hsym, 0, hedge_px, f'none sent: {why}'))
        else:
            tr = place(hsym, 'SELL', q, close=True)
            if tr is None:
                st = 'contract not found'
            else:
                ib.sleep(1.0)
                st = tr.orderStatus.status
                placed += 1
            print(f'  {hsym:<6} SELL {q} MOC (short, {target} target against '
                  f'${base:,.0f} long) -> {st}')
            hedge_lines.append((hsym, -q, hedge_px, st))
            if st in REJECTED:
                print(f'  !! THE {hsym} SHORT WAS NOT ACCEPTED ({st}): the '
                      f'long book is UNHEDGED tonight (beta ~1.4, not ~0.4). '
                      f'Margin account? Shortable? Rule 201?')
    if leg == 'open':
        # ...and COVERS every short stock position the account holds, the
        # mirror of selling every long: the hedge, or a stray short from a
        # failed morning. Widening the long filter to position != 0 instead
        # would send a SELL against a short and double it.
        covers = []
        for sym in sorted(shorts):
            q = shorts[sym]
            if working_side(sym, 'BUY'):
                print(f'  {sym:<6} skipped (order already working)')
                hedge_lines.append((sym, 0, None,
                                    f'none sent: a cover already working '
                                    f'({q} held short)'))
                continue
            tr = place(sym, 'BUY', q, close=False)
            if tr is None:
                print(f'  {sym:<6} skipped (contract not found)')
                hedge_lines.append((sym, q, None, 'contract not found'))
                continue
            covers.append((sym, q, tr))
            placed += 1
        if covers:
            ib.sleep(1.0)              # a rejection can arrive late
        for sym, q, tr in covers:
            st = tr.orderStatus.status
            print(f'  {sym:<6} BUY {q} MKT/OPG (cover) -> {st}')
            hedge_lines.append((sym, q, None, st))
        if not shorts:
            hedge_lines.append((hsym, 0, None, 'none sent: no short held'))
    ib.disconnect()
    return placed, hedge_lines


def write_logs(stamp, month, leg, md, names, total, capital, placed,
               hrows):
    """One orders_log.csv row for the run, and one orders_hedge_log.csv row
    per hedge line (symbol, sized shares, price, sent, status). Shares are
    the sized hedge and sent what reached paper ('' in a dry run), both
    signed: - = sold short at the close, + = bought to cover at the open.
    status is IBKR's answer, or why nothing was sent."""
    os.makedirs(os.path.dirname(OLOG), exist_ok=True)
    new = not os.path.exists(OLOG)
    with open(OLOG, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['run_at_et', 'month', 'leg', 'mode', 'names',
                        'notional', 'capital', 'placed'])
        w.writerow([stamp, month, leg, md, names, f'{total:.2f}',
                    f'{capital:.2f}', '' if placed is None else placed])
    new = not os.path.exists(HLOG)
    with open(HLOG, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['run_at_et', 'month', 'leg', 'mode', 'symbol',
                        'shares', 'price', 'long_notional', 'ratio', 'sent',
                        'status'])
        for s, q, p, sn, st in hrows:
            w.writerow([stamp, month, leg, md, s, q,
                        '' if p is None else f'{p:.2f}', f'{total:.2f}',
                        f'{HEDGE[1]:.2f}', sn, st])


def sent_line(s, q, p, st):
    """One hedge line as it reached paper, for the Telegram: what was sent,
    not what the dry sizing said."""
    if not q:
        return f'{s} {st}'
    act = 'SELL MOC' if q < 0 else 'BUY  MOO'
    return (f'{act} {s} x{abs(q)}' + (f' @ {p:.2f}' if p else '') +
            f' -> {st}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capital', type=float, required=True,
                    help='dollars for the long basket, split equally; the '
                         'QQQ short matches the long actually bought')
    ap.add_argument('--leg', choices=['close', 'open'], default=None,
                    help='close = BUY MOC basket + SELL MOC QQQ tonight; '
                         'open = SELL MOO longs + BUY MOO shorts. Default: '
                         'inferred from the ET clock (dry run only)')
    ap.add_argument('--submit', action='store_true',
                    help='place the orders in IBKR paper (default: dry run)')
    a = ap.parse_args()
    now = dt.datetime.now(ET)
    stamp = now.strftime('%Y-%m-%d %H:%M')
    md = 'paper' if a.submit else 'dry'
    mode = 'SUBMIT (paper)' if a.submit else 'DRY RUN'
    hsym = HEDGE[0]
    leg = a.leg
    if leg is None:
        if a.submit:
            raise SystemExit('--submit requires an explicit --leg')
        leg = 'open' if now.hour < 12 else 'close'
        print(f'leg not given -- inferred "{leg}" from ET clock '
              f'({now:%H:%M} ET); pass --leg to override')
    month = now.strftime('%Y-%m')

    def stand_aside(why, shorts=None):
        """Log the refusal and push it, then exit non-zero: under cron the
        only other trace is qqq.log, which nobody reads."""
        if not a.submit:
            held, risk = 'not read (dry run)', ''
        elif shorts is None:
            held = 'unknown (the account was not read)'
            risk = ('\n!! if a cover failed, a short is open with no long '
                    'against it' if leg == 'close' else
                    '\n!! nothing was sold or covered: last night\'s book '
                    'is still open')
        else:
            held = ' '.join(f'{k} -{v}' for k, v in sorted(shorts.items())) \
                or 'none'
            risk = ('\n!! NAKED SHORT tonight: no long against it'
                    if shorts and leg == 'close' else '')
        write_logs(stamp, month, leg, md, '', 0.0, a.capital,
                   0 if a.submit else None,
                   [(hsym, '', None, 0 if a.submit else '',
                     f'STOOD ASIDE: {why}')])
        print(f'logged -> reports/orders_log.csv and orders_hedge_log.csv '
              f'(stood aside)')
        if send(f'qqq {month} {leg} leg STOOD ASIDE [{mode}]\n{why}\n'
                f'open shorts: {held}{risk}'):
            print('telegram: sent')
        raise SystemExit(why)

    try:
        month, names, emitted = basket(now)
    except SystemExit as ex:
        if leg != 'open':
            stand_aside(str(ex))
        # no registration for this month yet: the open leg still has to exit
        # yesterday's positions (submit mode reads them from the account)
        print(f'{ex}\n-> open leg proceeds without a basket: it will liquidate '
              f'and cover whatever the account holds')
        month, names, emitted = now.strftime('%Y-%m'), [], '-'
    if leg == 'close':
        try:
            check_basket(names)
        except SystemExit as ex:
            stand_aside(str(ex))
    px = prices(names + [hsym]) if names else {}
    rows, per = size(names, px, a.capital) if names else ([], 0.0)
    _, hpx, hq, _ = hedge_leg(rows, px)
    if leg == 'close' and px.get(hsym) is None:
        stand_aside(f'no {hsym} price -- the hedged book cannot be sized; '
                    f'the close leg stands aside rather than trade the '
                    f'basket unhedged')
    if names and per < 10_000:
        print(f'WARNING: ${per:,.0f} per name is below ~$10k -- the broker\'s '
              f'per-order minimum commission alone is 2-4x the modelled '
              f'1 bps crossing at this size (RESULTS 20 cost note)')

    print(f'\n{month} basket (pre-registered {emitted}), {len(names)} names, '
          f'${a.capital:,.0f} -> ${per:,.0f}/name, short {hsym} '
          f'{HEDGE[1]:.2f}:1   [{mode}]')
    act = 'BUY  MOC' if leg == 'close' else 'SELL MOO'
    hact = 'SELL MOC' if leg == 'close' else 'BUY  MOO'
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
    hnot = hq * hpx if hq else 0.0
    if rows and hpx is None:
        print(f'  {hact}  {hsym:<6}     ?? no price -- the hedge is not sized')
    elif rows:
        print(f'  {hact}  {hsym:<6} {hq:>6} @ {hpx:>9.2f}  = '
              f'{hnot:>10,.2f}  {"short" if leg == "close" else "cover"}: '
              f'the hedge')
    print(f'  total {total:>10,.2f} of {a.capital:,.2f} '
          f'({total / a.capital * 100:.0f}% deployed)')
    if rows and hpx is not None:
        print(f'  hedge {hnot:>10,.2f} short {hsym} = '
              f'{hnot / total if total else 0:.3f} x the long (declared '
              f'{HEDGE[1]:.2f}, whole shares)')
    if rows and leg == 'close':
        gross = total + hnot
        print(f'  gross {gross:>10,.2f} overnight: equity to hold >= '
              f'${gross / (REGT_GROSS * REGT_USE):,.0f} (Reg T\'s '
              f'{REGT_GROSS:.0f}x end-of-day line, used to {REGT_USE:.0%}); '
              f'--submit refuses below it')
    if leg == 'close' and (now.hour, now.minute) >= (15, 45):
        print('  WARNING: past ~15:45 ET -- MOC cutoff is 15:50 NYSE / '
              '15:55 Nasdaq')
    if leg == 'open' and now.hour < 12 and (now.hour, now.minute) >= (9, 25):
        print('  WARNING: past 09:25 ET -- MOO/OPG cutoff is 09:28 '
              '(evening pre-placement for tomorrow is fine)')

    placed, sent = None, None
    if a.submit:
        print()
        try:
            placed, sent = submit_paper(rows, leg, now, hedge_px=hpx)
        except SystemExit as ex:
            stand_aside(str(ex), getattr(ex, 'shorts', None))
        print(f'{placed} order(s) placed in paper')

    sign = -1 if leg == 'close' else 1
    sized = sign * hq if rows else ''
    if sent is None:
        hrows = [(hsym, sized, hpx, '', '')] if rows else []
    else:
        hrows = [(s, sized if s == hsym else '',
                  p if p is not None else (hpx if s == hsym else None), q, st)
                 for s, q, p, st in sent]
    write_logs(stamp, month, leg, md, ';'.join(n for n, _, q, _ in rows if q),
               total, a.capital, placed, hrows)
    print(f'logged -> reports/orders_log.csv and orders_hedge_log.csv '
          f'(commit with reports/)')

    if sent is None:
        hedge_msg = (f'\n{hact} {hsym} x{hq} @ {hpx:.2f} (hedge '
                     f'{hnot / total if total else 0:.3f}x)' if hq else '')
    else:
        hedge_msg = ''.join('\n' + sent_line(*h) for h in sent)
    msg = (f'qqq {month} {leg} leg [{mode}]\n'
           + '\n'.join(f'{act} {n} x{q} @ {p:.2f}' for n, p, q, _ in rows
                       if q)
           + hedge_msg
           + f'\ntotal ${total:,.0f} / ${a.capital:,.0f}'
           + (f', short {hsym} ${hnot:,.0f}' if hq and sent is None else '')
           + (f'\n{placed} placed in paper' if placed is not None else '')
           + ''.join(f'\n!! {s} {q:+d} {st}: UNHEDGED tonight'
                     if leg == 'close' else
                     f'\n!! {s} cover {q:+d} {st}: the short stays open'
                     for s, q, _, st in (sent or []) if st in REJECTED))
    if send(msg):
        print('telegram: sent')


if __name__ == '__main__':
    main()
