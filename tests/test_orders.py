# Part of qqq-microstructure.
#
# orders.py is the only code that places orders, and since 2026-10-14 it
# trades the QQQ-hedged book: the basket long at the close, QQQ short against
# it, both unwound at the open. These tests pin the rails that make a short
# safe to automate: the short is sized from the long actually bought (never
# from --capital, never naked), a re-run tops up and never doubles it, the
# open leg covers every short as it sells every long, Reg T is checked before
# a single order, only the declared hedge is ever shorted, and the
# paper-only gate still holds. ib_insync is a fake installed into
# sys.modules: no test touches the network or a broker.
#
# Run: python -m pytest tests/test_orders.py -q

import os, sys, csv, types
from types import SimpleNamespace as NS
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import orders

H = orders.HEDGE[0]


class Stock:
    def __init__(self, symbol, exchange, currency):
        self.symbol, self.exchange, self.currency = symbol, exchange, currency
        self.secType = 'STK'


class Order:
    def __init__(self, action, orderType, totalQuantity, tif=''):
        self.action, self.orderType = action, orderType
        self.totalQuantity, self.tif = totalQuantity, tif


class FakeIB:
    """The slice of ib_insync.IB that orders.py uses. positions: (symbol,
    signed qty, avgCost); working: (symbol, action, orderType, qty[, status])
    from an earlier run; reject: symbols whose orders IBKR answers
    'Inactive'; late: {symbol: status} that arrives only after a pause of a
    second or more; missing: symbols qualifyContracts cannot find; down: TWS
    not reachable."""

    # ib_insync 0.9.86 order.py: openTrades() leaves out only these
    DONE = {'Filled', 'Cancelled', 'ApiCancelled'}

    def __init__(self, account='DU1234567', positions=(), nlv=1e6,
                 working=(), reject=(), late=None, missing=(), down=False):
        self.account, self._pos, self.nlv = account, positions, nlv
        self.working, self.reject = working, set(reject)
        self.late, self.missing, self.down = dict(late or {}), set(missing), down
        self.placed, self.trades, self.connected = [], [], False

    def connect(self, host, port, clientId, timeout):
        if self.down:
            raise ConnectionRefusedError
        self.connected = True

    def disconnect(self):
        self.connected = False

    def managedAccounts(self):
        return [self.account]

    def positions(self):
        return [NS(contract=NS(symbol=s, secType='STK'), position=float(q),
                   avgCost=c) for s, q, c in self._pos]

    def openTrades(self):
        out = []
        for w in self.working:
            s, a, t, q = w[:4]
            st = w[4] if len(w) > 4 else 'PreSubmitted'
            if st not in self.DONE:
                out.append(NS(contract=NS(symbol=s),
                              order=NS(action=a, orderType=t,
                                       totalQuantity=q),
                              orderStatus=NS(status=st)))
        return out

    def accountValues(self):
        if self.nlv is None:
            return []
        return [NS(tag='NetLiquidation', value=str(self.nlv), currency='USD',
                   account=self.account)]

    def qualifyContracts(self, c):
        return [] if c.symbol in self.missing else [c]

    def placeOrder(self, c, o):
        self.placed.append((c.symbol, o.action, o.orderType, o.tif,
                            o.totalQuantity))
        st = 'Inactive' if c.symbol in self.reject else 'PreSubmitted'
        tr = NS(contract=c, order=o, orderStatus=NS(status=st))
        self.trades.append(tr)
        return tr

    def sleep(self, s):
        if s >= 1.0:
            for tr in self.trades:
                if tr.contract.symbol in self.late:
                    tr.orderStatus.status = self.late[tr.contract.symbol]


@pytest.fixture
def broker(monkeypatch):
    def install(**kw):
        ib = FakeIB(**kw)
        mod = types.ModuleType('ib_insync')
        mod.IB, mod.Stock, mod.Order = (lambda: ib), Stock, Order
        monkeypatch.setitem(sys.modules, 'ib_insync', mod)
        return ib
    return install


# A basket with one name priced above its slice (B) and one unpriced (D):
# $10k over 4 names is $2,500 a slice, so A 25 sh, C 50 sh, B and D nothing.
NAMES = ['A', 'B', 'C', 'D']
PX = {'A': 100.0, 'B': 3000.0, 'C': 50.0, H: 740.0}
CAP = 10_000.0


def _book():
    rows, _ = orders.size(NAMES, PX, CAP)
    return rows


def test_the_hedge_is_sized_from_the_long_bought_not_from_capital():
    sym, p, q, long_ = orders.hedge_leg(_book(), PX)
    assert (sym, p, long_) == (H, 740.0, 5000.0)
    assert q == round(5000.0 / 740.0) == 7          # not round(10000 / 740)
    # nearest share, not floor: $25k at $739.53 is 33.8 -> 34 (33 would be
    # a lasting 2.4% under-hedge)
    assert orders.hedge_shares(25_000, 739.53) == 34
    assert orders.hedge_shares(7_400, 740.0, ratio=0.5) == 5
    # nothing bought, or no hedge price: no short at all
    assert orders.hedge_shares(0.0, 740.0) == 0
    assert orders.hedge_shares(5000.0, None) == 0
    assert orders.hedge_leg(orders.size(['B'], PX, 2_500)[0], PX)[2] == 0
    assert orders.HEDGE == ('QQQ', 1.0)


def test_reg_t_refuses_a_book_over_95pct_of_twice_equity(broker):
    # gross 99,000 vs 2 x 0.95 x 52,000 = 98,800: refused; at 53,000 it fits
    msg = orders.regt_check(50_000, -49_000, 52_000)
    assert msg and 'REG T' in msg and 'Nothing was placed' in msg
    assert orders.regt_check(50_000, -49_000, 53_000) is None
    assert 'NetLiquidation' in orders.regt_check(1.0, 1.0, None)
    # in paper: refused BEFORE any order, and the connection is closed
    rows = _book()                                  # 5,000 long + 7 x 740 short
    ib = broker(nlv=5_000)
    with pytest.raises(SystemExit) as ex:
        orders.submit_paper(rows, 'close', None, hedge_px=740.0)
    assert 'REG T' in str(ex.value)
    assert ib.placed == [] and not ib.connected
    # an account that cannot report its equity is refused too
    ib = broker(nlv=None)
    with pytest.raises(SystemExit):
        orders.submit_paper(rows, 'close', None, hedge_px=740.0)
    assert ib.placed == []


def test_the_close_leg_buys_the_basket_then_shorts_only_the_hedge(broker):
    ib = broker()
    placed, hedge = orders.submit_paper(_book(), 'close', None,
                                        hedge_px=740.0)
    assert ib.placed == [('A', 'BUY', 'MOC', '', 25),
                         ('C', 'BUY', 'MOC', '', 50),
                         (H, 'SELL', 'MOC', '', 7)]   # the basket first
    assert placed == 3 and hedge == [(H, -7, 740.0, 'PreSubmitted')]
    sells = [o for o in ib.placed if o[1] == 'SELL']
    assert [o[0] for o in sells] == [H]               # nothing else short


def test_the_short_is_sized_from_accepted_longs_and_never_naked(broker):
    ib = broker(reject={'C'})                         # C's BUY is rejected
    orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed[-1] == (H, 'SELL', 'MOC', '', round(2500 / 740))
    ib = broker(reject={'A', 'C'})                    # nothing accepted
    _, hedge = orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert [o for o in ib.placed if o[1] == 'SELL'] == []
    assert [(s, q) for s, q, _, _ in hedge] == [(H, 0)]   # none sent ...
    assert hedge[0][3].startswith('none sent: target 0')  # ... and why
    assert 'NAKED' not in hedge[0][3]
    # ... unless a short is still open from an earlier night: then it says so
    ib = broker(reject={'A', 'C'}, positions=[(H, -7, 740.0)])
    _, hedge = orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert hedge[0][:2] == (H, 0) and '7 held -- NAKED tonight' in hedge[0][3]


def test_the_close_leg_tops_up_a_short_and_never_doubles_it(broker,
                                                              capsys):
    assert orders.short_quantity(134, 0) == 134
    assert orders.short_quantity(134, 50) == 84
    assert orders.short_quantity(134, 134) == 0
    assert orders.short_quantity(134, 200) == 0
    # last night's cover failed: 3 sh still short, target 7 -> sell 4
    ib = broker(positions=[(H, -3, 740.0)])
    orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed[-1] == (H, 'SELL', 'MOC', '', 4)
    # the warning names the other way a short gets there: MOM in this account
    assert 'live_momentum.py' in capsys.readouterr().out
    # the whole hedge (and the basket) still held: no new order at all
    ib = broker(positions=[('A', 25, 100.0), ('C', 50, 50.0), (H, -7, 740.0)])
    _, hedge = orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed == []
    assert hedge == [(H, 0, 740.0,
                      'none sent: already at target (7 held, target 7)')]
    # a SELL already working for the hedge: skipped, not sent twice
    ib = broker(working=[(H, 'SELL', 'MOC', 7)])
    _, hedge = orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert [o for o in ib.placed if o[0] == H] == []
    assert hedge[0][3].startswith('none sent: a SELL already working')


def test_the_long_still_held_counts_in_the_hedge_base(broker):
    """The open leg failed and A is still held: A is not bought again, but it
    is still long overnight, so it is still hedged."""
    ib = broker(positions=[('A', 25, 100.0)])
    orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed == [('C', 'BUY', 'MOC', '', 50),
                         (H, 'SELL', 'MOC', '', 7)]


def test_the_open_leg_sells_every_long_and_covers_every_short(broker):
    ib = broker(positions=[('A', 25, 100.0), ('XYZ', 3, 10.0),
                           (H, -7, 740.0), ('SPY', -2, 600.0)])
    placed, hedge = orders.submit_paper(_book(), 'open', None)
    assert sorted(ib.placed) == sorted([
        ('A', 'SELL', 'MKT', 'OPG', 25),       # the basket long
        ('XYZ', 'SELL', 'MKT', 'OPG', 3),      # a long outside the basket
        (H, 'BUY', 'MKT', 'OPG', 7),           # the hedge cover
        ('SPY', 'BUY', 'MKT', 'OPG', 2)])      # a stray short, covered too
    assert placed == 4
    assert sorted((s, q) for s, q, _, _ in hedge) == sorted([('SPY', 2), (H, 7)])
    # no SELL is ever sent against a short (that would double it)
    assert not [o for o in ib.placed if o[0] in (H, 'SPY') and o[1] == 'SELL']
    # a cover already working is not sent twice
    ib = broker(positions=[(H, -7, 740.0)], working=[(H, 'BUY', 'MKT', 7)])
    _, hedge = orders.submit_paper(_book(), 'open', None)
    assert ib.placed == []
    assert hedge[0][:2] == (H, 0) and 'cover already working' in hedge[0][3]
    # nothing short: said so, nothing sent
    ib = broker(positions=[('A', 25, 100.0)])
    _, hedge = orders.submit_paper(_book(), 'open', None)
    assert hedge == [(H, 0, None, 'none sent: no short held')]


def test_the_hedge_symbol_in_the_basket_is_refused(broker, monkeypatch,
                                                  tmp_path):
    with pytest.raises(SystemExit) as ex:
        orders.check_basket(['MU', H, 'AMAT'])
    assert H in str(ex.value)
    orders.check_basket(['MU', 'AMAT'])
    # main() refuses the close leg before pricing anything
    monkeypatch.setattr(orders, 'basket', lambda now: ('2026-10', ['MU', H],
                                                       '20261001'))
    monkeypatch.setattr(orders, 'prices', lambda n: pytest.fail('priced'))
    monkeypatch.setattr(orders, 'OLOG', str(tmp_path / 'o.csv'))
    monkeypatch.setattr(orders, 'HLOG', str(tmp_path / 'h.csv'))
    monkeypatch.setattr(orders, 'send', lambda msg: False)
    monkeypatch.setattr(sys, 'argv', ['orders.py', '--capital', '1e5',
                                      '--leg', 'close'])
    with pytest.raises(SystemExit):
        orders.main()
    # the account holding the hedge LONG: a SELL would not short; refused
    ib = broker(positions=[(H, 5, 740.0)])
    with pytest.raises(SystemExit):
        orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed == []


def test_the_paper_only_gate_still_refuses_a_live_account(broker):
    ib = broker(account='U7654321')
    with pytest.raises(SystemExit) as ex:
        orders.connect_paper()
    assert 'not an IBKR PAPER account' in str(ex.value)
    assert not ib.connected
    ib = broker(account='U7654321')
    with pytest.raises(SystemExit):
        orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed == []


def _dry_run(monkeypatch, tmp_path, leg, px=PX, msgs=None, submit=False):
    monkeypatch.setattr(orders, 'basket', lambda now: ('2026-10', NAMES,
                                                       '20261001'))
    monkeypatch.setattr(orders, 'prices', lambda n: {k: px[k] for k in n
                                                     if k in px})
    sent = msgs if msgs is not None else []
    monkeypatch.setattr(orders, 'send', lambda msg: sent.append(msg) and False)
    monkeypatch.setattr(orders, 'OLOG', str(tmp_path / 'orders_log.csv'))
    monkeypatch.setattr(orders, 'HLOG', str(tmp_path / 'orders_hedge_log.csv'))
    monkeypatch.setattr(sys, 'argv', ['orders.py', '--capital', str(CAP),
                                      '--leg', leg]
                        + (['--submit'] if submit else []))
    orders.main()


def test_the_dry_run_shows_and_logs_the_hedge_and_places_nothing(
        broker, monkeypatch, tmp_path, capsys):
    ib = broker()
    olog = tmp_path / 'orders_log.csv'
    olog.write_text('run_at_et,month,leg,mode,names,notional,capital,placed\n'
                    '2026-08-22 15:08,2026-08,close,dry,LRCX;NBIS,2628.44,'
                    '5000.00,\n')
    _dry_run(monkeypatch, tmp_path, 'close')
    out = capsys.readouterr().out
    assert f'SELL MOC  {H}' in out and '7 @    740.00' in out
    assert 'x the long' in out and 'equity to hold >= $5,358' in out
    assert ib.placed == [] and not ib.connected
    # orders_log.csv: header and old rows untouched, one row appended
    log = pd.read_csv(olog, dtype=str)
    assert list(log.columns) == ['run_at_et', 'month', 'leg', 'mode',
                                 'names', 'notional', 'capital', 'placed']
    assert len(log) == 2 and log.names.iloc[1] == 'A;C'
    assert log.notional.iloc[1] == '5000.00'
    # the hedge leg in its own log, shares signed (- = short)
    h = list(csv.DictReader(open(tmp_path / 'orders_hedge_log.csv')))
    assert len(h) == 1 and h[0]['symbol'] == H and h[0]['shares'] == '-7'
    assert h[0]['long_notional'] == '5000.00' and h[0]['mode'] == 'dry'
    # the open leg's dry run shows the cover
    _dry_run(monkeypatch, tmp_path, 'open')
    out = capsys.readouterr().out
    assert f'BUY  MOO  {H}' in out and 'cover' in out
    h = list(csv.DictReader(open(tmp_path / 'orders_hedge_log.csv')))
    assert h[-1]['shares'] == '7' and h[-1]['leg'] == 'open'


def test_no_hedge_price_stands_the_close_leg_aside_logged_and_pushed(
        monkeypatch, tmp_path):
    px = {k: v for k, v in PX.items() if k != H}
    msgs = []
    with pytest.raises(SystemExit) as ex:
        _dry_run(monkeypatch, tmp_path, 'close', px=px, msgs=msgs)
    assert 'stands aside' in str(ex.value)
    # not silent: a log row and a Telegram, nothing traded
    log = pd.read_csv(tmp_path / 'orders_log.csv', dtype=str)
    assert len(log) == 1 and log['names'].isna().all()
    h = list(csv.DictReader(open(tmp_path / 'orders_hedge_log.csv')))
    assert h[0]['status'].startswith('STOOD ASIDE: no QQQ price')
    assert len(msgs) == 1 and 'close leg STOOD ASIDE [DRY RUN]' in msgs[0]


# --- an earlier run's rejected orders, late statuses, missing contracts -----

def test_a_rejected_order_from_an_earlier_run_is_not_working(broker):
    """openTrades() still lists an 'Inactive' order. It must neither block
    the retry nor count as a long in the hedge base."""
    earlier = [('A', 'BUY', 'MOC', 25, 'PreSubmitted'),
               ('C', 'BUY', 'MOC', 50, 'Inactive')]
    # C is retried; the working A and the retried C make the base: 7 sh
    ib = broker(working=earlier)
    orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed == [('C', 'BUY', 'MOC', '', 50),
                         (H, 'SELL', 'MOC', '', 7)]
    # C rejected again: the base is A alone, $2,500 -> 3 sh, not 7
    ib = broker(working=earlier, reject={'C'})
    orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed[-1] == (H, 'SELL', 'MOC', '', 3)
    # an Inactive SELL of the hedge does not stop the short being sent
    ib = broker(working=[(H, 'SELL', 'MOC', 7, 'Inactive')])
    orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed[-1] == (H, 'SELL', 'MOC', '', 7)
    # ... nor an Inactive cover the morning cover
    ib = broker(positions=[(H, -7, 740.0)],
                working=[(H, 'BUY', 'MKT', 7, 'Inactive')])
    orders.submit_paper(_book(), 'open', None)
    assert ib.placed == [(H, 'BUY', 'MKT', 'OPG', 7)]
    # done orders are not listed at all, and change nothing
    ib = broker(working=[('C', 'BUY', 'MOC', 50, 'Cancelled')])
    orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed[-1] == (H, 'SELL', 'MOC', '', 7)


def test_a_late_rejection_is_read_before_the_short_is_sized(broker, capsys):
    # C's BUY is accepted on placement and rejected a second later: the short
    # is sized from A alone
    ib = broker(late={'C': 'Inactive'})
    orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert ib.placed[-1] == (H, 'SELL', 'MOC', '', 3)
    # the short itself rejected late: reported, and said to be unhedged
    ib = broker(late={H: 'Inactive'})
    _, hedge = orders.submit_paper(_book(), 'close', None, hedge_px=740.0)
    assert hedge == [(H, -7, 740.0, 'Inactive')]
    assert 'UNHEDGED tonight' in capsys.readouterr().out
    # a cover rejected late is reported as rejected, not as working
    ib = broker(positions=[(H, -7, 740.0)], late={H: 'Inactive'})
    _, hedge = orders.submit_paper(_book(), 'open', None)
    assert hedge == [(H, 7, None, 'Inactive')]


def test_a_hedge_contract_not_found_is_an_unhedged_night(broker, capsys):
    ib = broker(missing={H})
    placed, hedge = orders.submit_paper(_book(), 'close', None,
                                        hedge_px=740.0)
    assert placed == 2 and [o[0] for o in ib.placed] == ['A', 'C']
    assert hedge == [(H, -7, 740.0, 'contract not found')]
    assert 'UNHEDGED tonight' in capsys.readouterr().out
    ib = broker(positions=[(H, -7, 740.0)], missing={H})
    placed, hedge = orders.submit_paper(_book(), 'open', None)
    assert placed == 0 and hedge == [(H, 7, None, 'contract not found')]


# --- main --submit, end to end on the fake broker --------------------------

def _submit(monkeypatch, tmp_path, leg, msgs):
    return _dry_run(monkeypatch, tmp_path, leg, msgs=msgs, submit=True)


def _hlog(tmp_path):
    return list(csv.DictReader(open(tmp_path / 'orders_hedge_log.csv')))


def test_main_submit_logs_and_pushes_what_was_sent(broker, monkeypatch,
                                                   tmp_path):
    # close leg, 3 sh already short from a failed cover: 4 sent, and the
    # Telegram says 4 -- not the sized 7
    msgs = []
    ib = broker(positions=[(H, -3, 740.0)])
    _submit(monkeypatch, tmp_path, 'close', msgs)
    assert ib.placed[-1] == (H, 'SELL', 'MOC', '', 4)
    h = _hlog(tmp_path)[-1]
    assert (h['mode'], h['shares'], h['sent'], h['status']) == \
        ('paper', '-7', '-4', 'PreSubmitted')
    assert 'SELL MOC QQQ x4 @ 740.00 -> PreSubmitted' in msgs[-1]
    assert 'x7' not in msgs[-1] and 'short QQQ $' not in msgs[-1]
    log = pd.read_csv(tmp_path / 'orders_log.csv', dtype=str)
    assert (log['mode'].iloc[-1], log.placed.iloc[-1]) == ('paper', '3')
    # nothing sent (already at target): the Telegram says why
    ib = broker(positions=[('A', 25, 100.0), ('C', 50, 50.0), (H, -7, 740.0)])
    _submit(monkeypatch, tmp_path, 'close', msgs)
    assert 'QQQ none sent: already at target (7 held, target 7)' in msgs[-1]
    assert _hlog(tmp_path)[-1]['sent'] == '0'
    # the short rejected late: the hedge log and the UNHEDGED push say so
    ib = broker(late={H: 'Inactive'})
    _submit(monkeypatch, tmp_path, 'close', msgs)
    assert _hlog(tmp_path)[-1]['status'] == 'Inactive'
    assert '!! QQQ -7 Inactive: UNHEDGED tonight' in msgs[-1]
    # open leg, 5 sh short: 5 covered, and the Telegram says 5
    ib = broker(positions=[('A', 25, 100.0), (H, -5, 740.0)])
    _submit(monkeypatch, tmp_path, 'open', msgs)
    h = _hlog(tmp_path)[-1]
    assert (h['leg'], h['shares'], h['sent'], h['status']) == \
        ('open', '7', '5', 'PreSubmitted')
    assert 'BUY  MOO QQQ x5 -> PreSubmitted' in msgs[-1]
    # a cover IBKR rejects is pushed: the short stays open
    ib = broker(positions=[(H, -5, 740.0)], reject={H})
    _submit(monkeypatch, tmp_path, 'open', msgs)
    assert '!! QQQ cover +5 Inactive: the short stays open' in msgs[-1]


def test_main_submit_refusals_are_logged_and_pushed(broker, monkeypatch,
                                                    tmp_path):
    msgs = []
    # Reg T refuses the night while this morning's cover failed: the open
    # short is named and called naked
    ib = broker(positions=[(H, -7, 740.0)], nlv=1_000)
    with pytest.raises(SystemExit) as ex:
        _submit(monkeypatch, tmp_path, 'close', msgs)
    assert 'REG T' in str(ex.value) and ib.placed == []
    assert 'close leg STOOD ASIDE [SUBMIT (paper)]' in msgs[-1]
    assert 'open shorts: QQQ -7' in msgs[-1] and 'NAKED SHORT' in msgs[-1]
    assert _hlog(tmp_path)[-1]['status'].startswith('STOOD ASIDE: REG T')
    log = pd.read_csv(tmp_path / 'orders_log.csv', dtype=str)
    assert (log['mode'].iloc[-1], log.placed.iloc[-1]) == ('paper', '0')
    # the account long the hedge symbol
    ib = broker(positions=[(H, 10, 740.0)])
    with pytest.raises(SystemExit):
        _submit(monkeypatch, tmp_path, 'close', msgs)
    assert 'LONG 10 QQQ' in msgs[-1] and 'open shorts: none' in msgs[-1]
    # TWS not reachable: the shorts are unknown, and the push says what
    # that can mean
    broker(down=True)
    with pytest.raises(SystemExit):
        _submit(monkeypatch, tmp_path, 'close', msgs)
    assert 'cannot reach TWS' in msgs[-1] and 'unknown' in msgs[-1]
    broker(down=True)
    with pytest.raises(SystemExit):
        _submit(monkeypatch, tmp_path, 'open', msgs)
    assert 'open leg STOOD ASIDE' in msgs[-1]
    assert "last night's book is still open" in msgs[-1]
    assert len(_hlog(tmp_path)) == 4


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
