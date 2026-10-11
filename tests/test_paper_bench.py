# Part of qqq-microstructure.
#
# paper_bench.py splits realized - graded into slippage + weighting +
# coverage + data, and prices commissions before a single order exists. A
# benchmark that is wrong is worse than none, so: the split must add up
# exactly and match hand arithmetic, the fee model must hit every branch of
# the schedule, and the pre-trade sizing must be the order code's own.
#
# Run: python -m pytest tests -q     (or: python tests/test_paper_bench.py)

import os, sys, math
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import paper_bench as pb
import orders


def test_fee_schedule_branches():
    reg = pb.REG
    cat = reg['cat']
    # Fixed, the minimum binds: 1 share of a $1,582 stock -- IBKR's own
    # example is 100 sh @ $25 = $1.00
    assert math.isclose(pb.order_cost('fixed', 1, 1582.0, 'BUY'), 1.00 + cat)
    assert math.isclose(pb.order_cost('fixed', 100, 25.0, 'BUY'),
                        1.00 + 100 * cat)
    # Fixed, per-share binds: IBKR's example 1,000 sh @ $25 = $5.00
    assert math.isclose(pb.order_cost('fixed', 1000, 25.0, 'BUY'),
                        5.00 + 1000 * cat)
    # the 1%-of-value cap binds: 1 share at $35 -> min $1 capped at $0.35
    assert math.isclose(pb.order_cost('fixed', 1, 35.0, 'BUY'), 0.35 + cat)
    # Tiered: commission + pass-throughs on it + exchange + clearing + CAT
    t = pb.FEES['tiered']
    assert math.isclose(pb.order_cost('tiered', 10, 100.0, 'BUY'),
                        0.35 * (1 + t['passthru']) +
                        10 * (t['exch']['BUY'] + t['clearing'] + cat))
    # sells add the SEC fee on proceeds; the TAF only outside its holiday
    in_holiday = pb.order_cost('fixed', 10, 100.0, 'SELL', day='20261105')
    after = pb.order_cost('fixed', 10, 100.0, 'SELL', day='20270105')
    assert math.isclose(in_holiday, 1.00 + 10 * cat + reg['sec_rate'] * 1000)
    assert math.isclose(after - in_holiday, 10 * reg['taf'])
    # the TAF cap binds on a large sale
    big = pb.order_cost('fixed', 100000, 10.0, 'SELL', day='20270105') - \
        pb.order_cost('fixed', 100000, 10.0, 'SELL', day='20261105')
    assert math.isclose(big, reg['taf_cap'])
    # an order that is never sent costs nothing
    assert pb.order_cost('fixed', 0, 100.0, 'BUY') == 0.0


def test_pretrade_sizing_is_the_order_code():
    rng = np.random.default_rng(7)
    for _ in range(50):
        names = [f'S{i}' for i in range(int(rng.integers(5, 15)))]
        px = {n: float(rng.uniform(5, 2000)) for n in names
              if rng.random() > 0.1}
        cap = float(rng.uniform(5e3, 5e5))
        mine, per = pb.sizing(names, px, cap)
        theirs, per2 = orders.size(names, px, cap)
        assert per == per2
        assert list(mine.shares) == [q for _, _, q, _ in theirs]


def _night():
    """One night, three filled names with unequal weights, one basket name
    never held, a buy above the close and a sell below the open."""
    D, Dn = '20260902', '20260903'
    prints = {('A', D): (99.0, 100.0), ('A', Dn): (101.0, 102.0),
              ('B', D): (49.0, 50.0), ('B', Dn): (49.5, 49.0),
              ('C', D): (19.0, 20.0), ('C', Dn): (20.4, 20.0),
              ('X', D): (9.0, 10.0), ('X', Dn): (10.2, 10.0)}
    legs = []
    for sym, q, buy, sell in (('A', 20, 100.10, 100.95),
                              ('B', 30, 50.00, 49.45),
                              ('C', 120, 20.02, 20.40)):
        legs.append(dict(night=D, sym=sym, qty=q, buy_px=buy, sell_px=sell,
                         sell_date=Dn, notional=q * buy,
                         bps=float(np.log(sell / buy) * 1e4), comm=2.0))
    return legs, ['A', 'B', 'C', 'X'], prints


def test_decomposition_adds_up_and_matches_hand_arithmetic():
    legs, basket, prints = _night()
    r_off = {s: np.log(prints[(s, '20260903')][0] / prints[(s, '20260902')][1])
             * 1e4 for s in basket}
    graded_logged = float(np.mean(list(r_off.values()))) - 0.37   # revised
    out = pb.decompose(legs, basket, graded_logged, prints)
    w = {l['sym']: l['notional'] / sum(x['notional'] for x in legs)
         for l in legs}
    realized = sum(w[l['sym']] * l['bps'] for l in legs)
    assert math.isclose(out['realized'], realized)
    assert math.isclose(out['gap'], realized - graded_logged)
    assert math.isclose(out['slippage'] + out['weighting'] + out['coverage']
                        + out['data'], out['gap'], abs_tol=1e-9)
    # each term by hand
    filled = ['A', 'B', 'C']
    assert math.isclose(out['coverage'],
                        np.mean([r_off[s] for s in filled]) -
                        np.mean(list(r_off.values())))
    assert math.isclose(out['weighting'],
                        sum(w[s] * r_off[s] for s in filled) -
                        np.mean([r_off[s] for s in filled]))
    assert math.isclose(out['data'], 0.37, abs_tol=1e-9)
    # the two crossings: A bought 10c above a $100 close, sold 5c under $101
    buyA = np.log(100.10 / 100.0) * 1e4
    sellA = np.log(101.0 / 100.95) * 1e4
    assert buyA > 0 and sellA > 0                   # both cost money
    assert math.isclose(out['slip_buy'] + out['slip_sell'], -out['slippage'],
                        abs_tol=1e-9)
    assert math.isclose(out['comm'],
                        6.0 / sum(l['notional'] for l in legs) * 1e4)


def test_a_night_without_prints_is_reported_not_guessed():
    legs, basket, prints = _night()
    del prints[('B', '20260903')]
    out = pb.decompose(legs, basket, 1.0, prints)
    assert 'why' in out and 'slippage' not in out
    out = pb.decompose(legs, basket, None, prints)
    assert out['why'] == 'not graded yet'


def test_a_split_night_in_paper_is_flagged_not_averaged():
    """Paper does not process splits: 10 pre-split shares sold at the
    post-split price look like a -90% night."""
    legs, basket, prints = _night()
    legs[0] = dict(legs[0], sell_px=legs[0]['sell_px'] / 10,
                   bps=float(np.log(legs[0]['sell_px'] / 10 /
                                    legs[0]['buy_px']) * 1e4))
    out = pb.decompose(legs, basket, 1.0, prints)
    assert 'corporate action' in out['why'] and 'gap' not in out


def test_end_to_end_on_files(capsys):
    legs, basket, prints = _night()
    rows = []
    for l in legs:
        rows.append(dict(exec_id=f'b{l["sym"]}', time_et='', date=l['night'],
                         symbol=l['sym'], side='BOT', qty=l['qty'],
                         price=l['buy_px'], commission=1.0))
        rows.append(dict(exec_id=f's{l["sym"]}', time_et='',
                         date=l['sell_date'], symbol=l['sym'], side='SLD',
                         qty=l['qty'], price=l['sell_px'], commission=1.0))
    fills = pd.DataFrame(rows)
    paper = pd.DataFrame([dict(month='2026-09', q5=';'.join(basket))])
    daily = pd.DataFrame([dict(date='20260902', q5_on_bps=5.0)])
    t = pb.realized_section(fills, paper, daily,
                            lambda *a: prints, ['fixed'])
    assert len(t) == 1 and t.iloc[0].filled == 3 and t.iloc[0].basket == 4
    r = t.iloc[0]
    assert math.isclose(r.slippage + r.weighting + r.coverage + r.data,
                        r.gap, abs_tol=1e-9)
    out = capsys.readouterr().out
    assert 'REALIZED vs GRADED: 1 paper night' in out
    live = r.graded + r.weighting + r.coverage + r.data - r.comm
    assert f'LIVE-EQUIVALENT net' in out and f'{live:+.2f}' in out


def test_panel_parity_matches_on_the_next_day_and_names_the_source(capsys):
    """Night D (buy at D's close, sell at D+1's open) is the panel's on_bps
    on day D+1. A one-day misalignment would still correlate a little; it
    must instead agree exactly on synthetic data."""
    rng = np.random.default_rng(3)
    days = ['20260901', '20260902', '20260903', '20260904']
    names = [f'N{i}' for i in range(6)]
    rows = [dict(ticker=n, day=d, month='2026-09', on_bps=float(rng.normal()),
                 src='yahoo') for n in names for d in days]
    df = pd.DataFrame(rows)
    on = df.set_index(['ticker', 'day']).on_bps
    daily = pd.DataFrame([dict(date=d, month='2026-09', n=6,
                               q5_on_masked=float(np.mean([on[(n, nx)]
                                                           for n in names])))
                          for d, nx in zip(days[:-1], days[1:])])
    paper = pd.DataFrame([dict(month='2026-09', q5=';'.join(names))])
    t = pb.panel_parity(daily, paper, df)
    assert len(t) == 3
    assert np.allclose(t.logged, t.panel)
    out = capsys.readouterr().out
    assert 'panel source yahoo' in out


if __name__ == '__main__':
    for n, fn in list(globals().items()):
        if n.startswith('test_') and 'capsys' not in fn.__code__.co_varnames:
            fn(); print(f'{n}: PASS')
