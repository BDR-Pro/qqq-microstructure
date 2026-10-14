# Part of qqq-microstructure.
#
# capital.py sets the hedged book's capital TARGET and MINIMUM by the rule
# the numbers research declared before computing any hedged-book cost,
# reproduced unchanged in its header. Amendments made after that rule's
# outcome was known are printed as proposals and must decide nothing. A rule
# is only as good as its application, so: the rule is checked on hand-made
# cost curves where the answer is known, its one R5 input is checked as a
# schedule against the rule itself, the ladder it reads must behave as the
# fee schedule says it must, the run must work offline from a prices file,
# and it must end on its one verdict line. No network: yfinance is made
# unimportable.

import os, sys, math
import numpy as np, pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import capital as cp
import paper_bench as pb
import roi_year as ry

OCT = ['MU', 'BE', 'STX', 'SNDK', 'NBIS', 'IREN', 'LRCX', 'WDC', 'AMAT',
       'INTC', 'CAT']
# hand-made October-like closes (not fetched): the spread of prices the real
# basket has, from a $35 name to a $1,500 one
PX = dict(MU=990.0, BE=262.0, STX=753.0, SNDK=1487.0, NBIS=210.0, IREN=34.8,
          LRCX=309.6, WDC=387.0, AMAT=497.5, INTC=101.4, CAT=799.4,
          QQQ=746.0)


def _curve(cost, fin=None, unbought=None):
    n = len(cost)
    return pd.DataFrame(dict(capital=list(cp.LADDER[:n]), cost=cost,
                             fin=fin or [0.0] * n,
                             unbought=unbought or [0] * n))


def test_the_rule_on_hand_made_cost_curves():
    # a curve shaped like an October ladder, 10k..500k
    base = [7.5, 3.96, 2.36, 1.85, 1.63, 1.41, 1.25, 1.16]
    d = cp.decide(_curve(base))
    assert (d['r3'], d['target'], d['minimum']) == (250e3, 250e3, 50e3)
    # thresholds are inclusive
    d = cp.decide(_curve([7.5, 3.0, 2.45, 1.3]))
    assert (d['target'], d['minimum']) == (75e3, 50e3)
    # R5: carry that pushes the R3 rung over 1.3 moves the target up a rung
    fin = [0.1] * 8
    d = cp.decide(_curve(base, fin=fin))
    assert (d['r3'], d['target']) == (250e3, 500e3)
    # ... and off the ladder when no rung passes with it
    assert cp.decide(_curve(base, fin=[0.2] * 8))['target'] is None
    # ... and a credit never pulls it below R3: $150k at 1.41 - 0.5 fails R3
    fin = [0.1, 0.1, 0.1, 0.1, 0.1, -0.5, -0.5, -0.7]
    assert cp.decide(_curve(base, fin=fin))['target'] == 250e3
    # a rung with an unbought name never qualifies, however cheap
    d = cp.decide(_curve([1.0, 1.2, 2.0], unbought=[2, 1, 0]))
    assert (d['target'], d['minimum']) == (None, 50e3)
    # nothing clears: no target, no minimum
    d = cp.decide(_curve([9.0, 5.0, 3.0, 2.5]))
    assert d == dict(r3=None, target=None, minimum=None)


def test_the_declared_rule_decides_and_the_amendments_do_not():
    # R4 and R5 exactly as declared before any hedged cost was computed
    assert cp.LADDER == (10e3, 25e3, 50e3, 75e3, 100e3, 150e3, 250e3, 500e3)
    assert cp.TARGET_BPS == 1.3
    assert cp.MIN_BPS == pytest.approx(2.45) == pb.REF['tilt_cens'] / 2
    assert cp.BORROW == 0.30
    r = cp.rung(OCT, PX, 500e3, 'tiered')          # the defaults are R5's
    book, _ = pb.sizing(OCT, PX, 500e3)
    h = ry.hedge_book('tiered', book, PX, 500e3, 1.0, cp.TAF_DAY, borrow=0.30)
    assert math.isclose(r['fin'], h['borrow'] / 500e3 * 1e4)   # no credit
    # A1: the amendment's 2.34 would not admit $50k at 2.36; R4 does
    base = [7.5, 3.96, 2.36, 1.85, 1.63, 1.41, 1.25, 1.16]
    assert cp.decide(_curve(base))['minimum'] == 50e3
    assert cp.decide(_curve(base), min_bps=cp.AMEND['min_bps'])[
        'minimum'] == 75e3
    assert cp.AMEND['min_bps'] == pytest.approx(2.34)
    if 'cost_neu' in pb.REF:                    # paper_bench's own NEU cost
        assert cp.AMEND['min_bps'] == pytest.approx(pb.REF['cost_neu'])


def test_the_r5_schedule_is_the_rule_at_every_rate():
    """The verdict states R5's input as a schedule; at any rate, the
    schedule's answer must be what the rule itself picks at that rate."""
    # October-like; dearer low-priced names (a lower floor); and a basket
    # whose floor never reaches 1.3 (an empty schedule: no target anywhere)
    for prices, n in ((PX, 2), (dict(PX, IREN=90.0, INTC=180.0), None),
                      (dict(PX, IREN=12.0, QQQ=600.0), 0)):
        t = cp.ladder(OCT, prices, 'tiered')
        sched = cp.r5_schedule(t)
        assert n is None or len(sched) == n
        assert all(b[0] > a[0] and b[1] > a[1]
                   for a, b in zip(sched, sched[1:]))
        for rate in np.arange(-0.5, 1.0, 0.01):
            per = t.short / t.capital * 1e4 * ry.DAYS / 360 / 100
            want = cp.decide(t.assign(fin=rate * per))['target']
            got = next((c for x, c in sched if rate <= x + 1e-12), None)
            assert got == want, (rate, got, want)


def test_the_ladder_behaves_as_the_fee_schedule_says():
    for plan in ('tiered', 'fixed'):
        t = cp.ladder(OCT, PX, plan)
        usd = t.cost * t.capital / 1e4
        # more capital never buys fewer shares or costs fewer dollars ...
        assert (np.diff(t.shares) >= 0).all() and (np.diff(usd) >= -1e-9).all()
        # ... and on this basket the cost per dollar of capital only falls
        assert (np.diff(t.cost) <= 1e-12).all(), t.cost.tolist()
        # the hedge only adds, and it is 1:1 to within half a QQQ share
        assert (t.cost > t.long_bps).all()
        assert ((t.short - t.long).abs() <= PX['QQQ'] / 2 + 1e-9).all()
        assert (t.long <= t.capital + 1e-9).all()
        # Reg T: 50% of each leg; a 1:1 book stays inside 2x equity
        assert np.allclose(t.regt, (t.long + t.short) / 2)
        assert np.allclose(t.fund, (t.long + t.short) / 1.9)
        assert (t.gross_x <= 2.0 + PX['QQQ'] / t.capital).all()


def test_dollar_costs_never_fall_as_capital_rises_on_any_basket():
    rng = np.random.default_rng(5)
    caps = np.arange(5e3, 400e3, 7.5e3)
    for _ in range(12):
        names = [f'N{i}' for i in range(int(rng.integers(5, 13)))]
        px = {n: float(np.exp(rng.uniform(np.log(5), np.log(2000))))
              for n in names}
        px['QQQ'] = float(rng.uniform(300, 900))
        t = cp.ladder(names, px, 'tiered', caps=caps)
        assert (np.diff(t.cost * t.capital) >= -1e-6).all()


def test_the_costs_are_paper_benchs_own_on_the_2027_schedule():
    book, _ = pb.sizing(OCT, PX, 50e3)
    assert math.isclose(cp.long_cost('tiered', book), pb.night_cost('tiered',
                                                                    book))
    r = cp.rung(OCT, PX, 50e3, 'tiered')
    q, leg = ry.hedge_leg('tiered', book.notional.sum(), PX['QQQ'],
                          cp.TAF_DAY)
    assert r['shares'] == q
    assert math.isclose(r['cost'], (cp.long_cost('tiered', book, cp.TAF_DAY)
                                    + leg) / 50e3 * 1e4)
    # the TAF is on in 2027 and falls on the sales: never cheaper than today
    today = cp.rung(OCT, PX, 50e3, 'tiered', day='20261105')
    assert r['cost'] > today['cost']


def test_the_carry_and_the_worlds_are_the_declared_ones():
    # A2's inputs: borrow less IBKR's interest on the short-sale proceeds
    a2 = dict(borrow=ry.BORROW, credit=True)
    small, big = (cp.rung(OCT, PX, c, 'tiered', **a2) for c in (75e3, 500e3))
    assert small['fin'] > 0                        # a $72k short earns nothing
    assert big['fin'] < 0                          # a $496k one earns more
    assert cp.rung(OCT, PX, 500e3, 'tiered')['fin'] > 0      # R5: borrow only
    big = cp.rung(OCT, PX, 500e3, 'tiered')
    w = cp.WORLDS
    assert w['as backtested'] == pytest.approx(4.93)
    assert w['half the edge'] == pytest.approx(4.93 - 0.5 * 4.63)
    assert w['no edge'] == pytest.approx(0.30)
    r = pd.Series(big)
    assert cp.net(r, 4.93) == pytest.approx(r.dep * 4.93 - r.cost - r.fin
                                            - r.dividend)


def test_the_verdict_is_one_line_with_its_margins_and_its_input():
    t = _curve([7.5, 3.96, 2.36, 1.85, 1.63, 1.41, 1.25, 1.16],
               fin=[0.12] * 8)
    t['fund'] = t.capital * 1.04
    t['short'] = t.capital
    line = cp.verdict(cp.decide(t), t, '2026-10', cp.r5_schedule(t))
    assert '\n' not in line and line.startswith('VERDICT (2026-10 basket')
    assert 'the rule as declared before computing' in line
    assert ("TARGET --capital $500,000 (1.16 bps/night; 1.28 with R5's "
            "0.30%/yr borrow, 0.02 inside 1.3)") in line
    assert '>= $520,000 equity' in line
    # the input that decides it, and where it flips the target
    assert "R5's input decides it" in line and 'read in TWS' in line
    assert 'up to 0.12%/yr $250,000, up to 0.35%/yr $500,000' in line
    assert 'MINIMUM $50,000 (2.36 <= 2.45, 0.09 inside)' in line
    none = cp.verdict(cp.decide(_curve([9.0, 5.0])), _curve([9.0, 5.0]),
                      '2026-10')
    assert 'TARGET none on the ladder' in none and 'MINIMUM none' in none


def test_it_runs_offline_from_a_prices_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, 'yfinance', None)    # no network
    paper = tmp_path / 'xsec_paper.csv'
    paper.write_text('emitted_on,month,universe_month,model_frozen,n_scored,'
                     'q5,q1\n20261001,2026-10,2026-09,2026-08-20,57,'
                     + ';'.join(OCT) + ',X;Y\n')
    monkeypatch.setattr(pb, 'PAPER', str(paper))
    f = tmp_path / 'px.csv'
    f.write_text(''.join(f'{k},{v}\n' for k, v in PX.items()))
    out = cp.main(['--prices', str(f)])
    text = capsys.readouterr().out
    lines = text.strip().splitlines()
    assert lines[-1] == out['verdict'] and lines[-1].startswith('VERDICT')
    for want in ('== TIERED (decides)', '== FIXED (reference only)',
                 'THE RULE, APPLIED', 'Reg T floor', 'hold @1.9x',
                 'proposed amendments', 'not binding', 'R4  cost <= 2.45',
                 'inside'):
        assert want in text
    assert 'Reg T needs' not in text
    # the declared rule decides; the amendments are printed beside it
    d, da = out['decision'], out['amended']
    assert d == cp.decide(out['tiered'])
    assert (d['minimum'], da['minimum']) == (50e3, 75e3)
    assert f'TARGET --capital ${d["target"]:,.0f}' in out['verdict']
    assert list(out['tiered'].capital) == list(cp.LADDER)
    # a missing QQQ price is refused, not silently left unhedged
    f.write_text(''.join(f'{k},{v}\n' for k, v in PX.items() if k != 'QQQ'))
    with pytest.raises(SystemExit) as ex:
        cp.main(['--prices', str(f)])
    assert "'QQQ'" in str(ex.value)
