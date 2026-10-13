# Part of qqq-microstructure.
#
# leverage.py answers "would leverage make the traded book perform better",
# and every column it prints rests on a piece of arithmetic that can be
# planted and checked: the growth-optimal fraction, the interest on the
# debit and the days it is charged for, the commissions at the levered
# notional, the Reg T cap, the maintenance liquidation, the fixed-notional
# account, the proxy's construction and banner, and the pre-declared
# decision rule. No test touches the network: series, bars, prices and the
# live log are all injected.

import os, sys, math
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import leverage as lv
import roi_year as ry
import paper_bench as pb

NAMES = [f'S{i}' for i in range(11)]
PX = {n: 40.0 + 61.0 * i for i, n in enumerate(NAMES)}
PX['QQQ'] = 740.0
CAP = 25000.0
ZERO = lambda d: lv.interest(d, flat=0.0)


def _acct(notional, cost=0.0, maint=0.25):
    return dict(L=notional / CAP, notional=float(notional), dep=notional / CAP,
                cost=cost, maint=maint, held=11, names=11)


def _no_book(f):
    return f, 0.0


def _no_fin(x):
    return 0.0


def test_kelly_recovers_mu_over_sigma2_on_planted_truth():
    mu, sd = 4e-4, 1e-2                       # 4 bps a night at 100 bps sd
    z = np.random.default_rng(11).standard_normal(50000)
    R = mu + sd * (z - z.mean()) / z.std()
    gaps = np.ones_like(R)
    f, g = lv.kelly(R, gaps, _no_book, _no_fin)
    assert abs(f / (mu / sd ** 2) - 1) < 0.02
    assert math.isclose(lv.kelly_closed(R, gaps, 0.0, 0.0), mu / sd ** 2,
                        rel_tol=1e-9)
    # and through the bootstrap the section actually uses: the optimum of
    # the nights it drew (whose own mean carries resampling noise)
    r = 1e4 * np.log1p(R)
    Rb, Gb, _ = lv.draw(r, gaps, 2000)
    fb, _ = lv.kelly(Rb, Gb, _no_book, _no_fin)
    assert abs(fb / (Rb.mean() / Rb.var()) - 1) < 0.01
    assert abs(fb / (mu / sd ** 2) - 1) < 0.10


def test_kelly_pays_for_financing_and_never_trades_a_losing_world():
    mu, sd = 4e-4, 1e-2
    z = np.random.default_rng(12).standard_normal(50000)
    R = mu + sd * (z - z.mean()) / z.std()
    gaps = np.ones_like(R)
    c = 1.5e-4                                # per unit borrowed per night
    fin = lambda x: max(0.0, x - 1) * c
    f, _ = lv.kelly(R, gaps, _no_book, fin)
    assert abs(f / ((mu - c) / sd ** 2) - 1) < 0.03
    f0, g0 = lv.kelly(R - 2 * mu, gaps, _no_book, _no_fin)
    assert f0 == 0.0 and g0 == 0.0


def test_the_feasible_kelly_obeys_reg_t_and_maintenance():
    # planted truth whose textbook f* is 4: no IBKR account can hold it
    mu, sd = 4e-4, 1e-2
    z = np.random.default_rng(13).standard_normal(50000)
    R = mu + sd * (z - z.mean()) / z.std()
    gaps = np.ones_like(R)
    f_free, _ = lv.kelly(R, gaps, _no_book, _no_fin)
    f_regt, g_regt = lv.kelly(R, gaps, _no_book, _no_fin, xcap=lv.REGT,
                              maint=0.25)
    assert f_free > 3.5 and f_regt == lv.REGT
    assert math.isclose(g_regt, lv.growth(R, gaps, 2.0, _no_book, _no_fin))
    assert lv.growth(R, gaps, 2.01, _no_book, _no_fin, xcap=lv.REGT) == -np.inf
    # maintenance inside the growth sum: at 3x a -12% night leaves equity
    # 0.64 against 25% of a 2.64 position (0.66) -- liquidated, and the
    # liquidator's slippage is paid on the position; -10% (0.70 vs 0.675)
    # is not a breach
    one = np.ones(1)
    g = lv.growth(np.array([-0.12]), one, 3.0, _no_book, _no_fin, maint=0.25)
    assert math.isclose(g, math.log1p(-0.36 - lv.LIQ_SLIP / 1e4 * 2.64))
    g = lv.growth(np.array([-0.10]), one, 3.0, _no_book, _no_fin, maint=0.25)
    assert math.isclose(g, math.log1p(-0.30))
    # above 1/m the account is in deficit at entry: every night is a breach
    g = lv.growth(np.zeros(5), np.ones(5), 4.2, _no_book, _no_fin, maint=0.25)
    assert math.isclose(g, math.log1p(-lv.LIQ_SLIP / 1e4 * 4.2))


def test_the_closed_form_sees_the_fixed_charge_corner():
    def planted(mu, sd, seed):
        z = np.random.default_rng(seed).standard_normal(50000)
        return mu + sd * (z - z.mean()) / z.std()
    # interior below 1x: mu/s2 = 0.5, growth there +1.25e-5 a night; a fixed
    # charge of 2e-5 makes not trading better, and both answers say 0
    R = planted(0.5e-4, 1e-2, 14)
    gaps = np.ones_like(R)
    assert math.isclose(lv.kelly_closed(R, gaps, 0.0, 0.0), 0.5, rel_tol=1e-9)
    assert lv.kelly_closed(R, gaps, 0.0, 0.0, 2e-5) == 0.0
    fixed = lambda f: (f, 2e-5 if f > 0 else 0.0)
    assert lv.kelly(R, gaps, fixed, _no_fin)[0] == 0.0
    # the kink (the reviewed cell): mu/s2 > 1 > (mu - c)/s2 put the first
    # draft at 1.00 however large the fixed charge
    R = planted(1.2e-4, 1e-2, 15)
    gaps = np.ones_like(R)
    c = 0.5e-4
    fin = lambda x: max(0.0, x - 1) * c
    assert lv.kelly_closed(R, gaps, 0.0, c) == 1.0
    assert lv.kelly_closed(R, gaps, 0.0, c, 0.5e-4) == 1.0   # g +0.2e-4
    assert lv.kelly_closed(R, gaps, 0.0, c, 1e-4) == 0.0     # g -0.3e-4
    big = lambda f: (f, 1e-4 if f > 0 else 0.0)
    assert lv.kelly(R, gaps, big, fin)[0] == 0.0


def test_interest_is_ibkrs_blended_tiers():
    # IBKR's own worked examples at BM 3.88: $50k all at 5.38%; $250k is
    # 100k at 5.38% + 150k at 4.88% = 5.08% blended
    assert math.isclose(lv.interest(50e3), 50e3 * 0.0538)
    assert math.isclose(lv.interest(250e3) / 250e3, 0.0508)
    assert math.isclose(lv.interest(2e6),
                        100e3 * .0538 + 900e3 * .0488 + 1e6 * .0463)
    assert lv.interest(-5000.0) == 0.0         # cash is not a debit
    assert math.isclose(lv.interest(40e3, flat=5.0), 2000.0)
    assert math.isclose(lv.interest(50e3, bm=5.0), 50e3 * 0.065)


def test_financing_cuts_expected_night_by_exactly_the_borrowed_fraction():
    r = np.random.default_rng(4).normal(9.0, 110.0, 3000)
    gm = 1.452
    rate = 5.0
    for L in (1.25, 1.5, 2.0):
        a = lv.account(NAMES, PX, CAP, L, 'tiered')
        e0 = lv.expected_night(r, a, CAP, gm, ZERO)
        e1 = lv.expected_night(r, a, CAP, gm, lambda d: lv.interest(d, flat=rate))
        assert math.isclose(e0 - e1, (a['dep'] - 1) * rate / 100 * gm / 360
                            * 1e4, rel_tol=1e-12)
    # 1x deploys less than capital: nothing is borrowed, nothing is charged
    a = lv.account(NAMES, PX, CAP, 1.0, 'tiered')
    assert a['dep'] < 1
    assert lv.expected_night(r, a, CAP, gm, lambda d: lv.interest(d, flat=rate)) \
        == lv.expected_night(r, a, CAP, gm, ZERO)


def test_each_night_is_charged_its_own_calendar_days_on_the_actual_debit():
    a = _acct(1.5 * CAP)
    rate = lambda d: lv.interest(d, flat=5.0)
    eq, called, fin = lv.simulate(np.zeros((1, 3)), np.array([[1, 1, 3]]),
                                  CAP, a, rate)
    e = [CAP]
    for g in (1, 1, 3):            # the debit grows by the interest itself
        e.append(e[-1] - (1.5 * CAP - e[-1]) * 0.05 * g / 360)
    assert np.allclose(eq[0], e, rtol=0, atol=1e-9)
    assert math.isclose(fin[0], CAP - e[-1])
    # a winning book borrows less: the basket up 5% (10% on equity), the next
    # debit is smaller
    eq, _, fin = lv.simulate(np.array([[0.05, 0.0]]), np.ones((1, 2)), CAP, a,
                             rate)
    e1 = CAP + 1.5 * CAP * 0.05 - 0.5 * CAP * 0.05 / 360
    assert math.isclose(eq[0, 2], e1 - (1.5 * CAP - e1) * 0.05 / 360)
    # at exactly 2x the first night's interest is a loss, so the cap binds
    # and the next debit is 2E - E = E
    eq, _, _ = lv.simulate(np.zeros((1, 2)), np.ones((1, 2)), CAP,
                           _acct(2 * CAP), rate)
    e1 = CAP - CAP * 0.05 / 360
    assert math.isclose(eq[0, 2], e1 - e1 * 0.05 / 360)


def test_commissions_at_the_levered_notional_are_paper_benchs():
    for L in lv.GRID:
        a = lv.account(NAMES, PX, CAP, L, 'tiered')
        book, _ = pb.sizing(NAMES, PX, L * CAP)
        assert a['cost'] == pb.night_cost('tiered', book)
        assert a['notional'] == book.notional.sum()
        assert math.isclose(a['cost_bps'], a['cost'] / CAP * 1e4)
    # the per-order minimum: per dollar traded the drag falls as L rises,
    # per dollar of equity it does not
    a1 = lv.account(NAMES, PX, CAP, 1.0, 'tiered')
    a2 = lv.account(NAMES, PX, CAP, 2.0, 'tiered')
    assert a2['cost_traded'] < a1['cost_traded']
    assert a2['cost_bps'] > a1['cost_bps']
    # and the fan charges exactly that, every night, while nothing cuts the
    # book back: a flat zero-return year at 1.5x (2E stays above the book)
    a15 = lv.account(NAMES, PX, CAP, 1.5, 'tiered')
    s = lv.fan(np.zeros(500), np.ones(500), a15, CAP, 20, ZERO)
    assert math.isclose(s['p50'], -lv.H * a15['cost'] / CAP, rel_tol=1e-12)
    # at 2x the same year's commissions bleed equity below N/2 by autumn, the
    # cap then shrinks the book, and the smaller book is charged less than
    # the first draft's full-book price every night
    s = lv.fan(np.zeros(500), np.ones(500), a2, CAP, 20, ZERO)
    assert s['p50'] > -lv.H * a2['cost'] / CAP
    E = CAP
    for _ in range(lv.H):
        E -= float(lv.night_cost_at(a2, min(a2['notional'], 2 * E)))
    assert math.isclose(s['p50'], E / CAP - 1, rel_tol=1e-12)


def test_commissions_are_repriced_when_reg_t_shrinks_the_book():
    a = lv.account(NAMES, PX, CAP, 2.0, 'tiered')
    xp, fp = a['curve']
    assert xp[-1] == a['notional'] and fp[-1] == a['cost']
    assert (np.diff(xp) > 0).all()
    # each node is paper_bench's own price for a smaller sizing of the basket
    for t in (0.25, 0.5, 0.8):
        b, _ = pb.sizing(NAMES, PX, t * 2 * CAP)
        d = float(b.notional.sum())
        assert abs(lv.night_cost_at(a, d) - pb.night_cost('tiered', b)) < 0.25
    # a 10% basket loss at 2x: the next night's book is 2E, not N, and its
    # orders cost what paper_bench charges for a book that size
    lg = {}
    eq, _, _ = lv.simulate(np.array([[-0.10, 0.0]]), np.ones((1, 2)), CAP, a,
                           ZERO, log=lg)
    e1 = CAP - a['notional'] * 0.10 - a['cost']
    assert math.isclose(eq[0, 1], e1)
    c2 = float(lv.night_cost_at(a, 2 * e1))
    assert 2 * e1 < a['notional'] and c2 < a['cost']
    b, _ = pb.sizing(NAMES, PX, 2 * e1)
    assert abs(c2 - pb.night_cost('tiered', b)) < 0.25
    assert math.isclose(eq[0, 2], e1 - c2)
    assert math.isclose(lg['comm'][0], a['cost'] + c2)
    # a synthetic account without a curve keeps its fixed cost
    assert lv.night_cost_at(_acct(2 * CAP, cost=7.0), 1000.0) == 7.0


def test_idle_cash_earns_ibkrs_credit_interest():
    # BM 3.88 - 0.5 = 3.38% on the balance above $10k, times min(1, NAV/100k)
    assert math.isclose(lv.credit(30e3, 100e3), 20e3 * 0.0338)
    assert math.isclose(lv.credit(30e3, 250e3), 20e3 * 0.0338)
    assert math.isclose(lv.credit(30e3, 50e3), 20e3 * 0.0338 * 0.5)
    assert lv.credit(8e3, 100e3) == 0.0 and lv.credit(-5e3, 100e3) == 0.0
    assert math.isclose(lv.credit(30e3, 100e3, bm=5.0), 20e3 * 0.045)
    # in the account, on calendar days: $100k holding a $70k book has $30k
    # idle, and the credit compounds into the next night's idle cash
    a = dict(L=0.7, notional=70e3, dep=0.7, cost=0.0, maint=0.25, held=11,
             names=11)
    cr = lambda cash, e: lv.credit(cash, e)
    lg = {}
    eq, _, fin = lv.simulate(np.zeros((1, 2)), np.array([[1, 3]]), 100e3, a,
                             ZERO, credit=cr, log=lg)
    e1 = 100e3 + 20e3 * 0.0338 / 360
    e2 = e1 + (e1 - 80e3) * 0.0338 * 3 / 360
    assert math.isclose(eq[0, 1], e1) and math.isclose(eq[0, 2], e2)
    assert math.isclose(fin[0], -(e2 - 100e3))         # net financing < 0
    assert math.isclose(lg['credit'][0], e2 - 100e3) and lg['interest'][0] == 0
    # a levered book has no idle cash: nothing is credited
    lg = {}
    lv.simulate(np.zeros((1, 3)), np.ones((1, 3)), CAP, _acct(1.5 * CAP), ZERO,
                credit=cr, log=lg)
    assert lg['credit'][0] == 0.0
    # and the credit is opt-in: without it the roi_year parity test holds


def test_the_margin_call_fires_on_a_planted_crash_and_never_on_calm_nights():
    a = lv.account(NAMES, PX, CAP, 2.0, 'tiered')
    rate = lambda d: lv.interest(d)
    rng = np.random.default_rng(5)
    calm = rng.normal(0.0, 0.002, (300, 252))
    calm -= calm.mean(axis=1, keepdims=True)
    gaps = np.ones_like(calm)
    eq, called, _ = lv.simulate(calm, gaps, CAP, a, rate)
    assert not called.any()
    crash = calm.copy()
    crash[:, 100] = -0.45
    eq, called, _ = lv.simulate(crash, gaps, CAP, a, rate)
    assert called.all()
    # liquidated at that open and out for the rest of the year
    assert np.allclose(eq[:, 102:], eq[:, [101]])
    assert lv.summary(eq, called, np.zeros(300), CAP)['mcall'] == 1.0


def test_the_maintenance_line_at_exactly_two_x_is_a_one_third_drop():
    a = _acct(2 * CAP)
    _, called, _ = lv.simulate(np.array([[-0.33]]), np.ones((1, 1)), CAP, a,
                               ZERO)
    assert not called[0]
    eq, called, _ = lv.simulate(np.array([[-0.34]]), np.ones((1, 1)), CAP, a,
                                ZERO)
    assert called[0]
    e_open = CAP - 2 * CAP * 0.34
    assert math.isclose(eq[0, 1], e_open - lv.LIQ_SLIP / 1e4 * 2 * CAP * 0.66)
    # fewer names held, higher house rate
    assert lv.house_maint(11) == 0.25 and lv.house_maint(8) == 0.281
    assert lv.house_maint(6) == 0.451 and lv.house_maint(3) == 0.753
    assert lv.house_maint(1) == 1.0


def test_reg_t_cuts_exposure_to_two_x_equity_after_a_loss():
    a = _acct(2 * CAP)
    eq, _, _ = lv.simulate(np.array([[-0.10, 0.10]]), np.ones((1, 2)), CAP, a,
                           ZERO)
    assert math.isclose(eq[0, 1], 0.8 * CAP)
    assert math.isclose(eq[0, 2], 0.8 * CAP + 1.6 * CAP * 0.10)   # not 2C
    # and under $2,000 no position opens
    eq, _, _ = lv.simulate(np.array([[-0.48, 0.5]]), np.ones((1, 2)), CAP,
                           _acct(2 * CAP, maint=0.0), ZERO)
    assert eq[0, 1] < lv.MIN_EQUITY and eq[0, 2] == eq[0, 1]


def test_day_gaps_come_from_the_calendar():
    # open-day dating: Friday 2026-09-18 -> Monday 09-21 is 3 days;
    # Labor Day: Friday 09-04 -> Tuesday 09-08 is 4
    g = lv.day_gaps(['20260917', '20260918', '20260921', '20260922'])
    assert list(g) == [1, 1, 3, 1]
    g = lv.day_gaps(['20260904', '20260908'], prev='20260903')
    assert list(g) == [1, 4]
    assert lv.day_gaps(['20260901', '20261101'], prev='20260831')[1] == \
        lv.MAX_GAP                               # a data hole is capped
    # buy-day dating (the live log): Friday's buy runs to Monday
    assert list(lv.buy_day_gaps(['20261008', '20261009'])) == [1, 3]
    assert list(lv.buy_day_gaps(['20260904', '20260908'])) == [4, 1]


def test_a_fixed_notional_book_adds_and_is_roi_years_fan_at_1x():
    a = lv.account(NAMES, PX, CAP, 1.0, 'tiered')
    s = lv.fan(np.full(500, 10.0), np.ones(500), a, CAP, 50, ZERO)
    per = a['notional'] * math.expm1(10 / 1e4) - a['cost']
    assert math.isclose(s['p50'], lv.H * per / CAP, rel_tol=1e-12)
    assert math.isclose(s['night'], per / CAP * 1e4, rel_tol=1e-12)
    # the same draws as roi_year.fan, so the 1x rows line up with RESULTS 26;
    # roi_year's 'night' is its closed form, which is leverage's 'night0'
    r = np.random.default_rng(6).normal(8.0, 40.0, 4000)
    mine = lv.fan(r, np.ones(4000), a, CAP, 500, ZERO)
    theirs = ry.fan(r, a['dep'], a['cost_bps'], 500,
                    np.random.default_rng(lv.SEED))
    for q in ('p5', 'p50', 'p95', 'loss', 'dd5'):
        assert math.isclose(mine[q], theirs[q], rel_tol=1e-9, abs_tol=1e-12), q
    assert math.isclose(mine['night0'], theirs['night'], rel_tol=1e-9)


def test_enet_is_the_fans_own_mean_and_the_closed_form_overstates_capped_rows():
    # no cut-back and no financing: P&L is linear, so the fan's mean night is
    # exactly N x (mean drawn return) - cost, and the closed form differs
    # from it only by the bootstrap's sampling noise
    a = lv.account(NAMES, PX, CAP, 1.5, 'tiered')
    r = np.random.default_rng(9).normal(8.0, 60.0, 4000)
    g = np.ones(4000)
    s = lv.fan(r, g, a, CAP, 400, ZERO)
    R, _, _ = lv.draw(r, g, 400)
    eq, _, _ = lv.simulate(R, np.ones_like(R), CAP, a, ZERO)
    assert eq.min() > a['notional'] / 2                # the cap never bound
    exact = (a['notional'] * R.mean() - a['cost']) / CAP * 1e4
    assert math.isclose(s['night'], exact, rel_tol=1e-9)
    assert abs(s['night'] - s['night0']) < 1.0
    # with a positive drift and a volatility where losses cut the 2x book back
    # (the live stress's ~2x the backtest's), the fan earns ~1 bp/night less
    # than the same closed form evaluated on the SAME draws (so no sampling
    # noise is in the gap); at the backtest's volatility, and at 1x, the gap
    # is ~0 -- the pattern the review measured on the proxy (1.3 vs 0.2-0.4)
    z = np.random.default_rng(10).standard_normal(4000)
    gap = {}
    for sd in (120.0, 250.0):
        r = 8.0 + sd * (z - z.mean()) / z.std()
        R, _, _ = lv.draw(r, g, 400)
        for L in (1.0, 2.0):
            a = lv.account(NAMES, PX, CAP, L, 'tiered')
            s = lv.fan(r, g, a, CAP, 400, lv.interest)
            fin0 = float(lv.interest(a['notional'] - CAP)) / lv.DAYCOUNT
            lin = (a['notional'] * R.mean() - a['cost'] - fin0) / CAP * 1e4
            gap[(sd, L)] = lin - s['night']
    assert gap[(250.0, 2.0)] > 0.8
    assert abs(gap[(250.0, 1.0)]) < 0.2 and abs(gap[(120.0, 1.0)]) < 0.2
    assert gap[(120.0, 2.0)] < 0.4


def _bars(seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2002-12-02', '2026-03-31')
    n = len(days)
    out = {}
    on_q = rng.normal(4.0, 75.0, n)
    for s, on in (('QQQ', on_q), ('SPY', 0.85 * on_q + rng.normal(-1, 30, n))):
        idv = rng.normal(0.0, 90.0, n)
        lc = np.log(100.0) + np.cumsum(on + idv) / 1e4
        lo = np.concatenate([[np.log(100.0)], lc[:-1]]) + on / 1e4
        out[s] = pd.DataFrame({'Open': np.exp(lo), 'Close': np.exp(lc)},
                              index=days)
    return out


def test_the_proxy_is_the_stated_factor_model():
    q5, F, gaps = lv.proxy_inputs(_bars(), '202603')
    assert q5.index[0] == '20030102' and q5.index[-1] == '20260331'
    assert math.isclose(q5.mean(), lv.PROXY['gross'], abs_tol=1e-9)
    alpha, cols, reg = lv.selection_alpha(q5, F)
    assert abs(reg['betas']['mkt_on'] - 0.69) < 0.05
    assert abs(reg['betas']['spy_on'] - 0.73) < 0.05
    X = np.column_stack([np.ones(len(F)), F.mkt_on, F.spy_on])
    resid = q5.values - X @ np.linalg.lstsq(X, q5.values, rcond=None)[0]
    assert abs(resid.std() - 4.13 * np.sqrt(252) / 1.26) < 1.0
    assert gaps.loc['20030106'] == 3            # Friday -> Monday
    assert gaps.index.equals(q5.index)


def test_the_proxys_recent_era_is_pinned_to_half_the_edge():
    # a constant alpha made the proxy's 2016+ world the best one, the
    # opposite of RESULTS 26's real 2016+ row (a median tie with half the
    # edge); the alpha steps at 2016 so the two worlds share a mean, with the
    # overall gross unchanged
    q5, F, _ = lv.proxy_inputs(_bars(), '202603')
    alpha, _, _ = lv.selection_alpha(q5, F)
    ws = ry.worlds(q5, alpha)
    assert math.isclose(ws[lv.RECENT].mean(), ws['half the edge'].mean(),
                        abs_tol=1e-6)
    assert ws[lv.RECENT].mean() < ws['as backtested'].mean()
    assert math.isclose(q5.mean(), lv.PROXY['gross'], abs_tol=1e-9)
    post = q5.index >= ry.RECENT
    assert q5[post].mean() < q5[~post].mean()


def _live(n=34, seed=8):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range('2026-08-21', periods=n).strftime('%Y%m%d')
    return pd.DataFrame({'date': d, 'q5_on_bps': rng.normal(0, 200, n),
                         'qqq_on_bps': rng.normal(0, 70, n)})


def test_every_table_in_proxy_mode_carries_the_banner(capsys):
    q5, F, gaps = lv.proxy_inputs(_bars(), '202603')
    out = lv.run(q5, F, '202603', NAMES, PX, _live(), CAP, 'tiered', 300,
                 proxy=True, gaps=gaps, kelly_paths=100)
    lines = capsys.readouterr().out.splitlines()
    titles = [i for i, l in enumerate(lines) if l.startswith('== ')]
    assert len(titles) >= 8
    for i in titles:
        assert 'PROXY' in lines[i - 1], lines[i]
    assert lines[0] == lv.BANNER
    assert out['passing'] == [] or all(L > 1 for L in out['passing'])
    for w in ry.worlds(q5, 6.0):
        assert set(out['tabs'][0][w]) == set(lv.GRID)
    # every line that names the recent era carries the proxy's tag, and the
    # note that explains it follows each table that shows it
    rec = [l for l in lines if 'recent era' in l]
    assert rec and all('recent era (2016+)' + lv.PROXY_TAG in l for l in rec)
    note = lv.PROXY_RECENT_NOTE.splitlines()[0]
    assert sum(l == note for l in lines) >= 4
    # Kelly: an f* over the Reg T ceiling is flagged, never reported as
    # attainable, and the feasible one never exceeds it
    for (w, v), r in out['kelly'].items():
        assert r['reach'] == (r['x'] <= lv.REGT + 1e-12), (w, v)
        assert r['x_regt'] <= lv.REGT + 1e-12, (w, v)
    assert any(not r['reach'] for r in out['kelly'].values())
    # the replay's commissions are what the account paid
    a1, a2 = out['accts'][0], out['accts'][-1]
    n = len(_live())
    assert math.isclose(out['replay']['book 1.00x']['comm'], a1['cost'] * n)
    assert out['replay']['book 2.00x']['comm'] <= a2['cost'] * n + 1e-9
    # and a real run says nothing of the kind
    lv.run(q5, F, '202603', NAMES, PX, _live(), CAP, 'tiered', 100,
           proxy=False, gaps=gaps, kelly_paths=50)
    real = capsys.readouterr().out
    assert 'PROXY' not in real
    assert 'recent era (2016+)' + lv.PROXY_TAG not in real


def test_the_decision_rule_is_the_one_declared():
    S = lambda p50, p5, mc=0.0: dict(p50=p50, p5=p5, mcall=mc)
    live = {'half the edge': {1.0: S(.10, -.20), 1.25: S(.12, -.19),
                              1.5: S(.13, -.25), 2.0: S(.09, -.18)},
            'as backtested': {L: S(.2, -.1) for L in lv.GRID}}
    ok, why = lv.decide(live, [live])
    assert ok == [1.25]                     # raises the median, keeps p5
    assert 'lowered' in why[1.5][0]         # median up, p5 down: fails
    assert 'not raised' in why[2.0][0]      # p5 up, median down: fails
    # an equal p5 does not lower it; an equal median does not raise it
    live['half the edge'][1.5] = S(.13, -.20)
    live['half the edge'][2.0] = S(.10, -.10)
    ok, why = lv.decide(live, [live])
    assert ok == [1.25, 1.5]
    # a margin call in ANY world, at either volatility, vetoes
    bt = {'no edge (beta only)': {L: S(0, 0, 0.02 if L == 1.25 else 0.0)
                                  for L in lv.GRID}}
    ok, why = lv.decide(live, [bt, live])
    assert ok == [1.5] and 'margin call' in why[1.25][0]
    bt['no edge (beta only)'][1.5] = S(0, 0, 0.0099)
    assert lv.decide(live, [bt, live])[0] == [1.5]
