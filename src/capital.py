# Part of qqq-microstructure.
#
# "What capital does the hedged book need?"
#
# The book, since the owner's decision of 2026-10-14 (orders.py's HEDGE): long
# the frozen model's monthly Q5 basket overnight (BUY MOC, SELL MOO) AND short
# QQQ overnight (SELL MOC, BUY OPG), 1:1 in dollars against the long notional
# actually bought. That is RESULTS 21's NEU: alpha +3.79 bps/night at t 5.25,
# total overnight beta 0.42 instead of the long book's 1.42. Its gross night
# is the forward test's paper-log metric, tilt (Q5 - QQQ overnight). The owner
# delegated the capital target, to be set by a rule declared before anything
# was computed.
#
# THE RULE, as the numbers research declared it on 2026-10-14 in the header
# of its ladder script, before any hedged-book cost was computed. It is
# reproduced here unchanged, and it alone decides:
#   R0 plan     IBKR Pro Tiered (RESULTS 25/26; SMART routing). Fixed is
#               printed for reference only.
#   R1 cost     the hedged book's whole nightly cost: BUY MOC + SELL MOO for
#               every basket name actually bought, plus SELL MOC (short) +
#               BUY OPG (cover) of QQQ, shares = round(deployed long notional
#               / QQQ price). Priced by paper_bench.order_cost on
#               paper_bench.sizing's book (= orders.size), the current basket,
#               latest Yahoo closes, FINRA TAF ON (the 2027 schedule: the book
#               cannot go live before a GO verdict, and the TAF holiday ends
#               2026-12-31). bps of CAPITAL per night (RESULTS 25's unit).
#   R2 edge     the corrected tilt, +4.9 bps/night gross (paper_bench.REF
#               tilt_cens; RESULTS 15/19) -- the hedged book's own
#               as-backtested gross.
#   R3 TARGET   the smallest rung of 10k/25k/50k/75k/100k/150k/250k/500k at
#               which the R1 Tiered cost <= 1.3 bps/night (RESULTS 26's level,
#               ~27% of the R2 edge) AND every basket name is bought.
#   R4 MINIMUM  the smallest rung at which the R1 Tiered cost <= 2.45
#               bps/night (half the R2 edge) AND every basket name is bought.
#               Below it the hedged book is not run live.
#   R5 robust   the TARGET must also pass R3 with an assumed QQQ borrow charge
#               of 0.30%/yr on the hedge notional, Actual/360, 1.45 calendar
#               days per night (the settled short never closes under T+1, as
#               leverage.py argues for the debit). If it fails, the next rung.
#
# Proposed amendments, NOT binding. A second pass (2026-10-14) changed R4 and
# R5 after the rule's outcome on the October basket was known, and both
# verdict numbers moved with them. A rule changed after its result is seen is
# not a rule declared before computing, so these are printed beside the
# verdict as proposals. If the owner accepts them, they first bind on the
# first basket emitted after that, whose outcome nobody has seen:
#   A1  R4 at 2.34 instead of 2.45: the cost the backtest itself assumed for
#       this object (stack_v2.py: NEU nets 2 x 1.0 bps of basket crossings
#       plus the house 0.34 QQQ round trip), RESULTS 25's criterion. On the
#       October basket it moves the MINIMUM from $50k to $75k.
#   A2  R5's input measured instead of assumed: QQQ borrow at 0.40%/yr
#       (IBKR's public feed, 2026-10-12 close; 13-week range 0.3-0.5) less
#       the interest IBKR Pro pays on short-sale proceeds (0 on the first
#       $100k, BM - 1.25% = 2.63%/yr from $100k to $1M; pricing pages, read
#       2026-10-14, not yet seen in this account). On the October basket it
#       moves the TARGET from $500k to $250k; without the proceeds interest,
#       0.40% alone leaves no rung on the declared ladder.
#   A3  a $1M rung.
#
# What the rule cannot do. The 1.3 inherited from the long-only book sits
# near the hedged book's cost floor (~1.1-1.25 bps/night once per-order
# minimums stop binding: per-share fees on low-priced names, and the SEC fee
# on two sales a night), so ~0.1 bps of carry moves the TARGET a whole rung.
# On the October basket the TARGET passes R5 by 0.02 bps, and R5's single
# input, the net financing rate on the short, puts it anywhere from $250k to
# off the ladder, so the verdict line carries that schedule and every margin.
# The MINIMUM is steadier: $50k clears 2.45 by ~0.09 bps (about $0.45 a
# night) and stayed at $50k in 150 of 150 draws of 10% per-name price noise
# (the TARGET left $500k in 51 of them). It is a threshold choice that moves
# it: under A1's 2.34, $50k misses by 0.02 and the MINIMUM is $75k.
# Re-apply the rule when each month's basket is emitted.
#
# Expected net per night, bps of capital, in the hedged book's own worlds
# (roi_year --hedge 1's, from the published numbers):
#   gross          +6.43 = NEU's +4.09 net + the 2.34 it assumed (RESULTS 16;
#                  the same series as RESULTS 19's +6.4 tilt)
#   alpha          +6.13 = NEU's +3.79 (RESULTS 21) + 2.34
#   as backtested  6.43 - 1.5 censoring = +4.93 (the QQQ leg is never
#                  censored, so all of RESULTS 15's -1.5 stays)
#   half the edge  4.93 - 0.5 x (6.13 - 1.5) = +2.61
#   no edge        6.43 - 6.13 = +0.30, the 0.42 of beta left
#   net = deployed x world - R1 cost - R5's borrow - the QQQ dividend the
#   short owes (0.414%/yr; the basket's own dividends are left out, as in
#   roi_year). All at backtest volatility. The live hedged sd is ~2.3x the
#   backtest's (152 vs ~66 bps, RESULTS 19, 27), and that is roi_year's
#   question.
#
# Reg T is a requirement printed beside the rule, not part of it: long +
# short <= 2 x equity at IBKR's end-of-day check, with no offset between a
# stock basket and QQQ. "Reg T floor" is (long + short) / 2; "hold" is
# (long + short) / 1.9, the equity orders.py --submit insists on (Reg T's line
# used to 95%, leverage.py's 1.9x). A cash account or an IRA cannot short.
#
# Known and not modelled: interest on idle cash (roi_year's convention);
# IBKR house margin above Reg T, maintenance (25% long at 9+ names plus ~30%
# short, i.e. a one-night loss of ~45% of equity at L = E); Rule 201 days, on
# which a short MOC may not fill while the long does; portfolio margin (from
# $110k NLV); taxes.
#
#   python src/capital.py                        # current basket, Yahoo closes
#   python src/capital.py --prices prices.csv    # offline: SYM,price, with QQQ
#   A2's inputs, as read in TWS: --borrow 0.40 --credit-spread 1.25 --bm 3.88

import os, re, sys, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_bench as pb
import roi_year as ry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = (10e3, 25e3, 50e3, 75e3, 100e3, 150e3, 250e3, 500e3)   # R3/R4
TARGET_BPS = 1.3                     # R3, RESULTS 26
MIN_BPS = pb.REF['tilt_cens'] / 2    # R4, half the R2 edge: 2.45
BORROW = 0.30                        # R5's assumed borrow, %/yr, no credit
TAF_DAY = '20270104'    # R1: the first session after the TAF holiday
EDGE = pb.REF['tilt_cens']
GROSS, ALPHA = 4.09 + 2.34, 3.79 + 2.34       # RESULTS 16 / 21, gross
WORLDS = {'as backtested': GROSS - ry.CENS,
          'half the edge': GROSS - ry.CENS - 0.5 * (ALPHA - ry.CENS),
          'no edge': GROSS - ALPHA}
# the proposed amendments (header): printed, decide nothing
AMEND = dict(min_bps=2.0 + 0.34,     # A1, stack_v2's NEU: 2 x c=1.0 + QQQ_RT
             borrow=ry.BORROW, credit=True,               # A2
             caps=LADDER + (1e6,))                        # A3


def long_cost(plan, book, day=None):
    """paper_bench.night_cost on a given fee day (night_cost uses today's)."""
    return sum(pb.order_cost(plan, r.shares, r.price, 'BUY', day) +
               pb.order_cost(plan, r.shares, r.price, 'SELL', day)
               for r in book.itertuples() if r.price)


def rung(names, px, capital, plan, day=TAF_DAY, borrow=BORROW, bm=ry.BM,
         spread=ry.SHORT_SPREAD, credit=False):
    """One rung of the ladder: the hedged book orders.py would send at this
    capital, its nightly cost and carry in bps of capital, and its Reg T.
    The defaults are the declared R5 (borrow only); credit=True adds IBKR's
    interest on the short-sale proceeds (A2)."""
    book, _ = pb.sizing(names, px, capital)
    h = ry.hedge_book(plan, book, px, capital, 1.0, day, borrow=borrow, bm=bm,
                      spread=spread)
    lc = long_cost(plan, book, day)
    bps = lambda usd: usd / capital * 1e4
    g = h['long'] + h['short']
    return dict(capital=capital, long=h['long'], dep=h['long'] / capital,
                unbought=int((book.shares == 0).sum()), shares=h['shares'],
                short=h['short'],
                ratio=h['short'] / h['long'] if h['long'] else np.nan,
                long_bps=bps(lc), leg_bps=bps(h['leg']),
                cost=bps(lc + h['leg']),
                fin=bps(h['borrow'] - (h['credit'] if credit else 0.0)),
                dividend=bps(h['div']), gross_x=g / capital, regt=g / ry.REGT,
                fund=g / ry.REGT_OP)


def ladder(names, px, plan, caps=LADDER, **kw):
    return pd.DataFrame([rung(names, px, c, plan, **kw) for c in caps])


def pick(t, ok):
    """The smallest rung where ok holds and every basket name is bought."""
    s = t[ok & (t.unbought == 0)]
    return None if s.empty else float(s.capital.min())


def decide(t, target_bps=TARGET_BPS, min_bps=MIN_BPS):
    """The rule on a Tiered ladder (columns capital, unbought, cost, fin):
    R3 alone, R3 with R5 (the TARGET), and R4 (the MINIMUM)."""
    return dict(r3=pick(t, t.cost <= target_bps),
                target=pick(t, (t.cost <= target_bps) &
                            (t.cost + t.fin <= target_bps)),
                minimum=pick(t, t.cost <= min_bps))


def r5_schedule(t, target_bps=TARGET_BPS):
    """R5 as a function of its one input, the net financing rate on the
    short (borrow less any proceeds interest, %/yr of the short): a list of
    (highest rate, capital), ascending. The TARGET at rate x is the capital
    of the first entry whose rate is >= x; above the last, no rung."""
    out = []
    for r in t.sort_values('capital').itertuples():
        if r.unbought or r.cost > target_bps or not r.short:
            continue
        per = r.short / r.capital * 1e4 * ry.DAYS / 360 / 100   # bps / 1%/yr
        top = (target_bps - r.cost) / per
        if not out or top > out[-1][0]:
            out.append((top, r.capital))
    return out


def net(r, gross):
    """Expected net per night, bps of capital, at a world's gross."""
    return r.dep * gross - r.cost - r.fin - r.dividend


def verdict(d, t, month, sched=None, min_bps=MIN_BPS, borrow=BORROW):
    """The one line the run ends on: the numbers, their margins, and the R5
    input that decides the TARGET."""
    at = lambda c: t[t.capital == c].iloc[0]
    if d['target'] is None:
        tgt = (f'TARGET none on the ladder (no rung clears {TARGET_BPS} '
               f'bps/night with R5\'s {borrow:.2f}%/yr borrow)')
    else:
        r = at(d['target'])
        tgt = (f'TARGET --capital ${d["target"]:,.0f} ({r.cost:.2f} '
               f'bps/night; {r.cost + r.fin:.2f} with R5\'s {borrow:.2f}%/yr '
               f'borrow, {TARGET_BPS - r.cost - r.fin:.2f} inside '
               f'{TARGET_BPS}) in a margin account holding >= '
               f'${r.fund:,.0f} equity')
    if sched:
        tgt += ('; R5\'s input decides it -- at a net financing rate on the '
                'short (IBKR\'s borrow less any short-proceeds interest, read '
                'in TWS) ' + ', '.join(f'up to {x:.2f}%/yr ${c:,.0f}'
                                       for x, c in sched)
                + ', above that no rung')
    if d['minimum'] is None:
        mn = 'MINIMUM none on the ladder'
    else:
        c = at(d['minimum']).cost
        mn = (f'MINIMUM ${d["minimum"]:,.0f} ({c:.2f} <= {min_bps:.2f}, '
              f'{min_bps - c:.2f} inside)')
    return (f'VERDICT ({month} basket, Tiered, 2027 fees; the rule as '
            f'declared before computing): {tgt}; {mn}; below the minimum, no '
            f'live hedged book.')


def scheduled_capital():
    """The --capital the crontab lines in roi.yml's comments pass today."""
    p = os.path.join(ROOT, '.github', 'workflows', 'roi.yml')
    m = re.search(r'orders\.py --capital (\d+)', open(p).read()) \
        if os.path.exists(p) else None
    return float(m.group(1)) if m else None


def show(t, title):
    print(f'\n{title}')
    print(f'{"capital":>10}{"dep":>6}{"unb":>4}{"QQQ sh":>7}{"h/L":>7}'
          f'{"long-only":>10}{"QQQ leg":>8}{"HEDGED":>8}{"%edge":>7}'
          f'{"carry":>7}{"+carry":>8}')
    for r in t.itertuples():
        print(f'{r.capital:>10,.0f}{r.dep:>6.0%}{r.unbought:>4}{r.shares:>7}'
              f'{r.ratio:>7.3f}{r.long_bps:>10.2f}{r.leg_bps:>8.2f}'
              f'{r.cost:>8.2f}{r.cost / EDGE:>7.0%}{r.fin:>+7.2f}'
              f'{r.cost + r.fin:>8.2f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='What capital does the hedged '
                                 'book need? The declared rule (header) on '
                                 'the current basket.')
    ap.add_argument('--prices', default=None,
                    help='offline: CSV of SYM,price (include QQQ) instead of '
                         'Yahoo')
    ap.add_argument('--borrow', type=float, default=ry.BORROW,
                    help='A2 input: QQQ borrow fee, %%/yr')
    ap.add_argument('--credit-spread', type=float, default=ry.SHORT_SPREAD,
                    help='A2 input: IBKR pays BM minus this on short proceeds '
                         'above $100k, %%/yr')
    ap.add_argument('--bm', type=float, default=ry.BM,
                    help='A2 input: IBKR benchmark rate, %%/yr')
    a = ap.parse_args(argv)
    reg = pd.read_csv(pb.PAPER, dtype=str).iloc[-1]
    names = reg.q5.split(';')
    px = ry.prices(names + ['QQQ'], a.prices)
    rates = dict(borrow=a.borrow, bm=a.bm, spread=a.credit_spread)

    print('WHAT CAPITAL DOES THE HEDGED BOOK NEED? Research only: no live '
          'money before a GO verdict\n(RESULTS 19). The rule was declared '
          'before any hedged cost was computed (header);\nit is applied here '
          'as written, and the later amendments decide nothing.\n')
    print(f'basket {reg.month} (pre-registered {reg.emitted_on}), '
          f'{len(names)} names; hedge: short QQQ 1:1 on the long bought')
    print('prices (' + (a.prices or 'Yahoo, latest close') + '): ' +
          ' '.join(f'{k} {px[k]:.2f}' for k in names + ['QQQ']))
    print(f'R5 input as declared: borrow {BORROW:.2f}%/yr on the short, no '
          f'proceeds interest; {ry.DAYS:.3f} calendar days a night, '
          f'Actual/360')

    t = ladder(names, px, 'tiered')
    show(t, '== TIERED (decides), 2027 fees: nightly cost, bps of capital; '
            'carry = R5\'s borrow ==')
    show(ladder(names, px, 'fixed'), '== FIXED (reference only), 2027 fees ==')

    print(f'\n== TIERED: expected net per night, bps of capital, at backtest '
          f'volatility; and Reg T ==')
    print(f'{"capital":>10}{"all-in":>8}' +
          ''.join(f'{w:>15}' for w in WORLDS) +
          f'{"long $":>11}{"short $":>11}{"gross":>7}{"Reg T floor":>13}'
          f'{"hold @1.9x":>12}')
    for r in t.itertuples():
        print(f'{r.capital:>10,.0f}{r.cost + r.fin + r.dividend:>8.2f}' +
              ''.join(f'{net(r, g):>+15.2f}' for g in WORLDS.values()) +
              f'{r.long:>11,.0f}{r.short:>11,.0f}{r.gross_x:>6.2f}x'
              f'{r.regt:>13,.0f}{r.fund:>12,.0f}')
    print('  worlds, gross bps/night on the hedged notional: ' +
          ', '.join(f'{w} {g:+.2f}' for w, g in WORLDS.items()) +
          f'\n  all-in = cost + R5\'s borrow + the QQQ dividend the short '
          f'owes ({ry.QQQ_DIV}%/yr); net = deployed x world - all-in')

    d = decide(t)
    sched = r5_schedule(t)
    at = lambda c: t[t.capital == c].iloc[0]
    below = lambda c: t[t.capital < c].iloc[-1] if (t.capital < c).any() \
        else None
    print('\nTHE RULE, APPLIED')
    if d['r3'] is None:
        print(f'  R3  no rung costs <= {TARGET_BPS} (lowest '
              f'{t.cost.min():.2f})')
    else:
        r, b = at(d['r3']), below(d['r3'])
        print(f'  R3  cost <= {TARGET_BPS}: first at ${d["r3"]:,.0f} '
              f'({r.cost:.2f}, {TARGET_BPS - r.cost:.2f} inside)' +
              (f'; ${b.capital:,.0f} costs {b.cost:.2f}' if b is not None
               else ''))
        g = None if d['target'] is None else at(d['target'])
        print(f'  R5  with {BORROW:.2f}%/yr borrow: ' +
              (f'${d["target"]:,.0f} passes ({g.cost + g.fin:.2f}, '
               f'{TARGET_BPS - g.cost - g.fin:.2f} inside)'
               if g is not None else 'no rung passes') +
              ('' if d['target'] == d['r3'] else
               f'; ${d["r3"]:,.0f} fails ({r.cost + r.fin:.2f})'))
        print('  R5  its input decides it: net financing rate on the short ' +
              ', '.join(f'<= {x:.3f}%/yr -> ${c:,.0f}' for x, c in sched) +
              ', above that none')
    if d['minimum'] is None:
        print(f'  R4  no rung costs <= {MIN_BPS:.2f}')
    else:
        r, b = at(d['minimum']), below(d['minimum'])
        print(f'  R4  cost <= {MIN_BPS:.2f}: ${d["minimum"]:,.0f} '
              f'({r.cost:.3f}, {MIN_BPS - r.cost:.3f} inside)' +
              (f'; ${b.capital:,.0f} costs {b.cost:.3f}'
               if b is not None else ''))
    if d['target'] is not None:
        r = at(d['target'])
        print(f'  Reg T at the target: long ${r.long:,.0f} + short '
              f'{int(r.shares)} QQQ ${r.short:,.0f} = {r.gross_x:.2f}x '
              f'capital; Reg T floor ${r.regt:,.0f}, hold >= ${r.fund:,.0f} '
              f'at {ry.REGT_OP}x; a margin account, not cash or IRA')

    print('\n  information, not the rule:')
    fmt = lambda c: 'none' if c is None else f'${c:,.0f}'
    caps = AMEND['caps']
    am = ladder(names, px, 'tiered', caps=caps, credit=True, **rates)
    da = decide(am, min_bps=AMEND['min_bps'])
    print(f'    proposed amendments (A1 minimum <= {AMEND["min_bps"]:.2f}; A2 '
          f'borrow {a.borrow:.2f}% less proceeds interest; A3 a $1M rung), '
          f'made after this outcome was known: TARGET {fmt(da["target"])}, '
          f'MINIMUM {fmt(da["minimum"])} -- not binding')
    for lab, kw in ((f'borrow {a.borrow:.2f}%, no proceeds interest',
                     dict(rates)),
                    ('borrow 0.50%, no proceeds interest',
                     dict(rates, borrow=0.50))):
        on = decide(ladder(names, px, 'tiered', **kw))['target']
        ext = decide(ladder(names, px, 'tiered', caps=caps, **kw))['target']
        print(f'    R5 at {lab:<34} TARGET {fmt(on)}' +
              ('' if ext == on else f' ({fmt(ext)} with A3\'s rung)'))
    today = decide(ladder(names, px, 'tiered', day=None))
    print(f'    {"TAF holiday (today)":<40} TARGET {fmt(today["target"])}, '
          f'MINIMUM {fmt(today["minimum"])}')
    cron = scheduled_capital()
    if cron:
        c = rung(names, px, cron, 'tiered')
        print(f'    roi.yml\'s scheduled --capital {cron:,.0f}: '
              f'{c["cost"]:.2f} bps/night hedged ({c["cost"] / EDGE:.0%} of '
              f'the edge), '
              + ('below the MINIMUM' if d['minimum'] is None or
                 cron < d['minimum'] else 'at or above the MINIMUM'))

    line = verdict(d, t, reg.month, sched)
    print('\n' + line)
    return dict(tiered=t, decision=d, amended=da, schedule=sched,
                verdict=line)


if __name__ == '__main__':
    main()
