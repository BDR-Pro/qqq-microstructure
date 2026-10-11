# Part of qqq-microstructure.
#
# Benchmark the strategy against the paper account: is the money the paper
# account makes the money the graded record says the strategy makes -- and if
# not, exactly where does it go?
#
# The chain, from what the backtest promises to what an account keeps:
#
#   backtest      RESULTS 15/19: the Q5 tilt, +6.4 bps/night gross over QQQ
#      |          (~+4.9 after the censoring correction in RESULTS 15)
#   graded        reports/xsec_paper_daily.csv: the emitted basket at the
#      |          official auction prints, equal weight, friction-free
#   realized      reports/fills.csv: what the paper account actually filled
#      |
#   net           realized minus commissions and fees
#
# graded vs backtest is the forward test (RESULTS 19) and is not re-judged
# here. This script owns the two links below it, and splits realized - graded
# into terms that add up EXACTLY, night by night:
#
#   slippage    sum_i w_i (r_i fill - r_i official), reported per crossing:
#               buy (fill vs D's official close) and sell (fill vs D+1's
#               official open), in bps -- the auction-realism number
#   weighting   integer shares make the weights unequal: what the filled
#               names earned at official prints under the account's weights,
#               minus the same names equal-weighted
#   coverage    names the account never held (price above the slice, a
#               rejected order): filled names minus the whole basket
#   data        the basket recomputed at official prints minus the logged
#               grade (a data revision between grading and now)
#
# What IBKR paper can and cannot tell you (IBKR's paper-trading guide and API
# docs, read 2026-10-11): paper trades never reach an exchange; fills are
# simulated from the top of the book, not from the opening or closing
# auction; partial market fills have their remainder rejected; splits and
# dividends are not processed; whether paper commission reports follow the
# account's Fixed/Tiered plan is undocumented. So paper MEASURES whether the
# pipe places, fills and liquidates every night, and the integer-share
# effects. It does NOT measure auction slippage -- a live MOC/MOO execution
# prints at the official cross by construction -- which is why the report
# prints a LIVE-EQUIVALENT net that leaves the simulator's slippage out. Live
# also receives dividends the graded record omits, so it is conservative.
#
# And before a single order exists, the PRE-TRADE benchmark: at this capital
# and this commission plan, what do minimums, integer shares and fees cost
# per night, against the edge they are paid out of?
#
#   python src/paper_bench.py --capital 25000                 # both plans
#   python src/paper_bench.py --capital 25000 --plan tiered
#   python src/paper_bench.py --prices prices.csv             # offline: SYM,price
#
# Fee constants are IBKR's published schedules as researched and verified on
# 2026-10-11 (see FEES and REG); change them there, not inline.

import os, sys, math, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fills import pair_legs
from orders import size

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, 'reports')
PAPER = os.path.join(REPORTS, 'xsec_paper.csv')
DAILY = os.path.join(REPORTS, 'xsec_paper_daily.csv')
FILLS = os.path.join(REPORTS, 'fills.csv')
OLOG = os.path.join(REPORTS, 'orders_log.csv')

# What the backtest says the costs are paid out of, and what it assumed they
# were. tilt: RESULTS 19's paper-log metric; tilt_cens: less RESULTS 15's
# measured censoring bias; cost: research_report.base_cost('ON') at c = 1 bps
# per crossing, i.e. one MOC buy plus one MOO sell per night.
REF = dict(tilt=6.4, tilt_cens=4.9, cost=2.0)

# Per-order costs, IBKR Pro, as published on 2026-10-11 (sources below; a
# verifier re-read each one). Fixed bundles exchange and clearing fees and
# passes through only regulatory fees; Tiered passes through everything:
#   commission     Fixed $0.005/sh, Tiered $0.0035/sh (<=300k sh/month);
#                  min $1.00 / $0.35 per ORDER; max 1% of trade value
#                  interactivebrokers.com/en/pricing/commissions-stocks.php
#   exchange       Tiered only, per share, by auction and listing venue:
#                  Nasdaq MOC $0.0016 / MOO $0.0015, NYSE MOC and MOO $0.0010
#                  (IBKR INETstkfee.php, NYSEstkfee.php). The basket is mostly
#                  Nasdaq-listed, so the Nasdaq rates are used: conservative.
#   clearing       Tiered only, $0.0002/sh (NSCC/DTC)
#   pass-through   Tiered only, NYSE 0.000175 x commission + FINRA 0.00056 x
#                  commission
#   regulatory     both plans: SEC Section 31 $20.60 per $1M of SALES (SEC
#                  advisory, effective 2026-04-04); FINRA TAF on SALES $0.00
#                  for 2026-10-01..12-31 (SR-FINRA-2026-021), then $0.000195/sh
#                  capped at $9.79; CAT $0.000003/sh on both sides.
# SMART-routed API orders (what orders.py sends) may use Tiered; orders
# directed to one exchange may not. IBKR Lite is commission-free but has no
# TWS API, so it cannot run this pipe.
FEES = {
    'fixed':  dict(per_share=0.005, min=1.00, max_pct=0.01,
                   exch={'BUY': 0.0, 'SELL': 0.0}, clearing=0.0, passthru=0.0),
    'tiered': dict(per_share=0.0035, min=0.35, max_pct=0.01,
                   exch={'BUY': 0.0016, 'SELL': 0.0015}, clearing=0.0002,
                   passthru=0.000175 + 0.00056),
}
REG = dict(sec_rate=20.60e-6, cat=0.000003, taf=0.000195, taf_cap=9.79,
           taf_holiday=('20261001', '20261231'))


# a fill whose overnight return differs from the official one by more than
# this is not slippage (a real auction fill IS the official print): it is a
# split or dividend the paper engine did not process, or a simulator outlier
OUTLIER = 1000.0


def order_cost(plan, shares, price, side, day=None):
    """Dollars for one order of `shares` at `price`. A zero-share order is
    never sent and costs nothing. day (YYYYMMDD) only matters for the FINRA
    TAF holiday; default today."""
    if shares <= 0:
        return 0.0
    import datetime as dt
    day = day or dt.date.today().strftime('%Y%m%d')
    p = FEES[plan]
    value = shares * price
    comm = min(max(p['min'], p['per_share'] * shares), p['max_pct'] * value)
    c = comm * (1 + p['passthru'])
    c += (p['exch'][side] + p['clearing'] + REG['cat']) * shares
    if side == 'SELL':
        c += REG['sec_rate'] * value
        lo, hi = REG['taf_holiday']
        if not lo <= day <= hi:
            c += min(REG['taf'] * shares, REG['taf_cap'])
    return c


def sizing(names, px, capital):
    """orders.py's own sizing, called rather than copied, so the pre-trade
    benchmark prices exactly the orders the pipe would send: an equal dollar
    slice per basket name, whole shares, an unpriced or too-expensive name's
    slice left in cash."""
    rows, per = size(names, px, capital)
    return pd.DataFrame([dict(sym=n, price=p, shares=q, notional=v)
                         for n, p, q, v in rows]), per


def night_cost(plan, book):
    return sum(order_cost(plan, r.shares, r.price, 'BUY') +
               order_cost(plan, r.shares, r.price, 'SELL')
               for r in book.itertuples() if r.price)


def pretrade(names, px, capital, plans, sweep=(10e3, 25e3, 50e3, 100e3,
                                                250e3, 500e3, 1e6)):
    book, per = sizing(names, px, capital)
    k = len(names)
    dep = book.notional.sum()
    w = book.notional / dep if dep else book.notional * 0
    print(f'\n== PRE-TRADE: the current basket ({k} names) at '
          f'${capital:,.0f} -> ${per:,.0f} per name ==')
    print(f'{"name":<7}{"price":>10}{"shares":>8}{"notional":>11}'
          f'{"weight":>8}{"vs 1/k":>8}' +
          ''.join(f'{p + " $/night":>17}' for p in plans))
    for r, wi in zip(book.itertuples(), w):
        if not r.price:
            print(f'{r.sym:<7}{"--":>10}   no price -- slice stays in cash')
            continue
        print(f'{r.sym:<7}{r.price:>10.2f}{r.shares:>8}{r.notional:>11,.0f}'
              f'{wi:>8.1%}{wi - 1 / k:>+8.1%}' +
              ''.join(f'{order_cost(p, r.shares, r.price, "BUY") + order_cost(p, r.shares, r.price, "SELL"):>17.2f}'
                      for p in plans))
    zero = book[(book.shares == 0) & book.price.notna()].sym.tolist()
    print(f'deployed ${dep:,.0f} = {dep / capital:.0%} of capital'
          + (f'; {len(zero)} name(s) unbought, price above the slice: '
             f'{", ".join(zero)}' if zero else '')
          + f'; max weight error {float((w - 1 / k).abs().max()):.1%}')
    for p in plans:
        c = night_cost(p, book)
        bps = c / capital * 1e4
        print(f'  {p:<7} ${c:6.2f}/night = {bps:5.2f} bps of capital '
              f'({c / dep * 1e4 if dep else float("nan"):.2f} of deployed) '
              f'-- {bps / REF["cost"]:.1f}x the backtest\'s {REF["cost"]:.0f} '
              f'bps, {bps / REF["tilt_cens"]:.0%} of the corrected tilt edge '
              f'(+{REF["tilt_cens"]} bps)')

    print(f'\n  commission drag by capital (bps of capital per night; the '
          f'backtest assumed {REF["cost"]:.0f}):')
    print(f'  {"capital":>11}{"deployed":>10}{"unbought":>10}' +
          ''.join(f'{p:>10}' for p in plans))
    for cap in sweep:
        b, _ = sizing(names, px, cap)
        d = b.notional.sum()
        print(f'  {cap:>11,.0f}{d / cap:>10.0%}'
              f'{int(((b.shares == 0) & b.price.notna()).sum()):>10}' +
              ''.join(f'{night_cost(p, b) / cap * 1e4:>10.2f}' for p in plans))
    return book


def official_prints(names, start, end):
    """{(sym, 'YYYYMMDD'): (open, close)} from Yahoo, the source the grader
    uses -- so slippage is measured against exactly what was graded."""
    import yfinance as yf
    v = yf.download([n.replace('.', '-') for n in names], start=start,
                    end=end, interval='1d', auto_adjust=False,
                    actions=False, group_by='ticker', progress=False)
    out = {}
    for n in names:
        try:
            s = v[n.replace('.', '-')][['Open', 'Close']].dropna()
        except KeyError:
            continue
        for d, r in zip(s.index.strftime('%Y%m%d'), s.itertuples()):
            out[(n, d)] = (float(r.Open), float(r.Close))
    return out


def decompose(legs, basket, graded, prints):
    """One night. legs: pair_legs() rows for the night; basket: the emitted
    names; graded: the logged q5_on_bps (or None); prints: official
    (open, close) by (sym, date). Returns the exact split of realized -
    graded, or a reason it cannot be split."""
    notional = sum(l['notional'] for l in legs)
    w = {l['sym']: l['notional'] / notional for l in legs}
    realized = sum(w[l['sym']] * l['bps'] for l in legs)
    comm = sum(l['comm'] for l in legs) / notional * 1e4
    out = dict(realized=realized, comm=comm, filled=len(legs),
               basket=len(basket), notional=notional, graded=graded)
    if graded is None:
        out['why'] = 'not graded yet'
        return out
    D, Dn = legs[0]['night'], legs[0]['sell_date']

    def r_off(sym):
        a, b = prints.get((sym, D)), prints.get((sym, Dn))
        return None if a is None or b is None else \
            float(np.log(b[0] / a[1]) * 1e4)

    ro = {l['sym']: r_off(l['sym']) for l in legs}
    if any(v is None for v in ro.values()):
        out['why'] = 'official prints missing for a filled name'
        return out
    # IBKR paper does not process splits or dividends: across a split the
    # account sells its pre-split share count at the post-split price, and
    # the fill return is a fiction of the simulator. A leg that disagrees with
    # the official overnight return by more than OUTLIER bps is reported, and
    # the night is kept out of every average.
    odd = {s: l['bps'] - ro[s] for s, l in ((l['sym'], l) for l in legs)
           if abs(l['bps'] - ro[l['sym']]) > OUTLIER}
    if odd:
        out['why'] = ('corporate action or simulator artefact: ' +
                      ', '.join(f'{s} fill vs official {v:+.0f} bps'
                                for s, v in odd.items()))
        return out
    rb = [x for x in (r_off(s) for s in basket) if x is not None]
    buy = {l['sym']: float(np.log(l['buy_px'] / prints[(l['sym'], D)][1]) * 1e4)
           for l in legs}
    sell = {l['sym']: float(np.log(prints[(l['sym'], Dn)][0] / l['sell_px'])
                            * 1e4) for l in legs}
    filled_eq = float(np.mean(list(ro.values())))
    out.update(
        slip_buy=sum(w[s] * buy[s] for s in w),     # + = paid above the close
        slip_sell=sum(w[s] * sell[s] for s in w),   # + = sold below the open
        slippage=sum(w[s] * (l['bps'] - ro[s]) for s, l in
                     ((l['sym'], l) for l in legs)),
        weighting=sum(w[s] * ro[s] for s in w) - filled_eq,
        coverage=filled_eq - float(np.mean(rb)),
        data=float(np.mean(rb)) - graded)
    out['gap'] = realized - graded
    # the identity this whole report rests on
    assert abs(out['slippage'] + out['weighting'] + out['coverage']
               + out['data'] - out['gap']) < 1e-6
    return out


def realized_section(fills, paper, daily, prints_fn, plans):
    legs = pair_legs(fills, quiet=True)
    if not legs:
        print('\n== REALIZED: no completed buy->sell round trips in '
              'reports/fills.csv ==')
        return None
    baskets = {r.month: r.q5.split(';') for r in paper.itertuples()}
    graded = dict(zip(daily.date, daily.q5_on_bps.astype(float)))
    by_night = {}
    for l in legs:
        by_night.setdefault(l['night'], []).append(l)
    names = sorted({n for b in baskets.values() for n in b} |
                   {l['sym'] for l in legs})
    nights = sorted(by_night)
    try:
        lo = pd.Timestamp(nights[0]) - pd.Timedelta(days=5)
        hi = pd.Timestamp(max(l['sell_date'] for l in legs)) + pd.Timedelta(days=3)
        prints = prints_fn(names, lo.strftime('%Y-%m-%d'),
                           hi.strftime('%Y-%m-%d'))
    except Exception as ex:
        print(f'  (official prints unavailable: {type(ex).__name__} -- the '
              f'split below is basket-level only)')
        prints = {}
    rows = []
    for D in nights:
        m = f'{D[:4]}-{D[4:6]}'
        rows.append(dict(night=D, **decompose(by_night[D], baskets.get(m, []),
                                              graded.get(D), prints)))
    t = pd.DataFrame(rows)
    print(f'\n== REALIZED vs GRADED: {len(t)} paper night(s) ==')
    cols = ['realized', 'graded', 'gap', 'slip_buy', 'slip_sell', 'weighting',
            'coverage', 'data', 'comm']
    show = [c for c in cols if c in t]
    print(f'{"night":>10}{"held":>6}' + ''.join(f'{c:>10}' for c in show))
    for r in t.itertuples():
        print(f'{r.night:>10}{f"{r.filled}/{r.basket}":>6}' +
              ''.join(f'{getattr(r, c):>+10.2f}' if pd.notna(getattr(r, c))
                      else f'{"--":>10}' for c in show)
              + (f'   {r.why}' if 'why' in t and isinstance(r.why, str) else ''))
    g = t.dropna(subset=['gap']) if 'gap' in t else t.iloc[0:0]
    if len(g):
        n = len(g)
        se = g.gap.std(ddof=1) / math.sqrt(n) if n > 1 else float('nan')
        print(f'\n  over {n} split night(s), bps/night (mean):')
        for c in ('gap', 'slip_buy', 'slip_sell', 'weighting', 'coverage',
                  'data', 'comm'):
            print(f'    {c:<10}{g[c].mean():>+8.2f}')
        print(f'    gap se {se:.2f}, t {g.gap.mean() / se:+.2f}' if n > 1
              else '    (one night -- no dispersion yet)')
        print(f'  net of commissions vs graded: '
              f'{(g.realized - g.comm).mean() - g.graded.mean():+.2f} bps/night;'
              f' commissions {g.comm.mean():.2f} vs the backtest\'s '
              f'{REF["cost"]:.0f}')
        # In a live auction every execution prints at the one cross price,
        # so a filled MOC/MOO order has no slippage against the official
        # print. What a live account would have kept is the graded night
        # plus what the account's own sizing did, minus what it paid. The
        # paper simulator's fills do not come from the auction, so its
        # slippage column is the simulator, not the market.
        live = (g.graded + g.weighting + g.coverage + g.data - g.comm)
        print(f'  LIVE-EQUIVALENT net (graded + weighting + coverage + data - '
              f'commissions): {live.mean():+.2f} bps/night vs graded '
              f'{g.graded.mean():+.2f}')
    return t


def panel_parity(daily, paper, df):
    """CODE vs CODE: every graded night recomputed from the backtest's own
    panel (load_panel's masked on_bps, ln(open / previous close)) for the
    same emitted basket. If the live grader and the backtest measure the
    same thing, the two series agree to data noise. A panel month built from
    Yahoo (xsec_extend, src='yahoo') shares the grader's source, so it is a
    consistency check there, not an independent one -- reported per month."""
    cal = sorted(df.day.unique())
    nxt = dict(zip(cal[:-1], cal[1:]))
    on = df.set_index(['ticker', 'day']).on_bps
    src = (df.assign(src=df['src'].fillna('hf')).groupby('month').src.first()
           if 'src' in df else pd.Series(dtype=str))
    baskets = {r.month: r.q5.split(';') for r in paper.itertuples()}
    rows = []
    for r in daily.itertuples():
        Dn = nxt.get(r.date)
        names = baskets.get(r.month)
        if Dn is None or not names or not np.isfinite(r.q5_on_masked):
            continue
        v = [on.get((n, Dn)) for n in names]
        v = [x for x in v if x is not None and np.isfinite(x)]
        if len(v) < 5:
            continue
        rows.append(dict(date=r.date, month=r.month, panel=float(np.mean(v)),
                         logged=float(r.q5_on_masked), n_panel=len(v),
                         n_logged=int(r.n)))
    t = pd.DataFrame(rows)
    print(f'\n== CODE vs CODE: graded nights recomputed from the backtest '
          f'panel ==')
    if t.empty:
        print('  no graded night is covered by the panel yet (the panel ends '
              f'{cal[-1]}; the log starts {daily.date.min()})')
        return t
    d = t.logged - t.panel
    print(f'  {len(t)} nights {t.date.min()} .. {t.date.max()}:  corr '
          f'{np.corrcoef(t.logged, t.panel)[0, 1]:+.4f}   mean(log - panel) '
          f'{d.mean():+.2f}   sd {d.std(ddof=1) if len(t) > 1 else 0:.2f}   '
          f'max |diff| {d.abs().max():.2f} bps')
    for m, g in t.groupby('month'):
        print(f'    {m}: {len(g):2d} nights  mean diff {(g.logged - g.panel).mean():+6.2f}'
              f'  names {g.n_panel.min()}-{g.n_panel.max()} panel vs '
              f'{g.n_logged.min()}-{g.n_logged.max()} log   panel source '
              f'{src.get(m, "?")}')
    worst = t.loc[d.abs().sort_values(ascending=False).index[:3]]
    for r in worst.itertuples():
        print(f'    largest: {r.date}  log {r.logged:+.1f}  panel {r.panel:+.1f}'
              f'  ({r.n_logged} vs {r.n_panel} names)')
    return t


def status(paper, daily, fills, olog):
    nights = 0 if fills is None else len({l['night'] for l in
                                          pair_legs(fills, quiet=True)})
    runs = olog if olog is not None else pd.DataFrame(columns=['mode'])
    sub = runs[runs['mode'] == 'paper'] if len(runs) else runs
    print(f'== STATUS ==\n  graded nights {len(daily)} '
          f'({daily.date.min() if len(daily) else "-"} .. '
          f'{daily.date.max() if len(daily) else "-"});  paper round trips '
          f'{nights};  orders_log runs {len(runs)} '
          f'({len(sub)} submitted to paper)')
    if nights == 0:
        print('  !! The paper account has never completed a round trip. Every '
              '"paper" number so far is\n     the graded record -- official '
              'prints, no fills, no commissions. Nothing below\n     REALIZED '
              'can be measured until the VM legs in roi.yml run against IBKR '
              'paper.')
    return nights


def main():
    ap = argparse.ArgumentParser(description='Benchmark the strategy against '
                                 'the paper account.')
    ap.add_argument('--capital', type=float, default=25000)
    ap.add_argument('--plan', choices=['fixed', 'tiered', 'both'],
                    default='both')
    ap.add_argument('--prices', default=None,
                    help='offline: a CSV of SYM,price instead of Yahoo')
    ap.add_argument('--panel', action='store_true',
                    help='also recompute every graded night from the backtest '
                         'panel (data/xsec) -- code vs code')
    a = ap.parse_args()
    plans = ['fixed', 'tiered'] if a.plan == 'both' else [a.plan]
    read = lambda p, **k: pd.read_csv(p, **k) if os.path.exists(p) else None
    paper = read(PAPER, dtype=str)
    daily = read(DAILY, dtype={'date': str, 'month': str},
                 float_precision='round_trip')
    fills = read(FILLS, dtype={'date': str})
    olog = read(OLOG, dtype=str)
    if paper is None or daily is None:
        raise SystemExit('no reports/xsec_paper*.csv -- nothing to benchmark')
    status(paper, daily, fills, olog)

    names = paper.iloc[-1].q5.split(';')
    if a.prices:
        px = dict(pd.read_csv(a.prices, header=None).values)
    else:
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
                pass
    pretrade(names, px, a.capital, plans)
    if fills is not None:
        realized_section(fills, paper, daily, official_prints, plans)
    if a.panel:
        from xsec_backtest import load_panel
        panel_parity(daily, paper, load_panel())


if __name__ == '__main__':
    main()
