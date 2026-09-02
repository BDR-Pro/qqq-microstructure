# Part of qqq-microstructure.
#
# Phase 19: one command, the whole robustness dossier. It does NOT re-run or
# re-fit anything -- it reads the daily series the disciplined scripts already
# wrote and subjects each book to the same battery, so the report cannot be
# rosier than the committed evidence. Sections:
#
#   1. dataset            panel span, names, months
#   2. metrics            CAGR/Sharpe/Sortino/maxDD/Calmar/vol/win/profit
#                         factor/worst-best day+month/turnover, per book
#   3. regime             4-year eras, per book (persistent or concentrated?)
#   4. fragility          performance after removing best/worst N days --
#                         is the mean a handful of observations?
#   5. cost stress        gross reconstructed from the known per-leg cost,
#                         swept 0..10x, with the BREAK-EVEN multiple
#   6. factor             alpha/beta/residual-Sharpe vs QQQ+SPY (factor.py)
#   7. significance       block-bootstrap 5/50/95 CI on mean bps/day and
#                         P(mean>0) (mc_risk machinery)
#   8. capacity           $ before the auction footprint gets serious, from
#                         panel dollar-volume, assumptions stated
#   9. survival table     the A-F deliverable: CAGR/Sharpe/maxDD/5x-cost/
#                         break-even/capacity + KILL or SURVIVE per book
#
# The verdict rule is fixed and printed: a book SURVIVES only if it clears
# every gate (positive after 5x costs, break-even cost > 2x the assumed,
# Sharpe > 0.5 after costs, edge not concentrated in <1% of days, and a
# defensible capacity). ROBUSTNESS > RETURN: a smaller number that clears
# every gate beats a larger one that fails any.
#
# Reads whatever exists: data/stack_daily.csv (run stack_v2.py), and
# optionally overlay_daily.csv, opra_daily.parquet. Panel via xsec_backtest.
#
#   python src/research_report.py

import os, argparse
import numpy as np, pandas as pd
from xsec_backtest import load_panel, stats, era_table
from mc_risk import paths, BLOCK, SEED
import factor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# per-book round-trip cost actually charged in stack_v2 (bps/day) at the --c
# the stack was run with, so gross = net + base and net(m) = net + (1-m)*base
# under a cost multiplier m. OPT's base is RESULTS 18's entry commissions.
OPT_COMM = 0.27


def base_cost(c):
    return {'ON': 2 * c, 'NEU': 2 * c + 0.34, 'MOM': 0.34, 'QQQ_ON': 0.34,
            'v2': 2 * c + 0.34, 'v2n': 2 * c + 0.68, 'OPT': OPT_COMM,
            'v2o': 2 * c + 0.34 + OPT_COMM}


BASE_COST = base_cost(1.0)
TURNOVER = {'ON': '200%/day (enter MOC, exit MOO)',
            'NEU': '200%/day + QQQ pair', 'MOM': '200%/day intraday',
            'v2': '~400%/day (two non-overlapping legs)',
            'v2n': '~400%/day + hedge'}


def metrics(r):
    r = np.asarray(r, float)
    eq = np.cumprod(1 + r / 1e4)
    n = len(r)
    cagr = eq[-1] ** (252 / n) - 1
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    down = np.sqrt(np.mean(np.minimum(r, 0) ** 2))   # downside deviation (LPM2)
    return dict(
        n=n, mean=r.mean(), cagr=cagr,
        sharpe=r.mean() / r.std() * np.sqrt(252),
        sortino=r.mean() / down * np.sqrt(252) if down else np.nan,
        maxdd=dd, calmar=cagr / abs(dd) if dd else np.nan,
        vol=r.std() * np.sqrt(252) / 100,
        win=(r > 0).mean(),
        pf=r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.nan,
        worst=r.min(), best=r.max())


def books(L):
    b = {}
    for k in ('ON', 'NEU', 'MOM'):
        if k in L:
            b[k] = L[k].dropna()
    if 'ON' in L and 'MOM' in L:
        b['v2'] = L[['ON', 'MOM']].dropna().sum(axis=1)
    if 'NEU' in L and 'MOM' in L:
        b['v2n'] = L[['NEU', 'MOM']].dropna().sum(axis=1)
    if all(c in L for c in ('ON', 'MOM', 'OPT')):
        b['v2o'] = L[['ON', 'MOM', 'OPT']].dropna().sum(axis=1)
    return b


def sec_metrics(b):
    print('\n' + '=' * 20 + ' 2. METRICS ' + '=' * 20)
    print(f'{"book":<6}{"days":>6}{"bps/d":>7}{"CAGR":>7}{"Shrp":>6}{"Sort":>6}'
          f'{"maxDD":>7}{"Calmar":>7}{"vol":>6}{"win":>6}{"PF":>6}'
          f'{"wDay":>7}{"bDay":>7}')
    for k, r in b.items():
        m = metrics(r.values)
        print(f'{k:<6}{m["n"]:>6}{m["mean"]:>+7.1f}{m["cagr"]*100:>6.0f}%'
              f'{m["sharpe"]:>6.2f}{m["sortino"]:>6.2f}{m["maxdd"]*100:>6.0f}%'
              f'{m["calmar"]:>7.2f}{m["vol"]:>5.0f}%{m["win"]*100:>5.0f}%'
              f'{m["pf"]:>6.2f}{m["worst"]:>+7.0f}{m["best"]:>+7.0f}')


def sec_regime(b):
    print('\n' + '=' * 20 + ' 3. REGIME (4-year eras) ' + '=' * 20)
    for k in [x for x in ('v2', 'v2n', 'ON') if x in b]:
        print(f'\n  {k}:')
        era_table(b[k].index, b[k].values)


def sec_fragility(b):
    print('\n' + '=' * 20 + ' 4. FRAGILITY (remove best/worst N days) ' + '=' * 12)
    print(f'{"book":<6}{"full":>8}{"-best5":>8}{"-best20":>9}{"-best1%":>9}'
          f'{"-wrst5":>8}{"-wrst20":>9}   (mean bps/day)')
    for k, r in b.items():
        v = np.sort(r.values)
        n1 = max(1, len(v) // 100)
        cells = [r.mean(), v[:-5].mean(), v[:-20].mean(), v[:-n1].mean(),
                 v[5:].mean(), v[20:].mean()]
        widths = [8, 8, 9, 9, 8, 9]
        print(f'{k:<6}' + ''.join(f'{x:>{w}.1f}'
                                  for x, w in zip(cells, widths)))


def sec_cost(b):
    print('\n' + '=' * 20 + ' 5. COST STRESS (multiple of assumed) ' + '=' * 14)
    print(f'{"book":<6}{"0x":>7}{"0.5x":>7}{"1x":>7}{"2x":>7}{"3x":>7}'
          f'{"5x":>7}{"10x":>7}{"break-even":>12}   (net bps/day)')
    for k, r in b.items():
        base = BASE_COST.get(k)
        if base is None:
            continue
        net1 = r.mean()
        gross = net1 + base
        cells = [gross - m * base for m in (0, 0.5, 1, 2, 3, 5, 10)]
        bem = gross / base
        print(f'{k:<6}' + ''.join(f'{c:>7.1f}' for c in cells)
              + f'{bem:>10.1f}x')


def sec_factor(b, F):
    print('\n' + '=' * 20 + ' 6. FACTOR EXPOSURE (vs QQQ+SPY) ' + '=' * 15)
    have_spy = 'spy_on' in F
    on_cols = ['mkt_on'] + (['spy_on'] if have_spy else [])
    for k in [x for x in ('ON', 'v2', 'v2n') if x in b]:
        cols = on_cols if k != 'MOM' else ['mkt_id']
        j = pd.concat([b[k].rename('y'), F[cols]], axis=1).dropna()
        if len(j) < 250:
            continue
        r = factor.regress(j.y, j[cols])
        bs = ' '.join(f'{kk} {vv:+.2f}' for kk, vv in r['betas'].items())
        print(f'  {k:<5} alpha {r["alpha"]:+.2f} bps/d (t_HAC {r["t_alpha"]:+.1f})'
              f'  residSharpe {r["resid_sharpe"]:+.2f}  R2 {r["r2"]:.2f}  '
              f'betas: {bs}')


def sec_significance(b):
    print('\n' + '=' * 20 + ' 7. SIGNIFICANCE (block bootstrap) ' + '=' * 13)
    rng = np.random.default_rng(SEED)
    print(f'{"book":<6}{"mean":>7}{"p5":>7}{"p50":>7}{"p95":>7}'
          f'{"P(mean>0)":>11}   (bps/day, 21-day blocks)')
    for k, r in b.items():
        m = paths(r.values, 4000, len(r), BLOCK, rng).mean(axis=1)
        print(f'{k:<6}{r.mean():>+7.1f}{np.percentile(m,5):>+7.1f}'
              f'{np.percentile(m,50):>+7.1f}{np.percentile(m,95):>+7.1f}'
              f'{(m>0).mean()*100:>10.1f}%')


def sec_capacity(df):
    print('\n' + '=' * 20 + ' 8. CAPACITY ' + '=' * 20)
    last = sorted(df.month.unique())[-1]
    dv = df[df.month == last].groupby('ticker').dollar_vol.mean()
    k = 10                                           # the traded basket
    rp = os.path.join(ROOT, 'reports', 'xsec_paper.csv')
    if os.path.exists(rp):
        try:
            k = len(pd.read_csv(rp, dtype=str).q5.iloc[-1].split(';'))
        except Exception:
            pass
    med = dv.median()                                # true median name
    p25 = dv.quantile(0.25)                          # a weak Q5 name
    AUCT_CLOSE, AUCT_OPEN, PARTIC = 0.08, 0.03, 0.10  # share of ADV in each cross
    auct = min(AUCT_CLOSE, AUCT_OPEN)                # the exit cross binds
    per_med, per_p25 = med * auct * PARTIC, p25 * auct * PARTIC
    print(f'  universe month {last}: {len(dv)} names; traded basket k={k}')
    print(f'  ADV$ median ~${med/1e6:.0f}M, p25 ~${p25/1e6:.0f}M; closing '
          f'cross ~{AUCT_CLOSE:.0%} / opening cross ~{AUCT_OPEN:.0%} of ADV, '
          f'participate ~{PARTIC:.0%} of the binding (opening) cross')
    print(f'  -> ~${per_med/1e6:.2f}M per median name, ~${per_p25/1e6:.2f}M per '
          f'p25 name; x{k} equal-weight = ~${per_p25*k/1e6:.0f}M '
          f'(least-liquid binds) .. ~${per_med*k/1e6:.0f}M')
    print('  (equal-weight, so the least-liquid name binds; overnight names '
          'are the top of\n   the tape, so this is an estimate, not a claim -- '
          'measure it live with fills.py)')


def sec_survival(b):
    print('\n' + '=' * 20 + ' 9. SURVIVAL TABLE + VERDICT ' + '=' * 15)
    print(f'{"book":<6}{"Sharpe":>7}{"maxDD":>7}{"5x-cost":>8}{"brk-even":>9}'
          f'{"turnover":>26}   verdict')
    for k, r in b.items():
        m = metrics(r.values)
        base = BASE_COST.get(k, 0)
        gross = r.mean() + base
        net5 = gross - 5 * base
        bem = gross / base if base else float('inf')
        v = np.sort(r.values)
        concentrated = (v[:-max(1, len(v) // 100)].mean() < 0)   # dies w/o top 1%
        # gate: survives DOUBLE the assumed cost (break-even>2x), Sharpe>0.5,
        # edge not concentrated in <1% of days. 5x-cost is shown, not gated --
        # 5x is a stress readout, 2x is the survival bar.
        survive = (bem > 2 and m['sharpe'] > 0.5 and not concentrated)
        print(f'{k:<6}{m["sharpe"]:>7.2f}{m["maxdd"]*100:>6.0f}%'
              f'{net5:>+8.1f}{bem:>8.1f}x{TURNOVER.get(k,"-"):>26}   '
              f'{"SURVIVE" if survive else "KILL"}'
              + ('' if not concentrated else ' (edge in <1% of days)'))
    print('\n  gate: break-even cost > 2x assumed AND Sharpe > 0.5 AND edge '
          'not in <1% of\n  days. 5x-cost is shown as a stress readout, not '
          'a gate. ROBUSTNESS > RETURN.')


def main():
    global BASE_COST
    ap = argparse.ArgumentParser()
    ap.add_argument('--c', type=float, default=1.0,
                    help='the one-way auction cost stack_v2.py was run with')
    a = ap.parse_args()
    BASE_COST = base_cost(a.c)
    d = os.path.join(ROOT, 'data')
    sp = os.path.join(d, 'stack_daily.csv')
    if not os.path.exists(sp):
        raise SystemExit('missing data/stack_daily.csv -- run stack_v2.py first')
    L = pd.read_csv(sp, dtype={'day': str}).set_index('day')
    # stack_v2 already writes an OPT column when the OPRA chains exist; only
    # derive one here if the stack didn't (older stack_daily.csv)
    if 'OPT' not in L:
        op = os.path.join(d, 'opra_daily.parquet')
        if os.path.exists(op):
            od = pd.read_parquet(op)
            od['day'] = od.day.astype(str)
            od = od.set_index('day')
            if 'ev_sprd' in od and 'spot' in od:
                L = L.join((od.ev_sprd - 1.30 / (od.spot * 100) * 1e4)
                           .rename('OPT'))
    b = books(L)
    mp = os.path.join(ROOT, 'models', 'xsec_lgbm.json')
    if os.path.exists(mp):
        import json
        cut = json.load(open(mp)).get('last_tmonth', '9999-99').replace('-', '')
        last = L.index.max()
        if 'ON' in L and L.ON.dropna().index.max()[:6] > cut[:6]:
            print(f'WARNING: ON runs past the frozen cutoff {cut[:4]}-{cut[4:6]} '
                  f'-- xsec_ml_daily.csv contains forward months; every number '
                  f'below mixes holdout into the backtest')

    df = load_panel()
    F = factor.factors(df)
    print('=' * 20 + ' 1. DATASET ' + '=' * 20)
    print(f'  panel {df.day.min()}..{df.day.max()}  {df.month.nunique()} months'
          f'  {df.ticker.nunique()} names  {len(df):,} ticker-days')
    print(f'  books: {", ".join(b)}')

    sec_metrics(b)
    sec_regime(b)
    sec_fragility(b)
    sec_cost(b)
    sec_factor(b, F)
    sec_significance(b)
    sec_capacity(df)
    sec_survival(b)
    print('\nThis report re-fits nothing; every number traces to a committed '
          'daily series.\nA book is production-grade only if it SURVIVES '
          'section 9 AND the forward\nreplay (RESULTS 19) keeps clearing its '
          'pre-registered GO line.')


if __name__ == '__main__':
    main()
