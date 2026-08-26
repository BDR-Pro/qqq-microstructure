# Part of qqq-microstructure.
#
# The replication question, asked the way that can be answered TODAY without
# waiting twelve forward months: is the overnight-persistence edge a broad
# market phenomenon, or a quirk of a handful of the 150 names it was found in?
#
# The test is disjoint name-subsets. Partition the universe by a STABLE hash of
# the ticker (md5, so a name lands in the same subset every month and across
# runs), then run the persistence signal INDEPENDENTLY inside each subset --
# each slice is its own self-contained mini-universe, ranked and traded only
# against itself. A real, broad edge shows up in every disjoint slice; an edge
# that lives in five names shows up in one slice and vanishes from the others.
# This is survivorship-free (it reuses the panel's own per-month universe) and
# needs no new data.
#
# Signal (the rule, not the ML -- we are testing the PHENOMENON the model
# exploits, not the model): rank names by their trailing-12-month overnight
# mean known at T-1, hold the top quintile minus the bottom quintile through
# trade month T's overnight sessions. This is the on_12m persistence that
# RESULTS 15b found carried ~37% of the ML gain. --signal on_1m tests the
# one-month version.
#
# What it prints, per partition (a 3-way hash split and a 2-way random split):
#   - each disjoint subset's L/S bps/day, t, % of months positive, rank IC,
#   - the MINIMUM across subsets (the binding number: the weakest disjoint
#     slice is the honest floor on how broad the edge is),
#   - the full-universe L/S as the reference,
#   - a BROAD / CONCENTRATED verdict: BROAD iff every subset is the same sign
#     and the weakest clears t>1.5; CONCENTRATED otherwise.
#
# Validated on planted truth: on a panel with persistence planted in EVERY
# name, all subsets replicate positive; on one with the effect planted in only
# a single hash-subset's names, only that subset shows it and the verdict flips
# to CONCENTRATED (see RESULTS / the unit checks in the docstring run).
#
#   python src/xsec_replicate.py                 # on_12m, 3-way + 2-way splits
#   python src/xsec_replicate.py --signal on_1m
#   python src/xsec_replicate.py --k 5           # 5-way disjoint split

import os, argparse, hashlib
import numpy as np, pandas as pd
from xsec_backtest import load_panel, stats, ETF, MIN_ON_DAYS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 7


def subset_of(ticker, k):
    """Stable, process-independent partition: md5(ticker) mod k."""
    return int(hashlib.md5(ticker.encode()).hexdigest(), 16) % k


def aggregate(df):
    """per (ticker, month): overnight mean and obs count."""
    g = df.groupby(['ticker', 'month']).agg(
        on_mean=('on_bps', 'mean'), on_n=('on_bps', 'count'))
    return {(t, m): (r.on_mean, int(r.on_n)) for (t, m), r in g.iterrows()}


def signal_rows(df, signal):
    """One row per (trade month T, name): the persistence signal known at
    T-1, and the realised overnight mean during T. Universe is lagged
    (names in T-1 and T), ETF-excluded, obs-gated -- the backtest's rule."""
    agg = aggregate(df)
    months = sorted(df.month.unique())
    uni = {m: set(g) for m, g in df.groupby('month')['ticker']}
    rows = []
    for i in range(13, len(months)):
        T = months[i]
        base = (uni[months[i - 1]] & uni[T]) - ETF
        look = [months[j] for j in range(i - 12, i)]
        for tk in base:
            cur = agg.get((tk, T))
            if cur is None or cur[1] < MIN_ON_DAYS or not np.isfinite(cur[0]):
                continue
            hist = [agg.get((tk, mm)) for mm in look]
            good = [h[0] for h in hist if h and h[1] >= 8]
            if signal == 'on_12m':
                if len(good) < 8:
                    continue
                sig = float(np.mean(good))
            else:                                   # on_1m: last month only
                last = hist[-1]
                if not last or last[1] < 8:
                    continue
                sig = float(last[0])
            rows.append((T, tk, sig, float(cur[0])))
    return pd.DataFrame(rows, columns=['tmonth', 'ticker', 'sig', 'realized'])


def ls_series(rows):
    """Monthly Q5-Q1: rank by sig within the given rows each month, top vs
    bottom quintile, realised spread. Returns (bps/day series, monthly IC)."""
    out, ics = {}, {}
    for T, g in rows.groupby('tmonth'):
        n = len(g)
        if n < 20:
            continue
        q = max(4, n // 5)
        s = g.sort_values('sig')
        lo = s.realized.iloc[:q].mean()
        hi = s.realized.iloc[-q:].mean()
        out[T] = hi - lo
        if g.sig.nunique() > 2:
            ics[T] = np.corrcoef(g.sig.rank(), g.realized.rank())[0, 1]
    return pd.Series(out).sort_index(), pd.Series(ics)


def report_subset(name, rows):
    ls, ic = ls_series(rows)
    if len(ls) < 12:
        return None
    t = ls.mean() / (ls.std() / np.sqrt(len(ls)))
    print(f'  {name:<14} {ls.mean():+6.2f} bps/day  t={t:+5.2f}  '
          f'Sharpe {ls.mean()/ls.std()*np.sqrt(252):+5.2f}  '
          f'{(ls > 0).mean()*100:3.0f}% mo+  IC {ic.mean():+.3f}  '
          f'({rows.ticker.nunique()} names, {len(ls)} mo)')
    return dict(mean=ls.mean(), t=t, n_names=rows.ticker.nunique())


def partition_report(title, rows, groups):
    print(f'\n{title}')
    res = []
    for gi in sorted(set(groups.values())):
        names = {tk for tk, g in groups.items() if g == gi}
        r = report_subset(f'subset {gi}', rows[rows.ticker.isin(names)])
        if r:
            res.append(r)
    if len(res) < 2:
        print('  too few populated subsets to judge')
        return
    signs = {np.sign(r['mean']) for r in res}
    weakest = min(res, key=lambda r: r['t'])
    broad = len(signs) == 1 and weakest['t'] > 1.5
    print(f'  --> weakest disjoint slice: {weakest["mean"]:+.2f} bps/day '
          f'(t={weakest["t"]:+.2f})   verdict: '
          + ('BROAD -- edge survives in every disjoint slice'
             if broad else
             'CONCENTRATED -- edge is not uniform across disjoint names'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signal', choices=['on_12m', 'on_1m'], default='on_12m')
    ap.add_argument('--k', type=int, default=3, help='hash-split subsets')
    a = ap.parse_args()
    df = load_panel()
    rows = signal_rows(df, a.signal)
    names = sorted(rows.ticker.unique())
    print(f'\nreplication of the overnight-persistence edge -- signal '
          f'{a.signal}\n{len(names)} names, {rows.tmonth.nunique()} trade '
          f'months {rows.tmonth.min()}..{rows.tmonth.max()}')

    print('\nfull universe (the rule baseline this study partitions; the ML '
          'ceiling was RESULTS 15b):')
    report_subset('FULL', rows)

    hashgrp = {tk: subset_of(tk, a.k) for tk in names}
    partition_report(f'{a.k}-way DISJOINT hash split (stable md5(ticker)):',
                     rows, hashgrp)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(names)
    randgrp = {tk: (0 if i < len(names) // 2 else 1)
               for i, tk in enumerate(perm)}
    partition_report('2-way RANDOM split (seed 7):', rows, randgrp)

    print('\nA broad edge appears in every disjoint slice; a data-mined quirk '
          'hides in one.\nThis is the fast replication -- it answers "is it '
          'these few names?" without\nwaiting on forward months. Out-of-'
          'universe replication (names never in the\ntop-150) is the next '
          'test and needs a survivorship-aware external feed.')


if __name__ == '__main__':
    main()
