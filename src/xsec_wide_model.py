# Part of qqq-microstructure.
#
# The FROZEN model on names it never saw. xsec_wide.py asks whether the
# overnight-persistence PHENOMENON reaches beyond the top-150 (a rule, no
# model). This asks the sharper question about the thing that is actually
# traded: does the frozen LightGBM ranking (models/xsec_on_lgbm.txt, trained
# 2026-08-20 on the core panel through 2026-03) still sort overnight returns
# when every name it ranks was never in its training panel?
#
# Same tiers as xsec_wide.py, from the wide panel (data/xsec1000, built by
# xsec_extract_fast.py), by each month's own dollar-volume rank:
#   CORE   ever ranked <= 150: names the model trained on (in-sample -- a
#          reference, not evidence)
#   MID    never ranked <= 150 in any month, median rank <= 400
#   DEEP   never ranked <= 150, deeper
# Each tier's feature table is built from that tier's names alone, so the
# month-ranked features mean what they meant in training (rank within the
# universe being traded). The model is loaded, never refitted.
#
# What it can and cannot show. The names are out of sample; the MONTHS are
# not: 1999-2026 overlaps the training window, so market-wide regimes the
# model learned are shared with the tiers. A positive MID/DEEP result says the
# learned mapping generalises across names -- it is not forward evidence, and
# it does not touch RESULTS 19's verdict.
#
# VERDICT RULE, declared 2026-10-14 before the first run: GENERALISES iff the
# Q5-Q1 overnight spread is positive with t > 2 in BOTH never-core tiers;
# PARTIAL iff positive in both but a t is <= 2; DOES NOT GENERALISE otherwise.
#
# FORWARD MODE (--after 2026-03): the same scoring restricted to trade months
# after the model's training cutoff -- months built by xsec_extend_wide.py from
# Yahoo, after the HF source stopped. Then the names AND the months are out of
# sample: the one forward, broad readout the frozen model can get today. Its
# verdict uses the same rule; few months make it noisy, and it is
# non-binding beside RESULTS 19's replay.
#
#   python src/xsec_wide_model.py [--dir data/xsec1000] [--after 2026-03]

import os, sys, argparse
import numpy as np, pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xsec_backtest import load_panel, stats
from xsec_ml import build_table, portfolio, monthly_ic, FEATS
from xsec_wide import monthly_rank, CORE_N
from xsec_replicate import signal_rows, ls_series

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, 'models', 'xsec_on_lgbm.txt')


def tiers(df):
    """xsec_wide.py's tiering, as sets of tickers."""
    rk = monthly_rank(df)
    best = rk.groupby('ticker').min()
    med = rk.groupby('ticker').median()
    names = set(df.ticker.unique())
    core = {tk for tk in names if best.get(tk, 1e9) <= CORE_N}
    never = names - core
    mid = {tk for tk in never if med[tk] <= 400}
    deep = never - mid
    assert not (mid | deep) & core
    return {'CORE': core, 'MID': mid, 'DEEP': deep}


def score_tier(df, names, booster, after=None):
    """The frozen model's Q5-Q1 overnight series and monthly IC inside one
    tier, the tier being its own universe; with `after`, only trade months
    after it (features still use the 12 months before each)."""
    sub = df[df.ticker.isin(names)]
    t = build_table(sub)
    if after:
        t = t[t.tmonth > after]
    if t.empty:
        return None
    pred = pd.Series(booster.predict(t[FEATS]), index=t.index)
    bym = {m: g[['ticker', 'day', 'on_bps']] for m, g in sub.groupby('month')}
    ls = portfolio(t, pred, bym, 'on_bps')
    ic = monthly_ic(t, pred, 'yon_r')
    if ls.empty:
        return None
    x = ls.ls.values
    mo = ls.ls.groupby(ls.index.str[:6]).mean()
    return dict(names=len(names), months=t.tmonth.nunique(), days=len(x),
                mean=float(x.mean()), t=float(x.mean() / (x.std(ddof=1) /
                                                          np.sqrt(len(x)))),
                pos_months=float((mo > 0).mean()), ic=float(ic.mean()),
                q5=float(ls.q5.mean()))


def rule_tiers(df, tiers_, after=None):
    """The persistence rule (no model), computed exactly as xsec_wide.py
    does -- signal rows on the whole panel, ranked within each tier -- for
    the same trade months, so the forward readout shows the phenomenon and
    the frozen model side by side. Monthly spreads; t across months."""
    rows = signal_rows(df, 'on_12m')
    if after:
        rows = rows[rows.tmonth > after]
    out = {}
    for lab, names in tiers_.items():
        ls, ic = ls_series(rows[rows.ticker.isin(names)])
        out[lab] = None if len(ls) < 2 else dict(
            months=len(ls), mean=float(ls.mean()),
            t=float(ls.mean() / (ls.std() / np.sqrt(len(ls)))),
            pos_months=float((ls > 0).mean()), ic=float(ic.mean()))
    return out


def verdict(res):
    oou = [res.get('MID'), res.get('DEEP')]
    if any(r is None for r in oou):
        return 'too thin to judge -- a never-core tier did not score'
    if all(r['mean'] > 0 and r['t'] > 2 for r in oou):
        return 'GENERALISES -- the frozen ranking sorts names it never saw'
    if all(r['mean'] > 0 for r in oou):
        return 'PARTIAL -- right sign in both never-core tiers, a t <= 2'
    return 'DOES NOT GENERALISE beyond the names it trained on'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/xsec1000')
    ap.add_argument('--after', default=None, metavar='YYYY-MM',
                    help='score only trade months after this one (forward)')
    a = ap.parse_args()
    path = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    df = load_panel(path)
    booster = lgb.Booster(model_file=MODEL)
    tr = tiers(df)
    print(('FORWARD: trade months after ' + a.after + ' only\n'
           if a.after else '') +
          f'frozen model on the wide panel: {df.month.nunique()} months, '
          f'{df.ticker.nunique()} names; tiers CORE {len(tr["CORE"])}, '
          f'MID {len(tr["MID"])}, DEEP {len(tr["DEEP"])}')
    print(f'  {"tier":<6}{"names":>7}{"months":>8}{"L/S bps/d":>11}{"t":>8}'
          f'{"months+":>9}{"IC":>8}{"Q5 bps/d":>10}')
    res = {}
    for lab, names in tr.items():
        r = res[lab] = score_tier(df, names, booster, a.after)
        if r is None:
            print(f'  {lab:<6} did not score (too few names a month)')
            continue
        print(f'  {lab:<6}{r["names"]:>7}{r["months"]:>8}{r["mean"]:>+11.2f}'
              f'{r["t"]:>+8.2f}{r["pos_months"]:>9.0%}{r["ic"]:>+8.3f}'
              f'{r["q5"]:>+10.2f}'
              + ('   (in-sample names: reference only)' if lab == 'CORE'
                 else ''))
    print(f'VERDICT: {verdict(res)}')
    if a.after:
        print(f'the persistence rule (no model), same tiers and months '
              f'(information; monthly spreads):')
        for lab, r in rule_tiers(df, tr, a.after).items():
            print(f'  {lab:<6}' + ('did not score' if r is None else
                  f'{r["months"]:>3} mo  {r["mean"]:+7.2f} bps/d  t '
                  f'{r["t"]:+5.2f}  {r["pos_months"]:.0%} mo+  IC '
                  f'{r["ic"]:+.3f}'))
    print('(names out of sample, months not: evidence about the mapping, not '
          'a forward test;\n RESULTS 19\'s verdict and the traded book are '
          'untouched)')
    return res


if __name__ == '__main__':
    main()
