# Part of qqq-microstructure.
#
# Can a learned model beat the quintile rules in the cross-section?
#
# The baselines are xsec_backtest.py's two rules: rank on 12-2 momentum (hold the
# month) and rank on last month's overnight mean (hold the nights). The challenger
# is LightGBM over ten features per name-month, each one a published effect, no
# inventions: 12-2 momentum, 1-month reversal, overnight persistence at 1 and 12
# months, 3-month volatility, 3-month range, 3-month first-hour return (the p60
# column), dollar volume and its trend, and price level. Two targets, one pipeline:
# next month's total return and next month's overnight mean, both demeaned across
# the universe by rank-transforming within each month. Predicting the CROSS-SECTION
# is the point -- the market component that wrong-footed the QQQ-only model in the
# Nov 2025 - Mar 2026 holdout cancels out of a within-month rank by construction.
#
# Discipline, same as momentum_ml.py:
#   - Walk-forward by trade year: each year predicted by a model trained only on
#     years before it. No tuning; PARAMS are inherited verbatim from momentum_ml.py
#     and were fixed before this file ran on real data.
#   - Features and targets are cross-sectional ranks in [-1, 1] (Gu, Kelly & Xiu
#     2020), so no scale drift across 27 years and no outlier leverage.
#   - Built-in leak canary: the same walk-forward with targets PERMUTED within each
#     training month must produce ~zero L/S. If the canary prints a live number,
#     something leaks -- stop trusting the rest.
#   - Universe, eligibility, split exclusion and portfolio mechanics are
#     xsec_backtest.py's, so ML-vs-rule differences are the model, nothing else.
#     Rule baselines are computed from this file's own feature columns on the SAME
#     name-months. The still-liquid-next-month conditioning documented there
#     applies here identically, to model and baseline alike.
#
# A third target answers RESULTS 15's open question: the FLOOR. The same features
# and the same pipeline, but the target and holding are the 09:45 exit (on15) --
# the return that does not capture the opening print. RESULTS 15 showed the rule
# loses half its spread there (+10.0 -> +4.8); this measures whether the model's
# doubling survives where the trade is actually collectable. Rule baseline for the
# floor is the same on_1m ranking, evaluated on the 09:45 holding. On the
# synthetic panel p15 equals the open by construction, so block C prints
# identically to block B -- the structural check that only the exit differs.
#
# Validation on a synthetic panel with planted truths (persistent per-name drift
# and overnight drift), recorded before any real-data run: both canaries ~0; the
# overnight model DOUBLED its rule (+10.5 vs +5.3 bps/day) by finding the planted
# 12-month persistence the 1-month rule cannot see (on_12m 37% of gain); the
# total-return model recovered only ~75% of its rule (+10.2 vs +13.6) because that
# synthetic world's one truth is linear in mom_12_2 and the other nine features
# are noise there -- the cost of splitting attention. That is the honest shape of
# the bet: ML pays exactly when the extra features carry real signal. PARAMS were
# not touched after seeing this.
#
#   python src/xsec_ml.py                  # needs data/xsec/ from xsec_extract.py

import os, argparse
import numpy as np, pandas as pd
import lightgbm as lgb
from xsec_backtest import (load_panel, stats, era_table, ETF,
                           MIN_CC_DAYS, MIN_NAMES)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATS = ['mom_12_2', 'ret_1m', 'on_1m', 'on_12m', 'vol_3m', 'rng_3m',
         'fh_3m', 'dvol', 'dvol_chg', 'lprice']
PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=15,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, verbosity=-1, seed=7)
ROUNDS = 300
MIN_TRAIN = 2000        # name-months of training data before the first test year


def rank_pm(s, by):
    return s.groupby(by).rank(pct=True) * 2 - 1


def build_table(df):
    """One row per (trade month, name): ranked features known at formation,
    ranked + raw targets realised during the trade month."""
    df = df.assign(fh=np.log(df.p60 / df.open) * 1e4,
                   rng=(df.high - df.low) / df.close * 1e4)
    m = df.groupby(['ticker', 'month']).agg(
        cc_sum=('cc_bps', 'sum'), cc_n=('cc_bps', 'count'),
        on_mean=('on_bps', 'mean'), on_n=('on_bps', 'count'),
        on15_mean=('on15_bps', 'mean'), on15_n=('on15_bps', 'count'),
        cc_std=('cc_bps', 'std'), fh_mean=('fh', 'mean'),
        rng_mean=('rng', 'mean'), dv=('dollar_vol', 'mean'),
        close_last=('close', 'last')).to_dict('index')
    months = sorted(df.month.unique())
    uni = {mm: set(g) for mm, g in df.groupby('month')['ticker']}
    rows = []
    for i in range(13, len(months)):
        T = months[i]
        base = (uni[months[i - 1]] & uni[T]) - ETF
        look = [months[j] for j in range(i - 12, i)]        # T-12 .. T-1
        for tk in sorted(base):
            hist = [m.get((tk, mm)) for mm in look]
            if any(h is None or h['cc_n'] < MIN_CC_DAYS for h in hist):
                continue
            if hist[-1]['on_n'] < 8:
                continue
            cur = m[(tk, T)]
            on12 = [h['on_mean'] for h in hist if h['on_n'] >= 8]
            rows.append(dict(
                tmonth=T, ticker=tk,
                mom_12_2=sum(h['cc_sum'] for h in hist[:-1]),
                ret_1m=hist[-1]['cc_sum'],
                on_1m=hist[-1]['on_mean'],
                on_12m=np.mean(on12) if len(on12) >= 8 else np.nan,
                vol_3m=np.nanmean([h['cc_std'] for h in hist[-3:]]),
                rng_3m=np.mean([h['rng_mean'] for h in hist[-3:]]),
                fh_3m=np.mean([h['fh_mean'] for h in hist[-3:]]),
                dvol=np.log10(hist[-1]['dv']),
                dvol_chg=np.log10(hist[-1]['dv'])
                         - np.mean([np.log10(h['dv']) for h in hist[:-1]]),
                lprice=np.log10(hist[-1]['close_last']),
                y_cc=cur['cc_sum'],
                y_on=cur['on_mean'] if cur['on_n'] >= 8 else np.nan,
                y_on15=cur['on15_mean'] if cur['on15_n'] >= 8 else np.nan))
    t = pd.DataFrame(rows).dropna(subset=FEATS)
    keep = t.groupby('tmonth')['ticker'].transform('size') >= MIN_NAMES
    t = t[keep].reset_index(drop=True)
    for f in FEATS:
        t[f] = rank_pm(t[f], t.tmonth)
    t['ycc_r'] = rank_pm(t.y_cc, t.tmonth)
    t['yon_r'] = rank_pm(t.y_on, t.tmonth)
    t['yon15_r'] = rank_pm(t.y_on15, t.tmonth)
    t['year'] = t.tmonth.str[:4].astype(int)
    return t


def walk_forward(t, ycol, shuffle=False):
    preds = pd.Series(np.nan, index=t.index)
    model = None
    for ty in sorted(t.year.unique()):
        tr = t[(t.year < ty) & t[ycol].notna()]
        te = t[t.year == ty]
        if len(tr) < MIN_TRAIN or not len(te):
            continue
        y = tr[ycol].values
        if shuffle:
            rng = np.random.default_rng(ty)
            y = tr.groupby('tmonth')[ycol].transform(
                lambda s: rng.permutation(s.values)).values
        m = lgb.train(PARAMS, lgb.Dataset(tr[FEATS], y, free_raw_data=True),
                      num_boost_round=ROUNDS)
        preds.loc[te.index] = m.predict(te[FEATS])
        model = m
    return preds, model


def portfolio(t, score, bym, col):
    """Daily Q5-Q1 series: top vs bottom 20% of `score` each trade month,
    daily equal-weight means of `col` -- the mechanics of xsec_backtest.py."""
    out = []
    for T, g in t[score.notna()].groupby('tmonth'):
        s = score.loc[g.index]
        k = len(g) // 5
        if k < 5:
            continue
        o = s.values.argsort()
        q1 = set(g.ticker.values[o[:k]])
        q5 = set(g.ticker.values[o[-k:]])
        sub = bym[T]
        out.append(pd.DataFrame({
            'ls': sub[sub.ticker.isin(q5)].groupby('day')[col].mean()
                - sub[sub.ticker.isin(q1)].groupby('day')[col].mean()}))
    return pd.concat(out).dropna() if out else pd.DataFrame(columns=['ls'])


def monthly_ic(t, score, ycol):
    ic = {}
    for T, g in t[score.notna() & t[ycol].notna()].groupby('tmonth'):
        a = score.loc[g.index].rank()
        ic[T] = np.corrcoef(a, g[ycol].rank())[0, 1]
    return pd.Series(ic)


def save_production(t):
    """Freeze production models for xsec_replay.py: train on everything supplied.
    The frozen targets are the two that survived RESULTS 15/15b -- the overnight
    ceiling (MOO/MOC execution) and the 09:45 floor. cc was a null; not frozen."""
    import json, datetime
    md = os.path.join(ROOT, 'models')
    os.makedirs(md, exist_ok=True)
    meta = {'features': FEATS, 'last_tmonth': str(t.tmonth.max()),
            'name_months': int(len(t)),
            'frozen_on': datetime.date.today().isoformat()}
    for tag, ycol in (('on', 'yon_r'), ('on15', 'yon15_r')):
        tr = t[t[ycol].notna()]
        m = lgb.train(PARAMS, lgb.Dataset(tr[FEATS], tr[ycol], free_raw_data=True),
                      num_boost_round=ROUNDS)
        m.save_model(os.path.join(md, f'xsec_{tag}_lgbm.txt'))
        meta[f'{tag}_rows'] = int(len(tr))
    json.dump(meta, open(os.path.join(md, 'xsec_lgbm.json'), 'w'), indent=2)
    print(f'production models -> models/xsec_on_lgbm.txt + models/xsec_on15_lgbm.txt')
    print(f'trained through {meta["last_tmonth"]}  ({len(t):,} name-months)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save-model', action='store_true',
                    help='train on all history and write models/xsec_*_lgbm.txt')
    ap.add_argument('--through', default=None,
                    help='with --save-model: train only on trade months <= YYYY-MM')
    a = ap.parse_args()
    df = load_panel()
    bym = {m: g[['ticker', 'day', 'cc_bps', 'on_bps', 'on15_bps']]
           for m, g in df.groupby('month')}
    t = build_table(df)
    print(f'\nfeature table: {len(t):,} name-months, '
          f'{t.tmonth.nunique()} trade months {t.tmonth.min()} .. {t.tmonth.max()}')
    if a.save_model:
        save_production(t[t.tmonth <= a.through] if a.through else t)
        return

    runs = {}
    for tag, ycol, col in (('cc', 'ycc_r', 'cc_bps'), ('on', 'yon_r', 'on_bps'),
                           ('on15', 'yon15_r', 'on15_bps')):
        pred, model = walk_forward(t, ycol)
        canary, _ = walk_forward(t, ycol, shuffle=True)
        oos = pred.notna()
        rule_col = 'mom_12_2' if tag == 'cc' else 'on_1m'
        rule_score = t[rule_col].where(oos)
        runs[tag] = dict(pred=pred, model=model, oos=oos,
                         ls=portfolio(t, pred, bym, col),
                         rls=portfolio(t, rule_score, bym, col),
                         cls=portfolio(t, canary, bym, col),
                         ic=monthly_ic(t, pred, ycol))

    print(f'\nout-of-sample trade months: {t[runs["cc"]["oos"]].tmonth.nunique()} '
          f'({t[runs["cc"]["oos"]].tmonth.min()} .. {t[runs["cc"]["oos"]].tmonth.max()})')

    print('\nA. next-month total return (hold the month):')
    stats(runs['cc']['rls'].ls.values, 'rule 12-2 L/S')
    stats(runs['cc']['ls'].ls.values, 'ML L/S')
    stats(runs['cc']['cls'].ls.values, 'canary (shuffled)')
    print('  by 4-year era (ML L/S):')
    era_table(runs['cc']['ls'].index, runs['cc']['ls'].ls.values)

    print('\nB. next-month overnight mean (hold the nights):')
    stats(runs['on']['rls'].ls.values, 'rule on_1m L/S')
    stats(runs['on']['ls'].ls.values, 'ML L/S')
    stats(runs['on']['cls'].ls.values, 'canary (shuffled)')
    print('  by 4-year era (ML L/S):')
    era_table(runs['on']['ls'].index, runs['on']['ls'].ls.values)

    print('\nC. the floor: same trade sold at 09:45 -- does the model survive '
          'without the opening print?')
    stats(runs['on15']['rls'].ls.values, 'rule on_1m L/S')
    stats(runs['on15']['ls'].ls.values, 'ML L/S')
    stats(runs['on15']['cls'].ls.values, 'canary (shuffled)')
    print('  by 4-year era (ML L/S):')
    era_table(runs['on15']['ls'].index, runs['on15']['ls'].ls.values)

    icc, ico = runs['cc']['ic'], runs['on']['ic']
    icf = runs['on15']['ic']
    print(f'\nmonthly rank IC: cc {icc.mean():+.3f} '
          f'(t={icc.mean()/(icc.std()/np.sqrt(len(icc))):.1f})   '
          f'on {ico.mean():+.3f} '
          f'(t={ico.mean()/(ico.std()/np.sqrt(len(ico))):.1f})   '
          f'on15 {icf.mean():+.3f} '
          f'(t={icf.mean()/(icf.std()/np.sqrt(len(icf))):.1f})')
    j = runs['cc']['ls'].join(runs['on']['ls'], rsuffix='_on').dropna()
    if len(j) > 100:
        print(f'correlation(ML-cc L/S, ML-on L/S) = '
              f'{np.corrcoef(j.ls, j.ls_on)[0, 1]:+.3f}')

    print(f'\n{"year":>6}{"ic_cc":>8}{"ic_on":>8}{"ml_cc":>9}{"ml_on":>9}'
          f'{"ml_on15":>9}   bps/day')
    for y in sorted(t[runs['cc']['oos']].year.unique()):
        ys = str(y)
        a = runs['cc']['ls'][runs['cc']['ls'].index.str[:4] == ys].ls
        b = runs['on']['ls'][runs['on']['ls'].index.str[:4] == ys].ls
        c = runs['on15']['ls'][runs['on15']['ls'].index.str[:4] == ys].ls
        print(f'{y:>6}{icc[icc.index.str[:4] == ys].mean():>8.3f}'
              f'{ico[ico.index.str[:4] == ys].mean():>8.3f}'
              f'{a.mean():>9.2f}{b.mean():>9.2f}{c.mean():>9.2f}')

    for tag in ('cc', 'on', 'on15'):
        m = runs[tag]['model']
        imp = m.feature_importance(importance_type='gain')
        top = sorted(zip(FEATS, imp), key=lambda x: -x[1])[:5]
        print(f'top features, {tag} model (last fold, gain): '
              + '  '.join(f'{f} {v/imp.sum()*100:.0f}%' for f, v in top))

    out = pd.DataFrame({'mlcc_ls': runs['cc']['ls'].ls, 'mlon_ls': runs['on']['ls'].ls,
                        'mlon15_ls': runs['on15']['ls'].ls,
                        'rule_mom_ls': runs['cc']['rls'].ls,
                        'rule_on_ls': runs['on']['rls'].ls,
                        'rule_on15_ls': runs['on15']['rls'].ls})
    out.index.name = 'day'
    p = os.path.join(ROOT, 'data', 'xsec_ml_daily.csv')
    out.to_csv(p, float_format='%.3f')
    print(f'daily series -> {p}')


if __name__ == '__main__':
    main()
