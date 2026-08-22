# Part of qqq-microstructure.
#
# RESULTS 12's standing instruction is "diagnose, don't filter". This is the
# diagnosis, built BEFORE it is needed, so a bad forward month gets a five-
# minute post-mortem instead of a panicked re-research session. Given a month,
# it rebuilds the frozen model's Q5/Q1 selection exactly as the replay does
# (same build_table features, same frozen booster, same top/bottom len//5 as
# xsec_ml.portfolio) and decomposes the result into the only three places a
# month can go wrong:
#
#   MARKET     QQQ's own overnight -- the component no selection controls
#   BREADTH    universe equal-weight minus QQQ -- did the premium itself thin
#              out across the board?
#   SELECTION  Q5 minus universe (long side) and universe minus Q1 (short
#              side) -- did the model pick badly?
#
# plus the per-name table (worst first), the worst nights with their culprit
# names, and the data flags (excluded split/gap days, missing days, names that
# left the top-150 the next month). The decomposition is an identity: long
# selection + short selection = the L/S; market + breadth + long selection =
# the Q5 basket vs zero. Nothing here decides anything -- it shows where the
# month happened, and RESULTS 12 says what NOT to do about it: no filters.
#
# A month is diagnosable once its panel month exists (i.e. after monthly.py's
# extend step) -- the in-flight month becomes diagnosable next month.
#
#   python src/diagnose.py                    # latest month after the freeze cutoff
#   python src/diagnose.py --month 2026-07    # any panel month (pre-cutoff = in-sample)

import os, json, argparse
import numpy as np, pandas as pd
import lightgbm as lgb
from xsec_backtest import load_panel
from xsec_ml import build_table, FEATS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = 12.2                       # RESULTS 15b walk-forward reference, bps/day


def pick(g, score):
    """Exactly xsec_ml.portfolio's selection: top/bottom len//5 by score."""
    k = len(g) // 5
    o = np.asarray(score).argsort()
    return list(g.ticker.values[o[-k:]]), list(g.ticker.values[o[:k]]), k


def name_table(sub, names, mdays, gone, label, reverse):
    rows = []
    for tk in names:
        s = sub[sub.ticker == tk]
        on = s.on_bps.dropna()
        rows.append((tk, on.mean() if len(on) else np.nan, len(on),
                     int(s.on_bps.isna().sum()), len(mdays) - len(s),
                     '*gone' if tk in gone else ''))
    rows.sort(key=lambda r: (np.inf if np.isnan(r[1]) else r[1]),
              reverse=reverse)
    print(f'  {label}  ({"worst shorts first" if reverse else "worst first"}; '
          f'excl = split/gap-excluded nights, miss = days not in panel)')
    print(f'  {"name":<8}{"bps/nt":>8}{"nights":>8}{"excl":>6}{"miss":>6}')
    for tk, mu, n, ex, ms, fl in rows:
        print(f'  {tk:<8}{mu:>+8.1f}{n:>8}{ex:>6}{ms:>6}  {fl}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', default=None, help='YYYY-MM (default: latest '
                    'month after the frozen training cutoff)')
    a = ap.parse_args()
    md = os.path.join(ROOT, 'models')
    mp = os.path.join(md, 'xsec_lgbm.json')
    if not os.path.exists(mp):
        raise SystemExit('no frozen models -- run xsec_replay.py first')
    meta = json.load(open(mp))
    cut = meta['last_tmonth']

    df = load_panel()
    t = build_table(df)
    M = a.month
    if M is None:
        after = sorted(t[t.tmonth > cut].tmonth.unique())
        if not after:
            raise SystemExit(f'no months after the cutoff {cut} in the panel '
                             f'-- run xsec_extend.py, or pass --month YYYY-MM')
        M = after[-1]
    g = t[t.tmonth == M]
    if g.empty:
        raise SystemExit(f'{M} is not in the feature table (panel month '
                         f'missing, or too few eligible names)')
    months = sorted(df.month.unique())
    nxt = months[months.index(M) + 1] if M in months \
        and months.index(M) + 1 < len(months) else None

    m = lgb.Booster(model_file=os.path.join(md, 'xsec_on_lgbm.txt'))
    q5, q1, k = pick(g, m.predict(g[FEATS]))
    sub = df[df.month == M]
    mdays = sorted(sub.day.unique())
    gone = (set(q5) | set(q1)) - set(df[df.month == nxt].ticker) if nxt \
        else set()

    qqq = sub[sub.ticker.isin({'QQQ', 'QQQQ'})].sort_values(
        ['day', 'ticker']).groupby('day').first().on_bps
    unid = sub[sub.ticker.isin(set(g.ticker))].groupby('day').on_bps.mean()
    q5d = sub[sub.ticker.isin(q5)].groupby('day').on_bps.mean()
    q1d = sub[sub.ticker.isin(q1)].groupby('day').on_bps.mean()
    ls = (q5d - q1d).dropna()

    print(f'\n{M}: {len(g)} eligible names, Q5/Q1 = {k} each, model frozen '
          f'{meta.get("frozen_on", "?")} (trained through {cut})'
          + ('   ** IN-SAMPLE month -- mechanics only, no evidential value'
             if M <= cut else ''))
    print(f'  L/S {ls.mean():+.2f} bps/day over {len(ls)} nights '
          f'({ls.sum()/100:+.2f}%)   walk-forward reference {REF:+.1f}')
    m15 = lgb.Booster(model_file=os.path.join(md, 'xsec_on15_lgbm.txt'))
    q5f, q1f, _ = pick(g, m15.predict(g[FEATS]))
    lsf = (sub[sub.ticker.isin(q5f)].groupby('day').on15_bps.mean()
           - sub[sub.ticker.isin(q1f)].groupby('day').on15_bps.mean()).dropna()
    print(f'  09:45 floor model, same month: {lsf.mean():+.2f} bps/day '
          f'(reference +7.3)')

    j = pd.concat([q5d, q1d, unid, qqq], axis=1,
                  keys=['q5', 'q1', 'uni', 'qqq']).dropna()
    print(f'\n  where the month happened (means over {len(j)} common nights, '
          f'bps/day):')
    print(f'    market    QQQ overnight                {j.qqq.mean():+8.2f}')
    print(f'    breadth   universe EW minus QQQ        '
          f'{(j.uni - j.qqq).mean():+8.2f}')
    print(f'    long sel  Q5 minus universe            '
          f'{(j.q5 - j.uni).mean():+8.2f}')
    print(f'    short sel universe minus Q1            '
          f'{(j.uni - j.q1).mean():+8.2f}')
    print(f'    identity: long+short sel = L/S         '
          f'{(j.q5 - j.q1).mean():+8.2f}')
    print(f'    basket vs QQQ (the paper-log tilt)     '
          f'{(j.q5 - j.qqq).mean():+8.2f}  '
          f'(= breadth + long sel)')

    print()
    name_table(sub, q5, mdays, gone, f'Q5 (long {k})', reverse=False)
    print()
    name_table(sub, q1, mdays, gone, f'Q1 (short {k})', reverse=True)

    worst = (j.q5 - j.q1).nsmallest(3)
    print(f'\n  worst nights (L/S):')
    for D, v in worst.items():
        night = sub[(sub.day == D) & sub.ticker.isin(q5)] \
            .dropna(subset=['on_bps'])
        culprit = night.loc[night.on_bps.idxmin()] if len(night) else None
        print(f'    {D}  L/S {v:+7.1f}   QQQ {j.qqq[D]:+7.1f}   '
              + (f'worst long: {culprit.ticker} {culprit.on_bps:+.0f}'
                 if culprit is not None else ''))
    if gone:
        print(f'\n  left the top-150 after {M}: {" ".join(sorted(gone))}')
    print(f'\n  RESULTS 12: diagnose, don\'t filter. Market and breadth are '
          f'nobody\'s fault;\n  selection is the model\'s; data flags are '
          f'the pipeline\'s. No new rules from one month.')


if __name__ == '__main__':
    main()
