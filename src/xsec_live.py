# Part of qqq-microstructure.
#
# The zero-capital live record -- monthly, because the strategy is monthly by
# construction: features and universe are frozen per calendar month, so the Q5
# basket computed tonight is identical every session of the month. One
# pre-registration per month is therefore exactly as much evidence as a daily
# run: emit the standing basket once, commit it, and the git timestamp makes it
# tamper-evident for every session it covers. Grading fills daily outcomes in
# retroactively from official open/close -- public history the selection never
# saw. Days BEFORE their month's emission date never grade: no pre-registration
# existed for them, so they are not evidence.
#
# Honesty rails, in code rather than promises:
#   - Feature parity check every run: the live feature builder must agree with
#     build_table's own rows to the decimal, or the run aborts.
#   - The live universe is the last complete month's top-150 (the lagged rule);
#     the backtest's still-in-top-150-next-month conditioning is unknowable
#     live, and live does not get it.
#   - Nothing retrains; models load frozen. A stale panel is announced loudly.
#
# Files (commit both -- the log IS the evidence):
#   reports/xsec_paper.csv        one row per month: the pre-registered basket
#   reports/xsec_paper_daily.csv  one row per graded session
#
#   python src/xsec_live.py        # emit this month's basket if new, grade the rest

import os, argparse, datetime as dt, json
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
import lightgbm as lgb
from xsec_backtest import load_panel, ETF, MIN_CC_DAYS
from xsec_ml import build_table, rank_pm, FEATS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, 'reports', 'xsec_paper.csv')
DAILY = os.path.join(ROOT, 'reports', 'xsec_paper_daily.csv')
ET = ZoneInfo('America/New_York')


def month_aggs(df):
    d = df.assign(fh=np.log(df.p60 / df.open) * 1e4,
                  rng=(df.high - df.low) / df.close * 1e4)
    return d.groupby(['ticker', 'month']).agg(
        cc_sum=('cc_bps', 'sum'), cc_n=('cc_bps', 'count'),
        on_mean=('on_bps', 'mean'), on_n=('on_bps', 'count'),
        cc_std=('cc_bps', 'std'), fh_mean=('fh', 'mean'),
        rng_mean=('rng', 'mean'), dv=('dollar_vol', 'mean'),
        close_last=('close', 'last')).to_dict('index')


def features(agg, look, universe):
    rows = []
    for tk in sorted(universe):
        hist = [agg.get((tk, mm)) for mm in look]
        if any(h is None or h['cc_n'] < MIN_CC_DAYS for h in hist):
            continue
        if hist[-1]['on_n'] < 8:
            continue
        on12 = [h['on_mean'] for h in hist if h['on_n'] >= 8]
        if len(on12) < 8:
            continue
        rows.append(dict(
            ticker=tk,
            mom_12_2=sum(h['cc_sum'] for h in hist[:-1]),
            ret_1m=hist[-1]['cc_sum'], on_1m=hist[-1]['on_mean'],
            on_12m=float(np.mean(on12)),
            vol_3m=float(np.nanmean([h['cc_std'] for h in hist[-3:]])),
            rng_3m=float(np.mean([h['rng_mean'] for h in hist[-3:]])),
            fh_3m=float(np.mean([h['fh_mean'] for h in hist[-3:]])),
            dvol=float(np.log10(hist[-1]['dv'])),
            dvol_chg=float(np.log10(hist[-1]['dv'])
                           - np.mean([np.log10(h['dv']) for h in hist[:-1]])),
            lprice=float(np.log10(hist[-1]['close_last']))))
    out = pd.DataFrame(rows)
    return out.dropna() if len(out) else out


def parity_check(df, t, agg, months):
    T = t.tmonth.max()
    i = months.index(T)
    uni_prev = set(df[df.month == months[i - 1]].ticker)
    uni_T = set(df[df.month == T].ticker)
    mine = features(agg, months[i - 12:i], (uni_prev & uni_T) - ETF)
    ref = t[t.tmonth == T][['ticker'] + FEATS]
    for f in FEATS:
        mine[f] = rank_pm(mine[f], pd.Series(0, index=mine.index))
    j = ref.merge(mine, on='ticker', suffixes=('_bt', '_lv'))
    d = max(float((j[f'{f}_bt'] - j[f'{f}_lv']).abs().max()) for f in FEATS)
    print(f'feature parity vs build_table ({T}, {len(j)} names): '
          f'max |rank diff| = {d:.6f}' + ('' if d < 1e-9 else '   <-- DRIFT'))
    return d < 1e-9


def grade(log, daily):
    import yfinance as yf
    today = dt.datetime.now(ET).strftime('%Y%m%d')
    done = set(daily.date) if len(daily) else set()
    for _, r in log.iterrows():
        m0 = r.month.replace('-', '')
        names = r.q5.split(';')
        d0 = dt.datetime.strptime(r.month + '-01', '%Y-%m-%d').date()
        d1 = (d0.replace(day=28) + dt.timedelta(days=12)).replace(day=1) \
            + dt.timedelta(days=9)
        try:
            v = yf.download([n.replace('.', '-') for n in names + ['QQQ']],
                            start=d0 - dt.timedelta(days=3), end=d1,
                            interval='1d', auto_adjust=False, actions=False,
                            group_by='ticker', progress=False)
            series = {}
            for n in names + ['QQQ']:
                try:
                    s = v[n.replace('.', '-')][['Open', 'Close']].dropna()
                    s.index = s.index.strftime('%Y%m%d')
                    series[n] = s
                except KeyError:
                    continue
        except Exception as ex:
            print(f'  {r.month}: grading fetch failed ({type(ex).__name__}) '
                  f'-- will retry next run')
            continue
        if 'QQQ' not in series:
            continue
        qd = series['QQQ'].index
        for k in range(len(qd) - 1):
            D, Dn = qd[k], qd[k + 1]
            if (D[:6] != m0[:6] or D < r.emitted_on or D >= today or D in done):
                continue
            ons = [float(np.log(series[n].Open[Dn] / series[n].Close[D]) * 1e4)
                   for n in names if n in series
                   and D in series[n].index and Dn in series[n].index]
            if len(ons) < 5:
                continue
            qqq = float(np.log(series['QQQ'].Open[Dn]
                               / series['QQQ'].Close[D]) * 1e4)
            daily = pd.concat([daily, pd.DataFrame([dict(
                date=D, month=r.month, n=len(ons),
                q5_on_bps=float(np.mean(ons)), qqq_on_bps=qqq,
                tilt_bps=float(np.mean(ons)) - qqq)])], ignore_index=True)
            done.add(D)
            print(f'graded {D}: basket {np.mean(ons):+.1f}  QQQ {qqq:+.1f}  '
                  f'tilt {np.mean(ons) - qqq:+.1f} bps')
    return daily


def main():
    argparse.ArgumentParser().parse_args()
    df = load_panel()
    months = sorted(df.month.unique())
    agg = month_aggs(df)
    t = build_table(df)
    if not parity_check(df, t, agg, months):
        raise SystemExit('live features drifted from build_table -- fix before '
                         'trusting any output')

    now = dt.datetime.now(ET)
    cur = now.strftime('%Y-%m')
    if months[-1] < (pd.Period(cur) - 1).strftime('%Y-%m'):
        print(f'WARNING: panel ends {months[-1]} but current month is {cur} -- '
              f'run xsec_extend.py first; emitting anyway on stale features')

    cols = ['emitted_on', 'month', 'universe_month', 'model_frozen',
            'n_scored', 'q5', 'q1']
    log = pd.read_csv(LOG, dtype=str) if os.path.exists(LOG) \
        else pd.DataFrame(columns=cols)
    dcols = ['date', 'month', 'n', 'q5_on_bps', 'qqq_on_bps', 'tilt_bps']
    daily = pd.read_csv(DAILY, dtype={'date': str, 'month': str}) \
        if os.path.exists(DAILY) else pd.DataFrame(columns=dcols)

    if cur in set(log.month):
        print(f'{cur} already pre-registered '
              f'(emitted {log[log.month == cur].emitted_on.iloc[0]})')
    else:
        live = features(agg, months[-12:],
                        set(df[df.month == months[-1]].ticker) - ETF)
        for f in FEATS:
            live[f] = rank_pm(live[f], pd.Series(0, index=live.index))
        mp = os.path.join(ROOT, 'models', 'xsec_lgbm.json')
        if not os.path.exists(mp):
            raise SystemExit('no frozen models -- run xsec_ml.py --save-model')
        meta = json.load(open(mp))
        m = lgb.Booster(model_file=os.path.join(ROOT, 'models',
                                                'xsec_on_lgbm.txt'))
        live['score'] = m.predict(live[FEATS])
        k = len(live) // 5
        q5 = live.nlargest(k, 'score').ticker.tolist()
        q1 = live.nsmallest(k, 'score').ticker.tolist()
        log = pd.concat([log, pd.DataFrame([dict(
            emitted_on=now.strftime('%Y%m%d'), month=cur,
            universe_month=months[-1],
            model_frozen=meta.get('frozen_on', '?'), n_scored=len(live),
            q5=';'.join(q5), q1=';'.join(q1))])], ignore_index=True)
        print(f'\npre-registered {cur}: hold overnight every session '
              f'(buy MOC, sell MOO), {len(q5)} names:')
        print('  ' + '  '.join(q5))

    daily = grade(log, daily)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log[cols].to_csv(LOG, index=False)
    daily[dcols].to_csv(DAILY, index=False)

    if len(daily):
        g = daily.astype({'q5_on_bps': float, 'qqq_on_bps': float,
                          'tilt_bps': float})
        print(f'\nscorecard: {len(g)} graded nights   basket '
              f'{g.q5_on_bps.mean():+.1f} bps/n   QQQ {g.qqq_on_bps.mean():+.1f}'
              f'   tilt {g.tilt_bps.mean():+.1f}   hit '
              f'{(g.tilt_bps > 0).mean() * 100:.0f}%   cum tilt '
              f'{g.tilt_bps.sum() / 100:+.2f}%')
    print(f'logs -> {LOG}\n        {DAILY}\n(commit both -- the log is the '
          f'evidence)')


if __name__ == '__main__':
    main()
