# Part of qqq-microstructure.
#
# Runs the saved model against data it has never seen and writes an end-of-day
# portfolio report (JSON + TXT) with balance, ROI and accuracy.
#
#   python src/evaluate.py                      # every .dbn.zst in eval_data/
#   python src/evaluate.py --from-zip 20251008  # a specific archive day (backtest)
#   python src/evaluate.py --from-zip last:12   # the days held out of training
#   python src/evaluate.py --reset              # clear the running account first
#
# Drop new Databento files into eval_data/ as they arrive -- days after 2025-10-08 were
# never in the archive, so they are a genuine out-of-sample test of the checkpoint.
#
# WHAT THE P&L ASSUMES, because it decides whether the number means anything:
#   - You were filled on every trade the model chose to quote into. You would not have
#     been; MBP-1 cannot see queue position. This is the phase-4 caveat and it is the
#     single largest source of optimism in these figures.
#   - Each fill is capped at --clip shares; you do not take the whole print.
#   - The position is unwound at the mid one second later (the markout horizon).
#   - The add rebate is earned on every filled share.

import os, sys, json, glob, argparse, zipfile, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, model as M

ROOT = dataset.ROOT
EVAL_DIR = os.path.join(ROOT, 'eval_data')
REPORT_DIR = os.path.join(ROOT, 'reports')
STATE_PATH = os.path.join(REPORT_DIR, 'portfolio_state.json')


def load_state(starting_capital):
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        'starting_capital': starting_capital,
        'balance': starting_capital,
        'cumulative_pnl': 0.0,
        'days': [],
        'total_fills_taken': 0,
        'total_shares': 0.0,
        'total_notional': 0.0,
    }


def evaluate_day(t, booster, man, clip, threshold, participation):
    """Score one day's fills and simulate the account. `t` comes from features_from_dbn.

    `participation` is the fraction of each qualifying print you actually win. Without
    it the simulation quietly assumes you beat every other maker to the front of the
    queue on every trade, which came to ~47% of all Nasdaq QQQ volume -- not a strategy,
    an impossibility. The report carries the implied venue share so it stays visible.
    """
    X = t[man['features']]
    pred = booster.predict(X)
    price, size = t.price.values.astype(float), t['size'].values.astype(float)
    actual = t[dataset.LABEL].values.astype(float)
    reb = M.rebate_bps(price)

    ev = pred + reb                       # expected net bps if filled here
    net = actual + reb                    # realised net bps
    take = ev > threshold
    per_fill = np.minimum(size, clip) * participation
    shares = per_fill * take
    pnl = shares * price * net / 1e4

    venue_volume = float(size.sum())
    taken = int(take.sum())
    day_pnl = float(pnl.sum())
    base_pnl = float((per_fill * price * net / 1e4).sum())      # quote everything

    r = {
        'day': str(t.day.iloc[0]),
        'fills_available': int(len(t)),
        'fills_taken': taken,
        'share_of_fills': float(take.mean()),
        'shares': float(shares.sum()),
        'notional_usd': float((shares * price).sum()),
        'venue_volume_shares': venue_volume,
        'pct_of_venue_volume': float(shares.sum() / venue_volume * 100),
        'participation_assumed': participation,
        'pnl_usd': day_pnl,
        'bps_per_share': float(np.average(net[take], weights=shares[take])) if taken else 0.0,
        'baseline_quote_everything_pnl_usd': base_pnl,
        'edge_vs_baseline_usd': day_pnl - base_pnl,
        'accuracy': {
            'sign_accuracy': float(np.mean(np.sign(ev) == np.sign(net))),
            'baseline_sign_accuracy': float(max(np.mean(net > 0), np.mean(net <= 0))),
            'precision_profitable': float(np.mean(net[take] > 0)) if taken else 0.0,
            'ic_pearson': float(np.corrcoef(pred, actual)[0, 1]),
            'rmse_bps': float(np.sqrt(np.mean((pred - actual) ** 2))),
        },
        'risk': {
            'max_clip_shares': clip,
            'peak_notional_per_fill_usd': float((shares * price).max()) if taken else 0.0,
            'worst_fill_usd': float(pnl.min()) if taken else 0.0,
            'best_fill_usd': float(pnl.max()) if taken else 0.0,
        },
    }
    return r


def write_reports(day_r, state):
    os.makedirs(REPORT_DIR, exist_ok=True)
    d = day_r['day']
    doc = {
        'generated_at': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        'model': {
            'checkpoint': os.path.basename(M.MODEL_PATH),
            'sha256': state.get('model_sha256'),
            'trees': state.get('model_trees'),
            'trained_days': state.get('model_trained_days'),
            'threshold_bps': state.get('threshold_bps'),
        },
        'session': day_r,
        'account': {
            'starting_capital_usd': state['starting_capital'],
            'balance_usd': state['balance'],
            'cumulative_pnl_usd': state['cumulative_pnl'],
            'roi_pct': state['cumulative_pnl'] / state['starting_capital'] * 100,
            'days_traded': len(state['days']),
            'total_fills_taken': state['total_fills_taken'],
            'total_shares': state['total_shares'],
            'total_notional_usd': state['total_notional'],
        },
        'assumptions': [
            'Fills are assumed on every quoted trade; queue position is not modelled '
            '(MBP-1 cannot see it). This is the largest source of optimism here.',
            f'You are assumed to win {state.get("participation", 0):.1%} of each qualifying '
            'print. At 100% the sim takes ~47% of all Nasdaq QQQ volume, which is not '
            'reachable; check pct_of_venue_volume before believing any P&L here.',
            'Position unwound at mid 1s after the fill (the markout horizon).',
            f'Add rebate ${M.REBATE_PER_SHARE}/share earned on every filled share.',
            'Single venue (Nasdaq), single symbol. No NBBO.',
        ],
    }
    jp = os.path.join(REPORT_DIR, f'eod_{d}.json')
    with open(jp, 'w') as f:
        json.dump(doc, f, indent=2)

    a, acc = doc['account'], day_r['accuracy']
    W = 66
    L = [
        '=' * W,
        f'  END OF DAY REPORT  --  {d}'.ljust(W),
        '=' * W, '',
        '  ACCOUNT',
        f'    Starting capital        ${a["starting_capital_usd"]:>14,.2f}',
        f'    Balance                 ${a["balance_usd"]:>14,.2f}',
        f'    Cumulative P&L          ${a["cumulative_pnl_usd"]:>14,.2f}',
        f'    ROI                      {a["roi_pct"]:>14.4f} %',
        f'    Days traded              {a["days_traded"]:>14,}', '',
        '  SESSION',
        f'    P&L                     ${day_r["pnl_usd"]:>14,.2f}',
        f'    Fills taken              {day_r["fills_taken"]:>14,}'
        f'  of {day_r["fills_available"]:,} ({day_r["share_of_fills"]*100:.1f}%)',
        f'    Shares                   {day_r["shares"]:>14,.0f}',
        f'    Notional                ${day_r["notional_usd"]:>14,.2f}',
        f'    Share of venue volume    {day_r["pct_of_venue_volume"]:>14.2f} %'
        f'  (participation {day_r["participation_assumed"]:.1%})',
        f'    Edge per share           {day_r["bps_per_share"]:>14.4f} bps', '',
        '  ACCURACY',
        f'    Sign accuracy            {acc["sign_accuracy"]*100:>14.2f} %'
        f'  (baseline {acc["baseline_sign_accuracy"]*100:.2f} %)',
        f'    Profitable of taken      {acc["precision_profitable"]*100:>14.2f} %',
        f'    IC (pred vs realised)    {acc["ic_pearson"]:>14.4f}',
        f'    RMSE                     {acc["rmse_bps"]:>14.4f} bps', '',
        '  VS QUOTING EVERYTHING',
        f'    Baseline P&L            ${day_r["baseline_quote_everything_pnl_usd"]:>14,.2f}',
        f'    Model edge              ${day_r["edge_vs_baseline_usd"]:>14,.2f}', '',
        '  ASSUMPTIONS',
    ]
    for s in doc['assumptions']:
        L += ['    - ' + s[:60]] + ['      ' + s[i:i+60] for i in range(60, len(s), 60)]
    L += ['', '=' * W]
    tp = os.path.join(REPORT_DIR, f'eod_{d}.txt')
    with open(tp, 'w') as f:
        f.write('\n'.join(L) + '\n')
    return jp, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-zip', default=None,
                    help='archive day(s) instead of eval_data/: YYYYMMDD, comma list, or last:N')
    ap.add_argument('--capital', type=float, default=100_000.0)
    ap.add_argument('--clip', type=float, default=100.0, help='max shares per fill')
    ap.add_argument('--participation', type=float, default=0.02,
                    help='fraction of each qualifying print you actually win (queue share)')
    ap.add_argument('--threshold', type=float, default=None,
                    help='override the threshold stored in the manifest')
    ap.add_argument('--reset', action='store_true', help='reset the running account')
    a = ap.parse_args()

    booster, man = M.load()
    if booster is None:
        raise SystemExit('no checkpoint in models/ -- run: python src/train.py')
    thr = a.threshold if a.threshold is not None else (man.get('decision_threshold_bps') or 0.0)

    if a.reset and os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)
        print('account reset')
    state = load_state(a.capital)
    state.update(model_sha256=man.get('model_sha256'), model_trees=man.get('num_trees'),
                 model_trained_days=len(man.get('trained_days', [])), threshold_bps=thr,
                 participation=a.participation)

    print(f'model: {man.get("num_trees")} trees, trained on '
          f'{len(man.get("trained_days", []))} days | threshold {thr:+.3f} bps '
          f'| clip {a.clip:.0f} sh | participation {a.participation:.1%}')

    # ---- assemble the day tables ----
    tables = []
    if a.from_zip:
        names = dataset.day_names()
        if a.from_zip.startswith('last:'):
            sel = names[-int(a.from_zip.split(':')[1]):]
        else:
            want = set(a.from_zip.split(','))
            sel = [n for n in names if n.split('-')[2].split('.')[0] in want]
        if not sel:
            raise SystemExit(f'no archive days matched {a.from_zip}')
        work = os.path.join(ROOT, 'data', '_work_eval')
        os.makedirs(work, exist_ok=True)
        for n in sel:
            day = n.split('-')[2].split('.')[0]
            tmp = os.path.join(work, f'{day}.zst')
            with zipfile.ZipFile(dataset.ZIP) as z, open(tmp, 'wb') as fo:
                fo.write(z.read(n))
            try:
                t = dataset.features_from_dbn(tmp, day)
                if t is not None:
                    tables.append(t)
            finally:
                os.remove(tmp)
    else:
        os.makedirs(EVAL_DIR, exist_ok=True)
        files = sorted(glob.glob(os.path.join(EVAL_DIR, '*.dbn.zst'))
                       + glob.glob(os.path.join(EVAL_DIR, '*.dbn')))
        if not files:
            raise SystemExit(
                f'no files in {os.path.relpath(EVAL_DIR, ROOT)}/ -- drop Databento '
                f'.dbn.zst files there, or use --from-zip for a backtest')
        for f in files:
            t = dataset.features_from_dbn(f)
            if t is not None:
                tables.append(t)

    if not tables:
        raise SystemExit('no usable trading days found')

    # ---- score, accumulate, report ----
    print(f'\n{"day":>10}{"fills":>9}{"taken":>8}{"P&L $":>11}{"bps/sh":>9}'
          f'{"venue%":>8}{"balance $":>13}')
    for t in tables:
        day = str(t.day.iloc[0])
        if day in [d['day'] for d in state['days']]:
            print(f'{day:>10}  already in the account -- skipping')
            continue
        r = evaluate_day(t, booster, man, a.clip, thr, a.participation)
        state['balance'] += r['pnl_usd']
        state['cumulative_pnl'] += r['pnl_usd']
        state['total_fills_taken'] += r['fills_taken']
        state['total_shares'] += r['shares']
        state['total_notional'] += r['notional_usd']
        state['days'].append({'day': day, 'pnl_usd': r['pnl_usd'],
                              'balance_usd': state['balance'],
                              'roi_pct': state['cumulative_pnl'] / state['starting_capital'] * 100})
        write_reports(r, state)
        print(f'{day:>10}{r["fills_available"]:>9,}{r["fills_taken"]:>8,}'
              f'{r["pnl_usd"]:>11,.2f}{r["bps_per_share"]:>9.4f}'
              f'{r["pct_of_venue_volume"]:>7.2f}%{state["balance"]:>13,.2f}')

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)

    roi = state['cumulative_pnl'] / state['starting_capital'] * 100
    print(f'\n{"":=<66}')
    print(f'  balance ${state["balance"]:,.2f}   P&L ${state["cumulative_pnl"]:+,.2f}   '
          f'ROI {roi:+.4f}%   over {len(state["days"])} day(s)')
    print(f'  reports -> {os.path.relpath(REPORT_DIR, ROOT)}/eod_*.json | .txt')


if __name__ == '__main__':
    main()
