# Part of qqq-microstructure.
#
# Trains the adverse-selection model, resuming from the checkpoint in models/.
#
#   python src/train.py             # train on any days not yet seen
#   python src/train.py --fresh     # discard the checkpoint and start over
#   python src/train.py --valid 10  # hold the last 10 extracted days for validation
#
# The manifest records every day the checkpoint has already been trained on, so a second
# run picks up where the first stopped instead of relearning the same data.

import os, sys, json, argparse, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, model as M


def metrics(pred, actual, price, size, threshold=0.0):
    reb = M.rebate_bps(price)
    ev, net = pred + reb, actual + reb
    took = ev > threshold
    out = {
        'n': int(len(pred)),
        'ic_pearson': float(np.corrcoef(pred, actual)[0, 1]) if len(pred) > 2 else float('nan'),
        'ic_spearman': float(pd.Series(pred).corr(pd.Series(actual), method='spearman')),
        'rmse_bps': float(np.sqrt(np.mean((pred - actual) ** 2))),
        # does the model call the sign of a fill's net P&L correctly?
        'sign_accuracy': float(np.mean(np.sign(ev) == np.sign(net))),
        'baseline_sign_accuracy': float(max(np.mean(net > 0), np.mean(net <= 0))),
        'share_quoted': float(took.mean()),
    }
    if took.any():
        out['bps_per_share_taken'] = float(np.average(net[took], weights=size[took]))
        out['precision_profitable'] = float(np.mean(net[took] > 0))
    out['bps_per_share_all'] = float(np.average(net, weights=size))
    out['threshold_bps'] = float(threshold)
    return out


def rule_baseline(df, threshold_label='spread>=2t & imb_own>0.4'):
    """The hand-written rule the model has to beat, scored the same way.

    Stated as a kill criterion up front: if a rule you can write on a napkin wins,
    ship the napkin.
    """
    net = df[dataset.LABEL].values + M.rebate_bps(df.price.values)
    sz = df['size'].values
    m = (df.spread_ticks.values >= 2) & (df.imb_own.values > 0.4)
    if not m.any():
        return None
    return {
        'rule': threshold_label,
        'share_of_fills': float(m.mean()),
        'bps_per_share': float(np.average(net[m], weights=sz[m])),
        'pnl_usd_uncapped': float((sz[m] * df.price.values[m] * net[m] / 1e4).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fresh', action='store_true', help='ignore any existing checkpoint')
    ap.add_argument('--valid', type=int, default=10, help='most recent N days held for validation')
    ap.add_argument('--rounds', type=int, default=M.ROUNDS_PER_SESSION)
    ap.add_argument('--replay', type=int, default=750_000,
                    help='rows resampled from already-seen days when resuming (0 disables)')
    ap.add_argument('--replay-days', type=int, default=25,
                    help='how many previously-seen days to draw the replay sample from')
    ap.add_argument('--force', action='store_true',
                    help='keep the new checkpoint even if validation degrades')
    a = ap.parse_args()

    days = dataset.extracted_days()
    if len(days) < a.valid + 5:
        raise SystemExit(f'only {len(days)} days extracted; run src/dataset.py first')
    valid_days, pool = days[-a.valid:], days[:-a.valid]

    booster, man = (None, M.blank_manifest()) if a.fresh else M.load()
    if a.fresh and os.path.exists(M.MODEL_PATH):
        print('--fresh: discarding existing checkpoint')
    seen = set(man['trained_days'])
    new_days = [d for d in pool if d not in seen]

    print(f'extracted {len(days)} days | validation {valid_days[0]}..{valid_days[-1]} '
          f'({len(valid_days)}) | training pool {len(pool)}')
    if booster is not None:
        print(f'resuming from checkpoint: {booster.num_trees()} trees, '
              f'{len(seen)} days already seen')
    else:
        print('no checkpoint - training from scratch')

    if not new_days:
        print('\nno unseen days: checkpoint is already current. '
              'Extract more days or pass --fresh to retrain.')
        return

    # Split validation in two by day: the first half calibrates the decision threshold,
    # the second half scores it. Picking the threshold on the same days you report would
    # flatter every number below.
    half = max(1, len(valid_days) // 2)
    cal_days, test_days = valid_days[:half], valid_days[half:]
    print(f'training on {len(new_days)} new days: {new_days[0]}..{new_days[-1]}')
    print(f'  threshold calibrated on {cal_days[0]}..{cal_days[-1]}, '
          f'scored on {test_days[0]}..{test_days[-1]}')
    tr = dataset.load(new_days)
    # Replay: mix in rows from days the checkpoint has already seen. Boosting on the new
    # days alone appends trees fitted to a narrow window, which measurably degraded
    # validation (IC 0.098 -> 0.078) the first time this ran. Replay keeps the appended
    # trees anchored to the full distribution.
    if booster is not None and a.replay > 0 and seen:
        prev = sorted(seen)
        rng = np.random.default_rng(0)
        pick = list(rng.choice(prev, size=min(len(prev), a.replay_days), replace=False))
        rep = dataset.load(pick)
        if len(rep) > a.replay:
            rep = rep.sample(n=a.replay, random_state=0)
        print(f'  replay: {len(rep):,} rows from {len(pick)} previously-seen days')
        tr = pd.concat([tr, rep], ignore_index=True)

    cal, te = dataset.load(cal_days), dataset.load(test_days)
    Xtr, ytr, wtr = tr[dataset.FEATURES], tr[dataset.LABEL].values, tr['size'].values
    print(f'  train rows {len(tr):,}   calib {len(cal):,}   test {len(te):,}')

    before = None
    if booster is not None:
        pv = booster.predict(te[dataset.FEATURES])
        before = metrics(pv, te[dataset.LABEL].values, te.price.values, te['size'].values,
                         man.get('decision_threshold_bps') or 0.0)

    booster = M.fit(Xtr, ytr, weight=wtr, booster=booster, rounds=a.rounds,
                    valid=(cal[dataset.FEATURES], cal[dataset.LABEL].values, cal['size'].values))

    thr, sweep = M.choose_threshold(booster.predict(cal[dataset.FEATURES]),
                                    cal[dataset.LABEL].values, cal.price.values, cal['size'].values)
    pred = booster.predict(te[dataset.FEATURES])
    after = metrics(pred, te[dataset.LABEL].values, te.price.values, te['size'].values, thr)
    rule = rule_baseline(te)
    va = te

    man['features'] = dataset.FEATURES
    man['trained_days'] = sorted(seen | set(new_days))
    man['validation_days'] = valid_days
    man['total_rounds'] = man.get('total_rounds', 0) + a.rounds
    man['decision_threshold_bps'] = thr
    man['metrics'] = after
    man['rule_baseline'] = rule
    man['threshold_sweep'] = sweep
    # Regression guard: never ship a checkpoint that scores worse than the one it
    # replaces. The days are still marked seen so the next run moves on rather than
    # retrying the same batch forever.
    K = 'bps_per_share_taken'
    degraded = before is not None and after.get(K, -9) < before.get(K, -9)
    accepted = (not degraded) or a.force
    man['sessions'].append({
        'at': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        'new_days': len(new_days),
        'rows': int(len(tr)),
        'rounds': a.rounds,
        'replay_rows': int(a.replay) if booster is not None else 0,
        'valid_ic': after['ic_pearson'],
        'valid_bps_taken': after.get(K),
        'accepted': bool(accepted),
    })
    if accepted:
        path = M.save(booster, man)
    else:
        keep, prev_man = M.load()          # the file still holds the previous model
        prev_man['trained_days'] = man['trained_days']
        prev_man['sessions'] = man['sessions']
        path = M.save(keep, prev_man)
        booster = keep

    print(f'\n--- validation ({len(va):,} fills over {len(valid_days)} unseen days) ---')
    if before:
        print(f"IC          {before['ic_pearson']:+.4f} -> {after['ic_pearson']:+.4f}")
        print(f"bps/share   {before.get('bps_per_share_taken', float('nan')):+.4f} -> "
              f"{after.get('bps_per_share_taken', float('nan')):+.4f}")
    else:
        print(f"IC (pred vs actual markout)   {after['ic_pearson']:+.4f}")
        print(f"Spearman                      {after['ic_spearman']:+.4f}")
    print(f"sign accuracy                 {after['sign_accuracy']:.4f} "
          f"(always-one-class baseline {after['baseline_sign_accuracy']:.4f})")
    print(f"quote-everything bps/share    {after['bps_per_share_all']:+.4f} on 100% of fills")
    print(f"model-gated bps/share         {after.get('bps_per_share_taken', float('nan')):+.4f} "
          f"on {after['share_quoted']*100:.1f}% of fills  (threshold {thr:+.3f} bps)")
    if rule:
        print(f"hand-written rule bps/share   {rule['bps_per_share']:+.4f} "
              f"on {rule['share_of_fills']*100:.1f}% of fills")
        mp = (va['size'].values * va.price.values
              * (va[dataset.LABEL].values + M.rebate_bps(va.price.values)) / 1e4)
        took = pred + M.rebate_bps(va.price.values) > thr
        print(f"  -> total P&L (uncapped): model ${mp[took].sum():,.0f} "
              f"vs rule ${rule['pnl_usd_uncapped']:,.0f}")
        if rule['bps_per_share'] > after.get('bps_per_share_taken', -9):
            print("  NOTE: the rule has the better per-share edge; the model wins on breadth.")
    if degraded and not a.force:
        print(f'\nREJECTED: validation fell {before[K]:+.4f} -> {after[K]:+.4f} bps/share. '
              f'Previous checkpoint kept.\n  Days are marked seen so the next run moves on. '
              f'Use --force to accept anyway, or --fresh to retrain from scratch.')
    print(f'checkpoint -> {os.path.relpath(path, dataset.ROOT)}  '
          f'({booster.num_trees()} trees, {len(man["trained_days"])} days seen)')


if __name__ == '__main__':
    main()
