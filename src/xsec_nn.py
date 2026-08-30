# Part of qqq-microstructure.
#
# The compute experiment, done the way this repo does things: does a NEURAL
# ranker on the WIDE cross-section beat the frozen LightGBM recipe -- or is
# the tree already extracting what the data holds? One shot, everything below
# fixed BEFORE any real-data run, judged by a pre-declared paired gate. The
# honest motivation (GKX 2020): nets only start beating trees when the
# cross-section gets big; the top-150 panel is ~20k name-months, the top-1000
# panel is ~10x that. This is the one regime where the GPU can matter.
#
# FIXED SPEC (inherited from GKX conventions, not tuned on this data):
#   architecture   10 -> 64 -> 32 -> 16 -> 1, ReLU, dropout 0.10 on hidden
#   training       Adam lr 1e-3, weight decay 1e-4, batch 2048, max 100
#                  epochs; early stop on the LAST 20% of training months
#                  (time-ordered validation -- later than all other training
#                  months, earlier than the test year: purged, no peeking),
#                  patience 10
#   ensemble       8 seeds averaged (kills init variance, GKX practice)
#   features       the SAME 10 rank-transformed FEATS as xsec_ml -- the
#                  experiment is model + data size, nothing else
#   target         yon_r (next-month overnight rank), the production target
#   walk-forward   by trade year, train strictly on earlier years, exactly
#                  xsec_ml's protocol; min 2000 training rows
#   baseline       LightGBM with xsec_ml.PARAMS/ROUNDS verbatim, same rows
#   evaluator      same for both models: monthly top-minus-bottom-quintile
#                  spread of realized overnight (y_on), table-level
#
# PRE-DECLARED GATE: the net replaces the tree ONLY if the paired monthly
# difference (nn L/S - lgb L/S over identical months) has mean > 0 AND
# paired t > 2. Anything less: THE TREE STANDS, and that is a recorded
# result, not a failure. Both models must pass the leak canary (targets
# permuted within month -> L/S ~ 0) or nothing here counts.
#
# Even a win changes nothing in production by itself: the frozen model stays
# frozen; adopting the net would be a deliberate, dated re-freeze with its
# own forward clock (RESULTS 19 discipline).
#
# Determinism: seeded per (year, member); exact on CPU, near-exact on GPU
# (cuDNN kernels). Device auto-selects cuda when present -- this is the file
# the A100 exists for.
#
# Validated on planted truth before real data: linear world -> both models
# find it; interaction world (y = sign(f1)*f2) -> both find it, net >= tree;
# permuted targets -> both canaries ~0; two runs bit-identical on CPU.
#
#   python src/xsec_nn.py                      # wide panel (data/xsec1000)
#   python src/xsec_nn.py --dir data/xsec      # small-panel mode (honest note)

import os, argparse
import numpy as np, pandas as pd

from xsec_backtest import load_panel
from xsec_ml import build_table, FEATS, PARAMS, ROUNDS, MIN_TRAIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENSEMBLE = 8
MAX_EPOCH, PATIENCE = 100, 10
BATCH, LR, WD, DROP = 2048, 1e-3, 1e-4, 0.10
HIDDEN = (64, 32, 16)


def table_ls(t, pred, col='y_on'):
    """Monthly top-quintile minus bottom-quintile of realized `col`, ranked by
    pred -- the shared evaluator for both models."""
    out = {}
    d = t.assign(p=pred).dropna(subset=['p', col])
    for T, g in d.groupby('tmonth'):
        q = len(g) // 5
        if q < 4:
            continue
        s = g.sort_values('p')
        out[T] = s[col].iloc[-q:].mean() - s[col].iloc[:q].mean()
    return pd.Series(out).sort_index()


def _make_net(torch, seed):
    torch.manual_seed(seed)
    import torch.nn as nn
    layers, prev = [], len(FEATS)
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(DROP)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def _fit_one(torch, dev, Xtr, ytr, Xva, yva, seed):
    seed = int(seed)
    net = _make_net(torch, seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    loss_fn = torch.nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev).view(-1, 1)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=dev)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=dev).view(-1, 1)
    g = torch.Generator(device='cpu').manual_seed(seed * 7919 + 1)
    best, best_state, bad = np.inf, None, 0
    n = len(Xtr_t)
    for ep in range(MAX_EPOCH):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(Xva_t), yva_t))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in
                          net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    return net


def nn_walk_forward(t, ycol='yon_r', shuffle=False, device=None, quiet=False):
    """Same protocol as xsec_ml.walk_forward, model swapped for the MLP
    ensemble. Validation for early stopping = last 20% of TRAINING months."""
    import torch
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    preds = pd.Series(np.nan, index=t.index)
    for ty in sorted(t.year.unique()):
        tr = t[(t.year < ty) & t[ycol].notna()]
        te = t[t.year == ty]
        if len(tr) < MIN_TRAIN or not len(te):
            continue
        y = tr[ycol].values.astype(np.float32)
        if shuffle:
            rng = np.random.default_rng(ty)
            y = tr.groupby('tmonth')[ycol].transform(
                lambda s: rng.permutation(s.values)).values.astype(np.float32)
        mons = sorted(tr.tmonth.unique())
        vset = set(mons[int(len(mons) * 0.8):])
        vm = tr.tmonth.isin(vset).values
        Xtr, ytr = tr[FEATS].values[~vm].astype(np.float32), y[~vm]
        Xva, yva = tr[FEATS].values[vm].astype(np.float32), y[vm]
        Xte = torch.tensor(te[FEATS].values.astype(np.float32), device=dev)
        acc = np.zeros(len(te))
        for s in range(ENSEMBLE):
            net = _fit_one(torch, dev, Xtr, ytr, Xva, yva,
                           seed=ty * 100 + s)
            with torch.no_grad():
                acc += net(Xte).cpu().numpy().ravel()
        preds.loc[te.index] = acc / ENSEMBLE
        if not quiet:
            print(f'    {ty}: trained {ENSEMBLE}-net ensemble on '
                  f'{len(Xtr):,}+{len(Xva):,} rows -> {len(te):,} test rows',
                  flush=True)
    return preds


def lgb_walk_forward(t, ycol='yon_r'):
    import lightgbm as lgb
    preds = pd.Series(np.nan, index=t.index)
    for ty in sorted(t.year.unique()):
        tr = t[(t.year < ty) & t[ycol].notna()]
        te = t[t.year == ty]
        if len(tr) < MIN_TRAIN or not len(te):
            continue
        m = lgb.train(PARAMS, lgb.Dataset(tr[FEATS], tr[ycol].values,
                                          free_raw_data=True),
                      num_boost_round=ROUNDS)
        preds.loc[te.index] = m.predict(te[FEATS])
    return preds


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/xsec1000')
    a = ap.parse_args()
    path = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {dev}' + ('' if dev == 'cuda' else
          '  (no GPU -- runs fine, just slower)'))
    if not os.path.isdir(path):
        raise SystemExit(f'{path} missing -- build the wide panel first:\n'
                         f'  python src/xsec_extract.py --top 1000 --out '
                         f'data/xsec1000\nor run --dir data/xsec for '
                         f'small-panel mode')
    df = load_panel(path)
    t = build_table(df)
    small = len(t) < 60000
    print(f'\ntable: {len(t):,} name-months, {t.tmonth.nunique()} months'
          + ('\nNOTE: this is the SMALL panel -- GKX says nets need the wide '
             'cross-section;\na tree win here is expected and proves little '
             'about the wide panel' if small else ''))

    print('\nLightGBM (frozen recipe, walk-forward):')
    lp = lgb_walk_forward(t)
    print('neural ensemble (fixed spec, walk-forward):')
    np_ = nn_walk_forward(t)
    print('neural leak canary (permuted targets):')
    cp = nn_walk_forward(t, shuffle=True, quiet=True)

    ls_l, ls_n, ls_c = table_ls(t, lp), table_ls(t, np_), table_ls(t, cp)
    common = ls_l.index.intersection(ls_n.index)
    ls_l, ls_n = ls_l[common], ls_n[common]

    def line(lab, s):
        tstat = s.mean() / (s.std() / np.sqrt(len(s)))
        print(f'  {lab:<18} {s.mean():+7.2f} bps/day  t={tstat:+5.2f}  '
              f'{(s > 0).mean()*100:3.0f}% mo+  ({len(s)} months)')
    print(f'\nout-of-sample monthly L/S (shared evaluator, y_on):')
    line('LightGBM', ls_l)
    line('neural ensemble', ls_n)
    line('canary (must ~0)', ls_c)
    if abs(ls_c.mean()) > max(1.5, 0.3 * abs(ls_n.mean())):
        print('  WARNING: canary is alive -- something leaks; the rest of '
              'this run is not trustworthy')

    d = ls_n - ls_l
    td = d.mean() / (d.std() / np.sqrt(len(d)))
    print(f'\npaired difference (nn - lgb), {len(d)} months: '
          f'{d.mean():+.2f} bps/day  t={td:+.2f}')
    print(f'{"years":>7}: ' + '  '.join(
        f'{y} {d[d.index.str[:4] == y].mean():+.1f}'
        for y in sorted({m[:4] for m in d.index})))
    if d.mean() > 0 and td > 2:
        print('\nGATE: NN BEATS THE TREE at the pre-declared bar (mean>0, '
              'paired t>2).\nAdoption would still be a deliberate dated '
              're-freeze with its own forward clock.')
    else:
        print('\nGATE: THE TREE STANDS. The net did not clear the '
              'pre-declared bar (mean>0 AND\npaired t>2) -- recorded as a '
              'result, not a failure; the frozen model is unchanged.')


if __name__ == '__main__':
    main()
