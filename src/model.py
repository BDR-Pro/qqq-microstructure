# Part of qqq-microstructure.
#
# The adverse-selection model: given the book state, predict the maker's markout in bps
# if a passive quote resting here were filled right now.
#
# Deliberately NOT a price-direction model. Direction is predictable in this data at
# IC 0.24 and still loses money, because the edge is smaller than the spread you cross
# to act on it. What pays is knowing which fills to want -- so the label is the P&L of
# an action, in bps, and the model's job is to gate quoting.
#
# Persistence lives here. A checkpoint is a LightGBM text model plus a JSON manifest
# recording which days it has already seen, so training resumes instead of restarting.

import os, json, hashlib, shutil
import numpy as np
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'adverse_selection.txt')
MANIFEST_PATH = os.path.join(MODEL_DIR, 'adverse_selection.manifest.json')

# Nasdaq add rebate, $/share. The whole business is rebate-funded, so it belongs in
# the decision rule rather than being bolted on afterwards.
REBATE_PER_SHARE = 0.00305

PARAMS = dict(
    objective='regression',
    metric='l2',
    learning_rate=0.04,
    num_leaves=63,
    min_data_in_leaf=2000,       # markout is mostly noise; keep leaves statistically real
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    verbosity=-1,
    num_threads=0,
)
ROUNDS_PER_SESSION = 300


def rebate_bps(price):
    """Add rebate expressed in bps of price, so it can be added to a markout figure."""
    return REBATE_PER_SHARE / np.asarray(price, dtype=float) * 1e4


def blank_manifest():
    return {
        'schema': 1,
        'model': 'lightgbm.regression',
        'target': 'markout_bps @1s (maker P&L, mid-unwind)',
        'features': [],
        'trained_days': [],
        'sessions': [],
        'total_rounds': 0,
        'rebate_per_share': REBATE_PER_SHARE,
        'decision_threshold_bps': None,
        'metrics': {},
    }


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return blank_manifest()


def save(booster, manifest):
    """Write checkpoint atomically so an interrupted save cannot corrupt the model."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp_m, tmp_j = MODEL_PATH + '.tmp', MANIFEST_PATH + '.tmp'
    booster.save_model(tmp_m)
    with open(tmp_m, 'rb') as f:
        manifest['model_sha256'] = hashlib.sha256(f.read()).hexdigest()[:16]
    manifest['num_trees'] = booster.num_trees()
    with open(tmp_j, 'w') as f:
        json.dump(manifest, f, indent=2)
    shutil.move(tmp_m, MODEL_PATH)
    shutil.move(tmp_j, MANIFEST_PATH)
    return MODEL_PATH


def load():
    """Return (booster, manifest). Booster is None when no checkpoint exists yet."""
    manifest = load_manifest()
    if not os.path.exists(MODEL_PATH):
        return None, manifest
    return lgb.Booster(model_file=MODEL_PATH), manifest


def fit(X, y, weight=None, booster=None, rounds=ROUNDS_PER_SESSION, valid=None):
    """Train, continuing from `booster` when one is supplied.

    LightGBM's init_model appends trees to the existing ensemble, so a resumed session
    refines the checkpoint rather than discarding it.
    """
    ds = lgb.Dataset(X, label=y, weight=weight, free_raw_data=False)
    valid_sets, names = [], []
    if valid is not None:
        vX, vy, vw = valid
        valid_sets = [lgb.Dataset(vX, label=vy, weight=vw, reference=ds)]
        names = ['valid']
    cb = [lgb.log_evaluation(period=100)] if valid_sets else []
    return lgb.train(PARAMS, ds, num_boost_round=rounds, init_model=booster,
                     valid_sets=valid_sets, valid_names=names, callbacks=cb)


def decide(pred_bps, price, threshold_bps=0.0):
    """Quote / don't quote.

    Expected value of a fill is the predicted markout plus the rebate earned for adding
    liquidity. Quote only where that clears `threshold_bps`.
    """
    return (np.asarray(pred_bps) + rebate_bps(price)) > threshold_bps


def choose_threshold(pred_bps, actual_bps, price, size, grid=None):
    """Pick the decision threshold that maximises realised P&L on validation data.

    Returns (threshold, table) where table holds the full sweep for inspection --
    a threshold that only wins at one point on the grid is a threshold to distrust.
    """
    if grid is None:
        grid = np.round(np.arange(-0.10, 0.201, 0.005), 4)
    ev = np.asarray(pred_bps) + rebate_bps(price)
    net = (np.asarray(actual_bps) + rebate_bps(price)) / 1e4 * np.asarray(price) * np.asarray(size)
    rows = []
    for t in grid:
        m = ev > t
        n = int(m.sum())
        rows.append({
            'threshold_bps': float(t),
            'fills': n,
            'share_of_fills': float(m.mean()),
            'pnl_usd': float(net[m].sum()) if n else 0.0,
            'pnl_bps_per_share': float(np.average(
                (np.asarray(actual_bps)[m] + rebate_bps(np.asarray(price)[m])),
                weights=np.asarray(size)[m])) if n else 0.0,
        })
    best = max(rows, key=lambda r: r['pnl_usd'])
    return best['threshold_bps'], rows
