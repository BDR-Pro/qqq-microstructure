# Part of qqq-microstructure.
#
# Regression tests for the defects the 2026-09 adversarial audit upheld, so
# none of them can silently return:
#   - factor.py must stitch QQQ/QQQQ and carry no seam point;
#   - poisoning the RAW parquets of future months (before load_panel, so the
#     derived on/cc/on15 columns and targets are poisoned too) must leave every
#     past feature and target byte-identical;
#   - opra_pull.window_utc must be DST-correct (the RESULTS 10 bug class);
#   - the overlay verdict must use common random numbers;
#   - no script may rebind its argparse namespace inside main() (the shadowed
#     `a` that crashed xsec_ml.py's meta sidecar after a full walk-forward).
#
# Run: python -m pytest tests -q      (or: python tests/test_audit_regressions.py)

import os, sys, glob, shutil, tempfile, datetime as dt
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
ROOT = os.path.join(os.path.dirname(__file__), '..')


def test_factor_stitches_qqq_qqqq_without_seam():
    import factor
    days = [f'2004{m:02d}01' for m in range(1, 13)] + \
           [f'2005{m:02d}01' for m in range(1, 9)]
    rows = []
    for i, d in enumerate(days):
        rows.append(dict(ticker='QQQ' if i < 10 else 'QQQQ', day=d,
                         on_bps=(np.nan if i == 10 else 5.0 + i),
                         cc_bps=1.0, id_bps=-1.0))
    F = factor.factors(pd.DataFrame(rows))
    assert len(F) == 20, 'both tickers must be present'
    assert F.mkt_on.abs().max() < 100, 'no rename seam point'
    assert F.loc['20050201':].mkt_on.notna().all(), 'QQQQ era must be populated'


def test_poison_raw_future_months_before_load_panel():
    from xsec_backtest import load_panel
    from xsec_ml import build_table, FEATS
    src = os.path.join(ROOT, 'data', 'xsec')
    files = sorted(glob.glob(os.path.join(src, '*.parquet')))
    if len(files) < 20:
        return                                   # no panel in this checkout
    months = [os.path.basename(f)[:7] for f in files]
    T = months[len(months) * 2 // 3]
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(0)
        for f, m in zip(files, months):
            df = pd.read_parquet(f)
            if m >= T:                           # poison the FUTURE at the source
                for c in ('open', 'high', 'low', 'close', 'p15', 'p30', 'p60',
                          'dollar_vol'):
                    if c in df:
                        df[c] = rng.uniform(1, 1e6, len(df))
            df.to_parquet(os.path.join(tmp, os.path.basename(f)), index=False)
        cols = ['tmonth', 'ticker'] + FEATS + ['y_on', 'y_cc']
        a = build_table(load_panel(src))
        b = build_table(load_panel(tmp))
        a = a[a.tmonth < T][cols].set_index(['tmonth', 'ticker']).sort_index()
        b = b[b.tmonth < T][cols].set_index(['tmonth', 'ticker']).sort_index()
        common = a.index.intersection(b.index)
        assert len(common) > 100
        # rows for months < T must be identical -- including TARGETS, which
        # come from month T-1's realised overnight, still in the past
        d = (a.loc[common] - b.loc[common]).abs().max().max()
        assert d < 1e-9, f'future poisoning moved a past feature/target: {d}'
        # and the poisoned rows must NOT have been silently dropped
        assert len(common) == len(a.index.intersection(b.index))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_opra_window_is_dst_correct():
    from opra_pull import window_utc
    a, b = window_utc('20240115')                # EST: 10:25 ET = 15:25Z
    assert (a.hour, a.minute) == (15, 25) and (b.hour, b.minute) == (15, 35)
    a, b = window_utc('20240715')                # EDT: 10:25 ET = 14:25Z
    assert (a.hour, a.minute) == (14, 25) and (b.hour, b.minute) == (14, 35)


def test_overlay_verdict_uses_common_random_numbers():
    import overlay
    r = pd.Series(np.random.default_rng(3).normal(8, 120, 3000))
    take = overlay.block_index(len(r), 300, np.random.default_rng(1))
    a = overlay.mc_row(r.values, 300, None, take)
    b = overlay.mc_row(r.values, 300, None, take)
    assert a['p50'] == b['p50'] and np.all(a['ratio'] == b['ratio'])


def test_no_script_shadows_its_argparse_namespace():
    """xsec_ml.main() bound `a = ap.parse_args()` and then reused `a` as a
    loop local, so the last line -- a.through, 40 lines later -- died with
    AttributeError AFTER the whole walk-forward had run. Cheap to check
    statically, expensive to hit at runtime, so check it for every script."""
    import ast
    bad = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'src', '*.py'))):
        tree = ast.parse(open(f, encoding='utf-8').read())
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)]:
            ns = {t.id for st in ast.walk(fn)
                  if isinstance(st, ast.Assign)
                  and isinstance(st.value, ast.Call)
                  and isinstance(st.value.func, ast.Attribute)
                  and st.value.func.attr == 'parse_args'
                  for t in st.targets if isinstance(t, ast.Name)}
            if not ns:
                continue
            for st in ast.walk(fn):
                tg = (st.targets if isinstance(st, ast.Assign) else
                      [st.target] if isinstance(st, (ast.AugAssign,
                                                     ast.AnnAssign, ast.For))
                      else [])
                for t in tg:
                    if isinstance(t, ast.Name) and t.id in ns and not (
                            isinstance(st, ast.Assign)
                            and isinstance(st.value, ast.Call)
                            and isinstance(st.value.func, ast.Attribute)
                            and st.value.func.attr == 'parse_args'):
                        bad.append(f'{os.path.basename(f)}:{t.lineno} '
                                   f'rebinds argparse namespace "{t.id}" '
                                   f'in {fn.name}()')
    assert not bad, 'argparse namespace shadowed: ' + '; '.join(bad)


if __name__ == '__main__':
    test_factor_stitches_qqq_qqqq_without_seam()
    test_poison_raw_future_months_before_load_panel()
    test_opra_window_is_dst_correct()
    test_overlay_verdict_uses_common_random_numbers()
    test_no_script_shadows_its_argparse_namespace()
    print('audit regression tests: PASS')
