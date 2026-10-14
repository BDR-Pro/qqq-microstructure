# Part of qqq-microstructure.
#
# xsec_extract_fast.collapse must return what xsec_extract.one_month does --
# the panel's definition -- on the same minute data. The reference is run
# for real, its download served from a synthetic month (no network), with
# the cases that bite: DST, a NaN open, a short session, a name below the
# top-N cut, missing snapshot minutes. dollar_vol may differ only in the
# last float bits (summation order).

import os, sys, io
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import xsec_extract as xe
import xsec_extract_fast as xf


def month(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    days = ['2025-03-07', '2025-03-10', '2025-03-11']     # DST starts 03-09
    for tk, scale in (('AAA', 5), ('BBB', 3), ('CCC', 1), ('DDD', 0.01)):
        for d in days:
            # 09:20 .. 16:05 ET, some minutes dropped
            start = pd.Timestamp(f'{d} 09:20', tz='America/New_York')
            n = 160 if (tk, d) == ('CCC', days[1]) else 405
            ts = start + pd.to_timedelta(np.arange(n), 'min')
            keep = rng.random(n) > 0.05
            keep[10] = True                    # 09:30, the first session bar
            ts = ts[keep]
            px = 100 + np.cumsum(rng.normal(0, 0.1, len(ts)))
            df = pd.DataFrame({'timestamp': ts.tz_convert('UTC'), 'open': px,
                               'high': px + 0.05, 'low': px - 0.05,
                               'close': px + 0.01,
                               'volume': rng.integers(100, 1000, len(ts))
                               * scale, 'ticker': tk})
            rows.append(df)
    t = pd.concat(rows, ignore_index=True)
    # AAA's first session bar on day 1 has no open: iloc[0] keeps the NaN,
    # a groupby first() would silently take the next bar's
    at = pd.Timestamp('2025-03-07 09:30', tz='America/New_York')
    t.loc[(t.ticker == 'AAA') & (t.timestamp == at), 'open'] = np.nan
    return t


def test_the_fast_collapse_is_the_reference(tmp_path, monkeypatch):
    t = month()
    src = tmp_path / 'src.parquet'
    t.to_parquet(src, index=False)

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(xe.urllib.request, 'urlopen',
                        lambda req, timeout=None: R(src.read_bytes()))
    monkeypatch.setattr(xe, 'OUT', str(tmp_path / 'ref'))
    monkeypatch.setattr(xe, 'TMP', str(tmp_path / 'tmp'))
    monkeypatch.setattr(xe, 'TOP_N', 3)
    os.makedirs(xe.OUT)
    os.makedirs(xe.TMP)
    assert xe.one_month('2025-03') > 0
    ref = pd.read_parquet(tmp_path / 'ref' / '2025-03.parquet')
    new = xf.collapse(pd.read_parquet(src, columns=xf.COLS), '2025-03', 3)
    assert list(new.columns) == list(ref.columns)
    assert new.dtypes.to_dict() == ref.dtypes.to_dict()
    assert 'DDD' not in set(new.ticker)                  # below the cut
    assert len(new) == len(ref) and len(new) < 9         # a short day dropped
    pd.testing.assert_frame_equal(new, ref, check_exact=False, rtol=1e-15,
                                  atol=0)
    assert new.open.isna().sum() == 1 == ref.open.isna().sum()
    assert all(((new[c] == ref[c]) | (new[c].isna() & ref[c].isna())).all()
               for c in new.columns if c != 'dollar_vol')
