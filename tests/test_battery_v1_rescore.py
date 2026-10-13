# Part of qqq-microstructure.
#
# The decider battery's v1 result is a committed record (RESULTS 27): the
# live run's calls, scored and selected at the declaration's commit. Any
# later change to the scorer must still turn those calls into the same
# summary and the same selection, byte for byte -- otherwise the selection
# could drift after the fact without anyone re-running anything.

import os, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import battery_score as bs

RUN = os.path.join(ROOT, 'bench', 'battery', 'runs', 'v1')


def test_the_committed_v1_run_rescores_to_the_same_bytes(tmp_path):
    d = tmp_path / 'v1'
    shutil.copytree(RUN, d)
    want = {f: (d / f).read_bytes() for f in ('summary.json',
                                              'selection_v1.json')}
    for f in want:
        (d / f).unlink()
    bs.score(str(d), plan_path=os.path.join(ROOT, 'bench', 'battery',
                                            'plan_v1.json'))
    bs.select(str(d))
    for f, b in want.items():
        assert (d / f).read_bytes() == b, f
