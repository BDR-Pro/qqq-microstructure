# Part of qqq-microstructure.
#
# The VM and GitHub Actions both append to reports/*.csv and both push, so a
# content conflict in an append-only log is a routine event, not an accident.
# The one unacceptable outcome is a LOST GRADED NIGHT: the forward record in
# RESULTS 19 is the only thing the probation clock runs on. This test builds a
# real conflict in a throwaway repo -- both sides appending, one side also
# adding the masked columns -- and asserts the resolver keeps every row and
# every column.
#
# Run: python -m pytest tests -q     (or: python tests/test_merge_daily.py)

import os, sys, shutil, subprocess, tempfile
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')
OLD = 'date,month,n,q5_on_bps,qqq_on_bps,tilt_bps\n'
NEW = OLD.rstrip('\n') + ',q5_on_masked,tilt_masked\n'


def _git(w, *a):
    return subprocess.run(['git'] + list(a), cwd=w, capture_output=True,
                          text=True)


def test_merge_keeps_every_graded_night_and_both_schemas():
    w = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(w, 'src'))
        os.makedirs(os.path.join(w, 'reports'))
        shutil.copy(os.path.join(ROOT, 'src', 'merge_daily.py'),
                    os.path.join(w, 'src'))
        log = os.path.join(w, 'reports', 'xsec_paper_daily.csv')
        _git(w, 'init', '-q', '.')
        _git(w, 'config', 'user.email', 't@t')
        _git(w, 'config', 'user.name', 't')
        open(log, 'w').write(OLD + '20260901,2026-09,10,5.0,2.0,3.0\n')
        _git(w, 'add', '-A'); _git(w, 'commit', '-qm', 'base')

        # the VM appends two nights on the old schema
        _git(w, 'checkout', '-qb', 'vm')
        open(log, 'a').write('20260902,2026-09,10,6.0,1.0,5.0\n'
                             '20260903,2026-09,10,-2.0,-1.0,-1.0\n')
        _git(w, 'commit', '-qam', 'vm')

        # Actions re-grades with the masked columns and adds a night of its own
        base = _git(w, 'rev-parse', 'HEAD~1').stdout.strip()
        _git(w, 'checkout', '-q', base)
        _git(w, 'checkout', '-qb', 'actions')
        open(log, 'w').write(NEW +
                             '20260901,2026-09,10,5.0,2.0,3.0,4.5,2.5\n'
                             '20260902,2026-09,10,6.0,1.0,5.0,5.5,4.5\n'
                             '20260904,2026-09,10,1.0,0.5,0.5,1.0,0.5\n')
        _git(w, 'commit', '-qam', 'actions')

        assert _git(w, 'merge', 'vm').returncode != 0, 'expected a conflict'
        r = subprocess.run([sys.executable, 'src/merge_daily.py'], cwd=w,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr

        out = pd.read_csv(log, dtype=str)
        assert list(out.date) == ['20260901', '20260902', '20260903',
                                  '20260904'], 'a graded night was lost'
        assert 'q5_on_masked' in out.columns, 'the newer schema was dropped'
        # the richer row wins where both sides have the same night ...
        assert out.set_index('date').loc['20260902', 'q5_on_masked'] == '5.5'
        # ... and the side that never had the column keeps its row anyway
        assert pd.isna(out.set_index('date').loc['20260903', 'q5_on_masked'])
        assert _git(w, 'commit', '-qm', 'merged').returncode == 0
    finally:
        shutil.rmtree(w, ignore_errors=True)


if __name__ == '__main__':
    test_merge_keeps_every_graded_night_and_both_schemas()
    print('merge_daily test: PASS')
