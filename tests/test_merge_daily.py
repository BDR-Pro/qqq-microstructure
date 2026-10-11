# Part of qqq-microstructure.
#
# The VM, GitHub Actions and anyone's own checkout all write reports/*.csv and
# all commit, so a content conflict in an append-only log is routine. The one
# unacceptable outcome is a LOST or DOUBLED graded night: the forward record
# in RESULTS 19 is the only thing the probation clock runs on. Every test here
# builds a real conflict in a throwaway repo and runs the real resolver --
# and the last one checks the cause, not the symptom: the grader itself must
# be able to rewrite the log without changing a byte.
#
# Run: python -m pytest tests -q     (or: python tests/test_merge_daily.py)

import os, sys, csv, shutil, subprocess, tempfile

ROOT = os.path.join(os.path.dirname(__file__), '..')
OLD = 'date,month,n,q5_on_bps,qqq_on_bps,tilt_bps\n'
NEW = OLD.rstrip('\n') + ',q5_on_masked,tilt_masked\n'
LOG = 'reports/xsec_paper_daily.csv'


def _git(w, *a):
    return subprocess.run(['git'] + list(a), cwd=w, capture_output=True,
                          text=True)


def _repo(base_text):
    w = tempfile.mkdtemp()
    os.makedirs(os.path.join(w, 'src'))
    os.makedirs(os.path.join(w, 'reports'))
    shutil.copy(os.path.join(ROOT, 'src', 'merge_daily.py'),
                os.path.join(w, 'src'))
    _git(w, 'init', '-q', '-b', 'main', '.')
    _git(w, 'config', 'user.email', 't@t')
    _git(w, 'config', 'user.name', 't')
    _git(w, 'config', 'core.autocrlf', 'false')
    _write(w, LOG, base_text)
    _git(w, 'add', '-A'); _git(w, 'commit', '-qm', 'base')
    return w


def _write(w, path, text):
    with open(os.path.join(w, path), 'w', newline='') as f:
        f.write(text)


def _branch_commit(w, branch, start, path, text, msg):
    _git(w, 'checkout', '-q', '-B', branch, start)
    _write(w, path, text)
    _git(w, 'add', '-A'); _git(w, 'commit', '-qm', msg)


def _resolve(w, *args):
    return subprocess.run([sys.executable, 'src/merge_daily.py'] + list(args),
                          cwd=w, capture_output=True, text=True)


def _rows(w, path=LOG):
    raw = open(os.path.join(w, path), 'rb').read()
    assert b'\r' not in raw, 'resolver wrote CRLF -- every line would churn'
    return list(csv.DictReader(raw.decode().splitlines()))


def test_both_sides_append_and_one_adds_columns():
    w = _repo(OLD + '20260901,2026-09,10,5.0,2.0,3.0\n')
    try:
        _branch_commit(w, 'vm', 'main', LOG, OLD +
                       '20260901,2026-09,10,5.0,2.0,3.0\n'
                       '20260902,2026-09,10,6.0,1.0,5.0\n'
                       '20260903,2026-09,10,-2.0,-1.0,-1.0\n', 'vm')
        _branch_commit(w, 'actions', 'main', LOG, NEW +
                       '20260901,2026-09,10,5.0,2.0,3.0,4.5,2.5\n'
                       '20260902,2026-09,10,6.0,1.0,5.0,5.5,4.5\n'
                       '20260904,2026-09,10,1.0,0.5,0.5,1.0,0.5\n', 'actions')
        assert _git(w, 'merge', 'vm').returncode != 0, 'expected a conflict'
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        rows = _rows(w)
        assert [x['date'] for x in rows] == ['20260901', '20260902',
                                             '20260903', '20260904']
        by = {x['date']: x for x in rows}
        assert by['20260902']['q5_on_masked'] == '5.5'   # the richer row wins
        assert by['20260903']['q5_on_masked'] == ''      # ...and the thin one stays
        assert _git(w, 'commit', '-qm', 'merged').returncode == 0
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_markers_committed_mid_merge_then_a_second_conflict():
    """The exact sequence seen on a real checkout: a conflict, then
    `git add . && git commit` WITHOUT resolving (markers now live in a commit),
    then upstream appends more nights and the next pull conflicts again."""
    w = _repo(OLD)
    try:
        # the local checkout graded two nights on its own (old pandas digits)
        _branch_commit(w, 'local', 'main', LOG, NEW +
                       '20260821,2026-08,10,-158.6,-53.1,-105.5,-158.6,-105.5\n'
                       '20260831,2026-08,10,-157.2,-131.6,-25.613501702212854,'
                       '-157.2,-25.613501702212854\n', 'local run')
        # Actions graded the same nights (digits drifted) plus one more
        _branch_commit(w, 'main', 'main', LOG, NEW +
                       '20260821,2026-08,10,-158.6,-53.1,-105.5,-158.6,-105.5\n'
                       '20260831,2026-08,10,-157.2,-131.6,-25.61350170221285,'
                       '-157.2,-25.61350170221285\n'
                       '20260902,2026-09,11,-67.6,22.7,-90.3,-67.6,-90.3\n',
                       'forward log (actions)')
        _git(w, 'checkout', '-q', 'local')
        assert _git(w, 'merge', 'main').returncode != 0
        # the mistake: commit the conflict markers
        _git(w, 'add', '.'); _git(w, 'commit', '-qm', 'update')
        assert '<<<<<<<' in open(os.path.join(w, LOG)).read()
        # upstream keeps appending
        _branch_commit(w, 'main', 'main', LOG, NEW +
                       '20260821,2026-08,10,-158.6,-53.1,-105.5,-158.6,-105.5\n'
                       '20260831,2026-08,10,-157.2,-131.6,-25.61350170221285,'
                       '-157.2,-25.61350170221285\n'
                       '20260902,2026-09,11,-67.6,22.7,-90.3,-67.6,-90.3\n'
                       '20260903,2026-09,11,156.0,23.5,132.5,156.0,132.5\n',
                       'forward log (actions)')
        _git(w, 'checkout', '-q', 'local')
        m = _git(w, 'merge', 'main')
        assert m.returncode != 0, 'expected the second conflict'
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        assert 'conflict-marker' in r.stdout          # it says what it did
        text = open(os.path.join(w, LOG)).read()
        assert '<<<<<<<' not in text and '=======' not in text
        rows = _rows(w)
        dates = [x['date'] for x in rows]
        assert dates == ['20260821', '20260831', '20260902', '20260903'], dates
        assert len(set(dates)) == len(dates), 'a night was doubled'
        # where the digits drifted, the shared record (theirs) wins
        assert {x['date']: x for x in rows}['20260831']['tilt_bps'] \
            == '-25.61350170221285'
        assert _git(w, 'commit', '-qm', 'resolved').returncode == 0
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_conflict_in_the_header_itself():
    """An older checkout without the masked columns: the HEADER line is in
    the conflict hunk. Every row must land under the header whose width it
    has, not be shifted into the wrong columns."""
    w = _repo(OLD + '20260901,2026-09,10,5.0,2.0,3.0\n')
    try:
        _branch_commit(w, 'old', 'main', LOG, OLD +
                       '20260901,2026-09,10,5.0,2.0,3.0\n'
                       '20260902,2026-09,10,6.0,1.0,5.0\n', 'old schema')
        _branch_commit(w, 'main', 'main', LOG, NEW +
                       '20260901,2026-09,10,5.0,2.0,3.0,4.5,2.5\n'
                       '20260903,2026-09,10,1.0,0.5,0.5,1.0,0.5\n', 'new')
        _git(w, 'checkout', '-q', 'old')
        assert _git(w, 'merge', 'main').returncode != 0
        _git(w, 'add', '.'); _git(w, 'commit', '-qm', 'update')   # markers in
        _branch_commit(w, 'main', 'main', LOG, NEW +
                       '20260901,2026-09,10,5.0,2.0,3.0,4.5,2.5\n'
                       '20260903,2026-09,10,1.0,0.5,0.5,1.0,0.5\n'
                       '20260904,2026-09,10,2.0,1.0,1.0,2.0,1.0\n', 'more')
        _git(w, 'checkout', '-q', 'old')
        assert _git(w, 'merge', 'main').returncode != 0
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        rows = _rows(w)
        assert [x['date'] for x in rows] == ['20260901', '20260902',
                                             '20260903', '20260904']
        by = {x['date']: x for x in rows}
        assert by['20260902']['tilt_bps'] == '5.0'     # not shifted
        assert by['20260904']['tilt_masked'] == '1.0'
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_order_log_keeps_both_legs_of_the_same_minute():
    """orders_log has two legitimate rows per session (close and open legs,
    dry and paper). A single-column key would collapse them."""
    H = 'run_at_et,month,leg,mode,names,notional,capital,placed\n'
    P = 'reports/orders_log.csv'
    w = _repo('')
    try:
        _write(w, P, H); _git(w, 'add', '-A'); _git(w, 'commit', '-qm', 'h')
        _branch_commit(w, 'vm', 'main', P, H +
                       '2026-09-02 15:25,2026-09,close,paper,MU;LRCX,24000,25000,2\n'
                       '2026-09-03 09:20,2026-09,open,paper,MU;LRCX,0,25000,2\n',
                       'vm')
        _branch_commit(w, 'main', 'main', P, H +
                       '2026-09-02 15:25,2026-09,close,dry,MU;LRCX,24000,25000,\n',
                       'dry run')
        _git(w, 'checkout', '-q', 'vm')
        assert _git(w, 'merge', 'main').returncode != 0
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        rows = _rows(w, P)
        assert len(rows) == 3, [tuple(x.values()) for x in rows]
        assert {(x['leg'], x['mode']) for x in rows} == {
            ('close', 'paper'), ('open', 'paper'), ('close', 'dry')}
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_refuses_paths_that_are_not_in_conflict():
    """A pasted sentence once became eight 'paths'. The answer must name the
    problem, not report 'neither side has content' eight times."""
    w = _repo(OLD + '20260901,2026-09,10,5.0,2.0,3.0\n')
    try:
        r = _resolve(w, 'resolves', 'a', 'real', 'conflict,')
        assert r.returncode != 0
        assert 'not a conflicted report log' in (r.stdout + r.stderr)
        assert 'nothing is in conflict' in (r.stdout + r.stderr)
        assert 'neither side' not in (r.stdout + r.stderr)
        assert _git(w, 'status', '--porcelain').stdout == ''  # touched nothing
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_line_one_marker_and_diff3_style():
    """git's diff3 style puts the BASE lines between ||||||| and =======; a
    stale base row must not come back. And with the header in the hunk, line
    1 of the committed side is a marker."""
    w = _repo(OLD + '20260901,2026-09,10,5.0,2.0,3.0\n')
    try:
        _git(w, 'config', 'merge.conflictStyle', 'diff3')
        _branch_commit(w, 'old', 'main', LOG, OLD +
                       '20260901,2026-09,10,5.0,2.0,3.1\n', 'regrade')
        _branch_commit(w, 'main', 'main', LOG, NEW +
                       '20260901,2026-09,10,5.0,2.0,3.2,4.5,2.5\n', 'new')
        _git(w, 'checkout', '-q', 'old')
        assert _git(w, 'merge', 'main').returncode != 0
        _git(w, 'add', '.'); _git(w, 'commit', '-qm', 'update')
        assert open(os.path.join(w, LOG)).read().startswith('<<<<<<<')
        _branch_commit(w, 'main', 'main', LOG, NEW +
                       '20260901,2026-09,10,5.0,2.0,3.2,4.5,2.5\n'
                       '20260902,2026-09,10,6.0,1.0,5.0,5.5,4.5\n', 'more')
        _git(w, 'checkout', '-q', 'old')
        assert _git(w, 'merge', 'main').returncode != 0
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        by = {x['date']: x for x in _rows(w)}
        assert sorted(by) == ['20260901', '20260902']
        assert by['20260901']['tilt_bps'] == '3.2'       # never the base 3.0
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_junk_rows_an_earlier_bad_merge_committed_are_dropped_and_named():
    """The previous resolver wrote marker lines and duplicated headers into
    the log as rows ('=======,,,,,,,'). They must go, and be reported."""
    junk = (NEW + '20260901,2026-09,10,5.0,2.0,3.0,4.5,2.5\n'
            '<<<<<<< HEAD,,,,,,,\n=======,,,,,,,\n>>>>>>> 48bb912,,,,,,,\n')
    w = _repo(junk)
    try:
        _branch_commit(w, 'a', 'main', LOG, junk +
                       '20260902,2026-09,10,6.0,1.0,5.0,5.5,4.5\n', 'a')
        _branch_commit(w, 'b', 'main', LOG, junk + NEW.rstrip('\n') + '\n' +
                       '20260903,2026-09,10,1.0,0.5,0.5,1.0,0.5\n', 'b')
        assert _git(w, 'merge', 'a').returncode != 0
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        assert 'junk row' in r.stdout
        assert [x['date'] for x in _rows(w)] == ['20260901', '20260902',
                                                 '20260903']
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_a_renamed_column_is_refused_not_guessed():
    """Both sides rename the same column differently, the header lands in the
    hunk, and the markers get committed. Two same-width headers with different
    names: which one does each row belong to? Unknowable -- refuse."""
    w = _repo(OLD + '20260901,2026-09,10,5.0,2.0,3.0\n')
    try:
        g = OLD.replace('q5_on_bps', 'q5_on_gross')
        n = OLD.replace('q5_on_bps', 'q5_on_net')
        _branch_commit(w, 'a', 'main', LOG, g +
                       '20260901,2026-09,10,5.0,2.0,3.0\n'
                       '20260902,2026-09,10,6.0,1.0,5.0\n', 'gross')
        _branch_commit(w, 'b', 'main', LOG, n +
                       '20260901,2026-09,10,5.0,2.0,3.0\n'
                       '20260903,2026-09,10,1.0,0.5,0.5\n', 'net')
        assert _git(w, 'merge', 'a').returncode != 0
        _git(w, 'add', '.'); _git(w, 'commit', '-qm', 'update')  # markers in
        _branch_commit(w, 'a', 'a', LOG, g +
                       '20260901,2026-09,10,5.0,2.0,3.0\n'
                       '20260902,2026-09,10,6.0,1.0,5.0\n'
                       '20260904,2026-09,10,2.0,1.0,1.0\n', 'more')
        _git(w, 'checkout', '-q', 'b')
        assert _git(w, 'merge', 'a').returncode != 0
        before = open(os.path.join(w, LOG), 'rb').read()
        r = _resolve(w)
        assert r.returncode != 0, r.stdout
        assert 'nothing was written' in (r.stdout + r.stderr)
        assert open(os.path.join(w, LOG), 'rb').read() == before
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_nothing_is_written_unless_every_file_merges():
    """Two conflicted logs, one unmergeable: the good one must not be
    half-applied either -- a half-resolved merge is worse than none."""
    H = 'run_at_et,month,leg,mode,names,notional,capital,placed\n'
    P = 'reports/orders_log.csv'
    w = _repo(OLD + '20260901,2026-09,10,5.0,2.0,3.0\n')
    try:
        _write(w, P, H); _git(w, 'add', '-A'); _git(w, 'commit', '-qm', 'h')
        _git(w, 'checkout', '-q', '-B', 'a', 'main')
        _write(w, LOG, OLD + '20260901,2026-09,10,5.0,2.0,3.0\n'
                            '20260902,2026-09,10,6.0,1.0,5.0\n')
        _write(w, P, H + 'not-a-time,2026-09,close,paper,MU,1,2,1\n')
        _git(w, 'add', '-A'); _git(w, 'commit', '-qm', 'a')
        _git(w, 'checkout', '-q', '-B', 'b', 'main')
        _write(w, LOG, OLD + '20260901,2026-09,10,5.0,2.0,3.0\n'
                            '20260903,2026-09,10,1.0,0.5,0.5\n')
        _write(w, P, H + '2026-09-02 15:25,2026-09,close,paper,MU,1,2,1\n')
        _git(w, 'add', '-A'); _git(w, 'commit', '-qm', 'b')
        assert _git(w, 'merge', 'a').returncode != 0
        before = open(os.path.join(w, LOG), 'rb').read()
        r = _resolve(w)
        assert r.returncode != 0, r.stdout
        assert 'orders_log.csv' in (r.stdout + r.stderr)
        assert open(os.path.join(w, LOG), 'rb').read() == before, \
            'the mergeable file was written although another file failed'
    finally:
        shutil.rmtree(w, ignore_errors=True)


def _conflict(w, path, header, ours_rows, theirs_rows):
    """Commit `ours_rows` on branch a and `theirs_rows` on branch b (both on
    top of a header-only base), check out a and merge b."""
    _write(w, path, header); _git(w, 'add', '-A'); _git(w, 'commit', '-qm', 'h')
    _branch_commit(w, 'b', 'main', path, header + theirs_rows, 'theirs')
    _branch_commit(w, 'a', 'main', path, header + ours_rows, 'ours')
    assert _git(w, 'merge', 'b').returncode != 0, 'expected a conflict'


def test_order_log_is_a_multiset_of_runs():
    """No identity column: two identical dry runs are two runs."""
    H = 'run_at_et,month,leg,mode,names,notional,capital,placed\n'
    P = 'reports/orders_log.csv'
    dry = '2026-09-02 15:08,2026-09,close,dry,MU;LRCX,2628.44,5000.00,\n'
    w = _repo('')
    try:
        _conflict(w, P, H, dry + dry,
                  dry + '2026-09-03 09:20,2026-09,open,paper,MU,0,25000,2\n')
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        rows = _rows(w, P)
        assert len(rows) == 3, rows                  # 2 dry + 1 paper
        assert sum(x['mode'] == 'dry' for x in rows) == 2
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_a_month_pre_registered_twice_is_refused():
    """The VM and Actions can both emit a basket on the 1st. Picking one
    silently would decide the pre-registration by merge direction."""
    H = 'emitted_on,month,universe_month,model_frozen,n_scored,q5,q1\n'
    P = 'reports/xsec_paper.csv'
    w = _repo('')
    try:
        _conflict(w, P, H,
                  '20261001,2026-10,2026-09,2026-08-20,57,MU;BE,MSFT;SGOV\n',
                  '20261001,2026-10,2026-09,2026-08-20,56,MU;STX,MSFT;WMT\n')
        before = open(os.path.join(w, P), 'rb').read()
        r = _resolve(w)
        assert r.returncode != 0, r.stdout
        assert 'pre-registration' in (r.stdout + r.stderr)
        assert open(os.path.join(w, P), 'rb').read() == before
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_digit_noise_is_not_a_disagreement_but_a_real_one_is_named():
    w = _repo('')
    try:
        _conflict(w, LOG, NEW,
                  '20260831,2026-08,10,-157.2,-131.6,-25.613501702212854,,\n'
                  '20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n',
                  '20260831,2026-08,10,-157.2,-131.6,-25.61350170221285,,\n'
                  '20260901,2026-09,10,5.0,2.0,9.9,5.0,9.9\n')
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        assert '20260831' not in r.stdout          # formatting, not data
        assert '20260901: the sides differ materially' in r.stdout
        by = {x['date']: x for x in _rows(w)}
        assert by['20260901']['tilt_bps'] == '9.9'    # the remote's version
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_a_side_without_the_key_column_is_refused():
    w = _repo('')
    try:
        _conflict(w, LOG, NEW, '20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n',
                  '20260902,2026-09,10,6.0,1.0,5.0,6.0,5.0\n')
        # rewrite theirs' stage without a date column: simulate by a branch
        _git(w, 'merge', '--abort')
        _branch_commit(w, 'b', 'b', LOG,
                       NEW.replace('date,', 'day,') +
                       '20260902,2026-09,10,6.0,1.0,5.0,6.0,5.0\n', 'renamed')
        _git(w, 'checkout', '-q', 'a')
        assert _git(w, 'merge', 'b').returncode != 0
        r = _resolve(w)
        assert r.returncode != 0, r.stdout
        assert 'nothing was written' in (r.stdout + r.stderr)
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_a_day_of_fills_keeps_every_execution():
    H = 'exec_id,time_et,date,symbol,side,qty,price,commission\n'
    P = 'reports/fills.csv'
    w = _repo('')
    try:
        _conflict(w, P, H,
                  '0001.a,2026-09-02 15:59:01,20260902,MU,BOT,10,100.0,1.0\n'
                  '0001.b,2026-09-02 15:59:01,20260902,LRCX,BOT,5,90.0,1.0\n',
                  '0001.a,2026-09-02 15:59:01,20260902,MU,BOT,10,100.0,1.0\n'
                  '0002.a,2026-09-03 09:30:02,20260903,MU,SLD,10,101.0,1.0\n')
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        assert [x['exec_id'] for x in _rows(w, P)] == ['0001.a', '0001.b',
                                                       '0002.a']
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_rebase_swaps_the_sides_and_the_continue_command():
    """`git pull --rebase`: stage :2 is UPSTREAM and stage :3 is the local
    commit being replayed. The remote's version must still win, and the
    command printed must be the rebase's own."""
    w = _repo(NEW + '20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n')
    try:
        _branch_commit(w, 'remote', 'main', LOG, NEW +
                       '20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n'
                       '20260902,2026-09,10,6.0,1.0,5.0,6.0,5.0\n', 'remote')
        _branch_commit(w, 'local', 'main', LOG, NEW +
                       '20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n'
                       '20260902,2026-09,10,6.0,1.0,7.7,6.0,7.7\n', 'local')
        assert _git(w, 'rebase', 'remote').returncode != 0
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        assert 'git rebase --continue' in r.stdout
        assert 'git commit --no-edit' not in r.stdout
        by = {x['date']: x for x in _rows(w)}
        assert by['20260902']['tilt_bps'] == '5.0'      # the remote's grade
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_a_night_both_sides_removed_does_not_come_back():
    base = NEW + ('20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n'
                  '20260999,2026-09,10,0,0,0,0,0\n')        # a bogus night
    w = _repo(base)
    try:
        good = '20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n'
        _branch_commit(w, 'a', 'main', LOG, NEW + good +
                       '20260902,2026-09,10,6.0,1.0,5.0,6.0,5.0\n', 'a')
        _branch_commit(w, 'b', 'main', LOG, NEW + good +
                       '20260903,2026-09,10,1.0,0.5,0.5,1.0,0.5\n', 'b')
        assert _git(w, 'merge', 'a').returncode != 0
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        assert [x['date'] for x in _rows(w)] == ['20260901', '20260902',
                                                 '20260903']
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_a_log_is_recognised_by_its_header_not_its_name():
    """A renamed graded log must still be merged on date -- as a multiset a
    re-graded night would appear twice."""
    P = 'reports/graded_v2.csv'
    w = _repo('')
    try:
        _conflict(w, P, NEW,
                  '20260831,2026-08,10,-157.2,-131.6,-25.613501702212854,,\n',
                  '20260831,2026-08,10,-157.2,-131.6,-25.61350170221285,,\n'
                  '20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n')
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        assert [x['date'] for x in _rows(w, P)] == ['20260831', '20260901']
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_committed_markers_are_found_without_a_conflict_and_repaired():
    """The state BEFORE the second pull: markers committed, nothing in
    conflict. 'nothing to do' would be a false all-clear."""
    broken = (NEW + '20260821,2026-08,10,-158.6,-53.1,-105.5,-158.6,-105.5\n'
              '<<<<<<< HEAD\n'
              '20260831,2026-08,10,-157.2,-131.6,-25.613501702212854,,\n'
              '=======\n'
              '20260831,2026-08,10,-157.2,-131.6,-25.61350170221285,,\n'
              '20260902,2026-09,11,-67.6,22.7,-90.3,-67.6,-90.3\n'
              '>>>>>>> 1f99899\n')
    w = _repo(broken)
    try:
        r = _resolve(w)
        assert r.returncode != 0 and 'BROKEN' in r.stdout, r.stdout + r.stderr
        assert open(os.path.join(w, LOG)).read() == broken      # untouched
        r = _resolve(w, '--repair')
        assert r.returncode == 0, r.stdout + r.stderr
        assert [x['date'] for x in _rows(w)] == ['20260821', '20260831',
                                                 '20260902']
        assert _resolve(w).returncode == 0          # clean now
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_an_unidentifiable_operation_refuses_material_disagreements():
    """`git stash pop` leaves no MERGE_HEAD or rebase directory, so there is
    no knowing which side is the shared record."""
    w = _repo(NEW + '20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n')
    try:
        _write(w, LOG, NEW + '20260901,2026-09,10,5.0,2.0,9.9,5.0,9.9\n')
        _git(w, 'stash')
        _branch_commit(w, 'main', 'main', LOG, NEW +
                       '20260901,2026-09,10,5.0,2.0,4.4,5.0,4.4\n', 'up')
        assert _git(w, 'stash', 'pop').returncode != 0
        r = _resolve(w)
        assert r.returncode != 0, r.stdout
        assert 'no remote version settles' in (r.stdout + r.stderr)
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_modify_delete_keeps_the_file_and_says_so():
    w = _repo(NEW + '20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n')
    try:
        _git(w, 'checkout', '-q', '-B', 'del', 'main')
        _git(w, 'rm', '-q', LOG); _git(w, 'commit', '-qm', 'rm')
        _branch_commit(w, 'keep', 'main', LOG, NEW +
                       '20260901,2026-09,10,5.0,2.0,3.0,5.0,3.0\n'
                       '20260902,2026-09,10,6.0,1.0,5.0,6.0,5.0\n', 'append')
        assert _git(w, 'merge', 'del').returncode != 0
        r = _resolve(w)
        assert r.returncode == 0, r.stdout + r.stderr
        assert 'DELETED' in r.stdout
        assert len(_rows(w)) == 2
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_a_failed_write_rolls_every_file_back(monkeypatch):
    """Two logs to write, the second cannot be (open in Excel): the first
    must be restored, not left half-resolved."""
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    import merge_daily as md
    w = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(w, 'reports'))
        for f in ('a.csv', 'b.csv'):
            _write(w, f'reports/{f}', 'orig ' + f)
        monkeypatch.setattr(md, 'ROOT', w)
        real, calls = os.replace, []

        def flaky(src, dst):
            calls.append(dst)
            if len(calls) == 2:
                raise PermissionError(13, 'Permission denied', dst)
            return real(src, dst)
        monkeypatch.setattr(md.os, 'replace', flaky)
        try:
            md.write_all({'reports/a.csv': ('new a', []),
                          'reports/b.csv': ('new b', [])})
            assert False, 'expected SystemExit'
        except SystemExit as ex:
            assert 'Nothing was changed' in str(ex)
        assert open(os.path.join(w, 'reports/a.csv')).read() == 'orig a.csv'
        assert open(os.path.join(w, 'reports/b.csv')).read() == 'orig b.csv'
        assert not [f for f in os.listdir(os.path.join(w, 'reports'))
                    if f.endswith('.tmp')]
    finally:
        shutil.rmtree(w, ignore_errors=True)


def test_the_grader_rewrites_the_forward_log_byte_for_byte():
    """The ROOT CAUSE of the routine conflicts: xsec_live re-read the log with
    pandas' inexact default float parser and wrote it all back, so every run
    changed the previous night's last digit (124.14844117239305 -> ...304),
    and two writers then edited the same line. Reading and writing the
    committed logs through the grader's own helpers must change no byte."""
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    import xsec_live
    w = tempfile.mkdtemp()
    try:
        a = os.path.join(ROOT, 'reports', 'xsec_paper.csv')
        b = os.path.join(ROOT, 'reports', 'xsec_paper_daily.csv')
        if not (os.path.exists(a) and os.path.exists(b)):
            return                              # a fresh clone with no log yet
        log, daily = xsec_live.read_logs(a, b)
        o1, o2 = os.path.join(w, 'p.csv'), os.path.join(w, 'd.csv')
        xsec_live.write_logs(log, daily, o1, o2)
        assert open(o1, 'rb').read() == open(a, 'rb').read()
        assert open(o2, 'rb').read() == open(b, 'rb').read(), \
            'a read->write round trip changed the forward record'
    finally:
        shutil.rmtree(w, ignore_errors=True)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and 'monkeypatch' not in \
                fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            fn()
            print(f'{name}: PASS')
