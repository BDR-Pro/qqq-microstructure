# Part of qqq-microstructure.
#
# The local decider runner (src/decide_local.py) and the guards it relies on
# in guard.py: downloads and the installed runtime are verified or refused;
# the server is ours (fresh key, 127.0.0.1, the pinned model, no config) or
# refused and stopped; tonight is logged once per backend on a complete state,
# a failing decider never costs the control's row; only the log is committed,
# and a night not pushed before its close is dropped; the schedule lands on
# ET weekdays from any time zone, with paths that cannot break the command.

import os, sys, io, json, time, socket, tarfile, hashlib, subprocess
import datetime as dt
import numpy as np, pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import decide_local as dl
import guard as g

STUB = [sys.executable, os.path.join(HERE, 'stub_llama_server.py')]
NAMES = ['MU', 'STX', 'INTC']
NOW = dt.datetime(2026, 10, 13, 9, 17, tzinfo=g.ET)
STUB_ENV = ('STUB_ALIAS', 'STUB_NOAUTH', 'STUB_ANYKEY', 'STUB_CONFIG',
            'STUB_MODEL', 'STUB_P')


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv('QQQ_DECIDER_HOME', str(tmp_path / 'home'))
    for k in STUB_ENV:
        monkeypatch.delenv(k, raising=False)
    return tmp_path / 'home'


def stub(m, threads=None):
    return dl.Server(m, cmd=STUB, wait=30)


def listening(port):
    return not dl.port_free(port)


# ------------------------------------------------------------------ pins --

def test_every_pin_is_a_full_sha256_and_names_carry_runtime_and_platform():
    for asset, sha, label in dl.LLAMA.values():
        assert dl.LLAMA_TAG in asset and len(sha) == 64 and int(sha, 16) >= 0
    for m, (repo, rev, f, sha, size) in dl.MODELS.items():
        assert len(rev) == 40 and len(sha) == 64 and size > 1e9
        for p, (*_, label) in dl.LLAMA.items():
            assert dl.model_name(m, p) == \
                f'{m}@{rev[:8]}+llama.cpp-{dl.LLAMA_TAG}-{label}'
    # the Linux and Windows builds are different deciders, never pooled
    assert dl.model_name('kev-4b-q8_0', 'linux-x86_64') != \
        dl.model_name('kev-4b-q8_0', 'windows-amd64')


def test_a_download_is_verified_or_discarded(tmp_path, home):
    src = tmp_path / 'blob'
    src.write_bytes(b'kev' * 1000)
    good = dl.sha256(src)
    with pytest.raises(SystemExit, match='mismatch'):
        dl.fetch(src.as_uri(), tmp_path / 'out' / 'x.gguf', '0' * 64)
    assert not list((tmp_path / 'out').iterdir())       # nothing kept
    out = dl.fetch(src.as_uri(), tmp_path / 'out' / 'x.gguf', good)
    assert out.read_bytes() == src.read_bytes()
    out.write_bytes(b'tampered')        # a corrupted copy is replaced
    dl.fetch(src.as_uri(), out, good)
    assert dl.sha256(out) == good


def test_the_installed_runtime_is_checked_against_its_archive(home,
                                                             monkeypatch):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        for name, data in (('llama-x/llama-server', b'#!server'),
                           ('llama-x/libggml.so.0.1', b'lib')):
            i = tarfile.TarInfo(name)
            i.size = len(data)
            t.addfile(i, io.BytesIO(data))
        link = tarfile.TarInfo('llama-x/libggml.so')
        link.type, link.linkname = tarfile.SYMTYPE, 'libggml.so.0.1'
        t.addfile(link)
    arc = home / 'downloads' / 'llama.tgz'
    arc.parent.mkdir(parents=True)
    arc.write_bytes(buf.getvalue())
    sha = hashlib.sha256(buf.getvalue()).hexdigest()
    monkeypatch.setattr(dl, 'plat', lambda: 'linux-x86_64')
    monkeypatch.setattr(dl, 'LLAMA', {'linux-x86_64': ('llama.tgz', sha,
                                                       'linux-x64')})
    with pytest.raises(SystemExit, match='differs'):
        dl.check_llama()                                # not extracted yet
    with tarfile.open(arc) as t:
        t.extractall(dl.llama_dir(), filter='data')
    dl.check_llama()                                    # intact: passes
    (dl.llama_dir() / 'llama-x' / 'llama-server').write_bytes(b'#!evil')
    with pytest.raises(SystemExit, match='llama-server differs'):
        dl.check_llama()
    arc.write_bytes(b'not the release')
    with pytest.raises(SystemExit, match='does not match its pin'):
        dl.check_llama()


# ---------------------------------------------------------------- server --

def test_the_server_is_ours_or_refused_and_stopped(home, monkeypatch):
    with stub('kev-4b-q8_0') as s:
        assert s.url == f'http://127.0.0.1:{s.port}/v1/systemone'
        p, model, _ = g.ask_http({'names': {}}, NAMES, s.url, s.key, s.alias,
                                 opener=dl.DIRECT.open)
        assert model == dl.model_name('kev-4b-q8_0')
        assert set(p) == set(NAMES) | {g.BOOK}
        with pytest.raises(SystemExit, match='401'):    # the key is required
            g.ask_http({'names': {}}, NAMES, s.url, None, s.alias, tries=1)
        port, proc = s.port, s.proc
    assert proc.poll() is not None and not listening(port)
    # refused and stopped: no key needed, any key accepted, another model,
    # a config file applied
    for env, val, msg in (('STUB_NOAUTH', '1', 'without our key'),
                          ('STUB_ANYKEY', '1', 'without our key'),
                          ('STUB_ALIAS', 'other', 'pinned decision model'),
                          ('STUB_CONFIG', '1', 'config file')):
        monkeypatch.setenv(env, val)
        srv = stub('kev-4b-q8_0')
        with pytest.raises(SystemExit, match=msg):
            with srv:
                pass
        assert srv.proc.poll() is not None              # no orphan left
        monkeypatch.delenv(env)
    # a port already taken is never asked
    busy = socket.socket()
    busy.bind(('127.0.0.1', 0))
    busy.listen()
    try:
        with pytest.raises(SystemExit, match='already in use'):
            with dl.Server('kev-4b-q8_0', port=busy.getsockname()[1],
                           cmd=STUB):
                pass
    finally:
        busy.close()


def test_the_server_never_inherits_llama_settings(home, monkeypatch):
    monkeypatch.setenv('LLAMA_ARG_CTX_SIZE', '99')
    monkeypatch.setenv('LLAMA_ARG_N_PARALLEL', '7')
    seen = {}
    real = dl.subprocess.Popen

    def spy(cmd, **kw):
        seen.update(kw['env'])
        return real(cmd, **kw)
    monkeypatch.setattr(dl.subprocess, 'Popen', spy)
    with stub('kev-4b-q8_0') as s:
        assert seen['LLAMA_API_KEY'] == s.key
    assert not [k for k in seen if k.startswith('LLAMA_') and
                k != 'LLAMA_API_KEY']
    assert seen['XDG_CONFIG_HOME'] == seen['APPDATA'] == \
        seen['PROGRAMDATA'] == str(home / 'empty-config')


# --------------------------------------------------------------- tonight --

def test_tonight_is_logged_once_per_backend(tmp_path, home):
    st = {'names': {'MU': {'earnings_tonight': True}, 'STX': {}, 'INTC': {}}}
    tonight = lambda t: ('20261013', NAMES, st)
    path = str(tmp_path / 'guard_log.csv')
    assert dl.run('kev-4b-q8_0', engine='local', server=stub, tonight=tonight, path=path,
                  now=NOW) == ['rule', 'http']
    log = g.read_log(path)
    assert len(log) == 2 * (len(NAMES) + 1)
    r = log[log.backend == 'rule'].set_index('ticker').decision
    assert r['MU'] == 'skip' and r['STX'] == 'hold'      # the earnings rule
    h = log[log.backend == 'http']
    assert set(h.model) == {dl.model_name('kev-4b-q8_0')}
    assert log.state_sha.nunique() == 1                 # one state, both
    # a second run the same night asks nothing and logs nothing
    assert dl.run('kev-4b-q8_0', engine='local', server=stub, tonight=tonight, path=path,
                  now=NOW) == []
    assert len(g.read_log(path)) == len(log)
    # a dry run asks but never writes
    dl.run('kev-4b-q8_0', engine='local', dry_run=True, server=stub, tonight=tonight,
           path=str(tmp_path / 'dry.csv'), now=NOW)
    assert not os.path.exists(tmp_path / 'dry.csv')
    # past the cutoff nothing is attempted
    with pytest.raises(SystemExit, match='cutoff'):
        dl.run('kev-4b-q8_0', engine='local', server=stub, tonight=tonight,
               path=str(tmp_path / 'late.csv'),
               now=NOW.replace(hour=15, minute=50))


def test_a_failing_decider_never_costs_the_controls_row(tmp_path, home,
                                                       monkeypatch):
    st = {'names': {n: {} for n in NAMES}}
    path = str(tmp_path / 'g.csv')
    monkeypatch.setenv('STUB_MODEL', 'some-other-model')    # wrong label
    assert dl.run('kev-4b-q8_0', engine='local', server=stub, tonight=lambda t: (
        '20261013', NAMES, st), path=path, now=NOW) == ['rule']
    assert set(g.read_log(path).backend) == {'rule'}
    # the catch-up run asks only the decider, about the SAME cached state
    monkeypatch.delenv('STUB_MODEL')
    other = {'names': {n: {'earnings_tonight': True} for n in NAMES}}
    assert dl.run('kev-4b-q8_0', engine='local', server=stub, tonight=lambda t: (
        '20261013', NAMES, other), path=path, now=NOW) == ['http']
    assert g.read_log(path).state_sha.nunique() == 1


def test_the_local_engine_is_not_the_declared_record(tmp_path, home,
                                                     monkeypatch):
    """DECLARED (k calls, a wording, a view, a provider pin) is the hosted
    decider's handling. Kev on --engine local is no declared record: one
    canonical call, logged under its own name with no suffix."""
    monkeypatch.setitem(dl.DECLARED, 'http', dict(
        dl.DECLARED['http'], k=3, version='q2', views='s3'))
    calls = []
    real = g.ask_http

    def spy(st, nm, *a, **kw):
        calls.append(kw)
        return real(st, nm, *a, **kw)
    monkeypatch.setattr(g, 'ask_http', spy)
    st = {'names': {n: {} for n in NAMES}}
    path = str(tmp_path / 'g.csv')
    assert dl.run('kev-4b-q8_0', engine='local', server=stub,
                  tonight=lambda t: ('20261013', NAMES, st), path=path,
                  now=NOW) == ['rule', 'http']
    assert len(calls) == 1
    assert not {'order', 'version', 'view', 'provider'} & set(calls[0])
    log = g.read_log(path)
    assert set(log[log.backend == 'http'].model) == \
        {dl.model_name('kev-4b-q8_0')}


def _px(names, idx):
    rng = np.random.default_rng(0)
    cols = pd.MultiIndex.from_product([names, ['Open', 'Close']])
    return pd.DataFrame(100 * np.exp(rng.normal(0, .02, (len(idx), len(cols)))
                                     .cumsum(0)), index=idx, columns=cols)


def test_tonight_refuses_a_partial_picture(tmp_path, monkeypatch):
    paper = tmp_path / 'xsec_paper.csv'
    pd.DataFrame({'month': ['2026-09'], 'q5': [';'.join(NAMES)]}) \
        .to_csv(paper, index=False)
    monkeypatch.setattr(g, 'PAPER', str(paper))
    idx = pd.bdate_range('2026-07-01', '2026-10-12')
    full = _px(NAMES + ['QQQ'], idx)
    no_ern = lambda names, today: {n: False for n in names}
    # the 1st, before this month's basket is registered: last month's names
    # are not tonight's book
    with pytest.raises(SystemExit, match='no basket registered for 2026-10'):
        g.tonight(NOW, download=lambda *a, **k: full, earnings=no_ern)
    pd.DataFrame({'month': ['2026-09', '2026-10'],
                  'q5': ['A;B', ';'.join(NAMES)]}).to_csv(paper, index=False)
    today, names, st = g.tonight(NOW, download=lambda *a, **k: full,
                                 earnings=no_ern)
    assert names == NAMES and set(st['names']) == set(NAMES)
    # a name (or QQQ) with no data -- e.g. no network yet after a wake
    for drop in ('STX', 'QQQ'):
        part = full.drop(columns=drop, level=0)
        with pytest.raises(SystemExit, match=f'no market data for {drop}'):
            g.tonight(NOW, download=lambda *a, **k: part, earnings=no_ern)


def test_an_unknown_earnings_calendar_is_refused_not_read_as_none():
    calls = []

    def flaky(n):
        calls.append(n)
        if n == 'STX':
            raise ConnectionError('429')
        return pd.DataFrame(index=pd.DatetimeIndex(
            ['2026-10-13 16:05'], tz='America/New_York'))
    with pytest.raises(SystemExit, match=r'STX \(429\)'):
        g.earnings_tonight(NAMES, '20261013', lookup=flaky, tries=3, wait=0)
    assert calls.count('STX') == 3                      # retried first
    ok = g.earnings_tonight(['MU'], '20261013', lookup=flaky, wait=0)
    assert ok == {'MU': True}                           # 16:05 tonight
    with pytest.raises(SystemExit, match='no calendar'):
        g.earnings_tonight(['MU'], '20261013', lookup=lambda n: None, wait=0)


def test_an_answer_that_arrives_after_the_cutoff_is_not_logged(tmp_path,
                                                               monkeypatch):
    path = str(tmp_path / 'g.csv')
    slow = lambda s, n: ({'MU': .9, g.BOOK: .1}, 'm', 1)
    # a 31-minute answer, seen by either clock: monotonic time, or wall time
    # (monotonic stops while a machine sleeps)
    for mono, wall in (([0.0, 31 * 60.0], [0.0, 1.0]),
                       ([0.0, 1.0], [0.0, 31 * 60.0])):
        m, w = iter(mono), iter(wall)
        monkeypatch.setattr(g.time, 'monotonic', lambda: next(m))
        monkeypatch.setattr(g.time, 'time', lambda: next(w))
        with pytest.raises(SystemExit, match='arrived 15:51'):
            g.decide('http', ['MU'], '20261013',
                     dt.datetime(2026, 10, 13, 15, 20, tzinfo=g.ET), {},
                     path, slow)
    assert not os.path.exists(path)


def test_guard_cli_no_longer_decides(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['guard.py', '--decide'])
    with pytest.raises(SystemExit, match='decide_local'):
        g.main()


# ------------------------------------------------------------------- git --

def _git(*a, cwd):
    subprocess.run(['git', *a], cwd=cwd, check=True, capture_output=True)


def _repos(tmp_path):
    remote, seed, mine, other = (tmp_path / n for n in
                                 ('remote.git', 'seed', 'mine', 'other'))
    _git('init', '-q', '--bare', str(remote), cwd=tmp_path)
    _git('init', '-q', str(seed), cwd=tmp_path)
    (seed / 'reports').mkdir()
    (seed / 'reports' / 'guard_log.csv').write_text('date\n')
    (seed / 'README').write_text('x\n')
    _git('add', '.', cwd=seed)
    _git('-c', 'user.email=t@example.com', '-c', 'user.name=t', 'commit',
         '-qm', 'base', cwd=seed)
    _git('push', '-q', str(remote), 'HEAD:refs/heads/main', cwd=seed)
    _git('symbolic-ref', 'HEAD', 'refs/heads/main', cwd=remote)
    for d in (mine, other):                 # clones track origin/main
        _git('clone', '-q', str(remote), str(d), cwd=tmp_path)
        _git('config', 'user.email', 't@example.com', cwd=d)
        _git('config', 'user.name', 't', cwd=d)
    return remote, mine, other


def _last(remote):
    return subprocess.run(['git', '--git-dir', str(remote), 'log', '-1',
                           '--name-only', '--format=%s'],
                          capture_output=True, text=True).stdout


def test_publish_commits_only_the_log_and_survives_a_moved_remote(tmp_path,
                                                                 home):
    remote, mine, other = _repos(tmp_path)
    (other / 'README').write_text('y\n')        # the remote moves on
    _git('commit', '-qam', 'elsewhere', cwd=other)
    _git('push', '-q', cwd=other)
    (mine / 'reports' / 'guard_log.csv').write_text('date\n20261013\n')
    (mine / 'notes.txt').write_text('private\n')
    _git('add', 'notes.txt', cwd=mine)          # staged, must NOT go out
    assert dl.publish(mine, 'guard: test')
    shown = _last(remote)
    assert 'guard: test' in shown and 'reports/guard_log.csv' in shown
    assert 'notes.txt' not in shown
    assert dl.publish(mine, 'guard: again')     # nothing new: still fine
    # an earlier same-day commit whose push failed goes out on the next run
    (mine / 'reports' / 'guard_log.csv').write_text('date\n20261013\n1\n')
    _git('commit', '-qm', 'guard: unpushed', '--', 'reports/guard_log.csv',
         cwd=mine)
    assert dl.publish(mine, 'guard: retry')
    assert 'guard: unpushed' in _last(remote)


def test_a_night_not_pushed_before_its_close_is_dropped(tmp_path, home):
    remote, mine, other = _repos(tmp_path)
    log = mine / 'reports' / 'guard_log.csv'
    log.write_text('date\n20261012,a\n')        # yesterday, never pushed
    _git('commit', '-qm', 'guard: yesterday', '--', 'reports/guard_log.csv',
         cwd=mine)
    dl.sync(mine, '20261013')
    assert log.read_text() == 'date\n'          # dropped, not pushed late
    assert 'base' in _last(remote)
    log.write_text('date\n20261013,a\n')        # today's stays, goes out
    _git('commit', '-qm', 'guard: today', '--', 'reports/guard_log.csv',
         cwd=mine)
    dl.sync(mine, '20261013')
    assert '20261013' in log.read_text()
    assert dl.publish(mine, 'x') and 'guard: today' in _last(remote)


def test_uncommitted_rows_of_an_earlier_night_are_dropped(tmp_path, home):
    remote, mine, other = _repos(tmp_path)
    log = mine / 'reports' / 'guard_log.csv'
    # a run killed after deciding but before committing, yesterday and today
    log.write_text('date\n20261012,a\n20261013,b\n')
    dl.sync(mine, '20261013')
    assert log.read_text() == 'date\n20261013,b\n'     # today's kept
    assert 'dropped uncommitted decisions for 20261012' in \
        (home / 'decider.log').read_text()


def test_a_real_run_only_from_the_dedicated_clone(home, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['decide_local.py'])
    with pytest.raises(SystemExit, match='dedicated clone'):
        dl.main()
    assert 'stopped: real decisions' in (home / 'decider.log').read_text()


def test_a_scheduled_run_finds_its_home_without_any_environment(tmp_path,
                                                                monkeypatch):
    monkeypatch.delenv('QQQ_DECIDER_HOME', raising=False)
    h = tmp_path / 'custom place'
    (h / 'repo').mkdir(parents=True)
    (h / dl.MARK).write_text('x')
    monkeypatch.setattr(dl, 'ROOT', h / 'repo')
    assert dl.home() == h and dl.dedicated() == (h / 'repo').resolve()


# -------------------------------------------------------------- schedule --

@pytest.mark.skipif(not hasattr(time, 'tzset'), reason='POSIX time zones')
def test_the_schedule_lands_on_et_weekdays_from_any_time_zone(monkeypatch):
    ref = dt.date(2026, 10, 13)
    try:
        monkeypatch.setenv('TZ', 'Asia/Riyadh')           # UTC+3, no DST
        time.tzset()
        assert dl.local_trigger('09:17', ref) == (
            '16:17', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
        monkeypatch.setenv('TZ', 'Pacific/Kiritimati')    # UTC+14
        time.tzset()
        hhmm, days = dl.local_trigger('09:17', ref)
        assert hhmm == '03:17' and days == ['Tuesday', 'Wednesday',
                                            'Thursday', 'Friday', 'Saturday']
        ps = dl.windows_task(ref=ref)
        assert "-DaysOfWeek Tuesday,Wednesday,Thursday,Friday,Saturday " \
               "-At '03:17'" in ps
    finally:
        monkeypatch.delenv('TZ')
        time.tzset()


def test_the_windows_task_wakes_runs_on_battery_and_holds_no_path():
    ps = dl.windows_task()
    for flag in ('-WakeToRun', '-StartWhenAvailable',
                 '-AllowStartIfOnBatteries', '-DontStopIfGoingOnBatteries',
                 '-MultipleInstances IgnoreNew', '@($t0,$t1)'):
        assert flag in ps
    # paths and arguments reach PowerShell only through the environment
    assert '$env:QQQ_PY' in ps and '$env:QQQ_SCRIPT' in ps
    assert '$env:QQQ_ARGS' in ps
    assert ':\\' not in ps and '/home' not in ps


def test_systemd_units_quote_paths_and_run_twice_a_day():
    service, timer = dl.systemd_units('/opt/py 3/python', '/r/a "b" 50%/x.py',
                                      '/r/50%')
    assert 'ExecStart="/opt/py 3/python" "/r/a \\"b\\" 50%%/x.py"\n' in service
    # the hosted engine's key comes from an optional user env file
    assert 'EnvironmentFile=-%h/.config/qqq-decider.env' in service
    local, _ = dl.systemd_units('/p', '/s.py', '/r', args=('--engine', 'local'))
    assert 'ExecStart="/p" "/s.py" "--engine" "local"' in local
    assert 'WorkingDirectory=/r/50%%' in service
    assert 'OnCalendar=Mon..Fri 09:17 America/New_York' in timer
    assert 'OnCalendar=Mon..Fri 13:47 America/New_York' in timer
    assert 'Persistent=true' in timer
