# Part of qqq-microstructure.
#
# Tonight's SHADOW decisions (guard.py, RESULTS 27), made entirely on this
# machine: no hosted API, no cloud runner. The model, its server and the
# questions never leave 127.0.0.1; market data comes in from Yahoo (as
# everywhere in this repo) and the answers go out by git push.
#
# The decider is Kev 1.0 (github.com/jaredpalmer/kev, Apache-2.0 code and
# weights), an open-weight Jev lookalike, in ggml-org's GGUF conversion,
# served by llama.cpp's own /v1/systemone server -- one native program, no
# Python ML stack. Checked against Kev's own reference server before it was
# declared (RESULTS 27): the same answers to within 0.015.
#
#   python src/decide_local.py --setup      once: own clone, download, verify
#   python src/decide_local.py --schedule   once: weekday task (Windows) or
#                                           systemd timer (Linux)
#   python src/decide_local.py --dry-run    any time: ask and print, write
#                                           nothing
#   python src/decide_local.py              what the schedule runs
#
# Every run checks, before anything is asked:
#   - the model file and the llama.cpp release against pinned sha256 sums;
#   - that the server it started is the pinned decision model, refuses a
#     missing or wrong key, and was shaped by no config file or LLAMA_*
#     variable (llama-server reads both silently);
#   - that the answer comes back under the pinned name, which carries the
#     file, its revision, the llama.cpp build and the platform -- a different
#     build is a different decider and starts its own record.
#
# The schedule decides from a DEDICATED clone (<home>/repo), never from your
# working copy: it pulls, decides, commits only reports/guard_log.csv and
# pushes. A night's decisions count only if they reach the remote before that
# night's close; any left unpushed from an earlier night are dropped.
#
# Speed: the server runs 16 parallel slots, so llama.cpp reads the ~650-token
# state once per request and shares it across the questions (with one slot it
# re-reads it for every question: 532 s instead of 85 s on 4 cloud vCPUs, and
# no closer to Kev's reference -- RESULTS 27). The schedule tries at 09:17 ET
# and again at 13:47 ET (a sleeping PC, no network yet, a basket registered
# late); a night already decided is never asked again, and an answer that
# arrives after the 15:50 ET cutoff is not logged.

import os, re, sys, json, time, shutil, socket, secrets, hashlib, \
    platform, argparse, subprocess, tarfile, zipfile, datetime as dt
import urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

LLAMA_TAG = 'b11429'                    # llama.cpp 0.6.0: first with Kev
LLAMA = {                               # platform -> (asset, sha256, label)
    'linux-x86_64': ('llama-b11429-bin-ubuntu-x64.tar.gz',
                     'f6d25dde8f51133143d1453da4fd5f73'
                     'b145127177612a283bf7995957af3392', 'linux-x64'),
    'windows-amd64': ('llama-b11429-bin-win-cpu-x64.zip',
                      '1283323272b04cd07905816a597a0da8'
                      '10918102de958f4ff6f7bbaa70ed2efe', 'win-x64'),
}
LLAMA_URL = ('https://github.com/ggml-org/llama.cpp/releases/download/'
             f'{LLAMA_TAG}/' + '{}')

# name -> (Hugging Face repo, revision, file, sha256, bytes). Kev 1.0 4B,
# converted by ggml-org from jaredpalmer/kev-4b, files identical to its v1.0
# tag (Qwen3.5-4B-Base; temperature 2.40605 as in Kev's head.pt).
MODELS = {
    'kev-4b-q8_0': ('ggml-org/Kev-4B-GGUF',
                    'd924f2e2c3872da8b8aaf3eb4453b4126deceb79',
                    'Kev-4B-Q8_0.gguf',
                    '7c2ebed90560522c2801389db482ac1d'
                    'c4c36d828f201f6074c1d60e433948da', 4483801504),
    'kev-4b-bf16': ('ggml-org/Kev-4B-GGUF',
                    'd924f2e2c3872da8b8aaf3eb4453b4126deceb79',
                    'Kev-4B-BF16.gguf',
                    'c690ee43d8244f7cd7dd960aebaadda3'
                    '666e68bd69aba1af9135737b4f0595d9', 8427020704),
}
DEFAULT_MODEL = 'kev-4b-q8_0'
HF_URL = 'https://huggingface.co/{}/resolve/{}/{}'

NP = 16                     # parallel slots: questions share the state's
                            # prefill (a basket of up to 15 names + the hedge)
CTX = 16384                 # the slots' shared KV pool (~750 tokens each)
TIMEOUT = 3 * 3600          # one request of 12 questions, worst case on a CPU
TASK = 'qqq-guard-decider'
RUNS_ET = ('09:17', '13:47')            # a try, and a catch-up
MARK = '.qqq-decider-home'
LOG = 'reports/guard_log.csv'
# the local server is reached directly: a system proxy must never see the
# key or the state (urllib would otherwise route 127.0.0.1 through it)
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def home():
    """QQQ_DECIDER_HOME if set; else, when this file runs from a dedicated
    clone, that clone's home (so a scheduled run needs no environment); else
    ~/qqq-decider."""
    if os.environ.get('QQQ_DECIDER_HOME'):
        return Path(os.environ['QQQ_DECIDER_HOME'])
    if (ROOT.parent / MARK).exists():
        return ROOT.parent
    return Path.home() / 'qqq-decider'


def dedicated():
    return (home() / 'repo').resolve()


def plat():
    k = f'{platform.system().lower()}-{platform.machine().lower()}'
    if k not in LLAMA:
        raise SystemExit(f'no pinned llama.cpp build for {k}; pinned: '
                         f'{", ".join(LLAMA)}')
    return k


def model_name(m, p=None):
    """What guard.py logs as the decider: the weights, the runtime, the
    platform build."""
    repo, rev, *_ = MODELS[m]
    return f'{m}@{rev[:8]}+llama.cpp-{LLAMA_TAG}-{LLAMA[p or plat()][2]}'


def redact(s):
    return re.sub(r'//[^/@\s]+@', '//***@', str(s))


def say(*a):
    line = redact(' '.join(str(x) for x in a))
    print(line, flush=True)
    try:
        home().mkdir(parents=True, exist_ok=True)
        with open(home() / 'decider.log', 'a', encoding='utf-8') as f:
            f.write(f'{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {line}\n')
    except OSError:
        pass


# --------------------------------------------------------- files and pins --

def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def fetch(url, dest, sha, opener=urllib.request.urlopen):
    """Download to dest, verified by sha256; a file already there that
    matches is kept. A mismatch is deleted, never used."""
    dest = Path(dest)
    if dest.exists():
        if sha256(dest) == sha:
            return dest
        say(f'{dest.name}: present but its sha256 does not match -- '
            f'downloading again')
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    say(f'downloading {url}')
    with opener(url) as r, open(part, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        got, step = 0, max(total // 10, 1)
        for b in iter(lambda: r.read(1 << 20), b''):
            f.write(b)
            got += len(b)
            if total and got // step != (got - len(b)) // step:
                print(f'  {100 * got // total}%', flush=True)
    if sha256(part) != sha:
        part.unlink()
        raise SystemExit(f'{dest.name}: sha256 mismatch -- deleted, not used')
    os.replace(part, dest)
    return dest


def llama_dir():
    return home() / 'llama' / LLAMA_TAG


def archive():
    return home() / 'downloads' / LLAMA[plat()][0]


def members(arc):
    """(relative path, kind, payload) for every entry of the pinned archive:
    kind 'file' with its bytes' sha256, or 'link' with its target."""
    if str(arc).endswith('.zip'):
        with zipfile.ZipFile(arc) as z:
            for i in z.infolist():
                if not i.is_dir():
                    yield i.filename, 'file', hashlib.sha256(
                        z.read(i)).hexdigest()
    else:
        with tarfile.open(arc) as t:
            for m in t.getmembers():
                if m.isfile():
                    yield m.name, 'file', hashlib.sha256(
                        t.extractfile(m).read()).hexdigest()
                elif m.issym():
                    yield m.name, 'link', m.linkname


def check_llama():
    """The archive matches its pin, and every file extracted from it still
    matches the archive -- run at setup and before every night."""
    arc = archive()
    if not arc.exists() or sha256(arc) != LLAMA[plat()][1]:
        raise SystemExit('the llama.cpp archive is missing or does not match '
                         'its pin -- run --setup')
    for name, kind, val in members(arc):
        f = llama_dir() / name
        ok = (f.is_symlink() and os.readlink(f) == val) if kind == 'link' \
            else (f.is_file() and not f.is_symlink() and sha256(f) == val)
        if not ok:
            raise SystemExit(f'llama.cpp file {name} differs from the pinned '
                             f'release -- run --setup')


def check_model(m):
    f = model_file(m)
    if not f.exists() or sha256(f) != MODELS[m][3]:
        raise SystemExit(f'{f.name} is missing or does not match its pin -- '
                         f'run --setup')


def server_binary():
    exe = 'llama-server.exe' if os.name == 'nt' else 'llama-server'
    hits = sorted(llama_dir().rglob(exe))
    if not hits:
        raise SystemExit('llama.cpp is not installed -- run '
                         'python src/decide_local.py --setup')
    return hits[0]


def model_file(m):
    return home() / 'models' / MODELS[m][2]


# --------------------------------------------------------------- the repo --

def git(*a, cwd=None, check=True, timeout=300):
    """git that never waits on a person: no terminal or credential-manager
    prompt (an expired login fails fast instead of hanging the night)."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    try:
        r = subprocess.run(['git', *a], cwd=cwd or ROOT, capture_output=True,
                           text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        r = subprocess.CompletedProcess(a, 124, '', 'timed out')
    if check and r.returncode:
        raise SystemExit(f'git {" ".join(a)} failed: '
                         f'{(r.stderr or r.stdout).strip()[:500]}')
    return r


def sync(root, today):
    """Bring the dedicated clone up to date for tonight. Decisions committed
    here but never pushed for an EARLIER night are dropped: they did not
    reach the remote before that night's close, so they cannot count."""
    drop_uncommitted(root, today)
    if git('fetch', '-q', cwd=root, check=False).returncode:
        say('git fetch failed -- deciding on the basket already here')
        return
    diff = git('diff', '@{u}...HEAD', '--', LOG, cwd=root, check=False).stdout
    late = sorted({l[1:9] for l in diff.splitlines()
                   if l.startswith('+') and l[1:9].isdigit()
                   and l[1:9] < today})
    if late:
        git('reset', '-q', '--hard', '@{u}', cwd=root)
        say(f'dropped unpushed decisions for {", ".join(late)}: they never '
            f'reached the remote before their close')
    if git('pull', '-q', '--rebase', '--autostash', cwd=root,
           check=False).returncode:
        git('rebase', '--abort', cwd=root, check=False)
        say('git pull failed -- deciding on the basket already here')


def drop_uncommitted(root, today):
    """Rows written but never committed (a run killed between deciding and
    publishing) are dropped if they belong to an earlier night -- pushed now
    they would reach the remote after that night's close. Today's are kept."""
    f = Path(root) / LOG
    r = git('show', f'HEAD:{LOG}', cwd=root, check=False)
    if r.returncode or not f.exists():
        return
    committed = r.stdout.splitlines()
    rows = f.read_text().splitlines()
    if rows[:len(committed)] != committed:
        return                  # not an append to the committed log: leave
    new = rows[len(committed):]
    late = sorted({l[:8] for l in new if l[:8].isdigit() and l[:8] < today})
    if late:
        keep = [l for l in new if not (l[:8].isdigit() and l[:8] < today)]
        with open(f, 'w', newline='') as out:
            out.write(''.join(l + '\n' for l in committed + keep))
        say(f'dropped uncommitted decisions for {", ".join(late)}: they '
            f'never reached the remote before their close')


def publish(root, message, tries=4):
    """Commit ONLY reports/guard_log.csv in the dedicated clone and push it.
    Returns True when the answers are on the remote."""
    git('add', LOG, cwd=root)
    if git('diff', '--cached', '--quiet', '--', LOG, cwd=root,
           check=False).returncode:
        git('commit', '-q', '-m', message, '--', LOG, cwd=root)
    ahead = git('rev-list', '--count', '@{u}..HEAD', cwd=root, check=False)
    if ahead.returncode == 0 and ahead.stdout.strip() == '0':
        say('nothing new to push')
        return True
    for i in range(tries):
        if git('push', '-q', cwd=root, check=False).returncode == 0:
            say('pushed: the answers are on the remote, time-stamped')
            return True
        # the dedicated clone holds nothing but these commits, so a rebase
        # onto whatever the remote gained (the forward log, code) is safe
        if git('pull', '--rebase', '--autostash', '-q', cwd=root,
               check=False).returncode:
            git('rebase', '--abort', cwd=root, check=False)
        time.sleep(2 ** i)
    say('PUSH FAILED -- the answers are committed in '
        f'{root} but not on the remote; they count only if pushed before '
        f'15:50 ET (the 13:47 catch-up run tries again)')
    return False


# ------------------------------------------------------------- the server --

def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def port_free(port):
    """Nothing is listening there. (A connect test, not a bind test: a port a
    stopped server just used lingers in TIME_WAIT and would read as taken.)"""
    try:
        socket.create_connection(('127.0.0.1', port), timeout=1).close()
        return False
    except OSError:
        return True


def get(url, key=None, timeout=5):
    req = urllib.request.Request(url, headers={'Authorization':
                                               f'Bearer {key}'} if key else {})
    with DIRECT.open(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode() or 'null')


class Server:
    """llama-server on 127.0.0.1, on a random free port, with a fresh random
    key (in its environment, never on its command line), in an environment
    stripped of LLAMA_* variables and pointing its config directories at an
    empty one -- for the length of a `with`. Refused unless it is the pinned
    decision model, rejects a missing or wrong key, and read no config."""

    def __init__(self, m, port=None, threads=None, cmd=None, wait=900):
        self.m, self.port, self.threads, self.wait = m, port, threads, wait
        self.cmd = cmd                    # tests: a stand-in server

    def __enter__(self):
        try:
            return self._start()
        except BaseException:
            self.__exit__()
            raise

    def _start(self):
        self.port = self.port or free_port()
        if not port_free(self.port):
            raise SystemExit(f'127.0.0.1:{self.port} is already in use -- '
                             f'not sending tonight\'s questions to a server '
                             f'this script did not start')
        self.key = secrets.token_hex(24)
        self.alias = model_name(self.m)
        cmd = self.cmd or [str(server_binary()), '-m',
                           str(model_file(self.m))]
        cmd = cmd + ['--alias', self.alias, '--host', '127.0.0.1', '--port',
                     str(self.port), '-c', str(CTX), '-np', str(NP)]
        if self.threads:
            cmd += ['-t', str(self.threads)]
        empty = home() / 'empty-config'
        empty.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items()
               if not k.upper().startswith('LLAMA_')}
        env.update(LLAMA_API_KEY=self.key, XDG_CONFIG_HOME=str(empty),
                   APPDATA=str(empty), PROGRAMDATA=str(empty))
        logp = home() / 'llama-server.log'
        self.offset = logp.stat().st_size if logp.exists() else 0
        self.out = open(logp, 'ab')
        self.proc = subprocess.Popen(cmd, stdout=self.out,
                                     stderr=subprocess.STDOUT, env=env)
        base = f'http://127.0.0.1:{self.port}'
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise SystemExit(f'llama-server exited ({self.proc.returncode})'
                                 f' -- see {logp}')
            try:
                if get(base + '/health')[0] == 200:
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.time() - t0 > self.wait:
                raise SystemExit('llama-server did not come up in time')
            time.sleep(1)
        for bad in (None, secrets.token_hex(24)):   # no key; a wrong key
            try:
                get(base + '/v1/models', bad)
            except urllib.error.HTTPError as ex:
                if ex.code == 401:
                    continue
                raise SystemExit(f'unexpected /v1/models status {ex.code}')
            raise SystemExit('the server answered without our key -- refusing')
        _, models = get(base + '/v1/models', self.key)
        if not [d for d in (models or {}).get('data', [])
                if d.get('id') == self.alias and 'decisions' in
                d.get('architecture', {}).get('output_modalities', [])]:
            raise SystemExit('the server is not serving the pinned decision '
                             'model')
        self.out.flush()
        with open(logp, 'rb') as f:
            f.seek(self.offset)
            if b'using config file' in f.read():
                raise SystemExit('llama-server applied a config file (see its '
                                 'log) -- the pinned decider must not be '
                                 'reshaped; remove it')
        if self.proc.poll() is not None:
            raise SystemExit('llama-server exited during the checks')
        self.url = base + '/v1/systemone'
        return self

    def __exit__(self, *exc):
        p = getattr(self, 'proc', None)
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        if getattr(self, 'out', None) and not self.out.closed:
            self.out.close()
        return False


# ---------------------------------------------------------------- tonight --

def night_state(today, build, t, cache=True):
    """Tonight's (date, names, state), built once per night and kept, so a
    catch-up run asks the decider about exactly what the control saw."""
    f = home() / 'state' / f'{today}.json'
    if cache and f.exists():
        d = json.loads(f.read_text())
        return d['date'], d['names'], d['state']
    date, names, state = build(t)
    if cache:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({'date': date, 'names': names,
                                 'state': state}))
    return date, names, state


def run(m, dry_run=False, threads=None, server=Server, tonight=None,
        path=None, now=None):
    """Decide tonight with the control (earnings rule) and the decider (Kev),
    or with --dry-run just ask and print. Returns the backends that logged.
    A failure of one backend never stops the other from being logged.
    `now` pins the clock (tests); left out, every answer is stamped with the
    real time it arrived."""
    import guard
    path = path or guard.GLOG
    t = now or dt.datetime.now(guard.ET)
    today = t.strftime('%Y%m%d')
    cutoff = t.replace(hour=guard.CUTOFF[0], minute=guard.CUTOFF[1],
                       second=0, microsecond=0)
    if not dry_run:
        if t >= cutoff:
            raise SystemExit(f'{t:%H:%M} ET is past the cutoff -- nothing '
                             f'to decide tonight')
        log = guard.read_log(path)
        todo = [b for b in ('rule', 'http')
                if not ((log.date == today) & (log.backend == b)).any()]
        if not todo:
            say(f'{today}: both backends already decided')
            return []
    else:
        todo = ['rule', 'http']
    if server is Server:
        check_model(m)
        check_llama()
    today, names, state = night_state(today, tonight or guard.tonight, t,
                                      cache=not dry_run)
    done = []
    if 'rule' in todo:
        try:
            guard.answer('rule', guard.ask_rule, names, today, state,
                         dry_run=dry_run, path=path, now=now)
            done.append('rule')
        except SystemExit as ex:
            say(f'rule: {ex}')
    if 'http' in todo:
        try:
            with server(m, threads=threads) as s:
                left = (cutoff - (now or dt.datetime.now(guard.ET))) \
                    .total_seconds() - 60
                if not dry_run and left <= 0:
                    raise SystemExit('too close to the cutoff to ask')

                def ask(st, nm):
                    p, model, ms = guard.ask_http(
                        st, nm, s.url, s.key, s.alias, tries=1,
                        opener=DIRECT.open,
                        timeout=TIMEOUT if dry_run else min(TIMEOUT, left))
                    if model != s.alias:
                        raise SystemExit(f'the server answered as {model!r}, '
                                         f'not the pinned {s.alias!r}')
                    return p, model, ms
                guard.answer('http', ask, names, today, state,
                             dry_run=dry_run, path=path, now=now)
                done.append('http')
        except SystemExit as ex:
            say(f'http: {ex}')
        except Exception as ex:                 # never lose the control's row
            say(f'http: {type(ex).__name__}: {ex}')
    return done


# --------------------------------------------------------------- schedule --

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
        'Sunday']


def local_trigger(at_et, ref=None):
    """The local wall time and weekdays at which a weekday ET run falls, for
    a scheduler that only knows local time. Returns ('HH:MM', [weekdays])."""
    import guard
    h, mi = map(int, at_et.split(':'))
    ref = ref or dt.date.today()
    monday = ref + dt.timedelta(days=(7 - ref.weekday()) % 7 or 7)
    days, hhmm = [], None
    for i in range(5):
        t = dt.datetime.combine(monday + dt.timedelta(days=i),
                                dt.time(h, mi), tzinfo=guard.ET).astimezone()
        days.append(DAYS[t.weekday()])
        hhmm = hhmm or f'{t:%H:%M}'
    return hhmm, days


def windows_task(runs_et=RUNS_ET, ref=None):
    """The PowerShell that registers the task. Paths travel in environment
    variables (QQQ_PY, QQQ_SCRIPT, QQQ_REPO), never inside the script text,
    so no quote in a path can break or rewrite it."""
    trig = []
    for i, at in enumerate(runs_et):
        hhmm, days = local_trigger(at, ref)
        trig.append(f"$t{i} = New-ScheduledTaskTrigger -Weekly -DaysOfWeek "
                    f"{','.join(days)} -At '{hhmm}'")
    return '\n'.join([
        "$ErrorActionPreference = 'Stop'",
        "$a = New-ScheduledTaskAction -Execute $env:QQQ_PY "
        "-Argument ('\"' + $env:QQQ_SCRIPT + '\"') "
        "-WorkingDirectory $env:QQQ_REPO",
        *trig,
        "$s = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable "
        "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
        "-ExecutionTimeLimit (New-TimeSpan -Hours 4) "
        "-MultipleInstances IgnoreNew",
        f"Register-ScheduledTask -TaskName '{TASK}' -Action $a -Trigger "
        f"@({','.join(f'$t{i}' for i in range(len(runs_et)))}) -Settings $s "
        f"-Description 'qqq shadow decider: Kev, local' -Force | Out-Null",
    ])


def systemd_quote(s):
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"') \
        .replace('%', '%%') + '"'


def systemd_units(python, script, repo, runs_et=RUNS_ET):
    service = '\n'.join([
        '[Unit]', 'Description=qqq shadow decider (Kev, local)', '',
        '[Service]', 'Type=oneshot',
        f'WorkingDirectory={str(repo).replace("%", "%%")}',
        f'ExecStart={systemd_quote(python)} {systemd_quote(script)}',
        'TimeoutStartSec=4h', 'Nice=10', ''])
    timer = '\n'.join([
        '[Unit]', 'Description=qqq shadow decider, weekdays before the close',
        '', '[Timer]',
        *[f'OnCalendar=Mon..Fri {at} America/New_York' for at in runs_et],
        'Persistent=true', '', '[Install]', 'WantedBy=timers.target', ''])
    return service, timer


def schedule(runs_et=RUNS_ET, remove=False):
    repo = dedicated()
    script = repo / 'src' / 'decide_local.py'
    if not script.exists() and not remove:
        raise SystemExit('run --setup first (the schedule runs the dedicated '
                         'clone)')
    python = sys.executable
    if os.name == 'nt':
        ps = (f"Unregister-ScheduledTask -TaskName '{TASK}' -Confirm:$false"
              if remove else windows_task(runs_et))
        r = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy',
                            'Bypass', '-Command', ps], capture_output=True,
                           text=True, env=dict(os.environ, QQQ_PY=python,
                                               QQQ_SCRIPT=str(script),
                                               QQQ_REPO=str(repo)))
        if r.returncode:
            raise SystemExit(f'Task Scheduler refused: {r.stderr.strip()}')
        if remove:
            say(f'removed the scheduled task {TASK}')
            return
        when = '; '.join(f'{local_trigger(at)[0]} local (= {at} ET)'
                         for at in runs_et)
        say(f'scheduled {TASK} on weekdays: {when}. It wakes the PC and runs '
            f'late if the PC was off; it runs while you are signed in (a '
            f'locked screen is fine). The local time stays fixed when US '
            f'clocks change, so the ET time moves by an hour -- still hours '
            f'before the 15:50 ET cutoff.')
        return
    d = Path.home() / '.config' / 'systemd' / 'user'
    if remove:
        subprocess.run(['systemctl', '--user', 'disable', '--now',
                        f'{TASK}.timer'], check=False)
        for ext in ('service', 'timer'):
            (d / f'{TASK}.{ext}').unlink(missing_ok=True)
        say(f'removed the {TASK} timer')
        return
    service, timer = systemd_units(python, script, repo, runs_et)
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{TASK}.service').write_text(service)
    (d / f'{TASK}.timer').write_text(timer)
    if shutil.which('systemctl'):
        for a in (['daemon-reload'], ['enable', '--now', f'{TASK}.timer']):
            r = subprocess.run(['systemctl', '--user', *a], capture_output=True,
                               text=True)
            if r.returncode:
                raise SystemExit(f'systemctl --user {" ".join(a)}: '
                                 f'{r.stderr.strip()}')
        say(f'timer {TASK}: Mon..Fri {" and ".join(runs_et)} '
            f'America/New_York. For it to run while you are logged out: '
            f'sudo loginctl enable-linger $USER')
    else:
        say(f'no systemctl here; units written to {d}. Or use cron (cronie '
            f'reads CRON_TZ): CRON_TZ=America/New_York, then for each of '
            f'{", ".join(runs_et)}: MM HH * * 1-5 {python} {script}')


# ------------------------------------------------------------------ setup --

def setup(m):
    h = home()
    h.mkdir(parents=True, exist_ok=True)
    (h / MARK).write_text('qqq-microstructure local decider home\n')
    say(f'decider home: {h}')
    repo = h / 'repo'
    if not (repo / '.git').exists():
        url = git('remote', 'get-url', 'origin').stdout.strip()
        say(f'cloning {url} -> {repo} (the schedule decides from this clone, '
            f'never from your working copy)')
        git('clone', '-q', url, str(repo), cwd=h, timeout=1800)
    else:
        git('pull', '--ff-only', '-q', cwd=repo)
    asset, sha, _ = LLAMA[plat()]
    arc = fetch(LLAMA_URL.format(asset), archive(), sha)
    dest = llama_dir()
    try:
        check_llama()
    except SystemExit:
        tmp = dest.with_name(LLAMA_TAG + '.part')
        shutil.rmtree(tmp, ignore_errors=True)
        if asset.endswith('.zip'):
            with zipfile.ZipFile(arc) as z:
                z.extractall(tmp)
        else:
            with tarfile.open(arc) as t:
                t.extractall(tmp, filter='data')
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
        check_llama()
    exe = server_binary()
    if os.name != 'nt':
        exe.chmod(exe.stat().st_mode | 0o111)
    repo_id, rev, fname, msha, size = MODELS[m]
    free = shutil.disk_usage(h).free
    if not model_file(m).exists() and free < size * 1.1:
        raise SystemExit(f'{fname} needs {size / 1e9:.1f} GB; '
                         f'{free / 1e9:.1f} GB free at {h}')
    fetch(HF_URL.format(repo_id, rev, fname), model_file(m), msha)
    say(f'ready: {exe.name} ({LLAMA_TAG}) + {fname}, both verified. Next: '
        f'python src/decide_local.py --dry-run, then --schedule')


def nightly(m, threads=None):
    """What the schedule runs, from the dedicated clone."""
    import guard
    if ROOT.resolve() != dedicated():
        raise SystemExit(f'real decisions run from the dedicated clone '
                         f'{dedicated()} (python src/decide_local.py --setup, '
                         f'then --schedule) so they never touch your working '
                         f'copy; here, use --dry-run')
    today = dt.datetime.now(guard.ET).strftime('%Y%m%d')
    sync(ROOT, today)
    run(m, threads=threads)
    # whatever this run, or an earlier one today, logged goes out now
    publish(ROOT, f'guard: decisions before the close (local, '
                  f'{model_name(m)})')


def main():
    ap = argparse.ArgumentParser(description='Tonight\'s shadow decisions, '
                                 'made entirely on this machine (Kev on '
                                 'llama.cpp, 127.0.0.1).')
    ap.add_argument('--setup', action='store_true')
    ap.add_argument('--schedule', action='store_true')
    ap.add_argument('--unschedule', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--model', choices=sorted(MODELS), default=DEFAULT_MODEL)
    ap.add_argument('--threads', type=int)
    a = ap.parse_args()
    try:
        if a.setup:
            setup(a.model)
        elif a.schedule or a.unschedule:
            schedule(remove=a.unschedule)
        elif a.dry_run:
            run(a.model, dry_run=True, threads=a.threads)
        else:
            nightly(a.model, a.threads)
    except SystemExit as ex:
        if ex.code not in (None, 0):
            say(f'stopped: {ex}')           # the log says why a night was
        raise                               # missed, even under a scheduler
    except Exception as ex:
        say(f'FAILED: {type(ex).__name__}: {ex}')
        raise


if __name__ == '__main__':
    main()
