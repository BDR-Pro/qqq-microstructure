# Part of qqq-microstructure.
#
# Tonight's SHADOW decisions (guard.py, RESULTS 27), run on the owner's own
# machine. It never trades.
#
# Two engines. The record's is HOSTED (the owner: "local host or API through
# OpenRouter, it is okay"):
#   - the decider: Perplexity Decider v1.1 27B, the best model on both
#     independent decision boards that can be reached (Decision Index 62.75,
#     Jev 60.11; JevBench intelligence 75.7, Jev 63.6; calibration on par);
#   - the reference: Jev 1.13 itself;
#   - both through OpenRouter's Decisions API (Jev's request shape; the
#     response names the dated build and the provider, which the record
#     logs), with OPENROUTER_API_KEY from this machine's environment;
#   - beside them the earnings-rule control, on the identical state.
# LOCAL is Kev 1.0 (Apache-2.0, an open-weight Jev lookalike) in llama.cpp's
# own /v1/systemone server on 127.0.0.1, no key, no network for the model.
#
#   python src/decide_local.py --setup      once: own clone (and, for
#                                           --engine local, the model)
#   python src/decide_local.py --dry-run    any time: ask and print, write
#                                           nothing
#   python src/decide_local.py --schedule   once: weekday task (Windows) or
#                                           systemd timer (Linux)
#   python src/decide_local.py              what the schedule runs
#   add --engine local to any of these for Kev on this machine
#
# Hosted: every answer must come back from the pinned model (any dated build
# of it; the build and provider name the record). How the decider is asked
# beyond that -- one call or k seeded orders averaged, the wording, a view,
# a provider pin -- is DECLARED beside HOSTED, names its own record suffix,
# and must match guard's gated slot or the decider is not asked (the control
# and the reference still decide, and decider.log says why). Local: the
# model file and the llama.cpp release are checked against pinned sha256 sums
# every night; the server must be the pinned decision model, refuse a missing
# or wrong key, and read no config file or LLAMA_* variable; its name carries
# the file, revision, llama.cpp build and platform.
#
# The schedule decides from a DEDICATED clone (<home>/repo), never from your
# working copy: it pulls, decides, commits only reports/guard_log.csv and
# pushes. A night's decisions count only if they reach the remote before that
# night's close; any left unpushed from an earlier night are dropped. It runs
# at 09:17 ET and again at 13:47 ET (a sleeping PC, no network yet, a basket
# registered late); a night already decided is never asked again, and an
# answer that arrives after the 15:50 ET cutoff is not logged.
#
# Local speed: 16 parallel slots let llama.cpp read the ~650-token state once
# per request (85 s a night on 4 cloud vCPUs; one slot took 532 s).

import os, re, sys, json, time, shutil, socket, secrets, hashlib, \
    platform, argparse, subprocess, tarfile, zipfile, datetime as dt
import urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import guard                            # for the declaration's check below

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
    'kev-9b-q8_0': ('ggml-org/Kev-9B-GGUF',
                    'ec2bbfe6620218aee2e01cc93bb78dee2a96ed58',
                    'Kev-9B-Q8_0.gguf',
                    'd30b225bfdc985d1856bb04a990be760'
                    '08d0cf40dccbf78938e3069744bc614c', 9529735648),
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

# The hosted engine (RESULTS 27): decision models on OpenRouter's Decisions
# API, which takes Jev's request shape and answers with the versioned model
# and the provider. backend in the log -> pinned model id.
DECISIONS = 'https://openrouter.ai/api/alpha/decisions'
HOSTED = {
    'http': 'perplexity/pplx-decider-v1.1-27b',  # the decider: best on both
                                                 # independent boards
    'jev': 'typesafe/jev-1.13',                  # the reference: Jev itself
}
# a record names a dated build: never a moving alias (~...) or a :free route
assert not any('~' in v or ':free' in v for v in HOSTED.values())

# How the record's decider is asked, beyond its pin (RESULTS 27; battery spec
# 5, R3-R5, R7). Today's declaration is the record's: one canonical call in
# the q1 wording, no view, no provider pin. The battery's selection ('k3' is
# k 3) and an iteration the holdout confirmed (a version, views, k 5) change
# these values only -- and guard.DECIDER's suffix with them. The check below
# runs before every decider call, not at import: a half-made change then
# costs the decider's night alone, logged, while the control and the
# reference decide and the nightly pull can still bring the fix:
#   k               1: one canonical call. k > 1: k sequential calls, call j
#                   in the night's seeded order j (the battery's E9 seeds,
#                   so the bytes sent are the bytes it tested), averaged
#   version         a wording in guard.QUESTIONS
#   views           guard.VIEWS tags, sorted ('b2s3'): what is sent is
#                   computed from the shared schema-2 state, whose sha the
#                   log keeps, so every backend still sees one state
#   provider        None, or the one provider every call is pinned to, with
#                   no fallback (it is not in the suffix: the logged
#                   '@provider' already splits the record)
#   dated           the decider is served under dated builds (guard.DECIDER's
#                   third field when False)
#   measured_build  the build the battery measured, or None: another build
#                   answering is a one-line drift note, never a refusal
# Only 'http' takes a handling. The reference ('jev') stays one raw canonical
# call with no suffix; its entry, if any, may name only its measured_build.
# The battery's selection (bench/battery/runs/v1/selection_v1.json, sha256
# fa28c603...; branch 'none': no candidate passed every gate) keeps the
# incumbent at k1.q1 with no provider pin, and names the build it measured.
DECLARED = {'http': {'k': 1, 'version': 'q1', 'views': '', 'provider': None,
                     'dated': True,
                     'measured_build':
                         'perplexity/pplx-decider-v1.1-27b-20261006'}}
HANDLING = ('k', 'version', 'views', 'provider', 'dated')


def suffix(how):
    """The record's handling suffix, guard.handling_suffix's -- the scheme
    the battery names its arms by: '' for one canonical call in q1 with no
    view (every night logged so far), else '#{h}.{version}{views}' where h is
    perm{k} for k seeded orders averaged, or k1 for one canonical call:
    '#perm3.q1', '#perm3.q2s3', '#perm5.q1', '#k1.q2', '#k1.q1b2'."""
    return guard.handling_suffix(how.get('k', 1), how.get('version', 'q1'),
                                 how.get('views', ''))


def declaration_problems(hosted=None, declared=None, decider=None):
    """What is wrong with the declaration (empty: nothing): the handling is
    well formed, only 'http' has one, and guard's gated decider slot is
    exactly (HOSTED pin, its suffix, dated)."""
    hosted = HOSTED if hosted is None else hosted
    declared = DECLARED if declared is None else declared
    decider = guard.DECIDER if decider is None else decider
    out = []
    if 'http' not in hosted or 'http' not in declared:
        return ['no http decider declared']
    out += [f'{b} is declared but not hosted' for b in declared
            if b not in hosted]
    for b, how in declared.items():
        allowed = (*HANDLING, 'measured_build') if b == 'http' \
            else ('measured_build',)
        out += [f'{b}: {x!r} cannot be declared' for x in how
                if x not in allowed]
        mb = how.get('measured_build')
        if mb is not None and not (isinstance(mb, str) and re.fullmatch(
                re.escape(hosted.get(b, '')) + r'(-\d{8})?', mb)):
            out.append(f'{b}: measured_build {mb!r} is not a build of its pin')
    how = declared['http']
    out += [f'http: {x!r} is not declared' for x in HANDLING if x not in how]
    if out:
        return out
    k, w = how['k'], how['views']
    if type(k) is not int or k < 1:
        out.append(f'http: k {k!r} is not a count of calls')
    if how['version'] not in guard.QUESTIONS:
        out.append(f'http: no wording {how["version"]!r} in guard.QUESTIONS')
    tags = re.findall('|'.join(map(re.escape, guard.VIEWS)), w) \
        if isinstance(w, str) else None
    if tags is None or ''.join(tags) != w or tags != sorted(set(tags)):
        out.append(f'http: views {w!r} are not guard.VIEWS tags, each once, '
                   f'sorted')
    p = how['provider']
    if p is not None and not (isinstance(p, str) and p.strip()):
        out.append(f'http: provider {p!r} is not a provider name')
    if type(how['dated']) is not bool:
        out.append('http: dated is not True or False')
    want = (hosted['http'], suffix(how)) + (() if how['dated'] else (False,))
    have = tuple(decider)
    if have[2:] == (True,):
        have = have[:2]             # (pin, suffix, True) is (pin, suffix)
    if have != want:
        out.append(f'guard.DECIDER is {tuple(decider)!r}, the runner sends '
                   f'{want!r} -- change them together')
    return out


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


# key-shaped strings, as the battery scrubs them (battery_run.SECRETS): an
# error body guard.ask_http quotes may echo the Authorization header
SECRETS = (re.compile(r'\bsk-[A-Za-z0-9_-]{8,}'), re.compile(r'Bearer\s+\S*'),
           re.compile(r'\bsk-or-[A-Za-z0-9_-]*'))


def redact(s):
    """A line as it may be printed or written to decider.log: no credential
    in a URL, no key-shaped string, and never this machine's key itself."""
    s = re.sub(r'//[^/@\s]+@', '//***@', str(s))
    key = os.environ.get('OPENROUTER_API_KEY')
    if key and len(key) >= 8:
        s = s.replace(key, '***')
    for r in SECRETS:
        s = r.sub('***', s)
    return s


def say(*a, echo=True):
    line = redact(' '.join(str(x) for x in a))
    if echo:
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
        path=None, now=None, engine='hosted', opener=None, proxy_auth=False):
    """Decide tonight, or with --dry-run just ask and print. Returns the
    backends that logged. A failure of one never stops the others.

    engine 'hosted' (the record's): the control, the decider (Perplexity
    Decider) and the reference (Jev), the last two through OpenRouter's
    Decisions API with OPENROUTER_API_KEY from this machine's environment.
    engine 'local': the control and Kev on a local llama.cpp, no key.

    `now` pins the clock (tests); left out, every answer is stamped with the
    real time it arrived. `proxy_auth`: send no key at all, because this
    machine's network proxy adds it (a cloud environment's network secret;
    measured working 2026-10-13)."""
    import guard
    path = path or guard.GLOG
    t = now or dt.datetime.now(guard.ET)
    today = t.strftime('%Y%m%d')
    cutoff = t.replace(hour=guard.CUTOFF[0], minute=guard.CUTOFF[1],
                       second=0, microsecond=0)
    backends = ['rule'] + (list(HOSTED) if engine == 'hosted' else ['http'])
    if not dry_run:
        if t >= cutoff:
            raise SystemExit(f'{t:%H:%M} ET is past the cutoff -- nothing '
                             f'to decide tonight')
        log = guard.read_log(path)
        todo = [b for b in backends
                if not ((log.date == today) & (log.backend == b)).any()]
        if not todo:
            say(f'{today}: every backend already decided')
            return []
    else:
        todo = backends
    if engine == 'hosted':
        key = os.environ.get('OPENROUTER_API_KEY')
        if not key and not proxy_auth:
            raise SystemExit('set OPENROUTER_API_KEY in this machine\'s '
                             'environment (never in a file in the repo), or '
                             'add --proxy-auth where a network proxy adds it, '
                             'or run --engine local')
        key = key or None               # None: no Authorization header
    elif server is Server:
        check_model(m)
        check_llama()
    today, names, state = night_state(today, tonight or guard.tonight, t,
                                      cache=not dry_run)
    done = []

    def once(b, ask):
        try:
            guard.answer(b, ask, names, today, state, dry_run=dry_run,
                         path=path, now=now)
            done.append(b)
        except SystemExit as ex:
            say(f'{b}: {ex}')
        except Exception as ex:                 # never lose another's row
            say(f'{b}: {type(ex).__name__}: {ex}')

    if 'rule' in todo:
        once('rule', guard.ask_rule)
    if engine == 'hosted':
        # checked here, not at import: a half-made declaration (DECLARED
        # without guard.DECIDER, or the other way round) costs the decider's
        # night alone, with the reason in decider.log, while the control and
        # the reference decide and the nightly pull can still bring the fix
        try:
            probs = declaration_problems()
        except Exception as ex:
            probs = [f'{type(ex).__name__}: {ex}']
        for b, pin in HOSTED.items():
            if b in todo:
                once(b, not_asked(probs) if b == 'http' and probs else
                     hosted_ask(pin, key, opener, backend=b, today=today))
    elif 'http' in todo:
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
                once('http', ask)
        except SystemExit as ex:
            say(f'http: {ex}')
        except Exception as ex:                 # never lose the control's row
            say(f'http: {type(ex).__name__}: {ex}')
    return done


def state_sha(state):
    """The night's state_sha exactly as guard.decide logs it. The k > 1
    orders are keyed on it, so they follow the one shared state, and a
    catch-up run on the cached state draws the same orders."""
    return hashlib.sha256(json.dumps(state, sort_keys=True)
                          .encode()).hexdigest()[:12]


def not_asked(probs):
    """The decider's ask under a half-made declaration: it asks nothing and
    says why, so nothing is logged under a suffix the grader does not gate."""
    def ask(st, nm):
        raise SystemExit('not asked, the declaration is half-made: '
                         + '; '.join(probs))
    return ask


def hosted_ask(pin, key, opener=None, backend=None, today=None, how=None):
    """A decider on OpenRouter's Decisions API, held to its pin: every answer
    must come back from that model (any dated build of it -- the build and
    the provider are what the record logs), or nothing is logged.

    backend's DECLARED handling applies (`how` overrides it, for tests).
    None, or today's declaration: one canonical call, the record's bytes,
    logged as 'build@provider'. k > 1: call j = 0..k-1 asks in the order
    question_order(f'{today}|{state_sha}|{j}|order-v1|http'), one call after
    another; all must come from one build@provider, else nothing is logged.
    The night logs their mean p per question (guard.mean_of_draws: a
    question is logged only if every call answered it, and a mean of
    hundredths that is 0.5 is 0.5), their summed latency, the stamp
    guard.decide puts on the last arrival, and 'build@provider' + the
    suffix. A 13:47
    catch-up asks about the cached state, so it sends the 09:17 bytes again.
    What the calls cost (usage.cost) and who answered go to decider.log --
    never to the record."""
    how = DECLARED.get(backend, {}) if how is None else how
    k, tag = how.get('k', 1), suffix(how)
    if k > 1 and not today:
        raise ValueError('k > 1 draws its orders from the night\'s date')
    kw = {'opener': opener} if opener else {}
    if how.get('version', 'q1') != 'q1':
        kw['version'] = how['version']
    if how.get('views'):
        kw['view'] = guard.view_of(how['views'])
    if how.get('provider'):
        kw['provider'] = {'order': [how['provider']], 'allow_fallbacks': False}

    def ask(st, nm):
        sha = state_sha(st)
        qids = [f'hold_{n}' for n in nm] + ['hedge_book']
        draws, seen, accepted = [], [], False
        try:
            for j in range(k):
                order = None if k == 1 else guard.question_order(
                    f'{today}|{sha}|{j}|order-v1|http', qids)
                info = {}
                p, who, ms = guard.ask_http(st, nm, DECISIONS, key, pin,
                                            info=info, order=order, **kw)
                seen.append((who, info))
                # the pinned id, optionally its dated build, optionally
                # @provider -- and with a suffix to follow, no '#' of its own
                if not re.fullmatch(re.escape(pin) + r'(-\d{8})?(@[^@]+)?',
                                    who) or (tag and '#' in who):
                    raise SystemExit(f'OpenRouter answered as {who!r}, not '
                                     f'the pinned {pin!r} -- not logged')
                if draws and who != draws[0][1]:
                    raise SystemExit(
                        f'call {j + 1} of {k} answered as {who!r}, call 1 as '
                        f'{draws[0][1]!r} -- a night is one build@provider; '
                        f'not logged')
                draws.append((p, who, ms))
            accepted = True
        finally:
            spent(backend or pin, seen, how, accepted)
        if k == 1:
            p, who, ms = draws[0]
            return p, who + tag, ms
        p = guard.mean_of_draws([d[0] for d in draws])
        return p, draws[0][1] + tag, sum(d[2] for d in draws)
    return ask


def spent(backend, seen, how, accepted=True):
    """decider.log, never the record: who answered and what it cost
    (OpenRouter's usage.cost, summed over the calls), whatever came of them;
    and, once the answers were accepted (accepted), a one-line note when the
    build is not the one the battery measured, or the provider not the
    pinned one. Notes only: they never stop a night. A night refused (two
    builds, an impostor) gets no drift note -- its refusal says who
    answered."""
    if not seen:
        return
    try:
        costs = [i['usage'].get('cost') if isinstance(i.get('usage'), dict)
                 else None for _, i in seen]
        got = [c for c in costs if isinstance(c, (int, float))
               and not isinstance(c, bool)]
        cost = 'n/a' if not got else f'{sum(got):.6g}' + (
            '' if len(got) == len(costs) else
            f' ({len(got)} of {len(costs)} calls reported one)')
        say(f'{backend}: {len(seen)} call(s) answered by '
            f'{", ".join(sorted({str(w) for w, _ in seen}))}; usage.cost '
            f'{cost}', echo=False)
        builds = sorted({str(i.get('model') or str(w).split('@')[0])
                         for w, i in seen})
        if not accepted:
            return
        mb = how.get('measured_build')
        if mb and builds != [mb]:
            say(f'{backend}: build drift -- answered by {", ".join(builds)}, '
                f'the battery measured {mb}; not a refusal (a new build is '
                f'its own record); re-run E1-E7 on it within 5 trading days')
        pv = how.get('provider')
        provs = sorted({str(i.get('provider')) for _, i in seen})
        if pv and provs != [pv]:
            say(f'{backend}: provider drift -- served by {", ".join(provs)}, '
                f'pinned to {pv}; not a refusal')
    except Exception as ex:                     # a note never costs a night
        say(f'{backend}: decider.log note failed ({type(ex).__name__})',
            echo=False)


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
    """The PowerShell that registers the task. Paths and arguments travel in
    environment variables (QQQ_PY, QQQ_SCRIPT, QQQ_ARGS, QQQ_REPO), never
    inside the script text, so no quote in a path can break or rewrite it."""
    trig = []
    for i, at in enumerate(runs_et):
        hhmm, days = local_trigger(at, ref)
        trig.append(f"$t{i} = New-ScheduledTaskTrigger -Weekly -DaysOfWeek "
                    f"{','.join(days)} -At '{hhmm}'")
    return '\n'.join([
        "$ErrorActionPreference = 'Stop'",
        "$a = New-ScheduledTaskAction -Execute $env:QQQ_PY "
        "-Argument ('\"' + $env:QQQ_SCRIPT + '\"' + $env:QQQ_ARGS) "
        "-WorkingDirectory $env:QQQ_REPO",
        *trig,
        "$s = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable "
        "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
        "-ExecutionTimeLimit (New-TimeSpan -Hours 4) "
        "-MultipleInstances IgnoreNew",
        f"Register-ScheduledTask -TaskName '{TASK}' -Action $a -Trigger "
        f"@({','.join(f'$t{i}' for i in range(len(runs_et)))}) -Settings $s "
        f"-Description 'qqq shadow decider' -Force | Out-Null",
    ])


def systemd_quote(s):
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"') \
        .replace('%', '%%') + '"'


def systemd_units(python, script, repo, runs_et=RUNS_ET, args=()):
    service = '\n'.join([
        '[Unit]', 'Description=qqq shadow decider', '',
        '[Service]', 'Type=oneshot',
        f'WorkingDirectory={str(repo).replace("%", "%%")}',
        'ExecStart=' + ' '.join(systemd_quote(x)
                                for x in (python, script, *args)),
        # the hosted engine's key, OPENROUTER_API_KEY=..., in this optional
        # file (mode 600): a user service does not see the login shell's env
        'EnvironmentFile=-%h/.config/qqq-decider.env',
        'TimeoutStartSec=4h', 'Nice=10', ''])
    timer = '\n'.join([
        '[Unit]', 'Description=qqq shadow decider, weekdays before the close',
        '', '[Timer]',
        *[f'OnCalendar=Mon..Fri {at} America/New_York' for at in runs_et],
        'Persistent=true', '', '[Install]', 'WantedBy=timers.target', ''])
    return service, timer


def schedule(runs_et=RUNS_ET, remove=False, engine='hosted',
             proxy_auth=False):
    repo = dedicated()
    script = repo / 'src' / 'decide_local.py'
    if not script.exists() and not remove:
        raise SystemExit('run --setup first (the schedule runs the dedicated '
                         'clone)')
    python = sys.executable
    args = (() if engine == 'hosted' else ('--engine', engine)) + \
        (('--proxy-auth',) if proxy_auth else ())
    if os.name == 'nt':
        ps = (f"Unregister-ScheduledTask -TaskName '{TASK}' -Confirm:$false"
              if remove else windows_task(runs_et))
        r = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy',
                            'Bypass', '-Command', ps], capture_output=True,
                           text=True, env=dict(os.environ, QQQ_PY=python,
                                               QQQ_SCRIPT=str(script),
                                               QQQ_ARGS=''.join(
                                                   ' ' + x for x in args),
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
    service, timer = systemd_units(python, script, repo, runs_et, args)
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

def setup(m, local=False):
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
    if not local:
        say('ready for the hosted engine: set OPENROUTER_API_KEY in this '
            'machine\'s environment, then --dry-run and --schedule. (For the '
            'local Kev engine: --setup --engine local downloads ~4.5 GB.)')
        return
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


def nightly(m, threads=None, engine='hosted', proxy_auth=False):
    """What the schedule runs, from the dedicated clone."""
    import guard
    if ROOT.resolve() != dedicated():
        raise SystemExit(f'real decisions run from the dedicated clone '
                         f'{dedicated()} (python src/decide_local.py --setup, '
                         f'then --schedule) so they never touch your working '
                         f'copy; here, use --dry-run')
    today = dt.datetime.now(guard.ET).strftime('%Y%m%d')
    sync(ROOT, today)
    run(m, threads=threads, engine=engine, proxy_auth=proxy_auth)
    # whatever this run, or an earlier one today, logged goes out now
    who = 'hosted: ' + ', '.join(HOSTED.values()) if engine == 'hosted' \
        else f'local: {model_name(m)}'
    publish(ROOT, f'guard: decisions before the close ({who})')


def main():
    ap = argparse.ArgumentParser(description='Tonight\'s shadow decisions: '
                                 'the earnings rule, Perplexity Decider and '
                                 'Jev through OpenRouter (hosted), or Kev on '
                                 'a local llama.cpp (local).')
    ap.add_argument('--setup', action='store_true')
    ap.add_argument('--schedule', action='store_true')
    ap.add_argument('--unschedule', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--model', choices=sorted(MODELS), default=DEFAULT_MODEL)
    ap.add_argument('--threads', type=int)
    ap.add_argument('--engine', choices=('hosted', 'local'), default='hosted',
                    help='hosted: OpenRouter, key in OPENROUTER_API_KEY '
                         '(default); local: Kev on llama.cpp, no key')
    ap.add_argument('--proxy-auth', action='store_true',
                    help='hosted, with no OPENROUTER_API_KEY here: send no '
                         'key, the network proxy adds it')
    a = ap.parse_args()
    try:
        if a.setup:
            setup(a.model, local=a.engine == 'local')
        elif a.schedule or a.unschedule:
            schedule(remove=a.unschedule, engine=a.engine,
                     proxy_auth=a.proxy_auth)
        elif a.dry_run:
            run(a.model, dry_run=True, threads=a.threads, engine=a.engine,
                proxy_auth=a.proxy_auth)
        else:
            nightly(a.model, a.threads, a.engine, a.proxy_auth)
    except SystemExit as ex:
        if ex.code not in (None, 0):        # the log says why a night was
            say(f'stopped: {ex}', echo=False)   # missed, even under a
        raise                                   # scheduler; Python prints it
    except Exception as ex:
        say(f'FAILED: {type(ex).__name__}: {ex}')
        raise


if __name__ == '__main__':
    main()
