# Part of qqq-microstructure.
#
# The decider battery's command line (RESULTS 27; spec v1, section 7). Each
# step from the committed declaration to the selection is one subcommand and
# a thin call into the module that owns it:
#
#   make-items   battery_items   the stimuli, generated once and locked
#   verify       battery_items   the lock, the bytes, the counts, a clean
#                + battery_run   tree, what the plan could cost
#   run          battery_run     send every item to every model, resumably
#   score        battery_score   every metric and gate -> summary.json
#   select       battery_score   section 3's rule -> selection_v1.json
#   scrub-check  battery_run     no key-shaped string in any output
#
# Why one script with fixed exit codes: the cloud session runs exactly these
# commands, and the code alone says what to do next (resume, fix, stop)
# without anyone reading output that could hold a secret. Every line printed
# passes battery_run.scrub.
#
#   python src/decider_battery.py verify --plan bench/battery/plan_v1.json
#   python src/decider_battery.py run --plan bench/battery/plan_v1.json \
#       --out bench/battery/runs/v1 --auth proxy --max-usd 8 --deadline-min 90
#   python src/decider_battery.py score --run bench/battery/runs/v1
#   python src/decider_battery.py select --run bench/battery/runs/v1
#   python src/decider_battery.py scrub-check --run bench/battery/runs/v1
#
# Exit codes: 0 done; 2 auth failure; 3 capped (resumable); 4 deadline
# (resumable); 5 refused; 6 insufficient analysis set (resume the run);
# 7 scrub hit.

import os, sys, json, time, shutil, argparse, tempfile
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery_items as bi
import battery_run as br
import battery_score as bs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONE, REFUSED, INSUFFICIENT, SCRUB_HIT = 0, br.REFUSED, \
    bs.EXIT_INSUFFICIENT, 7
say = br.say


def _refuse(why):
    say(f'refused: {why}')
    return REFUSED


def _read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _rel(path):
    p = os.path.abspath(path)
    return os.path.relpath(p, ROOT) if br._inside(p, ROOT) else p


# ------------------------------------------------------------ make-items --

def make_items(plan_path, force=False):
    """Generate plan, states, items and lock into the plan's directory from
    the committed S0 and anchors there. Built in a scratch directory first:
    when a lock already exists and the new bytes differ from it (another
    numpy, an edited generator), nothing is written -- the committed bytes
    are the declaration, and only --force replaces them."""
    plan_path = os.path.abspath(plan_path)
    d, name = os.path.split(plan_path)
    want = f'plan_{bi.VERSION}.json'
    if name != want:
        return _refuse(f'make-items writes {want}, not {name}')
    old = _read_json(bi.lock_path(plan_path))
    was = ((_read_json(plan_path) or {}).get('generator') or {}).get('numpy')
    with tempfile.TemporaryDirectory() as tmp:
        for f in (bi.S0_FILE, bi.ANCHORS_FILE):
            if not os.path.exists(os.path.join(d, f)):
                return _refuse(f'{f} is not in {_rel(d)}')
            shutil.copyfile(os.path.join(d, f), os.path.join(tmp, f))
        try:
            out = bi.make_items(tmp)
        except (AssertionError, KeyError, ValueError) as ex:
            return _refuse(f'make-items: S0 or the anchors are not what the '
                           f'generator expects ({type(ex).__name__})')
        new = out['lock']
        changed = sorted(k for k in new if old and old.get(k) != new[k])
        if changed and not force:
            return _refuse(
                f'the regenerated {", ".join(changed)} differ from the '
                f'committed lock (numpy {bi.GENERATOR["numpy"]} here, {was} '
                f'recorded); nothing written -- --force replaces the '
                f'declaration')
        names = [os.path.basename(p) for p in out['paths'].values()]
        names.append(os.path.basename(bi.lock_path(out['paths']['plan'])))
        for f in names:                 # the lock last
            shutil.copyfile(os.path.join(tmp, f), os.path.join(d, f + '.tmp'))
            os.replace(os.path.join(d, f + '.tmp'), os.path.join(d, f))
    status = 'new' if not old else ('replaced (--force)' if changed
                                    else 'unchanged')
    say(f'{len(out["items"])} items, {len(out["states"])} states -> '
        f'{_rel(d)}; lock {status}')
    for k, v in new.items():
        say(f'  {k:8s} {v}')
    return DONE


# ---------------------------------------------------------------- verify --

def _per_model(plan, items, sd):
    """{experiment: calls} for one model, after a scale-down."""
    by = {it['item_id']: it for it in items}
    m = plan['incumbent']
    return dict(sorted(Counter(
        by[k.split('|', 1)[1]]['experiment']
        for k in br.planned(plan, items, sd)
        if k.split('|', 1)[0] == m).items()))


def projection(plan, items):
    """What the plan could cost at catalogue prices: every model at the
    pessimistic per-call figure scaled by its input price (battery_run's
    estimate before a pilot lands), and the scale-down the run would apply
    if the pilot measured exactly that -- block 4, then unscored models,
    dearest first, never I or R. The run decides from measured costs; this
    is the bound, before any call."""
    sp, models = plan['spend'], br.models_of(plan)
    cand = {c['id']: c for c in plan['candidates']}

    def c0(m):
        price = cand.get(m, {}).get('price_in')
        return sp['pessimistic_per_call'] * (
            price / sp['price_ref'] if price is not None else 1.0)
    cm = {m: c0(m) for m in models}

    def proj(sd):
        n = Counter(k.split('|', 1)[0] for k in br.planned(plan, items, sd))
        return sum(cm[m] * n[m] for m in models), sum(n.values())
    exempt = {plan['incumbent'], plan['reference']}
    droppable = sorted((m for m in models if m not in exempt
                        and cand.get(m, {}).get('X') is None),
                       key=lambda m: (-cm[m], m))
    lim = 0.8 * sp['max_usd']
    sd = {'dropped_blocks': [], 'dropped_models': []}
    nominal = proj(sd)
    if nominal[0] > lim:
        sd['dropped_blocks'] = [plan['blocks']]
        for m in droppable:
            if proj(sd)[0] <= lim:
                break
            sd['dropped_models'].append(m)
    smallest = proj({'dropped_blocks': [plan['blocks']],
                     'dropped_models': droppable})
    return {'per_call': cm, 'limit_usd': lim, 'nominal': nominal,
            'scale_down': sd, 'after': proj(sd), 'smallest': smallest}


def verify(plan_path, allow_dirty=False, check_tree=None):
    """Everything the run will check before its first call, and more,
    offline: the lock and every body (battery_items.verify), the run's own
    refusal checks (battery_run.problems), the floor counts, a clean tree
    and the catalogue-price bound. Exit 0 when nothing is wrong, else 5."""
    plan_path = os.path.abspath(plan_path)
    probs = [f'[items] {p}' for p in bi.verify(plan_path)]
    try:
        plan, items, d = br.read_plan(plan_path)
    except (OSError, ValueError, KeyError) as ex:
        for p in probs + [f'cannot read {_rel(plan_path)}: '
                          f'{type(ex).__name__}']:
            say(p)
        return REFUSED
    probs += [f'[run] {p}' for p in br.problems(plan_path, plan, items, d)]
    models = br.models_of(plan)
    nominal = _per_model(plan, items, {})
    floor = _per_model(plan, items, {'dropped_blocks': [plan['blocks']]})
    want = (plan.get('calls_per_model') or {}).get('floor')
    if want is not None and floor != want:
        probs.append(f'[run] floor calls per model {floor} != plan {want}')
    pr = projection(plan, items)
    if pr['smallest'][0] > plan['spend']['max_usd']:
        probs.append(f'even the smallest run (block {plan["blocks"]} and '
                     f'every unscored model dropped) projects '
                     f'${pr["smallest"][0]:.2f} > max_usd '
                     f'{plan["spend"]["max_usd"]} at catalogue prices')
    if check_tree is None:
        check_tree = br._inside(plan_path, ROOT)
    dirty = br._dirty(os.path.join(d, 'runs')) if check_tree else []
    note = None
    if dirty:
        msg = (f'uncommitted changes under src/, tests/ or bench/battery/ '
               f'({len(dirty)}: {", ".join(dirty[:4])}'
               f'{", ..." if len(dirty) > 4 else ""}) -- run will refuse')
        if allow_dirty:
            note = msg
        else:
            probs.append(msg)
    say(f'plan {_rel(plan_path)} sha256 {br._sha(plan_path)[:16]}')
    say(f'{len(models)} models; per model {sum(nominal.values())} calls '
        f'nominal, {sum(floor.values())} at the floor; '
        f'{len(models) * sum(nominal.values())} calls in all')
    sd = pr['scale_down']
    say(f'catalogue-pessimistic cost ${pr["nominal"][0]:.2f} for '
        f'{pr["nominal"][1]} calls (scale-down above '
        f'${pr["limit_usd"]:.2f}: blocks {sd["dropped_blocks"]}, models '
        f'{sd["dropped_models"]} -> ${pr["after"][0]:.2f} for '
        f'{pr["after"][1]} calls); the pilot decides from measured costs')
    tree = 'not checked' if not check_tree else 'dirty' if dirty else 'clean'
    say(f'tree: {tree}' + (f' (allowed; {note})' if note else ''))
    for p in probs:
        say(p)
    say('ok' if not probs else f'{len(probs)} problem(s)')
    return DONE if not probs else REFUSED


# ------------------------------------------------------------------ score --

def score(run_dir):
    try:
        s = bs.score(run_dir)
    except (OSError, ValueError, KeyError) as ex:
        return _refuse(f'score: {type(ex).__name__}: {ex}')
    a = s['analysis_set']
    say(f'analysis set: blocks {a["blocks"]}, {len(a["states"])} states'
        + ('' if a['sufficient'] else
           f' -- fewer than {a["blocks_min"]} blocks: select will exit 6, '
           f'resume the run'))
    say('baselines: ' + ', '.join(f'{k} {v:.4f}' if v is not None else
                                  f'{k} n/a'
                                  for k, v in s['baselines'].items()))
    for m, r in s['models'].items():
        q = r['headline']['Q_h']
        say(f'  {m}: h={r["h"]} Q_h={"n/a" if q is None else f"{q:.4f}"} '
            f'failed={",".join(r["failed"]) or "-"}')
    say(f'-> {_rel(os.path.join(run_dir, "summary.json"))}')
    return DONE


# ----------------------------------------------------------------- select --

def select(run_dir):
    try:
        o = bs.select(run_dir)
    except bs.InsufficientAnalysisSet:
        return INSUFFICIENT             # select has said why, on stderr
    except (OSError, ValueError, KeyError) as ex:
        return _refuse(f'select: {type(ex).__name__}: {ex}')
    if o.get('mode') == 'iterate':
        say(f'adopted: {o.get("adopted") or "nothing"}')
        return DONE
    dec, ref = o['decider'], o['reference']
    say(f'eligible: {", ".join(o["eligible"]) or "none"}')
    say(f'branch {o["branch"]}: decider {dec["pin"]} ({dec["handling"]}'
        f'{", suffix " + dec["suffix"] if dec["suffix"] else ""}, provider '
        f'pin {"none" if not dec["provider_pin"] else dec["provider"]}); '
        f'reference {ref["id"]} {ref["status"]}')
    out = f'selection_{o.get("version") or "v1"}.json'
    say(f'-> {_rel(os.path.join(run_dir, out))}')
    return DONE


def scrub_check(run_dir):
    if not os.path.isdir(run_dir):
        return _refuse(f'no run directory {_rel(run_dir)}')
    hits = br.scrub_check(run_dir)
    for h in hits:
        say(f'scrub hit: {h}')
    say('clean' if not hits else f'{len(hits)} scrub hit(s)')
    return SCRUB_HIT if hits else DONE


# ------------------------------------------------------------------- main --

def parser():
    ap = argparse.ArgumentParser(
        prog='decider_battery.py',
        description='The decider battery (RESULTS 27): make-items, verify, '
                    'run, score, select, scrub-check.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('make-items', help='generate and lock the stimuli '
                       '(designing session only; offline)')
    p.add_argument('--plan', required=True)
    p.add_argument('--force', action='store_true',
                   help='replace a committed lock whose bytes differ')
    p = sub.add_parser('verify', help='offline checks before a run')
    p.add_argument('--plan', required=True)
    p.add_argument('--allow-dirty', action='store_true',
                   help='report uncommitted changes without failing '
                        '(the run itself still refuses them)')
    p = sub.add_parser('run', help='send every item to every model')
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--auth', choices=['proxy', 'env'], default='proxy')
    p.add_argument('--max-usd', type=float)
    p.add_argument('--deadline-min', type=float)
    p.add_argument('--concurrency', type=int)
    p.add_argument('--per-model', type=int)
    for name, what in (('score', 'metrics and gates -> summary.json'),
                       ('select', 'section 3 -> selection_v1.json'),
                       ('scrub-check', 'no key-shaped strings in outputs')):
        p = sub.add_parser(name, help=what)
        p.add_argument('--run', required=True)
    return ap


def main(argv=None, opener=urllib.request.urlopen, sleep=time.sleep,
         now=time.monotonic):
    """Exit code of one subcommand. opener, sleep and now reach the run
    unchanged: tests pass a stand-in API and a fake clock."""
    a = parser().parse_args(argv)
    if a.cmd == 'make-items':
        return make_items(a.plan, a.force)
    if a.cmd == 'verify':
        return verify(a.plan, a.allow_dirty)
    if a.cmd == 'run':
        try:
            return br.run(a.plan, a.out, auth=a.auth, max_usd=a.max_usd,
                          deadline_min=a.deadline_min,
                          concurrency=a.concurrency, per_model=a.per_model,
                          opener=opener, sleep=sleep, now=now)
        except (OSError, ValueError, KeyError) as ex:
            return _refuse(f'run: {type(ex).__name__}: {ex}')
    if a.cmd == 'score':
        return score(a.run)
    if a.cmd == 'select':
        return select(a.run)
    return scrub_check(a.run)


if __name__ == '__main__':
    sys.exit(main())
