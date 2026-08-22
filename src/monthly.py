# Part of qqq-microstructure.
#
# The whole forward-test ritual as one command. The strategy is monthly by
# construction, so this is the entire operating burden: run it in the first
# days of each month (or let Task Scheduler do it), commit reports/, done.
#
# Steps, each continuing past failure with a summary at the end:
#   1. xsec_extend        build newly completed panel months from Yahoo
#   2. xsec_replay        frozen-model verdict on everything past the cutoff
#   3. xsec_live          pre-register this month's basket, grade elapsed days
#   4. opra_pull --yes    extend the options record (~$0.50/month; skipped
#      + opra_value       unless DATABENTO_API_KEY is set)
#
#   python src/monthly.py
#
# Hands-free with failover (Windows; run the block once in PowerShell). Weekly
# on purpose: every step is idempotent, so extra runs are free and each month
# gets four to five chances instead of one. StartWhenAvailable catches up after
# sleep or a missed boot, WakeToRun wakes a sleeping machine, and RestartCount
# retries transient failures because this script exits nonzero when a step
# fails. Runs as the logged-on user, so set DATABENTO_API_KEY with setx to make
# it visible. The pushed commits double as the heartbeat: no commit for a
# month means the task is broken.
#
#   $repo    = "C:\Users\bader\qqq-microstructure"
#   $action  = New-ScheduledTaskAction -Execute "cmd.exe" -Argument ('/c cd /d ' + $repo +
#              ' && python src\monthly.py >> reports\monthly.log 2>&1' +
#              ' && git add reports data && (git diff --cached --quiet || ' +
#              '(git commit -m "forward log" && git push))')
#   $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 18:00
#   $set     = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
#              -RestartCount 3 -RestartInterval (New-TimeSpan -Hours 1) `
#              -ExecutionTimeLimit (New-TimeSpan -Hours 2)
#   Register-ScheduledTask -TaskName "qqq-monthly" -Action $action `
#              -Trigger $trigger -Settings $set

import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, 'src')


def run(name, args):
    print(f'\n{"=" * 20} {name} {"=" * 20}', flush=True)
    rc = subprocess.call([sys.executable, os.path.join(S, args[0])] + args[1:],
                         cwd=ROOT)
    return name, rc


def main():
    steps = [('extend', ['xsec_extend.py']),
             ('replay', ['xsec_replay.py']),
             ('live', ['xsec_live.py'])]
    if os.environ.get('DATABENTO_API_KEY'):
        steps += [('opra_pull', ['opra_pull.py', '--yes']),
                  ('opra_value', ['opra_value.py'])]
    else:
        print('DATABENTO_API_KEY not set -- options record skipped this month')
    results = [run(n, a) for n, a in steps]
    print(f'\n{"=" * 20} summary {"=" * 20}')
    for n, rc in results:
        print(f'  {n:<12} {"ok" if rc == 0 else f"FAILED (rc={rc})"}')
    print('\nnow: git add reports data && git commit -m "monthly forward log" '
          '&& git push\n(reports/xsec_paper*.csv are the evidence; '
          'data/opra_daily.parquet stays local)')
    try:
        from notify import send
        if send('qqq monthly: '
                + '  '.join(f'{n} {"ok" if rc == 0 else "FAILED"}'
                            for n, rc in results)
                + '\nnow commit reports/'):
            print('telegram: sent')
    except Exception:
        pass
    # nonzero exit when any step failed, so Task Scheduler's RestartCount can
    # retry -- every step is idempotent, so a retry only redoes what is missing
    sys.exit(1 if any(rc for _, rc in results) else 0)


if __name__ == '__main__':
    main()
