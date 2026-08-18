# Runbook

How to run everything in this repo, in the order it makes sense to run it, with what
each step costs in wall-clock time and what it is actually for.

Read [README.md](README.md) first if you want the findings. This file is operations.

---

## 0. Prerequisites

- Python 3.11+
- The Databento archive `XNAS-20251009-UVUB86RLRM.zip` (37.9 GB). Not in this repo.
- ~5 GB free disk. **Not** 38 GB — nothing here ever extracts the archive. Every script
  streams one day out of the zip, decodes it, aggregates, and deletes the temp file.
- For paper trading only: an IBKR **paper** account and TWS or IB Gateway.

```bash
pip install -r requirements.txt
```

The archives live in `~/OneDrive/المستندات/Trading/` and the scripts default there.
To keep them elsewhere, set:

```bash
export QQQ_DBN_ZIP=/path/to/XNAS-20251009-UVUB86RLRM.zip
```

On Windows PowerShell:

```powershell
$env:QQQ_DBN_ZIP = "C:\Users\you\Downloads\XNAS-20251009-UVUB86RLRM.zip"
```

Check it works — this decodes one day and prints the schema:

```bash
python src/probe_day.py
```

---

## 1. Run the analysis without touching the archive

Two things are committed so a fresh clone can reproduce results with no data at all.

Longer-horizon IC, out-of-sample split and intraday seasonality, from 65 days of
committed 1-minute bars (instant):

```bash
python src/bar_alpha.py
```

Score the committed model checkpoint on 12 held-out days and write EOD reports
(~3 min — this one does read the archive):

```bash
python src/evaluate.py --from-zip last:12 --reset
```

---

## 2. Build the data (only needed once)

### Daily spine — needed for everything momentum and 0DTE

515 days, 5 workers, **~25 min**. Writes `data/daily.parquet` (~25 KB).

```bash
python src/daily_bars.py 1
```

### Labelled fills — needed only to retrain the microstructure model

84 days at stride 6, **~13 min**, ~250 MB in `data/fills/`. Gitignored and regenerable.
The last 12 archive days are held out automatically.

```bash
python src/dataset.py 6
```

Use a bigger stride for a faster, smaller run (`12` → ~42 days, ~7 min).

---

## 3. The live strategy — intraday momentum in shares

This is the only thing in the repo with a positive, executable expectancy. Everything in
sections 4 and 5 is either dead or unproven.

Robustness sweep — entry times, cost sensitivity, leverage, drawdown (instant):

```bash
python src/momentum_backtest.py
```

Signal validation, conditional distribution and the strike table (instant):

```bash
python src/odte_strategy.py
```

**Read the band, not the best cell.** The 60-minute entry scores 10.0 bps/day; the band
mean across six entry times is 5.46. The honest estimate is ~5–6 bps/day at Sharpe ~0.9,
which is *below* what 515 days can establish. See RESULTS §8b.

### Paper trading

Wiring check, no broker needed — replays one archive day through the identical decision
function:

```bash
python src/live_momentum.py --dry-run
```

**Your setup, once:**

1. Open an IBKR **paper** account (I can't do this part — account opening and funding
   are yours).
2. Install TWS or IB Gateway and log into the paper account.
3. Enable the API: *Global Configuration → API → Settings* → tick **Enable ActiveX and
   Socket Clients**, socket port **7497**, and add `127.0.0.1` to trusted IPs.
4. Untick *Read-Only API*.

Then, with TWS running:

```bash
python src/live_momentum.py --paper
```

The script is **stateless and idempotent**. Each run reads the day's log and does only
the step that is due, so schedule it three times a day, US Eastern:

| Time (ET) | What it does |
|---|---|
| 09:31 | Records the opening reference price |
| 10:31 | Enters — long if the first hour is up, short if down |
| 15:58 | Flattens, unconditionally |

cron (if your box is on ET):

```
31 9  * * 1-5 cd /path/to/qqq-microstructure && python src/live_momentum.py --paper
31 10 * * 1-5 cd /path/to/qqq-microstructure && python src/live_momentum.py --paper
58 15 * * 1-5 cd /path/to/qqq-microstructure && python src/live_momentum.py --paper
```

On Windows use Task Scheduler with the same three times.

Each day writes `reports/live/YYYYMMDD.json`.

**Safety properties, all tested:**

- `--paper` is the only broker mode. There is no live-trading path in the file.
- The port is checked against IBKR paper ports (7497, 4002). Port 7496 is refused.
- Account codes must start with `DU` or it disconnects without trading.
- Default size is 10 shares.

**How long before the result means anything?** At ~5 bps/day against ~90 bps of daily
noise, roughly **200 trading days**. That is the real timeline. Backtesting more will not
shorten it — 515 days cannot separate Sharpe 0.9 from zero, and every extra variant tried
on those same days makes the estimate worse.

---

## 4. Options structures

No options data exists in this archive, so nothing here prices a real contract. What it
does is rank structures by **expected value** against the empirical distribution of where
QQQ actually lands, which is the input a payoff calculator does not have.

```bash
python src/option_structures.py --iv 16
```

```bash
python src/option_structures.py --iv 20 --width 0.75 --wing 1.5
```

Options: `--iv` annualised 0DTE IV, `--half-spread` dollars per leg (default 0.01),
`--width` / `--wing` strike offsets in %, `--unconditional` to ignore the momentum signal.

### Reading the output

Two tables. The first ranks by EV per contract. The second gives each structure's
**break-even IV** — the level at which it has zero edge.

The second table is the one that matters, because the first is largely an echo of the
`--iv` you passed in. At 16% every short-vol structure shows edge and every long-vol
structure does not; at 12% that reverses completely. The break-even column is invariant
to that guess.

What survives is the *margin*: how far the real IV sits from a structure's break-even.

| Structure | Break-even IV | Cushion at 16% |
|---|---|---|
| Put credit spread −0.5/−1.0 | 10.8% | 5.2 pts |
| Iron condor 0.5/1.0 | 12.2% | 3.8 pts |
| Short strangle ±0.5% | 13.2% | 2.8 pts |
| Long call ATM | 14.7% | needs IV *below* — no |

### The workflow with a live chain

1. Read the real 0DTE IV off the chain at 10:30 and re-run with `--iv`.
2. Pick from structures whose break-even sits far from the real IV. Anything close is a
   coin flip after commissions.
3. Take that structure to <https://www.optionsprofitcalculator.com/> with the real
   strikes and premiums to see the payoff shape, breakevens and margin. That site is the
   right tool for the step *after* selection — it will not rank structures for you,
   because it has no view on where the stock actually lands.
4. Check the max-loss column against your account, not against the EV.

### Two warnings that are not negotiable

**Max loss on the naked and short-vol rows is evaluated on a ±15% grid.** It is not a
bound. Their true loss is unbounded, and the `EV/risk` column is meaningless for them —
it only compares across the defined-risk rows.

**The 515-day sample contains no crash.** The worst aligned move in it is −5.87%. A
2020-03-16 style −12% session costs ~11% of spot on a −1% short put — about **1.3 years
of credit**. Naked short premium is not survivable. Defined-risk spreads only.

---

## 5. The microstructure work (finished, negative)

Kept because the negative results are the point. None of this is live.

| Command | Time | What it shows |
|---|---|---|
| `python src/signal_ic.py <day.zst>` | ~2 min | Queue imbalance IC 0.236, still sub-cost |
| `python src/markout_multiday.py` | ~10 min | Passive fills lose within 1 ms |
| `python src/selective_quoting.py` | ~10 min | The conditional edge, assuming fills |
| `python src/state_reachability.py` | ~8 min | Wide spreads are reachable at 10 ms |
| `python src/win_rate_study.py --days 6` | ~10 min | 75% win rates exist, all unprofitable |
| `python src/queue_sim.py --days 9` | ~35 min | **Phase 4 — the honest fill sim; kills it** |

Retrain the adverse-selection model (needs `data/fills/`, ~2 min):

```bash
python src/train.py --fresh --valid 10 --rounds 200
```

Resuming (`python src/train.py` with no `--fresh`) appends trees to the existing
checkpoint. It has never improved on a full rebuild, and the built-in regression guard
will reject the update and keep the previous model. That is correct behaviour, not a
failure — read the verdict it prints.

---

## 6. Gotchas

- **Never bulk-extract the archive.** It is larger than most free disks. Everything here
  streams.
- `df['flags']`, never `df.flags` — the latter hits pandas' own `DataFrame.flags`
  property and fails with a confusing `TypeError`.
- Filter to `F_LAST` rows (`flags & 128`) so each row is a coherent book state.
- RTH is 390 minutes and the UTC offset shifts with DST (13:30 UTC under EDT, 14:30 under
  EST). The scripts detect it from activity.
- All figures use `ts_recv`, not `ts_event` — what you would actually have observed.
- Markouts are **size-weighted**. Equal-weighting flatters small fills.
- On Windows, multiprocessing scripts must be run as files, not piped to `python` on
  stdin — `spawn` re-imports `__main__` and cannot re-import stdin.
- Long runs under `nohup` are block-buffered; `tail -f` will look frozen. The scripts
  flush their progress lines.

---

## 7. When to stop

Written down in advance so the decision gets made while you are still objective.

- Paper trading shows a negative mean after 200 days → stop. The backtest was noise.
- The realised paper edge is under ~2 bps/day → stop. It will not clear commissions and
  slippage at any size you can run.
- You find yourself re-optimising the entry time on the same 515 days → stop. That is
  where the +10 bps figure came from, and it was twice the honest number.
- Any option structure that needs the top of your IV estimate to be positive → treat as
  negative. You will not reliably get the best of a spread.
- Before real money: the paper result must hold, and you must have OPRA data if any
  option is involved. Both, not either.
