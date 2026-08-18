# qqq-microstructure

Microstructure research on 515 trading days of Nasdaq ITCH top-of-book data for QQQ,
and a build plan for a strategy that trades on it.

The short version: **the directional signal in this data is strong and unmonetizable.
The only edge that survives costs comes from deciding when *not* to quote.**

---

## The dataset

Databento batch job `XNAS-20251009-UVUB86RLRM` — 37.9 GB zstd, 519 files.

| | |
|---|---|
| Dataset | `XNAS.ITCH` (Nasdaq's own book, single venue) |
| Schema | `mbp-1` — top of book, every BBO change |
| Symbol | `QQQ` only |
| Range | 2023-09-20 → 2025-10-08 (515 trading days) |
| Volume | ~3.9 M events/day, ~2.0 B total |
| Nasdaq flow | ~$2.10 B notional/day |

The archive is **not** included here (37.9 GB). Point `QQQ_DBN_ZIP` at your own copy.

Three limits that each rule out a category of strategy:

- **One venue.** ~10–15% of consolidated QQQ volume, and no NBBO. Spreads measured
  here are wider than what a real order faces.
- **One symbol.** No cross-section — ranking, pairs and factor-neutral designs are out.
- **No queue position.** `mbp-1` gives aggregate size at the touch, not individual
  orders. For a passive strategy queue position *is* the strategy. You need `mbo` for that.

## Headline results

Measured on the real files — 65 days for bar-level tests, 20–24 days for event-level
markouts (1.12 M passive fills), sampled evenly across both years. Full tables in
[RESULTS.md](RESULTS.md).

**Passive fills capture +0.159 bps and lose it in under a millisecond.**

| Horizon | fill | 1 ms | 10 ms | 100 ms | 1 s | 10 s | 60 s |
|---|---|---|---|---|---|---|---|
| Markout (bps) | +0.159 | −0.021 | −0.054 | −0.067 | −0.074 | −0.077 | −0.094 |

Negative on 24/24 days at 10 ms, 100 ms and 1 s. Nasdaq's top-tier add rebate
(+0.065 bps at $473) does not close the gap — an indiscriminate quoter nets −0.015 bps.

**Directional signal is real and still loses.** Queue imbalance predicts the next mid
move at IC 0.236 @100 ms — a strong number for this field — but the decile edge per
side (0.069 bps) is smaller than the half-spread you pay to act (0.082 bps).

**5–60 minute horizons are noise.** Every directional feature flips sign out of sample.

**The one finding that pays is a sign flip.** Sorting fills by the book state
immediately *before* each trade:

| Quote only when… | Share of volume | Net + rebate | Per 100 shares |
|---|---|---|---|
| Always (no selection) | 100.0% | −0.015 bps | −$0.071 |
| Spread = 1 tick | 69.2% | −0.034 bps | −$0.161 |
| Spread ≥ 2 ticks | 30.8% | +0.028 bps | +$0.129 |
| **Spread ≥ 2 ticks, imbalance > 0.4** | **8.5%** | **+0.048 bps** | **+$0.226** |
| Spread ≥ 2 ticks, imbalance > 0.8 | 2.6% | −0.016 bps | −$0.075 |

The relationship is **non-monotonic** — pushing imbalance past 0.8 breaks the edge.
That is the honest argument for a learned model over a hand-written rule.

**The edge is reachable.** That table assumes you were already resting in the book when
the spread went wide, so it is worthless if wide states are microsecond flickers. They
are not: the median ≥2-tick run lasts ~2 ms and 36.7% last longer than 10 ms. Requiring
the state to have existed for 10 ms before the fill — a reaction window a well-built
non-colocated system can meet — still leaves 7.8% of volume at +0.077 bps net.

**But the magnitude is not yet established.** The same condition measured on two
different day samples gave +0.048 bps (20 days) and +0.095 bps (12 days). Only the
*sign* is currently supported; the size is sampling noise until the full 515 days are
run. Treat every bps figure here as provisional.

### What it's worth

The +0.048 bps is a **ceiling**: it assumes you are filled on every qualifying trade,
at zero latency, at a rebate tier that requires institutional volume. Qualifying flow is
~340 k shares/day; winning 10% of it is ~$19 k/yr gross against $100–250 k/yr of
colocation, feeds and membership. **As a standalone HFT business for one person this
does not clear its own infrastructure cost.** See [docs/strategy-plan.html](docs/strategy-plan.html)
for what to build instead.

## Layout

```
src/
  probe_day.py           decode one day, dump schema and a sample
  signal_ic.py           IC of queue imbalance / OFI / trade flow vs forward mid
  markout_day.py         markout + adverse selection, one day
  markout_multiday.py    size-weighted markout across sampled days
  selective_quoting.py   markout conditioned on spread state and imbalance
  state_reachability.py  wide-spread durations; does the edge survive latency?
  build_bars.py          streaming ETL: zip -> 1-minute bars with features
  bar_alpha.py           longer-horizon IC, out-of-sample split, seasonality
  dataset.py             labelled fills: 18 features + markout label per passive fill
  model.py               model definition, checkpoint save/load, decision rule
  train.py               train or resume; regression-guarded
  evaluate.py            score unseen days, write EOD portfolio reports
data/bars/               65 days of 1-minute bars (derived, 3.5 MB, committed)
data/fills/              labelled training fills (~250 MB, gitignored, regenerable)
models/                  the checkpoint + manifest (committed)
eval_data/               drop new .dbn.zst files here to score them
reports/                 eod_YYYYMMDD.json / .txt, running account state
docs/strategy-plan.html  the five-phase build plan
```

## Running it

```bash
pip install -r requirements.txt
```

```bash
export QQQ_DBN_ZIP=/path/to/XNAS-20251009-UVUB86RLRM.zip
```

`bar_alpha.py` runs against the committed bars with no archive needed:

```bash
python src/bar_alpha.py
```

Rebuild bars from the archive — arg is the day stride, `1` for all 515 days
(5 workers, ~3 s/day):

```bash
python src/build_bars.py 8
```

Then the event-level studies:

```bash
python src/markout_multiday.py
```

```bash
python src/selective_quoting.py
```

### Notes for anyone extending this

- **Never bulk-extract the archive.** It is larger than most free disks. Every script
  streams one day out of the zip, decodes, aggregates, and deletes the temp file.
- `df['flags']`, never `df.flags` — the latter collides with pandas' own
  `DataFrame.flags` property and fails with a confusing `TypeError`.
- Filter to `F_LAST` rows (`flags & 128`) so each row is a coherent book state.
- RTH is 390 minutes and the UTC offset shifts with DST (13:30 UTC in EDT,
  14:30 in EST). The scripts detect it by activity.
- All figures use `ts_recv` (what you would actually have observed), not `ts_event`.
- Markouts are **size-weighted**. Equal-weighting flatters small fills.

## Phase 4 killed it — read this first

The queue simulator (`src/queue_sim.py`) is the step that decides whether anything above
is real. It tracks an explicit position in the queue and fills you only once traded
volume exceeds the size resting ahead of you.

```
 P&L/day      top       mid     entry      base      none
  0.5ms   -804.95  -1467.50  -1994.83  -2469.19  -3566.18
  5.0ms  -1373.76  -1950.00  -2407.35  -2786.56  -3545.00
 50.0ms  -2372.58  -2767.39  -3122.07  -3342.49  -3753.66
```

**Zero of fifteen cells is profitable.** The best cell is also unreachable — the top
rebate tier requires >0.9% of consolidated US volume — and still loses $805/day.

The same model, gating the same decisions but *assuming* fills, produced +0.048 bps.
Under real queue mechanics it produces −0.009 bps. That gap is the whole lesson: when you
wait in a queue you do not get the fills you chose, you get the ones nobody faster wanted.

This trips the kill criterion written down before the phase began. Everything below is
still accurate, and none of it survives contact with a real queue. See RESULTS.md §7,
including the lookahead bug that made the first run of this look profitable.

## The model

Predicts the **maker's markout in bps** if a passive quote resting here were filled —
the P&L of an action, not the path of a price. Direction is predictable in this data at
IC 0.24 and still loses money, so a better directional model would only lose more
efficiently. LightGBM over 18 book-state features, ~3.4 M labelled fills.

**It is the baseline, not the challenger.** The plan called for gradient boosting as the
bar and *"a small temporal CNN over the raw event stream"* as the model that has to beat
it. Only the baseline was built. LightGBM sees 18 hand-engineered features at a single
instant and discards the sequence — the shape of how the book moved over the preceding
few hundred milliseconds — which is the natural thing to model on an event stream.

That gap is real. It is also probably not the binding constraint: gross edge is +0.077
bps against a 0.338 bps round trip (§6), so a challenger would need to be roughly **4x
better**, not incrementally better. Untested either way.

```bash
python src/dataset.py 6      # extract labelled fills (stride 6; ~7 min for 84 days)
python src/train.py          # train, or resume from the existing checkpoint
python src/evaluate.py       # score whatever is in eval_data/
```

Held-out validation (10 days the model never saw, threshold calibrated on a separate
earlier half so it is not scored on its own tuning data):

| | share of fills | bps/share |
|---|---|---|
| Quote everything | 100% | −0.0032 |
| Hand-written rule (spread ≥ 2t, imb > 0.4) | 3.1% | −0.0222 |
| **Model-gated** | **71.4%** | **+0.0138** |

IC +0.0965. Note the rule *loses* on this window while the model holds positive — the
same sample-instability the rest of this repo documents. Note also that **sign accuracy
is 52.9% against a 52.9% always-one-class baseline**: the model has essentially no edge
at calling the sign of an individual fill. All of its value is in the economics, which
is why bps/share is the metric that matters and accuracy is reported but not relied on.

### Checkpoints and resuming

`models/adverse_selection.txt` is the LightGBM model; the `.manifest.json` beside it
records the features, every day already trained on, a SHA of the model, the calibrated
decision threshold and a full session history. Both are committed, so a clone starts
from a trained model rather than from zero. Saves are atomic.

Two honest caveats, because the resume path did not behave the way it was designed to:

- **Appending trees to a converged model never improved it.** Resuming on 8 new days cost
  0.0138 → 0.0075 bps/share. Adding replay (750 k rows resampled from seen days) and
  shrinking the step to 25 rounds reduced the damage but never reversed it — the ensemble
  was already at its validation optimum, so extra capacity only overfits.
- `train.py` therefore ships a **regression guard**: a checkpoint that scores worse than
  the one it replaces is rejected and the previous model kept. It fires on exactly this
  case, which is the correct outcome rather than a bug.

So the thing that genuinely avoids starting from zero is the **cached extraction** in
`data/fills/` — the expensive step, never repeated. Once days are cached, a full rebuild
(`--fresh`) takes about a minute and is the refresh that actually works. Incremental
training is implemented and guarded; use it, but read the verdict it prints.

## The 0DTE pivot — and what actually came out of it

Competing with HFTs on microstructure is the one game where a retail account is
structurally disadvantaged. Moving to a daily horizon changes the arena. It also cuts the
sample from 2 billion events to **515 days**, where the minimum detectable Sharpe is
**1.40** — most of what a daily strategy could earn is smaller than the noise here.

One signal survived: **intraday momentum**. Sign of the first N minutes, held to the
close, traded in shares. Every other feature — overnight gap, prior-day return, prior
realised vol — flipped sign between train and test.

```
  entry    5m   10m   15m   30m   60m  120m
  bps/d  2.99  1.37  6.78  4.00 10.00  5.17
  t      0.66  0.31  1.54  0.95  2.56  1.49
```

**Quote the band, not the best cell.** 60m is an outlier, not the centre of a robust
hump — 30m scores below 15m, and only 60m clears t=2. Dropping the best 10 days takes it
from 10.00 to 4.23 bps/day. The honest estimate is **~5–6 bps/day, roughly 13–15%/yr at
Sharpe ~0.9** — below what this sample can establish. What supports it is that all six
cells are positive, all three years are positive, and it is a published effect
(Gao, Han, Li & Zhou, *Market Intraday Momentum*, JFE 2018) rather than something sifted
out of this archive.

It trades in **shares** — no colocation, no rebate tier, no queue. That is what makes it
categorically different from §1–§7.

**Buying 0DTE options on this signal gives the edge back.** Break-even IV is 14.8%
annualised and QQQ 0DTE rarely prints below 15%; the premium charged for leverage is
almost exactly the size of the edge. Selling the wing captures a different edge (+8.3
bps/day at the −0.5% strike, from the variance risk premium) but the worst day in this
crash-free sample is −5.87%, which is 65 days of credit. Defined-risk spreads only, and
none of it is evaluable without an OPRA feed — every option figure here assumes an IV.

## Forward testing — `live_momentum.py`

More backtesting cannot settle a Sharpe-0.9 effect on 515 days, and every extra variant
tried on the same days makes the estimate worse. Only new days add information.

```bash
python src/live_momentum.py --dry-run
```

```bash
python src/live_momentum.py --paper
```

Fixed rules, set in advance so there is nothing to tune while it runs: record the price
at 09:30 ET, enter at 10:30 (long if the first hour is up, short if down), flatten at
15:58 unconditionally. Each day writes `reports/live/YYYYMMDD.json`.

**It will not trade a live account.** `--paper` is the only broker mode, the port is
checked against IBKR's paper ports, and the account code must start with `DU`. Default
size is 10 shares — sizing up an unproven edge is how a marginal strategy becomes an
expensive one. Opening and funding an account is yours to do; this connects to one that
already exists.

## Evaluation and EOD reports

`eval_data/` is a drop folder for Databento files. The archive ends **2025-10-08**, so
anything dated later is genuinely out of sample. Evaluation reuses `features_from_dbn()`,
the same code path as training, so eval cannot drift from training.

Each day writes `reports/eod_YYYYMMDD.json` and `.txt` with balance, ROI, per-share edge,
accuracy and the running account.

On the 12 days held out of training entirely:

```
balance $100,135.02   P&L $+135.02   ROI +0.1350%   over 12 days
9 of 12 days profitable   |   ~1.0% of Nasdaq QQQ volume
```

**Read that ROI carefully.** The first version of this simulation assumed a fill on every
quoted trade and reported +7.94% over 12 days — while trading 2.46 M shares against a
venue that traded 5.29 M, i.e. **47% of all Nasdaq QQQ volume**. That is not a strategy,
it is an impossibility. The `--participation` flag (default 2% of each qualifying print)
fixes it, and every report now carries `pct_of_venue_volume` so the assumption stays
visible. The remaining +0.135% over 12 days is ~$11/day on $100 k — consistent with the
capacity arithmetic above, and still resting on assumed fills that phase 4 exists to
demolish.

## Method

Markout is P&L per share for the passive side of a fill, in bps of mid:

```
MM P&L = sign * (fill_price - mid[t+h]) / mid[t] * 1e4
```

where `sign` is +1 when a buy aggressor lifted the ask (the maker sold) and −1 when a
sell aggressor hit the bid. At `h=0` this is the captured half-spread; as `h` grows it
decays by adverse selection. Signals are computed from the book state at `t-1` — the
event *before* the trade — so there is no lookahead.

## Status

Research only. Nothing here has been traded, and none of it is investment advice.
The findings are measured on sampled days, not the full 515 — replicating at full
scale is phase 2 of the plan.
