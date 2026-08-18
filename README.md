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
data/bars/               65 days of 1-minute bars (derived, 3.5 MB, committed)
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
