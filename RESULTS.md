# Results

Raw output from the scripts in `src/`. Every number in the README and the plan traces
to a table here.

- Bar-level tests: 65 days (every 8th trading day), 25,350 one-minute bars.
- Event-level markouts: 20–24 days (every 22nd / 26th day), 1.12 M passive fills.
- Sampling is evenly spaced across 2023-09-20 → 2025-10-08, so results span both the
  low-vol 2024 stretch and the 2025 regime.

---

## 1. Day characterization — `probe_day.py`, `signal_ic.py`

2025-10-08, one day:

```
rows (all events)          4,028,906
RTH book updates           3,726,160      159 updates/sec
trades                        68,339
spread          1c 74.1%   2c 25.1%   3c 0.7%   4c 0.1%
half-spread     mean 0.104 bps   median 0.082 bps
top-of-book     bid med 410 sh   ask med 500 sh
```

Note the spread distribution is **Nasdaq-local**. Consolidated QQQ is ~1 tick almost
always; 25% two-tick states here are largely Nasdaq quote gaps, not a wide market.
This matters for section 4 and is the first thing to re-check with an NBBO feed.

### Information coefficient vs forward mid return

```
signal             100ms        1s       10s       60s
QueueImb          0.2360    0.1096    0.0402    0.0434
OFI_1s            0.0288    0.0155    0.0041   -0.0198
OFI_10s           0.0148   -0.0010   -0.0440   -0.0264
TradeFlow_1s      0.0001    0.0011    0.0046   -0.0165
```

Queue imbalance is a genuinely strong short-horizon predictor. The rest is weak.

### Why it does not pay

Decile edge conditioned on spread state, 1-second forward return:

```
spread=1c    n=2,760,048   half-spread 0.082 bps  | D0 -0.064  D9 +0.074  edge/side 0.069 bps
spread>=2c   n=  966,112   half-spread 0.168 bps  | D0 -0.032  D9 +0.068  edge/side 0.050 bps
```

Edge per side (0.069 bps) < half-spread (0.082 bps). **A taking strategy loses 0.013 bps
per round trip even with a top-decile signal.**

---

## 2. Passive markout — `markout_multiday.py`

Size-weighted, 24 days sampled across two years. `fill` is measured against the
pre-trade mid, i.e. the captured half-spread.

```
        mean      std    negative days
fill  +0.1587   0.0502     0/24
 1ms  -0.0212   0.0189    22/24
10ms  -0.0544   0.0156    24/24
100ms -0.0673   0.0192    24/24
  1s  -0.0737   0.0213    24/24
 10s  -0.0774   0.0523    22/24
 60s  -0.0940   0.0957    20/24
```

Per-day detail (bps):

```
     day  mm_fill  mm_1ms  mm_10ms  mm_100ms   mm_1s  mm_10s  mm_60s
20230920  +0.2040 -0.0171  -0.0526   -0.0534 -0.0526 -0.0792 -0.0689
20231020  +0.1673 -0.0462  -0.0759   -0.0841 -0.0907 -0.0820 -0.1070
20231121  +0.1368 -0.0441  -0.0615   -0.0773 -0.0774 -0.1002 +0.0111
20231222  +0.1517 -0.0423  -0.0546   -0.0585 -0.0608 -0.0601 -0.0530
20240126  +0.1422 -0.0403  -0.0621   -0.0621 -0.0757 -0.0333 -0.0474
20240228  +0.1246 -0.0446  -0.0640   -0.0733 -0.0797 -0.0764 +0.0099
20240401  +0.1224 -0.0345  -0.0652   -0.0812 -0.0765 -0.1087 -0.1923
20240501  +0.1671 -0.0386  -0.0750   -0.0899 -0.1076 +0.0225 +0.0225
20240603  +0.1489 -0.0260  -0.0581   -0.0780 -0.0934 -0.1198 -0.1199
20240705  +0.1360 -0.0106  -0.0450   -0.0652 -0.0804 -0.1247 -0.1788
20240806  +0.2431 -0.0175  -0.0892   -0.1091 -0.0673 -0.0581 -0.0545
20240906  +0.1945 -0.0104  -0.0619   -0.0685 -0.0806 -0.0946 -0.2158
20241008  +0.1462 -0.0084  -0.0472   -0.0626 -0.0891 -0.1479 -0.0634
20241107  +0.1587 -0.0027  -0.0363   -0.0453 -0.0725 -0.0077 -0.2020
20241210  +0.1434 -0.0069  -0.0380   -0.0631 -0.0678 -0.1353 -0.1747
20250114  +0.2628 +0.0200  -0.0427   -0.0727 -0.0986 -0.2138 -0.3934
20250214  +0.1549 -0.0008  -0.0257   -0.0341 -0.0353 -0.0205 -0.1008
20250319  +0.2187 -0.0137  -0.0671   -0.0896 -0.1047 -0.1075 -0.0247
20250421  +0.2685 +0.0159  -0.0462   -0.0977 -0.1103 -0.0686 -0.1049
20250521  +0.1292 -0.0302  -0.0645   -0.0635 -0.0557 -0.0838 -0.1138
20250624  +0.1120 -0.0166  -0.0341   -0.0409 -0.0431 +0.0012 +0.0092
20250725  +0.0943 -0.0177  -0.0292   -0.0332 -0.0390 -0.0376 -0.0255
20250826  +0.0962 -0.0322  -0.0526   -0.0522 -0.0541 -0.0690 -0.0294
20250926  +0.0852 -0.0429  -0.0582   -0.0608 -0.0569 -0.0519 -0.0392
```

With the Nasdaq top-tier add rebate ($0.00305/share = +0.0645 bps at the $473 mean price):

```
MM net @1s  incl rebate: -0.0093 bps  = -$0.0439 per 100 shares
MM net @10s incl rebate: -0.0129 bps  = -$0.0611 per 100 shares
```

The decay from +0.159 to negative inside 1 ms is partly mechanical — the trade that
fills you is the same event that moves the book. That is not a measurement artifact,
it is the cost of being filled indiscriminately, and it is what selection has to fix.

---

## 3. Selective quoting — `selective_quoting.py`

1,121,388 fills over 20 days. `fav` is the imbalance signed toward the side you filled
on, taken from the book state at `t-1` (the event **before** the trade — no lookahead).

Markout by favourable-imbalance decile, size-weighted:

```
 decile  fav mid    mo_1s  +rebate   mo_10s  share%
      0   -0.976  -0.1324  -0.0677  -0.1361     1.5
      1   -0.898  -0.1150  -0.0503  -0.0930     5.8
      2   -0.833  -0.1107  -0.0460  -0.1014     8.4
      3   -0.767  -0.1036  -0.0388  -0.0850     8.0
      4   -0.645  -0.0979  -0.0332  -0.0845     9.0
      5   -0.500  -0.0882  -0.0234  -0.0934    10.0
      6   -0.333  -0.0874  -0.0226  -0.0970    10.7
      7   -0.064  -0.0548   0.0099  -0.0076    11.6
      8    0.207  -0.0581   0.0066  -0.0726    14.3
      9    0.669  -0.0578   0.0070  -0.0599    20.7
```

Split by spread state — **this is the dominant variable, not imbalance**:

```
  spread=1c   fav>-1.0:  69.2% of vol  mo_1s -0.0988  net+reb -0.0341 bps  = $-0.1605/100sh
  spread=1c   fav> 0.4:  13.2% of vol  mo_1s -0.0851  net+reb -0.0204 bps  = $-0.0960/100sh
  spread=1c   fav> 0.8:   3.3% of vol  mo_1s -0.0782  net+reb -0.0134 bps  = $-0.0632/100sh
  spread>=2c  fav>-1.0:  30.8% of vol  mo_1s -0.0373  net+reb +0.0275 bps  = $+0.1294/100sh
  spread>=2c  fav> 0.4:   8.5% of vol  mo_1s -0.0167  net+reb +0.0481 bps  = $+0.2264/100sh
  spread>=2c  fav> 0.8:   2.6% of vol  mo_1s -0.0806  net+reb -0.0158 bps  = $-0.0747/100sh
```

Two things to carry forward:

1. At one tick you are at the back of a deep queue behind faster participants; the
   fills that reach you are the ones nobody faster wanted. That is what −0.099 bps means.
2. The `fav > 0.8` row **breaks the edge**. Extreme imbalance means the book is about
   to flip, not that it is safe. Non-monotonic — a threshold rule cannot express it.

Caveat that phase 4 of the plan exists to address: this assumes you are filled on every
qualifying trade. You would not have been.

---

## 4. Longer horizons — `bar_alpha.py`

25,350 one-minute bars over 65 days. Features z-scored within day.

IC vs forward return:

```
feature          f5      f15      f30      f60
qi_mean      0.0064  -0.0067  -0.0199  -0.0183
ofi_n       -0.0035  -0.0126  -0.0232  -0.0259
svol_n      -0.0082   0.0074   0.0095  -0.0047
ret         -0.0033  -0.0013  -0.0081  -0.0204
rv           0.0073   0.0290   0.0417   0.0185
spr_bps      0.0227   0.0456   0.0534   0.0238
vimb         0.0058  -0.0071  -0.0204  -0.0202
```

Out-of-sample split (train 39 days < 2024-12-16, holdout 26 days):

```
feature      IS f30  OOS f30    IS f5   OOS f5
qi_mean     -0.0401   0.0037  -0.0011   0.0158
ofi_n       -0.0453   0.0025   0.0039  -0.0126
svol_n       0.0135   0.0046  -0.0021  -0.0158
ret         -0.0168   0.0021  -0.0072   0.0015
rv           0.0528   0.0300   0.0085   0.0059
spr_bps      0.0359   0.0753   0.0218   0.0241
vimb        -0.0394   0.0018   0.0033   0.0089
```

**Every directional feature reverses sign out of sample.** `rv` and `spr_bps` hold their
sign but are volatility proxies, not direction — they do not support a long/short.

Decile long/short spreads, gross, against a 0.456 bps round-trip cost:

```
  qi_mean   f5  : L/S gross  +0.029 bps
  qi_mean   f30 : L/S gross  -2.016 bps
  ret       f5  : L/S gross  -0.302 bps
  ret       f30 : L/S gross  -0.485 bps
  ofi_n     f5  : L/S gross  +0.073 bps
  ofi_n     f30 : L/S gross  -0.833 bps
  svol_n    f5  : L/S gross  -0.307 bps
  svol_n    f30 : L/S gross  +0.264 bps
```

Mostly negative before paying to trade. **There is no minutes-to-hours alpha here.**

### Intraday seasonality

The one robust, immediately actionable pattern — and it needs no model:

```
         spr(bps)      vol     rv  |ret|
9:30      0.719    18053.8  0.075  4.147
10:00     0.599    12311.1  0.063  3.698
10:30     0.458    15690.4  0.050  4.049
11:00     0.444    12700.9  0.048  3.536
11:30     0.424    10954.3  0.046  3.206
12:00     0.424     9356.0  0.046  2.971
12:30     0.419     7929.6  0.045  2.629
13:00     0.424     8180.0  0.046  2.601
13:30     0.419     7359.8  0.046  2.521
14:00     0.423     8606.6  0.047  2.667
14:30     0.410     7810.4  0.046  2.491
15:00     0.411     8520.1  0.044  2.640
15:30     0.394    18797.6  0.045  2.951
```

Trading the opening half-hour costs ~0.72 bps against ~0.40 bps after 11:00 —
**roughly double**. Volume is the familiar U-shape. Any order that can wait, should.

---

## Reproducing

`bar_alpha.py` runs against the committed bars in `data/bars/` with no archive.
Everything else needs `QQQ_DBN_ZIP` pointing at the Databento file. Sampling strides are
hardcoded near the bottom of each script (`names[::8]`, `names[::22]`, `names[::26]`);
set them to `1` for the full 515 days.

---

## 5. Latency reachability — `state_reachability.py`

12 days (every 43rd), 1,338,246 wide-spread runs. This tests the biggest validity threat
to section 3: the +0.048 bps assumes you were *already resting* in the book when the
spread went wide. If wide states are microsecond flickers between quote updates, nobody
outside a colocation cage could ever have been there and the edge is unreachable.

### How long does a >=2-tick Nasdaq spread last?

```
  p25 :      0.181 ms
  p50 :      1.925 ms
  p75 :     44.078 ms
  p90 :    238.981 ms
  p95 :    524.116 ms
  p99 :   1985.869 ms

  runs lasting > 1 ms  :  54.6%
  runs lasting > 10 ms :  36.7%
  runs lasting > 100 ms:  17.5%
  runs lasting > 1 s   :   2.4%
```

Not flickers. The median run is ~2 ms and over a third last longer than 10 ms.

Note also that `spread >= 2t` covers **65.8% of book time** but only ~9% of traded
volume — wide states are time-heavy and trade-light. Time-weighting the opportunity
would badly overstate it; all edge figures here are volume-weighted.

### Does the edge survive requiring the state to pre-date your quote?

`L` is how long the wide state must already have existed before the fill — i.e. how much
reaction time you had to be resting there.

```
       L  % of all vol  gross mo_1s  net+rebate   per 100sh
0 (orig)         9.92%       0.0290      0.0945     +$0.4402
  0.1 ms         9.33%       0.0260      0.0915     +$0.4264
    1 ms         8.75%       0.0052      0.0706     +$0.3291
    5 ms         8.16%       0.0037      0.0692     +$0.3223
   10 ms         7.82%       0.0117      0.0771     +$0.3596
   50 ms         6.94%       0.0304      0.0959     +$0.4468
  100 ms         6.32%       0.0022      0.0676     +$0.3151
     1 s         3.16%      -0.0501      0.0154     +$0.0716
```

**The edge survives.** At `L = 10 ms` — a reaction window a well-built non-colocated
system can meet — 7.8% of volume still nets +0.077 bps. It only collapses at `L = 1 s`,
which is the sensible direction: chronically wide spreads mark genuinely bad moments,
not opportunities.

### The caveat this exposes

Compare the `L = 0` row here (+0.0945 bps net, 9.92% of volume) with the same condition
in section 3 (+0.0481 bps, 8.5% of volume). Same test, different day samples — 12 days
here vs 20 there — and the point estimate moved by a factor of two.

That spread is not a contradiction, it is sampling noise, and it is the argument for
running phase 2 before anything else. **Neither number should be trusted as the edge;
only the sign is currently established.** Full 515-day replication with a per-year split
is the next step.

### Still open

This does not address the NBBO question. A 2-tick spread on Nasdaq may be a genuinely
wide market or merely a Nasdaq-local quote gap while another venue holds the inside.
Those have very different economics and this dataset cannot tell them apart. Consolidated
data is required, and it should gate any capital commitment.

---

## 6. Can a directional strategy hit a 75% win rate? — `win_rate_study.py`

6 days, 12,000 entries/day, direction from queue imbalance (|imbalance| >= 0.30), entries
crossing the spread. Triple-barrier exits: profit target T ticks, stop S ticks, time
limit H. 90 configurations.

Two rates, and the gap between them is the whole point:

- `hit_rate` — target barrier reached before the stop
- `win_rate` — trade made money **after** paying the spread and fees

### Answer: yes, and it does not help

```
  H    T   S   hit_rate   win_rate   expectancy_bps
300    5  20     0.7960     0.7680          -0.5777
```

**One of 90 cells clears 75%. It loses 0.58 bps per trade.** Of the cells reaching a
75%+ win rate, the number that are profitable after costs is **zero**.

The geometry that produces the high win rate is what destroys it: a 5-tick target with a
20-tick stop wins four times out of five and pays 4:1 when it loses. Before costs that is
`0.796 x 5 - 0.204 x 20 = -0.10` ticks — already negative. Costs then take another
0.59 bps.

The best-expectancy cell in the grid runs the opposite geometry — a 10-tick target with a
1-tick stop, **12.3% win rate**, and it is the *least* bad configuration at -0.51 bps.
Win rate and profitability point in opposite directions here.

### Why every cell is negative

```
best gross edge anywhere in the grid   +0.0768 bps
round-trip cost charged in this study   0.5863 bps
```

Gross is *positive* — the signal genuinely works, which matches the IC 0.236 in section 1.
It is simply 7.6x too small to pay the toll.

That 0.586 bps cost is inflated by this being single-venue data: Nasdaq-local spreads
average 2.18c, while consolidated QQQ is ~1 tick. Against a realistic NBBO round trip:

```
1c spread + 2 x $0.0030/share fees at $473   0.3383 bps
gross edge                                  +0.0768 bps
net                                         -0.2615 bps   (short by 4.4x)
```

Still negative, by a factor of four. Correcting the cost narrows the gap and does not
close it.

### What this rules out

No model changes a 4x cost shortfall. The constraint is not prediction quality — the
signal is real and was never the problem — it is that QQQ's spread is wider than the
information content of its order book at these horizons. Leverage does not help either:
it scales expectancy per dollar without changing its sign, so leveraging a -0.26 bps
edge simply loses money faster.

The only positive-expectancy result anywhere in this repo remains the passive one in
section 3, where you are *paid* the spread instead of paying it.

---

## 7. Phase 4 — the honest fill simulation — `queue_sim.py`

9 most recent archive days, 100-share quotes on both sides, model-gated, explicit queue
position tracked through every event. Fills only occur once traded volume at our level
exceeds the size resting ahead of us.

### P&L per day, by latency and rebate tier

```
  latency            top            mid          entry           base           none
    0.5ms        -804.95      -1,467.50      -1,994.83      -2,469.19      -3,566.18
    5.0ms      -1,373.76      -1,950.00      -2,407.35      -2,786.56      -3,545.00
   50.0ms      -2,372.58      -2,767.39      -3,122.07      -3,342.49      -3,753.66
```

**Zero of fifteen cells is profitable.** The best case is not merely unprofitable but
unreachable: the top tier requires >0.9% of consolidated US volume, and it still loses
$805/day at −0.009 bps/share.

Latency orders correctly — 0.5 ms beats 5 ms beats 50 ms — and each tier beats the one
below it. Both gradients being the right way round is the main evidence the simulator is
behaving.

### A bug worth recording

The first version of this ran the opposite way: 50 ms latency *beat* 0.5 ms, and two
cells came out profitable. The cause was in the posting logic —

```python
j = L_arr[i]                       # index at t + latency
lp = L_bp[j] if s == 0 else L_ap[j]   # <- price at ARRIVAL
```

The order was placed at the price that would exist once it arrived. That is foresight,
and the more latency the more of it, which is precisely why slower looked better. Fixed
by aiming at the price visible at the decision (`t - latency`) and handling the three
outcomes on arrival: still the touch (join the back), improving on it (empty queue, and
the adverse selection of being alone at a stale price), or left behind (a miss).

After the fix, the profitable cells disappear.

### What killed it

Per-share edge under realistic fills is **−0.009 to −0.083 bps** depending on tier and
latency. Compare with section 3, where the same model gating the same decisions but
*assuming* fills produced +0.048 bps. The gap between those two numbers is the whole
content of phase 4: **when you have to wait in a queue, you do not get the fills you
chose — you get the ones nobody faster wanted.**

Note also that the sim quotes on 14–25% of venue volume. It is not being selective, and
the model's threshold was calibrated on assumed fills, not queued ones.

### Verdict against the stated kill criterion

> Kill if: the edge does not survive a realistic fill rate at a realistic rebate tier.

It does not survive at *any* tier or *any* latency. Phase 4 fired.

### 7b. Selectivity sweep — the verdict reverses

The §7 grid used the threshold calibrated in training (−0.055 bps), which quotes on
14–25% of venue volume. That is not a selective strategy. Sweeping the quoting threshold
under the same queue mechanics (5 days, 0.5 ms latency):

```
     lat/thr            top            mid          entry           base           none
  0.5ms -0.06        -963.17      -1,652.48      -2,183.51      -2,611.11      -3,683.30
  0.5ms +0.02        -614.85        -945.10      -1,205.62      -1,321.95      -1,387.50
  0.5ms +0.05        -372.37        -595.42        -660.90        -636.40        -640.56
  0.5ms +0.10         -15.66         -91.60        -161.88        -213.56        -147.48
  0.5ms +0.20         +93.93         +73.33         +62.81         +56.81         +52.55
  0.5ms +0.40         +42.76         +39.22         +34.17         +34.43         +33.85
```

At a +0.20 bps threshold **every tier turns positive**, including `none` — no rebate at
all, +$52.55/day. That is a materially stronger result than anything earlier in this
repo, because it does not depend on a rebate tier you cannot get. It quotes on **0.45%
of venue volume**: roughly 1/50th the activity of the §7 configuration.

Best cell: top tier, +$93.93/day, ROI 23.7%/yr on $100 k nominal. At the realistic entry
tier, +$62.81/day ≈ 15.8%/yr.

**Do not trust these numbers yet.** Six thresholds were swept on five days and the best
was reported — precisely the search that manufactures spurious results. The threshold
must be validated on days that were not used to choose it, and at latencies a real system
would actually have. Both are open.

### 7c. Out-of-sample validation — it does not hold up

The +0.20 threshold was chosen on the five days in §7b. Re-run on **six earlier days that
played no part in choosing it**, across three latencies:

```
     lat/thr          entry           none
  0.5ms +0.10         -77.14         -94.13
  0.5ms +0.20         +15.37         +28.41
  0.5ms +0.40          +5.68          +4.40
  5.0ms +0.10        -162.93        -155.81
  5.0ms +0.20          -6.37         -21.46
  5.0ms +0.40         +12.41          +3.27
 50.0ms +0.10        -209.53        -133.84
 50.0ms +0.20         -12.06         +14.15
 50.0ms +0.40         +17.15         +17.18
```

Three things to read here, none of them encouraging:

1. **Shrinkage.** At 0.5 ms / +0.20 the entry tier falls from **+$62.81 to +$15.37/day**
   and `none` from +$52.55 to +$28.41. A 2–4x haircut is the signature of a threshold
   fitted to the days it was chosen on.
2. **Latency inconsistency.** The configuration that survives at 0.5 ms is *negative* at
   5 ms (−$6.37 entry). A real edge should decay with latency, not flip sign and then
   recover at 50 ms.
3. **Tier ordering inverts.** At +0.20 / 0.5 ms, `none` (+$28.41) beats `entry`
   (+$15.37). More rebate produced *less* P&L. The two runs quote different fill sets, so
   this is not impossible, but a higher rebate scoring worse is a noise signature, not an
   edge.

9 of 18 cells positive is a coin flip, and the magnitudes are $5–28/day.

**Verdict: not established.** The §7b result does not survive out-of-sample. And even
taking the best cell at face value, $28.41/day is ~$7 k/yr, against the $100–250 k/yr of
colocation, market data and membership that 0.5 ms latency requires. The configuration
that is arguably profitable is one that cannot pay for the infrastructure it depends on.

Phase 4's verdict stands.

---

## 8. 0DTE pivot — the underlying half — `daily_bars.py`, `odte_study.py`, `odte_strategy.py`

Daily spine built over all **515 days**. No options data exists in this archive, so
nothing here prices a contract. It settles the two questions upstream of any option.

### Power first

```
open-to-close sd            106.1 bps/day
standard error of the mean    4.67 bps/day
minimum detectable Sharpe     1.40 annualised
```

A Sharpe 1.0 strategy is 1.43 sigma in this sample. Below that threshold, absence of
evidence is not evidence of absence. 515 days is four orders of magnitude less data than
the microstructure work and everything below inherits that weakness.

### One signal survives: intraday momentum

Sign of the first 60 minutes, held to the close, trading shares:

```
full sample  n=515   +10.00 bps/day   t=2.56   Sharpe 1.79   hit 54.2%

  year     n   bps/day      t   Sharpe   hit%
  2023    71     15.00   1.91     3.60   54.9
  2024   252      9.78   2.13     2.13   53.2
  2025   192      8.43   1.05     1.20   55.2

walk-forward (direction fitted on the past only)
  train 206d -> test 309d   +11.46 bps/day  t=2.01
  train 283d -> test 232d    +9.50 bps/day  t=1.36
  train 360d -> test 155d   +11.19 bps/day  t=1.20
  train 437d -> test  78d    +7.14 bps/day  t=1.57
```

Positive in every year and every walk-forward split. It is also a **published effect**
(Gao, Han, Li & Zhou, *Market Intraday Momentum*, JFE 2018), which matters at this sample
size — it was not found by sifting this archive. Note the decay, 15.0 → 9.78 → 8.43,
consistent with a known effect being competed away. Every other feature tested — overnight
gap, previous day's return, previous realised vol — **flipped sign** between train and test.

~10 bps/day is roughly 25%/year unleveraged, and it is executable in **shares**, with no
colocation, no rebate tier and no queue. That is a categorically different proposition
from everything in §1–§7.

### Buying 0DTE options gives the edge straight back

The realised signal-aligned entry-to-close distribution has mean +10.34 bps, sd 88.5 bps.
An ATM call struck at entry is worth **0.342% of spot** against that distribution. What
the market charges:

```
  0DTE IV   sigma 5.5h   mkt ATM call   our fair   edge
     12%        0.695%         0.277%     0.342%   +6.5 bps
     14%        0.811%         0.324%     0.342%   +1.8
     15%        0.869%         0.347%     0.342%   -0.5
     16%        0.927%         0.370%     0.342%   -2.8
     20%       1.159%          0.462%     0.342%  -12.0
     25%       1.449%          0.578%     0.342%  -23.6
```

**Break-even IV is 14.8% annualised.** QQQ 0DTE IV rarely prints below ~15%. The
directional edge is real and the premium charged for leverage is very close to exactly
the same size — so the shares keep the +10 bps and the long-option version hands it back.

### Selling the wing captures a different edge

Same distribution, valuing puts struck below entry against a 16% IV market:

```
   strike   P(breach)   our fair   mkt@16%   edge
   -0.50%       17.9%     0.090%    0.172%   +8.3 bps
   -0.75%       10.7%     0.055%    0.110%   +5.5
   -1.00%        6.2%     0.033%    0.066%   +3.3
   -1.50%        1.2%     0.016%    0.021%   +0.5
```

+8.3 bps/day at the −0.5% strike ≈ 20.8% of spot per year — comparable to the momentum
edge, from an unrelated source (the variance risk premium). Selling *in the momentum
direction* collects both.

### The tail this sample does not contain

```
worst aligned entry-to-close moves (%):  -5.87  -3.15  -2.71  -2.07  -1.73  -1.64

sell -0.5% put: worst day costs 5.37% of spot =  65 days of credit
sell -1.0% put: worst day costs 4.87% of spot = 146 days of credit
```

And a 2020-03-16 style −12% day, which is **not in these 515 days**, costs ~11% of spot
on a −1% put: **329 days of credit, about 1.3 years**. Naked short premium is not
survivable. Anything built here must be a defined-risk spread, and the long wing will eat
a large share of the credits above.

### What is still missing

Every option number on this page assumes an IV. That is the single most important input
and it is invented. It needs an OPRA feed (Databento sell `OPRA.PILLAR`) with per-strike
quotes, priced at the **ask when buying and the bid when selling**, never the mid. Until
then the momentum result stands on its own in shares, and the option overlay is a
hypothesis.

### 8b. Momentum robustness — the headline was the best cell, not the effect

§8 quoted +10.00 bps/day for a 60-minute entry. That was **one of six entry times tested**,
and it is the outlier:

```
  entry   bps/day      t  Sharpe   CAGR%   maxDD%   hit%
     5m      2.99   0.66    0.46    6.43   -16.89   51.8
    10m      1.37   0.31    0.22    2.24   -16.75   49.5
    15m      6.78   1.54    1.07   17.15   -14.52   53.8
    30m      4.00   0.95    0.66    9.34   -13.83   51.3
    60m     10.00   2.56    1.79   27.39    -9.70   54.2   <- the quoted cell
   120m      5.17   1.49    1.04   13.04    -8.62   52.2

  band mean (10m-120m): 5.46 bps/day, 5/5 positive
```

A robust effect gives a smooth hump around its best value. This is not that: 30m (4.00)
scores *below* 15m (6.78), and only the 60m cell clears t=2. That shape is what noise
looks like when you sample it six times.

Two more checks in the same direction:

```
  all 515 days          10.00 bps/day   t=2.56
  drop best  5 days      6.29           t=1.90
  drop best 10 days      4.23           t=1.32
  median day             5.93 bps
```

**Honest estimate: ~5–6 bps/day, not 10.** That is roughly 13–15%/yr unleveraged at a
Sharpe near 0.9 — *below* the 1.40 minimum this sample can detect (§8, power). The
direction of the effect is supported (all six cells positive, positive in all three
years, and it is a published result), but the magnitude quoted in §8 was inflated
about twofold by picking the best cell.

### Leverage

```
  leverage    CAGR%   maxDD%  worst day%
        1x    27.39    -9.70      -5.87
        3x    95.15   -26.74     -17.61
       10x   373.60   -66.68     -58.71
```

Computed on the inflated 60m series, so treat even these as optimistic. And the worst day
in the sample is −5.87% in a window containing no crash: at 3x a −12% session is −35% in
a single day, and drawdown scales with leverage while ruin does not scale back.

### What resolves this

Not more backtesting — 515 days cannot settle a Sharpe-0.9 effect, and every further
variant tested on the same days makes it worse. Only forward testing adds information.
That is the argument for paper trading this live rather than refining it further.

---

## 9. OPRA smoke test — one day, and it inverts §8's ranking

`OPRA-20260818-A3GELRSYU5.zip` — one day (2026-07-15), `cbbo-1m`, `QQQ.OPT` parent.
64 MB, 4.48 M rows, 11,530 instruments. Loader: `src/opra_load.py`.

### Two things that make the data cheaper than planned

**The `definition` schema is not needed.** A parent-symbology request embeds 11,530 OSI
symbol mappings in the DBN metadata, and OSI encodes everything:
`QQQ   260715C00718000` → 2026-07-15, call, $718 strike.

**The underlying is not needed either.** Put-call parity recovers spot from the chain
alone, `S = K + C − P`, read at the strike where call and put are closest in value:

```
  09:31  $723.77      first 60m  -76.4 bps  -> signal SHORT
  10:30  $718.24      entry -> close  -7.9 bps
  16:00  $717.67      signal-aligned  +7.9 bps
```

So an OPRA pull does **not** have to overlap the equity archive. It carries its own spot,
and the momentum signal can be computed from options data alone. The date-range advice in
§8 was over-constrained.

### Quote quality — better than assumed

```
  ATM $718   call $2.20   put $1.96   straddle $4.15
  straddle bid-ask $0.04            = 1.0% of premium
  29 call strikes within +/-2%, median spread $0.03 = 2.8% of premium
  0DTE chain: 184 strikes, 495-950, 406 minutes, 64.8% two-sided
```

### The number that matters

**0DTE IV at 10:30 was 12.5% annualised**, not the 16% assumed throughout §8. The
straddle cost 58 bps of spot against a historical mean |aligned move| of 58 bps — priced
almost exactly at fair value versus the two-year realised distribution. **No variance
risk premium on this day at all.**

Re-ranking at the measured 12.5% inverts §8 completely:

```
                              EV $/contract
  structure                   @16% (assumed)   @12.5% (measured)
  Long call ATM                      -14.42            +23.65
  Long call +0.5% OTM                -15.35            +16.29
  Short straddle ATM                 +74.13             -4.01
  Short strangle +/-0.5              +51.31            -13.75
  Iron condor 0.5/1.0                +29.91             +0.69
```

Every short-vol structure that looked good at 16% is dead at 12.5%, and buying calls in
the momentum direction becomes the best trade. This is exactly why the break-even IV
table was the invariant output and the EV table was not.

### But the long-call result is a bet on the signal, not on cheap options

The aligned distribution carries a +10.34 bps drift, and §8b established that ~5.5 bps is
the honest figure. Shrinking the drift:

```
  structure                 drift 10.3   drift 5.5   drift 0
  Long call ATM                  23.65       11.35     -1.35
  Long call +0.5% OTM            16.29       10.37     +4.28
  Bull call spread 0/+0.5         4.36       -2.02     -8.63
```

At the honest drift the long-call edge roughly halves; at zero drift it is negative. The
options are close to fairly priced, so **all of the edge comes from the momentum signal**
— the option is leverage on it, not a source of edge in itself.

### What one day cannot do

12.5% is a single draw. IV varies enormously day to day and 2026-07-15 was quiet — the
realised aligned move was 8 bps against a 58 bps breakeven, so short vol won handsomely
that day *despite* IV being low. One observation establishes neither the level nor the
premium. That is the argument for pulling the full range.

---

## 10. A one-hour session-offset bug — 169 of 515 days were wrong

Cross-validating against a free HuggingFace minute-bar dataset (`mito0o852/OHLCV-1m`)
exposed a bug in this repo that had contaminated every result above.

Every extraction script located the session open by *searching* for it:

```python
for off in (13, 14):                     # 09:30 ET under EDT / EST
    s_ns = d0 + (off * 3600 + 30 * 60) * 10**9
    if ((ts >= s_ns) & (ts < s_ns + 390*60*10**9)).sum() > 100_000:
        break                            # <-- 13 always wins
```

Under **EDT**, 13:30 UTC is 09:30 ET and the first branch is right. Under **EST**,
13:30 UTC is 08:30 ET — but that window still contains 5.5 hours of regular trading, so
it clears the 100,000-event threshold and the loop stops there. The result: on every EST
day the "session" ran **08:30–15:00 ET**, silently swapping an hour of thin pre-market
for the closing hour.

**169 of 515 days (33%)** — roughly November–March in both years.

Caught by comparing one day against the HF bars:

```
my "open"  518.09  =  HF 08:30  (diff $0.01)
my "p60"   523.02  =  HF 09:30  (diff $0.17)
my "close" 522.24  =  HF 15:00  (diff $0.01)
```

An exact match, shifted one hour. The prices were never wrong; the clock was.

Fixed by computing the open from the calendar rather than from activity:

```python
d = dt.datetime.fromtimestamp(first_ts_ns / 1e9, dt.timezone.utc).date()
et = dt.datetime.combine(d, dt.time(9, 30), ZoneInfo('America/New_York'))
return int(et.astimezone(dt.timezone.utc).timestamp()) * 10**9
```

Applied to all seven affected scripts: `dataset.py`, `daily_bars.py`, `build_bars.py`,
`markout_multiday.py`, `selective_quoting.py`, `state_reachability.py`,
`win_rate_study.py`.

### After the fix, the two sources agree exactly

```
  d_open    |mean| 0.39 bps   max 3.25
  d_entry   |mean| 0.28 bps   max 1.34
  d_close   |mean| 0.15 bps   max 0.40
  signal agrees 41/41 days (100%)     [44% before the fix]
```

Sub-tick agreement between Databento tick mids and HF trade bars. **The HF dataset is
validated** — free, minute bars back to 1992, and accurate to a fraction of a basis
point against paid tick data.

### The momentum result got stronger

Pre-market contamination was adding noise, not signal:

```
  entry   bps/day      t  Sharpe   CAGR%   maxDD%   hit%
     5m      1.45   0.32    0.22    2.34   -18.18   51.7
    10m      5.64   1.29    0.91   13.88   -10.71   52.0
    15m     13.95   3.22    2.25   40.39    -9.11   57.5
    30m      6.33   1.54    1.08   16.01    -9.27   53.6
    60m      9.62   2.52    1.76   26.23    -6.57   55.5
   120m      4.99   1.42    1.00   12.51    -9.53   53.4

  band mean 8.10 bps/day (was 5.46), 5/5 positive
  yearly at 15m: +17.43 / +9.22 / +18.87   at 60m: +17.11 / +8.38 / +8.47
```

The same discipline still applies: **15m is now the best cell and should not be quoted
as the result.** The band is still not a smooth hump, which is what noise looks like.
Honest estimate is the band mean, **~8 bps/day**, Sharpe ~1.3 — still near the 1.40 this
sample can resolve.

### What remains contaminated

Sections 1–7 were all produced with the buggy window and have **not** been re-run. On
EST days they included thin pre-market (wide spreads, inflating the `spread >= 2 ticks`
state that §3's result depends on) and excluded the closing hour. The scripts are fixed;
the numbers in those sections are not. Phase 4's verdict is unlikely to reverse — it
lost by a wide margin at every latency and tier — but §3's magnitudes should be treated
as unreliable until re-run.

---

## 11. The 27-year verdict: real, half the size, and ML doubles it honestly

Full history from the validated HF minute bars (QQQ stitched across the QQQQ era):
QQQ 6,285 days, SPY 6,273, IWM 5,188, DIA 5,855.

### The sign rule replicates -- and shrinks

```
 ticker   days  bps/day      t  Sharpe   CAGR%   maxDD%
    QQQ   6285     3.94   2.55    0.51    8.38   -46.18
    SPY   6273     2.30   2.19    0.44    5.03   -43.68
    IWM   5188     3.15   2.24    0.49    6.87   -36.57
    DIA   5855     0.99   0.96    0.20    1.72   -58.36
```

Three of four instruments clear t=2 independently, 18/27 QQQ years positive. The effect
is real and market-wide. It is also HALF the 515-day estimate (3.94 vs ~8), and heavily
regime-dependent: +21 to +30 bps/day in 2001/2002/2008/2022, near zero 2013-2017. The
recent sample sat in a hot regime.

### A lookahead bug worth recording

First ML run printed Sharpe 2.42 out of sample -- the tell, not a result. Top feature
was `rv_1m_bps`, which build_daily computes over the WHOLE session: at 10:30 it leaks
the afternoon's volatility, and vol-return asymmetry converts that into the sign of the
remaining move. Removed.

### The honest model (walk-forward, each year predicted only from prior years)

LightGBM, 12 features observable at 10:30. Tried and REJECTED, recorded so they are not
re-tried: trailing-vol features (cut OOS from 3.85 to 2.08 -- vol says how big, not
which way), vol-targeted sizing (0.47 vs 0.66), 3-ETF equal-weight portfolio (IWM's 0.28
drags more than the 0.13 correlations help).

```
 QQQ, 4,836 OOS days (2005-2025)   bps/day      t  Sharpe   CAGR%   maxDD%
 sign rule (baseline)                 2.07   1.56    0.36    4.23   -46.18
 ML sign                              3.85   2.91    0.66    9.02   -26.86
 SPY replication (untouched)          3.12   2.82    0.63    7.35   -28.16
```

IC +0.084 QQQ, +0.094 SPY. The model roughly doubles the rule's Sharpe and halves its
drawdown, and does it again on an instrument it was never developed on.

### Against the stated goal (1%+/month)

ML-sign QQQ averages **0.72%/month** unleveraged (Sharpe 0.66, maxDD -27%). ~55% of
months are positive; the worst are -5 to -8%. Reaching a 1%/month AVERAGE needs ~1.4x
leverage (maxDD ~-37%), or the current hot regime persisting (2022-2025 OOS ran 8-23
bps/day = 2-6%/month). "Positive every month" is not achievable at any realistic Sharpe
and no honest backtest will promise it.

---

## 12. The frozen-model holdout: Nov 2025 - Mar 2026, and it failed

Before any broker deployment, the production model (trained through 2025-10-31) was run
over 102 days it had never seen in any form -- HF bars 2025-11-03 .. 2026-03-31.

```
                 bps/day   total%   hit%      t
  ML (frozen)      -9.01    -9.19   45.1  -1.13
  sign rule        -5.86    -5.98   51.0  -0.73

  by month (ML):  Nov +0.75%  Dec -1.69%  Jan -0.69%  Feb +2.01%  Mar -9.58%
```

Holdout IC was **-0.148** against a walk-forward average of +0.08.

### Diagnosis

Not a data problem: the same files over the same window give SPY **+0.41 bps/day** with
the same frozen model, and March 2026 volatility is unremarkable (sd 115 vs 105
reference). March QQQ was a reversal regime -- 03-03: first hour -72 bps, rest of day
+161; 03-09: -13 then +238; 03-31 the model faded a +79 first hour and the market ran
+147. The model was wrong-footed on QQQ specifically while being fine on SPY, which is
regime plus instrument-level noise, not a broken pipeline.

Statistically the miss is ~1.4 sigma: at Sharpe 0.5, five-month stretches this bad have
~10% probability, so the holdout neither kills the 27-year result nor comes close to
confirming it. But you only live one path.

### The one pre-declared rescue, tested and rejected

Stand aside when the strategy's own trailing quarter (60 trading days, chosen in
advance, no sweep) is negative:

```
  27-year OOS:  always-on 2.94 bps/day Sharpe 0.50  ->  filtered 1.31 / 0.27   WORSE
  holdout:      always-on -9.19%       ->  filtered -3.59%                     less bad
```

It would have sat out March entirely and still halves the long-run edge. Rejected.

### Deployment verdict

- **Do not fund anything.** The recent five months were net losing and the 1-2%/month
  goal is unsupported in the current regime.
- **IBKR paper is optional, not urgent.** The strategy trades at two fixed timestamps,
  so monthly HF bar updates replay it exactly -- a free, zero-infrastructure forward
  test. Re-run the holdout evaluation as each month lands; consider paper/live only if
  the trailing 6-12 months turn positive again.
- The 27-year edge remains statistically real (t=2.3-2.9 across variants and
  instruments). What the holdout shows is its cost: Sharpe ~0.5 means years like this,
  and no filter tested honestly removes them without removing the edge too.

---

## 13. Signal #2: the overnight premium — stronger than momentum, uncorrelated

The documented overnight/intraday split (Lou, Polk & Skouras), tested on the validated
spines. Both unadjusted-price caveats are conservative: split days dropped, and
dividends land as negative overnight price moves the holder actually receives.

```
 ticker   days |  overnight bps/d      t |  intraday bps/d      t
    QQQ   6748 |             5.08   4.56 |           -1.43  -0.81
    SPY   6775 |             2.28   2.65 |            0.15   0.13
    IWM   5687 |             4.35   3.73 |           -1.28  -0.77
    DIA   6355 |             1.53   1.76 |            0.82   0.71
```

All four instruments positive, QQQ at t=4.56 over 27 years, positive in 7 of 8
four-year eras. Correlation with the momentum strategy: **+0.017**.

### The stack

The overnight leg holds 16:00 -> 09:30; the momentum leg trades 10:30 -> 16:00. They
never overlap, so one pot of capital runs both:

```
  overnight only    +4.74 bps/d  t=4.25  Sharpe 0.82  CAGR 11.50%  maxDD -30.6%
  momentum only     +3.67 bps/d  t=2.48  Sharpe 0.48  CAGR  7.67%  maxDD -47.0%
  STACK             +8.42 bps/d  t=4.50  Sharpe 0.87  CAGR 20.00%  maxDD -43.9%
```

**The stack averages 1.53%/month over 27 years with no leverage**, positive in every
4-year era (+11.4 bps/day in the current one). This is the first configuration in this
repo that reaches the stated goal on the full history rather than on a hot regime.

### The honest caveats

- **The holdout was bad for both**: overnight −1.91%, stack −7.89% over Nov 2025 - Mar
  2026. Zero long-run correlation did not prevent a jointly bad Feb-Mar 2026. For the
  overnight leg alone the miss is only ~0.7 sigma — normal variation — but the stack
  does not diversify away regime risk.
- The momentum leg here is the full-history sign rule; the deployable ML leg's OOS rate
  is similar (2.9-3.9 bps/day) but not identical.
- The overnight anomaly is widely published. It has nonetheless stayed positive through
  the most recent era in this data.
- Practicalities: two trades every day, all gains short-term for tax purposes, and
  ~500 spread-crossings/yr are modeled at 0.34 bps each.

---

## 14. Signal #3 candidates: two calendar effects, two honest nulls

Two pre-declared published effects, one pass each, no sweeps. Both point the direction
the literature says; neither clears any evidence bar on 6,747 QQQ days.

```
  turn-of-month (last + first 3):  TOM +6.95 vs other +2.56 bps/day, diff t=+0.83
    -> and INVERTED in the newest era (2023+: TOM +6.05 vs other +10.46)
  pre-holiday:                     +8.81 vs +3.14 bps/day, diff t=+0.55 (256 days)
```

Rejected, recorded so they are not re-tried. The stack stays two legs. Calendar effects
at ETF level are too small relative to ~100 bps/day noise for this sample to confirm.
