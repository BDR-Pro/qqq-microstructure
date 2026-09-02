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

---

## 15. The cross-section: momentum is a null, and the overnight premium is half an opening print — `xsec_extract.py`, `xsec_backtest.py`, `xsec_ml.py`

Panel: the top 150 US tickers per month by **that month's** dollar volume, 1999-03 →
2026-03, minute bars collapsed to daily rows at extraction. 325 months, 1,014,170
ticker-days, 1,348 names. The universe is lagged one month (month T trades on file
T-1's list), and a name must also appear in month T's file to be priceable — that
still-liquid-next-month conditioning runs 14.1%/month attrition (worst 22.0%) and is
measured, not hidden. ETFs/ETNs are excluded by a frozen list (28/month). Two signals,
pre-declared, one pass each, no sweeps: Jegadeesh-Titman 12-2 momentum and
Lou-Polk-Skouras overnight persistence, quintiles, equal weight.

### The panel had landmines, and the diagnostics existed to find them

- **Exchange test symbols** (ZVZZ.T and seven relatives, plus TESTA/B/C) bought their
  way into the top-150 on fake dollar volume — 2,539 ticker-days of test prints
  gapping ×50/×0.02, found because they were the entire top of the detected-split
  list on the first run. Now dropped from the panel at load.
- **Splits are excluded, not adjusted**, and the tolerance had to learn that a
  split-day gap is ratio × that day's real overnight move: AAPL's 4:1 printed ×0.256,
  AMZN's 20:1 ×0.051, SHOP's 10:1 ×0.098. At 3.5% log tolerance the verifier reads
  **9/9 known splits detected**, and the detector independently caught BRK.B 50:1,
  SOXL 15:1, TVIX 1:10, NUGT 1:10, LRCX 10:1, NFLX 10:1 — and one decimal-shift
  glitch (CTXS ×10.1 on 2000-02-29), which the same mechanism removes.
- All multi-period returns are sums of valid daily log returns, so an excluded day's
  level break cannot leak into any window. Dividends land as negative overnight price
  moves the holder actually receives, so every overnight number below is understated.

### A. Cross-sectional momentum — an honest null in mega-caps

```
                       bps/day      t   Sharpe    CAGR%   maxDD%
  L/S Q5-Q1              +3.05   1.26     0.25     2.82    -71.8
  long tilt Q5-EW        +1.17   0.89     0.18     1.54    -49.5
  turnover 26%/month/leg; net at 5 bps one-way: +2.80 bps/day

  eras: 2000-03 +1.8  04-07 +2.3  08-11 +3.4  12-15 +5.0  16-19 +1.1
        20-23 +7.1  24-26 -1.5
  worst months: 2001-10 -29.8%  2009-04 -28.9%  2000-11 -23.4%
```

The direction matches the literature and so does everything else about it: half the
canonical magnitude (momentum lives in mid/small caps, this universe is the megacap
top-150), and the worst months are *the* momentum-crash dates — Oct 2001, Apr 2009,
the 2000-01 unwind — recovered blind by the machinery. t=1.26 over 26 years does not
clear any bar. The walk-forward ML on the same trade (below) is a clean null too:
monthly rank IC **-0.002**, L/S -1.45 bps/day against its own canary at -1.18. The
monthly total-return cross-section of the most liquid 150 names is efficient.
Recorded so it is not re-tried.

### B. Overnight persistence — strong, and then the discriminator fires

Rank on last month's mean overnight return, hold close→open, refresh monthly:

```
                       bps/day      t   Sharpe    CAGR%   maxDD%
  L/S Q5-Q1             +10.01   9.10     1.75    27.35    -34.6
  Q5 overnight          +11.92   8.50     1.64    32.76    -44.0
  Q5 - QQQ (tilt)        +7.14   8.90     1.72    19.05    -32.1

  tug-of-war: Q5-Q1 INTRADAY -12.22 bps/day (t=-6.39)
  eras: 1999 +61.3  2000-03 +17.7  04-07 +15.6  08-11 +9.5  12-15 +5.9
        16-19 -0.4  20-23 +1.2  24-26 +11.4
```

t=9.1 over 26 years, and the tug-of-war is exactly as Lou-Polk-Skouras predict — so
exact that close-to-close the spread is ~zero (+10.01 − 12.22 ≈ −1 with the momentum
skip). The whole effect is *when* returns arrive, not *which* names go up. That is
either a pure timing anomaly or **bid-ask bounce at the opening print**: a first-bar
trade at the ask against a close near the bid manufactures the same persistent fake
overnight gain and the same offsetting intraday loss. Trade-price data cannot tell
those apart from the table above, so the discriminator was pre-declared: sell the
same quintiles at 09:45, which does not capture the opening print.

```
  L/S exit 09:45         +4.81   3.83     0.74    11.37    -51.1
  Q5  exit 09:45         +7.28   4.93     0.95    17.90    -49.5
```

**52% of the spread is the opening print; 48% survives as a holdable return at
t=3.8.** Q5-Q1 gives back 5.2 bps in the first 15 minutes — 43% of the entire day's
reversal — and then fades slowly for six hours. Where the truth sits between +4.81
and +10.01 depends on what the 09:30 bar's open actually is: if it is the official
auction cross, a market-on-open sell captures it and the overpricing is real and
sellable; if it is an ask-side print after the auction, it was never available.
This dataset cannot say. **Underwrite the floor: +4.81 L/S, and a long tilt of
roughly +2.5 bps/day over QQQ overnight** (7.28 vs the panel's 4.78 QQQ overnight).

The era shape carries its own warning: the effect died in 2016-2019 — the window in
which Lou-Polk-Skouras was published — and revived to +11.4 in 2024-2026. Whether
the revival is regime or noise is not answerable in-sample.

### The ML ranker doubles the rule, mechanism pre-registered

`xsec_ml.py`: LightGBM over ten features per name-month (each a published effect),
targets rank-transformed within month so the market component cancels, walk-forward
by trade year, params inherited frozen from `momentum_ml.py`, and a leak canary
(targets permuted within training months) built in. On a synthetic panel with
planted truths the pipeline predicted, before touching real data, that the model
would roughly double the 1-month rule by finding 12-month overnight persistence.
On the real panel, 279 OOS months (2003-01 → 2026-03):

```
                       bps/day      t   Sharpe    maxDD%
  rule on_1m L/S         +6.20   6.41     1.33     -26.4
  ML L/S                +12.38  12.58     2.61     -20.5
  canary (shuffled)      +1.60   2.54     0.53     -19.7

  monthly rank IC +0.214 (t=17.5); top feature on_12m at 28% of gain;
  positive in all six 4-year eras; one losing year (2022) in 24.
```

The doubling replicated with the predicted mechanism on top. The canary's +1.60 is
inside the ±1.6 wobble observed across runs and configurations, an order of
magnitude under the live number, and is reported rather than reasoned away. Two
caveats: these ML rows ran on the panel one fix earlier (TESTB still present — 624
of 1.0M rows, 2001-02 only, affecting nothing after 2003), and the target includes
the opening print, so the model's floor — trained and evaluated on the 09:45 exit —
is untested. Given the rule loses half its spread there, assume the ML does too
until shown otherwise.

### Costs, and what is actually deployable

The L/S pays 4 crossings/day: break-even one-way cost is 2.50 bps at the open exit
and 1.20 bps at the floor — thin against realistic single-name spreads, and the
short leg assumes borrow on ~30 names every night. The deployable object is the
**tilt**: the stack's overnight leg (§13) already pays its two crossings on QQQ, so
switching it to the top quintile adds only the single-name-minus-QQQ spread
difference, for +7.1 bps/day at the open exit and ~+2.5 at the floor, on top of
QQQ's own +5.1. Correlation with the momentum leg is +0.07.

### What would settle it

1. **Official auction prices** (the opening/closing cross) in place of first/last
   bar trades — this alone decides floor vs ceiling, and it is the same NBBO-class
   caveat §5 left open for the microstructure edge.
2. The **monthly HF replay** forward test, exactly as §12 runs it for the QQQ stack:
   the extractor is resumable and the strategy trades at two fixed timestamps.
3. If pursued, the ML re-targeted on the 09:45 exit, same pipeline, same canary.

Until (1), every bps figure in this section is a range, not a number, and the floor
of the range is the only one that should be underwritten.

### 15b. The discriminators fire — in favor of the ceiling

Both tests from "what would settle it" were run the same day. First, official
prices: `xsec_auction_check.py` fetches Yahoo's official daily open/close (the
primary-exchange auction prints) for the 79 most-frequent Q5/Q1 members since
2012-01 and compares overnight RETURNS on identical name-days, so Yahoo's split
back-adjustment cancels. The script was validated against a stub price source with
a planted +8 bps opening-print inflation, which it recovered at +7.6. On the real
thing (75 names resolved, 208,197 matched name-days):

```
  corr(panel, yahoo overnight)  0.9905     |gap| mean 4.40  median 2.10 bps
  signed gap (panel - yahoo):   Q5-frequent -0.04   Q1-frequent -0.26
  bounce loading (Q5-Q1):       +0.23 bps/day

  B on identical name-days (3,472 days):
    L/S panel prices     +5.64 bps/d   t=4.23
    L/S yahoo official   +5.73 bps/d   t=4.29     print component -0.08 bps/day
```

**The bounce hypothesis is refuted.** The first-bar open IS the auction cross for
these names, and the premium is identical at official prices: what Q5-Q1 gives
back by 09:45 is genuine post-auction reversion — overpricing a market-on-open
order sells INTO, not a print it cannot have. The ceiling is the tradeable number.
(The +5.6 on this sample vs +10.0 full-history is the window, not the source: 2012+
spans the 2016-2023 trough in the era table. Caveats: listed names only — 4 of 79
including renames failed to resolve — and one mapping quirk, CMCS.A.)

Second, the model at the floor anyway: `xsec_ml.py` block C re-targets the same
pipeline on the 09:45 exit (features, params, canary unchanged; on the synthetic
panel, where p15 equals the open, block C prints bit-for-bit identical to block B):

```
                       bps/day      t   Sharpe    maxDD%
  rule on_1m L/S         +2.46   2.21     0.46     -39.8
  ML L/S                 +7.29   6.61     1.37     -36.2
  canary (shuffled)      +0.04   0.05     0.01

  IC +0.108 (t=9.2); eras 2003-2026: +7.7 +7.2 +6.2 +11.6 +5.1 +6.6 +6.6
```

The rule keeps 40% of its spread at the floor; the model keeps 60% — and is
**positive in all seven 4-year eras**, including 2016-2019 where the rule went
negative. The floor model leans on a broader feature mix (mom_12_2, rng_3m, dvol
~11-12% each) where the ceiling model leans on on_12m at 29%: predicting the
post-reversion residual takes more than persistence alone, and the model finds it.

One number reported rather than reasoned away: the ceiling-target canary repeats
at +1.6 to +1.8 bps/day across runs (this run +1.79, t=2.85) while the cc and
floor canaries sit at zero. Treat ~+1.8 as that construction's achievable-by-
chance floor; the live +12.23 clears it by 7x, and the honest subtraction still
doubles the rule.

**Revised bottom line.** At MOO/MOC execution the ceiling stands: rule L/S +10.0
(t=9.1), tilt over QQQ +7.1 (t=9.0), ML L/S +12.2 (Sharpe 2.6, break-even one-way
cost ~3.1 bps), with the floor's +7.3 at t=6.6 as the robustness margin if
execution slips 15 minutes. What remains before any capital discussion is what
§12 demanded of the QQQ stack: the frozen-model forward replay as each month's
file lands — `xsec_ml.py --save-model` freezes production models, and
`xsec_replay.py` evaluates every month that postdates the freeze.

---

## 16. Signal #6 and Stack v2 — `xsec_intraday.py`, `stack_v2.py`

### Signal #6: cross-sectional intraday continuation — real, and the toll is bigger

At 10:30, rank the universe by its own first hour, hold Q5−Q1 to the close.
Pre-declared direction was continuation (the time-series version is §8/§11's
signal; HKS 2010 is the cross-sectional prior), with the sign genuinely at risk —
short-horizon cross-sections often reverse. Continuation won:

```
                       bps/day      t   Sharpe    CAGR%   maxDD%      6,787 days
  L/S Q5-Q1              +7.70   5.90     1.14    19.65    -30.9
  long tilt Q5-EW        +3.05   4.30     0.83     7.53    -31.9

  eras: 1999 +40.0   2000-03 +21.0   04-07 +2.5   08-11 +11.2
        12-15 +0.4   16-19 -0.2     20-23 +4.1   24-26 +9.6
```

The third real cross-sectional structure in this panel — and §1's verdict at
daily frequency: the signal was never the problem, the toll is. Break-even is
1.92 bps one-way and the 10:30 entries CROSS the spread (they are not auctions):
at the pre-declared c=2.5 the L/S nets **−2.30 bps/day**. Its correlation with
the QQQ momentum leg is +0.33 (same family) and ~0 with everything overnight.
Recorded, not traded. The residual use is as a which-name tilt inside entries a
book is already making, where it costs no new crossings.

### A truncated-input incident, recorded

The first stack run read the QQQ legs from `daily_hf_QQQ.parquet` rebuilt from
the committed `data/hf_bars/` — which turn out to be a partial subset (the
rebuilt QQQ spine ends 2005-01). The MOM leg silently covered 1999–2005, the
hottest momentum era in the sample, and the combined rows printed +19.5 bps/day.
Two rules now hold in `stack_v2.py`: the QQQ legs are derived from the xsec
panel itself (QQQ/QQQQ sit in every month's top-150 with open/p60/close), and
**every row prints its own date window and day count**. The panel-derived legs
then reproduce the earlier results from an independent price path: MOM +3.72
net vs §11's sign rule (3.94 gross − 0.34 cost), and v1 +8.26 vs §13's +8.42.

### Stack v2: the same stack with one leg swapped, on identical days

Legs net of stated costs (basket crossings c=1.0 bps one-way per §15b's MOO/MOC
finding; QQQ legs at the house 0.34 round trip):

```
                       bps/day      t   Sharpe    %/mo        window
  QQQ_ON                 +4.56   4.14     0.80   +0.87   1999-2026 (6,798d)
  MOM                    +3.72   2.53     0.49   +0.63   1999-2026 (6,803d)
  ON  (ML-Q5 basket)     +8.50   5.65     1.17   +1.66   2003-2026 (5,843d)
  NEU (basket - QQQ)     +4.09   5.22     1.09   +0.82   2003-2026 (5,841d)

  correlations: QQQ_ON/MOM +0.01 (the §13 stack's zero), ON/MOM +0.01,
  ON/QQQ_ON +0.88 (the basket is market-overnight plus edge -- the gain below is
  added edge, not diversification), NEU/QQQ_ON +0.39 (a 1:1 QQQ hedge
  UNDER-hedges: Q5 names carry overnight beta > 1; a beta-scaled hedge is a
  future pre-declared spec, not retrofitted here).
```

The comparison the exercise was for — v1 (§13's configuration) against v2 (same
stack, overnight leg swapped to the ML-Q5 basket), on the identical 5,843 days:

```
                       bps/day      t   Sharpe    CAGR%   maxDD%    %/mo
  v1 @ v2 window          +5.88   3.66     0.76    13.79    -43.7   +1.08
  v2  (ON + MOM)         +10.65   5.45     1.13    27.14    -43.1   +2.02
  v2n (NEU + MOM)         +6.24   4.27     0.89    15.21    -31.2   +1.19

  v2 by era: 2000-03 +8.1  04-07 +13.5  08-11 +15.4  12-15 +8.5
             16-19 +5.7   20-23 +7.0   24-26 +17.4     (positive in all seven)
```

**The stack's rate roughly doubles — +1.08%/mo to +2.02%/mo, Sharpe 0.76 to 1.13
— by swapping one leg, at stated costs, with the max drawdown unchanged.** v2's
worst era (+5.7, 2016-2019) is close to v1's average. At the pessimistic c=2.5
the upgrade still holds: v2 ≈ +7.7 vs v1's +5.9. Adding Signal #6 (v2x) makes
everything worse — its cost drag and −98% standalone drawdown contaminate every
combination — so the book is v2, with v2n (Sharpe 0.89, maxDD −31%) as the base
configuration if leverage is ever considered.

### What gates deployment — unchanged

The overnight leg answers to §15b's frozen-model replay (`xsec_replay.py`,
frozen 2026-08-20); the MOM leg remains under §12's monthly verdict, which
currently says do not fund it. Nothing here shortens either probation: v2 is
the configuration the forward test is now measuring, and no leverage decision
precedes forward months.

---

## 17. Does selling premium have a season? — `iv_regime.py`

§8 priced every 0DTE structure against an ASSUMED IV and showed the ranking
pivots entirely on it; §9's one measured day (12.5%) flipped it. Before paying
for the full OPRA pull, the free gate check: ^VIX1D (CBOE's 1-day SPX vol, the
0DTE era's own gauge, live 2023-04) scaled by the QQQ/SPY realised-vol ratio
measured on the proxy's own window — 1.309, against a dot-com-dominated 1.451
full-history ratio that inflated a first draft and is kept in the output as a
warning about window-matching.

### The proxy hit the only ground truth exactly

```
  2026-07-15:  ^VIX1D 9.5 x 1.309 = 12.5%   vs 12.5% measured from the chain (§9)
```

One point is one point, but the check existed before the answer did.

### The season, 735 days 2023-04 → 2026-03

```
  proxy: p10 11.8  p25 13.6  median 16.8  p75 21.4  p90 27.5
  days above 12.5 / 14.8 (break-even) / 16.0:   84.8% / 63.3% / 54.1%

  year  mean IV%  >14.8%   impl sd  real sd  impl/real    adj
  2023      17.3     65%       101       66       1.53   1.23
  2024      16.7     52%        97       72       1.34   1.08
  2025      21.3     70%       123      106       1.16   0.93
  2026      22.0     79%       128       72       1.78   1.42
```

`impl/real` is an upper bound — VIX1D prices the overnight gap the 10:30→close
window never realises; `adj` nets out the measured intraday variance share
(0.64). Read the adj column: **sellers were paid in three of four years (+8%
to +42% over realised), and in 2025 premium was cheap** — a year of
systematically selling would have collected less than the risk realised,
before tails and friction.

### Verdict

The season exists and is **conditional**: the gate is open ~63% of days, the
paid margin is thin on average and vanished for a full year. That supports
exactly one next step and rules out another. It justifies pricing the OPRA
date-range pull (`OPRA.PILLAR` `cbbo-1m`, `QQQ.OPT` parent, 2023→now) so the
structures can be valued against per-day measured chains — and it rules out
shipping any sell-premium overlay off this proxy alone. Three reasons the adj
margin overstates what a seller keeps: mean-implied vs realised-std ignores
the convex tail drag §8 quantified (one −12% day = 1.3 years of −1%-put
credits); §9 measured option friction at ~2.8% of premium per leg and a
defined-risk structure pays it twice; and the proxy has exactly one
ground-truth point. Whatever survives the real chains must still be
defined-risk and IV-gated — §8's tail rules are not relaxed by a thin average
edge, and 2025 shows the gate must be able to say "stand aside" for a year.

---

## 18. The chains arrive: §8 priced at real premiums — `opra_pull.py`, `opra_value.py`

The purchase §17 earned, made small: instead of the portal's $412 for 221 GB of
the whole OPRA universe (or ~$110 filtered to QQQ full days), ten ET minutes per
day around the 10:30 entry — QQQ.OPT parent, cbbo-1m, one parquet per day,
resumable, the bill printed before the spend. **851 sessions for ~$21.** The
valuer takes each day's minute closest to 10:30 ET (DST-correct; opra_load.py's
fixed-UTC snapshot constants carry the §10 bug and were not reused), recovers
spot by put-call parity and IV from the ATM straddle, and prices §8's
pre-declared structures in the momentum direction at the quotes a real order
faces — ask when buying, bid when selling. Signal and settlement come from the
equity panel. 749 days valued (2023-04 → 2026-03); 5 holidays; 97 days beyond
the panel's last month await `xsec_extract.py`.

### Two vendors, sub-basis-point agreement

```
  parity spot vs panel p60:  |diff| mean 0.4 bps   p95 1.0 bps
```

Databento option chains and the HuggingFace equity bars have no common
ancestor, and they agree at the level of rounding. This validates both
pipelines end to end — the DST windows, the OSI parsing, the parity read, and
retroactively the panel itself.

### Measured IV — and the proxy fails its exam

```
  0DTE IV at 10:30, 749 days:  p10 8.3   median 12.1   p90 18.5
  days above the 14.8 break-even:  25.5%      (§17's proxy said 63.3%)
```

§8's working assumption ("QQQ 0DTE IV rarely prints below ~15%") is measured
FALSE at the 10:30 entry, and §17's proxy — right to the decimal on its one
ground-truth day — was badly wrong in distribution. The wedge is the one §17's
adj column partially flagged: ^VIX1D prices a full day including the overnight
gap, while the 10:30 straddle prices the remaining session, and a realised-vol
ratio does not map one onto the other. The selling season is a quarter of
days, not two thirds. The proxy is retired for gating; measured IV now costs
~$0.50/month to keep current via the same slice pull.

### The verdict: the long side is dead, the defined-risk short side is not

```
  EV/day, bps of spot, GROSS      all days          IV<=14.8         IV>14.8
  long ATM (momentum)         +0.4  t=0.2  37%   -0.3 t=-0.2 36%   +2.4 t=0.4 40%
  put/call spread .5/1.5      +3.4  t=4.6  87%   +2.7 t= 3.9 90%   +5.4 t=2.7 81%
  iron condor .5/1.0          +2.3  t=3.6  70%   +2.5 t= 3.8 76%   +1.5 t=1.0 53%

  commissions ($0.65/contract/side, entry legs): long -0.14  spread -0.27  condor -0.54

  full period:  spread +3.37 bps/d  t=4.65  Sharpe 2.70  maxDD -2.1%  = +0.71%/mo
                condor +2.28 bps/d  t=3.57  Sharpe 2.07  maxDD -2.9%  = +0.48%/mo
  by year (spread): 2023 +4.7   2024 +4.0   2025 +2.5   2026(Q1) +0.2
```

Three findings, none the expected one:

- **Buying the momentum in options is dead.** +0.4 bps/day at t=0.2, 37% win
  rate, negative even in its own supposed low-IV regime. §9's one-day +23.65
  EV was one draw; over 749 days the ~5 bps honest drift (§8b) does not clear
  ~2.8%-of-premium friction per leg. The §1 pattern for the third time: the
  signal was never the problem.
- **The momentum-direction credit spread survives contact with real quotes.**
  +3.4 gross / +3.1 net bps/day, t=4.65, an 87% win rate with a −2.1% max
  drawdown, positive every year. It monetises three things at once — the
  premium, the skew, and the direction — which is why it clears the tolls that
  killed the long side.
- **The gate is not the discriminator.** Both IV cells are positive for the
  spread. §17's whole framing — sell only when IV is high — turns out to
  matter less than the structure itself. Reported, not tuned: the cells were
  pre-declared and both are printed.

### The tail check, finally at real prices

```
  worst 5 spread days:  -94  -94  -91  -91  -85 bps
```

That −94 is the cap by construction: (1% strike width − credit). §8's
arithmetic — one −12% day costs 65 to 329 days of credit — was for NAKED
shorts; the defined-risk wing §8 demanded bounds the same catastrophe at
roughly **28 days of credit**, and the sample's worst days sat exactly on the
cap and no further. The design survived its own worst case. The honest
remainder: 2023-2026 contains no crash, so the cap has been touched, not
stress-tested; an assignment on partial-ITM days is approximated as cash
settlement and actually leaves overnight stock exposure (closing shorts at
15:55 would remove it at the cost of a spread crossing); everything is
held-to-expiry with no management.

### The integration number, measured

The spread conditions on the same first-hour signal as the stack's MOM leg, so
its correlation against that leg decides whether it diversifies or doubles.
`stack_v2.py` carries it as the OPT leg, and the answer is the honest middle:

```
  corr(OPT, MOM) +0.67    corr(OPT, ON) -0.01    corr(OPT, QQQ_ON) -0.04
  OPT               +3.09 bps/d  t=4.26  Sharpe 2.47  maxDD  -2.2%   (749d)
  v2o (ON+MOM+OPT) +19.90 bps/d  t=3.32  Sharpe 1.93  maxDD -24.9%  = +3.97%/mo
                    (2023-04 .. 2026-03 only)
```

At +0.67 the spread is mostly THE SAME BET as the momentum leg — expressed
better: Sharpe 2.47 against the stock leg's 0.49 on this window, with the tail
capped by the wing — and uncorrelated with everything overnight. It joins the
book as more momentum, not as diversification: sizing must treat MOM + OPT as
one exposure family, and a regime that wrong-foots the first-hour signal
(March 2026 was one; the spread's 2026 quarter ran +0.2) hits both at once.
v2o's +3.97%/mo is a three-year hot-window number and is NOT comparable to
v2's 23-year +2.02%/mo — the same-window read is that the overlay adds its
~+3 bps/day nearly additively on top of the v2 book.

Probation unchanged: the monthly forward replay (a ~$0.50 slice pull plus the
panel extension), no leverage before forward months, and §12's standing
instruction to diagnose rather than filter when a month goes wrong. The
options thread, opened by §8 on an invented IV, closes on 749 measured days
with one dead hypothesis, one retired proxy, and one live, bounded,
+0.71%/month finding that the book must size as momentum.

---

## 19. The fan and the tripwires: Monte Carlo on the measured book — `mc_risk.py`

> "Given the returns I've actually measured, what range of outcomes could
> reasonably happen, and what future result would convince me that the
> backtest's edge is real?"

That question is the whole scope. Monte Carlo for *selecting* trades was
considered and rejected as waste — resampling one history cannot rank
strategies beyond what the measured means already say, and tuning against
resampled noise is overfitting with extra steps. What survives are the two
uses simulation is actually good at: the **range** (the fan of five-year
outcomes around the one path history happened to draw) and the **verdict**
(pass/fail thresholds for the forward test, committed before the months they
will judge).

Method: circular block bootstrap of the measured daily series — 21-day
blocks, so volatility clustering and within-month autocorrelation are
carried; 10,000 paths; seed 7; financing at 5%/yr on the borrowed fraction.
Nothing is simulated except the resampling: every path is a rearrangement of
days that actually happened. The machinery was validated on planted truth
before touching real series: on iid noise with a known mean the GO
thresholds print 14.9/10.5/7.3 bps (3/6/12mo) against the closed-form
14.7/10.4/7.33 and P(pass|real) 24% against the analytic 24.5%; on an
AR(0.5) series the blocks widen the 3-month standard error ×1.66 (an iid
bootstrap would leave it unchanged and overstate every confidence below;
theory says ×1.73).

### A. The range: 10,000 alternate five-years (p95 = the bad-tail 5% end)

```
  v2 (ON+MOM), 5,843 measured days:
  lev   CAGR p5/p50/p95     maxDD p50/p95   worst-mo p95  P(mo<-20%)  P(5y loss)
  1.0    8.4  27.4   48.4    -26.8  -48.8       -40.5%       25.7%       1.1%
  1.5    7.4  37.3   72.8    -39.1  -65.6       -61.0%       58.3%       1.9%
  2.0    4.5  46.0   98.7    -50.1  -77.9       -81.5%       84.1%       3.3%
  3.0   -6.7  57.9  151.9    -68.1  -92.3      -122.4%       97.1%       7.4%

  v2n (NEU+MOM), 5,841 days:
  1.0    2.2  15.1   30.7    -22.4  -36.4       -17.6%        1.1%       2.5%
  1.5   -0.4  19.0   43.9    -33.0  -52.5       -26.5%       23.4%       5.4%
  2.0   -3.7  22.1   57.2    -42.9  -65.1       -35.5%       52.1%       8.3%
  3.0  -12.2  25.7   83.3    -59.9  -82.9       -53.5%       90.7%      14.5%

  v2o (ON+MOM+OPT), 749 days, lev 1.0 only:
         34.1 60.0   88.5    -24.6  -37.7       -18.8%        0.0%       0.0%
```

Three decisions fall out:

- **The fan is wider than the path.** v2 unlevered shows a p95 max drawdown
  of −48.8% against history's single −43.1%, and a 1-in-4 chance of at least
  one month worse than −20% somewhere in five years. That is the price of
  admission the backtest line never displays, and it is what a funder must
  accept — not 27% p50 CAGR.
- **§16's leverage plan dies here.** "v2n as the base configuration if
  leverage is ever considered" does not survive financing: v2n at 1.5× (p50
  CAGR 19.0%, p95 maxDD −52.5%) and at 2× (22.1%, −65.1%) are both dominated
  by plain v2 at 1× (27.4%, −48.8%) on return AND tail. Corrected role: v2n
  is the low-tail configuration at 1× — worst-month p95 −17.6% vs v2's
  −40.5%, P(−20% month) 1.1% vs 25.7% — and anyone wanting more return
  should move to v2 unlevered, not lever the neutral book. At 3× the 21-day
  sums cross −100% (ruin territory) for both books; there is no leverage
  row this Monte Carlo endorses.
- **v2o's fan is an illustration, not a plan.** Five-year paths resampled
  from 749 bull-window days, with §18's spread edge already decaying by year
  (+4.7 → +0.2), print a p50 CAGR of 60% and zero loss probability. Upper
  bound; the honest read of the overlay stays §18's: ~+3 bps/day added
  nearly additively, sized as momentum.

### B. The verdict, pre-registered

Thresholds computed from the frozen walk-forward series alone (both end
2026-03, the training cutoff of the 2026-08-20 freeze). GO = the 95th
percentile of the no-edge world (same days, demeaned): a forward mean above
it has <5% probability if the edge is zero. KILL = the 5th percentile of the
as-measured world: a forward mean below it has <5% probability if the
backtest's edge is real. Units: mean bps/day over the forward window.

```
  replay metric: ON long/short, gross (measured +12.2 bps/day, 5,843 days)
  horizon    GO >   P(pass|real)   KILL <   P(kill|none)
      3mo    15.1        35%         -2.6        38%
      6mo    11.1        56%          1.6        61%
     12mo     7.8        83%          4.6        84%
     24mo     5.5        98%          6.8        98%     <- worlds fully separate

  paper-log metric: Q5 minus QQQ overnight tilt, gross (+6.4 bps/day, 5,841 days)
      3mo    12.5        18%         -5.0        24%
      6mo     9.6        25%         -2.1        36%
     12mo     7.0        41%          0.4        57%
     24mo     4.8        71%          2.0        78%     <- not yet separate at 24mo
```

The clock. The replay's first four quasi-holdout months — 2026-04 +5.00%,
-05 +4.88%, -06 +11.33%, -07 −1.91%, **+19.3% cumulative ≈ +23 bps/day** —
clear even the 3-month GO of 15.1. But their outcomes were known before
these thresholds were committed, so they count as a consistency check, not
as verdict evidence. The binding window starts with the first month that
completes after this section's commit: **2026-08**. First (weak) read early
November 2026 at the 3-month row; first high-power read early August 2027 at
the 12-month row, where a real edge passes 83% of the time and a dead one is
caught 84% of the time. The tilt metric, with half the mean, stays honest
longer: even at 24 months a real tilt clears its GO only 71% of the time —
patience is the modal outcome there and that is what the power column is for.

The rule, mechanical from here: average the replay's forward months into
bps/day and compare to the row for the elapsed horizon. Above GO — the edge
is hard to dismiss; size it off the range table, not the backtest line.
Below KILL — the backtest is hard to believe; stop and diagnose (§12's
instruction, unchanged). Between the two — neither vindication nor refutation,
only more waiting, which at short horizons is the likeliest outcome even
when everything is real. `mc_risk.py` reproduces these numbers exactly
(seed 7, inputs frozen at the cutoff); the binding copy is this section,
whose git timestamp precedes every month it will judge.

---

## 20. The overlays pass, and July gets its post-mortem — `overlay.py`, `diagnose.py`

Two sizing/hedging layers, specs and PASS/FAIL criteria committed before the
run (overlay.py's header), neither touching the frozen models nor the
registered metrics — §19's clock ran untouched through all of this. Both
were validated on planted truth first (beta 1.4 recovered to 0.02, beta=1
reproduces NEU to the float, no lookahead at hand-checked positions, the
criterion firing in both directions on synthetic worlds), and the vol
overlay carried a stated risk: v2 earned most in storms (2008-11, +15.4),
so de-levering storms could FAIL. It didn't. All four pre-declared verdicts
passed.

### The beta the 1:1 hedge was missing

§16 conjectured Q5 names carry overnight beta > 1. Measured (252-day
trailing, lagged, clamped [0.5, 2.0]):

```
  beta: mean 1.25   p5 0.87   p95 1.67   LAST 1.55      (5,715 days, 2003-2026)
  corr with QQQ_ON, same window:   NEU (1:1) +0.40   NEUb +0.03    -> PASS (<0.20)

                bps/day      t   Sharpe   CAGR%   maxDD%    %/mo
  NEU  (1:1)      +4.17   5.24     1.10   10.57    -31.7   +0.84
  NEUb (beta)     +3.40   4.85     1.02    8.56    -26.6   +0.69
```

The hedge does exactly what it was specified to do: residual market
exposure +0.40 -> +0.03. The 0.77 bps/day it costs is the QQQ overnight
premium the extra ~0.25 of hedge no longer collects — NEUb is the purest
number in the repo, the selection alpha net of ALL market exposure, and it
is +3.40 at t=4.85. The last reading matters live: **today's basket runs
overnight beta ~1.55**, so the 1:1-hedged book is quietly long ~0.55 of
QQQ's overnight without knowing it. Honest ranking note: PASSing its own
criterion did not earn NEUb a slot in the book table below — v2n-scaled
dominates v2nb-scaled on mean and tail alike. The hedge's value is
diagnostic (what is the alpha really?) and as the go-to configuration if
market-overnight exposure ever needs to be exactly zero.

### De-levering the storms: PASS on all three books

Exposure = min(1, target/vol), causal, capped at 1 — the mirror image of
the margin §19 vetoed. Raw and scaled on identical days (post-burn-in
window 2004-02 -> 2026-03, so raw rows differ slightly from §16's
full-window numbers); MC columns from §19's bootstrap at 1x:

```
                bps/day    t  Sharpe  %/mo | MC: CAGR p5   p50  maxDD p95  w-mo p95  P(-20%mo)  p50/|DD|
  v2   raw       +10.84 5.38   1.14  +2.06 |       9.2%  27.9%    -48.3%    -41.9%      27.9%      0.58
       scaled     +8.90 6.53   1.39  +1.77 |      11.1%  23.7%    -27.0%    -15.8%       0.0%      0.88
  v2n  raw        +6.74 4.52   0.96  +1.29 |       3.7%  16.5%    -34.5%    -17.6%       1.2%      0.48
       scaled     +5.41 5.06   1.08  +1.08 |       4.2%  13.6%    -23.2%     -8.7%       0.0%      0.59
  v2nb raw        +5.50 3.74   0.81  +1.04 |       1.0%  13.0%    -35.5%    -15.5%       0.0%      0.37
       scaled     +4.13 3.88   0.83  +0.80 |       0.8%  10.0%    -27.1%     -8.8%       0.0%      0.37*
                                             (* marginal at print precision)
  v2 exposure: mean 0.84, at the cap 51% of days, p5 0.39
  v2 by era, scaled | raw: 04-07 12.9|13.9   08-11 10.9|15.4   12-15 7.7|8.5
                           16-19  5.2|5.7    20-23  5.2|7.0    24-27 13.8|17.4
```

The headline is the v2 row. Giving up 18% of the mean (10.84 -> 8.90)
buys: the MC max drawdown nearly halved (-48.3% -> -27.0%), the worst-month
tail cut by 62% (-41.9% -> -15.8%), the chance of a -20% month from 27.9%
to zero — and the 5%-unlucky-world CAGR RISES (9.2% -> 11.1%). That last
one is §19's leverage table in mirror image: margin improved the median by
degrading the bad worlds; de-levering storms improves the bad worlds while
keeping most of the median. Return per unit of tail: 0.58 -> 0.88. Sharpe
1.14 -> 1.39, and the t-stat rises despite the lower mean.

The stated risk showed up and was survivable: storms did pay (08-11 scaled
kept 10.9 of raw's 15.4; 24-27 kept 13.8 of 17.4), but the premium in
storms is not proportional to variance — the overlay kept ~70% of the
storm-era mean at half the storm-era risk, and every era stayed positive.

The 1x menu this leaves, for the go-live day (nothing about probation
moves): raw v2 is the +2.06%/mo book at a -48% MC tail; scaled v2 is
+1.77%/mo at -27% with the highest Sharpe in the repo (1.39); scaled v2n
is the calm book (+1.08%/mo, worst-month p95 -8.7%). The forward metrics
are configuration-independent, so this choice is sizing, not evidence.
Whether modest leverage ON THE SCALED book beats raw v2 is a legitimate
future question for the same MC machinery — after the forward verdict,
not before.

### July 2026, diagnosed — `diagnose.py`

The one negative replay month (-1.91%, §19's quasi-holdout), decomposed by
the harness built for §12's "diagnose, don't filter" (selection rebuilt
exactly as the replay's, reconciled to `portfolio()` at 1e-9):

```
  2026-07: 52 eligible, Q5/Q1 = 10 each; L/S -8.70 bps/day x 22 nights (ref +12.2)
    market    QQQ overnight            +0.37
    breadth   universe EW minus QQQ    -2.36
    long sel  Q5 minus universe        -1.97
    short sel universe minus Q1        -6.73     <- the month
    identity  L/S                      -8.70
  worst nights: 07-28 -581.6 (MU -697)   07-07 -544.9 (AMAT -834)
                07-17 -395.3 (LRCX -548)
  data flags: none -- 20 names x 22 nights, zero exclusions, zero missing
```

The month was the SHORT side. The market was fine (+0.37) and the longs
only mildly lagged; the loss sat in Q1 rallying overnight — XOM +30.0,
CRWD +22.1, WFC +20.2, MA +18.6 bps/night: an old-economy/financials
rotation month that the momentum-flavored short book leaned against. The
long side was an earnings-dispersion wash (LRCX +70.0 and AMAT +24.1
against QCOM -62.2 and PLTR -39.5), and the worst nights are single-name
earnings gaps — the known texture of a 10-name book in July. The floor
model printed -26.91 the same month: the opening prints cushioned July
rather than inflating it. Under §12 and §19 together the verdict is
mechanical: -8.70 for one month sits inside the fan (the 3-month table
calls anything between -2.6 and +15.1 "waiting"), the data are clean, and
no rule changes.

One blemish surfaced in the open: **SGOV — a T-bill ETF — sat in Q1** (it
postdates the frozen ETF exclusion list). Its contribution was -0.3
bps/night of nothing, exactly what a cash ETF does overnight; the cost is
a wasted short slot, not a wrong number. The frozen experiment keeps its
universe rule as-is (the model was trained under it; changing it mid-probation
would create train/serve skew), and the ETF list gets amended at the next
deliberate re-freeze, which is its own dated commit. Found, sized
(negligible), deferred deliberately — that is what the harness is for.

---

## 21. Is it alpha or beta, and do the option numbers survive real quotes? — `factor.py`, `optbacktest.py`

Two questions the earlier sections deferred, answered on the real panel.

### The factor decomposition: v2's alpha is real, and so is its market beta

Every book regressed on QQQ + SPY overnight (and QQQ intraday), Newey-West
HAC t-stats. The two session-orthogonality falsification checks passed —
ON on the intraday factor beta +0.00, MOM on the overnight factor beta
−0.01 — so no session leaks into the other.

```
  book   alpha bps/d  t_HAC  residSharpe  R2   betas (mkt_on / spy_on)
  ON        +2.45     2.81      0.71      0.75   +0.20 / +1.19
  NEU       +1.13     1.47      0.36      0.15   -0.09 / +0.41
  MOM       +4.42     2.70      0.56      0.00   (vs intraday: -0.03)
  v2        +4.88     3.26      0.74      0.47   +0.18 / +1.24
  v2n       +3.45     2.32      0.54      0.05   -0.03 / +0.37
```

Read carefully, this is the most important table in the file, and it cuts
both ways:

- **The alpha is real.** v2's intercept is **+4.88 bps/day at t_HAC = 3.26**,
  residual Sharpe 0.74 — a return no combination of QQQ/SPY overnight and QQQ
  intraday reproduces, significant after the HAC penalty that the
  autocorrelation warrants. ON+MOM is its own source, not repackaged index
  exposure. v2n keeps a significant +3.45 (t 2.32) too. MOM standalone is
  near-**pure** alpha (R² 0.00, t 2.70) — uncorrelated with the market, just
  noisy on its own (Sharpe 0.49).
- **But v2 is about half market-overnight beta.** R² 0.47 and a total
  overnight-market beta near **1.4** mean roughly half of v2's raw +10.65
  bps/day is the overnight equity-risk premium levered ~1.4×, not selection
  skill. That premium is real and harvestable, but it is beta — it will draw
  down when the overnight market does, and it is not what a fee is paid for.
  The pure-selection piece is ~+4.9 bps/day (~1%/mo).
- **A collinearity caveat, stated so the betas are not over-read.** QQQ and
  SPY overnight returns are ~0.9 correlated, so the individual split
  (+0.20 QQQ / +1.24 SPY) is unstable — the regression hands the shared
  component mostly to SPY. The trustworthy quantity is the **sum**, ~1.4,
  which matches §20's independently measured basket overnight beta of ~1.55.
  Do not read "v2 is a SPY trade"; read "v2 carries ~1.4 units of
  overnight-market beta, however it is split."

The decision this forces is the one §20 already previewed, now on a rigorous
basis: **v2** is max return but half of it is market-overnight beta (spy_on
1.24); **v2n** is the cleaner, lower-beta expression (beta 0.37, alpha +3.45
still significant) at a lower raw return; **NEU alone** over-hedges into
statistically insignificant alpha (t 1.47) and is a tail tool, not a return
source. Nothing here is killed — v2 clears the "real alpha" bar — but "v2 =
+2.02%/mo" is honestly restated as "≈half selection alpha, ≈half levered
overnight beta."

### The option backtest: real quotes confirm §18, and the model flatters ~3×

`optbacktest.py` gives the structures the equity books' full treatment. Real
mode reproduces §18 exactly on measured chains, and comparing it to model
mode on the SAME QQQ structure is the whole lesson in one line:

```
  QQQ momentum credit spread, 833 days 2023-04 -> 2026-07
  REAL  (measured bid/ask):        +2.92 bps/d  Sharpe 2.32  win 86%  maxDD -2.8%
  MODEL (mid fills, IV x1.2):      +8.45 bps/d  Sharpe 4.69  win 82%  maxDD -5.8%
  QQQ condor, REAL:                +1.48 bps/d  Sharpe 1.33  win 69%
```

The **model overstates the same structure by ~3×** (8.45 vs 2.92). That gap
is exactly what the model omits: the 0DTE bid/ask a real order crosses, mid
fills, and — in model mode — an assumed IV multiple instead of the measured
premium. So every model-mode number is an **upper bound, not an edge**: the
AAPL directional model printing +8.77 bps/day at Sharpe 4.29 is the same
flattery (and directional-only at mult 1.0, with a −0.5 bps/day 2000–2003
era inside it). The only tradeable options number in the repo remains §18's
real-chain QQQ credit spread, +2.92 net, decaying by year (2023 +4.3 → 2026
+0.8), and it answers to the §19 forward clock like everything else.

The through-line of both tests: the edges that were real stay real under the
harsher lens (v2 alpha t 3.26; QQQ spread at measured quotes), and the harsher
lens is exactly what stops the flattering numbers — half of v2's raw return,
and two-thirds of the model option Sharpe — from being mistaken for skill.

---

## 22. The overnight edge is broad, not a few-name quirk — `xsec_replicate.py`

The forward test answers "is the edge real?" over twelve months. This answers
a different, faster question with data already in hand: **is the overnight
edge a broad market phenomenon, or does it live in a handful of the 150
names?** The test partitions the universe by a stable md5(ticker) hash — a
name is in the same disjoint slice every month — and runs the persistence
signal INDEPENDENTLY inside each slice, each ranked only against itself. A
real, broad edge appears in every disjoint slice; a data-mined quirk hides in
one. Survivorship-free, no new data.

### It replicates in every well-powered disjoint slice

```
  signal on_12m, 462 names, 316 months 2000-04 -> 2026-07
  full universe:            +13.16 bps/day  t 12.9   IC +0.197
  3-way hash split:  +13.65 / +14.74 / +10.47   weakest t 7.5   -> BROAD
  2-way random:      +12.78 / +13.47            weakest t 10.4  -> BROAD

  signal on_1m, 766 names:
  full universe:             +7.95 bps/day  t 8.0    IC +0.102
  3-way hash split:   +9.42 / +8.98 / +6.85    weakest t 4.8   -> BROAD
  2-way random:       +7.72 / +8.34            weakest t 6.8   -> BROAD
```

Every disjoint slice — names the others never touch — reproduces the edge at
t ≥ 4.8, on both the 12-month and 1-month persistence signals, across two
independent partitioning schemes. The edge is a property of the *population*
of names, not of a lucky few. This is the single strongest piece of evidence
in the file that the overnight book is not curve-fit to specific tickers, and
it arrived without spending a forward month.

### The honest caveat, and a tool that no longer cries wolf

At `--k 5` the verdict first flipped to CONCENTRATED — a **false alarm from
under-powering, not real concentration**. Splitting 150 names-per-month five
ways leaves ~30 per slice, too few to form stable quintiles, so most months
drop to the 20-name floor and one slice was judged on only 40 of 316 months
(t 1.16 on noise). The tool now power-gates: it judges only slices that keep
≥60% of the full month count, tags the rest UNDERPOWERED, and says "split too
fine to judge" rather than inventing concentration. The well-powered
partitions (k≤3, and the 2-way with all 316 months in each half) are the
trustworthy ones, and they are unanimously BROAD. The lesson is generic and
worth stating: a disjointness test must be powered on both sides, or the thin
side's noise masquerades as a finding.

### What remains

This proves the edge is broad WITHIN the top-150. The stronger claim — that it
exists in names that were *never* in the top-150 — needs an out-of-universe,
survivorship-aware feed (mid-cap names via Yahoo, recent years, attrition
measured) and is the next replication. But the datamined-to-specific-names
worry, the cheapest way for this whole book to be an illusion, is now
answered: it is not that.

---

## 23. The adversarial audit: what an independent 52-agent review found — `AUDIT.md`

Everything before this section was written by the people who wanted it to work.
§23 records what an independent, adversarial review — six subsystem readers,
six bias-dimension hunters, two refuters per finding defaulting to "refuted",
a completeness critic — found on `main 00db31f`. Full report, classifications
and the action log: **`AUDIT.md`**. Summary and the corrections it forces:

### No leak; several magnitudes wrong

Zero look-ahead leaks. The within-month overnight ranking edge stands (walk-
forward t 12.6, IC t 17.5, floor t 6.6, disjoint-slice replication weakest t
4.8). What the audit upheld instead — 18 findings, 0 refuted — are accounting,
calibration and construction-mismatch defects. The ones that change what this
file has been claiming:

- **§21 is void until re-run.** `factor.py` built its market factors from
  `ticker=='QQQ'` only with an unmasked `close.shift(1)`: the QQQQ era
  (2004-12..2011-03, including 2008, v2's best era) was silently dropped and the
  rename injected a ~+3,500 bps "overnight" regressor point. The +0.20/+1.19
  QQQ/SPY split, the "half of v2 is beta" narrative, and NEU's t 1.47 are that
  bug's signature — the "collinearity" explanation in §21 was explaining an
  artefact. Fixed (QQQ+QQQQ stitched, factors from the panel's masked returns,
  identity check printed); **the table must be re-run on the real panel and §21
  rewritten from it.** The independent evidence for a positive hedged residual
  (NEUb +3.40, t 4.85, §20) does not depend on the bug.
- **Every long-only series is optimistically biased by censoring, by an
  unrecorded amount.** The split-ratio band and the ±40% backstop (which in
  log space fires at −28.6% downside) NaN real crash nights out of Q5 while the
  live grade books them in full — direction-known, on the deployed object,
  plausibly 0.4–2.5 bps/day. The exclusion counts printed by `load_panel` were
  never pasted here. `xsec_gaps.py` now lists every censored non-verified-split
  night and re-inserts the backstop nights for a lower bound; the paper log now
  carries a like-for-like masked tilt. The frozen scoring path is untouched:
  changing the mask is a re-freeze, not a patch.
- **"ML doubles the rule" (§15b) compared the model with its weakest input.**
  The baseline was `on_1m`; the model's top feature is `on_12m`, and §22's
  `on_12m` rule prints +13.16 (t 12.9) in a different construction. The real
  increment is plausibly +2..+4 bps/day, not 2×. `xsec_ml.py` now prints
  `on_12m` and avg baselines on identical OOS rows and "ML minus best rule"
  with its own t — the number to quote from here on.
- **The ceiling canary's "construction floor" (§15b) was a post-hoc relabel of
  a single draw.** The pre-declared rule said stop trusting; one permuted draw
  is a random projection onto real feature signal, not a null. Multi-seed
  canary null now printed. Recorded as a dated decision, not subtracted from
  `REF`.
- **Costs are understated where the data is oldest and the size is smallest:**
  MOC/MOO collectability was asserted for 1999-2026 on a 2012+ check (Nasdaq
  had no closing cross before 2004); MOM's flat 0.34 bps is 5-15× too low for
  the 1/16-tick years; at the $25k rehearsal size the per-order commission
  minimum alone is 2-4.5× the modelled crossing. The forward test and GO/KILL
  (gross) are unaffected; the 1999-2004 rows of §11/§15 are.
- **Dividends were stated with the wrong sign for the short leg** (§15, and the
  header of `xsec_backtest.py`): the long leg's holder receives them (Q5/tilt
  understated), the short leg pays them (Q1 overstated), so the L/S is roughly
  neutral-to-optimistic. Corrected.
- **Model-mode option numbers (§21's AAPL/QQQ "directional-only" readings) were
  overstated:** the "fair" IV used full-day vol for a 10:30→close window and so
  already sold ~1.1-1.4× rich. Fixed to the window's own vol; re-validated on a
  planted overnight+first-hour world (condor at mult 1.0 ≈ 0). Real-mode
  numbers (§18, §21) are unaffected.
- Ops: a month-rollover hole could have left Aug-31 positions unsold for a
  week; the Yahoo candidate pool was monotone-shrinking; the paper log had zero
  graded nights at HEAD (the workflows had not fired). All three fixed in code;
  the third needs the secrets and a manual dispatch.

### Classification, from the audit (AUDIT.md §G)

```
  ON    ROBUST EDGE (long-only level optimistic by the censoring amount; ~70% overnight-market beta -- pending §21 re-run)
  MOM   PROMISING BUT INSUFFICIENT EVIDENCE (t ~2.0-2.3 after era-correct costs; only holdout negative)
  NEU   ROBUST EDGE, size disputed (+3.4 vs +1.1 until §21 is rebuilt)
  v2    ROBUST EDGE (inherits ON; the MOM increment is unproven)
  v2n   ROBUST EDGE (inherits NEU; weaker)
  OPT   PROMISING BUT INSUFFICIENT EVIDENCE (one regime, decaying, assignment unmodelled)
  v2o   PROMISING BUT INSUFFICIENT EVIDENCE (follows OPT)
```

### What is now different in the code, and what is not

Twelve changes, all off the frozen scoring path (AUDIT.md §I): the factor fix;
the censoring measurement (`xsec_gaps.py`, `load_panel(keep_backstop=)`); the
like-for-like paper-log columns; `on_12m` baselines and the multi-seed canary;
`--through` honoured everywhere and `mc_risk` refusing a series that runs past
the frozen cutoff; common random numbers in the overlay verdict; the window-vol
fair IV; capacity/Sortino/cost-base corrections in the report; the open-leg
liquidation and small-slice warning; the HF-only candidate pool; the dividend
text; and a regression test file that poisons the raw future parquets before
`load_panel`, checks the DST windows and the factor stitching, now run in CI.
Not changed, deliberately: the mask, the ETF list, the universe rule, `REF`,
the GO/KILL tables, the era-varying MOM cost, assignment modelling — each is a
re-freeze item, and the RESULTS 19 clock keeps running on the construction the
frozen models were trained on.

### What only the real panel can settle (run and paste)

```
python src/xsec_gaps.py                      # the censoring block -> here, §15
python src/factor.py                         # §21, rebuilt -- replaces the table
python src/xsec_ml.py                        # ML vs on_12m, canary null, exclusion counts
python src/research_report.py                # §6 inherits the factor fix
python src/xsec_extend.py --check 2026-03    # the Yahoo seam, never recorded
python -m pytest tests -q
```

The honest one-line reading: the edge is real and the book is fundable in
principle; several of the numbers this file quotes for it are too high by
amounts that are now measurable instead of assumed, and one table (§21) is
wrong and awaits its re-run.
