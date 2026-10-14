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

### The exclusion mask flatters the long leg by ~1.5 bps/day — measured

Splits are handled by exclusion (above), and the mask cannot know *why* a night
gapped: a −28.6% log move is a 2:1 split or it is Lehman. `xsec_gaps.py` re-runs
rule B with the ±40% backstop nights put back in, which is the direction of the
bias a live book actually eats:

```
  rule-B (on_1m) monthly legs, 316 months, bps/day: default mask vs backstop kept
             masked     kept   delta
    q5        +9.52    +8.00   -1.52   (t of delta -3.0)
    q1        +1.57    +0.65   -0.91   (t of delta -1.0)
    ls        +7.95    +7.35   -0.61   (t of delta -0.5)
    tilt      +6.02    +4.50   -1.52   (t of delta -3.0)
```

887 nights are censored across the panel — 227 in the split-ratio band, 660 at
the backstop, of which 502 are negative and only 9 are verified splits. The
biggest are events, not data errors: **LEH −267% (2008-09-15), BSC −224%
(2008-03-17), HAL −796% (2004-08-23)**, and the live grade books all of them in
full. So **every long-only overnight number in this file is optimistic by about
1.5 bps/day (t −3.0)**, and the long/short is nearly immune (−0.61, t −0.5)
because the mask censors both legs. Two things this is not: it is not a leak
(the censoring is symmetric in time and known at trade time), and it is not the
whole bias — only the backstop nights were re-inserted, so −1.52 is a **lower
bound**. Levels in the block above come from the gaps script's own monthly-leg
construction, not §15 B's daily series; read the delta, not the level.

The frozen scoring path is untouched on purpose — widening the mask changes the
object the frozen models were trained on and restarts the §19 forward clock — so
the correction is applied by *reading*: subtract ~1.5 bps/day from every Q5 and
tilt figure in this file, and leave the L/S ones alone.

### The ML ranker beats the rule it was measured against, and barely beats the right one

`xsec_ml.py`: LightGBM over ten features per name-month (each a published effect),
targets rank-transformed within month so the market component cancels, walk-forward
by trade year, params inherited frozen from `momentum_ml.py`, and a leak canary
(targets permuted within training months) built in. On a synthetic panel with
planted truths the pipeline predicted, before touching real data, that the model
would roughly double the 1-month rule by finding 12-month overnight persistence.
On the real panel, 283 OOS months (2003-01 → 2026-07):

```
                       bps/day      t   Sharpe    maxDD%
  rule on_1m L/S         +6.40   6.36     1.31     -26.4
  rule on_12m L/S       +11.44  10.98     2.26     -24.0
  rule avg(1m,12m)      +10.88  11.00     2.27     -15.5
  ML L/S                +12.42  11.89     2.45     -24.7

  ML minus best rule (on_12m): +0.99 bps/day   daily t +1.31   monthly t +1.24
  canary null, 5 permutation seeds: mean -0.29  sd 1.21  range [-1.42, +1.52]

  monthly rank IC +0.214 (t=17.5); top feature on_12m at 29% of gain;
  positive in all seven 4-year eras; two losing years (2022, and 2021 at +1.3) in 24.
```

The mechanism replicated — the model does find 12-month overnight persistence,
exactly as the synthetic pre-registration said it would. **The doubling did not
survive the right baseline.** "2×" was the model against `on_1m`, which is the
model's *weakest* input; against `on_12m` — the thing the model itself says is
29% of its gain, and a rule anyone can write in one line — the increment is
**+0.99 bps/day at monthly t = 1.24**, which is not significant at any
conventional bar. The same comparison at the floor (block C) gives +0.94 at t
0.83. Two readings are available and only one is defensible: the model is a
convenient wrapper around a simple ranking rule, and it is the *rule* that is
carrying 92% of the return.

The canary is now a distribution rather than one draw, and it vindicates the
caution: over five permuted-target seeds the null spans **[−1.42, +1.52]
bps/day**, so a single canary printing +1.5 is a construction floor, not
evidence of leakage — and by the same token an "edge" of +0.99 over the best
rule sits *inside* that band. One further caveat stands: the target includes
the opening print; block C re-runs the whole pipeline at the 09:45 exit.

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
  rule on_1m L/S         +2.58   2.25     0.46     -39.8
  rule on_12m L/S        +6.57   5.17     1.07     -32.5
  ML L/S                 +7.50   6.56     1.35     -36.2

  ML minus best rule (on_12m): +0.94 bps/day   daily t +0.89   monthly t +0.83
  canary null, 5 seeds: mean +0.14  sd 1.23  range [-1.03, +1.73]

  IC +0.108 (t=9.2); eras 2003-2026: +7.7 +7.2 +6.2 +11.6 +5.1 +6.6 +8.7
```

The rule keeps 40% of its spread at the floor and the model keeps 60%, and the
model is **positive in all seven 4-year eras**, including 2016-2019 where the
`on_1m` rule went negative. But the same correction applies here as in block B:
`on_12m` also survives the floor at +6.57 (t 5.17), so the model's genuine
increment over the best rule is **+0.94 bps/day at monthly t 0.83** — the same
not-significant number, at the same place in the distribution. The broader
feature mix the floor model leans on (mom_12_2, rng_3m, dvol ~11-12% each) is a
real observation about what predicts the post-reversion residual; it is not, on
this evidence, worth 0.9 bps/day of it.

The canary is now run as a distribution, and it settles a question this section
left open. The ceiling-target canary that kept repeating at +1.6 to +1.8 was
read here as "that construction's achievable-by-chance floor" rather than as
leakage. Five permuted-target seeds put the null at mean −0.29, sd 1.21, range
**[−1.42, +1.52]** — so +1.6 to +1.8 is one seed's draw from a wide null, and
the reading was right. The uncomfortable corollary is the one above: the ML's
+0.99 over `on_12m` is *inside* that band.

**Revised bottom line** (twice revised: once by the discriminators, once by the
2026-09 audit's re-runs). At MOO/MOC execution the ceiling stands: rule L/S
+10.0 (t=9.1), tilt over QQQ +7.1 (t=9.0) — **less the ~1.5 bps/day censoring
bias measured above, so call the tilt +5.6** — and ML L/S +12.4 (Sharpe 2.5),
with the floor's +7.5 at t=6.6 as the robustness margin if execution slips 15
minutes. What is no longer claimed: that the ML is worth roughly double the
rule. It is worth +0.99 bps/day over `on_12m`, inside its own canary band, and
the tradeable object is the **rule**, with the model as at best a small and
unproven overlay. What remains before any capital discussion is what
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

### The clock is running (status, 2026-10-11)

The audit's HIGH ops finding was that this apparatus had no data behind it —
`reports/xsec_paper_daily.csv` was header-only weeks after the basket was
emitted. Fixed: the monthly workflow fires daily, grades the elapsed nights and
commits them, and the log has run unattended since.

```
  33 graded nights, 20260821 .. 20261008, frozen models of 2026-08-20
  (three nights corrected 2026-10-12 -- see below)

  2026-08 (binding)    7 nights   q5 -19.08   qqq -15.27   tilt  -3.81
  2026-09             20 nights   q5 +25.14   qqq  +8.23   tilt +16.90
  2026-10 (partial)    6 nights   q5 -52.81   qqq +19.54   tilt -72.35
  all 33              tilt -3.72 bps/day   sd 152.3   se 26.5   t -0.14
                      95% CI [-55.7, +48.3];  tilt > 0 on 48% of nights
  q5_on_masked == q5_on_bps on 33/33 nights -- no night has hit the split mask
```

**A measurement correction to the binding record, made against the
strategy.** `paper_bench.py --panel` recomputed the August nights from the
backtest panel and found two that disagreed by up to 28 bps. A scan of all 33
found three nights the grader had graded on fewer names than the basket:

```
  night      names   q5 as logged -> full basket   missing
  20260827   9/10      -74.23 ->  -72.31           GEV
  20260828   9/10      -24.35 ->  -52.00           GEV (-301 bps that night)
  20260922   8/11      +11.36 ->   -5.29           CAT, ASML, GEV
```

The grader runs during D+1's session. Yahoo's in-progress bar can carry an
open and no close yet, and the grader dropped the whole row, taking the open
it needed with it. It then never revisited the night. The pre-registered
metric is the *emitted basket*, equal weight, at official prints, so a night
graded on a subset is a measurement error, not a definition.

The three rows were re-graded from Yahoo with `xsec_live.py --regrade`, which
only ever adds missing names, at the owner's decision. The commit names them,
and git keeps the originals.

The correction moves the all-night tilt from −2.43 to **−3.72** bps/day.
The drops had flattered the record by about 1.3 bps/night. The grader now
checks exactly the two prices a night needs and retries an incomplete night
for five days (§19's numbers above are the corrected ones).

**No verdict, and none is due.** 48 calendar days (~1.6 months) have elapsed
against a table whose shortest row is three months; that row becomes readable
around 2026-11-21. The interval spans both the 3-month GO (+12.5) and KILL
(−5.0) many times over: the corrected mean (−3.72) sits 1.3 bps above the KILL
line and 16 bps below GO, both well inside one standard error (26.5).

One thing the first seven weeks *do* show, recorded as a diagnostic and not
acted on — changing the metric or the bands now would void the pre-registration:
**the live basket is far more volatile and far more market-loaded than the
backtest it is judged against.**

```
                                live (33 nights)        backtest
  Q5 overnight beta to QQQ      2.37  (se ~0.3)         ~1.42   (§21, ON)
  beta left inside the tilt     ~1.37                   ~0.42   (§21, NEU)
  tilt daily sd, bps            152   (MAD 121)         ~66     (§15: +7.14 at t 8.9)
```

The baskets explain it. October's is MU, BE, STX, SNDK, NBIS, IREN, LRCX, WDC,
AMAT, INTC, CAT — memory, storage, semiconductor equipment and AI
infrastructure, one theme eleven times. The worst night, 2026-09-11, was the
basket −7.3% overnight against QQQ −1.6%: one sector gap, not eleven
independent draws. Two consequences:

1. **The power column above is optimistic for as long as this lasts.** It was
   computed from backtest-volatility blocks; at ~2.3× the sd the variance is
   ~5×, and a 3-month read under that variance discriminates about as well as
   the table's design would at a few weeks. Treat any 3-month read as close to
   a coin flip in both directions. Even the 12-month row would, at this
   volatility, have roughly the power the table assigns to its 3-month row.
2. **The paper-log metric is currently ~1.4 units of overnight-market beta
   plus selection.** Over a short window its sign is mostly the overnight
   market's sign. A beta-adjusted residual (Q5 − β̂·QQQ, β̂ from a trailing
   window) is worth printing beside it as a non-binding readout; it does not
   replace the pre-registered metric.

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

Two questions the earlier sections deferred, answered on the real panel. The
factor half of this section was rebuilt after the §23 audit voided its first
table; what follows is the re-run, not the original.

### The factor decomposition, rebuilt — there is MORE alpha here than §21 first said

The table that used to sit here was wrong, and the audit (§23) is why: `factor.py`
built its market factors from `ticker=='QQQ'` alone with an unmasked
`close.shift(1)`, so the QQQQ era (2004-12 → 2011-03, containing 2008 — v2's best
era) was silently dropped and the ticker rename injected a ~+3,500 bps "overnight"
point into the regressor. Everything downstream of that — the +0.20/+1.19 QQQ/SPY
split, the "half of v2 is beta" reading, NEU's insignificant t 1.47, and the
collinearity paragraph written to explain them — was the signature of the bug.

Rebuilt on the real panel (QQQ+QQQQ stitched, factors taken from the panel's own
masked returns, Newey-West HAC lag 5, and an OLS identity check printed for every
row):

```
  book           alpha bps/d  t_HAC  residSharpe   R2   betas (mkt_on / spy_on)  raw mean
  ON               +4.13      5.72      1.26      0.80    +0.69 / +0.73           +8.50
  NEU              +3.79      5.25      1.16      0.25    -0.31 / +0.73           +4.09
  MOM              +3.68      2.64      0.48      0.00    (vs intraday: -0.03)    +3.72
  ON+MOM  (v2)     +6.13      4.51      0.90      0.48    +0.76 / +0.66          +10.65
  NEU+MOM (v2n)    +5.79      4.26      0.85      0.07    -0.24 / +0.66           +6.24

  identity check (mean = alpha + sum beta*mean F): reconciles to the printed
  digit on every row -- ON +8.50 vs +8.50, v2 +10.65 vs +10.65, v2n +6.24 vs
  +6.24. The void table did not reconcile; that is how the bug was caught.
  5,841 days 20030102..20260331 (MOM: 6,887 days from 19990310).
```

The two session-orthogonality falsifications still pass, and pass harder: the
overnight book cannot load on the day (ON | mkt_id beta **+0.01**, alpha +8.48 at
t 6.16) and the intraday book cannot load on the night (MOM | mkt_on beta
**+0.02**). Adding every factor at once moves no alpha by more than 0.03 bps/day.

What the corrected table says, in the order it matters:

- **The alpha is bigger and much more significant than claimed.** v2's intercept
  is **+6.13 bps/day at t_HAC 4.51** (was: +4.88 at 3.26), and it is **58% of
  v2's raw +10.65** — not "about half is beta". ON is +4.13 (t 5.72) on a raw
  +8.50, i.e. 49% alpha.
- **NEU's alpha was never insignificant — the bug hid it.** +3.79 at **t 5.25**,
  against the void table's +1.13 at t 1.47, and it now agrees with the number §20
  measured by a completely independent route (NEUb +3.40, t 4.85). The old
  conclusion — "NEU over-hedges into statistically insignificant alpha, a tail
  tool and not a return source" — is **withdrawn**. NEU is 93% alpha (3.79 of
  4.09) with the second-highest residual Sharpe in the table.
- **The beta is real but balanced, and the two hedged books carry almost none.**
  Total overnight-market beta is **+1.42** for ON and for v2, and **+0.42** for
  NEU and v2n. The QQQ/SPY split is now +0.69/+0.73 — two ~0.9-correlated factors
  sharing the load about evenly, which is what collinearity actually looks like
  when the regressor is clean. The trustworthy quantity is still the sum, and
  ~1.42 sits below §20's independently measured basket overnight beta of ~1.55.
- **MOM is still near-pure alpha and still the weakest book.** R² 0.00, alpha
  +3.68 at t 2.64 — uncorrelated with everything — but residual Sharpe 0.48, and
  §24's survival gate kills it on cost, not on significance.

The decision this forces is the opposite of the one §21 printed. **v2n** is not
"the cleaner expression at a lower return": it keeps **+5.79 of alpha — 94% of
v2's +6.13 — while carrying 0.42 units of overnight beta instead of 1.42.** Put
plainly, the hedge costs about 4 bps/day of *market* return and about 0.3 bps/day
of *alpha*. If alpha is what is being underwritten, v2n is the better object, and
v2 is v2n plus a levered overnight-index position anyone can buy without us. The
honest restatement runs the other way from the old one: not "half of v2 is beta",
but **"~58% of v2 is alpha, and the hedged version keeps nearly all of it."**

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
harsher lens (v2 alpha t_HAC 4.51; QQQ spread at measured quotes), and the
harsher lens cuts both ways — it removed two-thirds of the model option Sharpe
as flattery, and it *added* alpha to v2 that a broken regressor had been
hiding. A correct measurement is not a pessimistic one; it is a correct one.

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
  graded nights at HEAD (the workflows had not fired). All three fixed in code,
  and the third has since resolved itself: the workflow now runs daily and
  commits the log (§19, "the clock is running").

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

Three of those lines carried an unresolved dependency on a re-run. **§24 has the
settled version**: NEU's size dispute closes at +3.79 (t 5.25), ON's beta is
measured at 1.42 (~49% of raw, not ~70%), v2n is upgraded to the preferred
expression, MOM's deployment verdict becomes a mechanical KILL, and the ML
overlay is added to the table as NO DEMONSTRABLE EDGE over the rule.

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

### What only the real panel could settle — run, and recorded in §24

```
python src/xsec_gaps.py                      # the censoring block -> §15
python src/factor.py                         # §21, rebuilt -- replaces the table
python src/xsec_ml.py                        # ML vs on_12m, canary null, exclusion counts
python src/research_report.py                # §6 inherits the factor fix
python src/xsec_extend.py --check 2026-03    # the Yahoo seam, never recorded
python -m pytest tests -q
```

All six were run on the full panel (329 months, 1,026,770 ticker-days,
19990301..20260731). **§24 records what they printed**, and §15, §15b and §21
have been rewritten from it.

The honest one-line reading: the edge is real and the book is fundable in
principle; several of the numbers this file quoted for it were wrong by amounts
that are now measured instead of assumed — two of them downward (the long-only
tilt, the ML's increment over the rule) and one of them **upward** (v2's alpha,
which a broken regressor had been hiding).

---

## 24. The verification run: what the corrected code prints on the real panel — `xsec_gaps.py`, `factor.py`, `xsec_ml.py`, `research_report.py`

§23 ended with six commands and a promise. They were run on the full panel —
**329 months, 1,026,770 ticker-days, 1,348 names, 19990301 → 20260731** — and
this section is what came back. Every open item from the audit closes here, and
one of them closes in the opposite direction from the one that was expected.

### The scoreboard

```
  item                       claimed              measured            direction
  long-only tilt / Q5        as printed           -1.52 bps/day       DOWN  (t -3.0)
  ML over the rule           "doubles the rule"   +0.99 bps/day       DOWN  (t 1.24, n.s.)
  v2 alpha (§21)             +4.88  t_HAC 3.26    +6.13  t_HAC 4.51   UP
  NEU alpha (§21)            +1.13  t_HAC 1.47    +3.79  t_HAC 5.25   UP (bug, not signal)
  the +1.6 canary            "construction floor" null [-1.42,+1.52]  VINDICATED
  MOM as a standalone book   "do not fund"        survival gate KILL  CONFIRMED
```

**Re-run on 2026-10-11 against a 330-month panel (19990301 → 20260831, one
month longer than the run above): `factor.py` reprints every alpha, t_HAC,
residual Sharpe, R² and beta in §21 to the printed digit.** The regression
windows end where the overnight books end (20260331), so a longer panel
cannot move them; the point is that the table is reproducible, which the void
one was not.

The three substantive re-runs are recorded where they belong — censoring in
**§15**, the rebuilt factor table in **§21**, the ML-vs-rule increment and the
canary null in **§15** and **§15b**. What follows is what only appears here.

### The survival table, with §6 inheriting the factor fix

`research_report.py` re-fits nothing; every number traces to a committed daily
series. The gate is pre-declared: break-even cost > 2× assumed **AND** Sharpe >
0.5 **AND** the edge is not concentrated in <1% of days.

```
  book    days  bps/d   CAGR  Shrp  Sort  maxDD Calmar   vol   win    PF
  ON      5843   +8.5    22%  1.17  1.63   -44%   0.50   18%   58%  1.25
  NEU     5841   +4.1    10%  1.09  1.57   -32%   0.33    9%   55%  1.23
  MOM     6887   +3.7     8%  0.49  0.71   -48%   0.16   19%   52%  1.10
  v2      5843  +10.7    27%  1.13  1.63   -43%   0.63   24%   55%  1.24
  v2n     5841   +6.2    15%  0.89  1.29   -31%   0.49   18%   54%  1.18
  v2o      749  +19.9    60%  1.93  2.79   -25%   2.39   26%   60%  1.41

  FRAGILITY (mean bps/day after removing the best/worst N days)
  book      full  -best5  -best20  -best1%  -wrst5  -wrst20
  ON         8.5     7.9      6.6      4.2     9.4     11.1
  NEU        4.1     3.8      3.0      1.7     4.5      5.3
  MOM        3.7     3.1      1.8     -1.2     4.3      5.5
  v2        10.7     9.8      8.0      5.0    11.7     13.7
  v2n        6.2     5.5      4.2      1.9     6.8      8.2
  v2o       19.9    15.8      8.3     14.6    24.9     34.2

  COST STRESS (net bps/day at a multiple of the assumed cost)
  book       0x   0.5x     1x     2x     3x     5x    10x   break-even
  ON       10.5    9.5    8.5    6.5    4.5    0.5   -9.5      5.3x
  NEU       6.4    5.3    4.1    1.7   -0.6   -5.3  -17.0      2.7x
  MOM       4.1    3.9    3.7    3.4    3.0    2.4    0.7     12.0x
  v2       13.0   11.8   10.7    8.3    6.0    1.3  -10.4      5.6x
  v2n       8.9    7.6    6.2    3.6    0.9   -4.5  -17.9      3.3x
  v2o      22.5   21.2   19.9   17.3   14.7    9.5   -3.6      8.6x

  SIGNIFICANCE (circular block bootstrap, 21-day blocks)
  book     mean     p5    p50    p95   P(mean>0)
  ON       +8.5   +6.2   +8.5  +10.7     100.0%
  NEU      +4.1   +2.7   +4.1   +5.5     100.0%
  MOM      +3.7   +1.5   +3.7   +6.0      99.7%
  v2      +10.7   +7.9  +10.7  +13.6     100.0%
  v2n      +6.2   +4.0   +6.2   +8.4     100.0%
  v2o     +19.9  +11.4  +20.1  +28.4     100.0%

  VERDICT   ON SURVIVE   NEU SURVIVE   v2 SURVIVE   v2n SURVIVE   v2o SURVIVE
            MOM KILL (edge in <1% of days)
```

**MOM is killed by its own fragility, not by its t-stat.** Remove the best 1% of
days and a 27-year book at +3.7 bps/day goes to **−1.2** — the whole return is
about seventeen sessions, which is what "concentrated in crisis regimes" looks
like when it is measured instead of described. Its Sharpe (0.49) misses the gate
independently. This confirms, on a mechanical rule, the "do not fund" that §15
reached by argument. It does not contradict §21's MOM alpha of +3.68 at t 2.64:
a return can be uncorrelated with everything and still be untradeable.

The other reading worth pulling out is **NEU's break-even of 2.7×**, the thinnest
in the table. NEU is the highest alpha *share* (93%) and the cheapest to kill
with slippage; v2's 5.6× is the comfortable one. Cost, not significance, is the
binding constraint on the hedged books.

### Capacity, on the binding auction

```
  universe month 2026-07: 150 names; traded basket k=11
  ADV$ median ~$1,866M, p25 ~$1,258M
  closing cross ~8% / opening cross ~3% of ADV; participate ~10% of the binding
  (opening) cross -> ~$5.60M per median name, ~$3.77M per p25 name
  x11 equal-weight, least-liquid name binds:  ~$42M .. ~$62M
```

Equal weighting means the least-liquid name binds, and the exit is the *opening*
cross, which is the thinner of the two — that is the constraint, not the closing
auction everyone quotes. An estimate, not a claim: `fills.py` measures it live.

### The Yahoo seam, finally measured

`xsec_extend.py --check 2026-03` compares the Yahoo-built stopgap month against
the HF panel's own version of that month:

```
  universe overlap:  132/150 of the HF top-150 recovered (88%)
  |open  diff|: median  0.0 bps   p95 44.3 bps
  |close diff|: median  1.5 bps   p95 19.3 bps
  |p60   diff|: median  1.1 bps   p95  8.4 bps      (2,904 rows)
  6 names look split-adjusted (Yahoo folds splits into pre-split history; the
  panel is raw -- this is exactly what the split detector exists for);
  6 downloads failed (SKX, CMCS-A, BPMC, FI delisted/renamed; SPLG, GTLS no data)
```

The read the script prints for itself is "overlap ≥90% and a few bps means the
extension can be trusted". At **88%** it is just under its own bar, and the two
missing name-classes are systematic (renames and delistings — the survivorship
direction). The extension is fit for keeping the forward clock running between HF
drops; it is **not** fit for re-deriving history, and §15's panel numbers do not
use it. One more caveat the run surfaced: 2026-03 is beyond Yahoo's 15-minute
retention, so p15/p30 were fabricated as the open and `on15` degrades to the open
exit for that month only.

### The tests, and the bug the verification run found in itself

`python -m pytest tests -q` → **6 passed** (the look-ahead canary, the raw-parquet
poisoning test, the factor stitching, the DST windows, the overlay CRN). The run
also found one defect in the audit batch itself: `xsec_ml.py` bound its argparse
namespace to `a` and then reused `a` as a local 100 lines later, so writing the
`.meta.json` sidecar died with `AttributeError: 'Series' object has no attribute
'through'` **after** the entire walk-forward had completed. Fixed, plus an AST
regression test that fails if any script in `src/` rebinds the name it took from
`parse_args()` — the class of bug that costs a full run to discover and nothing
to catch statically.

### Classification, settled

```
  ON     ROBUST EDGE      long-only level now corrected -1.5 bps/day (measured);
                          beta 1.42, so ~49% of the raw return is the overnight
                          premium levered, ~51% is selection alpha (t_HAC 5.72)
  NEU    ROBUST EDGE      dispute CLOSED at +3.79 (t 5.25); 93% alpha; break-even
                          2.7x is the binding risk, not significance
  v2     ROBUST EDGE      alpha +6.13 (t 4.51) = 58% of raw; MOM increment still
                          unproven
  v2n    ROBUST EDGE      upgraded: keeps 94% of v2's alpha at 0.42 beta -- the
                          preferred expression if alpha is what is underwritten
  MOM    PROMISING BUT INSUFFICIENT EVIDENCE -- deployment verdict KILL (Sharpe
                          0.49; -1.2 bps/day without its best 1% of days)
  ML     NO DEMONSTRABLE EDGE over the rule: +0.99 bps/day over on_12m at monthly
         overlay          t 1.24, inside its own permutation band [-1.42,+1.52].
                          The ranking edge is real; the MODEL is not the edge.
  OPT    PROMISING BUT INSUFFICIENT EVIDENCE (one regime, decaying, assignment
  v2o                     unmodelled) -- unchanged by this run
```

### What the verification did not touch

The frozen scoring path: `load_panel` defaults, `build_table`, `PARAMS/ROUNDS/
FEATS`, `portfolio()`, `REF`, the §19 GO/KILL tables, the mask, the ETF list, the
universe rule. Every correction above is applied by **reading** the numbers
differently, not by re-fitting anything. The forward clock in §19 keeps running on
the construction the frozen models were trained on; changing the mask to remove
the −1.5 bps/day censoring bias is a dated re-freeze decision, and it is not made
here.

The one-line reading, after verification: **the overnight cross-sectional edge is
robust, broad (§22), and worth roughly 1.5 bps/day less on the long side than
this file used to say; its alpha is larger than §21 claimed once the regressor is
correct; and the machine-learning layer on top of it is not, on 283 out-of-sample
months, distinguishable from a one-line ranking rule.**

---

## 25. Benchmarking against paper trading: what it costs, and what paper can measure — `paper_bench.py`

The forward test (§19) grades the emitted basket at official auction prints:
friction-free by construction. The paper account was supposed to supply the
other half of the benchmark (fills, commissions, the account's own sizing),
but on 2026-10-11 it had **never completed a round trip**: `orders_log.csv`
held three dry runs from 2026-08-22, and there was no `fills.csv`. The daily
ROI job had seen zero fills for seven weeks and said nothing; it now telegrams
"paper leg SILENT".

### Pre-trade: commissions at this size eat the edge

`paper_bench.py` prices the current basket with `orders.size()` itself, under
IBKR Pro's published schedule. Every number was re-read on the primary source
by an independent verifier on 2026-10-11 (IBKR commissions and exchange-fee
pages, SEC advisory 2026-2, FINRA SR-FINRA-2026-021, CAT fee alerts).
Separately, a synthesis agent re-derived the nightly total from scratch and
matched the script to the cent.

```
  October basket, 11 names, $25,000 -> $2,273/name, 88% deployed (SNDK buys 1 share)

  capital     Fixed bps/night   Tiered bps/night   (backtest assumed 2)
   10,000          18.1               6.6          2 names unbuyable
   25,000           9.0               3.4
   50,000           4.6               1.95
  100,000           2.5               1.3
  250,000           1.3               0.9

  $25k, per night: Fixed $22.45 = 22 x $1.00 minimum + SEC $0.45
                   Tiered $8.59 = 22 x $0.35 minimum + auction/clearing/SEC $0.89
```

The per-order **minimum** is the cost at this size: every one of the 22 nightly
orders is 1-64 shares, below the 200 (Fixed) or 100 (Tiered) shares where the
per-share rate overtakes it.

At $25k the corrected tilt edge (+4.9 bps/night, §15/§19) pays:
- **183% of itself** in commissions on Fixed;
- **70%** on Tiered.

The backtest's 2 bps assumption holds from about **$50k on Tiered** and
**$100k on Fixed**.

None of the obvious escapes work:
- **IBKR Lite** is commission-free but has no TWS API.
- **Fractional shares** are unavailable through the API, on opening/closing
  auction orders, and in paper.
- **Directed (non-SMART) API orders** lose Tiered pricing. `orders.py` routes
  SMART.

Capital is the lever. At small size the long-only book is mostly a leveraged
overnight-QQQ position with its stock-picking paid to the broker.

### What paper fills can and cannot measure

IBKR's own paper documentation says:
- paper trades never reach an exchange;
- fills are simulated from the **top of the book, not the opening or closing
  auction**;
- partial market fills have their remainder rejected;
- **splits and dividends are not processed**;
- whether paper commission reports follow the account's Fixed/Tiered plan is
  undocumented.

A live MOC/MOO execution prints at the official cross by construction, so:

- **Paper measures:** whether the pipe places, fills and liquidates every
  night, and the whole-share weighting and coverage effects.
- **Paper does not measure:** auction slippage. Its fills are the simulator's.

`paper_bench.py` therefore splits realized − graded exactly into slippage
(buy and sell legs separately), weighting, coverage and data revision. It also
prints a **live-equivalent net** that leaves the simulator's slippage out.

A leg more than 1,000 bps from the official overnight return is flagged as a
corporate action or simulator artefact and kept out of every average. An
unprocessed split makes a paper night look like −90%.

Live also receives dividends that both paper and the graded record omit, so
the live-equivalent net is conservative.

`--panel` adds the code-vs-code check: every graded night recomputed from the
backtest's own panel for the same basket.

### What this changes

Nothing on the frozen path. For the decision §19 will eventually force, a GO
verdict is necessary but not sufficient: at the $25k the crontab names, the
book's skill component does not survive its own commissions. A live account
should be sized at or above the capital where the drag falls to the backtest's
assumption, and run on Tiered. The paper leg should be switched on at that
size; paper capital is free and resets to up to 5x production.

---

## 26. One year, on the real fan: what the traded book pays against just holding QQQ — `roi_year.py`

"So in a year, what does the ROI look like?" — answered for the book `orders.py`
trades (long the frozen model's Q5 overnight, unhedged), on the owner's machine
on 2026-10-12, at the commission plan §25 recommends. The method is in the
script header.

- **Worlds:** five mean-shifted, block-bootstrapped worlds built from the real
  2003-01 → 2026-03 walk-forward nights.
- **Account model:** a fixed-notional account with whole-share deployment and
  IBKR Tiered costs.
- **Benchmarks:** the same money held in QQQ, and QQQ held overnight only.

Two independent reviewers had corrected the method before this run (fixed
notional, arithmetic-mean stress, two-factor alpha, benchmark rows).

**Read as the year after a GO verdict, not this year:** §19's rule is no live
money until the forward test passes.

The run's gross +10.50 bps/night and selection alpha +6.13 (on QQQ+SPY
overnight) match §21's **ON** row (long Q5 overnight: raw +8.50, alpha +4.13)
once §21's assumed 2 bps/night cost is added back. That +6.13 is not §21's v2
row, which happens to print the same number.

```
  ONE YEAR, backtest volatility, 21-day blocks           MEDIAN    p5  P(loss)  dip p50  dip 1/20
  $25,000 Tiered (88% deployed; costs 3.44 bps/night = -8.7%/yr certain)
    as backtested                                          +13%  -12%     18%     -10%      -26%
    half the edge                                           +8%  -17%     28%     -11%      -28%
    recent era (2016+)                                      +8%  -21%     31%     -12%      -33%
    no edge (beta only)                                     +3%  -22%     42%     -13%      -31%
    nothing (zero drift)                                    -8%  -34%     73%     -17%      -39%
    QQQ, bought and held                                   +15%  -17%     24%     -16%      -32%
    QQQ, overnight only                                    +10%  -10%     18%      -8%      -22%

  $100,000 Tiered (96% deployed; costs 1.26 bps/night = -3.2%/yr certain)
    as backtested                                          +21%   -7%     10%     -10%      -26%
    half the edge                                          +15%  -13%     17%     -11%      -28%
    recent era (2016+)                                     +15%  -17%     20%     -12%      -32%
    no edge (beta only)                                     +9%  -18%     27%     -12%      -30%
    nothing (zero drift)                                    -3%  -30%     58%     -16%      -37%
    QQQ, bought and held                                   +15%  -17%     24%     -16%      -32%
    QQQ, overnight only                                    +11%  -10%     17%      -8%      -22%
```

What it says, in the order it matters. Every comparison below is at the
backtest's volatility unless it says otherwise; the live-volatility stress
reverses several of them.

- **At $25k the book does not beat holding QQQ in any world.**
  - *Even as backtested*, its median (+13%) trails QQQ held (+15%). It trades
    that for a shallower left tail (p5 −12% vs −17%, P(loss) 18% vs 24%) at
    backtest volatility. At the live basket's volatility that tail is worse
    than QQQ's (p5 −31%, P(loss) 30%).
  - *Against costs:* the stock-picking covers its costs only as backtested.
    The edge adds about 4.1 bps/night of expected net over the beta-only
    world, against 3.44 bps/night of costs. At half the edge it adds about 2.0
    and does not.
  - The costs (commissions plus exchange, clearing and SEC fees) are a certain
    −8.7%/yr.
- **At $100k the comparison turns.**
  - *As backtested:* the median is +21% (p5 −7%, P(loss) 10%). That is ahead
    of QQQ held on median, left tail and loss probability; QQQ keeps the
    larger upside (p95 +54% vs +45%). At the live basket's volatility only the
    median lead survives (p5 −28% vs −17%, P(loss) 22% vs 24%).
  - *At half the edge:* the median ties QQQ held (+15%) with a shallower left
    tail (p5 −13% vs −17%). The mean and upside still trail (E +5.8 vs +6.2
    bps/night, p95 +40% vs +54%), and at live volatility the left tail is far
    worse (p5 −34%, P(loss) 29%).
  - *2016+ era:* the median also ties (+15%), with the same p5 as QQQ held
    (−17%) and a lower P(loss) (20% vs 24%).
  - *With no selection edge:* the book's median (+9%), p5 (−18%) and P(loss)
    (27%) trail both QQQ lines.
- **The 2016+ nights give a wider one-year fan, not just a lower one.**
  - At the same median as "half the edge", p5 is −21% vs −17% at $25k.
  - It stays wider even with both worlds scaled to the live sd (p5 −41% vs
    −36%), so nightly volatility is not the whole explanation.
- **Longer blocks deepen the tails, but only modestly.** The check covers the
  as-backtested world only.
  - p5 goes from −12% to −13%/−15% at $25k, and from −7% to −8%/−10% at $100k.
  - The 1-in-20 dip goes from −26% to −28%/−29% at both sizes.
  - The 21-day figures understate the tails by up to about 3 points.
- **At the live basket's volatility the bad tail worsens a lot.** The live sd
  is 207 bps/night, 90% CI [141, 265], from 33 nights. As backtested:
  - p5 −12% → −31% at $25k and −7% → −28% at $100k;
  - P(<−20%) 2% → 10% and 1% → 8%;
  - the 1-in-20 dip −26% → −48%/−49%.

  The QQQ rows are not stressed; they keep their own volatility. So the gap is
  the basket's extra live volatility, from its higher beta (2.37) and its
  single-theme concentration (§19).

**The decision this supports.**
- Run the book only at a size where costs fall to about 1.3 bps/night (1.26
  at $100k on Tiered; about 1 needs ~$250k, §25).
- Run it only after a GO verdict (§19). GO rules out no edge; it does not
  establish that half the edge survives.
- Run it only while the basket's volatility is near the backtest's. At the
  live basket's volatility so far, the book's left tail is worse than holding
  QQQ in every world.
- At $25k, holding QQQ has the higher median in every world, the best one
  included.

---

## 27. A Jev-style decision model, added in shadow mode — `guard.py`, `decide_local.py`

The owner asked to add "a locally hosted decision model like Jev for better
performance". Jev turned out to be real: TypeSafe AI's "System One" decision
model, launched 2026-09-15. It is after this file's other sources, so
everything below about it was read on primary sources (TypeSafe's docs and
API reference, the projects' repositories, the papers) and re-checked by an
independent verifier on 2026-10-12.

### What Jev is, and what can be run locally

**Jev itself:**
- **API:** `POST /v1/systemone` with a `state` and typed `questions` (`noul`
  yes/no, `choice`, `score`). It answers with probabilities in about 0.1 s.
- **Hosting:** hosted only, early access, an API key. No weights, no
  on-premises option, no disclosed training cutoff, no finance evaluation.
- **Known weaknesses** (TypeSafe's own notes for jev-1.13): numeric precision
  ("keep math in code"), and dates read as text.

**Local, Jev-API options:**
- **Ollama ≥ 0.35** serves `/v1/systemone` (release notes and source code
  verified). It runs purpose-trained decision models: Together AI's Tev1
  (Qwen3.5 4B / 0.8B) and Bespoke Labs' Nimble (9B). On Ollama's 13-dataset
  comparison Tev1 scores 73.3% (4B) and 63.5% (0.8B), Nimble 75.7%, against
  76.0% for Jev 1.13. These are vendor-reported, and none of the datasets is
  financial.
- **Laya** serves the same API from a 421M encoder on CPU. Its own benchmark
  puts it near chance zero-shot. Its server binds `0.0.0.0:8000` (every
  interface) by default, so on the VM it would be reachable from outside.
- **Kev** matches the API but is GPU-oriented.
- **SemIf** reads option probabilities from a frozen Qwen3.5-4B but has no
  HTTP server.

`jev_local.py` is that SemIf method behind the documented API: no generated
text, the model's own next-token probabilities. It runs Qwen3.5-0.8B on CPU in
the cloud environment: p(yes) 0.956 vs 0.053 on a sanity question, and
deterministic.

### Why it can only be tested forward

The literature on language models in return prediction is unambiguous:
- **Models memorise outcomes before their cutoff, most strongly for prominent
  stocks.** Anonymisation fails for mega-caps (Lopez-Lira, Tang & Zhu 2025),
  and a model "knew" Kodak's +318% day from the ticker and date alone (Gao,
  Jiang & Yan 2025).
- **Backtests shrink badly once the data moves past the model.** Published
  LLM trading agents lost 51–62% of their Sharpe ratio when re-run after the
  model's release (Profit Mirage, 2025), and leading LLMs lost money in live
  forward tests (DeepFund 2025; LiveTradeBench 2025).
- **The recommended designs** are point-in-time models (ChronoGPT, DatedGPT,
  Time Machine GPT) or forward tests. Closed hosted models with undisclosed
  training data should be avoided for prediction claims (Ludwig, Mullainathan
  & Rambachan 2024).

An 11-name basket of mega-caps is the worst case for memorisation. So
`guard.py` never backtests and never trades:
- Each afternoon, before the 15:50 ET MOC cutoff, it asks the decider
  "hold {name} tonight?" per name and "hedge the basket 1:1 with QQQ
  tonight?" for the book.
- It commits the answers, time-stamped and stamped with the model version,
  and grades the implied shadow book against the frozen book later, on
  identical nights at official prints.

Integrity is enforced in code:
- one decision per night per backend (the decider, and the control below);
- refused at or after the cutoff;
- a late row in the log is dropped by the grader;
- one record per decider: a new model starts its own record and its own
  63-night clock, and is never pooled with the old one.

**The control:** a code-only rule (`--backend rule`) that skips a name
reporting earnings before the next open and never hedges. It is logged and
graded on the same nights, so the decider has to beat a rule, not just the
frozen book.

**Gate (pre-declared):** at least 63 graded nights, paired mean > 0 and
paired t > 2, before the shadow book may even be proposed for a dated
re-freeze.

### What a decision layer could fix, measured first

On the 33 live nights, only **1 of 356** held name-nights crossed an earnings
report (MU, −1.0%). The losses were the basket falling together:
- 2026-09-11: all 11 names fell 4–9% overnight;
- 2026-10-06: all 11 fell.

A per-name veto has almost nothing to act on. The book-level question has
evidence behind it. Hedging with QQQ on the same nights cut the nightly sd
from 207 bps to:
- 152 (hedged 1:1);
- 134 (hedged at the backtest beta, 1.42);
- 115 (hedged at the in-sample live beta, 2.37).

The worst night fell from −734 to between −570 and −346 bps. §21 has the
backtest support: the 1:1-hedged book keeps 93% of the alpha.

### The first run, and the rule it forced

Qwen3.5-0.8B's answers on 2026-10-12 put **all 11 names at p(hold) 0.35–0.41,
and the hedge at 0.29**: a constant lean toward "No", not a view on any name.
A bare p < 0.5 would skip the whole basket every night, and the test would
degenerate into "flat vs the frozen book".

The rule was therefore fixed before any decided night had an outcome:
- skip a name only if p < 0.5 **and** p sits at least 0.10 below the night's
  median;
- hedge if p ≥ 0.5.

For a calibrated decider this reduces to the plain rule.

**The first forward decisions, 2026-10-12:**
- **Qwen3.5-0.8B through `jev_local.py`:** decided 14:59:20 ET, committed
  15:03 ET. Hold all 11, no hedge.
- **The earnings rule:** decided 15:18:20 ET, committed 15:18:38 ET. Hold all
  11 (none reports tonight), no hedge.

### Tev1 through Ollama, and the decider from 2026-10-13

Ollama 0.40.3 was installed in the cloud environment with both Tev1 builds:
`tev1:0.8b-q8_0` and `tev1:4b-q4_K_M` (digest `07be32e6e5e3`). Ollama
answers `/v1/systemone` natively, on localhost, with no API key.

**The prompt limit forced a smaller state.** Tev1 refused the first request:
"prompt 0 has 2762 tokens; expected 1–2050 (input is never truncated)".
- **Schema 1** (used for 2026-10-12 only) sent raw five-night series per name.
- **Schema 2** sends computed summaries instead, which is TypeSafe's own
  advice ("keep math in code"). Per name, in bps and rounded: last night's
  overnight move, the 5-night overnight mean, the 5-day return and 20-day
  volatility, plus the earnings flag. QQQ gets the same row. Nothing is taken
  from the decision day.

**Tonight's questions, asked again without logging** (the night was already
decided). Both builds gave identical answers on a rerun.

| | Tev1 0.8B | Tev1 4B |
|---|---|---|
| p(hold), range | 0.64–1.00 | 0.06–0.35 |
| median p(hold) | 0.94 | 0.16 |
| skipped under the rule | none | SNDK (0.0584, bar 0.0597) |
| p(hedge) | 0.88: hedge | 0.61: hedge |
| 12 questions, CPU, warm (cold) | 4 s (20 s) | 12 s (80–84 s) |

**What the table shows:**
- **The two sizes lean in opposite directions on every name.** The 0.8B leans
  to hold, the 4B to skip, and Qwen3.5-0.8B through `jev_local.py` leaned to
  "No". The level of p from a small decider is a lean, not information. This
  is why the rule is relative to each night's median, and it held: no leaning
  model skipped the basket wholesale.
- **Both Tev1 builds would have hedged tonight.** Every name is down 3–16%
  over five days while QQQ is flat (+0.2%). A QQQ hedge does not address a
  sell-off confined to the basket's theme, so this answer is not obviously
  sensible. Either way it is one night and carries no evidence.

**The decider from 2026-10-13 is Tev1-4B** (`tev1:4b-q4_K_M`, digest above).
This was declared before any decided night had an outcome. *Superseded the
next morning, before Tev1 decided a single night, by Kev on the owner's own
machine; see the next subsection.* The reasons:
- it was trained for these question types, unlike a chat model read through
  its next-token logits;
- it scores 73.3% against Jev's 76.0% on Ollama's comparison (vendor-reported).
  Nimble scores higher (75.7%) but is a 9B model and a 9.5 GB download;
- it runs on CPU in seconds and needs no key.

`guard.yml` runs it in GitHub Actions in the small hours of every weekday
(06:17 UTC, the decision day in ET), with the earnings rule beside it:
- The early slot is forced: GitHub started `monthly.yml` 2h45m–6h45m late on
  every run in Sept–Oct 2026, which would push a midday run past the cutoff.
  The state uses nothing from the decision day, so nothing is lost.
- Ollama is a released build, checked against its published sha256, CPU
  libraries only.
- The job refuses a model whose digest differs from the pinned one.
- The job holds no secrets and never trades.

Qwen3.5-0.8B's single night stays its own one-night record.
`python src/guard.py --dry-run` asks a decider and prints the answers without
writing anything, for trying one by hand.

### Everything local: Kev, the open-weight Jev lookalike — `decide_local.py`

The owner then set two rules: use an open-weight Jev lookalike, and host
everything locally. `guard.yml` broke the second rule: it ran Tev1 on
GitHub's cloud runner. It was removed (PR #28) before its first scheduled
night, so **Tev1 never decided a night.**

**The survey** was run on 2026-10-12. It read primary sources (repositories at
pinned commits, Hugging Face model cards and API) and put each claim through
an adversarial check. Third-party scores come from two public boards, JevBench
v1.6.1 and Decision Index v0.3.2; they were not re-measured here.

| Model | Open weights, licence | Jev API | vs Jev (who measured) | Verdict |
|---|---|---|---|---|
| **Kev 1.0** (jaredpalmer) | yes; Apache-2.0 code, adapter, head and base | closest: built to TypeSafe's spec, served by its own server and by llama.cpp | 4B: 0.817 vs 0.857 (Kev's author, dev set); Decision Index 39.5 vs 60.11. 27B: JevBench 78.9/64.2/93.7 vs 77.1/63.6/90.6 | **chosen (4B)** |
| H2O-Lightning-4B | yes, Apache-2.0 | via a Python shim that by default forces every yes/no to at least 0.801 | JevBench 75.0/60.0/90.0, measured on older weights than the download | no |
| Tev1 (Together) | yes; licence "being finalized" | via Ollama, prompt capped at 2,050 tokens | Decision Index 34.1 (4B) | retired |
| Nimble (Bespoke) | v3 non-commercial; v4 no weights | via Ollama | Ollama's own comparison: 75.7% vs 76.0% | no |
| Clef / Clef-Flash (Cloudflare) | yes, Apache-2.0 | not Jev's semantics: answers all questions jointly | JevBench capability 75.3 (Clef), 63.8 (Clef-Flash) | no |
| JevK5 | trained on OpenAI-terms outputs | GPU-only server | Decision Index 41.5 (4B) | no |
| Laya, SemIf, Verdict, jeff, CLM | — | — | near chance on both boards | no |

Kev-4B is the only candidate where every part is Apache-2.0, the training data
is stated, and two independent local servers exist. Its card says "No output
of Jev (TypeSafe's hosted decision model) was used." The price is accuracy:
on the independent Decision Index it scores 39.5 against Jev's 60.11.

**What runs.** ggml-org's GGUF conversion runs in llama.cpp's own
`/v1/systemone` server (release b11429). It is one native program, needs no
Python stack, and runs on Windows and Linux. All of the following were checked
here:
- **Same weights as Kev v1.0.** Every file of `jaredpalmer/kev-4b` at the
  conversion's source revision is blob-identical to the `v1.0` tag.
- **Same calibration.** The GGUF stores Kev's shipped temperature 2.40605,
  the value in Kev's `head.pt`. (`result.json`'s 2.297 is one
  cross-validation fold.)
- **llama.cpp loads it as a decision model.** It reports "decision model type:
  kev" and `output_modalities: ["decisions"]`.
- **Locked to this machine.** It refuses requests without the key (401) and
  binds 127.0.0.1.
- **Deterministic.** Two runs of tonight's request gave identical answers.

**Speed.** The server shares the ~650-token state across the questions of
a request, but only when it has more than one slot. *Corrected after review:*
an earlier version of this section said b11429 keeps no state cache for
decision requests, citing its "only for completion tasks" line. That line
governs reuse across requests. Within a request the server groups the
questions by slot (`server_decision_group_tasks`), and the runner's first
setting of one slot (`-np 1`) made it re-read the state for every question.
Measured on 4 cloud vCPUs, the same request each time:

| Slots | One night (12 questions) | Peak memory | Max \|Δp\| vs Kev's reference |
|---|---|---|---|
| 1 | 532 s | ~6 GB | 0.0148 |
| 12 | 83 s | 6.7 GB | 0.0090 |
| **16 (the runner)** | **85 s** | **6.9 GB** | **0.0090** (identical to 12) |

Sharing changed no decision, and it brings the answers closer to Kev's own
server. The runner uses 16 slots with a 16,384-token shared pool, enough for
a basket of up to 15 names. BF16 needs 9.7 GB (8.2 GB of it the mapped model
file). A desktop is faster still. The run starts hours before the cutoff, and
an answer that arrives late is not logged.

**Is the GGUF route really Kev?** The thresholds were declared before any
comparison was seen: max |Δp| ≤ 0.02 between Q8_0 and BF16; ≤ 0.03 between the
0.8B GGUF and Kev's fp32 reference; ≤ 0.05 between Q8_0 and Kev's bf16
reference (Kev's own bf16-vs-fp32 serving gap); and no decision change under
guard's rule in any of them.

| Check, threshold declared before running | Threshold | Result |
|---|---|---|
| A. Kev-4B Q8_0 vs BF16, both on llama.cpp | max \|Δp\| ≤ 0.02 | **0.0103**; decisions identical |
| B. Kev-0.8B on llama.cpp vs Kev's own fp32 reference server | ≤ 0.03 | **0.0028** (mean 0.0014); identical |
| C. Kev-4B Q8_0 on llama.cpp vs Kev's own reference server (bf16)* | ≤ 0.05 | **0.0148** (mean 0.0061); identical |

\*The fp32 4B reference needs ~19 GB of memory, more than the test machine
had. 0.05 is Kev's own stated bf16-vs-fp32 serving gap.

**The GGUF route is Kev.** At the reference precision (B), llama.cpp's
implementation agrees to 0.003. The Q8_0 file the owner will run stays within
0.015 of the reference and changes no decision.

**Kev's own server** takes 71 s warm and 141 s cold for 4B on the same CPU,
about what llama.cpp now takes. It needs a 9 GB Python/torch install that is
untested on Windows, so llama.cpp, which needs nothing, is the default.

**Tonight, as the decider will see it** (2026-10-13, state through the 10-12
close; Kev-4B Q8_0):
- **Hold:** p(hold) 0.51–0.70, median 0.605, so all 11 names are held.
- **Hedge:** p(hedge) 0.477, so no hedge.

Unlike Tev1's two sizes, Kev's answers sit near the middle: an ordinary night,
with mild doubt. This is one night, so it is not evidence.

**The decider from the first night it runs on the owner's machine** is
`kev-4b-q8_0@d924f2e2+llama.cpp-b11429-win-x64` (`-linux-x64` on the VM).
That name is logged with every answer and carries the file, its revision, the
runtime and the platform build, so any other build is a different decider
with its own record and its own 63-night clock. Run it on one machine only. The
earnings-rule control decides beside it, on the identical state. *Superseded
the same day, before Kev decided a single night, by the hosted deciders below
(the owner allowed OpenRouter). Kev remains the `--engine local` option.*

`decide_local.py` does the whole night locally:
- **Setup.** `--setup` makes its own clone of the repository, downloads
  llama.cpp and the model, and checks both against pinned sha256 sums. A file
  that does not match is deleted, never used.
- **The nightly run.** It pulls, logs the control, and starts llama-server on
  127.0.0.1 with a fresh random key passed in the environment. Before asking,
  it confirms the server refuses requests without the key and serves the pinned
  model. It then asks, stops the server, commits **only**
  `reports/guard_log.csv` from its own clone and pushes it. Nothing of the
  owner's working copy is touched.
- **Scheduling.** `--schedule` registers a weekday Task Scheduler task on
  Windows (wakes the PC, runs on battery, runs late if the PC was off) or a
  systemd timer on Linux. Both run at 09:17 ET and again at 13:47 ET. The
  second run is a catch-up after a wake without network or a basket
  registered late; a night already decided is never asked again.
- **Timestamps.** `guard.decide` now stamps an answer when it *arrives*, on
  the later of the monotonic and wall clocks, and refuses one at or after the
  cutoff.

**Refused rather than guessed.** An adversarial review (four lenses, every
finding independently re-checked) led to these rules:
- **This month's basket.** A night is decided only on the basket registered
  for its own month. On the 1st, before registration, there is no decision.
- **A complete state.** Every name and QQQ must have market data. Without it
  (no network yet after a wake) the night is not decided on part of the
  picture.
- **A known earnings calendar.** A lookup that still fails after retries
  stops the night. The control must never silently hold a name that reports.
- **Pushed before the close.** Answers count only if they reach the remote
  before that night's close. Unpushed answers from an earlier night are
  dropped, never pushed late.
- **One state per night.** It is cached, so a catch-up run asks Kev about
  exactly what the control saw.
- **A failing Kev never costs the control's row.**
- **A sealed server.** It is started in an environment with no `LLAMA_*`
  variables and empty config directories, and it is refused if it read a
  config file, accepts a wrong key, or answers under another name.
  (llama-server silently applies `config.ini` files and `LLAMA_ARG_*`
  variables.)
- **No proxy.** Requests to the local server bypass any system proxy.
- **Checked every night.** The llama.cpp files on disk are re-checked against
  the pinned release before each night.
- **Unfinished rows are dropped.** If a run is killed between deciding and
  committing, any uncommitted rows from an earlier night are discarded, not
  pushed late.

**One trust that remains:** the dedicated clone runs the code on `main` as
pulled each morning. Anyone who can push to `main` can change the decider's
code. That is the same trust as the repository itself. A code change that alters
the answers would not change the logged name, so watch
`src/decide_local.py` and `src/guard.py` in the history.

All of this was tested end to end here, against a stand-in remote and with
the real model, three times:
- **After the review fixes:** `--setup` downloaded and verified both files
  for real, and the nightly run found its home with no environment set.
- **With 16 slots:** the whole night, from building the state to pushing,
  took 115 s of wall time.
- **Every run:** the control and Kev were logged, one commit holding only the
  log was pushed, and the server was stopped.

### Nimble, the other open-weight candidate: measured, not adopted

The owner pointed at "Nimble: A Qwen-Based Alternative to Jev". That is
section 2 of a DataCamp roundup, *Top 7 Open-Source TypeSafe Jev
Alternatives* (2026-09-23); Kev is section 3 of the same piece. The article
covers only the first Nimble release and gives no accuracy, licence or
training-data facts, so Nimble was checked on primary sources and run here on
tonight's frozen state. The findings were re-checked adversarially.

**There are several Nimbles (Bespoke Labs, LoRA adapters on Qwen3.5-9B):**

| Release | Licence | What is known |
|---|---|---|
| original (2026-09-18) | Apache-2.0 | 2,676 synthetic rows (GPT-5.6, Claude Sonnet 5; no human review; no finance); fitted temperature 2.179 |
| v2 (2026-09-23) | Apache-2.0 | 4,926 rows; temperature copied from the original, not fitted. Decision Index 40.95 |
| current `Bespoke-Nimble-9B` main, "v3-12026" (2026-09-24) | Apache-2.0 | 12,026 rows, undescribed; temperature 1.0, not fitted; no evaluation published. **This is what Ollama's `nimble:9b` ships** (its licence layer names this adapter; inferred from its name "merged-current") |
| `Bespoke-Nimble-9B-v3` (2026-09-30) | **CC BY-NC 4.0** | a separate, larger adapter. Decision Index 54.67, the best Nimble; ECE 0.123, the worst calibrated |
| 27B-v3, v4 | — | hosted by Bespoke only: not local |

On the independent Decision Index (v0.3.2) the scores are:
- Jev 60.11;
- Nimble v3 54.67 (non-commercial);
- Nimble v2 40.95;
- Kev-4B 39.5, confirmed to be the exact Kev 1.0 weights run here.

Kev-4B's calibration error there is 0.032, against Jev's 0.074.

Bespoke says Nimble's labels were not derived from Jev, but its pipeline did
send 300 seed scenarios to Jev and publishes Jev's answers to them. The
Ollama page's "75.7% vs Jev 76.0%" compares Ollama's own run of its build
with Bespoke's run of Jev. It is not like for like.

**Measured here** (Ollama 0.40.3, `nimble:9b-q8_0`, the Apache build; the
same frozen 2026-10-13 state and the same 12 questions as Kev):

| Name | Kev-4B | Nimble, joint | Nimble, joint reversed | Nimble, one at a time |
|---|---|---|---|---|
| MU | 0.55 | 0.41 | 0.44 | 0.52 |
| BE | 0.66 | 0.09 **skip** | 0.41 | 0.64 |
| STX | 0.56 | 0.12 **skip** | 0.28 **skip** | 0.32 **skip** |
| SNDK | 0.58 | 0.10 **skip** | 0.24 **skip** | 0.21 **skip** |
| NBIS | 0.63 | 0.21 | 0.53 | 0.81 |
| IREN | 0.51 | 0.30 | 0.37 | 0.45 **skip** |
| LRCX | 0.60 | 0.76 | 0.64 | 0.97 |
| WDC | 0.58 | 0.50 | 0.46 | 0.88 |
| AMAT | 0.64 | 0.84 | 0.79 | 0.95 |
| INTC | 0.63 | 0.18 **skip** | 0.06 **skip** | 0.35 **skip** |
| CAT | 0.70 | 0.87 | 0.95 | 0.95 |
| hedge (BOOK) | 0.48 | 0.55 **hedge** | 0.72 **hedge** | 0.60 **hedge** |

Decisions are guard's pre-declared rule. Each Nimble column was
deterministic.

**What the table shows:**
- **Nimble's answer for a name depends on the other questions asked with
  it.** Every Nimble release writes the whole question list into each
  question's prompt (Bespoke's own prompt builder), and the effect is large.
  Reversing the order of the same 12 questions moved answers by up to 0.32
  (mean 0.14) and turned BE from skip into hold. Asked alone, BE went from
  0.09 to 0.64. The basket and its order change every month, so in its
  native mode Nimble's record would mix views on the stocks with artifacts of
  the question list. Kev's prompt is the state plus one question, so by
  construction its answers depend only on that question.
- **Asked one question at a time, Nimble is fair but slow and heavy.** It
  took 14 minutes a night here (Kev: 85 s), with 10.4 GB of memory at Q8_0
  (Kev: 6.9 GB). An Apache Q4_K_M build of 5.6 GB exists on Ollama but is
  unmeasured. Nimble is also still unevaluated and uncalibrated in that build.
- **It disagrees with Kev.** It would skip three or four names and hedge
  where Kev holds everything. That is one night; whether either is right only
  the forward record can say.

**Verdict: Kev stays the decider.** Each usable form of Nimble fails on one
count:
- the Nimble that scores best independently is non-commercial;
- the Apache Nimble that runs locally is unevaluated, uncalibrated and
  trained on undescribed data;
- in its own joint mode, its answers depend on the question list.

**If Nimble is wanted anyway,** the honest form is a second shadow record:
Ollama `nimble:9b-q8_0` pinned by digest, one question per request, declared
before its first night. Its gate must be tightened for two deciders: with
63 nights, Bonferroni halves the one-sided 2.5% level, so the gate becomes
paired t > 2.30 instead of 2. It is not added unless the owner asks.

### Hosted is allowed: Perplexity Decider and Jev, through OpenRouter

The owner then widened the rule: "local host or API through OpenRouter, it is
okay". That changes the answer, because **OpenRouter serves real decision
models, Jev included**. Its catalogue lists 21 models whose output is
"decisions", among them:
- `typesafe/jev-1.13`;
- `perplexity/pplx-decider-v1.1-27b`;
- `jaredpalmer/kev-4b`;
- Clef, Tev1, Mercury Decide;
- OpenAI's GPT-6 Luna Decisions and Microsoft-Decision-1.

OpenRouter's Decisions API (`POST /api/alpha/decisions`) takes Jev's request
shape and answers with the dated build and the provider. guard's client
speaks it unchanged.

**Measured on the two independent boards** (Decision Index v0.3.2, JevBench
v1.6.1; neither re-measured here):

| Model | Reachable as | Decision Index | JevBench intelligence / calibration |
|---|---|---|---|
| **Perplexity Decider v1.1 27B** | OpenRouter; Apache-2.0 weights | **62.75** | **75.7** / 89.9 |
| Jev 1.13 | OpenRouter, TypeSafe | 60.11 | 63.6 / 90.6 |
| Kev 27B | local only, about 16 GB at Q4 (unofficial file) | 58.78 | 64.2 / **93.7** |
| Clef | OpenRouter | 53.08 | 59.1 / 91.6, but answers jointly |
| Mercury Decide | OpenRouter | — | 65.9 / 81.3 |
| Kev 4B (the local decider until now) | local or OpenRouter | 39.50 | — |
| Qwen3-32B zero-shot (a plain LLM) | — | — | 33.9 / 89.9 |

GPT-6 Luna Decisions, Microsoft-Decision-1, Solar Decide, Drex v2.0 and
Span-01 are on neither board.

**What the boards rule out:**
- **A plain LLM read through its logprobs.** I had started building this
  before finding the catalogue. The one zero-shot LLM on the board scores
  half of Jev's intelligence. It was dropped.
- **The unscored decision models.** Unmeasured is not better.

**Kev-9B was also run here** (llama.cpp, Q8_0, ggml-org's file, built from
weights identical to Kev's `v1.0` tag; same temperature as its `head.pt`):
- 137–146 s a night and 11.8 GB of memory;
- deterministic;
- it would have skipped five names where Kev-4B skipped none.

Its own README puts it at 41.0 on the held-out index, against 38.0 for 4B.
That gap is too small to matter beside the hosted models.

**Declared 2026-10-13, before any of them decided a night:**
- **The decider** (backend `http`): `perplexity/pplx-decider-v1.1-27b`. It
  is the most accurate model reachable, on both boards, and its calibration
  matches Jev's. Its card scores each question as its own row, the state
  plus that one question, so it has no cross-question effect. Its weights
  are Apache-2.0. It needs about 49 GB of GPU memory, so it is hosted.
- **The reference** (backend `jev`): `typesafe/jev-1.13`, Jev itself. It is
  what the owner first asked for, and the standard every board measures
  against.
- **The control** (backend `rule`): the earnings rule, unchanged.
- **Kev-4B on the owner's machine** stays available as `--engine local`,
  with no key. It is not a declared record.
- **The record logs the dated build and the provider**, e.g.
  `typesafe/jev-1.13-20260917@TypeSafe`. An answer from any other model is
  refused. A new dated build starts its own record.
- **The gate is tightened.** Three records can each pass it, so 2's one-sided
  2.5% is split three ways (Bonferroni, 62 df): paired t > **2.46** over at
  least 63 nights.

**What it costs and sends:**
- Each night sends the ~650-token state and the 12 questions to OpenRouter
  and the model's provider. These are public-market summary numbers.
- At the listed rates ($0.02 and $0.042 per million input tokens) a night
  costs well under a tenth of a cent.
- Answers come back in under a second once warm (measured below).
- The key, `OPENROUTER_API_KEY`, lives only in the owner's machine's
  environment, never in the repository or in CI.

**Verified against a stand-in:** both decision endpoints exist (401 without
a key, against 404 for a nonexistent path). The client, the build-pinning
and refusal, the per-decider failure isolation and the retry on a 502 all
pass against a stand-in for the API.

### The first live answers, and the benchmark with and without a key

The key was added after this session started, and a session only picks up
new settings when it begins, so a second session ran the live test
(`bench/openrouter_livetest.md` on branch `claude/openrouter-livetest`; no key
appears in it). It ran on 2026-10-13 at 11:25–11:30 UTC, from `main` at
`e7fc1c9`, against that night's real state. That state was built from Yahoo
in 9–11 s, with 11 names and 1,416 bytes. The state was sent twice, through
the same `guard.ask_http` the runner uses.

**How the key arrives there.** In that cloud environment the key is not in
the process environment. The network proxy adds it to requests for
openrouter.ai, and a request with no `Authorization` header gets HTTP 200.
`decide_local.py` refused to start there because it found no key, which is
the intended default. `--proxy-auth` now covers that case: it sends no key
and relies on the proxy, and a test checks that no `Authorization` header is
sent. On the owner's machine the key stays an environment variable, and the
flag is not used.

**Both pinned models answered as themselves:**

| Backend | Answered as | Latency, run 1 / run 2 | Same answers twice? |
|---|---|---|---|
| `http`, the decider | `perplexity/pplx-decider-v1.1-27b-20261006@Perplexity` | 6.95 s (cold) / 0.53 s | yes, identical |
| `jev`, the reference | `typesafe/jev-1.13-20260917@TypeSafe` | 0.30 s / 0.40 s | no, up to ±0.06 |

Both builds match the runner's pin pattern, so neither would have been
refused. The whole test cost about $0.0049 of credit; that includes two full
nights for both models and two single-question probes. That is about a tenth
of a cent per night.

**What they said about 2026-10-13**, under the declared rule:

| | Its median p | Skips | Hedge (BOOK) |
|---|---|---|---|
| Perplexity Decider, both runs | 0.514 | none | yes (0.558) |
| Jev, run 1 | 0.56 | SNDK, IREN | no (0.35) |
| Jev, run 2 | 0.59 | MU, STX, SNDK, IREN | no (0.37) |
| Rule control | — | none (no earnings tonight) | no |

Four things to read before the first logged night:
- **The decider barely leans.** Every probability is between 0.46 and 0.61,
  and none is 0.10 under its median. On this night it skips nothing and
  hedges. If that stays true, its record measures the hedge and not the
  skips.
- **Jev is a sample, not a function.** Two identical requests minutes apart
  gave different skip lists, two names against four. The runner asks each
  backend once a night and logs what arrived, so Jev's record is one draw
  per night. Its noise is part of what that record measures.
- **The two disagree.** They give CAT 0.48 against 0.75, LRCX 0.53 against
  0.68–0.74, and BOOK 0.56 against 0.35–0.37. Their records are separate
  questions, and that is why the gate is split three ways.
- **Ties with the skip bar are real, and the rule now handles them.** Jev
  answers in hundredths. In run 1, SNDK sat exactly 0.10 under the median,
  at 0.46 against 0.56. The code compared floats bare, so it skipped 0.46
  under 0.56 but held 0.49 under 0.59, which is the same distance. It now
  compares tie-safe, which is what "at least `SKIP_MARGIN` below" always said.
  Run 1's skips are unchanged by the fix. The fix was made before any hosted
  night was logged; the only rows in the log are the superseded Tev1 dry run
  of 2026-10-12, where p is continuous and ties never happen.

**Benchmark, with and without a key.** These are the same night's state and
the same 12 questions. The local engine ran on this 4-core cloud container on
the CPU, with llama.cpp b11429 and 16 slots sharing the prefix. "Warm" means
the files were already read once; "cold" means the first read after a
container restart.

| Engine | Key | Answer time | Night, warm / cold | Peak memory | Same answers twice? |
|---|---|---|---|---|---|
| Perplexity Decider, hosted | yes | 0.5 s (7 s cold start) | ~10 s (Yahoo state) | — | yes |
| Jev, hosted | yes | 0.3–0.4 s | ~10 s | — | no (±0.06) |
| `--engine hosted`, no key | none | — | refuses in 2.3 s, exit 1, nothing logged | — | — |
| Kev-4B Q8_0, `--engine local` | none needed | 86–87 s | 130 s / 512 s | 6.85 GB | yes, exactly |
| Kev-9B Q8_0, local | none needed | 143 s | 206 s / 1,016 s | 11.79 GB | yes, exactly |

The cold nights are the container's disk, not the model. This container
reads a file for the first time at about 11–12 MB/s. Hashing a model takes
14–19 s once it has been read, and loading it takes 15–19 s. On a normal SSD
a local night is the warm figure. Either engine finishes long before the
09:17 and 13:47 runs matter.

Without a key, the hosted engine does not quietly fall back to the local
model. It refuses and logs nothing, and the message names the three ways
forward: set the key, use `--proxy-auth` where a proxy adds it, or use
`--engine local`. A night is never decided by an undeclared model.

### The battery: randomized, outcome-free, and declared before it ran — `decider_battery.py`

The owner then asked for the deciders to be tested here, with randomization,
and for the best one to be chosen, built in and iterated. One live night says
nothing about which model reads this state well. And the record can never be
backtested (above). So the test is a **battery of randomized experiments
whose right answers are known by construction**, and no market outcome
enters it anywhere.

**What it sends.** 377 requests per model, the same bytes to every model:
- **24 synthetic states**, in 4 blocks of 6, half Stressed and half Calm.
  - Each state draws 11 of the 17 tickers that have been in a q5 basket.
  - Each name is given a condition: Ordinary (hold), Earnings or an Extreme
    move (skip), or Moderate (unlabelled). Every state has at least 6
    Ordinary names, so every label can be reached through `guard.rule`.
  - Moves are folded coherently into the 5-day return and the volatility,
    anchored to each ticker's real volatility on 2026-10-12.
- **The real 2026-10-13 state (S0)**, byte-identical to what both hosted
  models answered live. It is diagnostic only and carries no labels.
- **What is randomized, every draw from a committed seed:**
  - the question and name order;
  - opaque question ids and key order;
  - ±3 bps jitter, a date shift, and a paraphrase of the book sentence;
  - which name gets an earnings flag (E4) or an extreme move of 1.5, 3 or 6
    times its volatility (E5);
  - which regime a state's QQQ row is swapped into (E6);
  - a derangement of the tickers' feature rows, or anonymised labels (E7);
  - negated questions (E8);
  - the order in which every (model, request) pair is sent.

**What it measures**, per model:

| Gate | Passes if | Catches |
|---|---|---|
| G1 | ≥ 98% valid answers, ≥ 95% complete, ≥ 90% from one build and provider, p95 ≤ 60 s, median ≤ $0.01 | a model that cannot serve the night |
| G2a | the excess wobble under nuisances is ≤ 0.05, and ≤ half the cross-name spread (which must be ≥ 0.01) | answers moved by order, ids or rounding; a lethargic model gets no credit for stability |
| G2b | the lower bound of its context dependence is ≤ 0.05 | a joint answerer (Nimble measured 0.14) |
| G3a | an earnings flag lowers p(hold) by ≥ 0.05 against the 10 untreated names, in ≥ 75% of states, Holm p < 0.05 | a model that ignores the one fact the question names |
| G3b | a stressed QQQ row raises p(hedge) by ≥ 0.05 and raises the hedge rate, Holm p < 0.05 | a model that cannot see market stress |
| G4 | ≤ 10% of decisions flip on a retest, under its handling | a coin |
| G5 | decision accuracy Q ≥ the earnings rule's + 0.05 (the rule scores 0.599) | a model no better than the control |
| G6 | its answers follow the feature rows, not the ticker names (Φ ≥ 0.5) | a model answering from its memory of a ticker |

Q is the balanced decision accuracy through the declared rule: skips of
skip-criteria names, holds of Ordinary names, hedges of Stressed states and
no hedge in Calm ones, a quarter each. A noisy model is asked three times in
a seeded order and averaged ("k3") if its retest noise or order sensitivity
calls for it. That handling is fixed before Q is looked at.

**The candidates.** All 18 paid decision models on OpenRouter, each pinned.
Moving aliases (`~...`) and `:free` routes are refused. The incumbent
(Perplexity Decider) and the reference (Jev) are never dropped for budget.

**The selection rule, fixed in code** (`battery_score.select`):
1. The eligible models are those that pass all eight gates.
2. They are ranked by Q, with ties broken by AUC, lifts, flips, board
   score, cost and id.
3. **The incumbent is replaced only if:**
   - it fails a gate (then the best eligible model is declared); or
   - the leader beats it by at least 0.03 in Q, with the lower bound of a
     paired, regime-stratified bootstrap above 0 at a Bonferroni-corrected
     alpha.
4. If nothing passes, the incumbent stays and the failures are recorded.
5. The control, the reference and the 2.46 gate never change.

**Spend.** The cap is $8, and a resume can raise it to at most $12.
- The run refuses to start unless at least the cap plus $10 of credit
  remains.
- A pilot call per model measures the real cost first. If the projection
  exceeds the cap, block 4 is dropped first, then the dearest unscored
  models, and that decision is written once and never changed.

**Integrity.**
- **What the lock pins.** `lock_v1.json` pins every byte the run sends: plan
  `56311a1b…`, items `2c1db06e…`, states `a2ac1bec…`, S0 `a974818a…`,
  anchors `0b70e99f…`.
- **When the run refuses.** It refuses a dirty tree, a lock mismatch, a
  second run directory for the same plan, and a resume of a run that has
  been scored. Whether to extend a run is never decided after seeing its
  result.
- **Auth.** With `--auth proxy` no key is ever read. Every printed or written
  string is scrubbed, and `scrub-check` fails on anything key-shaped.

**Built and reviewed before any live call.**
- **The stand-ins.** Ten stand-in models with known behaviour run end to end
  through the real items:
  - The oracle and the noisy, Jev-like model pass.
  - The constant, random, lethargic, ticker-prior and rule-clone models fail
    the gates they should.
  - The joint answerer fails G2b.
- **The adversarial review.** An adversarial review (4 lenses, every finding
  checked by a refuter) confirmed and fixed 16 of 19 findings. Among them:
  - a failed pilot call could have dropped most candidates;
  - a hung model could stall every block;
  - an API cache could make a noisy model look deterministic, so the retest
    now sends different bytes;
  - a partial result could have been inspected before deciding to extend.

**Its limits.**
- **Coherence is not profit.** The battery picks the model that reads the
  state as the question asks, and stably. Only the forward 63-night gate
  measures value.
- **Q weighs the hedge half equally with the skip half.** A model that never
  skips but hedges well can qualify. That follows the repo's own reading,
  that the hedge question is the one with evidence behind it, and it is
  declared rather than patched.
- **Stressed states are synthetic.**

**The grader gains randomization inference** (`guard.randomization`). Each
record's verdict now adds these lines below the unchanged gate line:
- the skips and hedges it made;
- the split of its edge into skip and hedge parts;
- a sign-flip p-value;
- an activity-matched null: random skip sets of the same size each night,
  and the same number of hedged nights placed at random.

Its label says whether the selection beats random choices of the same
activity, or the edge is just exposure. The 63-night gate at t > 2.46 is
untouched. Records outside the three declared slots print as
`[not a gated record]`.

### The battery's result: Perplexity Decider is the best by far, and stays

The run went out from a cloud session at the declaration's merge commit
(`6a864cc`), with the tree clean, on 2026-10-13 from 18:00 to 19:05 UTC.
- **Delivery.** All 6,786 requests were settled; the analysis set is all 4
  blocks; nothing was scaled down.
- **Cost.** **$1.04**, against a pilot projection of $1.82 and a cap of $8.
- **Models that could not answer.** Three never returned a usable answer and
  were smoke-stopped after 3 calls: Span-01, Span-01-lite and classify-1.
- **A brief outage.** GPT-6 Luna tripped the circuit breaker once, was
  parked, and completed in the retry pass.
- **Outputs.** `bench/battery/runs/v1/` holds every call, `summary.json` and
  `selection_v1.json`. Its scrub check is clean.

| Model | Q | handling | ex_nuis | earnings DiD | stress dH | flips | fails |
|---|---|---|---|---|---|---|---|
| **perplexity/pplx-decider-v1.1-27b** | **0.905** | k3 | 0.11 | −0.88 | +0.54 | 0.02 | G2a |
| liquid/d1 | 0.782 | k3 | 0.17 | −0.70 | +0.27 | 0.13 | G2a, G4 |
| openai/gpt-6-luna-decisions | 0.745 | k3 | 0.25 | −0.67 | +0.10 | 0.11 | G2a, G4 |
| inception/mercury-decide | 0.744 | k3 | 0.35 | −0.74 | +0.34 | 0.14 | G2a, G4 |
| jaredpalmer/kev-4b | 0.716 | k3 | 0.08 | −0.26 | +0.04 | 0.06 | G1, G2a, G3a, G3b |
| upstage/solar-decide | 0.711 | k3 | 0.06 | −0.54 | +0.09 | 0.12 | G2a, G4 |
| cloudflare/clef | 0.674 | k3 | 0.09 | −0.53 | +0.04 | 0.06 | G2a, G2b, G3b |
| nace-ai/drex-v2.0 | 0.651 | k3 | 0.14 | −0.73 | +0.09 | 0.06 | G2a |
| typesafe/jev-1.13 (reference) | 0.645 | k3 | 0.10 | −0.55 | +0.04 | 0.07 | G2a, G3b, G5 |
| microsoft/microsoft-decision-1 | 0.623 | k3 | 0.20 | −0.62 | +0.07 | 0.09 | G2a, G3b, G5 |
| nace-ai/drex-v1.5 | 0.615 | k3 | 0.14 | −0.89 | +0.08 | 0.02 | G2a, G5 |
| cloudflare/clef-flash | 0.610 | k3 | 0.10 | −0.36 | +0.01 | 0.06 | G1, G2a, G2b, G3a, G3b, G5 |
| cloudflare/clef-omni | 0.602 | k3 | 0.09 | −0.34 | −0.01 | 0.08 | G2a, G3b, G5 |
| togethercomputer/tev1-4b-experimental | 0.568 | k3 | 0.14 | −0.41 | +0.06 | 0.11 | G2a, G3b, G4, G5 |
| upstage/solar-decide-flash | 0.499 | k3 | 0.01 | +0.07 | −0.01 | 0.05 | G3a, G3b, G5 |
| scaledown/classify-1, respan/span-01, -lite | — | — | — | — | — | — | all (no valid answer) |
| *earnings rule (control), offline* | *0.599* | | | | | | |

**No model passed all eight gates, so the declared rule takes branch (c).**
The incumbent stays declared at `k1.q1` and the runtime is unchanged.

It stays on merit, not by default. **Perplexity Decider leads on every axis
but one:**
- **Q:** 0.905, ahead of the next model (0.782) and of the rule (0.599).
- **Its bootstrap:** it ranks first in 99.96% of draws.
- **Both halves:**
  - it skips 89% of skip-criteria names and wrongly skips 11% of Ordinary
    ones;
  - it hedges 83% of Stressed states and 0% of Calm ones;
  - with the earnings labels removed, Q is still 0.888.
- **Earnings:** an earnings flag lowers its p(hold) by 0.88, in every state.
- **Stress:** a stressed QQQ row raises its p(hedge) by 0.54.
- **Features, not tickers:** its answers follow the feature rows, not the
  ticker names (Φ 0.997).
- **Context:** it shows no dependence on the other questions asked with it.
- **Repeats:** identical bytes answer identically.
- **Delivery:** every call was valid, and its p95 latency was 0.9 s.

**Its one failure, G2a, is the order of the names.** Its answers move by
0.11 when only the order of the names and questions changes. Those order
draws are in every nuisance arm; the other nuisances (opaque ids, key order,
jitter, a date shift, a paraphrase) each regress to about zero. Averaging
three seeded orders (k3) brings its decision flips down to 2.4%, and Q from
0.890 to 0.905. But G2a is judged on single draws, and the declared
iteration menu allows k5 only for repeat noise, which this model does not
have.
- **What iterates next.** Branch (c) iterates on the incumbent with the menu's
  q2, s3 and b2 variants. A variant is adopted only if it makes the model
  pass every gate in a fresh holdout.

**Jev, the reference, never hedges.** Its hedge true-positive rate is 0, and
a stressed market row moves its p(hedge) by only 0.04. It does not beat the
rule by the required margin either. It stays the reference: it is what the
owner first asked for, not the best model.

**Silent model change under one build string.**
- **The bytes were identical.** The battery's requests for the real
  2026-10-13 state are byte-identical to the record's (sha `5489fee7…`), and
  the API reported the same build, `pplx-decider-v1.1-27b-20261006@Perplexity`.
- **The answers were not.** At 18:00–18:51 UTC eight answers were identical
  to each other: hedge 0.248, IREN 0.379, so IREN is skipped and the book not
  hedged. At 11:25 UTC, twice, the same bytes had drawn hedge 0.558 and IREN
  0.587, with every answer between 0.46 and 0.61. The largest difference on
  any question is 0.455.
- **Conclusion.** The build string does not pin the model's behaviour. The
  morning's "lethargic" decider was not this model's normal output.
- **The guard against it.** A nightly canary: one fixed reference request
  logged apart from the record, flagging a change the build string does not
  show.

**What the battery cannot settle, and the forward gate must.**
- **The hedge half has to beat QQQ's overnight premium.**
  - The gate grades a hedged night as the shadow book minus QQQ's overnight
    return. That return has averaged about +4.5 bps a night live and about +5
    over the long run.
  - So a decider that hedges a third of the nights passes only if QQQ
    averages about −40 bps on exactly those nights. That is crash timing.
  - RESULTS 21's evidence for hedging is about tail and alpha retention, not
    the mean. The 2.46 gate is unchanged, so this is stated now, before any
    night, not discovered later.
- **It also skips after big up-moves.** Its p(hold) falls by 0.49 after an
  extreme up-move, against 0.63 after a down-move. The basket is selected on
  overnight persistence (§15B), so the decider may skip names the strategy
  wants to hold. The shadow record will show whether that costs anything.

**Cost and the key.**
- In the stretches when nothing else was using the key, its usage moved in
  step with the run's own measured spend: $0.248 against $0.241, $0.242
  against $0.233.
- During the run's last 17 minutes the key rose $4.59 against $0.26 of
  battery spend. Earlier in the day it fell $11.77 before the run started.
  Something else is drawing on the same key.
- About $18 of credit remained afterwards.

### Iteration round 1: nothing adopted, so the decider stands as declared

Branch (c) iterates on the incumbent. Round 1 was declared before its run
(`bench/battery/iter_r1/`, PR #36). It ran the same pinned model in four
arms, each on the same 18 fresh states with a new seed, 262 calls per arm.
It cost $0.21.

| Arm | Q | ex_nuis (G2a needs ≤ 0.05) | AUC | Fails |
|---|---|---|---|---|
| **k1.q1, the control (today's wording)** | **0.907** | 0.13 | 0.970 | G2a |
| k1.q2, the hedge question points at the QQQ row | 0.907 | 0.13 | 0.982 | G2a |
| k1.q1s3, z-scores computed in code | 0.704 | 0.08 | 0.943 | G2a, G3b |
| k1.q1b2, every field defined in the book sentence | 0.759 | 0.09 | 0.967 | G2a |

- **The result replicates.** On new states the control's Q is 0.907,
  against 0.905 in v1.
- **q2 changes nothing:** dQ is exactly 0.
- **s3 and b2 trade accuracy for stability.** They roughly halve the
  model's sensitivity to name order, but cost 15–20 points of Q, and
  neither reaches the 0.05 bar. With z-scores (s3) it also stops
  responding to market stress, failing G3b.
- **The decision.** No variant clears the adopt rule, so the declared
  stopping rule ends the iteration after round 1 (`next: stop`). There is
  no round 2 and no holdout.
- **What stands.** The decider stays `perplexity/pplx-decider-v1.1-27b` at
  `k1.q1`: today's wording, one canonical call a night. The runtime does
  not change.
- **What G2a means here.** Its order sensitivity is a known property of the
  record's decider, not a defect a request view can fix.

### Reading

A decision model's speed is irrelevant to a book that trades twice a day in
auctions. What it might add is judgment on the nights where the basket's
shared risk is high. That can only be measured forward, and only the gate
decides whether it ever touches capital. Until then, the evidence-backed lever
for this book's risk is the book-level hedge (§21, and the numbers above), and
that is a change of the traded object for the owner to decide.


---

## 28. Leverage on the traded book: tested, and 1× stands — `leverage.py`

The owner asked whether leverage might perform well. §19 had rejected
leverage on the v2 book, at a flat 5%/yr. This asks again for the book that
`orders.py` actually trades: long the frozen model's Q5 basket overnight,
unhedged. It uses IBKR's real financing, commissions and margin rules, and
§26's five worlds and one-year fan, imported rather than copied.

**The decision rule, written into the script's header before any run.**
Leverage above 1× is worth considering only if all three hold:
- at the live-volatility stress, it **raises** the half-the-edge world's
  median one-year return;
- it does **not lower** that world's 5th percentile;
- its chance of a margin call stays **under 1%** in every world.

Otherwise 1× stands. Half the edge is the world a GO verdict leaves most
plausible. The live stress is the volatility the basket is actually showing.
An account that is liquidated cannot wait for its edge.

**What the account pays, as measured on 2026-10-13:**
- **Financing.** IBKR Pro's blended tiers on the actual debit: the benchmark
  is 3.88% (EFFR on 2026-10-09; the Fed hiked in September), so 5.38% on the
  first $100k and 4.88% to $1M.
- **Interest every calendar day.** Under T+1, each morning's sale and each
  evening's buy settle on the same day. The settled debit therefore never
  closes while the book trades nightly, so interest is charged every
  calendar day, about 1.45 days a trading night, which is ~2.2 bps per unit
  borrowed.
- **Reg T.** Exposure is capped at 2× equity at the end-of-day check.
- **Maintenance.** IBKR's own calculator puts maintenance at 25% on the
  11-name October basket. It rises to about 75% if only 3 names are held.
- **Portfolio margin.** It needs $110k, so neither capital tested gets it.
- **Commissions.** Leverage spreads the per-order minimums over more
  notional. At $25k, cost per traded dollar falls from 3.90 bps at 1× to 2.07
  bps at 2×. This is the one channel where leverage helps a small account.

**The data.** The real 2003–2026 walk-forward series lives only on the
owner's machine. Here the book is a **factor proxy**: real QQQ and SPY
overnight returns up to the frozen cutoff, with §21's alpha and betas and a
Student-t residual scaled to §21's residual Sharpe. Every table prints a
PROXY banner. The authoritative run is
`python src/leverage.py --real --capital 25000` on the owner's machine.

**The decision world: half the edge, at the live basket's volatility**
(34 live nights, sd 205 bps). The fan is one year, 10,000 paths, at fixed
notional:

```
                           $25,000                         $100,000
                 median     p5   P(<-20%)  dip 1/20   median     p5   P(<-20%)  dip 1/20
  1.00x           +8.9%  -35.8%     13%      -52%     +16.3%  -33.3%     10%      -53%
  1.25x          +12.8%  -48.0%     16%      -63%     +20.0%  -48.3%     14%      -65%
  1.50x          +16.8%  -61.0%     20%      -72%     +23.1%  -57.9%     17%      -71%
  2.00x          +19.0%  -67.7%     28%      -78%     +24.8%  -64.0%     25%      -77%
  QQQ held 1x    +15.6%  -18.0%      4%      -32%     +15.6%  -18.0%      4%      -32%
  QLD-like 2x    +21.0%  -40.7%     16%      -57%     +21.0%  -40.7%     16%      -57%
  (P(margin call) 0.0% in every row; financing at 2x: $940/yr at $25k, $3,935/yr at $100k)
```

**The verdict: 1× stands at both capitals.**
- Every level above 1× raises the median and lowers the 5th percentile, and
  the p5 clause vetoes all of them.
- At $25k, 2× buys 10 points of median for 32 points of p5. At $100k it buys
  8 points for 31.
- Even for someone who wants 2× exposure, holding a daily-2× QQQ fund beats
  the book at 2× on both median and tail at the live volatility.
- At $25k, plain QQQ beats the book's own 1× median.

**Read the verdict for what it is.** Leverage scales the whole fan, so
whenever the 5th percentile is a loss, it is a deeper loss at any L above 1.
The rule's p5 clause therefore could not be met by any leverage on this
book, and "1× stands" was settled by the rule's shape, not by the data. The
informative results are the price of the trade, 10 points of median for 32
of p5 at 2×, and these:
- **Funding beats borrowing.** The one real gain from leverage at $25k is
  that it spreads IBKR's per-order minimums over more notional: commission
  per traded dollar falls from 3.90 bps at 1× to 2.07 bps at 2×. That is
  the sign of a sub-scale account. More capital gets the same saving
  without the debit or the tail.
- **This was the worst candidate book.** RESULTS 20 left open modest
  leverage on the scaled or hedged book. This section levered the raw,
  unhedged book, whose live beta is 2.4. If more return is ever wanted,
  the evidence-backed order is to hedge, then volatility-scale, and only
  then, after a GO, to consider leverage.
- **Proxy only.** The Kelly figures and "P(margin call) 0.0%" come from a
  factor proxy with an iid residual, with no clustering inside the basket.
  They are not the book's numbers until `--real` has run.

**What the growth view adds (Kelly, with Reg T and maintenance applied).**
- **At the backtest's volatility (~116 bps):** the growth-optimal leverage
  sits at the Reg T cap, about 2×, in every world with an edge. 2× compounds
  faster than 1× in 4 of 5 worlds.
- **At the live volatility:** the growth-optimal leverage drops to:

  | Capital | As backtested | Half the edge | No edge |
  |---|---|---|---|
  | $25k | 1.69 | 1.24 | 0 |
  | $100k | 1.62 | 1.12 | 1.02 |

  At the live volatility, 2× out-compounds 1× only as backtested. At half
  the edge it compounds at +1% a year against +3% at $25k, and +5% against
  +10% at $100k.
- **The textbook optimum is unreachable.** The no-margin figure of ~4.8× is
  out of reach under Reg T, and portfolio margin needs $110k.

**Realized, on the 34 real live nights:**

| Capital | 1× | 1.25× | 1.5× | 2× | QQQ overnight 2× |
|---|---|---|---|---|---|
| $25k | −1.3% (dip −7.7%) | −1.4% | −1.7% | −2.7% (dip −16.1%) | +2.0% |
| $100k | −0.5% (dip −8.3%) | −0.8% | −1.0% | −2.8% (dip −16.6%) | +2.0% |

This is one short path, and evidence of nothing about the edge. It shows how
fast leverage deepens the dips of a basket whose live overnight beta to QQQ
is 2.38. At 2× the account carries about 4.5 units of overnight QQQ beta.

**Reading.** Leverage could earn its keep only if two things hold:
1. A GO verdict, which rules out "no edge" but not "half the edge".
2. The basket's volatility falling back toward the backtest's.

Even then, the fixed-notional tail cost is large. The script is the
machinery for asking again then, with `--real`. It answers nothing about this
year: no live money moves before the forward test passes (§19).

**Reviewed.** The script was reviewed through three adversarial lenses:
mechanics, statistics and honesty. 8 of 16 findings were confirmed and
fixed. Among them:
- Kelly figures ignored the margin rules;
- commissions were not re-priced when the 2× cap shrank the book;
- the expected return overstated the levered rows at the live stress;
- one proxy world was mislabelled.

The 20 tests check planted Kelly truth, financing and day counts, commission
parity with `paper_bench`, the margin-call rule, and that every proxy table
carries its banner.

---

## 29. The traded book becomes the hedged book, and the capital it needs — `orders.py`, `capital.py`

On 2026-10-14 the owner decided two things:
1. **The traded book becomes the QQQ-hedged book.** Until today `orders.py`
   held the Q5 basket long overnight, unhedged.
2. **The capital target is delegated to us**, to be set by a rule declared
   before anything was computed.

Nothing here touches the forward test. This section changes what the paper
account trades, and at what size.

### The hedged book, and why this object

Every night the book is:
- long the frozen model's Q5 basket at the close (BUY MOC, then SELL MOO at
  the next open);
- short QQQ over the same night (SELL MOC, then BUY MOO), 1:1 in dollars
  against the long notional actually bought.

That is §21's **NEU**:
- **Alpha** +3.79 bps/night at t 5.25, against the long-only book's +4.13.
  Total overnight beta is 0.42 instead of 1.42, so 93% of the return is
  alpha.
- **Measured independently (§20):** NEU 1:1 +4.17, t 5.24, Sharpe 1.10, max
  drawdown −31.7%. A trailing-beta hedge (NEUb) came out at +3.40, t 4.85.
- **Robustness (§16/§24):** +4.09 net at t 5.22, with 9%/yr volatility. It
  breaks even at 2.7× the assumed cost, the thinnest margin in the table.
- **Its gross night is the paper log's own metric**, `tilt_bps` = Q5 − QQQ
  overnight: +6.4 gross in the backtest, +4.9 after §15's 1.5 bps censoring.

**Chosen from the backtest, not from the live nights.** On the live nights:
- a 1:1 hedge cut the nightly sd from 207 to 152 bps (§27);
- a hedge at the in-sample live beta (2.37) would have cut it to 115.

That beta was fitted on the very nights being read, one theme eleven times,
so it was not used. The ratio is §21's 1:1 dollar hedge, the object with the
published alpha. §20 already established that the forward metrics do not
depend on the configuration, so this is a sizing choice, not evidence.

### What changes in the pipe

- **`orders.py`** trades the hedged book. The hedge is a dated constant,
  `HEDGE = ('QQQ', 1.0)`, not a flag.
  - **Close leg.** The basket's BUY MOC orders go in first. The short is then
    sized as round(long notional IBKR accepted / QQQ price) to the nearest
    share, from the same price snapshot, and sent as a plain SELL MOC. A
    failure can leave the old unhedged book, never a naked short.
  - **Re-runs.** A re-run tops up the short and never doubles it.
  - **Open leg.** Every long is sold, as before, and every short is covered
    with a BUY MKT/OPG.
  - **Checks before any order.** Reg T is checked against the account's
    NetLiquidation. Only QQQ is ever shorted, and a basket containing QQQ, or
    an account long QQQ, is refused.
  - **Refusals are never silent.** No QQQ price, Reg T, an account long QQQ,
    or TWS unreachable: each is logged and pushed to Telegram, naming any
    short still open.
  - **Telegram** reports what reached paper, not the dry sizing.
  - **Logs.** The hedge orders get their own log,
    `reports/orders_hedge_log.csv`. `orders_log.csv` keeps its header and
    meaning.
- **`fills.py`** pairs each QQQ cover with the latest earlier short, and
  grades hedged nights on `tilt_bps`.
- **`paper_bench.py`** prices the QQQ pair beside the long book. It closes the
  hedged night's identity against `tilt_bps` exactly, with three hedge terms:
  slippage, ratio and data.
- **`roi_year.py`** gains `--hedge RATIO`. Its default is now the traded
  book's 1.0, and `--hedge 0` reproduces §26 byte for byte.
- **`capital.py`** (new) applies the capital rule below.

### What does not change

- **The forward test.** §19's pre-registered replay (`xsec_replay.py`, the
  L/S book) and its GO/KILL table are untouched. The paper log's `tilt_bps`
  already is the hedged book's gross, so the log now grades the traded object
  directly.
- **The frozen path.** `models/`, the scoring code and the `reports/*.csv`
  history are untouched.
- **Live submission is still not implemented.** `--submit` reaches IBKR paper
  only ('D' accounts).
- **The decider's hedge question is now moot.**
  - Each afternoon `guard.py` asks whether to hedge the basket 1:1 with QQQ
    tonight. The traded book is now hedged every night, whatever the answer,
    so the answer no longer corresponds to anything the account does.
  - `guard.py`'s pre-declared grading is left exactly as it is. Its record
    still grades the shadow book against the frozen, unhedged book, as
    declared. Changing a declared gate now would void it.
  - If a decider ever passes its gate, its hedge half will need a new question
    the hedged book can act on, declared in a dated commit with a fresh
    record.

### The capital rule, declared before computing

The numbers research wrote this rule into the header of its ladder script
before any hedged-book cost was computed. `capital.py` reproduces it
unchanged, and only this rule decides:

- **R0** IBKR Pro Tiered. Fixed is printed for reference only.
- **R1** The cost is the whole nightly bill: BUY MOC + SELL MOO for each
  basket name bought, plus the QQQ SELL MOC and BUY OPG. It is priced by
  `paper_bench.order_cost` with the FINRA TAF on (the 2027 schedule), in bps
  of capital.
- **R2** The edge is the corrected tilt, +4.9 bps/night.
- **R3 TARGET** The smallest rung of $10k…$500k where every name is bought
  and the cost is ≤ 1.3 bps/night (§26's level).
- **R4 MINIMUM** The smallest rung where every name is bought and the cost is
  ≤ 2.45, half the edge. Below it, there is no live hedged book.
- **R5** The TARGET must also pass R3 with an assumed QQQ borrow of 0.30%/yr
  (Actual/360, 1.45 calendar days a night). Otherwise, the next rung up.

**A second pass changed the rule after the October outcome was known.** It
moved R4 to 2.34 (the backtest's own NEU cost) and R5's input to the measured
0.40% borrow less IBKR's interest on short-sale proceeds, and added a $1M
rung. Both verdict numbers moved with it, to $250k and $75k. Review caught
this. Those changes are kept as **proposed amendments A1–A3**:
- they are printed beside the verdict;
- they decide nothing;
- they would bind only from the first basket emitted after the owner accepts
  them.

### The ladder, the target and the minimum

October basket at the 2026-10-13 closes, Tiered, 2027 fees
(`python src/capital.py`):

```
  capital  deployed  QQQ sh  long-only  QQQ leg  HEDGED  % of edge  + R5 borrow  equity to hold
   10,000    70%*        9      6.63     0.87     7.50     153%        7.58          7,203
   25,000    87%        29      3.45     0.50     3.95      81%        4.06         22,794
   50,000    95%        64      1.97     0.38     2.36      48%        2.47         49,973
   75,000    96%        97      1.51     0.34     1.85      38%        1.97         75,794
  100,000    98%       131      1.28     0.34     1.62      33%        1.74        102,600
  150,000    98%       199      1.07     0.35     1.41      29%        1.53        155,600
  250,000    99%       332      0.90     0.35     1.25      25%        1.37        259,910
  500,000    99%       667      0.81     0.35     1.16      24%        1.28        522,049
  bps of capital per night; * 2 names unbought
```

- **TARGET $500,000.** R3 is first met at $250k (1.25, 0.05 inside), but R5's
  borrow takes $250k to 1.37. $500k passes at 1.28, **0.02 inside**.
- **MINIMUM $50,000** (2.36 ≤ 2.45, 0.09 inside).
- The research's own run on the 2026-10-12 closes gave the same answer.

**Neither number is solid, and the line that states them says so.**
- **R5's one input decides the target.** That input is the net financing rate
  on the short: the borrow fee, less any interest on short proceeds.
  - Up to 0.13%/yr, the target is $250k. Up to 0.35%/yr, it is $500k. Above
    that, no rung on the ladder qualifies.
  - IBKR's public feed showed a 0.40% borrow on 2026-10-12 (13-week range
    0.3–0.5).
  - IBKR's pricing pages say short proceeds above $100k earn BM − 1.25%
    (2.63%). If this account is paid that, the net rate is negative and $250k
    would pass. Without it, 0.40% leaves no rung.
  - The owner can read both rates in TWS for free. Under the declared rule
    that reading is information; adopting it is amendment A2.
- **The 1.3 bar sits near this book's cost floor**, about 1.1–1.25 bps/night.
  The floor is set by per-share fees on low-priced names and the SEC fee on
  two sales a night.
  - About 0.1 bps of carry moves the target a whole rung.
  - In 150 draws of 10% per-name price noise, the target left $500k 51 times.
- **The minimum is steadier.** It held at $50k in all 150 draws. A threshold
  choice moves it, not prices: under A1's 2.34, $50k misses by 0.02 and the
  minimum becomes $75k.
- **Re-apply the rule** each time a month's basket is emitted.

**The crontab moves to the target.** It passed `--capital 25000`: 3.95
bps/night, 81% of the edge, below the minimum. Paper capital is free (§25),
so the paper leg moves to the target, and the crontab lines are updated with
this section.

### Reg T

- **No offset between legs.** Reg T charges 50% of each leg at IBKR's
  end-of-day check, and a stock basket does not offset a QQQ short (12 CFR
  220.12). A 1:1 book at capital C needs about C of equity, half the long-only
  book's ceiling.
- **At the target:** long $495,932 + short 667 QQQ ($495,961) = $991,893
  gross, 1.98× capital. The Reg T floor is $495,947. `orders.py --submit` uses
  that line only to 95%, so it refuses below **$522,049 of NetLiquidation**.
- **The account type.** The short needs a non-IRA margin account; a cash
  account or an IRA rejects it. The order is a plain SELL: IBKR marks it short
  and handles the locate. IBKR's default $1M paper account clears the equity.
- **Maintenance** (25% on a 9+ name basket, ~30% on the short) leaves room for
  a one-night loss of about 45% of equity. The worst live hedged night was
  −5.7%.

### Costs

At $500k the nightly bill is certain:
- the long book costs 0.81 bps/night;
- the QQQ pair costs 0.35, about 0.2 of it the SEC fee on the short sale;
- that is 1.16 in all, 24% of the edge;
- adding R5's borrow (0.12) and the QQQ dividend the short owes (0.414%/yr,
  0.16) gives **1.44 bps/night all-in**: about $72 a night, or 3.6%/yr of
  capital.

At $25k the same book costs 3.95 (4.20 all-in).

What these books do not see:
- **Dividends.** `xsec_live.py` grades on unadjusted prices, which credit the
  short with QQQ's ex-dividend drop. A real account therefore trails the tilt
  by about 0.17 bps/night, partly offset by the basket's own dividends.
- **Paper flatters both.** IBKR paper processes no dividends, and whether it
  charges borrow is unconfirmed.

### What to expect, in each world

- **Method.** Net per night at $500k is deployed × world − all-in, in bps of
  capital. Per year it is 252 nights at a fixed notional.
- **An approximation.** The volatility columns are a normal approximation, not
  the block-bootstrap fan. That fan needs the walk-forward series, which
  exists only on the owner's machine:
  `python src/roi_year.py --capital 500000 --plan tiered`, which now defaults
  to the hedged book.

```
  $500,000, 252 nights          net/night   per year   P(loss), backtest vol (9%/yr)   P(loss), live vol (152 bps/night ~ 24%/yr)
  as backtested (+4.93 gross)     +3.45       +8.7%              ~17%                                ~36%
  half the edge (+2.61)           +1.15       +2.9%              ~37%                                ~45%
  no edge, beta only (+0.30)      -1.14       -2.9%              ~63%                                ~55%
  QQQ held, §26 (for scale)                   median +15%, p5 -17%, P(loss) 24%
```

- **This book is almost pure alpha.** With no edge it simply loses its costs,
  about −2.9% a year. The long-only book at least kept the overnight beta
  premium.
- **At backtest volatility it is a calm book:** 9%/yr against QQQ's ~20%. As
  backtested it earns about +9% a year, at under half of QQQ's risk.
- **At the live volatility its risk is about QQQ's own,** 24%/yr. In the
  half-the-edge world, the one a GO verdict leaves most plausible, it expects
  about +3% a year for that risk, and a losing year is close to a coin flip.
- **The live record so far settles nothing.** On 34 graded nights the tilt
  averaged −6.83 bps (sd 151, t −0.26). The first readable row of §19's table
  arrives around 2026-11-21.
- **No live money moves before a GO verdict** (§19). What this section sets
  is what paper rehearses, and at what size.

### Known gaps, stated now

- **Rule 201.** After QQQ falls 10% from the prior close, a short MOC does not
  take part in a cross priced at or below the national best bid, for the rest
  of that day and all of the next.
  - The long then fills unhedged, on the most volatile nights.
  - QQQ has been under Rule 201 on 2 days on record, most recently
    2020-03-17.
  - A stand-aside rule is the owner's to declare.
- **Shorting in paper is unverified.** Before the first `--submit`, a 1-share
  `whatIfOrder` on a QQQ SELL MOC should be accepted at about 31.5% initial
  margin.
- **One account, one book.** `live_momentum.py` trades QQQ from 10:31 to 15:58
  and flattens whatever QQQ the account holds at 15:58. It must not run in the
  account `orders.py` trades.
- **Cron does not read `~/.profile`.** Without it, `--submit` dials 7497
  instead of the Gateway's 4002, and no alert can reach Telegram. The updated
  crontab lines source it.

**Reviewed.** Three adversarial reviews (mechanics, integrity, numbers)
confirmed 14 findings, and all were addressed.
- **The most serious:** the capital rule had been amended after its outcome
  was known. It is now applied as declared, with the amendments shown as
  proposals.
- **The others included:**
  - an order IBKR had rejected in an earlier run still counted as working,
    which blocked the retry and inflated the hedge base;
  - refusals exited without a log line or an alert;
  - the Telegram reported the sized hedge, not what was sent;
  - the verdict line hid its margins and the input that decides it.

The tests use a fake `ib_insync` and never touch a broker. They check:
- the hedge sizing and the order of the legs;
- topping up without doubling, and the covers;
- Reg T;
- rejected and late order statuses;
- `main --submit` end to end;
- the declared rule on hand-made cost curves;
- R5's schedule against the rule itself at every rate.
