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
