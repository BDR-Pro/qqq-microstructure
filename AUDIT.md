# Adversarial audit of the trading research — 2026-09

**How this was produced.** An independent 52-agent audit run on the repository at
`main` `00db31f`: six readers (one per subsystem) built the architecture map;
six hunters (one per bias dimension — look-ahead, survivorship/universe,
execution/costs, P&L accounting, overfitting/selection, options correctness +
reproducibility) reported findings with file:line evidence; every non-LOW
finding was then attacked by two independent verifiers with distinct lenses
(code trace; materiality), defaulting to "refuted" if unconfirmable; a baseline
extractor and an edge-decomposition agent worked from the RESULTS record; a
completeness critic listed what was still missing; a final agent wrote
sections A–H. 42 raw findings → 41 unique → 18 verified upheld (2 contested),
0 refuted, 23 LOW unverified. The container holds only the synthetic panel, so
all real numbers cited are the RESULTS.md record of the user's runs.

**Bottom line.** No look-ahead leak was found; the within-month overnight
ranking edge is established beyond reasonable statistical doubt on the record
(walk-forward t 12.6, IC t 17.5, floor exit t 6.6, disjoint-slice replication
weakest t 4.8). What the audit found instead are accounting, calibration,
construction-mismatch and documentation defects that make several MAGNITUDES
optimistic or unreliable — most importantly a real bug in `factor.py` that
invalidates RESULTS 21's alpha/beta table, and a direction-known optimistic
bias in every long-only series from split/backstop censoring. The action log at
the end records what was fixed, what was made measurable, and what is
deliberately deferred to the next dated re-freeze so the RESULTS 19 clock keeps
running on the construction the frozen models were trained on.

---

# qqq-microstructure — Final Audit Report

Provenance convention used throughout: **[REAL]** = number recorded in RESULTS.md from the user's machine; **[container]** = observed on the synthetic 60-month panel in this checkout (1999-03..2004-02, 226 fake names); models/xsec_*.txt and models/xsec_lgbm.json are the user's REAL frozen artefacts (`frozen_on` 2026-08-20, `last_tmonth` 2026-03, 20,341 name-months). All line numbers were re-verified read-only in this checkout (HEAD `00db31f`).

---

## A. Current Strategy Diagnosis

**Data.** HuggingFace `mito0o852/OHLCV-1m` minute bars, per-month top-150 by that month's RTH dollar volume (`src/xsec_extract.py:27-29,59-62`), collapsed to one row per ticker-day: `open` = first bar's open, `close` = **15:59 bar's close** (16:00 bar excluded, `:59,72`), `p15/p30/p60` = open of the last bar with minutes-from-open ≤ k (`:75-77`), `dollar_vol`, `bars≥150` (`:69`). Prices unadjusted; no dividends, no delisting returns. HF stalled at 2026-03; from 2026-04 months are built from Yahoo daily/60m/15m bars by `src/xsec_extend.py` (universe = union of last 12 files re-ranked on consolidated `Volume*Close`, `:44-49,97-99`; `bars` hard-coded 390, `:121`; p15/p30 = open when the month is >55 days old, `:102,122-124`). Validation of HF vs Databento: 0.15-0.39 bps (`src/hf_history.py:4-5`; RESULTS.md:749-806). Options: Databento OPRA cbbo-1m, 10:25-10:35 ET window per day (`src/opra_pull.py:33-40`). Container has no `data/opra*`.

**Cleaning** (`src/xsec_backtest.py:load_panel:126-201`, the single path used by every consumer). Drop test symbols (`:71-73,136-140`); `on = log(open/prev_close)` only if calendar-adjacent, not within 3.5% log of a split ratio when |on|>25%, and |on| ≤ log 1.40 (`:63-64,147-155`); `id_bps = log(close/open)` always (`:156`); `cc = on+id` (`:157`); `on15` under the same mask (`:165-166`). Frozen ETF exclusion list applied at signal time (`:84-100`).

**Features → signal.** `build_table` (`src/xsec_ml.py:73-125`): for trade month T, universe `base = (uni[T-1] & uni[T]) - ETF` (`:90`), ten features from months T-12..T-1 (`mom_12_2, ret_1m, on_1m, on_12m, vol_3m, rng_3m, fh_3m, dvol, dvol_chg, lprice`; `:60-61,102-112`), rank-transformed within T (`:69-70,119-123`); targets = month-T `cc_sum / on_mean / on15_mean` (`:113-115`), rank-transformed. LightGBM regression (lr 0.03, 15 leaves, min_leaf 100, l2 10, seed 7, 300 rounds; `:62-65`), walk-forward by trade year (`:128-145`), canary = targets permuted within tmonth (`:137-140`). Rule baselines: rule B ranks on last month's mean `on_bps` (`src/xsec_backtest.py:286-290`); the ML script's only rule is `on_1m` (`src/xsec_ml.py:219-220`). Momentum leg: `sign(log(p60/open)) * log(close/p60)` on QQQ/QQQQ (`src/stack_v2.py:57-63`). Options leg: direction from the same p60 sign, 0DTE credit spread short 0.995S / long 0.985S (put) or mirror (call), at NBBO touch (`src/opra_value.py:85-97`).

**Position.** `portfolio()` (`src/xsec_ml.py:148-164`): k = n//5 per side, daily equal-weight mean of `on_bps` over names with a valid value (NaN names drop silently), L/S = Q5−Q1. Real eligible set ≈ 65 names/month on average (20,341/312), 52-53 in mid-2026 → 10 per leg (RESULTS.md:1687; `reports/xsec_paper.csv:2`). No vol targeting, sizing, leverage or stops anywhere in the xsec/stack code.

**Execution (as modelled).** Buy at the 15:59-bar close, sell at the 09:30 first-bar open ("ceiling") or the 09:45 bar open ("floor"); labelled MOC/MOO (`src/xsec_live.py:192-193`). MOM enters at the 10:30 bar open and exits at the 15:59 close with zero latency. OPT fills the whole structure at the 10:30 NBBO snapshot, held to expiry, cash-settled at the panel close (`src/opra_value.py:86-92`).

**Costs / P&L.** `QQQ_RT = 0.34` bps per day for QQQ legs (`src/stack_v2.py:39,59,63`); ON = `mlon_q5 − 2c`, NEU = `mlon_q5 − qqq_on − 2c − 0.34`, c default 1.0 (`:48-49,69-71`); OPT = `ev_sprd − $1.30/(spot·100)` (`:90`). No borrow, financing, dividends, fees or slippage beyond these constants. `stats()` compounds daily bps, iid t, Sharpe √252 (`src/xsec_backtest.py:103-116`). Books = legs summed on common days as "one pot of capital" (`src/stack_v2.py:124-126`).

**The books** [REAL, RESULTS.md:1260-1263,1278-1279,1457-1459]: QQQ_ON +4.56; MOM +3.72; ON +8.50 (long-only ML-Q5 basket); NEU +4.09 (ON − QQQ 1:1); v2 = ON+MOM +10.65; v2n = NEU+MOM +6.24; OPT +3.09; v2o = ON+MOM+OPT +19.90 (749 days only). Overlays (`src/overlay.py`): rolling-beta hedge NEUb and 21d-vol/expanding-median scaling, one-day shifted (`:64-76`).

**Risk.** In code: none beyond split/backstop NaN'ing, MIN_NAMES=40, paper-only account guard (`src/orders.py:114-121`), stale-basket refusal (`:60-64`). Risk framework = `src/mc_risk.py` circular block bootstrap (block 21, seed 7, 5% financing on the borrowed fraction, `:42-63`) producing GO/KILL thresholds (`:87-89,128-136`).

**What is frozen.** `models/xsec_on_lgbm.txt`, `xsec_on15_lgbm.txt` trained on all rows through tmonth 2026-03 (`src/xsec_ml.py:175-193`; json above). `xsec_replay.py` scores only tmonth > cut against REF 12.2 (ceiling) / 7.3 (floor) (`src/xsec_replay.py:39,81`). PARAMS/FEATS unchanged since the first commit (git).

**The forward test.** Pre-registered in RESULTS 19 (commit b599382, 2026-08-22): binding window starts 2026-08, verdict = mean of replay forward months vs the horizon row (RESULTS.md:1578-1592). Quasi-holdout 2026-04..07: +5.00/+4.88/+11.33/−1.91%, +19.3% cumulative, explicitly non-binding (RESULTS.md:1574-1577). Paper log: 2026-08 basket emitted 2026-08-21 from universe 2026-07, 10 names (`reports/xsec_paper.csv:2`); `reports/xsec_paper_daily.csv` is header-only (0 graded nights) at HEAD. What ops actually trades: **long Q5 only**, MOC buy / OPG sell, $25k, no MOM, no Q1, no hedge (`src/orders.py:37-38`; `.github/workflows/roi.yml:11,13`) — not any of the books in RESULTS 19-21.

---

## B. Bugs Found

Severity shown as the reconciled verdict; **[contested]** marks findings where verifiers split (severity given as HIGHER/lower vote). None of the upheld findings is a look-ahead leak; they are accounting, calibration, construction-mismatch and documentation defects.

### B1. Upheld

**1. [HIGH / MEDIUM — contested] Realised-return censoring of −22..−28% (and everything ≤ −28.6%) and +28.7..+38.1% / ≥ +40% overnight gaps.**
`src/xsec_backtest.py:151-155`: `is_split = adjacent & (near < 0.035) & (|on| > log 1.25)`, `too_big = |on| > log 1.40`, `on_bps = NaN` otherwise; same mask on `on15` (`:165-166`). Because the mask is symmetric in log space the downside backstop fires at −28.6%, not −40%. Basket means skip NaN names (`src/xsec_ml.py:161-162`; `src/xsec_backtest.py:300-306`), so a Q5 name's −25% earnings night vanishes from every backtest/ML/replay/MC series, while live grading (`src/xsec_live.py:128-131`) and `fills.py` book it in full. The comment claims conservatism "for both L/S signals" (`:28-30`) but the deployable object is long-only Q5/tilt (`:37-38`; `src/orders.py:37-38`). RESULTS.md never records the `overnight exclusions:` count printed at `:171-172` (grep: none). Impact: direction-known optimistic bias on Q5 +11.92, tilt +7.14 (RESULTS.md:1066-1067), ON +8.50, v2 +10.65, and on the paper-log KILL row; verifiers' plausible range 0.4-2.5 bps/day (3-21% of Q5), concentrated in the drawdown tail the MC fan claims to price. Sign and t survive. **Fix:** do not touch `load_panel` for the frozen path (see H, fix-interaction). Add a read-only `--report-gaps` mode printing every censored non-KNOWN_SPLITS gap (ticker/day/sign/quintile) and Q5/tilt with those nights re-inserted; at the next dated re-freeze replace the ratio band with an external split whitelist, keeping the band as fallback for names without split data; apply the identical mask (or none) in `xsec_live.grade` so backtest and paper log are like-for-like.

**2. [HIGH — uncontested, four independent confirmations] `factor.py` builds market factors from `ticker=='QQQ'` only with an unmasked `close.shift(1)`.**
`src/factor.py:80-88`: `g = df[df.ticker == tk]`, `log(g.open / g.close.shift(1))`, `leg('QQQ', …)`; `:96-97` inner-join + dropna; `show()` prints no n. Every other consumer aliases QQQQ (`src/stack_v2.py:57`, `diagnose.py:100`, `opra_value.py:127`, `iv_regime.py:50`, `xsec_backtest.py:306`); the real panel carries QQQQ 2004-12..2011-03 (`src/hf_history.py:31-34`; RESULTS.md:1248). Consequences on the real panel: RESULTS 21 regressions (RESULTS.md:1733-1739) run on ~73% of each book's days, omit 2005-2011 including 2008 (v2's best era, RESULTS.md:1281), and contain a ~+3,400-3,550 bps `mkt_on` point at the 2011-03 rename (and −6,931 bps at the 2000-03-20 split for MOM). Simulation by three verifiers reproduces the reported +0.20/+1.19 QQQ/SPY split from one such point; the "collinearity" explanation at RESULTS.md:1758-1764 is the bug's signature. Corroboration inside RESULTS: betas(ON)−betas(NEU) should be ≈(1,0) but is (0.29, 0.78). **Fix (exact):** in `factors()` select `df.ticker.isin({'QQQ','QQQQ'})`, `sort_values(['day','ticker']).groupby('day').first()`, and build `mkt_on/mkt_cc/mkt_id` from load_panel's masked `on_bps/cc_bps/id_bps` (or require dpos adjacency); print n and date window in `show()`; re-run RESULTS 21 and `research_report.py` section 6 (`src/research_report.py:226`). Production models, stack_v2 and xsec_backtest are unaffected.

**3. [MEDIUM / LOW — contested] Month-T membership conditioning (`uni[T-1] & uni[T]`) in every historical series, absent live.**
`src/xsec_ml.py:90`, `src/xsec_backtest.py:227,283`, `src/xsec_replicate.py:70`, `src/xsec_replay.py:80-81` (inherits), `src/mc_risk.py:128-136` (inherits); live uses `months[-1]` only (`src/xsec_live.py:173-174`). Documented (`src/xsec_backtest.py:15-18`; RESULTS.md:1015-1017) but only attrition (14.1%/mo) is reported, never returns; unmeasurable from `data/xsec` because dropped names have no month-T rows (`src/xsec_extract.py:61-63`; verified [container]). Materiality bounded: eligible-set loss is ~1-5%/month, not 14% (synthetic 1.3%); two sub-claims (on_n≥8 target gate: 2 of 20,341 real rows; MIN_ON_DAYS in replicate) are immaterial. `tests/test_no_lookahead.py:55,73` intersect indices so cannot detect it. RESULTS.md:1814 "survivorship-free" overstates. **Fix:** measure on the wide panel (H.2 #4); re-derive the paper-log tilt thresholds without `& uni[T]`; correct the wording.

**4. [MEDIUM / LOW — contested] Yahoo extension: closed, monotone-shrinking candidate pool; different `dollar_vol` definition; Yahoo months never replaced by HF.**
`src/xsec_extend.py:44-49` (no `src` filter → pool(m+1) ⊆ pool(m)); `:97-98,121` consolidated `Volume*Close` vs RTH minute dollars (`src/xsec_extract.py:59-60,74`); `src/xsec_extract.py:44-46` returns −1 if file exists and `src/xsec_replay.py:45-48` only fetches months > last file, so the header's "until HF catches up" (`src/xsec_extend.py:14-16`) is false as coded. No `--check` output recorded (grep RESULTS.md: none). Every binding forward month from 2026-08 is Yahoo-built; from 2027-04 the lookback holds no HF file and the live basket becomes a 2025-26 cohort. `dvol_chg` straddles the seam for tmonths 2026-05..2027-04 (rank-transform mitigates). **Fix:** filter `candidates()` to rows with `src != 'yahoo'` or a frozen HF pool plus an external liquidity screen; allow HF to overwrite `src=='yahoo'` months; run and record `--check`.

**5. [MEDIUM / LOW — contested] `c = 1.0` bps per auction crossing is asserted, mis-cited to §15b, and below the commission floor at the rehearsal size.**
`src/stack_v2.py:26-27,48-49,69,71`; RESULTS.md:1255-1256. §15b is a price-source test (RESULTS.md:1155-1175; `src/xsec_auction_check.py:26-28`), not a cost model; grep `commission` hits only `fills.py` and option scripts. At `--capital 25000` / 10 names (`roi.yml:11,13`; `reports/orders_log.csv:4` $2,212/order) IBKR Pro fixed $1.00 minimum = 4.5 bps one-way (9.0/night vs modelled 2.0); tiered ≈1.6-1.7. ON net at that size ≈ +1.5 (fixed) / +7.1 (tiered) vs +8.50; tilt ≈ −2.6 / +3.0 vs +6.4. GO/KILL are gross and unaffected. **Fix:** decompose c (schedule with per-order minimum as a function of slice size, exchange cross fees, TAF/SEC on sells); print implied c for the `--capital` run; `orders.py` warn when `capital/len(names) < $10,000`; carry `fills.csv` commission bps back into `--c`.

**6. [MEDIUM / LOW — contested] MOO/MOC collectability asserted for 1999-2026 on a 2012+ test.**
RESULTS.md:1175,1205 generalise the 75-name 2012+ Yahoo check (RESULTS.md:1155-1169) to a series starting 1999-03; Nasdaq had no closing cross before 2004-04 / opening cross before 2004-12 and fractional ticks to 2001-04 — nowhere caveated (grep). Pre-2004 eras carry ~4 of the ~10 bps/day rule headline (RESULTS.md:1070-1071); ML OOS starts 2003-01 (23 of 279 months affected), v2 starts 2003, MC window 2004-02+, forward test unaffected. **Fix:** caveat in RESULTS 15/15b; print the floor era table (only the ceiling era table is printed, `src/xsec_backtest.py:332-334`).

**7. [MEDIUM / LOW — contested] MOM leg charged a flat 0.34 bps RT for 27 years.**
`src/stack_v2.py:39,61-63`; `src/momentum_backtest.py:11-13` calibrates to "~1 tick on ~$500"; same constant in `overnight_study.py:18`, `momentum_ml.py:23`, `research_report.py:45`. Committed real `data/hf_bars/QQQ` show medians $103-211 (1999-2000, 1/16 ticks) and $22-43 (2001-04, $0.01), Roll spreads 3-31 bps. Full-sample MOM +3.72 (RESULTS.md:1261) overstated by ~0.4-1.0 bps/day (t 2.53 → ~2.0-2.3); v2 (2003+) moves <2%; QQQ_ON, replay, verdict unaffected. **Fix:** `cost_t = max(0.34, k·tick_t/price_t·1e4)` with tick 1/16 before 2001-04, else 0.01; re-print era tables.

**8. [MEDIUM — uncontested] `optbacktest` "fair" IV (mult 1.0) already sells rich.**
`src/optbacktest.py:78` annualises 20-day std of `cc_bps` (overnight + intraday, `src/xsec_backtest.py:157`) and prices the 10:30→close window with `T = SF/252` (`:57,80`); implied window variance = 0.846 × daily vs realised ≤ intraday share minus first hour. Header `:19-23` and printout `:173-176` ("variance premium is ZERO") are false; [container] var(window)/var(cc) = 0.64 and a delta-neutral condor prints t 2.5 positive at "zero premium". No tradeable (real-mode) number moves; RESULTS.md:1791-1797 model-mode readings are overstated. **Fix:** rv from `log(close/p60)` scaled by √(252/SF), or scale cc-sigma by √(var(ec)/var(cc)); re-run the planted-truth check on a synthetic with overnight + first-hour variance and record it.

**9. [MEDIUM / LOW — contested] Physical assignment / overnight stock on breached-strike days unmodelled.**
`src/opra_value.py:86-92` cash-settles intrinsic at the 15:59 panel close, entry-leg commissions only (`:20-22`); feeds `src/stack_v2.py:88-90` → OPT/v2o and the v2o MC row. RESULTS.md:1444-1446 discloses but never quantifies. On real `data/daily.parquet` (515 days) ~12% of days finish partial-ITM. Hold-through: Sharpe 2.70 → ~1.5-1.6, −94 cap non-binding. Documented remedy (buy back at 15:55): ~0.1-0.2 bps/day, Sharpe ≈2.6. Also breaks the "never hold at the same time" one-pot premise (`src/stack_v2.py:6-8`). **Fix:** add an explicit exit rule (book next-day open on ±100 sh, or model a 15:55 buy-back from a 15:55 chain slice) and state which variant v2o uses.

**10. [MEDIUM / LOW — contested] "ML doubles the rule" is measured only against `on_1m`.**
`src/xsec_ml.py:219-220`; RESULTS.md:1113-1114,1203. `on_12m` is the model's top feature (28% gain; real model 28.1% vs on_1m 7.9%) and §22's on_12m rule prints +13.16 t 12.9 (RESULTS.md:1819) in a different construction; no same-window ML-vs-on_12m comparison exists. [container] on the planted panel: on_1m +4.94, on_12m +11.27, ML +9.84 — the model trails the one-line rule. Real increment plausibly +2..+4 bps/day, not 2×. **Fix:** add `t['on_12m'].where(oos)` (and the on_1m/on_12m average) as baselines in `xsec_ml.main()` on identical OOS rows; report ML−best-rule with its own t; restate RESULTS 15/15b.

**11. [LOW / MEDIUM — contested; leak hypothesis refuted] The ceiling canary (+1.6..+1.8, t 2.85) and the post-hoc "chance floor".**
`src/xsec_ml.py:22-24` pre-declares "stop trusting"; RESULTS.md:1199-1203 re-labels. Both verifiers agree: the canary is one deterministic draw (`rng = default_rng(ty)`, `:137`), a permuted-target GBM is a random projection onto real feature signal so its iid daily t is inflated; [container] 12 alternative seeds give mean +0.12, sd 1.44, |t|>2 in 3/12. No code asymmetry exists (identical mask for on/on15, `portfolio()` exactly antisymmetric) and the on15/cc canaries at ~0 refute a feature-level leak. What stands: the rule was overridden after the number, and "floor" is a single draw, not a null distribution. **Fix:** multi-seed canary null + year-clustered t; record the reinterpretation as a dated decision. No subtraction from REF is warranted.

**12. [LOW / MEDIUM — contested] Paper-log GO/KILL rows calibrated on a construction the live grade does not share.**
`src/mc_risk.py:134-136` tilt = `mlon_q5 − qqq_on` (conditioned universe, k≈13 historically, masked prices) vs `src/xsec_live.py:126-131,173-174` (T-1 universe, 10 names, raw Yahoo, ≥5-name floor). Price-source difference measured nil (RESULTS.md:1160-1168); universe difference bounded ~0.2-1 bps; backstop difference is #1. The **binding** replay rule (RESULTS.md:1585-1591) is internally consistent. **Fix:** rebuild the tilt series live-style or footnote the paper-log row's power columns as approximate; do not restart the clock.

**13. [LOW — uncontested] Vol-overlay PASS/FAIL compares raw vs scaled on independent bootstrap draws.** `src/overlay.py:92` consumes one rng twice; `src/mc_risk.py:52` draws fresh block starts; strict `>` at `:100-101`. v2nb "0.37 vs 0.37*" (RESULTS.md:1651-1653) is inside noise (synthetic unpaired SD ≈0.006). v2/v2n gaps are 19-50 SD. **Fix:** draw `start` once and index both series; bootstrap the paired ratio difference with a CI.

**14. [LOW — uncontested] Capacity estimate: k from the 150-name file (=30) not the 10-14-name basket; single AUCT=0.08 applied to both the open and close cross; p25 ADV labelled "median".** `src/research_report.py:166-175`. Advisory only; never in RESULTS. **Fix:** k from `xsec_paper.csv`; separate open/close auction shares, take the min; relabel.

### B2. Refuted / cleared concerns (do not re-raise)

- **Feature look-ahead in `build_table`**: every feature uses months ≤ T-1, targets month T only, rank transform is within-month (`src/xsec_ml.py:69-70,91-123`); confirmed by test and parity check.
- **Walk-forward embargo**: not needed — Dec ty-1 targets realise before Jan ty formation (`:131-134`).
- **Frozen-model contamination**: freeze commit 1a5e8f5 (2026-08-20 01:56 UTC) precedes the first post-2026-03 month on disk (d67908d, 23:59 UTC); PARAMS/FEATS unchanged since a41598a.
- **GO/KILL thresholds contain forward information**: no — derived from the series ending at the cutoff (`src/mc_risk.py:128-136`), and Apr-Jul 2026 are declared non-binding.
- **Overlay lags**: beta and vol weights are shifted once, compared on identical days (`src/overlay.py:64-76,152`).
- **Options DST/timing**: `opra_pull.window_utc` and `entry_chain` are per-day America/New_York-correct (verified numerically for EST/EDT dates); the RESULTS 10 bug is not in the live path.
- **Put-call-parity spot, IV formula, structure sides/payoffs, commission arithmetic**: all correct; parity spot vs p60 |diff| 0.4 bps (RESULTS.md:1378).
- **MOM signal/entry share the 10:30 print**: stated assumption; bounce cannot bias the sign.
- **Auction-print "bounce" manufacturing the overnight edge (2012+)**: refuted by the Yahoo official-print check (print component −0.08 bps/day, corr 0.9905, RESULTS.md:1163-1170).
- **Ticker renames manufacturing returns**: per-ticker shift + calendar adjacency NaN the first night (`src/xsec_backtest.py:143-147`).
- **QQQ/QQQQ alias**: handled everywhere except `factor.py` (finding #2).
- **stats()/cost algebra/MC bootstrap/Newey-West estimator**: arithmetic verified; HAC ratio on AR(0.5) matches theory.
- **Hyperparameter/feature tuning on walk-forward numbers**: git shows none; entry-time and threshold sweeps were retracted/killed and reported as such.
- **Test symbols / exchange placeholders**: dropped at load, count recorded (RESULTS.md:1023-1026).

### B3. LOW hygiene items (unverified by a second reader; act on the cheap ones)

- `tests/test_no_lookahead.py:46-50` poisons prices after `load_panel`, so `cc/on/on15`-derived features and all targets are never poisoned; and no CI step runs pytest (`monthly.yml:51-60`).
- Month-boundary night (T-1 close → T open) is attributed to month T's basket in backtest/replay (`src/xsec_ml.py:160-162`) but to T-1's basket by `grade()` (`src/xsec_live.py:124-126`); ops emits mid-month so that night is untradeable.
- `range60` includes the 10:30 bar's high/low (`src/hf_history.py:108-113`); `momentum_ml.py:112,119-120` sizing variants use `bfill` and a full-sample median (rejected variants only).
- Yahoo months >55 days old carry `p15 = open`, so the on15 "floor" equals the ceiling for 2026-04..06 in replay/diagnose output without a flag (`src/xsec_extend.py:102,122-124`).
- `src/xsec_intraday.py:41-43` drops days on the realised hold-period return (|rest| > 2500).
- `queue_sim.py:129-134` phantom fills between decision and arrival (killed subsystem).
- `signal_ic.py:12-14`, `markout_day.py:10` still hard-code 13:30 UTC; `selective_quoting.py`/`markout_multiday.py`/`state_reachability.py` do not import (RESULTS §1-5 not reproducible).
- Dividend direction misstated for the short leg (`src/xsec_backtest.py:31-33`; RESULTS.md:1034-1035): Q1 is overstated, so L/S is roughly neutral-to-optimistic, not "understated"; Q5/tilt remain understated. Magnitude unmeasured (~0.3-0.6 bps/day/leg).
- TVIX missing from the ETF list (`src/xsec_backtest.py:91`) in addition to the documented SGOV leak.
- `research_report.py:45-46,187-190`: hard-coded BASE_COST ignores `--c`; v2o gets `base=0` → infinite break-even, auto-passes SURVIVE.
- Sortino = mean/std(negative days) in three scripts (`research_report.py:59`, `optbacktest.py:313-315`, `momentum_backtest.py:50,55`), ~13% high; never recorded.
- `fills.py:230-237` charges full b.comm+s.comm against `q = min(qty)`; sell pairs to the latest prior buy so multi-night holds/orphans are misreported (`:86-91`).
- Momentum ML recorded at three values (3.85 / 3.1 / 2.94; RESULTS.md:874-876, `models/momentum_lgbm.json`, RESULTS.md:927); feature set selected on OOS.
- §18 strikes are §8's best in-sample cell on overlapping days (RESULTS.md:578-583 vs `src/opra_value.py:90-101`).
- `xsec_ml.py` walk-forward path ignores `--through` (`:200-211,291-292`); a re-run after `xsec_extend` silently folds 2026-04..07 into `xsec_ml_daily.csv` and every downstream threshold.
- Settlement at the 15:59 bar, not the official close; commission bps at mean spot in `opra_value.py:166` vs per-day in `optbacktest.py:109`; SF applied on early-close days; legacy `opra_load.py` writes the same parquet path with an incompatible schema; `near()` + zero-bid filter can narrow the declared width; no options tests; unpinned dependencies (pandas 3.0.5 / numpy 2.4.6 in container vs `>=` bounds).
- Ops: month-rollover hole — `orders.py:60-64` raises with no basket, open leg sells only current-basket names (`:133-138`), so Aug-31 MOC positions can sit unsold until the Monday cron emits September's basket; `notify._chat` pins any stranger who messages the bot (`notify.py:30-36`).

---

## C. Baseline Results

All [REAL] unless noted. Net of the stated (and, per B1 #5/#7, understated) costs.

| Book | bps/day | t (iid daily) | Sharpe | CAGR / maxDD | %/mo | Window | Cost | Source |
|---|---|---|---|---|---|---|---|---|
| QQQ_ON | +4.56 | 4.14 | 0.80 | — | +0.87 | 1999-2026, 6,798d | 0.34 RT | RESULTS.md:1260 |
| MOM (sign rule) | +3.72 | 2.53 | 0.49 | (HF spine: 7.67% / −47.0%) | +0.63 | 1999-2026, 6,803d | 0.34 RT | :1261, :971 |
| ON (ML-Q5 long) | +8.50 (gross 10.50) | 5.65 | 1.17 | — | +1.66 | 2003-2026, 5,843d | 2×1.0 | :1262 |
| NEU (ON − QQQ) | +4.09 | 5.22 | 1.09 | (§20 window: 10.57% / −31.7%) | +0.82 | 5,841d | 2.34 | :1263, :1620 |
| NEUb (β-hedged, β̄ 1.25) | +3.40 | 4.85 | 1.02 | 8.56% / −26.6% | +0.69 | 5,715d | 0.34β | :1614-1621 |
| v2 (ON+MOM) | +10.65 | 5.45 | 1.13 | 27.14% / −43.1% | +2.02 | 5,843d | 2.34 | :1278 |
| v2 vol-scaled | +8.90 | 6.53 | 1.39 | MC p95 DD −27.0% | +1.77 | 2004-02→2026-03 | 2.34 | :1646 |
| v2n (NEU+MOM) | +6.24 | 4.27 | 0.89 | 15.21% / −31.2% | +1.19 | 5,841d | 2.68 | :1279 |
| v1 @ v2 window | +5.88 | 3.66 | 0.76 | 13.79% / −43.7% | +1.08 | 5,843d | 0.68 | :1277 |
| OPT (spread .5/1.5) | +3.09 (gross 3.37) | 4.26 | 2.47 | — / −2.2% | +0.71 (arith.) | 2023-04→2026-03, 749d | comm −0.27 | :1404-1412, :1457 |
| OPT REAL (§21, to 2026-07) | +2.92 | — | 2.32 | — / −2.8% | — | 833d | measured quotes | :1782 |
| v2o (ON+MOM+OPT) | +19.90 | 3.32 | 1.93 | — / −24.9% | +3.97 | 749d | 2.34 + comm | :1458-1459 |
| ML L/S ceiling (replay metric) | +12.38 / +12.2 | 12.58 | 2.61 | — / −20.5% | — | 279 OOS mo, 2003-01→2026-03 | gross; BE ~3.1 one-way | :1114, :1207, :1560 |
| ML L/S floor (09:45) | +7.29 | 6.61 | 1.37 | — / −36.2% | — | same | gross; BE 1.20 | :1185-1190 |
| Rule on_1m L/S | +6.20 | 6.41 | 1.33 | — / −26.4 | — | same | gross | :1113 |
| Rule B Q5 / tilt | +11.92 / +7.14 | 8.50 / 8.90 | 1.64 / 1.72 | 32.8 / 19.1% ; −44.0 / −32.1% | — | 1999-2026 | gross | :1066-1067 |

**Forward record.** Quasi-holdout Apr-Jul 2026 L/S: +5.00 / +4.88 / +11.33 / −1.91%, +19.3% cum. ≈ +23 bps/day (RESULTS.md:1574-1575; only July's floor, −26.91, is recorded, :1704). Binding: 2026-08 onward; paper log empty at HEAD.

**Pre-registered GO/KILL** (RESULTS.md:1558-1571; `src/mc_risk.py:129-135`): replay metric GO/KILL 3mo 15.1/−2.6, 6mo 11.1/1.6, 12mo 7.8/4.6 (P(pass|real) 83%), 24mo 5.5/6.8; tilt 3mo 12.5/−5.0 … 12mo 7.0/0.4 (P(pass|real) 41%), 24mo 4.8/2.0.

**Benchmarks.** QQQ overnight-only B&H +5.08 gross / +4.74 net, t 4.56, Sharpe 0.82, CAGR 11.5%, maxDD −30.6% (RESULTS.md:958,970); QQQ intraday −1.43 (t −0.81); SPY overnight +2.28 (t 2.65) (:958-959). QQQ close-to-close B&H is **not recorded**; derived here as 5.08 + (−1.43) ≈ +3.65 bps/day gross (no t/Sharpe/DD on record). No benchmark row exists in `research_report.py`.

**Gaps (not recorded anywhere).** Sortino/Calmar/profit factor/win-rate for the equity books; measured basket turnover (`TURNOVER` strings are hard-coded, `src/research_report.py:46-50`); drop-best-N-days fragility for ON/NEU/v2; cost break-even sweep and SURVIVE verdicts; block-bootstrap CI on mean bps/day per book (`research_report.py:143-152` exists, never run on record); replay per-month bps/day + IC and the Apr-Jun floor; capacity; ceiling-ML era table (only "positive in all six eras", RESULTS.md:1118); the load_panel exclusion counts.

**Recorded inconsistencies.** Two series both called "ON" (replay metric = `mlon_ls` +12.2 vs book = `mlon_q5 − 2c` +8.50); QQQ overnight gross reported as 5.08 / 4.90 / 4.78 and QQQ day counts 6,285-6,803 across sections; v2 on three windows (+10.65 / +10.84); NEU/QQQ_ON corr +0.39 / +0.40 / +0.55 (`src/overlay.py:10` stale); §21 OLS identity fails for MOM/NEU/v2n (mean ≠ alpha + Σβ·mean F) — consistent with B1 #2; ceiling ML +12.38 / +12.23 / +12.2; §12's "do not fund" holdout is for the momentum ML, applied to the sign-rule MOM leg.

---

## D. Edge Decomposition

Arithmetic from the recorded legs on the common 5,843-day window (identities from `src/stack_v2.py:59-71`): v2 +10.65 = MOM +2.15 (20%; v2−ON = v2n−NEU) + QQQ overnight premium net +3.73 (35%; v1@v2 − MOM) + NEU +4.09 (38%) + hedge cost 0.68 (6%). MOM's full-sample +3.72 therefore implies ~13 bps/day in the 1999-2002 stub (consistent with RESULTS.md:855-857).

**ON.** *Supported:* (i) overnight-market beta ≈1.4-1.55 explains ~70% of the book (R² 0.75; betas sum 1.39, RESULTS.md:1735; rolling β̄ 1.25, last 1.55, :1616,1628-1630) — but note the beta *split* and R² come from the buggy §21 regression; the sum and the rolling estimate are independent and agree. (ii) A within-month overnight-persistence ranking edge exists: ML L/S +12.38 t 12.6, IC +0.214 t 17.5, positive in all six eras, one losing year (RESULTS.md:1113-1118); on_12m rule replicates BROAD in disjoint hash slices, weakest t 7.5 (RESULTS.md:1819-1826); on15 floor +7.29 t 6.6 positive in all 7 eras (:1185-1195). (iii) Mechanism is timing, not stock selection: Q5−Q1 intraday −12.22 t −6.39, cc ML null (:1055,1074-1078); half the ceiling is given back by 09:45 (:1084-1092). *Suggestive:* the size of the selection component inside the long-only book (alpha +2.45 t_HAC 2.81 from the buggy regression vs NEUb +3.40 t 4.85). *Unsupported:* day-concentration (never measured); model value-add over on_12m (B1 #10); the rule's decay (16-19 −0.4, 20-23 +1.2, RESULTS.md:1070-1071) versus the claim the ML smooths it (era table not printed).

**MOM.** *Supported:* a long-run first-hour continuation in QQQ, sign rule +3.94 t 2.55 gross, 18/27 years positive, replicates 3/4 ETFs (RESULTS.md:848-857); zero beta by construction (R² 0.00). *Suggestive:* crisis-regime concentration (+21-30 in 2001/02/08/22, ~0 in 2013-17). *Contrary:* only true holdout is negative (−9.01 momentum-ML, −5.86 sign rule, :899-903); costs understated pre-2005 (B1 #7); §21 alpha +4.42 exceeds the leg's mean on any window (identity fails).

**NEU.** *Supported:* a positive hedged residual exists (NEUb t 4.85, corr with QQQ_ON +0.03, RESULTS.md:1616-1628). *Contested in the record:* §21 says +1.13 t_HAC 1.47 — attributable to B1 #2 (sample truncation + leverage point) but unproven until re-run. Residual beta ~+0.3 (1:1 under-hedges; §21 text says "over-hedges", :1769 — wording error).

**v2.** *Supported:* ~54% beta, ~20% MOM direction, ~26% net selection; identity reconciles exactly (implied factor mean 4.06 = QQQ_ON gross 4.07); positive all seven eras, best in 2008-11 (:1281-1287). *Unsupported:* share of P&L from top-N days.

**OPT.** *Suggestive only:* premium collected in both IV cells (+2.7 / +5.4), direction-only long ATM dead (+0.4 t 0.2), condor 51-68% of the spread EV, corr(OPT, MOM) +0.67 (:1404,1415-1419,1456); no premium/skew/direction split recorded; decay 4.7→4.0→2.5→0.2 by year (:1412); 2023-26 window has no crash; strikes are §8's in-sample best cell; assignment unmodelled (B1 #9); model-mode "directional-only" claims are void (B1 #8).

---

## E. Robustness Evidence Already on Record

| Test | Where | What it shows | Limit |
|---|---|---|---|
| Walk-forward by year, 279 OOS months | `src/xsec_ml.py:128-145`; RESULTS 15/15b (:1109-1118) | ML L/S +12.38, IC t 17.5, all eras positive | Same universe conditioning throughout; canary single-seed |
| Floor vs ceiling (09:45 exit) | RESULTS 15/15b (:1084-1096,1185-1195) | +7.29 t 6.6 survives, all 7 eras | Floor never era-tabled for the rule |
| Official-print cross-check | `src/xsec_auction_check.py`; RESULTS 15b (:1155-1175) | corr 0.9905, print component −0.08 bps/day | 75 names, 2012+, listed only; censored nights excluded by construction |
| Frozen-model quasi-holdout | RESULTS 19 (:1574-1577) | +19.3% over Apr-Jul 2026 | Non-binding; Yahoo-built months; conditioned universe |
| Disjoint-subset replication | `src/xsec_replicate.py`; RESULTS 22 (:1819-1849) | on_12m and on_1m rules BROAD, weakest t 4.8 | Rule not ML; within top-150; k=5 underpowered |
| Cost stress | RESULTS 16 (:1287-1290) | v2 ≈ +7.7 at c=2.5 still > v1 | Only two cost points; flat costs; no per-order minimum |
| Monte Carlo (block 21, 10k paths, 5% financing) | `src/mc_risk.py`; RESULTS 19 (:1509-1526) | v2 1× CAGR p5/p50/p95 8.4/27.4/48.4, maxDD p95 −48.8, P(month<−20%) 25.7%; 3× ruin | Bootstraps the censored series; no ruin truncation; v2o on a 749-day bull window (P(5y loss) 0.0% is meaningless) |
| Regime / era tables | RESULTS 15, 16 (:1070-1071,1281-1282) | Rule decays 2016-19; v2 positive all seven eras | Ceiling-ML era table unprinted |
| Factor regression, NW lag 5 | `src/factor.py`; RESULTS 21 (:1733-1739) | ON/v2 alpha t_HAC 2.8/3.3 | **Computed on a truncated, corrupted factor series (B1 #2); must be re-run** |
| Beta hedge / vol overlay | `src/overlay.py`; RESULTS 20 (:1608-1675) | v2 scaled: worst-month −15.8%, P(−20%) 0 | v2nb verdict inside MC noise |
| Rejected/killed tests recorded | RESULTS 7-8b, 12, 14 | Market-making, entry-time sweep, calendar, trailing-quarter filter | — |

**Remains untested:** return impact of `uni[T]` conditioning (wide panel never built); count/identity of censored non-split gaps; dividends per leg; day-concentration for ON/NEU/v2; measured turnover; bootstrap CIs per book; ML vs on_12m same-window; Yahoo seam (`--check`); ceiling-ML era table; recent-era (2016+) GO/KILL power; any options tail regime; any ops-side realized-vs-graded gap (paper log empty).

---

## F. Statistical Significance

**Independent observations.** Formation is monthly; nights within a month share one basket, so the effective sample for the cross-sectional edge is ~279 trade months (2003-2026) or 316-325 for the rule, not 5,843 days. The monthly rank-IC t of 17.5 over 279 months (RESULTS.md:1117) is the right headline statistic and is, by itself, overwhelming. Every daily t in RESULTS other than §21 is the iid `mean/(std/√n)` of `src/xsec_backtest.py:110-111`; no HAC, no month clustering. Daily overnight basket returns are close to serially uncorrelated in mean but strongly vol-clustered, so the iid t overstates by a modest factor (order 1.2-1.5), not by the √21 a naive month-clustering would imply.

**Where it matters.** ML L/S t 12.6 and IC t 17.5 survive any plausible correction. Tilt t 8.9 / NEU t 5.2 survive. MOM t 2.53 (→ ~2.0-2.3 after B1 #7) does not survive a family-wise correction. OPT t 4.65 is on 749 days of one regime with an 87% win rate — a short-vol strategy's t-stat is not evidence about its tail. v2o t 3.32 on 749 days is the sum of a bull-window ON (2024-26 era +17.4) and OPT.

**Multiple-testing exposure.** At least eight strategy families were tried across RESULTS 1-22 (ITCH market-making, QQQ momentum sign/ML/entry-time sweep, calendar effects, cc/on/on15 ML targets, XID, 0DTE long/spread/condor, IV gate); within the surviving family the visible degrees of freedom are: target choice (cc dropped as null, disclosed), exit (ceiling vs floor, both reported), book configuration (v1/v2/v2n/v2x/v2o, all printed), rule baseline (on_1m only — B1 #10), strike (§8 best cell), momentum feature set (chosen on OOS). Nothing in git indicates hyperparameter tuning of the frozen model. The overnight cross-sectional persistence is also an externally documented anomaly, which reduces (but does not remove) the data-mining prior.

**Bootstrap CIs.** Block-bootstrap ranges exist only for CAGR/maxDD/monthly sums (`src/mc_risk.py`); a CI on mean bps/day per book is coded (`src/research_report.py:143-152`) but never recorded. The overlay verdict is unpaired (B1 #13). The canary "floor" is a single deterministic draw with no null distribution (B1 #11).

**Frank assessment.** The existence of a within-month overnight ranking edge in the top-150 universe is established beyond reasonable statistical doubt on the record, subject to two magnitude caveats that are direction-known (censoring, pre-2004 execution) and one that is not (uni[T]). The statistical case for the MOM and OPT legs is weak-to-marginal after accounting for costs, regime concentration and selection. The factor-alpha table that is supposed to separate selection from beta is presently unreliable. The forward test has zero binding observations.

---

## G. Final Classification per Book

**ON — ROBUST EDGE.** The underlying ranking signal is confirmed by walk-forward (t 12.6, IC t 17.5), floor exit (t 6.6), disjoint-slice replication (t ≥ 4.8), and an official-print cross-check, with no look-ahead found in the feature or split construction. Qualifiers: ~70% of the book's return is overnight-market beta (≈1.4), so it is a levered overnight-premium exposure with a selection tilt; the long-only level is optimistically biased by an unrecorded amount from crash-night censoring (B1 #1) and by pre-2004 execution assumptions; net-of-cost figures are valid only above ~$10k per order. The forward test has not yet produced a binding month.

**MOM — PROMISING BUT INSUFFICIENT EVIDENCE.** A 27-year t of 2.5 (≈2.0-2.3 after era-correct costs), 18/27 years positive and zero beta is real but marginal, concentrated in crisis regimes, and its only true holdout is negative (RESULTS.md:899-903). The record's own verdict is "do not fund" (RESULTS.md:935,1296-1297); inside v2 it contributes ~2.15 bps/day. Its §21 alpha does not reconcile with its mean.

**NEU — ROBUST EDGE (size disputed).** A hedged residual exists at t 4.85 (NEUb, corr +0.03 with QQQ_ON) and the unhedged 1:1 version prints t 5.2 over 23 years; this is the cleanest expression of the selection component. The size is unresolved between +3.4 and +1.1 (t_HAC 1.47) until the factor regression is rebuilt (B1 #2); the residual beta ~+0.3 means "neutral" is a misnomer.

**v2 — ROBUST EDGE (inherits ON; MOM increment unproven).** Positive in all seven eras, identity reconciles with §21, MC tails priced at 1×. About half is beta, a fifth is the marginal MOM leg, and the 26% probability of a −20% month at 1× is the cost of the beta. Dropping MOM would not change the classification; adding leverage was correctly vetoed.

**v2n — ROBUST EDGE (inherits NEU; weaker).** t 4.27 / Sharpe 0.89 with lower drawdown; a third of its mean is the MOM leg, so its margin over the well-established NEU is not itself demonstrated. Scaled v2n is the record's preferred low-tail configuration (RESULTS.md:1647-1648).

**OPT — PROMISING BUT INSUFFICIENT EVIDENCE.** 749-833 days in a single no-crash regime, edge decaying 4.7→0.2 by year, bundled premium + direction with the direction-only version dead and corr 0.67 with MOM, strikes taken from an in-sample sweep on overlapping days, physical assignment unmodelled, and every model-mode "directional-only" reading void (B1 #8). A Sharpe of 2.7 on a short-gamma structure with 3 years of data is not evidence of robustness.

**v2o — PROMISING BUT INSUFFICIENT EVIDENCE.** +19.9 bps/day is the sum of ON in its hottest era (+17.4 in 2024-26) and OPT over 749 bull days; the MC "0% P(5y loss)" bootstraps that window and is uninformative; OPT's overnight-assignment exposure violates the one-pot premise. Classification would follow OPT's until OPT has a regime.

---

## H. Recommended Actions

### H.1 Bug fixes to make now (in order)

1. **`src/factor.py` → `factors()` / `show()`** → select `df.ticker.isin({'QQQ','QQQQ'})`, `sort_values(['day','ticker']).groupby('day').first()`, build `mkt_on/mkt_cc/mkt_id` from load_panel's masked `on_bps/cc_bps/id_bps` (same for SPY); print n and date window; re-run `factor.py` and `research_report.py` section 6, replace RESULTS 21, check the OLS identity per book → **reason:** RESULTS 21 currently omits 2004-12..2011-03 and carries a ~+3,500 bps regressor point; the beta split and the NEU alpha are artefacts.
2. **`src/xsec_backtest.py` → new read-only `--report-gaps` (or a script)** → list every censored `is_split`/`too_big` gap not in KNOWN_SPLITS with ticker/day/sign/quintile; print Q5, tilt, ON with those nights re-inserted; paste the existing `overnight exclusions:` line into RESULTS 15 → **reason:** the censoring count was never recorded and the long-only bias is direction-known. Do **not** change the mask on the frozen path (see H.3).
3. **`src/xsec_live.py` → `grade()`** → apply the identical adjacency/split/backstop mask as `load_panel` (or record both masked and unmasked tilt) → **reason:** the paper-log KILL row must be like-for-like with its calibration series.
4. **`src/xsec_ml.py` → `main()`** → add `on_12m` (and mean of on_1m/on_12m) as rule baselines on the same OOS rows; print ML − best rule with its own t; multi-seed canary (≥10 seeds per year) with year-clustered t → **reason:** "doubles the rule" is against the weakest feature; the canary "floor" is one draw.
5. **`src/xsec_ml.py` → walk-forward path** → honour `--through` for `build_table`/`portfolio` outputs and stamp the cutoff into `xsec_ml_daily.csv`; **`src/mc_risk.py`, `src/research_report.py`** → refuse/warn if the input series extends past `models/xsec_lgbm.json:last_tmonth` → **reason:** one routine re-run after `xsec_extend` would silently fold the holdout months into the thresholds.
6. **`src/xsec_extend.py` → `candidates()`** → restrict to rows with `src != 'yahoo'` (or a frozen HF pool + external liquidity screen); **`src/xsec_extract.py` → `one_month()`** → allow overwriting a `src=='yahoo'` month from HF; fix the header → **reason:** the pool is monotone-shrinking and the stated remedy does not exist in code.
7. **`src/orders.py` → open leg** → when no current-month basket exists or a held symbol is absent from it, liquidate it anyway; emit the basket daily/on the 1st rather than Mondays; warn when `capital/len(names) < $10,000` → **reason:** month-rollover leaves Aug-31 positions unsold for up to a week; rehearsal size sits under the per-order commission minimum.
8. **`src/overlay.py` → `mc_compare()`** → draw block starts once and index raw and scaled with the same `take`; report the paired ratio difference with a CI → **reason:** v2nb's PASS is inside bootstrap noise.
9. **`src/optbacktest.py` → rv construction (:78)** → use rolling std of `log(close/p60)` scaled by √(252/SF); fix header/printout; re-run the planted-truth check with overnight variance and record it → **reason:** mult 1.0 sells ~1.1-1.4× rich, so "directional-only" is false.
10. **`src/opra_value.py` / `optbacktest.py`** → add an explicit breach-day exit (next-day open on ±100 sh, or 15:55 buy-back cost) and print the partial-ITM count; **`src/stack_v2.py:39`** → price/era-dependent `QQQ_RT`; **`src/research_report.py`** → read cost parameters from the stack run, give OPT/v2o an explicit base, fix Sortino and the capacity k/label; **`src/xsec_backtest.py:31-33` + RESULTS.md:1034-1035** → correct the dividend direction text; add TVIX at the next re-freeze; wire `pytest tests -q` into `monthly.yml` and poison the raw panel before `load_panel`; add a DST regression test for `opra_pull.window_utc`/`opra_value.entry_chain`.

### H.2 Checks the user must run on the real panel (paste outputs into RESULTS)

1. Exclusion count and censored-gap list: `python src/xsec_backtest.py 2>&1 | grep -A14 'overnight exclusions'` (then the `--report-gaps` output from H.1 #2).
2. Delisting coverage + manifest: `python3 -c "import pandas as pd,glob;df=pd.concat(pd.read_parquet(f,columns=['ticker','month']) for f in glob.glob('data/xsec/*.parquet'));g=df.groupby('ticker').month.agg(['min','max']);print(g.loc[g.index.intersection(['LEH','BSC','WCOM','ENRN','WAMU','YHOO','TWTR','JDSU','SUNW','BBBY','SIVB','FRC','ATVI','SPLK'])]);print((g['max']<'2026-03').sum(),'names end before 2026-03 of',len(g))"` then `sha256sum data/xsec/*.parquet > reports/xsec_manifest.txt` and `pip freeze` into RESULTS.
3. Era tables and recent-era power: `python src/xsec_ml.py 2>&1 | grep -B2 -A8 'eras:'`; then `python3 -c "import pandas as pd,numpy as np,sys;sys.path.insert(0,'src');import mc_risk as m;ml=pd.read_csv('data/xsec_ml_daily.csv',dtype={'day':str}).set_index('day');bt=pd.read_csv('data/xsec_daily.csv',dtype={'day':str}).set_index('day');r=np.random.default_rng(7);s=ml.index>='20160101';m.verdict_table('replay 2016+',ml.mlon_ls[s].dropna().values,10000,r);m.verdict_table('tilt 2016+',(ml.mlon_q5-bt.qqq_on)[s].dropna().values,10000,r)"` — record 2016+ thresholds beside the full-history ones (do not replace them).
4. uni[T] conditioning, measured: `python src/xsec_extract.py --top 1000 --out data/xsec1000` (once; HF is stalled), then `python src/xsec_wide.py`, plus month-T Q5/Q1 overnight means for names in uni_core[T-1] \ uni_core[T]; report conditioned vs unconditioned Q5, tilt, L/S.
5. Yahoo seam: `python src/xsec_extend.py --check 2026-03` and `--check 2025-12`; and `python3 -c "import pandas as pd;a=pd.read_parquet('data/xsec/2026-03.parquet');b=pd.read_parquet('data/xsec/2026-04.parquet');j=a.groupby('ticker').dollar_vol.mean().to_frame('hf').join(b.groupby('ticker').dollar_vol.mean().rename('yh'),how='inner');print(len(j),(j.yh/j.hf).describe(percentiles=[.05,.25,.5,.75,.95]))"` — a name-specific ratio spread wider than ±10% means `dvol_chg` ranks are perturbed at the seam.
6. Full replay printout and dossier: `python src/xsec_replay.py` (paste both tag blocks: per-month bps/day + IC, Apr-Jun floor flagged as fabricated); `python src/research_report.py > reports/research_report_$(date +%F).txt` **after** H.1 #1 (section 6 inherits `factor.py`) and noting BASE_COST assumes c=1.0.
7. Dividends per leg (trailing yield of the registered Q5/Q1): `python3 -c "import yfinance as yf,pandas as pd;r=pd.read_csv('reports/xsec_paper.csv',dtype=str).iloc[-1];[print(q,sum(yf.Ticker(t.replace('.','-')).dividends['2025-08':'2026-08'].sum()/yf.Ticker(t.replace('.','-')).fast_info['last_price'] for t in r[q].split(';'))/len(r[q].split(';'))) for q in('q5','q1')]"` → bps/night = yield·1e4/252 with the correct signs.
8. Ops health: `gh run list --workflow monthly.yml --limit 5`, `gh run list --workflow roi.yml --limit 5`; if empty, `gh workflow run monthly.yml` and confirm a commit touching `reports/xsec_paper_daily.csv`; `python src/orders.py --leg open --capital 25000` dry-run before Sept 1 to inspect unsold positions. Until this works the 2026-08 clock has no paper-log data.
9. `python -m pytest tests -q` (currently never run in CI).
10. Month-clustered inference: report the t of `monthly_ic` (`src/xsec_ml.py:167-172`) over 279 months and a month-clustered t for tilt, NEU, OPT and MOM next to the iid daily t; add a one-line count of strategy families tried.

### H.3 What NOT to change

- **`models/xsec_on_lgbm.txt`, `xsec_on15_lgbm.txt`, `models/xsec_lgbm.json`**, `PARAMS/ROUNDS/FEATS` (`src/xsec_ml.py:60-66`), **REF 12.2/7.3** (`src/xsec_replay.py:39`), and the RESULTS 19 threshold tables — the frozen models were trained on the current `load_panel` cleaning; changing the split/backstop mask, the ETF list (SGOV/TVIX) or the universe rule on the scoring path would put the models on a different feature distribution than they were trained on, void REF and the GO/KILL calibration, and restart the pre-registration clock. Bundle all cleaning/universe changes into the next dated re-freeze, and keep the current binding clock running on the current construction.
- **The mechanical replay rule** (RESULTS.md:1585-1591) and its binding window (2026-08). It is internally consistent (calibration and forward metric share `build_table` + `portfolio`). Only the paper-log row needs a footnote or a live-style recalculation.
- **Do not subtract the canary from REF** — its no-leak expectation is ~0.
- **Do not re-run `xsec_ml.py` walk-forward over the extended panel** until H.1 #5 is in; it would overwrite `data/xsec_ml_daily.csv` with the holdout months.
- **Do not add leverage or fund MOM/OPT/v2o** on the current evidence; the record's own veto (3× ruin, 1.5× v2n dominated by 1× v2; MOM do-not-fund) stands.
- **Do not "fix" the Yahoo months already on disk by deleting them** without a manifest and a documented re-extract; the forward months' provenance must stay auditable.

---

## I. Actions taken (this commit) — FILE → FUNCTION → CHANGE → REASON

Everything below leaves the frozen scoring path byte-identical: `load_panel`
defaults, `build_table`, `PARAMS/FEATS`, `portfolio()`, `REF`, the models and
the RESULTS 19 threshold tables are unchanged (H.3).

| # | Finding | File → function | Change | Reason |
|---|---|---|---|---|
| 1 | B1#2 HIGH | `src/factor.py` → `factors()`, `show()` | Factors from `load_panel`'s masked `on_bps/cc_bps/id_bps`, QQQ+QQQQ stitched (`isin`, groupby-day-first); `show()` prints n, window and the OLS identity check | RESULTS 21 was computed on ~73% of days with the QQQQ era missing and a +3,500 bps seam regressor. **RESULTS 21 must be re-run** (`python src/factor.py`, `research_report.py` §6). |
| 2 | B1#1 HIGH | `src/xsec_backtest.py` → `load_panel(keep_backstop=False)`; new `src/xsec_gaps.py` | Measurement switch (default byte-identical) + a read-only tool listing every censored non-verified-split night and printing rule-B Q5/Q1/L/S/tilt with backstop nights re-inserted (a lower bound on the long-only bias) | The censoring count was never recorded and the bias is direction-known on the deployed long-only object; the frozen path is not touched. |
| 3 | B1#1/#12 | `src/xsec_live.py` → `_masked()`, `grade()` | Paper log gains `q5_on_masked` / `tilt_masked` columns (same split-band/backstop mask as the backtest) beside the raw night | The paper-log KILL row must be judged like-for-like with its calibration series. |
| 4 | B1#10, #11 | `src/xsec_ml.py` → `main()`, `walk_forward(seed_offset)` | `on_12m` and avg(1m,12m) rule baselines on identical OOS rows; "ML minus best rule" with daily and monthly t; `--canary-seeds` multi-seed permutation null (mean/sd/range) | "ML doubles the rule" compared the model with its weakest input; one permuted draw is not a null distribution. |
| 5 | B3 (`--through`) | `src/xsec_ml.py` → `main()`; `src/mc_risk.py` → `main()`; `src/research_report.py` → `main()` | `--through` now truncates the walk-forward path too and writes `xsec_ml_daily.meta.json`; `mc_risk` REFUSES a series past `models/xsec_lgbm.json:last_tmonth` (`--allow-forward` overrides, loudly); `research_report` warns | A routine re-run after `xsec_extend` would have folded the holdout months into the pre-registered thresholds. |
| 6 | B1#13 | `src/overlay.py` → `block_index()`, `mc_row(take)`, `mc_compare()` | Common random numbers: block starts drawn once, both series indexed with the same draw; paired per-path CAGR/\|maxDD\| difference with p5/p50/p95 and P(scaled better) printed | The PASS/FAIL carried Monte-Carlo noise (v2nb passed at print precision). Criterion itself unchanged. |
| 7 | B1#8 | `src/optbacktest.py` → `model_series(df=None)`, header | "Fair" IV = trailing vol of the 10:30→close window's own return, annualised by its own length; test injection via `df` | The old cc-based vol already sold ~1.1–1.4× rich at mult 1.0 while claiming zero variance premium. Re-validated: condor at mult 1.0 ≈ 0 (t −2.2 on 3k days), mult 1.3 → +5.0. |
| 8 | B1#14, B3 | `src/research_report.py` → `sec_capacity()`, `metrics()`, `base_cost(c)`, `--c` | Basket k from the registered basket; true median and p25 ADV; opening-cross share binds; Sortino as downside deviation (LPM2); BASE_COST derived from `--c`; OPT/v2o get explicit bases (no infinite break-even) | p25 was labelled median, k=30 was never held, v2o auto-passed SURVIVE. |
| 9 | Critic #4, B1#5 | `src/orders.py` → `submit_paper()`, `main()` | Open leg liquidates every held STK position, basket or not, and no longer aborts on a missing month registration; warns when `capital/len(names) < $10k` | Month-rollover hole (Aug-31 buys unsold for a week); per-order minimums exceed the modelled crossing at rehearsal size. |
| 10 | B1#4 | `src/xsec_extend.py` → `candidates()`; `src/xsec_extract.py` → `one_month()`, `main()` | Candidate pool drawn only from HF-sourced months (loud fallback); HF re-extraction overwrites `src=='yahoo'` months; header corrected | The pool was monotone-shrinking and the header's "until HF catches up" remedy did not exist in code. |
| 11 | B3 (dividends) | `src/xsec_backtest.py` header | Direction corrected: long leg understated, short leg overstated, L/S ≈ neutral-to-optimistic | The note (and RESULTS 15) had the short leg's sign wrong. |
| 12 | Critic #11, B3 | `tests/test_audit_regressions.py`; `.github/workflows/monthly.yml`; `.github/workflows/roi.yml` | Tests: factor stitching; poison the RAW future parquets before `load_panel` (targets included); DST for `opra_pull.window_utc`; CRN; pytest step in CI; daily cron so the basket registers on the 1st; VM crontab gets a daily `xsec_live` line | Tests were never run in CI and the existing canary poisoned after load. |

**Deliberately NOT changed (deferred to the next dated re-freeze, per H.3):** the split/backstop mask on the scoring path; the ETF list (SGOV, TVIX); the universe rule; `REF`; the GO/KILL tables; era-varying `QQQ_RT` (B1#7); explicit assignment modelling in `opra_value` (B1#9 — the caveat stands in RESULTS 18); the legacy §1–5 scripts.

**Checks only the real panel can answer (H.2, unchanged list):** the censoring block (`python src/xsec_gaps.py`), the re-run of `factor.py` (RESULTS 21 is void until then), the ML-vs-on_12m block (`python src/xsec_ml.py`), the exclusion counts, 2016+ GO/KILL thresholds, the delisting-coverage probe, the Yahoo seam (`xsec_extend.py --check`), and ops health (`gh run list`) — the paper log had zero graded nights at HEAD.

## J. Completeness critic (verbatim)

GAPS (prioritized; container is synthetic, so every CMD below is for the user's REAL checkout unless noted)

1. [HIGH] Fix-interaction: the upheld split/backstop fix (whitelist true splits, keep real gaps) would silently change the training targets and the on_1m/on_12m/vol_3m features (src/xsec_backtest.py:155-157,165-166 -> src/xsec_ml.py:78-84,102-105,113-115), so the frozen models (models/xsec_lgbm.json frozen 2026-08-20) would be scored on a different cleaning than they were trained on, REF 12.2/7.3 (src/xsec_replay.py:39) and the GO/KILL tables (src/mc_risk.py:128-136) would be void, and the pre-registration clock (RESULTS.md:1578-1596) would restart. Secondary hazard: a yfinance-based whitelist has no split data for delisted names, so any missed 2:1 on a dead Q5 name re-enters as a -6,931 bps "night" (one such night = -690 bps on a 10-name basket, far worse than the censoring it replaces); also a 5:4 split (ratio 1.25) is below the |on|>log1.25 trigger at :151 and is already uncaught. DO: do not touch load_panel for the frozen path; add a read-only `--report-gaps` mode that prints every censored non-KNOWN_SPLITS gap with ticker/day/sign/quintile and the Q5/tilt series with those nights re-inserted; keep the ratio band as fallback for names without external split data; move any cleaning change to the next dated re-freeze together with SGOV/TVIX. CMD (real panel, prints the never-recorded exclusion count): `python src/xsec_backtest.py 2>&1 | grep -A14 'overnight exclusions'` and paste into RESULTS 15.

2. [HIGH] Unverified premise "dead/delisted names are retained for the months they were alive" (src/xsec_extract.py:8-12; accepted by the survivorship hunter on the header alone). The HF URL is unpinned `resolve/main` (src/xsec_extract.py:27-28), so the dataset's own delisting coverage and any upstream revision are untested; monthly.yml re-bootstraps from HF on cache eviction (monthly.yml:45-52) with no panel hash. CMD: `python3 -c "import pandas as pd,glob;df=pd.concat(pd.read_parquet(f,columns=['ticker','month']) for f in glob.glob('data/xsec/*.parquet'));g=df.groupby('ticker').month.agg(['min','max']);print(g.loc[g.index.intersection(['LEH','BSC','WCOM','ENRN','WAMU','YHOO','TWTR','JDSU','SUNW','BBBY','SIVB','FRC','ATVI','SPLK'])]);print((g['max']<'2026-03').sum(),'names end before 2026-03 of',len(g))"` then `sha256sum data/xsec/*.parquet > reports/xsec_manifest.txt` and record the manifest + `pip freeze` (pandas/numpy/lightgbm versions; requirements.txt pins only lower bounds, monthly.yml:51 installs unpinned).

3. [HIGH] GO/KILL power (RESULTS.md:1558-1571) is bootstrapped from the full 2003-2026 series, including the 2003-2011 era where the rule printed +15.6/+9.5 vs -0.4 in 2016-19 (RESULTS.md:1070); the ceiling-ML era table is printed by src/xsec_ml.py:242 but never pasted (RESULTS.md:1118 says only "positive in all six eras"). P(kill|real) and P(pass|real) for the forward window depend on the RECENT mean, not the 23-year mean. CMD: `python src/xsec_ml.py 2>&1 | grep -B2 -A8 'eras:'` (paste all three era tables) and `python3 -c "import pandas as pd,numpy as np,sys;sys.path.insert(0,'src');import mc_risk as m;ml=pd.read_csv('data/xsec_ml_daily.csv',dtype={'day':str}).set_index('day');bt=pd.read_csv('data/xsec_daily.csv',dtype={'day':str}).set_index('day');r=np.random.default_rng(7);s=ml.index>='20160101';m.verdict_table('replay 2016+',ml.mlon_ls[s].dropna().values,10000,r);m.verdict_table('tilt 2016+',(ml.mlon_q5-bt.qqq_on)[s].dropna().values,10000,r)"` and record the 2016+ thresholds beside the full-history ones.

4. [HIGH] Ops month-rollover hole (no finding raised): src/orders.py:60-64 raises for any month without a pre-registered basket; the basket is emitted only by the Monday cron (monthly.yml:30; Sept 2026: first run Sept 7), so on Sept 1 the 09:20 open leg (roi.yml crontab) aborts and the Aug 31 MOC positions are held unhedged for a week; the open leg then sells only names in the CURRENT basket (src/orders.py:133-138), so August names absent from September's basket are never sold, and src/fills.py:86-91 later pairs any eventual sell to the Aug 31 buy as one "night". DO: make the open leg liquidate every long position when no current-month basket exists or the position is not in the basket; run xsec_live emission daily (or on the 1st) instead of Mondays; add a rollover test. CMD (verify paper account state before Sept 1): `python src/orders.py --leg open --capital 25000` (dry run; check the "skipped (no position)" lines vs `ib.positions()`).

5. [HIGH] Ops-health: reports/xsec_paper_daily.csv is header-only although the basket was emitted 2026-08-21, HEAD is 2026-08-30 (git log), and the Aug 24 cron should have graded >=1 night (src/xsec_live.py:126 `D >= emitted_on`, `D < today`); reports/ last changed 2026-08-22 (0f79414). Either monthly.yml never fired (cron fires only from the DEFAULT branch, monthly.yml:24) or the push failed silently. CMD: `gh run list --workflow monthly.yml --limit 5` and `gh run list --workflow roi.yml --limit 5`; if empty, `gh workflow run monthly.yml` and confirm a commit touching reports/xsec_paper_daily.csv appears. Until this works, the "binding window starts 2026-08" clock (RESULTS.md:1578-1580) has no paper-log data behind it.

6. [MEDIUM] RESULTS claims still lacking recorded output: (a) day-concentration/fragility for ON/NEU/v2 (src/research_report.py:110-113; only §8b's QQQ study exists, RESULTS.md:636-637); (b) cost break-even sweep and SURVIVE verdict (research_report.py:120-132,186-196); (c) replay per-month bps/day + IC for 2026-04..07 and the on15 floor for Apr-Jun (printed by src/xsec_replay.py:100-103, only July recorded); (d) `xsec_extend.py --check` (src/xsec_extend.py:135-156) never run on record. CMD: `python src/research_report.py > reports/research_report_$(date +%F).txt` (note section 6 inherits the factor.py bug until fixed, and BASE_COST at :45-46 assumes stack_daily.csv was written at c=1.0), `python src/xsec_replay.py` (paste both tag blocks), `python src/xsec_extend.py --check 2026-03` and `--check 2025-12` (paste overlap %, |open|/|close| diff medians/p95, "look split-adjusted" count).

7. [MEDIUM] Fix-interaction on the factor finding: after stitching QQQ/QQQQ and using masked on_bps in src/factor.py:80-88, RESULTS 21 will change, and with it the unresolved 3x disagreement on "pure selection alpha" (NEUb +3.40 t 4.85 at RESULTS.md:1621-1628 vs NEU alpha +1.13 t_HAC 1.47 at :1736) and the "half of v2 is beta" narrative (:1751-1757) that motivated the NEUb/vol-scaling menu (RESULTS.md:1670-1675). DO: after the fix, print n and the date window in show() (factor.py:103-106), re-run and check the OLS identity mean = alpha + sum(beta*mean F) for every book (currently fails for MOM/NEU/v2n per the decomposition), and re-state which "selection alpha" the record adopts. research_report.py:226 shares the same factors() and must be re-run with it.

8. [MEDIUM] Never-examined: the frozen model scored live on Yahoo-built T-1 features. For the 2026-08 basket (reports/xsec_paper.csv universe_month 2026-07) dvol/lprice/ret_1m/on_1m come from a Yahoo month (consolidated Volume*Close, src/xsec_extend.py:97,121) while dvol_chg's T-12..T-2 baseline is HF RTH dollar volume (src/xsec_ml.py:109-111); from 2027-04 the eligible pool is a closed cohort (upheld finding) and no genuinely new entrant can be scored. No RESULTS number exists for the live-vs-backtest feature drift. CMD: `python3 -c "import pandas as pd;a=pd.read_parquet('data/xsec/2026-03.parquet');b=pd.read_parquet('data/xsec/2026-04.parquet');j=a.groupby('ticker').dollar_vol.mean().to_frame('hf').join(b.groupby('ticker').dollar_vol.mean().rename('yh'),how='inner');r=(j.yh/j.hf);print(len(j),r.describe(percentiles=[.05,.25,.5,.75,.95]))"` -- a name-specific spread of the ratio wider than ~+/-10% means dvol_chg ranks are perturbed at the seam; record it, and add `src != 'yahoo'` filtering to candidates() (xsec_extend.py:44-49) before 2027-04.

9. [MEDIUM] Never-examined bias class: uni[T] conditioning was bounded by argument only; the wide panel that measures it (src/xsec_wide.py:13, `python src/xsec_extract.py --top 1000 --out data/xsec1000`) has never been built on record. HF is stalled at 2026-03 so the history is extractable once. CMD: run that extraction (heavy, resumable), then `python src/xsec_wide.py`, and additionally compute month-T Q5/Q1 overnight means for names in uni_core[T-1] but not uni_core[T] (rows present in xsec1000). Report the delta vs the conditioned Q5, tilt and L/S; this also settles the RESULTS.md:1814 "survivorship-free" wording.

10. [MEDIUM] Never-examined accounting class: dividends. Both the code (src/xsec_backtest.py:31-33) and RESULTS.md:1034-1035 state the wrong direction for the short leg, and no number exists for either leg. The Q5/Q1 yield differential decides the sign of the L/S bias and the size of the long-only understatement. CMD (Yahoo, one recent month, read-only): `python3 -c "import yfinance as yf,pandas as pd;r=pd.read_csv('reports/xsec_paper.csv',dtype=str).iloc[-1];[print(q,sum(yf.Ticker(t.replace('.','-')).dividends['2025-08':'2026-08'].sum()/yf.Ticker(t.replace('.','-')).fast_info['last_price'] for t in r[q].split(';'))/len(r[q].split(';'))) for q in('q5','q1')]"` -> trailing yield per leg; then bps/night = yield*1e4/252 per leg, applied with the correct signs.

11. [LOW] No test runs anywhere: tests/test_no_lookahead.py is not invoked by monthly.yml (no pytest step, monthly.yml:51-60) or roi.yml, and it poisons prices after load_panel (LOW finding). DO: add `python -m pytest tests -q` before the "monthly ritual" step; add a poison-before-load_panel case and a membership-perturbation case; add a DST regression test for src/opra_pull.window_utc and src/opra_value.entry_chain (2024-01-15 -> 15:30Z, 2024-07-15 -> 14:30Z). CMD now: `python -m pytest tests -q`.

12. [LOW] Never-examined program-level multiple testing and inference: the surviving books are the winners of >=8 strategy families across RESULTS 1-22 (market-making, QQQ momentum sign/ML, calendar, cc/on/on15 ML, XID, 0DTE long/spread/condor, IV gate), and every t in RESULTS except §21 is the iid daily t of src/xsec_backtest.py:110-111 with no HAC or month-clustering, while formation is monthly. The ML L/S t=12.6 survives any correction, but the tilt (t 8.9), NEU (t 5.2), OPT (t 4.65 on 749 days) and MOM (t 2.5) should be re-stated with a month-clustered t (monthly_ic at src/xsec_ml.py:167-172 already exists; report its t over 279 months) and a note on the number of families tried.


---

## K. Verification run — which findings the real panel closed

The six commands §H asked for were run on the full panel (329 months, 1,026,770
ticker-days, 19990301 → 20260731). Full numbers: **RESULTS §24**, with the
re-runs folded into §15, §15b and §21. Status of the open findings:

| finding | status | measured |
|---|---|---|
| B1 #1 — censoring flatters the long-only book | **CLOSED, upheld** | Q5 and tilt **−1.52 bps/day, t −3.0** (887 censored nights; 660 backstop, 502 negative; LEH −267%, BSC −224% among them). Lower bound: band nights not re-inserted. |
| B1 #2 — `factor.py` QQQ-only, §21 void | **CLOSED, upheld; direction reversed** | Rebuilt table reconciles the OLS identity on every row. ON +4.13 (t_HAC 5.72), NEU **+3.79 (t 5.25)**, v2 **+6.13 (t 4.51)**, v2n +5.79 (t 4.26), MOM +3.68 (t 2.64). Beta sum 1.42 (ON, v2) and 0.42 (NEU, v2n); split +0.69/+0.73, not +0.20/+1.19. |
| B1 #10 — "ML doubles the rule" (contested) | **CLOSED, upheld** | ML +12.42 vs `on_12m` +11.44 → **+0.99 bps/day, daily t 1.31, monthly t 1.24**. Floor: +7.50 vs +6.57 → +0.94, t 0.83. Not significant on either target. |
| B1 #11 — one canary draw is not a null | **CLOSED; the record's reading vindicated** | 5-seed permutation null: mean −0.29, sd 1.21, **range [−1.42, +1.52]**. The historical +1.6–1.8 canary is one draw from that band, as §15b argued — and the ML's +0.99 increment sits inside it. |
| §G — NEU "size disputed +3.4 vs +1.1" | **DISPUTE CLOSED at +3.79 (t 5.25)** | Agrees with the independently measured NEUb +3.40 (t 4.85). The +1.13/t 1.47 was the bug. |
| §G — MOM "promising but insufficient" | **Deployment verdict now mechanical: KILL** | Sharpe 0.49 (gate 0.5); **−1.2 bps/day** with its best 1% of days removed. Break-even 12.0× is the only leg it passes. |
| §H #11 — no test runs anywhere | **CLOSED** | `pytest` step added to `monthly.yml`; `python -m pytest tests -q` → 6 passed on the user's machine, 7 with the shadowing test added since. |
| §H #8 — the Yahoo seam was never measured | **CLOSED, partially reassuring** | 2026-03: overlap **132/150 (88%)**, \|close\| median 1.5 bps / p95 19.3, \|p60\| median 1.1 / p95 8.4; 6 split-adjusted names, 6 failed downloads (renames + delistings — the survivorship direction). Below the script's own ≥90% bar. |

**Also closed since: §B #5 (HIGH, ops-health).** The paper log is no longer
header-only — the monthly workflow fires daily and has committed nine graded
nights (20260821..20260903, 2026-08 binding). The clock in RESULTS 19 now has
data behind it; the horizon is still under one month, so there is no verdict.

One defect the verification run found in the audit batch itself: `xsec_ml.py`
rebound its argparse namespace and crashed writing the `.meta.json` sidecar
**after** the full walk-forward completed. Fixed, with an AST regression test
that fails if any script in `src/` rebinds the name it took from `parse_args()`.

Still open, unchanged: #8's live-vs-backtest feature drift at the seam, #9's
wide-panel `uni[T]` conditioning measurement (needs the top-1000 extraction),
#10's dividend numbers, and #12's program-level multiple-testing restatement.
