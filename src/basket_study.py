# Part of qqq-microstructure.
#
# Would a BROADER and/or THEME-CAPPED basket give a better hedged book, after
# realistic costs at the $500k target, than the one traded today? Research
# only: nothing here changes the traded book, the frozen models, the frozen
# scoring path (xsec_ml / xsec_backtest / xsec_replay / xsec_live), orders.py,
# guard.py or reports/*.csv. A winner is at most a PROPOSAL for a dated
# re-freeze, decided by the owner after the November forward read (RESULTS
# 19); it binds nothing before then.
#
# Why it is asked. The traded hedged book (RESULTS 29: long the frozen
# model's Q5 overnight, ~11 names of the lagged top-150, short QQQ 1:1) has
# been one theme eleven times live (memory, storage, semis, AI
# infrastructure): overnight beta ~2.4 to QQQ and a tilt sd of ~151
# bps/night against the backtest's ~66 (RESULTS 19, 29). The wide panel
# (data/xsec1000, RESULTS 30) showed the overnight edge holds far beyond the
# top 150. The question is whether breadth and/or a cap per theme buys a
# similar net mean at much lower risk, or a higher risk-adjusted return.
#
# Everything from this header down to the rule that opens "THE STUDY --
# machinery" was written before any return of any basket in this study was
# computed, and is not edited after. Its sha256 is printed by every run and
# checked against the pin below the rule (as event_test.py does).
#
# ============================================================================
# DECLARATIONS (pre-registered 2026-10-14)
# ============================================================================
#
# DATA. The wide panel data/xsec1000 (top 1000 US tickers by each month's own
# dollar volume, 1999-03 .. 2026-09; 2026-04 .. 09 rebuilt from Yahoo by
# xsec_extend_wide.py), cleaned once by xsec_backtest.load_panel with its
# DEFAULT mask (test symbols dropped; overnight NaN across missing days,
# split-ratio gaps and the +-40% backstop). on_bps is the overnight log
# return in bps; cc_bps the close-to-close one.
#
# FORMATION, every trade month T, exactly as xsec_live forms the live basket:
#   Universe U_N(T): the names whose rank by month T-1's total dollar volume
#     within the wide panel (xsec_wide.monthly_rank) is <= N, minus the
#     frozen ETF list (xsec_backtest.ETF). No month-T information: neither
#     month-T membership nor month-T volume enters.
#   The universe's own panel: the wide-panel rows (month m, name) with
#     month-m rank <= N -- what an extraction of the top N would hold. Its
#     month aggregates and features are xsec_live.month_aggs and
#     xsec_live.features over months T-12 .. T-1 (build_table's construction;
#     xsec_live's parity check holds them equal to build_table's), with
#     features()'s own eligibility. A month with fewer than MIN_NAMES (40)
#     scored names is skipped. The ten features are ranked across the
#     scored names of U_N(T) with rank_pm (within the universe).
#   Scores, two families on the SAME scored names:
#     (a) MODEL: the frozen booster models/xsec_on_lgbm.txt, loaded and never
#         refitted, predicting on the ranked features -- the traded score;
#     (b) RULE:  on_12m, xsec_replicate.signal_rows' signal: the mean of the
#         monthly overnight means over months T-12 .. T-1 with >= 8 valid
#         nights, at least 8 such months (features()'s raw on_12m, before
#         ranking). signal_rows' own looser eligibility is NOT used, so the
#         model and the rule choose from identical names.
#   Ordering: score descending, ties by ticker ascending (xsec_live's
#     nlargest over the sorted universe).
#   Basket N: the first N names of the ordering (all of them if fewer).
#   Theme cap C:
#     none  -- no cap.
#     cap3  -- at most 3 names per cluster. Clusters are formed at T's
#              formation over the scored names of U_N(T), from the 63 trading
#              days of the wide panel's calendar ending on T-1's last trading
#              day (strictly before T), on each name's cc_bps as it stands in
#              the wide panel (masked days missing). Pearson correlation on
#              pairwise-complete days; a pair with fewer than 40 common valid
#              days has correlation 0. Distance 1 - correlation, average
#              linkage (scipy), flat clusters cut at distance 0.4 (merge iff
#              average correlation >= 0.6). Walk down the ordering and admit
#              a name unless its cluster already holds 3 admitted names; stop
#              at N. If the scored names run out first, the basket holds
#              fewer than N (reported as names/night).
#
# HOLDING. The basket of T is held on every night D whose close D is a
#   trading day of month T in the panel calendar and whose next trading day
#   d exists: buy at D's close (MOC), sell at d's open (MOO). A name's night
#   outcome is on_bps of its row at d (default mask). Against it, a 1:1
#   dollar QQQ short over the same night: QQQ's rows, QQQQ's from 2004-12 to
#   2011-03, one row per day (the panel's stitch). A night with no valid
#   QQQ outcome is dropped for every configuration.
#
# GRID, and nothing else:
#   universe U in {top150 (current), top300, top500, top1000}
#   basket   N in {11 (current), 20, 40}
#   cap      C in {none, cap3}
#   24 configurations x 2 score families. The CURRENT book is (top150, 11,
#   none, MODEL); the RULE family's baseline is (top150, 11, none, RULE).
#
# SIZING AND COSTS at the $500,000 target (RESULTS 29), each night:
#   A basket of n_b names gets per = C / n_b dollars a name (orders.size). A
#   name is HELD iff it has a row at D (a close) and q = floor(per / close_D)
#   >= 1; otherwise its slice stays in cash. long_D = sum q x close_D over
#   held names; dep_D = long_D / C. The short is orders.hedge_shares(long_D,
#   QQQ close_D) whole shares (round to nearest); S_D = shares x QQQ close_D.
#   cost_D = for every held name, paper_bench.order_cost('tiered', q,
#   close_D, 'BUY') + order_cost('tiered', q, sell price, 'SELL') -- the sell
#   price is d's open when the outcome is valid, else close_D -- plus
#   order_cost('tiered', shares, QQQ close_D, 'SELL') + order_cost('tiered',
#   shares, QQQ open_d, 'BUY'): one BUY MOC + one SELL MOO per name and the
#   QQQ SELL MOC / BUY OPG pair. Fee day '20270104' for every night: the
#   FINRA TAF on, capital.py R1's 2027 schedule. Today's IBKR Pro Tiered
#   schedule is applied to all history (a today's-cost overlay).
#   gross_D, bps of C = dep_D x (equal-weight mean on_bps over held names
#     with a valid outcome) - (S_D / C) x QQQ on_bps. A night on which no
#     held name has a valid outcome is dropped for that configuration (the
#     backtest's dropna).
#   cens_D = 1.5 x dep_D bps of C: RESULTS 15's measured lower bound on the
#     optimism of every long-only Q5 / tilt night under the backstop mask.
#     The hedged book's gross IS a tilt and its QQQ leg is never censored, so
#     it applies on every night the long basket is held, per dollar of long.
#   net_D = gross_D - cost_D / C x 1e4 - cens_D.        Gross and net are
#   both reported; net decides.
#   INFORMATION (decides nothing): (i) measured censoring -- gross recomputed
#     with the backstop nights put back in (load_panel(keep_backstop=True)'s
#     outcome; split-band nights stay out, as xsec_gaps.py), and net_meas =
#     that gross - cost, i.e. the flat 1.5 replaced by the configuration's
#     own measured censoring; (ii) unpriced share: basket name-nights with no
#     row at D or no row at d in the wide panel; (iii) the per-name slice as
#     a share of the name's day-D dollar volume.
#
# METRICS, per configuration and window, on the net nightly series unless
#   labelled: net mean bps/night; sd; net Sharpe = mean / sd x sqrt(252) on
#   the NIGHTLY series; t = mean / (sd / sqrt(nights)); worst night; p1 night
#   (1st percentile, linear interpolation); max drawdown at fixed notional
#   (cumulative sum of net bps of C from 0 at the window's start, worst fall
#   from a running peak, in % of C); beta to QQQ overnight (OLS slope of net
#   on QQQ on_bps, with intercept); average pairwise correlation inside the
#   basket (the mean off-diagonal entry of the formation correlation matrix
#   above among the basket's names, nights-weighted); names/night (held
#   names with a valid outcome); cost bps/night; turnover (share of the
#   basket replaced at each monthly formation, 1 - |B_T & B_T-1| / |B_T|,
#   nights-weighted); gross mean and cens mean beside them.
#
# WINDOWS, by night D: SELECTION 2001-01 .. 2015-12; CONFIRMATION 2016-01 ..
#   2026-09; RECENT 2024-01 .. 2026-09 (the live-like regime, information);
#   FORWARD 2026-04 .. 2026-09 (after the frozen model's 2026-03 cutoff,
#   information).
#
# SELECTION RULE, mechanical; the MODEL configurations decide:
#   S1 In SELECTION, the candidates are the MODEL configurations whose net
#      mean >= 0.80 x the CURRENT book's net mean AND whose sd <= 0.85 x the
#      CURRENT's sd. (If the CURRENT's SELECTION net mean is <= 0, the mean
#      condition reads net mean >= the CURRENT's.)
#   S2 Among the candidates pick the highest SELECTION net Sharpe; ties at
#      two decimals go to the smaller max drawdown. No candidate: "the
#      current basket stands".
#   S3 The pick is PROPOSED only if, in CONFIRMATION, its net Sharpe >= the
#      CURRENT's AND its max drawdown is no worse (no deeper) AND its p1
#      night is no worse (no lower) than the CURRENT's. Otherwise "the
#      current basket stands".
#   The same S1-S3 applied to the RULE family against its own baseline is
#   reported as the model-free check; it decides nothing. The best
#   configuration by net Sharpe in each window is reported (not decided), so
#   the selection effect is visible.
#
# WHAT THE EVIDENCE CAN AND CANNOT SAY, stated before the run:
#   - The frozen model was trained through trade month 2026-03 on the
#     top-150 core names. For most of history the MODEL family is in-sample
#     on the core names and out-of-sample on names never in the top 150 --
#     which tilts every MODEL comparison toward the narrow CURRENT book. The
#     RULE family and the FORWARD months are the cleaner evidence for the
#     breadth and cap effects.
#   - Exit conditioning: a name is priced at D and d only while it is in the
#     wide panel's top 1000 for those months; a lagged top-1000 name that
#     drops out is unpriced (counted, item ii), which flatters the widest
#     universes most.
#   - Costs are commissions and fees only: no auction impact, no spread. The
#     panel's open/close are first/last minute-bar trades (Yahoo daily bars
#     from 2026-04), shown to be the official auction prints only for the
#     most frequent core names (RESULTS 15b). Item iii sizes the impact risk.
#   - Dividends: the basket receives them, the short owes QQQ's; neither is
#     in the unadjusted overnight returns. QQQ borrow is not charged.
#
#   python src/basket_study.py [--dir data/xsec1000] [--cache DIR]
# ============================================================================


# ============================================================================
# THE STUDY -- machinery
# ============================================================================

import os, sys, glob, time, argparse, hashlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

# sha256 of every line above the rule that opens "THE STUDY -- machinery",
# printed at 2026-10-14 15:55Z before any basket return was computed.
PREREG_SHA256 = \
    '561af434b07b377072f096154fbe9b2f9d73fe5aa61d705f11d7d87ebb0916a4'

MODEL = os.path.join(ROOT, 'models', 'xsec_on_lgbm.txt')
CAPITAL = 500_000.0
FEE_DAY = '20270104'          # FINRA TAF on: capital.py R1's 2027 schedule
PLAN = 'tiered'
CENS = 1.5                    # RESULTS 15, bps per dollar of long, per night
UNIVERSES = (('top150', 150), ('top300', 300), ('top500', 500),
             ('top1000', 1000))
NS = (11, 20, 40)
CAPS = ('none', 'cap3')
FAMILIES = ('MODEL', 'RULE')
CURRENT = ('top150', 11, 'none', 'MODEL')
RULE_BASE = ('top150', 11, 'none', 'RULE')
CAP_K = 3
CORR_DAYS = 63
CORR_MIN = 40
CORR_CUT = 0.6
WINDOWS = (('SELECTION', '20010101', '20151231'),
           ('CONFIRMATION', '20160101', '20260930'),
           ('RECENT', '20240101', '20260930'),
           ('FORWARD', '20260401', '20260930'))
S_MEAN, S_SD = 0.80, 0.85     # S1
CACHE_VERSION = 1

# ---------------------------------------------------------------------------
# POST-HOC (added 2026-10-14 AFTER the declared run, from an adversarial
# review; NOT evidence). Nothing here changes the declared run: its tables
# and verdict are computed exactly as before. These constants feed the
# labelled POST-HOC variants printed after the declared verdict.
# ---------------------------------------------------------------------------
# IBKR Pro Tiered US stock commission by MONTHLY cumulative share volume (all
# US stock and ETF shares, both sides), applied at the marginal rate as the
# month accumulates: <=300k $0.0035, 300k-3M $0.0020, 3M-20M $0.0015,
# 20M-100M $0.0010, >100M $0.0005; the $0.35 minimum and the 1% cap stay per
# order. paper_bench.FEES['tiered'] carries only the first band (its own
# comment: "<=300k sh/month"), which every declared number uses.
TIER_BANDS = ((300_000, 0.0035), (3_000_000, 0.0020), (20_000_000, 0.0015),
              (100_000_000, 0.0010), (float('inf'), 0.0005))

# Exchange-traded products (ETFs, ETNs, commodity trusts and pools, leveraged
# and inverse funds) that the frozen top-150 list (xsec_backtest.ETF) misses
# below rank 150. Built after the declared run by: (1) a return signature on
# the wide panel -- a name whose trailing-63-day close-to-close correlation
# with another panel name is >= 0.95 in absolute value, or whose daily sd is
# < 30 bps, in >= 30% of its months -- 364 tickers, each classified by hand
# (dual share classes, merger pairs, ADR pairs and preferreds kept as
# stocks); (2) every basket ticker of the declared and screened runs checked
# against SEC company_tickers_mf.json and the EDGAR (ticker, year) -> entity
# map on disk, and by hand; (3) the reviewer's pattern list. Tickers whose
# symbol belonged to a company before the product existed are screened only
# from ETP_FROM's month (Tandy was TAN, Seagram VO, Ethan Allen ETH, Triton
# Energy OIL, Apex Silver SIL, Litton LIT, Spyglass SPYG); DRIV (Digital
# River) and HERO (Hercules Offshore) are never products while in the panel.
ETP_SCREEN = frozenset('''
AAXJ ACWI ACWX AGQ AMJ AMLP ANGL AVEM BAI BAL BBEU BBJP BGU BGZ BIB BIS BITQ
BIV BKLN BLOK BMNU BNDX BNKU BNO BOTZ BOXX BRZU BSV BTC BULZ CALF CGDV CGGR
CHAU CIBR CLOU CONY COPX CORN CPER CSJ CURE CYB DBA DBB DBE DBEF DBEU DBJP
DBO DBP DBS DBV DEM DFEN DGAZ DGL DGLD DGP DGRO DIG DJP DOG DPST DRIP DRN
DRV DSLV DTO DUG DVY DWT DWTI DXGE DXJ DXO DYNF DZZ ECH EDC EDV EDZ EEMV EEV
EFAV EFG EFU EFV EFZ EIDO EIS EMLC EMXC EPI EPOL EPP EPU ERUS ESGE ESGU ESPO
ETH ETHD ETHU EUM EUO EURL EWD EWI EWK EWL EWN EWO EWP EWQ EWS EWW EZA EZU
FALN FBND FDN FETH FEZ FINX FLOT FNDX FNGA FNGD FNGO FNGS FNGU FTSM FV FVD
FXA FXB FXC FXF FXP GAZ GDXD GDXU GLDM GLL GLTR GSG GUSH GXC HACK HDV HEDJ
HEFA HEZU HIBL HYLB IAI IAT IAUM ICF ICLN IDEV IEFA IEI IEMG IEO IEUR IEV
IGE IGIB IGSB IGV IHI IJJ IJK IJS IJT ILF INDL IONZ ITA ITOT IUSB IUSG IVE
IVW IWB IWD IWF IWN IWO IWP IWR IWS IWV IXUS IYE IYF IYG IYH IYM IYT IYW IYZ
JAAA JEPI JEPQ JETS JJA JJC JJG JO JPST KBWB KCE KIE KORU LIT MAGS MBB MEXX
MGK MIDU MINT MOAT MTUM MUB MUU MVV MZZ NAIL NEAR NIB NOBL NRGD NRGU NVD
NVDQ NVDX NVDY OEF OIL OILD OILU PALL PBW PCY PFF PGX PHYS PILL PIN PLTY
PPLT PSCE PSLV PTIR QCLN QQQI QQQM QUAL QYLD REMX RETL REW RKH ROBO ROM RUSL
RUSS RWM RWR RYLD SAA SARK SCHA SCHB SCHD SCHE SCHF SCHG SCHH SCHO SCHP SCHR
SCHX SCHZ SCZ SDD SDY SGG SGOL SGOV SHYG SIL SILJ SIVR SJNK SKF SKYY SMCX
SMDD SMIN SMN SMST SNXX SOYB SPAB SPDN SPDW SPEM SPHB SPHD SPHY SPIB SPLB
SPLG SPLV SPMD SPMO SPSB SPSM SPTI SPTL SPTS SPYD SPYG SPYI SPYM SPYV SRLN
SRS SSG STIP SVOL TAN TBF TECL TECS TFI TFLO THD TLH TPOR TSDD TSLG TSLR
TSLT TSLY TVIX TVIZ TYD TYH TYO UDN UGA UGAZ UGL UGLD ULE ULTY UNL URA URE
USCI USD USFR USHY USL USLV USMV UTSL UWT UWTI UYG UYM VAW VB VBIL VBR VCIT
VCLT VCR VCSH VDC VDE VEA VEU VFH VGIT VGK VGLT VGSH VGT VHT VIG VIIX VIS
VLUE VMAX VMBS VMIN VNM VO VONG VOX VPL VPU VT VTEB VTIP VTV VTWO VUG VV VWO
VXUS VXXB VYM WCLD WEAT WEBL XAR XES XLG XSD XTN XVZ XYLD YCS YMAX ZSL
'''.split())
ETP_FROM = {'TAN': '2008-04', 'VO': '2004-01', 'ETH': '2024-07',
            'SPYG': '2005-01', 'LIT': '2010-07', 'EFU': '2007-07',
            'OIL': '2006-08', 'BNKU': '2019-04', 'SCZ': '2007-12',
            'MAGS': '2023-04', 'SIL': '2010-04'}
MODEL_CUTOFF = '20260331'     # the frozen model's last training trade month
FWD_FIRST = '2026-04'         # first wide-panel month rebuilt from Yahoo
PANEL_COLS = ['ticker', 'day', 'month', 'open', 'close', 'high', 'low',
              'p60', 'dollar_vol', 'prev_close', 'dpos', 'on_bps', 'id_bps',
              'cc_bps']


def prereg_sha256(path=None):
    """sha256 of the pre-registered region: every line above the '# ===='
    rule that opens 'THE STUDY -- machinery'."""
    L = open(path or os.path.abspath(__file__)).read().split('\n')
    i = next(i for i, l in enumerate(L)
             if l.startswith('# THE STUDY -- machinery'))
    return hashlib.sha256(('\n'.join(L[:i - 1]) + '\n').encode()).hexdigest()


def cfg_name(c):
    return f'{c[0]}/{c[1]}/{c[2]}/{c[3]}' if isinstance(c, tuple) else str(c)


def grid():
    return [(u, n, cap, fam) for fam in FAMILIES for u, _ in UNIVERSES
            for n in NS for cap in CAPS]


# ---------------------------------------------------------------- the panel

def panel_key(path):
    """Cache key: the panel files (name, size, mtime) and the code version."""
    h = hashlib.sha256(f'v{CACHE_VERSION}'.encode())
    for f in sorted(glob.glob(os.path.join(path, '*.parquet'))):
        st = os.stat(f)
        h.update(f'{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}'
                 .encode())
    return h.hexdigest()[:16]


def on_keep(df):
    """load_panel(keep_backstop=True)'s on_bps, derived from the DEFAULT
    panel (sorted by ticker, day, as load_panel returns it): the default
    series plus the backstop nights -- adjacent, |log gap| > BACKSTOP, not
    within SPLIT_TOL of a split ratio. Split-band nights stay out
    (xsec_gaps.py). Used for the information column only."""
    from xsec_backtest import SPLIT_LOGR, SPLIT_TOL, BACKSTOP
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = np.log(df.open.values / df.prev_close.values)
    prev = df.groupby('ticker', sort=False).dpos.shift().values
    adj = (df.dpos.values - prev) == 1
    big = adj & np.isfinite(raw) & (np.abs(raw) > BACKSTOP)
    near = np.full(len(df), np.inf)
    if big.any():
        near[big] = np.abs(raw[big][:, None] - SPLIT_LOGR[None, :]).min(1)
    back = big & (near >= SPLIT_TOL)
    return np.where(back, raw * 1e4, df.on_bps.values)


def load_wide(path, cache):
    """The wide panel through load_panel's default cleaning, cached."""
    key = panel_key(path)
    f = os.path.join(cache, f'panel_{key}.parquet')
    if os.path.exists(f):
        print(f'panel from cache {f}')
        return pd.read_parquet(f), key
    from xsec_backtest import load_panel
    df = load_panel(path)[PANEL_COLS].copy()
    df['on_keep'] = on_keep(df)
    os.makedirs(cache, exist_ok=True)
    df.to_parquet(f, index=False)
    return df, key


def next_month(m):
    return (pd.Period(m, 'M') + 1).strftime('%Y-%m')


def universe(rank_prev, n, extra=frozenset()):
    """U_N(T) from month T-1's ranks ALONE ({ticker: rank}): rank <= n, minus
    the frozen ETF list. Nothing from month T can enter. `extra` (POST-HOC
    only; empty in the declared run) removes further tickers."""
    from xsec_backtest import ETF
    return {tk for tk, r in rank_prev.items() if r <= n} - ETF - set(extra)


def etp_excluded(T):
    """POST-HOC: the ETP_SCREEN tickers screened for trade month T."""
    return frozenset(t for t in ETP_SCREEN if ETP_FROM.get(t, '') <= T)


def attach_ranks(df):
    """monthly_rank (xsec_wide: each month's total dollar volume within the
    panel) on every row, as column mrank."""
    from xsec_wide import monthly_rank
    rk = monthly_rank(df)
    idx = pd.MultiIndex.from_arrays([df.month.values, df.ticker.values])
    return df.assign(mrank=rk.reindex(idx).values)


# ---------------------------------------------------------------- scoring

def feature_table(df, n, months=None, exclude=None, raw_hook=None,
                  t_from=None):
    """Every trade month's scored names for U_N: xsec_live.month_aggs and
    xsec_live.features on the universe's own panel (rows with that month's
    rank <= n), over T-12 .. T-1, universe from T-1's ranks; features ranked
    with rank_pm across the scored names; on12_raw = the RULE's signal.
    Includes the month after the panel's last (formation needs no month-T
    data), for reconciliation with the live log.
    POST-HOC hooks (all None in the declared run): exclude(T) -> tickers
    taken out of U_N(T); raw_hook(live, T) -> the raw feature frame altered
    before ranking; t_from -> skip trade months before it."""
    from xsec_live import month_aggs, features
    from xsec_ml import rank_pm, FEATS
    from xsec_backtest import MIN_NAMES
    months = months or sorted(df.month.unique())
    sub = df[df.mrank <= n]
    agg = month_aggs(sub)
    rprev = {m: dict(zip(g.ticker, g.mrank))
             for m, g in sub[['month', 'ticker', 'mrank']]
             .drop_duplicates(['month', 'ticker']).groupby('month')}
    out = []
    for i in range(13, len(months) + 1):
        T = months[i] if i < len(months) else next_month(months[-1])
        if t_from is not None and T < t_from:
            continue
        live = features(agg, months[i - 12:i],
                        universe(rprev.get(months[i - 1], {}), n,
                                 exclude(T) if exclude else frozenset()))
        if len(live) < MIN_NAMES:
            continue
        if raw_hook is not None:
            live = raw_hook(live, T)
        live['on12_raw'] = live.on_12m.astype(float)
        for f in FEATS:
            live[f] = rank_pm(live[f], pd.Series(0, index=live.index))
        live['tmonth'] = T
        out.append(live)
    return pd.concat(out, ignore_index=True)


def scored(df, n, booster, cache=None, key=None, exclude=None, tag=''):
    """feature_table for U_N (cached per universe) with both families'
    scores: s_MODEL (the frozen booster) and s_RULE (on_12m). exclude / tag:
    POST-HOC screen and its cache tag (None / '' in the declared run)."""
    from xsec_ml import FEATS
    f = cache and key and os.path.join(cache,
                                       f'feat_{key}_top{n}{tag}.parquet')
    if f and os.path.exists(f):
        t = pd.read_parquet(f)
    else:
        t = feature_table(df, n, exclude=exclude)
        if f:
            t.to_parquet(f, index=False)
    t['s_MODEL'] = booster.predict(t[FEATS])
    t['s_RULE'] = t.on12_raw
    return t


def ordering(g, fam):
    """Score descending, ties by ticker ascending (xsec_live's nlargest
    over the sorted universe)."""
    return g.sort_values([f's_{fam}', 'ticker'],
                         ascending=[False, True]).ticker.tolist()


# ---------------------------------------------------------------- the cap

def pair_corr(X, min_common=CORR_MIN):
    """Pearson correlation of X's columns (days x names, NaN = missing) on
    pairwise-complete days; a pair with < min_common common days -> 0."""
    X = np.asarray(X, float)
    M = np.isfinite(X).astype(float)
    Z = np.where(M > 0, X, 0.0)
    n = M.T @ M
    sx = Z.T @ M                       # sum of x_i over days i and j share
    sxx = (Z * Z).T @ M
    sxy = Z.T @ Z
    with np.errstate(divide='ignore', invalid='ignore'):
        cov = sxy - sx * sx.T / n
        vx = sxx - sx ** 2 / n
        c = cov / np.sqrt(vx * vx.T)
    c[(n < min_common) | ~np.isfinite(c)] = 0.0
    c = np.clip(c, -1.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def clusters(corr, cut=CORR_CUT):
    """Average linkage on 1 - correlation, flat clusters at distance
    1 - cut: two groups merge iff their average correlation >= cut."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    n = len(corr)
    if n < 2:
        return np.ones(n, int)
    d = 1.0 - np.asarray(corr, float)
    d = np.clip((d + d.T) / 2, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method='average')
    return fcluster(Z, t=1.0 - cut, criterion='distance')


def pick(order, n, labels=None, cap=None):
    """The basket: the first n of `order`; with a cap, walk down the order
    and skip a name whose cluster already holds `cap` admitted names. Fewer
    than n if the order runs out."""
    if cap is None:
        return list(order[:n])
    out, cnt = [], {}
    for tk in order:
        c = labels[tk]
        if cnt.get(c, 0) >= cap:
            continue
        out.append(tk)
        cnt[c] = cnt.get(c, 0) + 1
        if len(out) == n:
            break
    return out


def cc_matrix(df, tk_code, n_days):
    """Dense days x names cc_bps (float32, NaN = no valid row)."""
    A = np.full((n_days, len(tk_code)), np.nan, np.float32)
    A[df.dpos.values, df.ticker.map(tk_code).values] = df.cc_bps.values
    return A


def window_rows(first_pos, days=CORR_DAYS):
    """The trailing formation window: the `days` calendar positions strictly
    before T's first trading day (= ending on T-1's last)."""
    return np.arange(first_pos - days, first_pos)


def form_month(g, A, rows, tk_code, ns=NS, caps=CAPS, fams=FAMILIES):
    """All baskets of one (universe, month): {(n, cap, fam): (names,
    avg pairwise corr)}. g = that month's scored names."""
    names = g.ticker.tolist()
    X = A[rows][:, [tk_code[t] for t in names]]
    corr = pair_corr(X)
    lab = dict(zip(names, clusters(corr)))
    ix = {t: i for i, t in enumerate(names)}
    out = {}
    for fam in fams:
        order = ordering(g, fam)
        for n in ns:
            for cap in caps:
                b = pick(order, n, lab, CAP_K if cap == 'cap3' else None)
                k = [ix[t] for t in b]
                sub = corr[np.ix_(k, k)]
                ac = (sub.sum() - len(k)) / (len(k) * (len(k) - 1)) \
                    if len(k) > 1 else np.nan
                out[(n, cap, fam)] = (b, float(ac))
    return out, len(set(lab.values()))


# ---------------------------------------------------------------- costs

def order_cost_vec(plan, shares, price, side, day=FEE_DAY, per_share=None):
    """paper_bench.order_cost, vectorised: same constants (imported, not
    copied), same arithmetic; a zero-share order costs nothing. per_share
    (POST-HOC, scalar or per order) replaces the plan's commission rate --
    the volume-tier sensitivity; None (the declared run) keeps the plan's."""
    import paper_bench as pb
    p, R = pb.FEES[plan], pb.REG
    q = np.asarray(shares, float)
    px = np.asarray(price, float)
    value = q * px
    rate = p['per_share'] if per_share is None else \
        np.asarray(per_share, float)
    comm = np.minimum(np.maximum(p['min'], rate * q),
                      p['max_pct'] * value)
    c = comm * (1 + p['passthru']) + (p['exch'][side] + p['clearing']
                                      + R['cat']) * q
    if side == 'SELL':
        c = c + R['sec_rate'] * value
        lo, hi = R['taf_holiday']
        if not lo <= day <= hi:
            c = c + np.minimum(R['taf'] * q, R['taf_cap'])
    return np.where(q > 0, c, 0.0)


def tier_total(shares, bands=TIER_BANDS):
    """POST-HOC: the Tiered per-share commission on a month's first `shares`
    shares (before per-order minimums and caps), piecewise linear."""
    x, y, lo, c = [0.0], [0.0], 0.0, 0.0
    for hi, r in bands:
        top = hi if np.isfinite(hi) else 1e15
        c += (top - lo) * r
        x.append(top)
        y.append(c)
        lo = top
    return np.interp(np.asarray(shares, float), x, y)


def tier_rates(vol, month):
    """POST-HOC: the marginal per-share rate of each trade day's shares, by
    IBKR's monthly cumulative volume (it resets each calendar month). vol:
    shares traded per calendar position; month: its month label. A day with
    no shares gets the rate its next share would pay."""
    vol = np.asarray(vol, float)
    cum = pd.Series(vol).groupby(np.asarray(month)).cumsum().values
    before = cum - vol
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (tier_total(cum) - tier_total(before)) / vol
    nxt = tier_total(before + 1) - tier_total(before)
    return np.where(vol > 0, r, nxt)


# ---------------------------------------------------------------- nights

class Nights:
    """Name-night outcomes keyed by (month of D, name), and the QQQ leg per
    night position p (night D = cal[p], outcome row d = cal[p + 1])."""

    def __init__(self, df, cal, months, tk_code):
        self.cal, self.months = cal, months
        mcode = {m: i for i, m in enumerate(months)}
        tk = df.ticker.map(tk_code).values.astype(np.int64)
        pos = df.dpos.values.astype(np.int64)
        same = np.r_[tk[1:] == tk[:-1], False]
        adj = same & (np.r_[pos[1:], -9] == pos + 1)
        nxt = lambda a: np.where(adj, np.r_[a[1:], np.nan], np.nan)
        key = df.month.map(mcode).values.astype(np.int64) * 16384 + tk
        o = np.argsort(key, kind='stable')
        self.key = key[o]
        self.pos = pos[o]
        self.close = df.close.values[o]
        self.dv = df.dollar_vol.values[o]
        self.has_next = adj[o]
        self.open_next = nxt(df.open.values)[o]
        self.on = nxt(df.on_bps.values)[o]
        self.on_keep = nxt(df.on_keep.values)[o]
        # QQQ, QQQQ 2004-12 .. 2011-03: one row per day, QQQ first
        q = df[df.ticker.isin(['QQQ', 'QQQQ'])].sort_values(['dpos', 'ticker']) \
            .drop_duplicates('dpos')
        L = len(cal)
        qc, qo, qn = (np.full(L + 1, np.nan) for _ in range(3))
        qc[q.dpos.values] = q.close.values
        qo[q.dpos.values] = q.open.values
        qn[q.dpos.values] = q.on_bps.values
        self.q_close, self.q_open, self.q_on = qc[:L], qo[1:L + 1], qn[1:L + 1]
        self.ok = np.isfinite(self.q_close) & np.isfinite(self.q_open) & \
            np.isfinite(self.q_on)
        dm = pd.Series(cal).str[:4] + '-' + pd.Series(cal).str[4:6]
        self.pmonth = dm.map(mcode).fillna(-1).astype(int).values
        self.mcode = mcode
        # calendar month of every trade-day position, plus one past the end
        # (POST-HOC volume tiers)
        self.tmonth6 = np.append(pd.Series(cal).str[:6].values.astype(str),
                                 '999999')

    def gather(self, mk):
        """Row indices of every (month, name) key in mk, and which key."""
        lo = np.searchsorted(self.key, mk, 'left')
        hi = np.searchsorted(self.key, mk, 'right')
        ln = hi - lo
        tot = int(ln.sum())
        start = np.repeat(lo, ln)
        off = np.arange(tot) - np.repeat(np.cumsum(ln) - ln, ln)
        return start + off, np.repeat(np.arange(len(mk)), ln)


def evaluate(N, bk, tk_code, capital=CAPITAL, plan=PLAN, day=FEE_DAY,
             cens=CENS):
    """One configuration's nights. bk: DataFrame (tmonth, ticker, nb) of its
    baskets. Returns a per-night DataFrame (index = night D) with gross,
    cost, cens, net, gross_keep, qqq_on, names, dep, unpriced counts and
    slice/ADV sums.

    gross / gross_keep are the DECLARED measure: dep x the mean overnight
    LOG return of the held names - hedge x QQQ's LOG return, in bps of C --
    a log-return proxy for the book's dollar P&L, low by about half each
    name's overnight variance. POST-HOC columns (decide nothing):
      gross_d / gross_keep_d  the dollar P&L of the shares actually held,
          sum q x (open_d - close_D) over held names with a valid outcome,
          scaled by long / (their long) -- the declared measure's own
          imputation of the rest -- minus the short's hq x (QQQ open_d -
          close_D), in bps of C;
      cost_t  the costs with IBKR's monthly cumulative volume tiers
          (TIER_BANDS) in place of the flat first-band rate, on the
          configuration's own kept nights' shares by trade date."""
    L = len(N.cal)
    mc = bk.tmonth.map(N.mcode)
    bk = bk[mc.notna()]
    mk = mc[mc.notna()].astype(np.int64).values * 16384 + \
        bk.ticker.map(tk_code).values.astype(np.int64)
    r, j = N.gather(mk)
    nb = bk.nb.values[j].astype(float)
    p = N.pos[r]
    use = N.ok[p]
    r, nb, p = r[use], nb[use], p[use]
    close = N.close[r]
    per = capital / nb
    q = np.floor(per / close)
    held = q >= 1
    q = np.where(held, q, 0.0)
    on, onk = N.on[r], N.on_keep[r]
    valid = held & np.isfinite(on)
    validk = held & np.isfinite(onk)
    sell_px = np.where(np.isfinite(on), N.open_next[r], close)
    c_long = order_cost_vec(plan, q, close, 'BUY', day) + \
        order_cost_vec(plan, q, sell_px, 'SELL', day)
    bc = lambda w: np.bincount(p, weights=w, minlength=L)
    long_ = bc(q * close)
    s_on, n_on = bc(np.where(valid, on, 0.0)), bc(valid.astype(float))
    s_k, n_k = bc(np.where(validk, onk, 0.0)), bc(validk.astype(float))
    cost_l = bc(c_long)
    n_held = bc(held.astype(float))
    priced = bc((N.has_next[r]).astype(float))
    s_adv = bc(np.where(held, per / N.dv[r], 0.0))
    # expected name-nights: the basket size on every valid night of its month
    nb_m = bk.drop_duplicates('tmonth').set_index('tmonth').nb
    nbm = np.zeros(len(N.months))
    for m, v in nb_m.items():
        if m in N.mcode:
            nbm[N.mcode[m]] = v
    has_b = (N.pmonth >= 0) & (nbm[np.maximum(N.pmonth, 0)] > 0)
    expected = np.where(has_b & N.ok, nbm[np.maximum(N.pmonth, 0)], 0.0)
    hq = np.where((long_ > 0) & N.ok,
                  np.rint(np.where(N.ok, long_ / N.q_close, 0.0)), 0.0)
    S = hq * np.nan_to_num(N.q_close)
    c_h = order_cost_vec(plan, hq, np.nan_to_num(N.q_close), 'SELL', day) + \
        order_cost_vec(plan, hq, np.nan_to_num(N.q_open), 'BUY', day)
    with np.errstate(divide='ignore', invalid='ignore'):
        dep = long_ / capital
        gross = dep * s_on / n_on - S / capital * N.q_on
        gross_k = dep * s_k / n_k - S / capital * N.q_on
    cost = (cost_l + c_h) / capital * 1e4
    keep = has_b & N.ok & (n_on >= 1)
    # ---- POST-HOC columns (the declared ones above are untouched)
    notl = q * close
    pl = bc(np.where(valid, notl * np.expm1(np.where(valid, on, 0) / 1e4),
                     0.0))
    lv = bc(np.where(valid, notl, 0.0))
    plk = bc(np.where(validk, notl * np.expm1(np.where(validk, onk, 0)
                                               / 1e4), 0.0))
    lvk = bc(np.where(validk, notl, 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        q_simple = np.expm1(N.q_on / 1e4) * 1e4
        gross_d = pl * long_ / lv / capital * 1e4 - S / capital * q_simple
        gross_kd = plk * long_ / lvk / capital * 1e4 - S / capital * q_simple
    kr = keep[p]                          # name-nights on kept nights
    hk = np.where(keep, hq, 0.0)
    vol = np.bincount(p[kr], weights=q[kr], minlength=L + 1) + \
        np.bincount(p[kr] + 1, weights=q[kr], minlength=L + 1)
    vol[:L] += hk
    vol[1:] += hk
    rate = tier_rates(vol, N.tmonth6)
    c_long_t = order_cost_vec(plan, q, close, 'BUY', day, rate[p]) + \
        order_cost_vec(plan, q, sell_px, 'SELL', day, rate[p + 1])
    c_h_t = order_cost_vec(plan, hq, np.nan_to_num(N.q_close), 'SELL', day,
                           rate[:L]) + \
        order_cost_vec(plan, hq, np.nan_to_num(N.q_open), 'BUY', day,
                       rate[1:])
    cost_t = (bc(c_long_t) + c_h_t) / capital * 1e4
    out = pd.DataFrame(dict(
        gross=gross, cost=cost, cens=cens * dep, gross_keep=gross_k,
        qqq_on=N.q_on, names=n_on, held=n_held, dep=dep,
        expected=expected, priced=priced, s_adv=s_adv,
        month=(pd.Series(N.cal).str[:4] + '-' +
               pd.Series(N.cal).str[4:6]).values,
        gross_d=gross_d, gross_keep_d=gross_kd, cost_t=cost_t),
        index=pd.Index(N.cal, name='night'))[keep]
    out['net'] = out.gross - out.cost - out.cens
    out['net_meas'] = out.gross_keep - out.cost
    return out


# ---------------------------------------------------------------- metrics

def max_dd(x):
    """Worst fall from a running peak of the cumulative sum of x (bps of
    C, fixed notional, from 0 at the start), in % of C."""
    cum = np.cumsum(np.asarray(x, float)) / 1e4 * 100
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return float((cum - peak).min()) if len(cum) else np.nan


def metrics(ev, mo):
    """The declared metrics on one window of one configuration's nights.
    mo: per-month table (avgcorr, turnover) of the configuration."""
    x = ev.net.values
    n = len(x)
    if n < 3:
        return None
    sd = x.std(ddof=1)
    q = ev.qqq_on.values
    vq = q.var(ddof=1)
    beta = float(np.cov(x, q, ddof=1)[0, 1] / vq) if vq > 0 else np.nan
    m_ = mo.reindex(ev.month.values)
    exp = ev.expected.sum()
    held = ev.held.sum()
    return dict(
        nights=n, gross=float(ev.gross.mean()), cost=float(ev.cost.mean()),
        cens=float(ev.cens.mean()), net=float(x.mean()), sd=float(sd),
        sharpe=float(x.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
        t=float(x.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan,
        worst=float(x.min()), p1=float(np.percentile(x, 1)),
        maxdd=max_dd(x), beta=beta,
        corr=float(np.nanmean(m_.avgcorr.values)),
        names=float(ev.names.mean()),
        turnover=float(np.nanmean(m_.turnover.values)),
        cens_meas=float((ev.gross - ev.gross_keep).mean()),
        net_meas=float(ev.net_meas.mean()),
        sharpe_meas=float(ev.net_meas.mean() / ev.net_meas.std(ddof=1)
                          * np.sqrt(252)),
        unpriced=float(1 - ev.priced.sum() / exp) if exp else np.nan,
        slice_adv=float(ev.s_adv.sum() / held * 100) if held else np.nan)


# ---------------------------------------------------------------- the rule

def select(sel, conf, current):
    """S1-S3 on one family. sel / conf: DataFrames indexed by configuration
    with columns net, sd, sharpe, maxdd, p1 (SELECTION / CONFIRMATION).
    Returns dict(candidates, pick, checks, proposed, verdict)."""
    c = sel.loc[current]
    floor = S_MEAN * c.net if c.net > 0 else c.net
    cand = sel[(sel.net >= floor) & (sel.sd <= S_SD * c.sd)]
    out = dict(candidates=list(cand.index), pick=None, checks={},
               proposed=False, mean_floor=floor, sd_cap=S_SD * c.sd)
    if cand.empty:
        out['verdict'] = ('the current basket stands -- no SELECTION '
                          'candidate passes S1')
        return out
    k = cand.assign(_s=-cand.sharpe.round(2), _d=-cand.maxdd) \
        .sort_values(['_s', '_d'], kind='stable')
    pk = k.index[0]
    out['pick'] = pk
    a, b = conf.loc[pk], conf.loc[current]
    ch = {'sharpe': bool(a.sharpe >= b.sharpe),
          'maxdd': bool(a.maxdd >= b.maxdd),
          'p1': bool(a.p1 >= b.p1)}
    out['checks'] = ch
    out['proposed'] = all(ch.values())
    out['verdict'] = (f'PROPOSED: {cfg_name(pk)}' if out['proposed'] else
                      f'the current basket stands -- the pick {cfg_name(pk)} '
                      f'fails S3 on ' +
                      ', '.join(k_ for k_, v in ch.items() if not v))
    return out


def form_all(tabs, A, first, n_cal, tk_code):
    """Every configuration's baskets from the scored tables {u: table}:
    B (cfg, tmonth, ticker, nb, rank), MO (cfg, tmonth, avgcorr, nb,
    turnover) and the clusters per (universe, month)."""
    B, MO, ncl = [], [], []
    for u, _ in UNIVERSES:
        if u not in tabs:
            continue
        for T, g in tabs[u].groupby('tmonth'):
            p0 = int(first[T]) if T in first.index else n_cal
            res, k = form_month(g, A, window_rows(p0), tk_code)
            ncl.append((u, T, len(g), k))
            for (nn, cap, fam), (b, ac) in res.items():
                c = (u, nn, cap, fam)
                B.append(pd.DataFrame(dict(cfg=cfg_name(c), tmonth=T,
                                           ticker=b, nb=len(b),
                                           rank=range(len(b)))))
                MO.append((cfg_name(c), T, ac, len(b)))
    B = pd.concat(B, ignore_index=True)
    MO = pd.DataFrame(MO, columns=['cfg', 'tmonth', 'avgcorr', 'nb'])
    ncl = pd.DataFrame(ncl, columns=['u', 'tmonth', 'scored', 'clusters'])
    turn = {}
    for c, g in B.groupby('cfg'):
        prev = None
        for T, h in g.groupby('tmonth'):
            s = set(h.ticker)
            turn[(c, T)] = np.nan if prev is None else 1 - len(s & prev) / len(s)
            prev = s
    MO['turnover'] = [turn[(c, T)] for c, T in zip(MO.cfg, MO.tmonth)]
    return B, MO, ncl


# ---------------------------------------------------------------- POST-HOC
# Everything below this line down to "report" is POST-HOC: written after the
# declared run was seen, from the review, and printed under POST-HOC labels.

def view(ev, measure='declared'):
    """A configuration's nights under a measure: 'declared'; 'dollar' (the
    dollar P&L of the shares held, gross_d); 'tiers' (IBKR monthly volume
    tiers, cost_t); 'dollar+tiers'. The flat censoring stays as declared."""
    dollar, tiers = 'dollar' in measure, 'tiers' in measure
    g = ev.gross_d if dollar else ev.gross
    gk = ev.gross_keep_d if dollar else ev.gross_keep
    c = ev.cost_t if tiers else ev.cost
    return ev.assign(gross=g, gross_keep=gk, cost=c, net=g - c - ev.cens,
                     net_meas=gk - c)


def table(evs, MO, measure='declared', windows=WINDOWS):
    """The declared metrics of every configuration and window, under a
    measure (view)."""
    rows = []
    for c, ev in evs.items():
        nm = cfg_name(c)
        e0 = view(ev, measure)
        mo = MO[MO.cfg == nm].set_index('tmonth')
        for w, lo, hi in windows:
            r = metrics(e0[(e0.index >= lo) & (e0.index <= hi)], mo)
            if r:
                rows.append(dict(cfg=nm, u=c[0], n=c[1], cap=c[2], fam=c[3],
                                 window=w, **r))
    return pd.DataFrame(rows)


def rule(R):
    """S1-S3 per family on a metrics table: {fam: select(...)} plus the
    tables used, keyed as the declared run does."""
    out = {}
    for fam, base in (('MODEL', CURRENT), ('RULE', RULE_BASE)):
        S = R[(R.window == 'SELECTION') & (R.fam == fam)].copy()
        Cf = R[(R.window == 'CONFIRMATION') & (R.fam == fam)].copy()
        for d in (S, Cf):
            d.index = d.cfg.values
        out[fam] = (select(S, Cf, cfg_name(base)), S, Cf, cfg_name(base))
    return out


def paired(ea, eb, lo, hi, col='net'):
    """Mean and t of a - b on the nights both configurations hold, in
    [lo, hi]."""
    a, b = ea[col], eb[col]
    d = (a - b).dropna()
    d = d[(d.index >= lo) & (d.index <= hi)]
    if len(d) < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan, len(d)
    return (float(d.mean()), float(d.mean() / d.std(ddof=1)
                                   * np.sqrt(len(d))), len(d))


def etp_share(B):
    """Share of each configuration's basket name-months that ETP_SCREEN
    marks as an exchange-traded product in that trade month."""
    ex = {T: etp_excluded(T) for T in B.tmonth.unique()}
    hit = np.array([t in ex[T] for t, T in zip(B.ticker, B.tmonth)])
    return pd.Series(hit, index=B.index).groupby(B.cfg).mean()


def seam_splits(df, first=FWD_FIRST):
    """Names whose first-day open after the HF / Yahoo seam gaps from the
    last HF close by a split ratio: Yahoo back-adjusts history for splits
    up to its download date, so their Yahoo prices are scaled. Returns
    {ticker: as-traded price / Yahoo price if the split came after}."""
    from xsec_backtest import SPLIT_LOGR, SPLIT_TOL
    if first not in set(df.month):
        return {}
    p0 = int(df.loc[df.month == first, 'dpos'].min())
    a = df.loc[df.dpos == p0 - 1].set_index('ticker').close
    b = df.loc[df.dpos == p0].set_index('ticker').open
    j = pd.concat([a, b], axis=1, join='inner', keys=['c', 'o']).dropna()
    g = np.log(j.o / j.c).values
    k = np.abs(g[:, None] - SPLIT_LOGR[None, :]).argmin(1)
    hit = (np.abs(g - SPLIT_LOGR[k]) < SPLIT_TOL) & (np.abs(g) > np.log(1.25))
    return dict(zip(j.index[hit], np.exp(-SPLIT_LOGR[k[hit]])))


def lprice_bound_hook(fac, t_from):
    """raw_hook for feature_table: lprice at the as-traded price for every
    seam-split name from trade month t_from on -- the UPPER bound of the
    contamination (every split assumed after the formation)."""
    def hook(live, T):
        if T < t_from:
            return live
        live = live.copy()
        live['lprice'] = live.lprice + np.log10(
            live.ticker.map(fac).fillna(1.0).astype(float))
        return live
    return hook


# ---------------------------------------------------------------- report

HDR = (f'{"config":<26}{"nts":>5}{"gross":>7}{"cost":>6}{"cens":>5}'
       f'{"NET":>7}{"sd":>7}{"Shrp":>6}{"t":>6}{"worst":>8}{"p1":>7}'
       f'{"maxDD%":>8}{"beta":>6}{"corr":>6}{"nm/n":>6}{"turn":>5}'
       f' |{"c_meas":>7}{"n_meas":>7}{"Sh_m":>6}{"unpr%":>6}{"%ADV":>6}')


def fmt_row(c, r):
    return (f'{cfg_name(c):<26}{r["nights"]:>5}{r["gross"]:>+7.2f}'
            f'{r["cost"]:>6.2f}{r["cens"]:>5.2f}{r["net"]:>+7.2f}'
            f'{r["sd"]:>7.1f}{r["sharpe"]:>+6.2f}{r["t"]:>+6.2f}'
            f'{r["worst"]:>8.0f}{r["p1"]:>7.0f}{r["maxdd"]:>8.1f}'
            f'{r["beta"]:>+6.2f}{r["corr"]:>6.2f}{r["names"]:>6.1f}'
            f'{r["turnover"] * 100:>4.0f}%'
            f' |{r["cens_meas"]:>+7.2f}{r["net_meas"]:>+7.2f}'
            f'{r["sharpe_meas"]:>+6.2f}{r["unpriced"] * 100:>6.1f}'
            f'{r["slice_adv"]:>6.2f}')


def reconcile(tabs, N, ev_cur, log_path, daily_path):
    """Information: the replicated formation against the live registry
    (xsec_live's Q5 = n_scored // 5 on top150, frozen model), and the
    CURRENT replica's gross against the graded paper log."""
    print('\nRECONCILIATION with the live log (information):')
    if os.path.exists(log_path):
        log = pd.read_csv(log_path, dtype=str)
        t = tabs['top150']
        for _, r in log.iterrows():
            g = t[t.tmonth == r.month]
            if g.empty:
                print(f'  {r.month}: not formable from the wide panel')
                continue
            k = len(g) // 5
            mine = ordering(g, 'MODEL')[:k]
            live = r.q5.split(';')
            print(f'  {r.month}: n_scored live {r.n_scored} / replica '
                  f'{len(g)};  Q5 overlap {len(set(mine) & set(live))} of '
                  f'{len(live)}  (replica-only: '
                  f'{",".join(sorted(set(mine) - set(live))) or "-"}; '
                  f'live-only: {",".join(sorted(set(live) - set(mine))) or "-"})')
    if os.path.exists(daily_path):
        d = pd.read_csv(daily_path, dtype={'date': str})
        j = d.set_index('date').join(ev_cur[['gross', 'dep']], how='inner')
        if len(j):
            g = j.gross / j.dep
            print(f'  {len(j)} common nights: paper-log tilt mean '
                  f'{j.tilt_bps.mean():+.1f}  replica (11 names, wide panel) '
                  f'{g.mean():+.1f}  corr {np.corrcoef(j.tilt_bps, g)[0, 1]:+.2f}')


FWD_NOTE = ('  FORWARD caveat: the MODEL rows\' lprice input is split-'
            'contaminated (Yahoo back-adjusts history for splits up to its '
            'download)\n  and the Yahoo months carry build-date survivorship;'
            ' the RULE rows are scale-free. See POST-HOC.')


def print_tables(R, extra=None):
    """The four window tables of a metrics table, as the declared run prints
    them. extra: {window: [(label, row)]} printed after the grid's rows."""
    for w, lo, hi in WINDOWS:
        print(f'\n== {w} ({lo[:4]}-{lo[4:6]} .. {hi[:4]}-{hi[4:6]}) ' +
              '=' * 60)
        print(HDR)
        for fam in FAMILIES:
            for c in [c for c in grid() if c[3] == fam]:
                r = R[(R.cfg == cfg_name(c)) & (R.window == w)]
                if len(r):
                    print(fmt_row(c, r.iloc[0]))
            print()
        for lab, r in (extra or {}).get(w, []):
            print(fmt_row(lab, r))
        z = R[R.window == w]
        b = z.loc[z.sharpe.idxmax()]
        bm = z[z.fam == 'MODEL'].loc[z[z.fam == 'MODEL'].sharpe.idxmax()]
        br = z[z.fam == 'RULE'].loc[z[z.fam == 'RULE'].sharpe.idxmax()]
        print(f'  best by net Sharpe (reported, not decided): overall '
              f'{b.cfg} {b.sharpe:+.2f};  MODEL {bm.cfg} {bm.sharpe:+.2f};  '
              f'RULE {br.cfg} {br.sharpe:+.2f}')
        if w == 'FORWARD':
            print(FWD_NOTE)


def print_rule(rr):
    """S1-S3 per family, as the declared run prints it."""
    for fam in ('MODEL', 'RULE'):
        s, S, Cf, base = rr[fam]
        b = S.loc[base]
        print(f'\n {fam} family ({"DECIDES" if fam == "MODEL" else "model-free check, decides nothing"}); '
              f'baseline {cfg_name(base)}: SELECTION net {b.net:+.2f}, sd '
              f'{b.sd:.1f}, Sharpe {b.sharpe:+.2f}')
        print(f'  S1 candidates (net >= {s["mean_floor"]:+.2f} and sd <= '
              f'{s["sd_cap"]:.1f}): ' + (', '.join(
                  f'{cfg_name(k)} [net {S.loc[k].net:+.2f} sd {S.loc[k].sd:.1f}'
                  f' Sh {S.loc[k].sharpe:+.2f}]' for k in s['candidates'])
                  or 'none'))
        if s['pick'] is not None:
            p, q = Cf.loc[s['pick']], Cf.loc[base]
            print(f'  S2 pick: {cfg_name(s["pick"])}')
            print(f'  S3 CONFIRMATION, pick vs baseline: Sharpe '
                  f'{p.sharpe:+.2f} vs {q.sharpe:+.2f} '
                  f'[{"ok" if s["checks"]["sharpe"] else "FAIL"}];  maxDD '
                  f'{p.maxdd:.1f}% vs {q.maxdd:.1f}% '
                  f'[{"ok" if s["checks"]["maxdd"] else "FAIL"}];  p1 '
                  f'{p.p1:.0f} vs {q.p1:.0f} '
                  f'[{"ok" if s["checks"]["p1"] else "FAIL"}]')
        print(f'  -> {s["verdict"]}')


def posthoc(df, tabs, A, first, cal, tk_code, N, booster, evs, B, MO, R,
            cache, key, t0):
    """The review's corrections, run AFTER the declared verdict and printed
    as POST-HOC: not evidence, they decide nothing."""
    from xsec_ml import FEATS
    bar = '#' * 100
    print(f'\n{bar}\nPOST-HOC -- written after the declared run was seen, '
          'from an adversarial review. NOT evidence; the declared\nverdict '
          f'above stands as declared. Each variant changes one thing the '
          f'review found and keeps everything else.\n{bar}')

    # (1) ETPs in the declared baskets
    sh = etp_share(B)
    print('\n[P3 input] exchange-traded products the frozen top-150 ETF list '
          'misses (ETP_SCREEN), as a share of\nbasket name-months in the '
          'DECLARED formation (MODEL / RULE):')
    for u, _ in UNIVERSES:
        print('  ' + '   '.join(
            f'{u}/{n}/{cap}: ' + ' / '.join(
                f'{sh.get(cfg_name((u, n, cap, f)), 0) * 100:4.1f}%'
                for f in FAMILIES) for n in NS for cap in CAPS))

    # (2) the screened formation: ETP_SCREEN out of every universe
    tag = '_etp' + hashlib.sha256(' '.join(sorted(ETP_SCREEN)).encode() +
                                  repr(sorted(ETP_FROM.items())).encode()
                                  ).hexdigest()[:8]
    tabs_e = {u: scored(df, n, booster, cache, key, exclude=etp_excluded,
                        tag=tag) for u, n in UNIVERSES}
    B_e, MO_e, _ = form_all(tabs_e, A, first, len(cal), tk_code)
    evs_e = {c: evaluate(N, B_e[B_e.cfg == cfg_name(c)], tk_code)
             for c in grid()}
    left = etp_share(B_e).max()
    print(f'\n[P3] screened formation: {len(MO_e)} baskets; ETP share left in '
          f'any basket {left * 100:.1f}%   [{time.time() - t0:.0f}s]')
    # the as-traded baselines (frozen list) stay the baselines
    evs4 = dict(evs_e)
    MO4 = MO_e[~MO_e.cfg.isin([cfg_name(CURRENT), cfg_name(RULE_BASE)])]
    for c in (CURRENT, RULE_BASE):
        evs4[c] = evs[c]
    MO4 = pd.concat([MO4, MO[MO.cfg.isin([cfg_name(CURRENT),
                                          cfg_name(RULE_BASE)])]])

    variants = [('declared (as run)', evs, MO, 'declared'),
                ('P1 dollar P&L', evs, MO, 'dollar'),
                ('P2 IBKR volume tiers', evs, MO, 'tiers'),
                ('P3 ETP screen', evs4, MO4, 'declared'),
                ('P4 all three', evs4, MO4, 'dollar+tiers')]
    tabsR, out = {}, []
    print('\nTHE SELECTION RULE under each POST-HOC variant (baselines: the '
          'CURRENT book and the RULE baseline as\ntraded, frozen ETF list; '
          'S1 floor / sd cap move with the baseline):')
    for lab, ev_, mo_, meas in variants:
        Rv = table(ev_, mo_, meas)
        tabsR[lab] = Rv
        out.append(Rv.assign(variant=lab))
        rr = rule(Rv)
        for fam in ('MODEL', 'RULE'):
            s, S, Cf, base = rr[fam]
            b = S.loc[base]
            pk = s['pick']
            s3 = '' if pk is None else (
                f'; S3 {cfg_name(pk)}: Sh {Cf.loc[pk].sharpe:+.2f} vs '
                f'{Cf.loc[base].sharpe:+.2f}, DD {Cf.loc[pk].maxdd:.1f} vs '
                f'{Cf.loc[base].maxdd:.1f}, p1 {Cf.loc[pk].p1:.0f} vs '
                f'{Cf.loc[base].p1:.0f}')
            print(f'  {lab:<20} {fam:<5} base net {b.net:+6.2f} sd '
                  f'{b.sd:5.1f} Sh {b.sharpe:+.2f}; S1 (>= {s["mean_floor"]:+.2f},'
                  f' <= {s["sd_cap"]:.1f}): {len(s["candidates"])} cand.'
                  f'{s3}\n  {"":<20} {"":<5} -> {s["verdict"]}')

    R4 = tabsR['P4 all three']
    scr = table({CURRENT: evs_e[CURRENT], RULE_BASE: evs_e[RULE_BASE]},
                MO_e, 'dollar+tiers')
    extra = {w: [(cfg_name(c) + '*',
                  scr[(scr.cfg == cfg_name(c)) & (scr.window == w)].iloc[0])
                 for c in (CURRENT, RULE_BASE)
                 if len(scr[(scr.cfg == cfg_name(c)) & (scr.window == w)])]
             for w, _, _ in WINDOWS}
    print('\nP4 TABLES (POST-HOC, not evidence): dollar P&L of the shares held,'
          ' IBKR monthly volume tiers, ETP_SCREEN out of\nevery universe; the '
          'two baselines as traded. Columns as declared (cost = tiered; '
          'n_meas / Sh_m on the dollar P&L).\n'
          '* = the baseline re-formed with ETP_SCREEN (information; the '
          'unstarred baselines are as traded).')
    print_tables(R4, extra)

    # (3) in-sample lift and like-for-like breadth (paired nights)
    W3 = (('SELECTION', '20010101', '20151231'),
          ('CONF to 2026-03', '20160101', MODEL_CUTOFF),
          ('FORWARD', '20260401', '20260930'))
    v4 = {c: view(e, 'dollar+tiers') for c, e in evs4.items()}
    print('\nIN-SAMPLE LIFT AND LIKE-FOR-LIKE BREADTH (POST-HOC information).'
          ' The frozen model was trained on top-150\nname-months through '
          'trade month 2026-03: MODEL rows are in-sample through 2026-03 and'
          ' out of sample only in\nFORWARD. Paired nights, mean bps/night '
          '(t); each cell "declared measure | P4".')
    print(f'  {"":<44}' + ''.join(f'{w:>26}' for w, _, _ in W3))

    def cell(fa, fb, lo, hi, col):
        m1, t1, _ = paired(fa[0], fb[0], lo, hi, col)
        m2, t2, _ = paired(fa[1], fb[1], lo, hi, col)
        return f'{m1:+6.2f} ({t1:+5.1f}) |{m2:+6.2f} ({t2:+5.1f})'
    print('  gross, MODEL - RULE, same universe /11/none:')
    for u, _ in UNIVERSES:
        m, r_ = (u, 11, 'none', 'MODEL'), (u, 11, 'none', 'RULE')
        print(f'    {u:<42}' + ''.join(
            f'{cell((evs[m], v4[m]), (evs[r_], v4[r_]), lo, hi, "gross"):>26}'
            for _, lo, hi in W3))
    print('  net, MODEL config - CURRENT:')
    for c in [(u, 11, 'none', 'MODEL') for u in ('top300', 'top500',
                                                  'top1000')] + \
            [('top150', 11, 'cap3', 'MODEL'), ('top150', 20, 'cap3', 'MODEL')]:
        print(f'    {cfg_name(c):<42}' + ''.join(
            f'{cell((evs[c], v4[c]), (evs[CURRENT], v4[CURRENT]), lo, hi, "net"):>26}'
            for _, lo, hi in W3))
    print('  net, RULE config - RULE baseline (top150/11/none/RULE):')
    for c in [(u, 11, 'none', 'RULE') for u in ('top300', 'top500',
                                                 'top1000')] + \
            [('top150', 11, 'cap3', 'RULE'), ('top500', 11, 'cap3', 'RULE')]:
        print(f'    {cfg_name(c):<42}' + ''.join(
            f'{cell((evs[c], v4[c]), (evs[RULE_BASE], v4[RULE_BASE]), lo, hi, "net"):>26}'
            for _, lo, hi in W3))

    # (4) FORWARD: the seam splits and an upper bound on lprice
    fac = seam_splits(df)
    t_from = next_month(FWD_FIRST)
    fb = B[(B.tmonth >= FWD_FIRST) & B.ticker.isin(fac)]
    print(f'\nFORWARD DATA (POST-HOC information): {len(fac)} names gap at '
          f'the 2026-03 / {FWD_FIRST} seam by a split ratio\n(Yahoo folds '
          f'later splits into earlier prices), e.g. ' + ', '.join(
              f'{t} x{1 / f:.3g}' for t, f in sorted(
                  fac.items(), key=lambda kv: -abs(np.log(kv[1])))[:6]) +
          f'. FORWARD basket name-months holding them: MODEL '
          f'{int(fb.cfg.str.endswith("MODEL").sum())}, RULE '
          f'{int(fb.cfg.str.endswith("RULE").sum())}.')
    if fac:
        tabs_b = {}
        for u, n in UNIVERSES:
            t = feature_table(df, n, raw_hook=lprice_bound_hook(fac, t_from),
                              t_from=t_from)
            t['s_MODEL'] = booster.predict(t[FEATS])
            t['s_RULE'] = t.on12_raw
            tabs_b[u] = t
        B_b, _, _ = form_all(tabs_b, A, first, len(cal), tk_code)
        B_b = B_b[B_b.cfg.str.endswith('MODEL')]
        old = B[(B.tmonth >= t_from) & B.cfg.str.endswith('MODEL')]
        sa = old.groupby(['cfg', 'tmonth']).ticker.apply(frozenset)
        sb = B_b.groupby(['cfg', 'tmonth']).ticker.apply(frozenset)
        ch = (sa != sb.reindex(sa.index)).groupby(level=0).sum()
        Bn = pd.concat([B[~((B.tmonth >= t_from) &
                            B.cfg.str.endswith('MODEL'))], B_b])
        print(f'  Upper bound (every seam split assumed after formation, '
              f'lprice at the as-traded price, trade months {t_from}..): '
              f'{int(ch.sum())} of {len(sa)} MODEL baskets change. FORWARD '
              f'MODEL net / Sharpe, declared measure, as run -> bound:')
        lines = []
        for c in [c for c in grid() if c[3] == 'MODEL']:
            nm = cfg_name(c)
            e1 = evs[c][(evs[c].index >= '20260401') &
                        (evs[c].index <= '20260930')]
            e2 = evaluate(N, Bn[Bn.cfg == nm], tk_code)
            e2 = e2[(e2.index >= '20260401') & (e2.index <= '20260930')]
            sh_ = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252)
            lines.append(f'{nm:<22}{e1.net.mean():+6.2f} {sh_(e1.net):+5.2f}'
                         f' -> {e2.net.mean():+6.2f} {sh_(e2.net):+5.2f}'
                         f' ({int(ch.get(nm, 0))} chg)')
        for i in range(0, len(lines), 3):
            print('    ' + '   '.join(lines[i:i + 3]))

    P = pd.concat(out, ignore_index=True)
    f = os.path.join(cache, 'basket_study_posthoc.csv')
    P.to_csv(f, index=False, float_format='%.4f')
    print(f'\nPOST-HOC metrics -> {f}   [{time.time() - t0:.0f}s]')
    return dict(posthoc=tabsR, evs_etp=evs_e, B_etp=B_e, MO_etp=MO_e)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Broader / theme-capped basket '
                                 'study (declared in the header).')
    ap.add_argument('--dir', default='data/xsec1000')
    ap.add_argument('--cache', default=os.path.join(ROOT, 'data',
                                                    '_work_basket'))
    ap.add_argument('--allow-edited-prereg', action='store_true')
    ap.add_argument('--no-posthoc', action='store_true',
                    help='print the declared run only')
    a = ap.parse_args(argv)
    t0 = time.time()
    sha = prereg_sha256()
    ok = sha == PREREG_SHA256
    print(f'pre-registered region sha256 {sha} -- '
          + ('matches the pin' if ok else
             'DOES NOT MATCH the pin ' + PREREG_SHA256))
    if not ok and not a.allow_edited_prereg:
        raise SystemExit('the declarations were edited after they were '
                         'pinned; refusing (--allow-edited-prereg to override, '
                         'and say so in the report)')
    import lightgbm as lgb
    path = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    os.makedirs(a.cache, exist_ok=True)
    df, key = load_wide(path, a.cache)
    df = attach_ranks(df)
    months = sorted(df.month.unique())
    cal = np.sort(df.day.unique())
    assert (df.dpos.values == np.searchsorted(cal, df.day.values)).all()
    tickers = sorted(df.ticker.unique())
    tk_code = {t: i for i, t in enumerate(tickers)}
    print(f'wide panel: {len(months)} months {months[0]}..{months[-1]}, '
          f'{len(df):,} rows, {len(tickers)} names   [{time.time() - t0:.0f}s]')
    booster = lgb.Booster(model_file=MODEL)

    tabs = {}
    for u, n in UNIVERSES:
        tabs[u] = scored(df, n, booster, a.cache, key)
        g = tabs[u].groupby('tmonth').size()
        print(f'  {u}: {g.size} trade months {g.index.min()}..{g.index.max()}, '
              f'scored names/month mean {g.mean():.0f} (min {g.min()}, last '
              f'{g.iloc[-1]})   [{time.time() - t0:.0f}s]')

    A = cc_matrix(df, tk_code, len(cal))
    first = df.groupby('month').dpos.min()
    B, MO, ncl = form_all(tabs, A, first, len(cal), tk_code)
    print(f'formation: {len(MO)} baskets   [{time.time() - t0:.0f}s]')
    for u, _ in UNIVERSES:
        z = ncl[ncl.u == u]
        print(f'  {u}: clusters/month mean {z.clusters.mean():.0f} of '
              f'{z.scored.mean():.0f} scored; fewest {z.clusters.min()} '
              f'({z.tmonth.values[z.clusters.values.argmin()]})')

    N = Nights(df, cal, months, tk_code)
    print(f'nights: {int(N.ok.sum())} with a valid QQQ leg, '
          f'{int((~N.ok[:-1]).sum())} dropped   [{time.time() - t0:.0f}s]')
    evs = {c: evaluate(N, B[B.cfg == cfg_name(c)], tk_code) for c in grid()}
    R = table(evs, MO)
    print(f'evaluation: {len(evs)} configurations   [{time.time() - t0:.0f}s]')

    print('\nUnits: bps of the $500,000 capital per night, on the DECLARED '
          'measure -- dep x mean overnight LOG return - hedge x QQQ LOG '
          'return,\na log proxy that understates the dollar P&L (see '
          'POST-HOC); net = gross - cost - cens; maxDD in % of capital at '
          'fixed notional;\n'
          'corr = avg pairwise trailing-63d cc correlation in the basket at '
          'formation; nm/n = held names with a valid outcome per night;\n'
          'turn = basket replaced per month. Right of "|" is INFORMATION: '
          'c_meas = measured censoring (gross - gross with backstop nights),\n'
          'n_meas / Sh_m = net and Sharpe with the measured censoring in '
          'place of the flat 1.5, unpr% = basket name-nights unpriced,\n'
          '%ADV = per-name slice as % of the name\'s day-D dollar volume.')
    print_tables(R)

    print('\nTHE SELECTION RULE, applied')
    rr = rule(R)
    print_rule(rr)
    verdicts = {fam: rr[fam][0] for fam in rr}

    reconcile(tabs, N, evs[CURRENT],
              os.path.join(ROOT, 'reports', 'xsec_paper.csv'),
              os.path.join(ROOT, 'reports', 'xsec_paper_daily.csv'))
    out = os.path.join(a.cache, 'basket_study_metrics.csv')
    R.to_csv(out, index=False, float_format='%.4f')
    print(f'\nVERDICT (MODEL family decides): {verdicts["MODEL"]["verdict"]}')
    print(f'model-free check (RULE family): {verdicts["RULE"]["verdict"]}')
    print(f'metrics -> {out}')
    res = dict(R=R, verdicts=verdicts, evs=evs, B=B, MO=MO)
    if not a.no_posthoc:
        res.update(posthoc(df, tabs, A, first, cal, tk_code, N, booster,
                           evs, B, MO, R, a.cache, key, t0))
    nightly = pd.concat({cfg_name(c): e[['gross', 'cost', 'cens', 'net',
                                         'net_meas', 'qqq_on', 'names',
                                         'gross_d', 'gross_keep_d', 'cost_t']]
                         for c, e in evs.items()}, names=['cfg'])
    nightly.to_parquet(os.path.join(a.cache, 'basket_study_nightly.parquet'))
    print(f'\nrun time {time.time() - t0:.0f}s; '
          f'pre-registered region sha256 {sha}')
    return res


if __name__ == '__main__':
    main()
