# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
# Streams one day at a time out of the zip: the archive is larger than most free disks.
import os
import databento as db, pandas as pd, numpy as np, sys
np.seterr(all='ignore')
df = db.DBNStore.from_file(sys.argv[1]).to_df().reset_index()
df = df[(df['flags'].values.astype(np.uint8) & 128) != 0]
ts = df.ts_recv.values.astype('datetime64[ns]').astype(np.int64)
day0 = (ts[0]//86_400_000_000_000)*86_400_000_000_000
m = (ts >= day0+13*3600*10**9+30*60*10**9) & (ts < day0+20*3600*10**9)
d, ts = df[m], ts[m]
bp, ap = d.bid_px_00.values, d.ask_px_00.values
bs, as_ = d.bid_sz_00.values.astype(float), d.ask_sz_00.values.astype(float)
ok = np.isfinite(bp)&np.isfinite(ap)&(ap>bp)&(bs>0)&(as_>0)
d, ts, bp, ap, bs, as_ = d[ok], ts[ok], bp[ok], ap[ok], bs[ok], as_[ok]
mid, spr = (bp+ap)/2.0, ap-bp

act, sd = d.action.values, d.side.values
px, sz = d.price.values, d['size'].values.astype(float)
isT = (act=='T') & ((sd=='B')|(sd=='A'))
sign = np.where(sd=='B', 1.0, -1.0)            # +1 buy aggressor lifts ask
ti = np.flatnonzero(isT)
print(f"trades: {len(ti):,}  shares: {sz[ti].sum():,.0f}  notional ${(sz[ti]*px[ti]).sum()/1e6:,.0f}M")

H = [('0ms',0),('1ms',10**6),('10ms',10**7),('100ms',10**8),('1s',10**9),('10s',10**10),('60s',6*10**10)]
print("\n=== MARKET-MAKER markout P&L per share (bps of price), passive fill ===")
print("  (= half-spread captured minus adverse selection; needs to stay >0)")
print(f"{'horizon':>8}" + f"{'all':>10}{'spr=1c':>10}{'spr>=2c':>10}")
s1 = np.round(spr[ti]*100).astype(int)==1
for name,h in H:
    j = np.minimum(np.searchsorted(ts, ts[ti]+h, 'left'), len(ts)-1)
    pnl = sign[ti]*(px[ti]-mid[j])/mid[ti]*1e4
    print(f"{name:>8}{pnl.mean():>10.4f}{pnl[s1].mean():>10.4f}{pnl[~s1].mean():>10.4f}")

print("\n=== AGGRESSOR (liquidity-taker) markout, bps — is following flow profitable? ===")
for name,h in H[3:]:
    j = np.minimum(np.searchsorted(ts, ts[ti]+h, 'left'), len(ts)-1)
    pnl = sign[ti]*(mid[j]-px[ti])/mid[ti]*1e4
    print(f"{name:>8}: {pnl.mean():+.4f} bps   (needs > 0 to beat the spread already paid)")

print("\n=== Non-overlapping fixed-grid return autocorrelation (true mean-reversion) ===")
for lbl,step in [('1s',10**9),('10s',10**10),('60s',6*10**10),('300s',3*10**11)]:
    g = np.arange(ts[0], ts[-1], step)
    k = np.minimum(np.searchsorted(ts, g, 'right')-1, len(ts)-1)
    r = np.diff(np.log(mid[k]))*1e4
    r = r[np.isfinite(r)]
    print(f"  {lbl:>5}: n={len(r):>6}  AC1 {np.corrcoef(r[:-1],r[1:])[0,1]:+.4f}  vol {np.std(r):7.3f} bps")

print("\n=== Queue-imbalance edge conditioned on spread state ===")
qi = (bs-as_)/(bs+as_)
j = np.minimum(np.searchsorted(ts, ts+10**9,'left'), len(ts)-1)
f1 = (mid[j]-mid)/mid*1e4
w1 = np.round(spr*100).astype(int)==1
for lbl,msk in [('spread=1c',w1),('spread>=2c',~w1)]:
    q = pd.qcut(pd.Series(qi[msk]).rank(method='first'),10,labels=False)
    mm = pd.Series(f1[msk]).groupby(q).mean()
    print(f"  {lbl:<11} n={msk.sum():>8,}  half-spr {np.mean(spr[msk]/2/mid[msk])*1e4:.3f} bps"
          f"  | D0 {mm.iloc[0]:+.3f}  D9 {mm.iloc[-1]:+.3f}  edge/side {(mm.iloc[-1]-mm.iloc[0])/2:.3f} bps")
