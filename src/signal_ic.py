# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
# Streams one day at a time out of the zip: the archive is larger than most free disks.
import os
import databento as db, pandas as pd, numpy as np, sys
np.seterr(all='ignore')
f = sys.argv[1]
df = db.DBNStore.from_file(f).to_df()
df = df.reset_index()
# F_LAST bit = 128 -> final message of an event packet (consistent book state)
df = df[(df['flags'].values.astype(np.uint8) & 128) != 0]
ts = df.ts_recv.values.astype('datetime64[ns]').astype(np.int64)
# RTH: 13:30-20:00 UTC (EDT). derive day start
day0 = (ts[0] // 86_400_000_000_000) * 86_400_000_000_000
rth_s, rth_e = day0 + 13*3600*10**9 + 30*60*10**9, day0 + 20*3600*10**9
m = (ts >= rth_s) & (ts < rth_e)
d = df[m]
ts = ts[m]
bp, ap = d.bid_px_00.values, d.ask_px_00.values
bs, as_ = d.bid_sz_00.values.astype(float), d.ask_sz_00.values.astype(float)
ok = np.isfinite(bp) & np.isfinite(ap) & (ap > bp) & (bs>0) & (as_>0)
d, ts, bp, ap, bs, as_ = d[ok], ts[ok], bp[ok], ap[ok], bs[ok], as_[ok]
mid = (bp+ap)/2.0; spr = ap-bp
print(f"=== {f.split('/')[-1]} | RTH book updates: {len(ts):,}")
print(f"updates/sec: {len(ts)/23400:,.0f}   trades: {(d.action=='T').sum():,}")
print(f"spread ticks: " + "  ".join(f"{k}c:{v/len(spr)*100:.1f}%" for k,v in
      sorted(pd.Series(np.round(spr*100).astype(int)).value_counts().head(4).items())))
print(f"half-spread bps: mean {np.mean(spr/2/mid)*1e4:.3f}  median {np.median(spr/2/mid)*1e4:.3f}")
print(f"top-of-book size: bid med {np.median(bs):.0f} ask med {np.median(as_):.0f} shares")

# ---- signals ----
qi = (bs - as_) / (bs + as_)                       # queue imbalance
# Cont-Kukanov-Stoikov order flow imbalance
dbp, dap = np.diff(bp, prepend=bp[0]), np.diff(ap, prepend=ap[0])
pbs, pas = np.roll(bs,1), np.roll(as_,1); pbs[0], pas[0] = bs[0], as_[0]
e = (np.where(dbp>0, bs, np.where(dbp==0, bs-pbs, 0))
     - np.where(dap<0, as_, np.where(dap==0, as_-pas, 0)))
def roll_sum(x, ts, win_ns):
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.searchsorted(ts, ts - win_ns, 'left')
    return c[np.arange(len(x))+1] - c[lo]
ofi1 = roll_sum(e, ts, 1_000_000_000)              # 1s OFI
ofi10 = roll_sum(e, ts, 10_000_000_000)            # 10s OFI
# trade-flow imbalance (signed volume, 1s)
tsz = np.where(d.action.values=='T', d['size'].values.astype(float), 0.0)
tsign = np.where(d.side.values=='B', 1.0, -1.0) * tsz
tfi1 = roll_sum(tsign, ts, 1_000_000_000)

def fwd(h_ns):
    j = np.searchsorted(ts, ts + h_ns, 'left'); j = np.minimum(j, len(ts)-1)
    return (mid[j]-mid)/mid*1e4                    # bps

sigs = {'QueueImb':qi, 'OFI_1s':ofi1, 'OFI_10s':ofi10, 'TradeFlow_1s':tfi1}
hors = [('100ms',10**8), ('1s',10**9), ('10s',10**10), ('60s',6*10**10)]
print("\n=== IC (Pearson corr of signal vs forward mid-return, bps) ===")
print(f"{'signal':<14}" + "".join(f"{h:>10}" for h,_ in hors))
F = {h: fwd(n) for h,n in hors}
for name,s in sigs.items():
    row = "".join(f"{np.corrcoef(s, F[h])[0,1]:>10.4f}" for h,_ in hors)
    print(f"{name:<14}{row}")

print("\n=== Decile edge: mean forward return (bps) by signal decile ===")
hs = np.mean(spr/2/mid)*1e4
for name in ['QueueImb','OFI_1s']:
    s = sigs[name]
    q = pd.qcut(pd.Series(s).rank(method='first'), 10, labels=False)
    print(f"\n{name}  (half-spread = {hs:.3f} bps)")
    for h,_ in hors[:3]:
        mm = pd.Series(F[h]).groupby(q).mean()
        print(f"  {h:>6}: D0 {mm.iloc[0]:+.3f}  D9 {mm.iloc[-1]:+.3f}  |spread D9-D0| {mm.iloc[-1]-mm.iloc[0]:.3f} bps")

print("\n=== Mid-return autocorrelation (mean reversion check) ===")
for h,n in hors:
    r = fwd(n); k = max(1, int(n/10**8))
    print(f"  {h:>6}: AC(1 step) {np.corrcoef(r[:-k], r[k:])[0,1]:+.4f}   vol {np.std(r):.3f} bps")
