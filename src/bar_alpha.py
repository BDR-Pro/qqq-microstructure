# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
# Streams one day at a time out of the zip: the archive is larger than most free disks.
import os
import pandas as pd, numpy as np, glob, os
np.seterr(all='ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(ROOT+'/data/bars/*.parquet'))], ignore_index=True)
df = df.sort_values(['day','b']).reset_index(drop=True)
print(f"bars {len(df):,} over {df.day.nunique()} days  ({df.day.min()} .. {df.day.max()})")
px = df.close.values
print(f"QQQ mid range ${np.nanmin(px):.2f}-${np.nanmax(px):.2f} | avg spread {df.spr_mean.mean()*100:.2f}c "
      f"= {df.spr_mean.mean()/df.close.mean()*1e4:.3f} bps | Nasdaq notional/day ${df.groupby('day').notional.sum().mean()/1e9:.2f}B")

g = df.groupby('day')
df['ret'] = g.close.transform(lambda s: np.log(s).diff()*1e4)
for h in (5,15,30,60):
    df[f'f{h}'] = g.close.transform(lambda s,h=h: (np.log(s.shift(-h))-np.log(s))*1e4)
# features (z-scored within day to remove level drift)
df['ofi_n']  = df.ofi/(df.bsz+df.asz)
df['svol_n'] = df.svol/df.vol.replace(0,np.nan)
df['spr_bps']= df.spr_mean/df.close*1e4
df['vimb']   = (df.bsz-df.asz)/(df.bsz+df.asz)
feats = ['qi_mean','ofi_n','svol_n','ret','rv','spr_bps','vimb']
for f in feats:
    df['z_'+f] = g[f].transform(lambda s: (s-s.mean())/s.std())
zf = ['z_'+f for f in feats]

print("\n=== IC: feature vs forward return (all 65 days) ===")
print(f"{'feature':<10}" + "".join(f"{'f'+str(h):>9}" for h in (5,15,30,60)))
for f in zf:
    print(f"{f[2:]:<10}" + "".join(
        f"{df[[f,f'f{h}']].dropna().corr().iloc[0,1]:>9.4f}" for h in (5,15,30,60)))

days = sorted(df.day.unique()); cut = days[int(len(days)*0.6)]
print(f"\n=== Out-of-sample stability (train < {cut} <= test) ===")
print(f"{'feature':<10}{'IS f30':>9}{'OOS f30':>9}{'IS f5':>9}{'OOS f5':>9}")
for f in zf:
    r=[]
    for h in (30,5):
        for msk in (df.day<cut, df.day>=cut):
            s=df.loc[msk,[f,f'f{h}']].dropna(); r.append(s.corr().iloc[0,1])
    print(f"{f[2:]:<10}{r[0]:>9.4f}{r[1]:>9.4f}{r[2]:>9.4f}{r[3]:>9.4f}")

print("\n=== Decile edge, best features (bps forward return) ===")
RT = 2*df.spr_mean.mean()/df.close.mean()*1e4/2   # round-trip = 1 full spread if crossing both ways
print(f"round-trip cost if crossing spread both sides ~ {2*(df.spr_mean.mean()/2/df.close.mean()*1e4):.3f} bps")
for f in ['z_qi_mean','z_ret','z_ofi_n','z_svol_n']:
    for h in (5,30):
        s = df[[f,f'f{h}']].dropna()
        q = pd.qcut(s[f].rank(method='first'),10,labels=False)
        mm = s[f'f{h}'].groupby(q).mean()
        print(f"  {f[2:]:<9} f{h:<3}: D0 {mm.iloc[0]:+.3f}  D9 {mm.iloc[-1]:+.3f}  L/S gross {(mm.iloc[-1]-mm.iloc[0]):.3f} bps")

print("\n=== Intraday seasonality (robust, non-predictive-but-actionable) ===")
tod = df.groupby(df.b//30).agg(spr=('spr_bps','mean'), vol=('vol','mean'),
                               rv=('rv','mean'), absret=('ret', lambda s: s.abs().mean()))
tod.index = [f"{9+ (i*30+30)//60}:{(i*30+30)%60:02d}" for i in tod.index]
print(tod.round(3).to_string())
