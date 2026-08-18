# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
# Latency-reachability test: how long >=2-tick spreads persist, and whether the
# selective-quoting edge survives requiring the state to pre-date your quote.
import databento as db, pandas as pd, numpy as np, os, zipfile
from concurrent.futures import ProcessPoolExecutor
np.seterr(all='ignore')
ZIP = os.environ.get("QQQ_DBN_ZIP",
    os.path.expanduser("~/Downloads/XNAS-20251009-UVUB86RLRM.zip"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = ROOT+'/data/_work_reach'; os.makedirs(WORK,exist_ok=True)
def one(name):
    day=name.split('-')[2].split('.')[0]; tmp=f'{WORK}/{day}.zst'
    try:
        with zipfile.ZipFile(ZIP) as z, open(tmp,'wb') as fo: fo.write(z.read(name))
        d=db.DBNStore.from_file(tmp).to_df().reset_index()
        d=d[(d['flags'].values.astype(np.uint8)&128)!=0]
        ts=d.ts_recv.values.astype('datetime64[ns]').astype(np.int64)
        d0=(ts[0]//86_400_000_000_000)*86_400_000_000_000
        for off in (13,14):
            s=d0+(off*3600+1800)*10**9; m=(ts>=s)&(ts<s+390*60*10**9)
            if m.sum()>100_000: break
        d,ts=d[m],ts[m]
        bp,ap=d.bid_px_00.values,d.ask_px_00.values
        bs,as_=d.bid_sz_00.values.astype(float),d.ask_sz_00.values.astype(float)
        ok=np.isfinite(bp)&np.isfinite(ap)&(ap>bp)&(bs>0)&(as_>0)
        d,ts,bp,ap,bs,as_=d[ok],ts[ok],bp[ok],ap[ok],bs[ok],as_[ok]
        mid=(bp+ap)/2.0; tick=np.round((ap-bp)*100).astype(int); qi=(bs-as_)/(bs+as_)
        n=len(ts); w=tick>=2
        # age of the current wide-spread run at every event
        prev=np.concatenate([[False],w[:-1]]); rs=w&~prev
        idx=np.where(rs,np.arange(n),0); start=np.maximum.accumulate(idx)
        age=np.where(w, ts-ts[start], -1)
        # duration of each completed wide run (for the distribution)
        si=np.flatnonzero(rs); ends=np.concatenate([si[1:],[n]])
        durs=[]
        for a,b in zip(si,ends):
            e=a+np.argmin(w[a:b]) if (~w[a:b]).any() else b-1
            durs.append(ts[min(e,n-1)]-ts[a])
        sd=d.side.values; ti=np.flatnonzero((d.action.values=='T')&((sd=='B')|(sd=='A')))
        ti=ti[ti>0]
        sg=np.where(sd=='B',1.0,-1.0)[ti]; px=d.price.values[ti]; sz=d['size'].values.astype(float)[ti]
        j=np.minimum(np.searchsorted(ts,ts[ti]+10**9,'left'),n-1)
        mo=sg*(px-mid[j])/mid[ti]*1e4
        pre=ti-1
        return dict(day=day, durs=np.array(durs,dtype=np.int64),
                    wide_frac_time=float(np.mean(w)), px=float(np.mean(mid)),
                    fav=-sg*qi[pre], age=age[pre], tick=tick[pre], mo=mo, sz=sz)
    except Exception: return None
    finally:
        if os.path.exists(tmp): os.remove(tmp)
if __name__=='__main__':
    with zipfile.ZipFile(ZIP) as z: names=sorted(n for n in z.namelist() if n.endswith('.dbn.zst'))
    rs=[r for r in ProcessPoolExecutor(max_workers=5).map(one,names[::43]) if r]
    durs=np.concatenate([r['durs'] for r in rs])
    PX=np.mean([r['px'] for r in rs]); reb=0.00305/PX*1e4
    print(f"days={len(rs)}  wide runs={len(durs):,}  spread>=2t is {np.mean([r['wide_frac_time'] for r in rs])*100:.1f}% of book time\n")
    print("=== How long does a >=2-tick Nasdaq spread last? ===")
    for p in (25,50,75,90,95,99):
        v=np.percentile(durs,p)
        print(f"  p{p:<3}: {v/1e6:>10.3f} ms")
    for th,lab in [(10**6,'1 ms'),(10**7,'10 ms'),(10**8,'100 ms'),(10**9,'1 s')]:
        print(f"  runs lasting > {lab:<6}: {np.mean(durs>th)*100:>5.1f}%")
    fav=np.concatenate([r['fav'] for r in rs]); age=np.concatenate([r['age'] for r in rs])
    tick=np.concatenate([r['tick'] for r in rs]); mo=np.concatenate([r['mo'] for r in rs])
    sz=np.concatenate([r['sz'] for r in rs])
    base=(tick>=2)&(fav>0.4)
    print(f"\n=== The headline edge, now requiring the wide state to have EXISTED for >= L ===")
    print("    (L = how long your quote needed to already be resting there)")
    print(f"{'L':>8}{'% of all vol':>14}{'gross mo_1s':>13}{'net+rebate':>12}{'per 100sh':>12}")
    for L,lab in [(0,'0 (orig)'),(10**5,'0.1 ms'),(10**6,'1 ms'),(5*10**6,'5 ms'),
                  (10**7,'10 ms'),(5*10**7,'50 ms'),(10**8,'100 ms'),(10**9,'1 s')]:
        m=base&(age>=L); w=sz[m]
        if w.sum()<500: print(f"{lab:>8}{w.sum()/sz.sum()*100:>13.2f}%{'  too few fills':>25}"); continue
        a=np.average(mo[m],weights=w)
        print(f"{lab:>8}{w.sum()/sz.sum()*100:>13.2f}%{a:>13.4f}{a+reb:>12.4f}{(a+reb)/1e4*PX*100:>+12.4f}")
