# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
# Streams one day at a time out of the zip: the archive is larger than most free disks.
import os
import databento as db, pandas as pd, numpy as np, os, zipfile
from concurrent.futures import ProcessPoolExecutor
np.seterr(all='ignore')
ZIP = os.environ.get("QQQ_DBN_ZIP",
    os.path.expanduser("~/Downloads/XNAS-20251009-UVUB86RLRM.zip"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); WORK=SP+'/work3'; os.makedirs(WORK,exist_ok=True)
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
        mid=(bp+ap)/2.0; spr=ap-bp; qi=(bs-as_)/(bs+as_)
        sd=d.side.values; ti=np.flatnonzero((d.action.values=='T')&((sd=='B')|(sd=='A')))
        ti=ti[ti>0]
        if len(ti)<1000: return None
        sg=np.where(sd=='B',1.0,-1.0)[ti]; px=d.price.values[ti]; sz=d['size'].values.astype(float)[ti]
        pre=ti-1                                   # book state BEFORE the trade (no lookahead)
        fav=-sg*qi[pre]                            # >0 = imbalance favours the passive side we filled
        out={}
        for nm,h in [('1s',10**9),('10s',10**10)]:
            j=np.minimum(np.searchsorted(ts,ts[ti]+h,'left'),len(ts)-1)
            out['mo_'+nm]=sg*(px-mid[j])/mid[ti]*1e4
        return dict(day=day, fav=fav, sz=sz, spr=np.round(spr[pre]*100).astype(int),
                    px=np.mean(mid), **out)
    except Exception as ex: return None
    finally:
        if os.path.exists(tmp): os.remove(tmp)
if __name__=='__main__':
    with zipfile.ZipFile(ZIP) as z: names=sorted(n for n in z.namelist() if n.endswith('.dbn.zst'))
    sel=names[::26]
    rs=[r for r in ProcessPoolExecutor(max_workers=5).map(one,sel) if r]
    print(f"days={len(rs)}  fills={sum(len(r['fav']) for r in rs):,}")
    fav=np.concatenate([r['fav'] for r in rs]); sz=np.concatenate([r['sz'] for r in rs])
    spr=np.concatenate([r['spr'] for r in rs]); PX=np.mean([r['px'] for r in rs])
    mo={n:np.concatenate([r['mo_'+n] for r in rs]) for n in ('1s','10s')}
    reb=0.00305/PX*1e4
    print(f"avg px ${PX:.0f}   add-rebate {reb:+.4f} bps\n")
    print("=== Passive markout by favourable-imbalance decile (size-weighted, bps) ===")
    print(f"{'decile':>7}{'fav mid':>9}{'mo_1s':>9}{'+rebate':>9}{'mo_10s':>9}{'share%':>8}")
    q=pd.qcut(pd.Series(fav).rank(method='first'),10,labels=False).values
    for k in range(10):
        m=q==k; w=sz[m]
        print(f"{k:>7}{np.median(fav[m]):>9.3f}{np.average(mo['1s'][m],weights=w):>9.4f}"
              f"{np.average(mo['1s'][m],weights=w)+reb:>9.4f}"
              f"{np.average(mo['10s'][m],weights=w):>9.4f}{w.sum()/sz.sum()*100:>8.1f}")
    print("\n=== Selective quoting: only quote when fav > threshold ===")
    print(f"{'thresh':>8}{'%fills':>8}{'mo_1s':>9}{'net+reb':>9}{'mo_10s':>9}{'net10s':>9}")
    for th in (-1.0,0.0,0.2,0.4,0.6,0.8):
        m=fav>th; w=sz[m]
        if w.sum()==0: continue
        a=np.average(mo['1s'][m],weights=w); b=np.average(mo['10s'][m],weights=w)
        print(f"{th:>8.1f}{w.sum()/sz.sum()*100:>8.1f}{a:>9.4f}{a+reb:>9.4f}{b:>9.4f}{b+reb:>9.4f}")
    print("\n=== Same, split by spread state at quote time ===")
    for lbl,m0 in [('spread=1c',spr==1),('spread>=2c',spr>=2)]:
        for th in (-1.0,0.4,0.8):
            m=m0&(fav>th); w=sz[m]
            if w.sum()<1000: continue
            a=np.average(mo['1s'][m],weights=w)
            print(f"  {lbl:<11} fav>{th:>4.1f}: {w.sum()/sz.sum()*100:>5.1f}% of vol"
                  f"  mo_1s {a:+.4f}  net+reb {a+reb:+.4f} bps"
                  f"  = ${(a+reb)/1e4*PX*100:+.4f}/100sh")
