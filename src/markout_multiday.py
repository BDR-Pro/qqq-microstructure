# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
# Streams one day at a time out of the zip: the archive is larger than most free disks.
import os
import databento as db, pandas as pd, numpy as np, os, zipfile
from concurrent.futures import ProcessPoolExecutor
np.seterr(all='ignore')

def session_start_ns(first_ts_ns):
    """UTC nanoseconds of 09:30 America/New_York on the day containing `first_ts_ns`.

    Deriving this from event counts is unsafe and was wrong here for 169 of 515 days.
    Under EST a 13:30-20:00 UTC window still clears any activity threshold while
    actually spanning 08:30-15:00 ET -- silently trading an hour of pre-market for the
    closing hour. Compute it from the calendar instead.
    """
    d = dt.datetime.fromtimestamp(first_ts_ns / 1e9, dt.timezone.utc).date()
    et = dt.datetime.combine(d, dt.time(9, 30), ZoneInfo('America/New_York'))
    return int(et.astimezone(dt.timezone.utc).timestamp()) * 10**9


ZIP = os.environ.get("QQQ_DBN_ZIP",
    os.path.expanduser("~/Downloads/XNAS-20251009-UVUB86RLRM.zip"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); WORK=SP+'/work2'; os.makedirs(WORK,exist_ok=True)
H=[('fill',0),('1ms',10**6),('10ms',10**7),('100ms',10**8),('1s',10**9),('10s',10**10),('60s',6*10**10)]
def one(name):
    day=name.split('-')[2].split('.')[0]; tmp=f'{WORK}/{day}.zst'
    try:
        with zipfile.ZipFile(ZIP) as z, open(tmp,'wb') as fo: fo.write(z.read(name))
        d=db.DBNStore.from_file(tmp).to_df().reset_index()
        d=d[(d['flags'].values.astype(np.uint8)&128)!=0]
        ts=d.ts_recv.values.astype('datetime64[ns]').astype(np.int64)
        d0=(ts[0]//86_400_000_000_000)*86_400_000_000_000
        s_ns = session_start_ns(ts[0])
        m = (ts >= s_ns) & (ts < s_ns + 390*60*10**9)
        d,ts=d[m],ts[m]
        bp,ap=d.bid_px_00.values,d.ask_px_00.values
        bs,as_=d.bid_sz_00.values.astype(float),d.ask_sz_00.values.astype(float)
        ok=np.isfinite(bp)&np.isfinite(ap)&(ap>bp)&(bs>0)&(as_>0)
        d,ts,bp,ap=d[ok],ts[ok],bp[ok],ap[ok]
        mid=(bp+ap)/2.0; spr=ap-bp
        sd=d.side.values; ti=np.flatnonzero((d.action.values=='T')&((sd=='B')|(sd=='A')))
        if len(ti)<1000: return None
        sg=np.where(sd=='B',1.0,-1.0)[ti]; px=d.price.values[ti]; sz=d['size'].values.astype(float)[ti]
        r={'day':day,'trades':len(ti),'shares':sz.sum(),'px':np.mean(mid),
           'spr_bps':np.mean(spr/mid)*1e4}
        for nm,h in H:
            j=np.minimum(np.searchsorted(ts,ts[ti]+h,'left'),len(ts)-1)
            r['mm_'+nm]=np.average(sg*(px-mid[j])/mid[ti]*1e4, weights=sz)  # size-weighted
        return r
    except Exception as ex: return {'day':day,'err':str(ex)}
    finally:
        if os.path.exists(tmp): os.remove(tmp)
if __name__=='__main__':
    with zipfile.ZipFile(ZIP) as z: names=sorted(n for n in z.namelist() if n.endswith('.dbn.zst'))
    sel=names[::22]
    print(f'markouts on {len(sel)} days',flush=True)
    rs=[r for r in ProcessPoolExecutor(max_workers=5).map(one,sel) if r and 'err' not in r]
    t=pd.DataFrame(rs)
    print(f"\ndays={len(t)}  avg trades/day {t.trades.mean():,.0f}  avg QQQ px ${t.px.mean():.0f}  avg spread {t.spr_bps.mean():.3f} bps")
    print("\n=== SIZE-WEIGHTED MARKET-MAKER MARKOUT (bps), per day ===")
    print(t[['day']+['mm_'+n for n,_ in H]].to_string(index=False, float_format=lambda x:f'{x:+.4f}'))
    print("\n=== MEAN across days (bps of notional) ===")
    for nm,_ in H:
        v=t['mm_'+nm]; print(f"  {nm:>6}: {v.mean():+.4f}  (std {v.std():.4f}, neg on {(v<0).sum()}/{len(v)} days)")
    reb=0.00305/t.px.mean()*1e4
    print(f"\nNasdaq top-tier ADD rebate $0.00305/sh = {reb:+.4f} bps at ${t.px.mean():.0f}")
    for nm in ('1s','10s'):
        print(f"  MM net @{nm} incl rebate: {t['mm_'+nm].mean()+reb:+.4f} bps"
              f"  -> ${(t['mm_'+nm].mean()+reb)/1e4*t.px.mean()*100:.4f} per 100 shares")
