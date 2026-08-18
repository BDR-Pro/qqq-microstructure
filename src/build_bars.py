# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
# Streams one day at a time out of the zip: the archive is larger than most free disks.
import os
import databento as db, pandas as pd, numpy as np, sys, os, subprocess, zipfile
from concurrent.futures import ProcessPoolExecutor
np.seterr(all='ignore')
ZIP = os.environ.get("QQQ_DBN_ZIP",
    os.path.expanduser("~/Downloads/XNAS-20251009-UVUB86RLRM.zip"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK, OUT = os.path.join(ROOT,'data','_work_bars'), os.path.join(ROOT,'data','bars')
os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
BAR = 60*10**9

def one(name):
    day = name.split('-')[2].split('.')[0]
    out = os.path.join(OUT, f'{day}.parquet')
    if os.path.exists(out): return day, 'cached'
    tmp = os.path.join(WORK, f'{day}.zst')
    try:
        with zipfile.ZipFile(ZIP) as z, open(tmp,'wb') as fo: fo.write(z.read(name))
        df = db.DBNStore.from_file(tmp).to_df().reset_index()
        df = df[(df['flags'].values.astype(np.uint8) & 128) != 0]
        ts = df.ts_recv.values.astype('datetime64[ns]').astype(np.int64)
        d0 = (ts[0]//86_400_000_000_000)*86_400_000_000_000
        # RTH in UTC: EDT 13:30-20:00, EST 14:30-21:00 -> detect via activity
        for off in (13,14):
            s_ns = d0 + (off*3600 + 30*60)*10**9          # 09:30 ET
            e_ns = s_ns + 390*60*10**9                     # full 6.5h RTH -> 16:00 ET
            m = (ts>=s_ns)&(ts<e_ns)
            if m.sum() > 100_000: break
        d, ts = df[m], ts[m]
        bp,ap = d.bid_px_00.values, d.ask_px_00.values
        bs,as_= d.bid_sz_00.values.astype(float), d.ask_sz_00.values.astype(float)
        ok = np.isfinite(bp)&np.isfinite(ap)&(ap>bp)&(bs>0)&(as_>0)
        d,ts,bp,ap,bs,as_ = d[ok],ts[ok],bp[ok],ap[ok],bs[ok],as_[ok]
        if len(ts) < 10_000: return day,'thin'
        mid,spr = (bp+ap)/2.0, ap-bp
        qi = (bs-as_)/(bs+as_)
        dbp,dap = np.diff(bp,prepend=bp[0]), np.diff(ap,prepend=ap[0])
        pbs,pas = np.roll(bs,1),np.roll(as_,1); pbs[0],pas[0]=bs[0],as_[0]
        ofi = (np.where(dbp>0,bs,np.where(dbp==0,bs-pbs,0))
              -np.where(dap<0,as_,np.where(dap==0,as_-pas,0)))
        isT = (d.action.values=='T')
        sgn = np.where(d.side.values=='B',1.0,-1.0)
        tsz = np.where(isT, d['size'].values.astype(float), 0.0)
        tpx = np.where(isT, d.price.values, 0.0)
        b = (ts - s_ns) // BAR
        g = pd.DataFrame({'b':b,'mid':mid,'spr':spr,'qi':qi,'ofi':ofi,
                          'tv':tsz,'stv':sgn*tsz,'notional':tsz*tpx,
                          'ntr':isT.astype(float),'bs':bs,'as':as_,'lr':np.log(mid)})
        a = g.groupby('b').agg(
            n=('mid','size'), open=('mid','first'), close=('mid','last'),
            hi=('mid','max'), lo=('mid','min'), mid_mean=('mid','mean'),
            spr_mean=('spr','mean'), qi_mean=('qi','mean'), ofi=('ofi','sum'),
            vol=('tv','sum'), svol=('stv','sum'), notional=('notional','sum'),
            ntrades=('ntr','sum'), bsz=('bs','mean'), asz=('as','mean'),
            rv=('lr', lambda x: np.std(np.diff(x))*1e4 if len(x)>2 else np.nan)).reset_index()
        a['day']=day; a.to_parquet(out, index=False)
        return day, f'{len(a)} bars'
    except Exception as ex:
        return day, f'ERR {type(ex).__name__}: {ex}'
    finally:
        if os.path.exists(tmp): os.remove(tmp)

if __name__=='__main__':
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.dbn.zst'))
    step = int(sys.argv[1]) if len(sys.argv)>1 else 8
    sel = names[::step]
    print(f'{len(names)} days total -> processing {len(sel)}', flush=True)
    with ProcessPoolExecutor(max_workers=5) as ex:
        for i,(day,st) in enumerate(ex.map(one, sel),1):
            print(f'[{i}/{len(sel)}] {day} {st}', flush=True)
