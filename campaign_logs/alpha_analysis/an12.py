import json, math, statistics as st
rs=json.load(open("/tmp/xse/runs.json"))
H=json.load(open("/tmp/xse/hist2.json")); H1=json.load(open("/tmp/xse/hist.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
def rw(r):  return H.get(r["name"]+"|"+r["id"], [])
def pairs(rows,k): return [(x["_step"],x[k]) for x in rows if k in x]
def mean(x): return sum(x)/len(x) if x else float("nan")
NODP=[r for r in rs if g(r,"noise_multiplier")==0 and r["state"]=="finished"
      and sm(r,"_step")==260 and r["name"]+"|"+r["id"] in H]

print("="*104)
print("1.  IS EXPLORATION PRODUCTIVE EARLY AND UNPRODUCTIVE LATE?")
print("    promoted fraction = promotion_count / r_keep  = share of the RETAINED set that was")
print("    freshly discovered at that rotation.  (promotion_count is capped at r_keep, xse.py:588)")
print("="*104)
BINS=[(0,40),(40,80),(80,130),(130,180),(180,220),(220,261)]
sel=[r for r in NODP if r["name"] in (
  "med-nodp-fixed-re5-s42","med-nodp-fixed-re9-s42","med-nodp-fixed-re13-s42",
  "renyi-ad-nodp-ainf-m1-s42","renyi-ad-nodp-a1-m2-s42","renyi-nodp-s42","m0-nodp-a2-m0-s42")]
print(f'{"run":30s}{"depth":>6s}  ' + "".join(f'{f"{a}-{b}":>10s}' for a,b in BINS) + f'{"trend":>9s}')
for r in sorted(sel, key=lambda x: mean([v for _,v in pairs(rw(x),"rotation/r_e_dyn")])):
    rows=rw(r)
    pc=dict(pairs(rows,"rotation/promotion_count")); dd=dict(pairs(rows,"rotation/r_e_dyn"))
    if not pc: continue
    d=mean(list(dd.values()))
    cells=[]; vals=[]
    for a,b in BINS:
        v=[pc[s]/max(1e-9,16-dd[s]) for s in pc if a<=s<b and s in dd]
        cells.append(mean(v) if v else float("nan")); vals.append(cells[-1])
    first,last=vals[0],vals[-1]
    tr = "falls" if last<first*0.8 else ("rises" if last>first*1.25 else "flat")
    print(f'{r["name"][:29]:30s}{d:6.1f}  ' + "".join(f'{c:10.4f}' for c in cells) + f'{tr:>9s}')

print()
print("="*104)
print("2.  WHAT DOES THE CURRENT RULE DO OVER TIME?  (does it explore MORE or LESS late?)")
print("="*104)
print(f'{"run":30s}{"alpha":>7s}{"m":>3s}  ' + "".join(f'{f"{a}-{b}":>9s}' for a,b in BINS) + f'{"direction":>12s}')
for r in sorted(NODP, key=lambda x:x["name"]):
    rows=rw(r); dd=dict(pairs(rows,"rotation/r_e_dyn"))
    if not dd or max(dd.values())==min(dd.values()): continue   # skip fixed-depth
    cells=[]
    for a,b in BINS:
        v=[dd[s] for s in dd if a<=s<b]
        cells.append(mean(v) if v else float("nan"))
    mg=None
    for tag,val in {"m1":1,"m2":2,"m3":3,"m0":0}.items():
        if "-"+tag+"-" in r["name"]: mg=val
    dirn = "explores MORE" if cells[-1]>cells[0]+0.05 else ("explores LESS" if cells[-1]<cells[0]-0.05 else "flat")
    print(f'{r["name"][:29]:30s}{str(sm(r,"rotation/alpha")):>7s}{str(mg):>3s}  '
          + "".join(f'{c:9.2f}' for c in cells) + f'{dirn:>12s}')

print()
print("="*104)
print("3.  THE ONE TIME-VARYING-SCHEDULE DATA POINT WE ALREADY HAVE")
print("="*104)
CURVE={1:0.34700,5:0.34429,9:0.34368,13:0.34354,15:0.34366}   # 15 from the two m=0 depth-15 runs
def interp(d):
    ks=sorted(CURVE)
    if d<=ks[0]: return CURVE[ks[0]]
    if d>=ks[-1]: return CURVE[ks[-1]]
    for i in range(len(ks)-1):
        if ks[i]<=d<=ks[i+1]:
            f=(d-ks[i])/(ks[i+1]-ks[i]); return CURVE[ks[i]]+f*(CURVE[ks[i+1]]-CURVE[ks[i]])
for nm in ["m0-nodp-a025-m0-s42","m0-nodp-a01-m0-s42","lowa-nodp-a01-m2-s42","lowa-nodp-a015-m2-s42"]:
    for r in NODP:
        if r["name"]!=nm: continue
        rows=rw(r); d=[v for _,v in pairs(rows,"rotation/r_e_dyn")]
        dm=mean(d); l=sm(r,"eval/loss_min")
        print(f'  {nm:26s} depth {d[0]:5.2f} -> {d[-1]:5.2f} (mean {dm:5.2f}, RISING schedule)')
        print(f'  {"":26s} loss_min {l:.5f}   curve at mean depth {interp(dm):.5f}'
              f'   residual {l-interp(dm):+.2e}   ({abs(l-interp(dm))/2.8e-4:.1f}x the deep floor)')

print()
print("="*104)
print("4.  IS THERE A ROTATION AT THE VERY LAST STEP?  (cool-down question)")
print("="*104)
print("  rotation fires when  new_step % interval == 0   (xse.py:725)")
print("  interval = None -> max(1, round(0.5/(1-momentum))) = round(5) = 5  at momentum 0.9")
for tot in (260,):
    for iv in (3,5):
        last=(tot//iv)*iv
        print(f'  total steps {tot}, interval {iv}: last rotation at step {last} '
              f'({tot-last} steps of training left afterwards)')
print()
print("  => with interval 5 and 260 steps the LAST rotation lands exactly ON the final step.")
print("     r_e of the 16 directions are then freshly random with essentially no training after.")
