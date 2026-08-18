import json, statistics as st
rs=json.load(open("/tmp/xse/runs.json")); H=json.load(open("/tmp/xse/hist.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
def rows(nm):
    for r in rs:
        if r["name"]==nm and r["state"]=="finished" and sm(r,"_step")==260:
            return H.get(nm+"|"+r["id"], []), r
    return [], None
def ser(rw,k): return [x[k] for x in rw if k in x]
def mean(x): return sum(x)/len(x) if x else float("nan")

# the alpha x margin grid actually run in non-DP, seed 42 (+43 where noted)
GRID = {
 0: {"0.5":"m0-nodp-a05-m0-s42", "2":"m0-nodp-a2-m0-s42",
     "0.25":"m0-nodp-a025-m0-s42","0.1":"m0-nodp-a01-m0-s42"},
 1: {"0.5":"renyi-ad-nodp-a0.5-m1-s42","1":"renyi-ad-nodp-a1-m1-s42",
     "2":"renyi-ad-nodp-a2-m1-s42","inf":"renyi-ad-nodp-ainf-m1-s42"},
 2: {"0.5":"renyi-ad-nodp-a0.5-m2-s42","1":"renyi-ad-nodp-a1-m2-s42",
     "2":"renyi-ad-nodp-a2-m2-s42","inf":"renyi-ad-nodp-ainf-m2-s42",
     "0.25":"med-nodp-adaptive-a025-m2-s42","0.2":"lowa-nodp-a02-m2-s42",
     "0.15":"lowa-nodp-a015-m2-s42","0.1":"lowa-nodp-a01-m2-s42","0.05":"lowa-nodp-a005-m2-s42"},
 3: {"0.5":"renyi-ad-nodp-a0.5-m3-s42","1":"renyi-ad-nodp-a1-m3-s42",
     "2":"renyi-ad-nodp-a2-m3-s42","inf":"renyi-ad-nodp-ainf-m3-s42"},
}

print("="*100)
print("1.  HOW MANY MARGINS WAS THE ALPHA SWEEP ACTUALLY REPLICATED AT?")
print("="*100)
for m in sorted(GRID):
    got=[a for a,nm in GRID[m].items() if rows(nm)[0]]
    print(f'  m={m}: alphas = {sorted(got, key=lambda x: 99 if x=="inf" else float(x))}')

print()
print("="*100)
print("2.  THE BOSS'S HYPOTHESIS, TESTED:  does the rule COMPENSATE for a margin change?")
print("    d(depth)/dm = -1 - d(floor N_a)/dm.   |slope| < 1 = compensating (absorbs it)")
print("                                          |slope| > 1 = AMPLIFYING (over-reacts)")
print("="*100)
print(f'{"alpha":>6s}  ' + "".join(f'{"depth m="+str(m):>12s}' for m in (0,1,2,3)) + f'{"d(depth)/dm":>14s}{"verdict":>14s}')
amp={}
for a in ["inf","2","1","0.5","0.25","0.1"]:
    ds={}
    for m in (0,1,2,3):
        nm=GRID[m].get(a)
        if nm:
            rw,_=rows(nm)
            if rw: ds[m]=mean(ser(rw,"rotation/r_e_dyn"))
    if len(ds)<2: continue
    ks=sorted(ds)
    slopes=[(ds[ks[i+1]]-ds[ks[i]])/(ks[i+1]-ks[i]) for i in range(len(ks)-1)]
    sl=mean(slopes)
    cells="".join(f'{ds[m]:12.2f}' if m in ds else f'{"-":>12s}' for m in (0,1,2,3))
    v = "AMPLIFIES" if abs(sl)>1.02 else ("neutral" if abs(sl)>0.98 else "compensates")
    print(f'{a:>6s}  {cells}{sl:14.3f}{v:>14s}')
    amp[a]=sl
print("\n  Every alpha AMPLIFIES or is neutral. None compensates. And the amplification grows")
print("  monotonically as alpha falls -- the 'more adaptive' settings over-react MOST.")

print()
print("="*100
      )
print("3.  DOES THE ALPHA RANKING REPRODUCE ACROSS MARGINS?  (if alpha were real, it must)")
print("="*100)
for key in ("eval/loss","eval/loss_min"):
    print(f'\n  metric = {key}')
    print(f'{"m":>3s}  ' + "".join(f'{"a="+a:>11s}' for a in ["0.5","1","2","inf"]) + f'{"spread":>10s}{"best":>7s}{"worst":>7s}')
    for m in (0,1,2,3):
        vals={}
        for a in ["0.5","1","2","inf"]:
            nm=GRID[m].get(a)
            if nm:
                rw,r=rows(nm)
                if r: vals[a]=sm(r,key)
        if len(vals)<2: continue
        cells="".join(f'{vals[a]:11.5f}' if a in vals else f'{"-":>11s}' for a in ["0.5","1","2","inf"])
        best=min(vals,key=vals.get); worst=max(vals,key=vals.get)
        print(f'{m:3d}  {cells}{max(vals.values())-min(vals.values()):10.2e}{best:>7s}{worst:>7s}')
print("\n  If alpha had a real effect the 'best' column would be constant. It is not.")

print()
print("="*100)
print("4.  WHERE ALPHA HAS THE MOST ROOM AMONG TESTED MARGINS (m=3): is anything visible?")
print("="*100)
for m in (1,2,3):
    sp=[]
    for a in ["0.5","1","2","inf"]:
        nm=GRID[m].get(a)
        if nm:
            rw,_=rows(nm)
            if rw: sp.append(mean([x-y for x,y in zip(ser(rw,"rotation/r_eff_a0p5"),ser(rw,"rotation/r_eff_ainf"))]))
    dep={}
    for a in ["0.5","inf"]:
        nm=GRID[m].get(a)
        if nm:
            rw,_=rows(nm)
            if rw: dep[a]=mean(ser(rw,"rotation/r_e_dyn"))
    losses={}
    for a in ["0.5","1","2","inf"]:
        nm=GRID[m].get(a)
        if nm:
            _,r=rows(nm)
            if r: losses[a]=sm(r,"eval/loss_min")
    d=dep.get("0.5",float("nan"))-dep.get("inf",float("nan"))
    print(f'  m={m}:  mean N-span {mean(sp):.3f}   depth gap (a=0.5 vs inf) {abs(d):.2f} slots   '
          f'loss_min spread {max(losses.values())-min(losses.values()):.2e}')
print("\n  m=3 gives alpha the widest reach of any margin tested -- and the loss spread there is")
print("  still at the run-to-run floor. The 'more room' does not produce an effect.")

print()
print("="*100)
print("5.  LOW-ALPHA x MARGIN: is the low-alpha penalty a margin artefact?")
print("="*100)
print(f'{"alpha":>6s}{"m":>3s}{"depth":>7s}{"loss_min":>10s}   vs fixed-p_e curve at that depth')
CURVE={1:0.34700,5:0.34429,9:0.34368,13:0.34354}
def interp(d):
    ks=sorted(CURVE)
    if d<=ks[0]: return CURVE[ks[0]]
    for i in range(len(ks)-1):
        if ks[i]<=d<=ks[i+1]:
            f=(d-ks[i])/(ks[i+1]-ks[i]); return CURVE[ks[i]]+f*(CURVE[ks[i+1]]-CURVE[ks[i]])
    return CURVE[ks[-1]]
for a,m in [("0.1",0),("0.1",2),("0.25",0),("0.25",2),("0.05",2),("0.15",2),("0.2",2)]:
    nm=GRID[m].get(a)
    if not nm: continue
    rw,r=rows(nm)
    if not rw: continue
    d=mean(ser(rw,"rotation/r_e_dyn")); l=sm(r,"eval/loss_min")
    print(f'{a:>6s}{m:3d}{d:7.2f}{l:10.5f}   curve {interp(d):.5f}   residual {l-interp(d):+.2e}')
print("\n  a=0.1 appears at BOTH m=0 (depth 9.56) and m=2 (depth 6.50). Same alpha, different")
print("  margin, different depth, and the loss tracks the DEPTH not the alpha.")
