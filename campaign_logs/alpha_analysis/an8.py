import json, statistics as st
rs=json.load(open("/tmp/xse/runs.json")); H=json.load(open("/tmp/xse/hist.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
def rows(r): return H.get(r["name"]+"|"+r["id"], [])
def ser(rw,k): return [x[k] for x in rw if k in x]
def mean(x): return sum(x)/len(x) if x else float("nan")
NODP=[r for r in rs if g(r,"noise_multiplier")==0 and r["name"]+"|"+r["id"] in H
      and r["state"]=="finished" and sm(r,"_step")==260]

print("="*106)
print("1.  IS floor(N_alpha)=1 REALLY UNIVERSAL FOR alpha>=1, OR ONLY AT THE DEEP OPERATING POINTS?")
print("    Check every non-DP run: is max_t N_1 < 2 ?  is max_t N_0.5 < 2 ?  is min_t p_1 > 0.7071 ?")
print("="*106)
print(f'{"run":32s}{"depth":>7s}  {"max N_0.5":>10s}{"max N_1":>9s}{"max N_2":>9s}{"min p_1":>9s}   '
      f'{"a>=1 collapsed":>15s}{"a=0.5 too":>11s}')
bad1=[]; bad05=[]; badp=[]
for r in sorted(NODP, key=lambda x: mean(ser(rows(x),"rotation/r_e_dyn") or [0])):
    rw=rows(r); d=mean(ser(rw,"rotation/r_e_dyn"))
    if d!=d: continue
    m05=max(ser(rw,"rotation/r_eff_a0p5")); m1=max(ser(rw,"rotation/r_eff_a1"))
    m2=max(ser(rw,"rotation/r_eff_a2")); p1=1/max(ser(rw,"rotation/r_eff_ainf"))
    ok1 = m1<2 and m2<2; ok05 = m05<2
    if not ok1: bad1.append(r["name"])
    if not ok05: bad05.append(r["name"])
    if p1<=2**-0.5: badp.append(r["name"])
    print(f'{r["name"][:31]:32s}{d:7.2f}  {m05:10.3f}{m1:9.3f}{m2:9.3f}{p1:9.3f}   '
          f'{("YES" if ok1 else "no"):>15s}{("YES" if ok05 else "no"):>11s}')
print(f'\n  runs where alpha>=1 is NOT collapsed : {len(bad1)}/{len(NODP)}  {bad1}')
print(f'  runs where alpha=0.5 also collapses  : {len(NODP)-len(bad05)}/{len(NODP)}')
print(f'  runs violating the p_1>0.7071 certificate: {len(badp)}/{len(NODP)}  {badp}')

print()
print("="*106)
print("2.  DOES THE ALPHA SPAN DEPEND ON THE OPERATING POINT?  (i.e. did m=2 choose the answer?)")
print("    span = mean_t (N_0.5 - N_inf).  span<1 => alpha in [0.5,inf] cannot cross a boundary.")
print("="*106)
print(f'{"depth band":>14s}{"n":>3s}{"mean span":>11s}{"max span":>10s}{"mean p_1":>10s}{"mean N_0.5":>12s}')
bands=[(0,4,"1 - 4"),(4,7,"4 - 7"),(7,10,"7 - 10"),(10,12.5,"10 - 12.5"),(12.5,14.5,"12.5 - 14.5"),(14.5,16,"14.5 - 15")]
for lo,hi,lab in bands:
    sel=[]
    for r in NODP:
        rw=rows(r); d=mean(ser(rw,"rotation/r_e_dyn") or [float("nan")])
        if d==d and lo<=d<hi: sel.append(r)
    if not sel: continue
    sp=[mean([a-b for a,b in zip(ser(rows(r),"rotation/r_eff_a0p5"),ser(rows(r),"rotation/r_eff_ainf"))]) for r in sel]
    p1=[1/mean(ser(rows(r),"rotation/r_eff_ainf")) for r in sel]
    n05=[mean(ser(rows(r),"rotation/r_eff_a0p5")) for r in sel]
    print(f'{lab:>14s}{len(sel):3d}{mean(sp):11.3f}{max(sp):10.3f}{mean(p1):10.3f}{mean(n05):12.3f}')
print("\n  => the span GROWS as exploration gets shallower (flatter spectrum). m=2 sits in the")
print("     deep/spiky band where the span is smallest.")

print()
print("="*106)
print("3.  MARGIN AS OPERATING POINT: what alpha DOSE is realised at each m, and is it detectable?")
print("="*106)
def dep(nm):
    for r in NODP:
        if r["name"]==nm: return dict(zip(ser(rows(r),"rotation/r_e_dyn") and
            [x["_step"] for x in rows(r) if "rotation/r_e_dyn" in x], ser(rows(r),"rotation/r_e_dyn")))
def dose(n1,n2):
    a,b=dep(n1),dep(n2)
    if not a or not b: return None
    ts=sorted(set(a)&set(b)); return mean([abs(a[t]-b[t]) for t in ts]), max(abs(a[t]-b[t]) for t in ts)
print(f'{"m":>3s}{"depth":>7s}{"a=0.5 vs inf dose (mean/max)":>30s}{"local slope /slot":>19s}{"max effect":>12s}{"floor":>9s}{"ratio":>8s}')
# local slope from the fixed-p_e mediator curve: 1->5 = 6.78e-4, 5->9 = 1.52e-4, 9->13 = 3.52e-5
def slope(d):
    if d<5:  return 6.78e-4
    if d<9:  return 1.52e-4
    return 3.52e-5
def floor(d):
    if d<7:  return 3.0e-5
    if d<12: return 1.0e-4   # UNMEASURED - interpolated, flagged below
    return 3.0e-4
for m,d,n1,n2 in [(1,13.9,"renyi-ad-nodp-a0.5-m1-s42","renyi-ad-nodp-ainf-m1-s42"),
                  (2,12.8,"renyi-ad-nodp-a0.5-m2-s42","renyi-ad-nodp-ainf-m2-s42"),
                  (3,11.7,"renyi-ad-nodp-a0.5-m3-s42","renyi-ad-nodp-ainf-m3-s42")]:
    dd=dose(n1,n2)
    if not dd: print(m,"missing"); continue
    eff=dd[1]*slope(d); fl=floor(d)
    print(f'{m:3d}{d:7.1f}{f"{dd[0]:.3f} / {dd[1]:.3f}":>30s}{slope(d):19.2e}{eff:12.2e}{fl:9.1e}{eff/fl:8.2f}x')
print("\n  PROJECTION to margins never tested with an alpha comparison (dose extrapolated from the")
print("  m=1,2,3 trend of ~+0.09 slots per unit m; slope and floor from the tables above):")
for m in (4,6,8,10):
    d = 15-m + 0.0   # depth ~ 16 - 1 - m
    ds = 0.628 + 0.09*(m-1)
    eff = ds*slope(d); fl=floor(d)
    print(f'{m:3d}{d:7.1f}{f"~{ds:.2f} (extrapolated)":>30s}{slope(d):19.2e}{eff:12.2e}{fl:9.1e}{eff/fl:8.2f}x')

print()
print("="*106)
print("4.  IS THERE ANY MATCHED-DEPTH ALPHA COMPARISON AT SHALLOW DEPTH?  (the actual gap)")
print("="*106)
byd={}
for r in NODP:
    rw=rows(r); d=mean(ser(rw,"rotation/r_e_dyn") or [float("nan")])
    a=sm(r,"rotation/alpha")
    if d!=d: continue
    byd.setdefault(round(d*2)/2, []).append((str(a), r["name"], sm(r,"eval/loss")))
print("  realised depth -> distinct alphas present at that depth (+-0.25 slots):")
for d in sorted(byd):
    al=sorted({x[0] for x in byd[d]})
    flag = "  <== alpha CONTRAST" if len(al)>1 else ""
    print(f'   {d:6.1f}  n={len(byd[d]):2d}  alphas={al}{flag}')
print("\n  Every multi-alpha depth is >= 11.3. There is NO alpha contrast at depth < 11 anywhere")
print("  in the corpus, matched or otherwise.")

print()
print("="*106)
print("5.  CLAMP CHECK for low alpha at m=2:  r_e = clip(16-floor(N)-m, 1, 15)")
print("="*106)
for nm in ["lowa-nodp-a005-m2-s42","lowa-nodp-a01-m2-s42","lowa-nodp-a015-m2-s42","lowa-nodp-a02-m2-s42"]:
    for r in NODP:
        if r["name"]!=nm: continue
        rw=rows(r); d=mean(ser(rw,"rotation/r_e_dyn"))
        n05=mean(ser(rw,"rotation/r_eff_a0p5")); n0=mean(ser(rw,"rotation/r_eff_a0"))
        print(f'  {nm:26s} depth {d:6.3f}  implied mean floor(N) {16-d-2:6.3f}   '
              f'N_0.5 {n05:5.2f}  N_0 {n0:5.2f}   clamp risk: floor(N)>=13 needed for r_e=1')
