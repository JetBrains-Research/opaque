import json, math, statistics as st
rs = json.load(open("/tmp/xse/runs.json"))
H  = json.load(open("/tmp/xse/hist.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
def rows_of(r): return H.get(r["name"]+"|"+r["id"], [])
def ser(rows,k): return [(row["_step"], row[k]) for row in rows if k in row]
def mean(xs): return sum(xs)/len(xs)
def sd(xs): return st.stdev(xs) if len(xs)>1 else float("nan")

NODP=[r for r in rs if g(r,"noise_multiplier")==0 and sm(r,"rotation/r_eff_a1") is not None
      and r["state"]=="finished" and sm(r,"_step")==260]

print("="*104)
print("1.  MEDIATOR CURVE  L(d)  from FIXED-p_e runs (adaptive OFF, no feedback), non-DP seed 42")
print("="*104)
fixed=[r for r in NODP if r["name"].startswith("med-nodp-fixed")]
pts=sorted((sm(r,"rotation/r_e_dyn"), sm(r,"eval/loss"), sm(r,"eval/loss_min"), r["name"]) for r in fixed)
print(f'{"depth":>6s}{"loss":>10s}{"loss_min":>10s}   marginal slope /slot')
prev=None
for d,l,lm,nm in pts:
    s = "" if prev is None else f'{(l-prev[1])/(d-prev[0]):+.3e}'
    print(f'{d:6.2f}{l:10.5f}{lm:10.5f}   {s}')
    prev=(d,l)
d1,l1=pts[0][0],pts[0][1]; d13,l13=pts[-1][0],pts[-1][1]
print(f'\ntotal 1->13 : {l13-l1:+.3e}')
# plateau slope 9->13
p9=[p for p in pts if p[0]==9][0]; p13=[p for p in pts if p[0]==13][0]
slope_plateau=(p13[1]-p9[1])/(p13[0]-p9[0])
p5=[p for p in pts if p[0]==5][0]
print(f'slope 5->9   : {(p9[1]-p5[1])/4:+.3e} /slot')
print(f'slope 9->13  : {slope_plateau:+.3e} /slot   <-- PLATEAU SLOPE (use for the alpha bound)')

print()
print("="*104)
print("2.  REPLICATE (NONDETERMINISM) FLOORS — groups that are the SAME ALGORITHM")
print("="*104)
def show(label, names, key="eval/loss"):
    got=[]
    for nm in names:
        cand=[r for r in rs if r["name"]==nm and r["state"]=="finished" and sm(r,"_step")==260]
        for r in cand: got.append((nm, sm(r,key), sm(r,"rotation/r_e_dyn")))
    ls=[v for _,v,_ in got]
    if len(ls)<2: print(f'{label}: n<2'); return None
    print(f'{label}')
    for nm,v,d in got: print(f'    {nm:34s} depth={d if d is None else round(d,4)}  {key}={v:.6f}')
    print(f'    n={len(ls)}  mean={mean(ls):.6f}  sd={sd(ls):.2e}  spread={max(ls)-min(ls):.2e}')
    return sd(ls)

f_shallow = show("[A] depth-5 non-adaptive, IDENTICAL config+seed (pure nondeterminism)",
     ["renyi-alpha-nodp","renyi-nodp-s42"])
show("[B] depth-5 non-adaptive, seeds 42/43/44 (+dup)",
     ["renyi-nodp-s42","renyi-nodp-s43","renyi-nodp-s44"])
f_deep = show("[C] depth-14 adaptive m=1, alpha in {1,2,inf} @ seed 42  (PROVEN same algorithm, see 3)",
     ["renyi-ad-nodp-a1-m1-s42","renyi-ad-nodp-a2-m1-s42","renyi-ad-nodp-ainf-m1-s42"])
f_deep_all = show("[D] depth-14 adaptive m=1, alpha in {2,inf} x seeds 42/43",
     ["renyi-ad-nodp-a2-m1-s42","renyi-ad-nodp-ainf-m1-s42",
      "seedrep-ad-nodp-a2-m1-s43","seedrep-ad-nodp-ainf-m1-s43"])
show("[E] depth-13 adaptive alpha=inf m=2, two separate submissions (identical config+seed)",
     ["med-nodp-adaptive-ainf-m2-s42","renyi-ad-nodp-ainf-m2-s42"])
show("[C'] same as [C] but loss_min", ["renyi-ad-nodp-a1-m1-s42","renyi-ad-nodp-a2-m1-s42",
      "renyi-ad-nodp-ainf-m1-s42"], key="eval/loss_min")
show("[D'] same as [D] but loss_min", ["renyi-ad-nodp-a2-m1-s42","renyi-ad-nodp-ainf-m1-s42",
      "seedrep-ad-nodp-a2-m1-s43","seedrep-ad-nodp-ainf-m1-s43"], key="eval/loss_min")

print()
print("="*104)
print("3.  THE MEDIATION BOUND:  observed alpha 'effect' vs MAXIMUM POSSIBLE mediated effect")
print("="*104)
def depth_traj(nm):
    for r in rs:
        if r["name"]==nm and r["state"]=="finished":
            d=dict(ser(rows_of(r),"rotation/r_e_dyn"))
            if d: return d
    return None
def dose(nm1,nm2):
    a,b=depth_traj(nm1),depth_traj(nm2)
    ts=sorted(set(a)&set(b))
    return mean([abs(a[t]-b[t]) for t in ts]), max(abs(a[t]-b[t]) for t in ts)
def lossof(nm,key="eval/loss"):
    for r in rs:
        if r["name"]==nm and r["state"]=="finished" and sm(r,"_step")==260: return sm(r,key)
S=abs(slope_plateau)
print(f'plateau |dL/d(depth)| = {S:.2e} per slot     (from fixed-p_e curve, 9->13)')
print()
print(f'{"contrast":48s}{"mean dose":>10s}{"max dose":>9s}{"max mediated |dL|":>19s}{"observed |dL|":>15s}{"ratio":>9s}')
CONTRASTS=[("renyi-ad-nodp-a1-m1-s42","renyi-ad-nodp-ainf-m1-s42"),
           ("renyi-ad-nodp-a2-m1-s42","renyi-ad-nodp-ainf-m1-s42"),
           ("renyi-ad-nodp-a1-m1-s42","renyi-ad-nodp-a2-m1-s42"),
           ("renyi-ad-nodp-a0.5-m1-s42","renyi-ad-nodp-ainf-m1-s42"),
           ("renyi-ad-nodp-a1-m2-s42","renyi-ad-nodp-ainf-m2-s42"),
           ("renyi-ad-nodp-a2-m2-s42","renyi-ad-nodp-ainf-m2-s42"),
           ("renyi-ad-nodp-a0.5-m2-s42","renyi-ad-nodp-ainf-m2-s42"),
           ("renyi-ad-nodp-a1-m3-s42","renyi-ad-nodp-ainf-m3-s42"),
           ("renyi-ad-nodp-a2-m3-s42","renyi-ad-nodp-ainf-m3-s42"),
           ("renyi-ad-nodp-a0.5-m3-s42","renyi-ad-nodp-ainf-m3-s42"),
           ("m0-nodp-a05-m0-s42","m0-nodp-a2-m0-s42")]
for n1,n2 in CONTRASTS:
    try:
        md,mx=dose(n1,n2); obs=abs(lossof(n1)-lossof(n2))
        pred=mx*S
        lab=f'{n1.replace("renyi-ad-nodp-","").replace("m0-nodp-","m0:")} vs {n2.split("-")[-2] if "m0" not in n2 else "a2"}'
        print(f'{n1[:26]+" vs "+n2[-14:]:48s}{md:10.4f}{mx:9.4f}{pred:19.2e}{obs:15.2e}{obs/pred:9.0f}x')
    except Exception as e: print(n1,n2,"ERR",e)

print()
print("="*104)
print("4.  alpha -> DEPTH MAP AT MATCHED MARGIN m=2 (the sensitivity / conditioning of the knob)")
print("="*104)
m2=[("0.05","lowa-nodp-a005-m2-s42"),("0.1","lowa-nodp-a01-m2-s42"),("0.15","lowa-nodp-a015-m2-s42"),
    ("0.2","lowa-nodp-a02-m2-s42"),("0.25","med-nodp-adaptive-a025-m2-s42"),
    ("0.5","renyi-ad-nodp-a0.5-m2-s42"),("1","renyi-ad-nodp-a1-m2-s42"),
    ("2","renyi-ad-nodp-a2-m2-s42"),("inf","renyi-ad-nodp-ainf-m2-s42")]
print(f'{"alpha":>6s}{"mean depth":>11s}{"floor(N)=16-d-2":>16s}{"loss":>10s}{"loss_min":>10s}   d(depth)/d(alpha)')
prev=None
tab=[]
for a,nm in m2:
    r=[x for x in rs if x["name"]==nm and x["state"]=="finished"][0]
    d=mean([v for _,v in ser(rows_of(r),"rotation/r_e_dyn")])
    l=sm(r,"eval/loss"); lm=sm(r,"eval/loss_min")
    av=float("inf") if a=="inf" else float(a)
    slope="" if prev is None or math.isinf(av) else f'{(d-prev[1])/(av-prev[0]):+9.2f}'
    print(f'{a:>6s}{d:11.3f}{16-d-2:16.3f}{l:10.5f}{lm:10.5f}   {slope}')
    tab.append((a,av,d,l,lm))
    if not math.isinf(av): prev=(av,d)

print()
print("="*104)
print("5.  ALTERNATIVE RULE ALREADY LOGGED:  spike count  #{sigma_i > 2*median(sigma)}  (Donoho-Gavish-ish)")
print("="*104)
print(f'{"run":34s}{"depth":>8s}{"floor(N_a)":>11s}{"rec_rank med":>13s}{"min":>5s}{"max":>5s}{"std":>7s}  {"implied r_e = 16-rec-m":>22s}')
for r in sorted(NODP,key=lambda x:x["name"]):
    rows=rows_of(r)
    rr=[v for _,v in ser(rows,"xs_spread/rec_rank_median")]
    rmin=[v for _,v in ser(rows,"xs_spread/rec_rank_min")]
    rmax=[v for _,v in ser(rows,"xs_spread/rec_rank_max")]
    rstd=[v for _,v in ser(rows,"xs_spread/rec_rank_std")]
    d=mean([v for _,v in ser(rows,"rotation/r_e_dyn")])
    if not rr: continue
    m=None
    for tag,v in {"m1":1,"m2":2,"m3":3,"m0":0}.items():
        if "-"+tag+"-" in r["name"]: m=v
    fl = (16-d-m) if m is not None else float("nan")
    print(f'{r["name"]:34s}{d:8.3f}{fl:11.3f}{mean(rr):13.2f}{mean(rmin):5.1f}{mean(rmax):5.1f}{mean(rstd):7.3f}  {16-mean(rr)-(m if m is not None else 2):22.2f}')
