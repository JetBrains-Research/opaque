import json, math, statistics as st
rs = json.load(open("/tmp/xse/runs.json"))
H  = json.load(open("/tmp/xse/hist.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
def rows_of(r): return H.get(r["name"]+"|"+r["id"], [])
def ser(rows,k): return [(row["_step"], row[k]) for row in rows if k in row]
def mean(xs): return sum(xs)/len(xs)

print("="*100)
print("6.  DOSE-RESPONSE TEST:  is the observed |dL| correlated with the alpha dose?")
print("="*100)
dose=[0.1378,0.0306,0.1071,0.6276,0.2092,0.0969,0.6939,0.3214,0.1020,0.8061,0.4286]
obs =[4.62e-4,5.71e-4,1.08e-4,2.31e-4,1.25e-4,1.05e-3,1.52e-4,2.51e-4,3.11e-5,1.39e-5,3.82e-5]
lab =["a1vinf m1","a2vinf m1","a1va2 m1","a.5vinf m1","a1vinf m2","a2vinf m2",
      "a.5vinf m2","a1vinf m3","a2vinf m3","a.5vinf m3","m0 a.5va2"]
def rank(xs):
    s=sorted(range(len(xs)), key=lambda i:xs[i]); r=[0]*len(xs)
    for pos,i in enumerate(s): r[i]=pos+1
    return r
rd, ro = rank(dose), rank(obs)
n=len(dose)
sp = 1 - 6*sum((rd[i]-ro[i])**2 for i in range(n))/(n*(n*n-1))
mx,my=mean(dose),mean(obs)
pear = sum((dose[i]-mx)*(obs[i]-my) for i in range(n))/math.sqrt(
        sum((d-mx)**2 for d in dose)*sum((o-my)**2 for o in obs))
for i in range(n): print(f'  {lab[i]:12s} dose={dose[i]:.4f}  |dL|={obs[i]:.2e}')
print(f'\n  Spearman rho(dose, |dL|) = {sp:+.3f}     Pearson r = {pear:+.3f}   (n={n})')
print("  A real dose-response requires rho >> 0. Negative/zero => the differences are not caused by alpha.")

print()
print("="*100
      )
print("7.  NOISE RESPONSE OF N_alpha (for the theory; DP shown for contrast only)")
print("="*100)
def regime(r):
    if g(r,"noise_multiplier")==0: return "nonDP"
    e=g(r,"target_epsilon")
    return f"eps{e}"
groups={}
for r in rs:
    if sm(r,"rotation/r_eff_a1") is None: continue
    if r["state"]!="finished": continue
    rows=rows_of(r)
    if not rows: continue
    d=[v for _,v in ser(rows,"rotation/r_e_dyn")]
    key=(regime(r), g(r,"lora_r"), g(r,"lora_xse_p_e"))
    vals={}
    for k,lab in [("a0p5","0.5"),("a1","1"),("a2","2"),("ainf","inf")]:
        s=[v for _,v in ser(rows,"rotation/r_eff_"+k)]
        if s: vals[lab]=mean(s)
    rr=[v for _,v in ser(rows,"xs_spread/rec_rank_median")]
    ns=[v for _,v in ser(rows,"train/noise_std")]
    if vals: groups.setdefault(key,[]).append((r["name"], vals, mean(rr) if rr else float("nan"),
                                               mean(ns) if ns else float("nan"),
                                               mean(d) if d else float("nan")))
print(f'{"regime":8s}{"r":>3s}{"p_e":>7s}{"n":>3s}  {"N_0.5":>7s}{"N_1":>7s}{"N_2":>7s}{"N_inf":>7s}  {"spike#":>7s}{"noise_std":>10s}')
agg={}
for k in sorted(groups, key=lambda x:(str(x[0]),str(x[1]),str(x[2]))):
    v=groups[k]
    row=[mean([x[1].get(a,float("nan")) for x in v]) for a in ["0.5","1","2","inf"]]
    print(f'{k[0]:8s}{str(k[1]):>3s}{str(k[2]):>7s}{len(v):3d}  ' + "".join(f'{x:7.3f}' for x in row)
          + f'  {mean([x[2] for x in v]):7.2f}{mean([x[3] for x in v]):10.4f}')
    agg.setdefault(k[0],[]).append(row+[mean([x[2] for x in v])])
print()
for reg in ["nonDP","eps3","eps1"]:
    if reg not in agg: continue
    rows=agg[reg]
    print(f'{reg:6s} pooled: N_0.5={mean([r[0] for r in rows]):.3f}  N_1={mean([r[1] for r in rows]):.3f} '
          f' N_2={mean([r[2] for r in rows]):.3f}  N_inf={mean([r[3] for r in rows]):.3f}  spike#={mean([r[4] for r in rows]):.2f}')
print("\n  Interpretation: N_alpha is computed on the NORMALISED spectrum p_i = s_i^2/sum(s_j^2), so it is")
print("  invariant to the overall scale of the momentum -- it cannot see the signal-to-noise ratio.")
print("  Under DP noise the spectrum FLATTENS => N_alpha RISES => r_e = r - floor(N_a) - m FALLS,")
print("  i.e. the rule explores LESS when more of the spectrum is noise. That is the wrong sign.")

print()
print("="*100)
print("8.  HETEROGENEITY AUDIT: how many distinct depths does the rule actually assign across 196 matrices?")
print("="*100)
print("   mean floor(N_a) = 16 - depth - m ; if it equals 1.000 then EVERY matrix got floor(N)=1")
print(f'{"run":34s}{"m":>3s}{"mean floor(N)":>14s}{"implied frac at 2+":>19s}   {"spike# std (per-layer)":>22s}')
for r in sorted([x for x in rs if x["state"]=="finished" and g(x,"noise_multiplier")==0
                 and sm(x,"rotation/r_eff_a1") is not None and sm(x,"_step")==260], key=lambda x:x["name"]):
    m=None
    for tag,v in {"m1":1,"m2":2,"m3":3,"m0":0}.items():
        if "-"+tag+"-" in r["name"]: m=v
    if m is None: continue
    rows=rows_of(r); d=mean([v for _,v in ser(rows,"rotation/r_e_dyn")])
    fl=16-d-m
    rrstd=mean([v for _,v in ser(rows,"xs_spread/rec_rank_std")])
    a=sm(r,"rotation/alpha")
    if not (isinstance(a,(int,float)) and a>=0.5) and a!=float("inf") and str(a)!="Infinity": continue
    print(f'{r["name"]:34s}{m:3d}{fl:14.4f}{max(0.0,fl-1):19.3f}   {rrstd:22.3f}')
