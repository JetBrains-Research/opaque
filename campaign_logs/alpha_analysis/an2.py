import json, math, statistics as st
rs = json.load(open("/tmp/xse/runs.json"))
H  = json.load(open("/tmp/xse/hist.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
idx = {r["name"]+"|"+r["id"]: r for r in rs}
def hist_of(name):
    out=[]
    for k,v in H.items():
        if k.split("|")[0]==name: out.append((idx[k], v))
    return out
def ser(rows,k): return [(row["_step"], row[k]) for row in rows if k in row]

M = {"m1":1,"m2":2,"m3":3,"m0":0}
def margin_of(name):
    for tag,v in M.items():
        if "-"+tag+"-" in name: return v
    return None

print("="*112)
print("A.  NON-DP ADAPTIVE RUNS — realised depth trajectory and the Renyi grid (layer means)")
print("="*112)
hdr=f'{"run":32s}{"a":>5s}{"m":>3s}{"n":>4s}  {"depth@1st":>9s}{"depth@last":>10s}{"depth mean":>11s}  {"N.5":>6s}{"N1":>6s}{"N2":>6s}{"Ninf":>6s}  {"N0":>6s}  {"loss":>9s}'
print(hdr)
NODP=[r for r in rs if g(r,"noise_multiplier")==0 and sm(r,"rotation/r_eff_a1") is not None]
rowsout=[]
for r in sorted(NODP, key=lambda x:(str(sm(x,"rotation/alpha")), x["name"])):
    nm=r["name"]; rows=H.get(nm+"|"+r["id"])
    if not rows: continue
    d=ser(rows,"rotation/r_e_dyn")
    if not d: continue
    a=sm(r,"rotation/alpha"); m=margin_of(nm)
    def mean(k):
        s=ser(rows,k); return sum(v for _,v in s)/len(s) if s else float("nan")
    rowsout.append(dict(name=nm,alpha=a,m=m,n=len(d),d0=d[0][1],dL=d[-1][1],
        dmean=sum(v for _,v in d)/len(d),
        N05=mean("rotation/r_eff_a0p5"),N1=mean("rotation/r_eff_a1"),
        N2=mean("rotation/r_eff_a2"),Ninf=mean("rotation/r_eff_ainf"),
        N0=mean("rotation/r_eff_a0"),
        loss=sm(r,"eval/loss"), state=r["state"], step=sm(r,"_step"), seed=g(r,"seed"),
        pe=g(r,"lora_xse_p_e"), rid=r["id"]))
    o=rowsout[-1]
    print(f'{nm:32s}{str(a):>5s}{str(m):>3s}{o["n"]:4d}  {o["d0"]:9.4f}{o["dL"]:10.4f}{o["dmean"]:11.4f}  '
          f'{o["N05"]:6.3f}{o["N1"]:6.3f}{o["N2"]:6.3f}{o["Ninf"]:6.3f}  {o["N0"]:6.2f}  {o["loss"]:9.5f} {o["state"]} {o["step"]}')
json.dump(rowsout, open("/tmp/xse/nodp_rows.json","w"))

print()
print("="*112)
print("B.  THE FLOOR COLLAPSE:  span of N_alpha over alpha in [0.5, inf] per run  (needs to be <1 for collapse)")
print("="*112)
print(f'{"run":32s}{"a":>5s}  {"max_t N.5":>9s}{"min_t Ninf":>11s}{"max span_t":>11s}{"mean span":>10s}   {"N.5<2 always?":>14s}')
for o in rowsout:
    rows=H[o["name"]+"|"+o["rid"]]
    s05=dict(ser(rows,"rotation/r_eff_a0p5")); sinf=dict(ser(rows,"rotation/r_eff_ainf"))
    common=sorted(set(s05)&set(sinf))
    if not common: continue
    spans=[s05[t]-sinf[t] for t in common]
    print(f'{o["name"]:32s}{str(o["alpha"]):>5s}  {max(s05[t] for t in common):9.3f}{min(sinf[t] for t in common):11.3f}'
          f'{max(spans):11.3f}{sum(spans)/len(spans):10.3f}   {str(all(s05[t]<2 for t in common)):>14s}')

print()
print("="*112)
print("C.  ALPHA DOSE: matched-margin depth trajectories, alpha in {1,2,inf} at m=1 (seed 42)")
print("="*112)
trip=["renyi-ad-nodp-a1-m1-s42","renyi-ad-nodp-a2-m1-s42","renyi-ad-nodp-ainf-m1-s42","renyi-ad-nodp-a0.5-m1-s42"]
D={}
for nm in trip:
    for r,rows in hist_of(nm):
        D[nm]=dict(ser(rows,"rotation/r_e_dyn"))
steps=sorted(set.intersection(*[set(v) for v in D.values()]))
print("step  " + "".join(f'{nm.split("-")[3]:>10s}' for nm in trip) + "   max|diff| over {1,2,inf}")
for t in steps:
    vals=[D[nm][t] for nm in trip]
    sub=[D[nm][t] for nm in trip[:3]]
    print(f'{t:5d} ' + "".join(f'{v:10.4f}' for v in vals) + f'      {max(sub)-min(sub):.4f}')
for nm in trip:
    v=[D[nm][t] for t in steps]
    print(f'{nm:34s} mean depth {sum(v)/len(v):.4f}   final {v[-1]:.4f}')
print()
mad = sum(max(D[n][t] for n in trip[:3])-min(D[n][t] for n in trip[:3]) for t in steps)/len(steps)
print(f'MEAN ABSOLUTE DEPTH SEPARATION across alpha in (1,2,inf) at m=1 : {mad:.4f} slots')
mad2 = sum(abs(D["renyi-ad-nodp-a0.5-m1-s42"][t]-D["renyi-ad-nodp-ainf-m1-s42"][t]) for t in steps)/len(steps)
print(f'MEAN ABSOLUTE DEPTH SEPARATION alpha=0.5 vs inf at m=1          : {mad2:.4f} slots')
