import json, math, statistics as st
rs = json.load(open("/tmp/xse/runs.json"))
H  = json.load(open("/tmp/xse/hist.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
def rows_of(r): return H.get(r["name"]+"|"+r["id"], [])
def ser(rows,k): return [(row["_step"], row[k]) for row in rows if k in row]
def mean(xs): return sum(xs)/len(xs)

print("="*100)
print("9.  'INERT WHERE STABLE, UNSTABLE WHERE LIVE':  alpha vs (sensitivity, within-run drift)")
print("="*100)
SET=[("0.05","lowa-nodp-a005-m2-s42"),("0.1","lowa-nodp-a01-m2-s42"),("0.15","lowa-nodp-a015-m2-s42"),
     ("0.2","lowa-nodp-a02-m2-s42"),("0.25","med-nodp-adaptive-a025-m2-s42"),
     ("0.5","renyi-ad-nodp-a0.5-m2-s42"),("1","renyi-ad-nodp-a1-m2-s42"),
     ("2","renyi-ad-nodp-a2-m2-s42"),("inf","renyi-ad-nodp-ainf-m2-s42")]
print(f'{"alpha":>6s}{"depth first":>12s}{"depth last":>11s}{"DRIFT":>8s}{"depth sd_t":>11s}{"mean floor(N)":>14s}{"frac het.":>10s}')
for a,nm in SET:
    r=[x for x in rs if x["name"]==nm and x["state"]=="finished"][0]
    d=[v for _,v in ser(rows_of(r),"rotation/r_e_dyn")]
    fl=16-mean(d)-2
    print(f'{a:>6s}{d[0]:12.3f}{d[-1]:11.3f}{d[-1]-d[0]:8.3f}{st.stdev(d):11.3f}{fl:14.3f}{max(0,fl-1):10.3f}')
print()
print("Also, the m=0 arms (no margin) where the feedback loop is strongest:")
for nm in ["m0-nodp-a01-m0-s42","m0-nodp-a025-m0-s42","m0-nodp-a05-m0-s42","m0-nodp-a2-m0-s42"]:
    r=[x for x in rs if x["name"]==nm and x["state"]=="finished"][0]
    d=[v for _,v in ser(rows_of(r),"rotation/r_e_dyn")]
    print(f'  {nm:24s} depth {d[0]:7.3f} -> {d[-1]:7.3f}   drift {d[-1]-d[0]:+7.3f}  sd_t {st.stdev(d):.3f}  loss {sm(r,"eval/loss"):.5f}')

print()
print("="*100)
print("10. r-DEPENDENCE of N_alpha (does a bigger r escape the floor's null space?)")
print("="*100)
print(f'{"regime":8s}{"r":>4s}{"n":>3s}  {"N_0.5":>7s}{"N_1":>7s}{"N_2":>7s}{"N_inf":>7s}  {"p_1=1/N_inf":>12s}{"delta":>7s}  {"floor(N_1)":>11s}')
buck={}
for r in rs:
    if sm(r,"rotation/r_eff_a1") is None or r["state"]!="finished": continue
    rows=rows_of(r)
    if not rows: continue
    reg = "nonDP" if g(r,"noise_multiplier")==0 else f'eps{g(r,"target_epsilon")}'
    key=(reg, g(r,"lora_r"))
    v=[mean([x for _,x in ser(rows,"rotation/r_eff_"+k)] or [float("nan")])
       for k in ["a0p5","a1","a2","ainf"]]
    buck.setdefault(key,[]).append(v)
for k in sorted(buck, key=lambda x:(x[0],x[1])):
    vs=buck[k]; row=[mean([v[i] for v in vs]) for i in range(4)]
    p1=1/row[3]; delta=1-p1
    print(f'{k[0]:8s}{k[1]:4d}{len(vs):3d}  ' + "".join(f'{x:7.3f}' for x in row)
          + f'  {p1:12.3f}{delta:7.3f}  {math.floor(row[1]):11d}')

print()
print("="*100)
print("11. THEOREM-2 CERTIFICATE per run:  p_1 = 1/N_inf  -> does it certify floor(N_a)=1 for a>=1 ?")
print("    (a) alpha>=2 certificate:   p_1 > 2^(-1/2) = 0.7071")
print("    (b) alpha=1  certificate:   exp(H_b(delta) + delta*log(r-1)) < 2")
print("="*100)
def Hb(d):
    if d<=0 or d>=1: return 0.0
    return -d*math.log(d)-(1-d)*math.log(1-d)
print(f'{"run":34s}{"N_inf":>7s}{"p_1":>7s}{"delta":>7s}  {"a>=2 cert":>10s}{"N_1 bound":>10s}{"a=1 cert":>9s}  {"measured N_1":>13s}{"realised frac het":>18s}')
for r in sorted([x for x in rs if x["state"]=="finished" and g(x,"noise_multiplier")==0
                 and sm(x,"rotation/r_eff_a1") is not None and sm(x,"_step")==260], key=lambda x:x["name"]):
    m=None
    for tag,v in {"m1":1,"m2":2,"m3":3,"m0":0}.items():
        if "-"+tag+"-" in r["name"]: m=v
    rows=rows_of(r)
    ninf=mean([v for _,v in ser(rows,"rotation/r_eff_ainf")])
    n1=mean([v for _,v in ser(rows,"rotation/r_eff_a1")])
    p1=1/ninf; delta=1-p1
    b1=math.exp(Hb(delta)+delta*math.log(15))
    d=mean([v for _,v in ser(rows,"rotation/r_e_dyn")])
    het = "" if m is None else f'{max(0.0,16-d-m-1):.4f}'
    print(f'{r["name"]:34s}{ninf:7.3f}{p1:7.3f}{delta:7.3f}  {("YES" if p1>2**-0.5 else "no"):>10s}'
          f'{b1:10.3f}{("YES" if b1<2 else "no"):>9s}  {n1:13.3f}{het:>18s}')
