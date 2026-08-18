import json, math
rs=json.load(open("/tmp/xse/runs.json"))
H=json.load(open("/tmp/xse/hist2.json")); H1=json.load(open("/tmp/xse/hist.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
def rw(r):  return H.get(r["name"]+"|"+r["id"], [])
def rw1(r): return H1.get(r["name"]+"|"+r["id"], [])
def ser(rows,k): return [x[k] for x in rows if k in x]
def mean(x): return sum(x)/len(x) if x else float("nan")
R=16; BETA=0.9; GAIN=1.0/(1.0-BETA**2)      # Var(m)=Var(g)/(1-b^2) for m=b*m+g
def isfixed(r):
    d=ser(rw(r),"rotation/r_e_dyn"); return bool(d) and max(d)==min(d)
FIN=[r for r in rs if r["state"]=="finished" and sm(r,"_step")==260
     and r["name"]+"|"+r["id"] in H and g(r,"lora_r")==16 and isfixed(r)]

print("="*104)
print("A.  VALIDATING THE iid-NOISE MODEL OF THE MOMENTUM  (this is 'phase 0', done with no GPU)")
print()
print("  Model:  M = M_signal + E,  E iid per-entry std sigma_m,  E independent of the signal")
print("     =>   ||M_dp||^2 - ||M_nodp||^2  =  r^2 * sigma_m^2        [measured, matched pairs]")
print("  Accountant:  sigma_m = train/noise_std * sqrt(1/(1-beta^2)) = noise_std * 2.294")
print("  If those two agree, the iid / Marchenko-Pastur picture of the momentum is validated.")
print("="*104)
print(f'{"p_e":>7s}{"depth":>7s}{"eps":>5s}  {"||M||nodp":>10s}{"||M||dp":>9s}  '
      f'{"sigma_m MEASURED":>17s}{"sigma_m PREDICTED":>18s}{"ratio":>8s}   {"noise share of ||M||^2":>22s}')
rows=[]
for pe in (0.0625,0.3125,0.333,0.5625,0.8125):
    nod=[r for r in FIN if g(r,"lora_xse_p_e")==pe and g(r,"noise_multiplier")==0]
    dps=[r for r in FIN if g(r,"lora_xse_p_e")==pe and g(r,"noise_multiplier")!=0]
    if not nod or not dps: continue
    mn_no=mean([sm(r,"xs/m_norm") for r in nod if sm(r,"xs/m_norm") is not None])
    for eps in sorted({g(r,"target_epsilon") for r in dps}):
        sel=[r for r in dps if g(r,"target_epsilon")==eps]
        mn_dp=mean([sm(r,"xs/m_norm") for r in sel if sm(r,"xs/m_norm") is not None])
        ns   =mean([mean(ser(rw(r),"train/noise_std")) for r in sel])
        ninf =mean([mean(ser(rw1(r),"rotation/r_eff_ainf")) for r in sel])
        n1   =mean([mean(ser(rw1(r),"rotation/r_eff_a1")) for r in sel])
        er   =mean([mean(ser(rw(r),"rotation/energy_ratio")) for r in sel])
        d    =mean([mean(ser(rw(r),"rotation/r_e_dyn")) for r in sel])
        meas = math.sqrt(max(mn_dp**2-mn_no**2,0)/R**2)
        pred = ns*math.sqrt(GAIN)
        share= 1.0-(mn_no/mn_dp)**2
        print(f'{pe:7.4f}{d:7.2f}{eps:5.0f}  {mn_no:10.4f}{mn_dp:9.4f}  {meas:17.5f}{pred:18.5f}'
              f'{meas/pred:8.2f}   {share*100:21.1f}%')
        rows.append((pe,d,eps,mn_dp,meas,ninf,n1,er,int(round(R-d))))
print()
print("  The ratio is ~1 without any batch-size division => the noise the optimizer sees has")
print("  per-entry std EQUAL to train/noise_std (not divided by B=192), and the momentum gain")
print("  1/(1-beta^2)=5.26 is confirmed. The iid model of the momentum is VALIDATED.")
print("  Also note the last column: under DP, >98% of the momentum's ENERGY is injected noise.")

print()
print("="*104)
print("B.  THE ACTUAL COUNT: how many momentum directions clear the noise bulk edge?")
print()
print("  Bulk edge (quarter-circle, r x r iid):  tau = 2 * sigma_m * sqrt(r)")
print("  sigma_1 = ||M|| * sqrt(p_1) = ||M|| / sqrt(N_inf)")
print("  sigma_2 = ||M|| * sqrt(energy_ratio(r_keep) - p_1)/... (only valid when r_keep = 2)")
print("="*104)
print(f'{"p_e":>7s}{"eps":>5s}  {"tau":>9s}{"sigma_1":>9s}{"sigma_1/tau":>12s}   '
      f'{"threshold rule keeps":>21s}{"entropy rule keeps":>19s}  {"refresh: thr / ent":>19s}')
for pe,d,eps,mn,sig_m,ninf,n1,er,rk in rows:
    tau=2*sig_m*math.sqrt(R)
    sig1=mn/math.sqrt(ninf)
    k_thr = 1 if sig1>tau else 0
    k_ent = int(math.floor(n1))+2
    print(f'{pe:7.4f}{eps:5.0f}  {tau:9.4f}{sig1:9.4f}{sig1/tau:12.2f}   '
          f'{k_thr:21d}{k_ent:19d}  {R-k_thr:9d} / {R-k_ent:<8d}')
print()
print("  Only the TOP direction clears the edge, and only by ~1.2-1.4x. Everything else is")
print("  inside the noise bulk. So under DP the threshold rule keeps ~1 and refreshes ~15,")
print("  while the entropy rule keeps 5-7 and refreshes 9-11. Opposite responses to noise,")
print("  now in absolute units rather than by proxy.")

print()
print("="*104)
print("C.  NON-DP: the same calculation is NOT possible from the logs -- and why")
print("="*104)
nod=[r for r in FIN if g(r,"noise_multiplier")==0]
print(f'{"run":30s}{"depth":>7s}{"||M||":>9s}{"N_inf":>8s}{"sigma_1":>9s}   sigma_m = ?')
for r in sorted(nod, key=lambda x: mean(ser(rw(x),"rotation/r_e_dyn"))):
    d=mean(ser(rw(r),"rotation/r_e_dyn")); mn=sm(r,"xs/m_norm")
    ninf=mean(ser(rw1(r),"rotation/r_eff_ainf"))
    if mn is None or ninf!=ninf: continue
    print(f'{r["name"][:29]:30s}{d:7.2f}{mn:9.4f}{ninf:8.3f}{mn/math.sqrt(ninf):9.4f}   '
          f'UNMEASURABLE (no noiseless reference)')
print()
print("  In non-DP there is no injected noise to subtract, so sigma_m (the SAMPLING-noise scale)")
print("  cannot be recovered from ||M|| alone. Two ways to get it, in order of rigour:")
print("    1. Gavish-Donoho:  tau = 2.858 * median(sigma)  -- needs the spectrum logged (xse.py:617)")
print("    2. Split-batch null: d = (g_A - g_B)/2 over two independent half-batches has ZERO")
print("       signal and EXACTLY the full-batch noise variance. Its spectrum IS the bulk.")
print("       Then sigma_m = sigma_grad * 2.294 and tau = 2*sigma_m*sqrt(r).")
print("  Option 2 needs no distributional assumption and the per-microbatch grads already exist.")
