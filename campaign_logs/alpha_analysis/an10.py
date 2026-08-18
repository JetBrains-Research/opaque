import json, math, statistics as st
rs=json.load(open("/tmp/xse/runs.json"))
H=json.load(open("/tmp/xse/hist2.json"))
H1=json.load(open("/tmp/xse/hist.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
def rows(r): return H.get(r["name"]+"|"+r["id"], [])
def rows1(r): return H1.get(r["name"]+"|"+r["id"], [])
def ser(rw,k): return [x[k] for x in rw if k in x]
def mean(x): return sum(x)/len(x) if x else float("nan")
R=16; BETA=0.9
MOM_VAR_GAIN = 1.0/(1.0-BETA**2)          # torchopt: m = b*m + g  =>  Var(m) = Var(g)/(1-b^2)

def isfixed(r):
    d=ser(rows(r),"rotation/r_e_dyn"); return bool(d) and max(d)==min(d)
FIN=[r for r in rs if r["state"]=="finished" and sm(r,"_step")==260
     and r["name"]+"|"+r["id"] in H and g(r,"lora_r")==16 and isfixed(r)]

print("="*100)
print("MEASURING THE NOISE BULK EDGE WITHOUT ANY NORMALISATION ASSUMPTION")
print()
print("  For a matched pair (same p_e, same fixed depth, only noise differs):")
print("      ||M_dp||^2  ~  ||M_signal||^2 + r^2 * sigma_m^2")
print("      ||M_nodp||^2 ~ ||M_signal||^2")
print("  =>  sigma_m = sqrt( (||M_dp||^2 - ||M_nodp||^2) / r^2 )     [per-entry momentum noise std]")
print("  Bulk edge for an r x r iid matrix (quarter-circle law):  tau = 2 * sigma_m * sqrt(r)")
print("  Rigorous count bound (Sum sigma_i^2 = ||M||^2, each counted sigma_i > tau):")
print("      k = #{i : sigma_i > tau}  <=  ||M||^2 / tau^2")
print("="*100)
print()
hdr=(f'{"p_e":>7s}{"depth":>7s}  {"||M|| nodp":>11s}{"||M|| dp":>10s}{"eps":>5s}  '
     f'{"sigma_m":>10s}{"tau":>10s}  {"sig_1 dp":>10s}{"k<=":>7s}  {"floor(N_1)":>11s}')
print(hdr)
out=[]
for pe in (0.0625,0.3125,0.333,0.5625,0.8125):
    nod=[r for r in FIN if g(r,"lora_xse_p_e")==pe and g(r,"noise_multiplier")==0]
    dps=[r for r in FIN if g(r,"lora_xse_p_e")==pe and g(r,"noise_multiplier")!=0]
    if not nod or not dps: continue
    mn_no=mean([sm(r,"xs/m_norm") for r in nod if sm(r,"xs/m_norm") is not None])
    for eps in sorted({g(r,"target_epsilon") for r in dps}):
        sel=[r for r in dps if g(r,"target_epsilon")==eps]
        mn_dp=mean([sm(r,"xs/m_norm") for r in sel if sm(r,"xs/m_norm") is not None])
        n1  =mean([mean(ser(rows1(r),"rotation/r_eff_a1")) for r in sel])
        ninf=mean([mean(ser(rows1(r),"rotation/r_eff_ainf")) for r in sel])
        d   =mean([mean(ser(rows(r),"rotation/r_e_dyn")) for r in sel])
        dif=mn_dp**2 - mn_no**2
        if dif<=0:
            print(f'{pe:7.4f}{d:7.2f}  {mn_no:11.4f}{mn_dp:10.4f}{eps:5.0f}   (dp norm <= nodp norm; skip)')
            continue
        sig_m=math.sqrt(dif/R**2)
        tau=2*sig_m*math.sqrt(R)
        sig1=mn_dp*math.sqrt(1.0/ninf)
        kmax=mn_dp**2/tau**2
        print(f'{pe:7.4f}{d:7.2f}  {mn_no:11.4f}{mn_dp:10.4f}{eps:5.0f}  {sig_m:10.5f}{tau:10.5f}'
              f'  {sig1:10.4f}{kmax:7.2f}  {int(math.floor(n1)) if n1==n1 else -1:11d}')
        out.append((pe,d,eps,sig_m,tau,sig1,kmax,math.floor(n1)))

print()
print("="*100)
print("WHAT THE TWO RULES WOULD DO, IN THE SAME UNITS")
print("="*100)
print(f'{"p_e":>7s}{"eps":>5s}  {"threshold rule keeps k <=":>26s}{"=> refresh >=":>15s}   '
      f'{"entropy rule keeps":>19s}{"=> refresh":>12s}')
for pe,d,eps,sig_m,tau,sig1,kmax,fl in out:
    m=2
    print(f'{pe:7.4f}{eps:5.0f}  {kmax:26.2f}{R-kmax:15.2f}   {fl+m:19d}{R-fl-m:12d}')
print()
print("  The two rules point in OPPOSITE directions under noise: the threshold count collapses")
print("  (few directions clear the inflated bulk edge => refresh nearly everything), while the")
print("  entropy count RISES (flatter spectrum => keep more => refresh less).")

print()
print("="*100)
print("SANITY: is the measured sigma_m consistent with the accountant?")
print("   train/noise_std logs  noise_multiplier * clipping_norm  (std on the SUMMED grads).")
print("   Per-entry std on the AVERAGED grad = that / batch_size; momentum inflates by 1/sqrt(1-b^2).")
print("="*100)
B=192
for pe in (0.333,0.5625):
    for r in FIN:
        if g(r,"lora_xse_p_e")!=pe or g(r,"noise_multiplier")==0: continue
        ns=mean(ser(rows(r),"train/noise_std"))
        pred=ns/B*math.sqrt(MOM_VAR_GAIN)
        print(f'  {r["name"][:34]:35s} eps={g(r,"target_epsilon")} noise_std(logged)={ns:.5f}'
              f'  => predicted sigma_m={pred:.3e}')
        break
print("  Compare with the measured sigma_m column above. Agreement within ~2x validates the")
print("  whole Marchenko-Pastur framing; disagreement localises the normalisation convention.")
