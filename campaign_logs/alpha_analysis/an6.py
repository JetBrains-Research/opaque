import json, statistics as st
rs=json.load(open("/tmp/xse/runs.json")); H=json.load(open("/tmp/xse/hist2.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
def rows(r): return H.get(r["name"]+"|"+r["id"], [])
def ser(rw,k): return [x[k] for x in rw if k in x]
def mean(x): return sum(x)/len(x) if x else float("nan")
NODP=[r for r in rs if g(r,"noise_multiplier")==0 and r["name"]+"|"+r["id"] in H]
DP  =[r for r in rs if g(r,"noise_multiplier")!=0 and r["name"]+"|"+r["id"] in H]

print("="*112)
print("A.  THE CUT-POINT TEST.  spectral_gap = sigma[r_keep]/sigma[r_keep-1] -- the drop AT the cut.")
print("    Small value = the rule cut at a real cliff.  ~1 = it cut through a smooth continuum.")
print("="*112)
print(f'{"run":34s}{"depth":>7s}{"r_keep":>7s}{"gap@cut":>9s}{"energy kept":>12s}{"loss":>10s}')
tab=[]
for r in sorted(NODP, key=lambda x: -mean(ser(rows(x),"rotation/r_e_dyn") or [0])):
    rw=rows(r); d=mean(ser(rw,"rotation/r_e_dyn"))
    if d!=d: continue
    gap=mean(ser(rw,"rotation/spectral_gap")); en=mean(ser(rw,"rotation/energy_ratio"))
    rk=16-d
    print(f'{r["name"][:33]:34s}{d:7.2f}{rk:7.2f}{gap:9.4f}{en:12.5f}{sm(r,"eval/loss"):10.5f}')
    tab.append((rk,gap,en,sm(r,"eval/loss"),r["name"]))

print()
print("="*112)
print("B.  CONSECUTIVE-RATIO PROFILE of the momentum spectrum, assembled across runs")
print("    (each run reports the ratio at ITS OWN cut, so together they trace the spectrum shape)")
print("="*112)
byrk={}
for rk,gap,en,loss,nm in tab:
    byrk.setdefault(round(rk), []).append((gap,en))
print(f'{"r_keep":>7s}{"n":>3s}  {"sigma[k]/sigma[k-1]":>20s}{"energy in top k":>17s}   interpretation')
prev=None
for rk in sorted(byrk):
    v=byrk[rk]; gp=mean([x[0] for x in v]); en=mean([x[1] for x in v])
    note = "<-- BIGGEST relative drop" if False else ""
    print(f'{rk:7d}{len(v):3d}  {gp:20.4f}{en:17.5f}   {note}')
gaps={rk: mean([x[0] for x in byrk[rk]]) for rk in byrk}
lo=min(gaps, key=lambda k: gaps[k])
print(f'\n  minimum consecutive ratio at r_keep = {lo}  (ratio {gaps[lo]:.4f})')
print("  => the sharpest cliff in the momentum spectrum sits right after direction", lo)

print()
print("="*112)
print("C.  IS EXPLORATION PRODUCTIVE?  promotion_count = explore directions that became important")
print("="*112)
print(f'{"run":34s}{"depth":>7s}{"promoted/rot":>13s}{"promoted/slot":>14s}{"grad in explore":>16s}{"m in explore":>13s}')
for r in sorted(NODP, key=lambda x: mean(ser(rows(x),"rotation/r_e_dyn") or [0])):
    rw=rows(r); d=mean(ser(rw,"rotation/r_e_dyn"))
    if d!=d: continue
    pc=mean(ser(rw,"rotation/promotion_count"))
    gf=mean(ser(rw,"xs/grad_explore_frac")); mr=mean(ser(rw,"xs/m_explore_ratio"))
    print(f'{r["name"][:33]:34s}{d:7.2f}{pc:13.3f}{pc/d:14.4f}{gf:16.4f}{mr:13.4f}')

print()
print("="*112)
print("D.  xs/grad_snr  -- the quantity N_alpha structurally cannot see (non-DP vs DP)")
print("="*112)
for lab,grp in (("non-DP",NODP),("DP",DP)):
    v=[mean(ser(rows(r),"xs/grad_snr")) for r in grp]
    v=[x for x in v if x==x]
    if v: print(f'  {lab:8s} n={len(v):3d}  grad_snr mean {mean(v):8.4f}  min {min(v):8.4f}  max {max(v):8.4f}')
print()
print("  matched pairs (same p_e, fixed depth, adaptive off):")
def fixed(r):
    d=ser(rows(r),"rotation/r_e_dyn"); return d and max(d)==min(d)
for pe in (0.0625,0.3125,0.333,0.5625,0.8125):
    for lab,grp in (("non-DP",NODP),("DP",DP)):
        sel=[r for r in grp if g(r,"lora_xse_p_e")==pe and fixed(r) and g(r,"lora_r")==16]
        if not sel: continue
        snr=[mean(ser(rows(r),"xs/grad_snr")) for r in sel]
        gap=[mean(ser(rows(r),"rotation/spectral_gap")) for r in sel]
        cond=[mean(ser(rows(r),"xs/r_condition")) for r in sel]
        print(f'    p_e={pe:<7} {lab:7s} n={len(sel):2d}  grad_snr {mean(snr):8.4f}   gap@cut {mean(gap):7.4f}   R cond {mean(cond):9.2f}')

print()
print("="*112)
print("E.  THE ALPHA SPAN IS ALREADY A LOGGED METRIC:  rotation/renyi_gap_a0p5_ainf = N_0.5 - N_inf")
print("    If this is < 1 the floor cannot separate any alpha in [0.5, inf].")
print("="*112)
for lab,grp in (("non-DP",NODP),("DP",DP)):
    v=[(mean(ser(rows(r),"rotation/renyi_gap_a0p5_ainf")), r["name"]) for r in grp]
    v=[x for x in v if x[0]==x[0]]
    below=[x for x in v if x[0]<1]
    print(f'  {lab:8s} n={len(v):3d}  mean span {mean([x[0] for x in v]):6.3f}   '
          f'runs with span<1: {len(below)}/{len(v)}   range {min(x[0] for x in v):.3f}-{max(x[0] for x in v):.3f}')
