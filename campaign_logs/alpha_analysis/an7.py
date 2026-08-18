import json, statistics as st
rs=json.load(open("/tmp/xse/runs.json")); H=json.load(open("/tmp/xse/hist2.json"))
def g(r,k,d=None): return r.get("config",{}).get(k,d)
def sm(r,k,d=None): return r.get("summary",{}).get(k,d)
def rows(r): return H.get(r["name"]+"|"+r["id"], [])
def ser(rw,k): return [x[k] for x in rw if k in x]
def mean(x): return sum(x)/len(x) if x else float("nan")
NODP=[r for r in rs if g(r,"noise_multiplier")==0 and r["name"]+"|"+r["id"] in H]
def isfixed(r):
    d=ser(rows(r),"rotation/r_e_dyn"); return bool(d) and max(d)==min(d)

print("="*104)
print("C-CORRECTED.  promotion_count is CAPPED AT r_keep by construction (xse.py:588 loops")
print("   over range(r_keep)), so 'promotions fall with depth' was a ceiling artefact.")
print("   The scale-free quantity is promotion_count / r_keep = the FRACTION of the retained")
print("   set that was freshly discovered rather than inherited.")
print("="*104)
print(f'{"run":32s}{"depth":>7s}{"r_keep":>7s}{"promo/rot":>10s}{"promo/r_keep":>13s}{"loss":>10s}  {"fixed?":>7s}')
prof={}
for r in sorted(NODP, key=lambda x: mean(ser(rows(x),"rotation/r_e_dyn") or [0])):
    rw=rows(r); d=mean(ser(rw,"rotation/r_e_dyn"))
    if d!=d: continue
    rk=16-d; pc=mean(ser(rw,"rotation/promotion_count"))
    print(f'{r["name"][:31]:32s}{d:7.2f}{rk:7.2f}{pc:10.3f}{pc/rk:13.4f}{sm(r,"eval/loss"):10.5f}  {str(isfixed(r)):>7s}')
    prof.setdefault(round(d), []).append(pc/rk)

print()
print("  Grouped by realised depth:")
print(f'  {"depth":>6s}{"n":>3s}{"promo fraction of retained set":>32s}')
for d in sorted(prof):
    print(f'  {d:6d}{len(prof[d]):3d}{mean(prof[d]):32.4f}')
lo=min(prof, key=lambda k: mean(prof[k]))
print(f'\n  minimum at depth {lo} ({mean(prof[lo]):.4f}); values at depth>=5 sit in a flat band.')

print()
print("="*104)
print("A-CAVEATED.  The consecutive-ratio profile is CONFOUNDED by the feedback loop.")
print("   Deeper refresh -> less accumulation -> spikier R -> reported cliff moves.")
print("   Only the FIXED-depth runs assign r_keep independently of the spectrum:")
print("="*104)
print(f'{"run":32s}{"r_keep k":>9s}{"sigma[k+1]/sigma[k]":>21s}{"energy in top k":>17s}{"loss":>10s}')
for r in sorted([x for x in NODP if isfixed(x) and x["name"].startswith("med-nodp-fixed")],
                key=lambda x: -mean(ser(rows(x),"rotation/r_e_dyn"))):
    rw=rows(r); d=mean(ser(rw,"rotation/r_e_dyn")); rk=int(round(16-d))
    print(f'{r["name"][:31]:32s}{rk:9d}{mean(ser(rw,"rotation/spectral_gap")):21.4f}'
          f'{mean(ser(rw,"rotation/energy_ratio")):17.5f}{sm(r,"eval/loss"):10.5f}')
print("\n  Even these four are four different trajectories, not one spectrum. The metric that")
print("  WOULD settle it -- 'singular_values_top' (top-8 singular values) -- is computed at")
print("  xse.py:617 and then DROPPED from the logged per-layer dict (xse.py:803-815). Free fix.")

print()
print("="*104)
print("D-RETRACTED.  xs/grad_snr is NOT a signal-to-noise ratio.")
print("   train_causal_lm.py:2374  ->  grad_snr = ||m_R|| / ||g_R - m_R||")
print("   That is a momentum-consistency ratio. With momentum 0.9, ||m|| ~ 10x||g||, so the")
print("   ratio pins near 1 regardless of noise: non-DP 0.9547-0.9585, DP 0.9651-0.9693.")
print("   It CANNOT be used for the noise argument. Flag as mislabelled.")
print("="*104)

print()
print("="*104)
print("F.  WHAT DOES CLEANLY SEPARATE non-DP FROM DP:  R's condition number")
print("="*104)
DP=[r for r in rs if g(r,"noise_multiplier")!=0 and r["name"]+"|"+r["id"] in H]
for pe in (0.0625,0.3125,0.333,0.5625,0.8125):
    out=[]
    for lab,grp in (("non-DP",NODP),("DP",DP)):
        sel=[r for r in grp if g(r,"lora_xse_p_e")==pe and isfixed(r) and g(r,"lora_r")==16]
        if sel: out.append((lab,len(sel),mean([mean(ser(rows(r),"xs/r_condition")) for r in sel]),
                            mean([mean(ser(rows(r),"xs/r_effective_rank")) for r in sel])))
    if len(out)==2:
        (l1,n1,c1,e1),(l2,n2,c2,e2)=out
        print(f'  p_e={pe:<7} cond(R): non-DP {c1:.2e} (n={n1})  vs  DP {c2:.2e} (n={n2})   ratio {c1/c2:7.1f}x'
              f'   |  r_eff(R): {e1:.2f} vs {e2:.2f}')
print("\n  R is ~30x more ill-conditioned without DP noise (cond ~5e9 vs ~1.5e8). DP noise")
print("  regularises the core. Consistent with rank-1 dominance being a non-DP phenomenon.")

print()
print("="*104)
print("E.  rotation/renyi_gap_a0p5_ainf = N_0.5 - N_inf  -- the alpha span, ALREADY LOGGED")
print("="*104)
for lab,grp in (("non-DP",NODP),("DP",DP)):
    v=[mean(ser(rows(r),"rotation/renyi_gap_a0p5_ainf")) for r in grp]
    v=[x for x in v if x==x]
    print(f'  {lab:8s} n={len(v):3d}  mean {mean(v):6.3f}  range {min(v):.3f}-{max(v):.3f}  '
          f'span<1 in {sum(1 for x in v if x<1)}/{len(v)} runs')
print("\n  The adaptive m=1 family specifically (the runs used as the alpha comparison):")
for nm in ["renyi-ad-nodp-ainf-m1-s42","renyi-ad-nodp-a2-m1-s42","renyi-ad-nodp-a1-m1-s42",
           "renyi-ad-nodp-a0.5-m1-s42","seedrep-ad-nodp-ainf-m1-s43","seedrep-ad-nodp-a2-m1-s43"]:
    for r in NODP:
        if r["name"]==nm:
            print(f'    {nm:32s} span = {mean(ser(rows(r),"rotation/renyi_gap_a0p5_ainf")):.3f}')
