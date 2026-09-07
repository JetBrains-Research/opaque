"""refute-sensitivity: independent recomputation of the design's cost table (section 2.6) with the real accountant/allocator."""
import math, time
t0=time.time()
from opaque.dpsgd import accounting as A
from opaque.types import PerGroup
from opaque.api.engine.noise_allocation import per_group_noise_stddev
nm=0.5622; q=256/5e5; T=15625; d=1e-6; Bbar=256; k=8; E=64; Cg=0.9; L=28
Dh=math.sqrt(k*(1-k/E))
print(f"Delta_h centred = {Dh:.4f}; uncentred sqrt(k)={math.sqrt(k):.4f}; replace-one tight sqrt(2k)={math.sqrt(2*k):.3f}; 2*Delta_h={2*Dh:.3f}; per-layer sqrt(kL(1-k/E))={math.sqrt(k*L*(1-k/E)):.3f}")
base=A.poisson(A.gaussian(nm),q)*T; eb=base.epsilon_at(d); print(f"baseline eps={eb:.4f}  [{time.time()-t0:.1f}s]")
ema=lambda b: math.sqrt((1-b)/(1+b))
print(f"EMA factors: .95={ema(.95):.4f} .99={ema(.99):.4f}; window256={1/16:.4f}")
MF_ema99=0.0249; MF_single=1.4309; MF_w256=0.0198
print("rho | c | grad_infl | mahal | eps_nm_held | r1 | SGD ema99 | SGD w256 | MF single | MF ema99 | JS thr SGD/MF")
for rho in (0.5,0.2,0.1,0.05,0.02,0.01):
    Ch=rho*Cg
    mn=PerGroup(groups={"fallback":("a",),"router_load_probe":("z",)},values={"fallback":Cg/Bbar,"router_load_probe":Ch/Bbar})  # bounds already / Bbar as clipped_grad stores them
    s=per_group_noise_stddev(mn,nm)
    infl=s.values["fallback"]/(nm*Cg/Bbar)
    mahal=sum((mn.values[x]/s.values[x])**2 for x in mn.values)*nm**2
    c=math.sqrt(1+1/rho)
    eps_held=(A.poisson(A.gaussian(nm)|A.gaussian(c*nm),q)*T).epsilon_at(d)
    lam=Ch/Dh
    r1=(s.values["router_load_probe"]/lam)/(k/E)   # per-entry noise std of d_hat (already /Bbar) in units of k/E
    r1_closed=nm*Dh*c/(Bbar*k/E)
    js_sgd=r1*ema(.99)*math.sqrt((E-1)/E); js_mf=r1*MF_ema99*math.sqrt((E-1)/E)
    print(f"{rho:5.2f} | {c:6.3f} | x{infl:.4f} | {mahal:.6f} | {eps_held:.3f} | {100*r1:5.1f}% (closed {100*r1_closed:5.1f}%) | {100*r1*ema(.99):5.2f}% | {100*r1/16:5.2f}% | {100*r1*MF_single:5.1f}% | {100*r1*MF_ema99:5.2f}% | {js_sgd:.4f}/{js_mf:.4f}")
for c,m,qq in ((2,4,q),(1,4,q),(2,16,q)):
    p=A.poisson(A.gaussian(nm),q)*T | A.poisson(A.gaussian(c*nm),qq)*(T//m)
    print(f"indep c={c} m={m}: eps={p.epsilon_at(d):.4f}")
# gradient noise norm for LoRA r=16 q/k/v/o
dparam=28*(2*2304*16+2*16*4096 + 2*(2304*16+16*512) + 2*(4096*16+16*2304))
print(f"LoRA d={dparam}, sqrt(d)={math.sqrt(dparam):.0f}, noise norm nm*Cg*sqrt(d)/Bbar={nm*Cg*math.sqrt(dparam)/Bbar:.2f}")
print(f"total {time.time()-t0:.1f}s")
