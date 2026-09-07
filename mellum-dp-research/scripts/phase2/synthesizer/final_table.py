"""Final cost table for phase2-final-design.md (synthesizer). CPU, < 30 s.
Preset regime: nm=0.5622 (DP-SGD/Poisson eps=3 calibration), Bbar=256, k=8, E=64, C_g=0.9, q=256/5e5, T=15625, delta=1e-6.
Centred release d(x)=h(x)-k/E, Delta_h=sqrt(k(1-k/E)).  Filter factors: DP-SGD EMA sqrt((1-b)/(1+b)), window 1/sqrt(W);
band-MF(bands=64, momentum=0.95) factors from judge-impl/mf_rownorm.py rerun this session (1.4309/0.0824/0.0249/0.0198)."""
import math, time
t0=time.time()
from opaque.dpsgd import accounting as A
from opaque.types import PerGroup
from opaque.api.engine.noise_allocation import per_group_noise_stddev
nm=0.5622; q=256/5e5; T=15625; d=1e-6; Bbar=256; k=8; E=64; Cg=0.9
Dh=math.sqrt(k*(1-k/E)); print(f"Delta_h={Dh:.4f}  replace-one sqrt(2k)={math.sqrt(2*k):.3f} vs 2*Delta_h={2*Dh:.3f}")
base=A.poisson(A.gaussian(nm),q)*T; print("baseline eps", round(base.epsilon_at(d),4))
MF={"single":1.4309,"ema95":0.0824,"ema99":0.0249,"w256":0.0198}
SGD={"single":1.0,"ema95":math.sqrt(0.05/1.95),"ema99":math.sqrt(0.01/1.99),"w256":1/16}
print("SGD factors", {kk:round(v,4) for kk,v in SGD.items()})
print("rho | c | grad_infl(real allocator) | chk | eps_if_nm_held | r1 | SGD ema99 | SGD w256 | MF single | MF ema99 | MF w256 | per-layer(x5.29) SGD/MF ema99 | JS thr per-coord SGD/MF")
for rho in (0.5,0.2,0.1,0.05,0.02,0.01):
    mn=PerGroup(groups={"fallback":("a",),"router_load_probe":("z",)},values={"fallback":Cg,"router_load_probe":rho*Cg})
    s=per_group_noise_stddev(mn,nm)
    infl=s.values["fallback"]/(nm*Cg); chk=sum((mn.values[x]/s.values[x])**2 for x in mn.values)*nm**2
    c=math.sqrt(1+1/rho)
    eps_held=(A.poisson(A.gaussian(nm)|A.gaussian(c*nm),q)*T).epsilon_at(d)
    # per-entry noise std of d_hat in units of k/E: sigma_h/(lambda*Bbar) with sigma_h=nm*sqrt(Ch*(Cg+Ch)), lambda=Ch/Dh
    Ch=rho*Cg; lam=Ch/Dh; sig_h=s.values["router_load_probe"]; r1=(sig_h/lam/Bbar)/(k/E)
    r1b=nm*Dh*c/(Bbar*k/E)
    assert abs(r1-r1b)<1e-9
    L=math.sqrt(28)
    js_sgd=r1*SGD["ema99"]*math.sqrt(63/64); js_mf=r1*MF["ema99"]*math.sqrt(63/64)
    print(f"{rho:5.2f} | {c:6.3f} | x{infl:.4f} | {chk:.4f} | {eps_held:.3f} | {100*r1:5.1f}% | {100*r1*SGD['ema99']:5.2f}% | {100*r1*SGD['w256']:5.2f}% | {100*r1*MF['single']:5.1f}% | {100*r1*MF['ema99']:5.2f}% | {100*r1*MF['w256']:5.2f}% | {100*r1*SGD['ema99']*L:5.1f}%/{100*r1*MF['ema99']*L:5.1f}% | {js_sgd:.4f}/{js_mf:.4f}")
# independent draw rows (DP-SGD only)
for c,m,qq in ((2,4,q),(1,4,q),(2,16,q)):
    p=A.poisson(A.gaussian(nm),q)*T | A.poisson(A.gaussian(c*nm),qq)*(T//m)
    r1=nm*Dh*c/(Bbar*k/E)
    print(f"indep c={c} m={m}: eps={p.epsilon_at(d):.4f}  r1={100*r1:.1f}%  after EMA .9 over releases {100*r1*math.sqrt(0.1/1.9):.2f}% (lag {m*10} steps)")
# monitor false alarm at rho=0.02, EMA .99, tau=0.5: z = tau/(r_smoothed)
r_s=nm*Dh*math.sqrt(1+1/0.02)/(Bbar*k/E)*SGD["ema99"]; print(f"monitor: smoothed per-entry noise SGD {100*r_s:.2f}% -> tau=0.5 is {0.5/r_s:.1f} sigma; MF {100*nm*Dh*math.sqrt(1+1/0.02)/(Bbar*k/E)*MF['ema99']:.2f}%")
# noise norm vs C for LoRA r=16
dparam=28*(2*2304*16+2*16*4096 + 2*(2304*16+16*512) + 2*(4096*16+16*2304))
print(f"LoRA params d={dparam}, sqrt(d)={math.sqrt(dparam):.0f}, per-step noise norm nm*C*sqrt(d)/Bbar={nm*Cg*math.sqrt(dparam)/Bbar:.2f} = {nm*math.sqrt(dparam)/Bbar:.2f} C")
print(f"total {time.time()-t0:.1f}s")
