import time, math, torch
t0=time.time()
from opaque.dpsgd import accounting as A
nm=0.5622; q=256/5e5; T=15625; d=1e-6
def eps(p): return p.epsilon_at(d)
base=A.poisson(A.gaussian(nm),q)*T
print("baseline eps", round(eps(base),4), f"[{time.time()-t0:.1f}s]")
for rho in (0.05,0.02):
    c=math.sqrt(1+1/rho)
    p=A.poisson(A.gaussian(nm)|A.gaussian(c*nm),q)*T
    print(f"pay-in-eps rho={rho} c={c:.3f} eps={eps(p):.4f}")
# optimal's independent draw: c=2, every m=4 steps, q2=q
for c,m in ((2,4),(1,4),(2,16)):
    p=A.poisson(A.gaussian(nm),q)*T | A.poisson(A.gaussian(c*nm),q)*(T//m)
    print(f"indep draw c={c} m={m}: eps={eps(p):.4f}")
# 4x batch c=1 m=4
p=A.poisson(A.gaussian(nm),q)*T | A.poisson(A.gaussian(nm),4*q)*(T//4)
print(f"indep draw 4x batch c=1 m=4: eps={eps(p):.4f}  [{time.time()-t0:.1f}s]")
# Mahalanobis identity
from opaque.types import PerGroup
from opaque.api.engine.noise_allocation import per_group_noise_stddev
C=0.9
for rho in (0.02,0.05,0.1):
    mn=PerGroup(groups={"fallback":("a",),"router_load_probe":("z",)},values={"fallback":C,"router_load_probe":rho*C})
    s=per_group_noise_stddev(mn,nm)
    chk=sum((mn.values[k]/s.values[k])**2 for k in mn.values)*nm**2
    print(f"rho={rho} sig_g/(nm C)={s.values['fallback']/(nm*C):.4f} chk={chk:.4f}")
