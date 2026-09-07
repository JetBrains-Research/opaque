"""refute-sensitivity: (a) band-MF(64, .95) filter factors at n=1024 (independent re-run of judge-dp/mf.py logic);
(b) EMA warm-up bias; (c) per-layer -> pooled noise equivalence; (d) JS+ shrinkage behaviour near the noise floor."""
import time, math, numpy as np, torch
torch.set_num_threads(4)
from opaque.dpftrl.noise import band_mf_strategy
t0=time.time(); n=1024; bands=64
s=band_mf_strategy(bands=bands, momentum=0.95)
c=s.coefficients(n_steps=n).double().numpy(); c=np.concatenate([c,np.zeros(n-len(c))])
Cm=np.zeros((n,n))
for i in range(n): Cm[i:, i]=c[:n-i]
Ci=np.linalg.inv(Cm); rn=np.linalg.norm(Ci,axis=1)
print(f"band-MF(64,.95): sens(coeffs)={np.linalg.norm(c):.4f}  ||row_t C^-1|| t=0:{rn[0]:.4f} t=7:{rn[7]:.4f} t=63:{rn[63]:.4f} t=n-1:{rn[-1]:.4f}")
def filt(w):
    Fm=np.zeros((n,n))
    for i in range(n): Fm[i:, i]=w[:n-i]
    return float(np.linalg.norm((Fm@Ci)[-1]))
for b in (0.95,0.99):
    w=(1-b)*b**np.arange(n); print(f"  EMA {b}: MF factor={filt(w):.4f}  iid factor={math.sqrt((1-b)/(1+b)):.4f}")
for W in (256,):
    w=np.zeros(n); w[:W]=1/W; print(f"  window {W}: MF factor={filt(w):.4f} iid={1/math.sqrt(W):.4f}")
print(f"  [{time.time()-t0:.1f}s]")
# (b) EMA warm-up bias: E[d_tilde_t] = (1-beta^t) d_true ; noise std phi_t
b=0.99
for t in (1,10,50,100,200,300,460,1000):
    phi=(1-b)*math.sqrt((1-b**(2*t))/(1-b**2))
    print(f"  EMA.99 t={t:4d}: signal gain (1-b^t)={1-b**t:.3f}  noise factor phi_t={phi:.4f}  -> SNR relative to stationary={(1-b**t)/phi*math.sqrt(0.01/1.99):.3f}")
# (c) per-layer release then pooled vs direct pooled: same noise on the pooled vector?
E,k,L=64,8,28; rho=0.02; Cg=0.9; nm=0.5622; Bbar=256
Dh=math.sqrt(k*(1-k/E)); DhL=math.sqrt(k*L*(1-k/E))
lam_p=rho*Cg/Dh; lam_L=rho*Cg/DhL; Ch=rho*Cg; S=Cg+Ch; sig_h=nm*math.sqrt(Ch*S)/Bbar
noise_pooled_direct = sig_h/lam_p
noise_pooled_from_layers = sig_h/lam_L/math.sqrt(L)   # mean over L independent coordinates
print(f"(c) per-entry noise on pooled d_hat: direct={noise_pooled_direct:.5f}  via per-layer release then mean over L={noise_pooled_from_layers:.5f}  ratio={noise_pooled_from_layers/noise_pooled_direct:.4f}; per-layer entry noise = {sig_h/lam_L:.5f} (x{(sig_h/lam_L)/noise_pooled_direct:.2f})")
# (d) JS+ near the floor: simulate d_true with per-coord RMS delta*k/E, noise s per coord, E-1 dof
rng=np.random.default_rng(0); s_=0.0235*0.125
for delta in (0.0,0.01,0.023,0.05,0.1,0.3):
    err=[]; zero=0
    for _ in range(2000):
        d=rng.normal(0,delta*0.125,E); d-=d.mean()
        y=d+rng.normal(0,s_,E); y-=y.mean()
        f=max(0.0,1-(E-1)*s_**2/float(y@y)); est=f*y
        zero+= f==0
        err.append(np.linalg.norm(est-d)/max(np.linalg.norm(d),1e-12))
    print(f"(d) delta={delta:5.3f}: zeroed {100*zero/2000:5.1f}%  median rel err of d_tilde+ = {np.median(err):.3f}")
