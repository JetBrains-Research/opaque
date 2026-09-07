import time, torch, numpy as np
torch.set_num_threads(2)
from opaque.dpftrl.noise import band_mf_strategy
n=1024; bands=64
def rows(mom):
    t0=time.time()
    s=band_mf_strategy(bands=bands, momentum=mom)
    c=s.coefficients(n_steps=n).double().numpy(); c=np.concatenate([c,np.zeros(n-len(c))])
    sens=float(np.linalg.norm(c))
    # C lower-triangular Toeplitz with c; C^-1 rows via solve
    Cm=np.zeros((n,n))
    for i in range(n): Cm[i:, i]=c[:n-i]
    Ci=np.linalg.inv(Cm)
    rn=np.linalg.norm(Ci,axis=1)
    out={"sens":round(sens,4),"c0":round(float(c[0]),4),"rowC^-1 t0,t7,t63,t511,tn-1":[round(float(rn[i]),4) for i in (0,7,63,511,n-1)]}
    def filt(w):  # w: (n,) toeplitz filter coefficients, factor at last row
        Fm=np.zeros((n,n))
        for i in range(n): Fm[i:, i]=w[:n-i]
        M=Fm@Ci
        return round(float(np.linalg.norm(M[-1])),4)
    for b in (0.95,0.99):
        w=(1-b)*b**np.arange(n); out[f"ema{b}"]=(filt(w), round(float(np.sqrt((1-b)/(1+b))),4))
    for W in (64,256):
        w=np.zeros(n); w[:W]=1/W; out[f"win{W}"]=(filt(w), round(1/np.sqrt(W),4))
    out["time"]=round(time.time()-t0,1)
    return out
for mom in (0.95,1.0):
    print("momentum",mom,rows(mom))
