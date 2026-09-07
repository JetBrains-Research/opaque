import math, torch
E,k,L,T=64,8,4,16; Dh=math.sqrt(k*(1-k/E))
torch.manual_seed(0)
rnd = torch.stack([torch.stack([torch.randperm(E)[:k] for _ in range(T)]) for _ in range(L)])
def h_of(routes, mask):
    onehot=(routes[...,None]==torch.arange(E)).float().sum(-2); w=mask.float()[None,:,None]
    return (onehot*w).sum((0,1))/(L*mask.float().sum())
m=torch.zeros(T); m[0]=1.0; m[1]=-0.5
hm=h_of(rnd,m); print(f"[A6 mixed-sign mask] sum_e h={hm.sum().item():.3f} min h={hm.min().item():.3f} max h={hm.max().item():.3f} ||d||={(hm-k/E).norm().item():.3f}  Dh={Dh:.3f}  broken={(hm-k/E).norm().item()>Dh}")
logits=torch.zeros(1000,E, dtype=torch.bfloat16)
p=torch.softmax(logits, dtype=torch.float, dim=-1); _,idx=torch.topk(p,k,dim=-1)
dups=sum(len(set(r.tolist()))!=k for r in idx); print(f"[ties] all-equal logits: rows with duplicate top-k indices = {dups}/1000; distinct experts in row 0 = {len(set(idx[0].tolist()))}")
