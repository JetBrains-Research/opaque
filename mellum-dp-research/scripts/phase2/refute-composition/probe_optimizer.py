"""refute-composition check 3: a zeroed probe leaf through DP-aware optimizers with
PerGroup noise metadata (design §2.2 step 7, risk 8).  Also: does the probe's sigma_h
leak into other leaves' bias correction?  And is the noise on the probe leaf independent
of the gradient leaves' noise (same step key, sequential draws)?
"""
import torch

torch.manual_seed(0)
from opaque.dpsgd.noise import gaussian_noise
from opaque.optimizers import adamw, sgd
from opaque.random import key
from opaque.types import PerGroup, clipped

E = 64
paths = {("w",): "fallback", ("router_load_probe",): "router_load_probe"}
pg = PerGroup(groups=paths, values={"fallback": 0.9 / 256, "router_load_probe": 0.018 / 256})
tree = {"w": torch.randn(1000) * 1e-3, "router_load_probe": torch.randn(E) * 1e-5}
cp = clipped(tree, max_norm=pg)
nf, ns = gaussian_noise(noise_multiplier=0.5622, key=key(0))
noisy, ns = nf(cp, ns)
print("sigma per group:", {k: f"{v:.3e}" for k, v in noisy.noise_stddev.values.items()})

# independence of probe noise from gradient noise: same step key, different draws.
n_probe = noisy.pytree["router_load_probe"] - tree["router_load_probe"]
n_w = noisy.pytree["w"] - tree["w"]
print("probe-noise vs w-noise[:64] equal?", bool(torch.allclose(n_probe / noisy.noise_stddev.values["router_load_probe"],
                                                             n_w[:E] / noisy.noise_stddev.values["fallback"])))

noisy.pytree["router_load_probe"].zero_()  # design §2.2 step 7
for name, opt in [("adamw-BC", adamw(lr=1e-2, weight_decay=0.1, noise_bias_correction=True)),
                  ("adamw", adamw(lr=1e-2, weight_decay=0.1)),
                  ("sgd-momentum", sgd(lr=1e-2, momentum=0.95))]:
    params = {"w": torch.zeros(1000), "router_load_probe": torch.zeros(E)}
    ost = opt.init(params)
    for t in range(3):
        upd, ost = opt.update(noisy, ost, params=params)
        for k in params:
            params[k] = params[k] + upd[k]
    print(f"{name}: probe update after 3 steps max|.| = {upd['router_load_probe'].abs().max():.3e}, "
          f"probe param max|.| = {params['router_load_probe'].abs().max():.3e}, finite: "
          f"{bool(torch.isfinite(params['router_load_probe']).all())}; w update finite: "
          f"{bool(torch.isfinite(upd['w']).all())}")
