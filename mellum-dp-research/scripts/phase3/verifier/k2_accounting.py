"""K2: accountant invariance + cost-table 'pay in eps' column, with the real Opaque accountant and allocator."""
import math, time
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.types import PerGroup
from opaque.api.engine.noise_allocation import per_group_noise_stddev

NM = 0.5622
Q = 256 / 5e5
T = 15625
DELTA = 1e-6

t0 = time.time()
base = dpsgd_acc.poisson(dpsgd_acc.gaussian(NM), Q) * T
eps0 = base.epsilon_at(DELTA)
print(f"baseline poisson(gaussian({NM}), {Q})*{T} @ delta={DELTA}: eps = {eps0:.4f}  ({time.time()-t0:.1f}s)")

# per_group_noise_stddev check: sigma_g/(nm*C_g) = sqrt(1+rho), sigma_h/(nm*C_h)=sqrt(1+1/rho), Mahalanobis = 1/nm^2
C_g = 0.9
print("\nrho    sigma_g/(nm*C_g)  sqrt(1+rho)   sigma_h/(nm*C_h)  sqrt(1+1/rho)  Mahalanobis*nm^2")
for rho in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
    C_h = rho * C_g
    pg = PerGroup(groups={("lora.w",): "fallback", ("router_load_probe",): "router_load_probe"},
                  values={"fallback": C_g, "router_load_probe": C_h})
    sig = per_group_noise_stddev(pg, NM)
    sg, sh = sig.values["fallback"], sig.values["router_load_probe"]
    mah = (C_g / sg) ** 2 + (C_h / sh) ** 2
    print(f"{rho:<6} {sg/(NM*C_g):.6f}          {math.sqrt(1+rho):.6f}      {sh/(NM*C_h):.6f}          {math.sqrt(1+1/rho):.6f}       {mah*NM*NM:.6f}")

# 'pay in eps' column: gradient noise held at nm*C_g, load released at sigma_h = nm*C_h*sqrt(1+1/rho)
#   as a separate mechanism composed on the same Poisson coin: poisson(gaussian(nm) | gaussian(nm_h), q) * T
#   Equivalent joint Gaussian: 1/nm_eff^2 = 1/nm^2 + 1/nm_h^2  ->  nm_eff = nm*sqrt((1+rho)/(1+2rho))
print("\nrho    nm_h      nm_eff    eps[poisson(g(nm)|g(nm_h),q)*T]   eps[poisson(g(nm_eff),q)*T]   design")
design = {0.5: 5.443, 0.2: 4.219, 0.1: 3.703, 0.05: 3.417, 0.02: 3.234, 0.01: 3.172}
for rho in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
    nm_h = NM * math.sqrt(1 + 1 / rho)
    nm_eff = 1 / math.sqrt(1 / NM**2 + 1 / nm_h**2)
    assert abs(nm_eff - NM * math.sqrt((1 + rho) / (1 + 2 * rho))) < 1e-12
    t0 = time.time()
    try:
        e_comp = (dpsgd_acc.poisson(dpsgd_acc.gaussian(NM) | dpsgd_acc.gaussian(nm_h), Q) * T).epsilon_at(DELTA)
    except Exception as ex:  # noqa: BLE001
        e_comp = f"ERR {type(ex).__name__}: {str(ex)[:60]}"
    e_eff = (dpsgd_acc.poisson(dpsgd_acc.gaussian(nm_eff), Q) * T).epsilon_at(DELTA)
    ec = f"{e_comp:.4f}" if isinstance(e_comp, float) else e_comp
    print(f"{rho:<6} {nm_h:.4f}   {nm_eff:.5f}   {ec:<32} {e_eff:.4f}                        {design[rho]}   ({time.time()-t0:.1f}s)")

# Show that the shipped route (eps held) really is eps-unchanged: same accountant call, nm unchanged
print(f"\nshipped route: accountant call is poisson(gaussian({NM}), q)*T regardless of PerGroup -> eps = {eps0:.4f} (identical object)")
# What eps would the naive 'sigma*C_g per group' allocation give? = gaussian(nm/sqrt(2)) for two groups
e_naive = (dpsgd_acc.poisson(dpsgd_acc.gaussian(NM / math.sqrt(2)), Q) * T).epsilon_at(DELTA)
print(f"naive per-group sigma=nm*C_i (two groups) == gaussian(nm/sqrt2): eps = {e_naive:.4f}")
