"""Statistical comparison of dist_run dumps (base vs branch)."""

import sys

import torch
from scipy import stats

base = torch.load(sys.argv[1], weights_only=False)
br = torch.load(sys.argv[2], weights_only=False)
assert base["side"] == "base" and br["side"] == "branch"

print("=== Gaussian mechanism, nm=1.3, sens=1.0 -> expected iid N(0, 1.69) ===")
for name, d in (("base", base), ("branch", br)):
    s = d["gaussian_samples"].double().flatten()
    print(f"{name:7s} n={s.numel()} mean={s.mean():+.5f} std={s.std():.5f} (expect 1.30000)")
a = base["gaussian_samples"].double().flatten().numpy()
b = br["gaussian_samples"].double().flatten().numpy()
ks = stats.ks_2samp(a, b)
print(f"two-sample KS: stat={ks.statistic:.5f} p={ks.pvalue:.4f}")
sw_a = stats.kstest(a, "norm", args=(0, 1.3))
sw_b = stats.kstest(b, "norm", args=(0, 1.3))
print(f"KS vs N(0,1.3^2): base p={sw_a.pvalue:.4f} branch p={sw_b.pvalue:.4f}")

print("\n=== MF (band_mf, bands=4, n=8) noise: marginal stds per step ===")
sb = base["mf_streams"].double()  # 300 x 8 x 16
sr = br["mf_streams"].double()
std_b = sb.std(dim=(0, 2))
std_r = sr.std(dim=(0, 2))
for i in range(8):
    print(f" step {i}: base={std_b[i]:.4f} branch={std_r[i]:.4f} ratio={std_r[i] / std_b[i]:.4f}")

print("\n=== MF cross-step correlation matrices (avg over coords) ===")


def corr(s):
    # s: R x 8 x 16 -> correlation across steps, averaged over coordinates
    x = s.permute(2, 1, 0)  # 16 x 8 x 300
    mats = []
    for c in range(x.shape[0]):
        mats.append(torch.corrcoef(x[c]))
    return torch.stack(mats).mean(0)


cb, cr = corr(sb), corr(sr)
diff = (cb - cr).abs()
print("base corr row0:  ", " ".join(f"{v:+.3f}" for v in cb[0]))
print("branch corr row0:", " ".join(f"{v:+.3f}" for v in cr[0]))
print(f"max |corr diff| over 8x8: {diff.max():.4f}  mean: {diff.mean():.4f}")
se = (1 / (300 - 3)) ** 0.5  # Fisher-z SE for n=300
print(f"(Fisher-z 3*SE at n=300 ~= {3 * se:.4f} -> diffs below this are sampling noise)")

print("\n=== adafactor probe (max |param diff| final step) ===")
for k in ("wd0", "wd01"):
    tb = base["adafactor_probe"][k][-1]
    tr = br["adafactor_probe"][k][-1]
    m = max((x.double() - y.double()).abs().max().item() for x, y in zip(tb, tr))
    print(f" {k}: {m:.3e}")
