"""Compare base.pt vs branch.pt parity dumps; print a numeric report."""

import sys

import torch

base = torch.load(sys.argv[1], weights_only=False)
br = torch.load(sys.argv[2], weights_only=False)

print(f"torch versions: base={base['torch_version']} branch={br['torch_version']}\n")


def tdiff(a, b):
    if a.shape != b.shape:
        return None, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return None, f"dtype {a.dtype} vs {b.dtype}"
    a64, b64 = a.double(), b.double()
    mad = (a64 - b64).abs().max().item()
    denom = b64.abs().max().item() or 1.0
    return mad, f"max_abs_diff={mad:.3e} rel={mad / denom:.3e}"


def walk(a, b, path, stats):
    if torch.is_tensor(a) and torch.is_tensor(b):
        mad, msg = tdiff(a, b)
        stats.append((path, mad, msg))
    elif isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        for k in keys:
            if k not in a or k not in b:
                stats.append((f"{path}.{k}", None, "missing on one side"))
            else:
                walk(a[k], b[k], f"{path}.{k}", stats)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            stats.append((path, None, f"len {len(a)} vs {len(b)}"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", stats)
    else:
        eq = a == b
        stats.append((path, 0.0 if eq else None, "EQUAL" if eq else f"{a!r} vs {b!r}"))


for section in [
    "key_derivation",
    "generator_streams",
    "clip_fixed",
    "clip_autos",
    "gaussian_noise_stream",
    "optimizers",
    "e2e_dpsgd",
    "accounting",
    "dpftrl_mf",
]:
    if section not in base or section not in br:
        print(f"== {section}: MISSING ({section in base=} {section in br=})")
        continue
    stats = []
    walk(base[section], br[section], section, stats)
    n = len(stats)
    exact = sum(1 for _, m, _ in stats if m == 0.0)
    close = sum(1 for _, m, _ in stats if m is not None and 0 < m <= 1e-6)
    far = [(p, m, s) for p, m, s in stats if m is None or m > 1e-6]
    worst = max((m for _, m, _ in stats if m is not None), default=0.0)
    print(f"== {section}: {n} comparisons | bit-exact {exact} | <=1e-6 {close} | mismatched {len(far)} | worst {worst:.3e}")
    for p, m, s in far[:12]:
        print(f"   MISMATCH {p}: {s}")
    if len(far) > 12:
        print(f"   ... {len(far) - 12} more mismatches")
print()
