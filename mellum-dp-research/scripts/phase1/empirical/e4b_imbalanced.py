"""E4b: same as E4 but with an artificially imbalanced router (rows 0,1 of every gate x3),
so f(B) - k/E is a persistent signal rather than sampling noise."""
import sys, re
src = open("/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical/e4_aux.py").read()
src = src.replace('model = build_unpatched(seed=0); model.train()',
 'model = build_unpatched(seed=0); model.train()\nwith torch.no_grad():\n    for n_, p_ in model.named_parameters():\n        if ".mlp.gate." in n_: p_[0:2].mul_(3.0)')
src = src.replace('dump("e4_results.json", res)', 'dump("e4b_imbalanced_results.json", res)')
src = src.replace('for kind in ["random", "structured", "random_padded"]:', 'for kind in ["random", "structured"]:')
exec(compile(src, "e4b", "exec"))
