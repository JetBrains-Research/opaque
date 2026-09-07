"""Auditor: run ONLY the V6 band-MF arm from the patched prototype (paper-p b-min-sep + explicit latch check)."""
import sys, json, time
sys.path.insert(0, "/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/auditor")
import prototype_load_release_patched as P
r = json.load(open("/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/auditor/rerun_results.json"))
C_g, NM, rho_star = r["calibration"]["C_g"], r["rho_star"]["NM"], r["rho_star"]["rho_star"]
cfg = dict(C_g=C_g, n_steps=P.N_STEPS_MF, router_scale=P.ROUTER_SCALE, noise="mf", seed=11, name="MF_DP_rho0.34_paperp",
           alpha=P.ALPHA_LAB, rho=rho_star, nm=NM, f_mode="dp")
t0 = time.time(); res = P.run_arm(cfg); s = P.summarize_arm(res)
orig = [a for a in r["arms"] if a["noise"] == "mf"][0]
keys = ("batch_size_mean", "delta_heldout_final", "mean_cos", "mean_cos_pop", "dead_zone_rate", "clip_rate_mean", "probe_clip_events", "s_inf_over_kE", "pooled_noise_std_empirical", "pooled_noise_std_predicted")
print(f"{'metric':28s} {'prototype(p=q)':>16s} {'patched(paper p)':>18s}")
for k in keys:
    print(f"{k:28s} {orig[k]:16.4f} {s[k]:18.4f}")
print("latch_ok (explicit check):", res["mf_latch_ok"], "| sigma relerr probe/fallback:", f"{res['mf_sigma_relerr_probe']:.2e}", f"{res['mf_sigma_relerr_fallback']:.2e}",
      "| probe noise emp/pred:", f"{res['mf_probe_noise_empirical_std_over_pred']:.3f}")
print("delta_heldout:", [round(x, 3) for x in s["delta_heldout"]], "| elapsed", round(time.time() - t0, 1), "s")
