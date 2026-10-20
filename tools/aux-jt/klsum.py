import json, sys, os
L = open(os.path.expanduser("~/jt/kl.log")).read().splitlines()
s = [l for l in L if "SUMMARY" in l]
row = [l for l in L if " row 1/1" in l]
if not s:
    print(sys.argv[1], "FAILED", L[-3:]); sys.exit()
d = json.loads(s[-1].split("SUMMARY ")[1])
r = row[-1].split(" I ")[-1][:60] if row else ""
print(f"{sys.argv[1]:55s} KL mean {d['kl_mean']:.5f} p99 {d['kl_p99']:.4f} max {d['kl_max']:.3f} top1 {d['top1_agree']:.4f} | {r}")
