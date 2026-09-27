import sys, json, torch, numpy as np
torch.set_num_threads(1)
from mb_check import optimize
sig = float(sys.argv[1]); mode = sys.argv[2]; ns = int(sys.argv[3]); seed = int(sys.argv[4])
out = optimize(sig, mode=mode, nstarts=ns, seed=seed)
res = {k: v for k, v in out.items() if k not in ("x", "S", "vecs")}
res["x"] = out["x"].tolist()
json.dump(res, open(f"res_{mode}_{sig:.0e}_s{seed}.json", "w"), indent=1)
print(sig, mode, seed, res["B"], res["deficit"])
