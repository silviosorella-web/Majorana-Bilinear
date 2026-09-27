"""Sec. VIII: structure of the optimum when the vectors are generated from Hermite-Gaussian
seeds as in the v5/v6 construction (f = (1+s)phi, f' = (1+s) i phi).  Reads the optimum
saved by scan.py (results/res_paper_1e-03_s1.json) and prints the real Gram matrix of Alice's
vectors: eigenvalues {0,1,1,2} with Re<h,h'> = 0 mean that the second factor of one observable
equals a factor of the other, so that one observable is the parity -i psi(f)psi(f')."""
import json, sys, numpy as np, torch
from mb_check import Setup, paper_vectors, chsh_from_vectors
fn = sys.argv[1] if len(sys.argv) > 1 else "../results/res_paper_1e-03_s1.json"
r = json.load(open(fn)); S = Setup(r["sigma"])
P = torch.tensor(np.array(r["x"]).reshape(10, 6))
f, fp, h, hp, g, gp, p, pp = paper_vectors(S, P)
nrm = lambda u: u / torch.sqrt(S.ip(u, u).real)
A = [nrm(u) for u in (f, fp, h, hp)]
GA = np.array([[S.ip(u, v).real.item() for v in A] for u in A])
val, _ = chsh_from_vectors(S, (f, fp, h, hp, g, gp, p, pp))
print(f"sigma = {r['sigma']}: B = {val.real.item():.10f}")
print("Alice real Gram matrix (order f, f', h, h'):\n", np.round(GA, 6))
print("eigenvalues:", np.round(np.linalg.eigvalsh(GA), 6))
