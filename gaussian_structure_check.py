"""Gaussian seeds (Sec. VIII): optimum in the polar parametrization with N = 6 Hermite
functions at sigma = 1e-3, with the real Gram matrices of the two parties (eigenvalues
{0,1,1,2}: three Majorana directions, orthogonal planes) and the four correlators."""
import numpy as np, torch
from math import pi, sqrt
from scipy.optimize import minimize
from mb_check import Setup, corr
torch.set_num_threads(4)
sigma, N, nstarts = 1e-3, 6, 12
S = Setup(sigma, nmodes=N, N=80)
phase = torch.tensor([1.0 if n % 2 == 0 else 0.0 for n in range(N)], dtype=torch.complex128) \
      + 1j * torch.tensor([0.0 if n % 2 == 0 else 1.0 for n in range(N)], dtype=torch.complex128)
Bc = S.B.to(torch.complex128) * phase[:, None]              # basis chi_0, i chi_1, chi_2, ...
ep2 = torch.exp(torch.tensor(pi * sigma * S.x.numpy() / 2)).to(torch.complex128)   # delta^{-1/4}
def vecs(P):
    R = [P[k].to(torch.complex128) @ Bc for k in range(8)]
    f, fp, h, hp = [ep2 * r for r in R[:4]]                 # Alice: delta^{-1/4} r
    g, gp, p, pp = [1j * r / ep2 for r in R[4:]]            # Bob: i delta^{1/4} r'
    return f, fp, S.gs(f, h), S.gs(fp, hp), g, gp, S.gs(g, p), S.gs(gp, pp)
def correlators(P):
    f, fp, h, hp, g, gp, p, pp = vecs(P)
    return [corr(S, h, f, p, g), corr(S, hp, fp, p, g), corr(S, h, f, pp, gp), corr(S, hp, fp, pp, gp)]
def fun(z):
    P = torch.tensor(z.reshape(8, N), requires_grad=True); c = correlators(P)
    v = -(c[0] + c[1] + c[2] - c[3]).real; v.backward()
    return v.item(), P.grad.numpy().ravel().copy()
rng = np.random.default_rng(3); best = None
for k in range(nstarts):
    r = minimize(fun, rng.normal(size=8 * N), jac=True, method="BFGS", options=dict(gtol=1e-14, maxiter=5000))
    if best is None or r.fun < best.fun: best = r
P = torch.tensor(best.x.reshape(8, N)); f, fp, h, hp, g, gp, p, pp = vecs(P)
nrm = lambda u: u / torch.sqrt(S.ip(u, u).real)
gram = lambda vs: np.array([[S.ip(nrm(u), nrm(v)).real.item() for v in vs] for u in vs])
print(f"B = {-best.fun:.10f}   deficit/sigma^2 = {(2*sqrt(2)+best.fun)/sigma**2:.3f}")
print("correlators <AB>, <A'B>, <AB'>, <A'B'> =", np.round([c.real.item() for c in correlators(P)], 7))
print("Alice real Gram eigenvalues:", np.round(np.linalg.eigvalsh(gram([f, fp, h, hp])), 6))
print("Bob   real Gram eigenvalues:", np.round(np.linalg.eigvalsh(gram([g, gp, p, pp])), 6))
