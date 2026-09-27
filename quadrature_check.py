"""Appendix C: stability of the Gauss-Hermite quadrature (80 vs 120 vs 200 nodes) for the
Bell parameter of random configurations in the polar parametrization."""
import numpy as np, torch
from math import pi
from mb_check import Setup, corr
def B_for(P, sigma, nodes, N=6):
    S = Setup(sigma, nmodes=N, N=nodes)
    phase = torch.tensor([1.0 if n % 2 == 0 else 0.0 for n in range(N)], dtype=torch.complex128) \
          + 1j * torch.tensor([0.0 if n % 2 == 0 else 1.0 for n in range(N)], dtype=torch.complex128)
    Bc = S.B.to(torch.complex128) * phase[:, None]
    ep2 = torch.exp(torch.tensor(pi * sigma * S.x.numpy() / 2)).to(torch.complex128)
    R = [P[k].to(torch.complex128) @ Bc for k in range(8)]
    f, fp, h, hp = [ep2 * r for r in R[:4]]; g, gp, p, pp = [1j * r / ep2 for r in R[4:]]
    h, hp, p, pp = S.gs(f, h), S.gs(fp, hp), S.gs(g, p), S.gs(gp, pp)
    return (corr(S, h, f, p, g) + corr(S, hp, fp, p, g) + corr(S, h, f, pp, gp) - corr(S, hp, fp, pp, gp)).real.item()
rng = np.random.default_rng(7)
for sigma in [1e-2, 1e-3]:
    P = torch.tensor(rng.normal(size=(8, 6)))
    b80, b120, b200 = B_for(P, sigma, 80), B_for(P, sigma, 120), B_for(P, sigma, 200)
    print(f"sigma = {sigma}: B(80) = {b80:.15f}  |B80-B120| = {abs(b80-b120):.1e}  |B80-B200| = {abs(b80-b200):.1e}")
