"""Checks of the analytic picture:
 (1) polar form  H(W_R) = delta^{-1/4} R,  Bob's space = i delta^{1/4} R,
     R = { r : conj(r(-w)) = r(w) }   (even real  +  i * odd real)
 (2) three Majoranas per side + Horodecki  ->  B = 2 sqrt(m1^2 + m2^2)
 (3) deficit ~ kappa * sigma^2 (clean power law)
"""
import numpy as np, torch
from math import sqrt, pi
from mb_check import Setup

torch.set_num_threads(1)


def real_on(S, vecs):
    """Gram-Schmidt w.r.t. Re<.,.> (keeps modular localization: real combinations)."""
    out = []
    for v in vecs:
        for u in out:
            v = v - S.ip(u, v).real * u
        out.append(v / torch.sqrt(S.ip(v, v).real))
    return out


def bil(S, x, y, z, w):
    ip = S.ip
    return (-(ip(x, y) * ip(z, w) - ip(x, z) * ip(y, w) + ip(x, w) * ip(y, z))).real.item()


def horodecki(S, Avec, Bvec):
    """Alice: 3 real-orthonormal vectors, Bob: 3.  sigma_a = -i psi_b psi_c (abc cyclic)."""
    cyc = [(1, 2), (2, 0), (0, 1)]
    M = np.array([[bil(S, Avec[b], Avec[c], Bvec[e], Bvec[f]) for (e, f) in cyc] for (b, c) in cyc])
    m = np.linalg.svd(M, compute_uv=False)
    return 2 * sqrt(m[0] ** 2 + m[1] ** 2), M, m


def hermite_block_eig(nmodes=6):
    """compressed x^2 on even and odd Hermite subspaces (units of sigma^2)"""
    X = np.zeros((nmodes, nmodes))
    for n in range(nmodes - 1):
        X[n, n + 1] = X[n + 1, n] = sqrt((n + 1) / 2)
    X2 = np.zeros((nmodes, nmodes))
    # exact x^2 compressed (not (compressed x)^2): include the n -> n+2 leak properly
    for n in range(nmodes):
        X2[n, n] = n + 0.5
        if n + 2 < nmodes:
            X2[n, n + 2] = X2[n + 2, n] = sqrt((n + 1) * (n + 2)) / 2
    ev = [i for i in range(nmodes) if i % 2 == 0]
    od = [i for i in range(nmodes) if i % 2 == 1]
    ee, Ve = np.linalg.eigh(X2[np.ix_(ev, ev)])
    eo, Vo = np.linalg.eigh(X2[np.ix_(od, od)])
    return ee, Ve, ev, eo, Vo, od


def explicit_config(sigma, nmodes=6, N=80):
    S = Setup(sigma, nmodes=nmodes, N=N)
    ee, Ve, ev, eo, Vo, od = hermite_block_eig(nmodes)
    xs = S.x
    half = torch.exp(pi * sigma * xs / 2).to(torch.complex128)

    def fn(coeffs, idx):
        c = torch.zeros(nmodes, dtype=torch.float64)
        c[idx] = torch.tensor(coeffs)
        return S.seed(c)
    # R elements: even real (e1, e2) and i*odd real (o1)
    e1, e2 = fn(Ve[:, 0], ev), fn(Ve[:, 1], ev)
    o1 = 1j * fn(Vo[:, 0], od)
    Rtriple = [e1, o1, e2]
    # polar form: Alice u = e^{+pi w/2} r,  Bob v = i e^{-pi w/2} r
    A = [half * r for r in Rtriple]
    B = [1j * r / half for r in Rtriple]
    # sanity: s u = u, s^dagger v = -v
    res_s = max((S.s(u) - u).abs().max().item() for u in A)
    res_sd = max((S.sd(v) + v).abs().max().item() for v in B)
    A, B = real_on(S, A), real_on(S, B)
    Bm, M, m = horodecki(S, A, B)
    return dict(sigma=sigma, B=Bm, deficit=2 * sqrt(2) - Bm, sv=m.tolist(),
                res_s=res_s, res_sd=res_sd, nu=(ee[0], eo[0], ee[1]))


if __name__ == "__main__":
    ee, Ve, ev, eo, Vo, od = hermite_block_eig(6)
    print("compressed x^2 eigenvalues  even:", np.round(ee, 5), " odd:", np.round(eo, 5))
    for s in [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]:
        r = explicit_config(s)
        print(f"sigma={s:.0e}  B={r['B']:.12f}  deficit={r['deficit']:.4e}  deficit/sigma^2={r['deficit']/s**2:.4f}"
              f"  sv={np.round(r['sv'],8)}  [s-res {r['res_s']:.1e}, s+-res {r['res_sd']:.1e}]")
    nu1, nu2, nu3 = sorted([ee[0], eo[0], ee[1]])
    print("leading-order prediction (no tau term): pi^2/sqrt2*(2nu1+nu2+nu3) =",
          pi**2 / sqrt(2) * (2 * nu1 + nu2 + nu3))
