"""Independent re-implementation of the Majorana-bilinear CHSH construction
(Caribe, Guimaraes, Roditi, Sorella, v5/v6), written from the equations of the
manuscript only.  Modular-frequency variable omega = sigma * x; Gauss-Hermite
quadrature in x.  A one-particle vector u is stored as U_i = exp(x_i^2/2) u(sigma x_i),
so <u,v> = sigma * sum_i w_i conj(U_i) V_i (the factor sigma cancels everywhere).
"""
import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from math import factorial, sqrt, pi
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)
CD = torch.complex128


class Setup:
    def __init__(self, sigma, nmodes=6, N=80):
        x, w = hermgauss(N)
        self.sigma, self.nmodes, self.N = sigma, nmodes, N
        self.x = torch.tensor(x)
        self.w = torch.tensor(w)
        H = [np.ones_like(x), 2 * x]
        for n in range(2, nmodes):
            H.append(2 * x * H[-1] - 2 * (n - 1) * H[-2])
        B = np.array([H[n] / sqrt(2.0**n * factorial(n)) for n in range(nmodes)])
        self.B = torch.tensor(B)                       # (nmodes, N)
        self.ep = torch.exp(torch.tensor(pi * sigma * x)).to(CD)   # e^{+pi omega}
        self.em = torch.exp(torch.tensor(-pi * sigma * x)).to(CD)  # e^{-pi omega}
        self.rev = torch.arange(N - 1, -1, -1)        # index of -x_i

    def seed(self, c):
        return (c @ self.B).to(CD)

    def s(self, U):   # (s u)(w) = e^{pi w} conj(u(-w))
        return self.ep * torch.conj(U[self.rev])

    def sd(self, U):  # (s^dagger u)(w) = e^{-pi w} conj(u(-w))
        return self.em * torch.conj(U[self.rev])

    def ip(self, U, V):
        return torch.sum(self.w * torch.conj(U) * V)

    def L(self, U):   # 1 + s   (Alice, right wedge)
        return U + self.s(U)

    def R(self, U):   # 1 + s^dagger
        return U + self.sd(U)

    def gs(self, u, v):  # real ("symplectic" in the paper) Gram-Schmidt
        return v - (self.ip(u, v).real / self.ip(u, u).real) * u


I = 1j


def paper_vectors(S, P):
    """P: (10, nmodes) real seeds  phi, l, eta, l', eta', ghat, alpha, beta, alpha', beta'.
    Exactly Eqs. (29)-(44) of v6 followed by the Gram-Schmidt of Eq. (45)."""
    phi, l, eta, lp, etap, gh, al, be, alp, bep = [S.seed(P[k]) for k in range(10)]
    f, fp = S.L(phi), S.L(I * phi)
    h, hp = S.L(l + I * eta), S.L(lp + I * etap)
    g, gp = I * S.R(gh), -I * S.R(I * gh)
    p, pp = I * S.R(al + I * be), -I * S.R(alp + I * bep)
    h, hp, p, pp = S.gs(f, h), S.gs(fp, hp), S.gs(g, p), S.gs(gp, pp)
    return f, fp, h, hp, g, gp, p, pp


def general_vectors(S, P):
    """P: (16, nmodes): eight independent complex seeds; same localization maps.
    This is the unrestricted bilinear problem inside the same 6-mode space."""
    Z = [S.seed(P[2 * k]) + I * S.seed(P[2 * k + 1]) for k in range(8)]
    f, fp, h, hp = S.L(Z[0]), S.L(Z[1]), S.L(Z[2]), S.L(Z[3])
    g, gp, p, pp = I * S.R(Z[4]), I * S.R(Z[5]), I * S.R(Z[6]), I * S.R(Z[7])
    h, hp, p, pp = S.gs(f, h), S.gs(fp, hp), S.gs(g, p), S.gs(gp, pp)
    return f, fp, h, hp, g, gp, p, pp


def corr(S, h, f, p, g):
    """<(-i psi(h)psi(f)/|h||f|) (-i psi(p)psi(g)/|p||g|)>  via Eq. (55)."""
    ip = S.ip
    n = torch.sqrt(ip(h, h).real * ip(f, f).real * ip(p, p).real * ip(g, g).real)
    return -(ip(h, f) * ip(p, g) - ip(h, p) * ip(f, g) + ip(h, g) * ip(f, p)) / n


def chsh_from_vectors(S, vecs):
    f, fp, h, hp, g, gp, p, pp = vecs
    AB, ApB = corr(S, h, f, p, g), corr(S, hp, fp, p, g)
    ABp, ApBp = corr(S, h, f, pp, gp), corr(S, hp, fp, pp, gp)
    return AB + ApB + ABp - ApBp, (AB, ApB, ABp, ApBp)


def diagnostics(S, vecs):
    f, fp, h, hp, g, gp, p, pp = vecs
    alice, bob = [f, fp, h, hp], [g, gp, p, pp]
    loc = max(abs(S.ip(u, v).real.item()) / sqrt(S.ip(u, u).real.item() * S.ip(v, v).real.item())
              for u in alice for v in bob)
    pairs = [(f, h), (fp, hp), (g, p), (gp, pp)]
    car = max(abs(S.ip(u, v).real.item()) for u, v in pairs)
    auto = (S.ip(f, fp).real.item() / sqrt(S.ip(f, f).real.item() * S.ip(fp, fp).real.item()),
            S.ip(g, gp).real.item() / sqrt(S.ip(g, g).real.item() * S.ip(gp, gp).real.item()))
    # modular localization check: s f = f for Alice, s^dagger g = -g for Bob
    sloc = max((S.s(u) - u).abs().max().item() / u.abs().max().item() for u in alice)
    sdloc = max((S.sd(v) + v).abs().max().item() / v.abs().max().item() for v in bob)
    return dict(max_rel_Re_alice_bob=loc, max_intra_pair_Re=car,
                Re_f_fp_normalized=auto[0], Re_g_gp_normalized=auto[1],
                s_fixed_point_residual=sloc, sdagger_minus1_residual=sdloc)


def optimize(sigma, mode="paper", nstarts=40, seed=0, nmodes=6, N=80, tol=1e-15, maxiter=4000):
    S = Setup(sigma, nmodes=nmodes, N=N)
    nseeds = 10 if mode == "paper" else 16
    build = paper_vectors if mode == "paper" else general_vectors
    rng = np.random.default_rng(seed)

    def fun(z):
        P = torch.tensor(z.reshape(nseeds, nmodes), requires_grad=True)
        val, _ = chsh_from_vectors(S, build(S, P))
        loss = -val.real
        loss.backward()
        return loss.item(), P.grad.numpy().ravel().copy()

    best = None
    for k in range(nstarts):
        z0 = rng.normal(size=nseeds * nmodes)
        r = minimize(fun, z0, jac=True, method="BFGS",
                     options=dict(gtol=tol, maxiter=maxiter))
        if best is None or r.fun < best.fun:
            best = r
    P = torch.tensor(best.x.reshape(nseeds, nmodes))
    vecs = build(S, P)
    val, cs = chsh_from_vectors(S, vecs)
    return dict(sigma=sigma, mode=mode, B=val.real.item(), ImB=val.imag.item(),
                deficit=2 * sqrt(2) - val.real.item(),
                correlators=[c.real.item() for c in cs],
                diag=diagnostics(S, vecs), x=best.x, S=S, vecs=vecs)


if __name__ == "__main__":
    import sys, json
    sig = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
    mode = sys.argv[2] if len(sys.argv) > 2 else "paper"
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    out = optimize(sig, mode=mode, nstarts=ns)
    print(json.dumps({k: v for k, v in out.items() if k not in ("x", "S", "vecs")}, indent=1))
