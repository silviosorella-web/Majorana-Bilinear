"""Well-conditioned version of the variational problem: parametrize Alice's and Bob's
Majorana vectors directly in the polar form
    u = delta^{-1/4} r = e^{+pi w/2} r ,   v = i delta^{1/4} r' = i e^{-pi w/2} r' ,
    r, r' in R_N = real span{h0, i h1, h2, i h3, ...}  (N Hermite functions of width sigma).
General bilinears A=-i psi(h)psi(f), A'=-i psi(h')psi(f') (4 independent Alice vectors, 4 Bob)."""
import sys, json, numpy as np, torch
from math import pi, sqrt
from scipy.optimize import minimize
from mb_check import Setup, corr
torch.set_num_threads(1)
def run(sigma, N, nstarts=20, seed=0):
    S = Setup(sigma, nmodes=N, N=max(80, 2*N+40))
    phase = torch.tensor([1.0 if n % 2 == 0 else 0.0 for n in range(N)], dtype=torch.complex128) \
          + 1j*torch.tensor([0.0 if n % 2 == 0 else 1.0 for n in range(N)], dtype=torch.complex128)
    Bc = S.B.to(torch.complex128) * phase[:, None]         # rows: h0, i h1, h2, i h3, ...
    ep2 = torch.exp(torch.tensor(pi*sigma*S.x.numpy()/2)).to(torch.complex128)
    def vecs(P):
        R = [P[k].to(torch.complex128) @ Bc for k in range(8)]
        f, fp, h, hp = [ep2*r for r in R[:4]]
        g, gp, p, pp = [1j*r/ep2 for r in R[4:]]
        h, hp, p, pp = S.gs(f, h), S.gs(fp, hp), S.gs(g, p), S.gs(gp, pp)
        return f, fp, h, hp, g, gp, p, pp
    def chsh(P):
        f, fp, h, hp, g, gp, p, pp = vecs(P)
        return (corr(S,h,f,p,g) + corr(S,hp,fp,p,g) + corr(S,h,f,pp,gp) - corr(S,hp,fp,pp,gp)).real
    def fun(z):
        P = torch.tensor(z.reshape(8, N), requires_grad=True); v = -chsh(P); v.backward()
        return v.item(), P.grad.numpy().ravel().copy()
    rng = np.random.default_rng(seed); best = None
    for k in range(nstarts):
        r = minimize(fun, rng.normal(size=8*N), jac=True, method='BFGS', options=dict(gtol=1e-14, maxiter=5000))
        if best is None or r.fun < best.fun: best = r
    B = -best.fun
    return dict(sigma=sigma, N=N, B=B, deficit=2*sqrt(2)-B, coef=(2*sqrt(2)-B)/sigma**2)
if __name__ == "__main__":
    sigma = float(sys.argv[1]); N = int(sys.argv[2]); ns = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    print(json.dumps(run(sigma, N, ns)))
