"""Finite-dimensional check with sharp boost-frequency modes.
Mode k (frequency w_k>0) contributes 2 Alice Majoranas and 2 Bob Majoranas with
 Gamma_AA = t_k J, Gamma_BB = -t_k J, Gamma_AB = c_k 1,  t=tanh(pi w), c=sech(pi w)
(derived from the polar form H = delta^{-1/4} R). Correlator of bilinears = Pfaffian.
We (i) build the covariance directly from wave packets on a grid and compare, and
(ii) optimise ALL bilinears (4+4 Majoranas) to compare with 2 sqrt(1 + c1^2 c2^2)."""
import numpy as np, torch
from math import pi, sqrt, tanh, cosh
from scipy.optimize import minimize
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

def gamma_sharp(ws):
    n = len(ws); G = np.zeros((4*n, 4*n))   # order: Alice (a_k1,a_k2)_k, then Bob (b_k1,b_k2)_k
    for k, w in enumerate(ws):
        t, c = tanh(pi*w), 1/cosh(pi*w)
        a1, a2, b1, b2 = 2*k, 2*k+1, 2*n+2*k, 2*n+2*k+1
        G[a1,a2], G[a2,a1] = t, -t
        G[b1,b2], G[b2,b1] = -t, t
        G[a1,b1], G[b1,a1] = c, -c
        G[a2,b2], G[b2,a2] = c, -c
    return G

def gamma_from_packets(ws, eta=1e-3, M=200001, L=None):
    """Build Alice/Bob Majorana vectors from narrow Gaussian packets at +-w_k on a grid,
    using the polar form u = e^{pi w/2} r, v = i e^{-pi w/2} r, r in R = {conj(r(-w)) = r(w)}."""
    L = L or (max(ws) + 12*eta)
    w = np.linspace(-L, L, M); dw = w[1]-w[0]
    ip = lambda u, v: np.sum(np.conj(u)*v)*dw
    A, B = [], []
    for wk in ws:
        chi_p = np.exp(-(w-wk)**2/(4*eta**2)); chi_m = chi_p[::-1]
        e = (chi_p + chi_m); o = 1j*(chi_p - chi_m)      # both in R
        for r in (e, o):
            A.append(np.exp(pi*w/2)*r); B.append(1j*np.exp(-pi*w/2)*r)
    def on(vs):
        out = []
        for v in vs:
            for u in out: v = v - ip(u, v).real*u
            out.append(v/np.sqrt(ip(v, v).real))
        return out
    V = on(A) + on(B)
    G = np.array([[ip(u, v) for v in V] for u in V])
    return G.imag, np.abs(G.real - np.eye(len(V))).max()

def pf_corr(G, x, y, z, w):
    return G[x,y]*G[z,w] - G[x,z]*G[y,w] + G[x,w]*G[y,z]

def best_bilinear_chsh(G, nA, nstart=30, seed=0):
    """maximise CHSH over all bilinears -i psi(x)psi(y) with x,y orthonormal in Alice's R^nA, same for Bob."""
    Gt = torch.tensor(G); n = G.shape[0]; nB = n - nA
    rng = np.random.default_rng(seed)
    def frame(v1, v2):
        v1 = v1/torch.linalg.norm(v1); v2 = v2 - (v1@v2)*v1; v2 = v2/torch.linalg.norm(v2)
        return v1, v2
    def emb(v, side):
        z = torch.zeros(n, dtype=torch.float64)
        return torch.cat([v, torch.zeros(nB)]) if side == 'A' else torch.cat([torch.zeros(nA), v])
    def corr(xa, ya, zb, wb):
        x, y, z, w = emb(xa,'A'), emb(ya,'A'), emb(zb,'B'), emb(wb,'B')
        g = lambda a, b: a @ Gt @ b
        return g(x,y)*g(z,w) - g(x,z)*g(y,w) + g(x,w)*g(y,z)
    def fun(p):
        P = torch.tensor(p, requires_grad=True)
        q = P.reshape(8, -1)
        A1 = frame(q[0,:nA], q[1,:nA]); A2 = frame(q[2,:nA], q[3,:nA])
        B1 = frame(q[4,:nB], q[5,:nB]); B2 = frame(q[6,:nB], q[7,:nB])
        val = corr(*A1,*B1) + corr(*A2,*B1) + corr(*A1,*B2) - corr(*A2,*B2)
        (-val).backward(); return -val.item(), P.grad.numpy().copy()
    best = 0
    for k in range(nstart):
        r = minimize(fun, rng.normal(size=8*max(nA,nB)), jac=True, method='BFGS', options=dict(gtol=1e-12))
        best = max(best, -r.fun)
    return best

if __name__ == "__main__":
    for ws in [(0.05, 0.08), (0.1, 0.1), (0.2, 0.3), (0.3, 0.5)]:
        G = gamma_sharp(ws)
        # the packet comparison needs distinct frequencies (equal ones give coinciding packets)
        Gp, metric_err = gamma_from_packets(ws) if ws[0] != ws[1] else (None, None)
        c1, c2 = [1/cosh(pi*w) for w in ws]
        formula = 2*sqrt(1 + c1**2*c2**2)
        # 3+3 Majoranas: Alice a11,a12,a21 ; Bob b11,b12,b21  (indices 0,1,2 ; 4,5,6)
        idx = [0,1,2,4,5,6]; G3 = G[np.ix_(idx, idx)]
        b3 = best_bilinear_chsh(G3, 3)
        b4 = best_bilinear_chsh(G, 4)
        pk = f"{np.abs(Gp-G).max():.1e} (metric err {metric_err:.1e})" if Gp is not None else "n/a (equal frequencies)"
        print(f"w={ws}: |Gamma_packets - Gamma_sharp|max={pk}"
              f" | formula 2sqrt(1+c1^2c2^2)={formula:.10f} | optimum 3+3 bilinears={b3:.10f} | optimum 4+4 bilinears={b4:.10f}")
