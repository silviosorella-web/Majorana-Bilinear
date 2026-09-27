"""Sanity check of the 'necessity' bound: for any dichotomic A in M(W_R) entering a CHSH
configuration with deficit eps = 2sqrt2 - <C>,   || (Delta^{1/2} - 1) A Omega ||^2 <= 2 sqrt2 eps (1 + O(sqrt eps)).
For a bilinear A = -i psi(x) psi(y) (x,y real-orthonormal): (Delta^{1/2}-1) A Omega = -i (delta^{1/2}x ^ delta^{1/2}y - x ^ y)."""
import numpy as np
from math import pi, sqrt
from sorella08_check import N0, bump
def run(a, M=200001):
    d = a; L = d + a + 0.05
    w = np.linspace(-L, L, M); dw = w[1]-w[0]
    ip = lambda u, v: np.sum(np.conj(u)*v)*dw
    ba = N0*bump(w/a)/sqrt(a); q = N0*bump((w-d)/a)/sqrt(a)
    S  = lambda u: np.exp(pi*w)*np.conj(u[::-1]); Sd = lambda u: np.exp(-pi*w)*np.conj(u[::-1])
    Al = lambda u: u + S(u); Bo = lambda u: 1j*(u + Sd(u))
    V = [Al(ba), Al(1j*ba), Al(q), Bo(ba), Bo(1j*ba), Bo(q)]
    V = [v/np.sqrt(ip(v, v).real) for v in V]
    f, fp, h, g, gp, p = V
    Im = np.array([[ip(u, v).imag for v in V] for u in V])
    corr = lambda x, y, z, t: Im[x,y]*Im[z,t] - Im[x,z]*Im[y,t] + Im[x,t]*Im[y,z]
    # Horodecki-optimal settings with shared primary Majoranas f (Alice), g (Bob):
    Tzz, t = corr(0,1,3,4), corr(0,2,3,5)          # <Z_A Z_B>, <X_A X_B>
    chi = np.arctan2(abs(t), abs(Tzz))
    # Alice: A = -i psi(f)psi(f'), A' = -i psi(f)psi(h); Bob: B,B' = -i psi(g) psi(w_pm), w_pm = c g' +- s p
    wp, wm = np.cos(chi)*gp + np.sin(chi)*p, np.cos(chi)*gp - np.sin(chi)*p
    def c2(x, y, z, t_):  # generic correlator with explicit vectors
        P = lambda u, v: ip(u, v)
        return (-(P(x,y)*P(z,t_) - P(x,z)*P(y,t_) + P(x,t_)*P(y,z))).real
    vals = {}
    for sA in (1,-1):
        for sB in (1,-1):
            AB, ApB = sA*c2(f,fp,g,wp), sB*c2(f,h,g,wp)
            ABp, ApBp = sA*c2(f,fp,g,wm), sB*c2(f,h,g,wm)
            vals[(sA,sB)] = (AB + ApB + ABp - ApBp, sA, sB)
    Bval, sA, sB = max(vals.values(), key=lambda z: z[0])
    eps = 2*sqrt(2) - Bval
    dh = np.exp(-pi*w)            # delta^{1/2}
    def wedge_norm2(x1, y1, x2, y2):   # <x1^y1, x2^y2>
        return (ip(x1,x2)*ip(y1,y2) - ip(x1,y2)*ip(y1,x2))
    def dist2(x, y):
        X, Y = dh*x, dh*y
        return (wedge_norm2(X,Y,X,Y) + wedge_norm2(x,y,x,y) - 2*wedge_norm2(X,Y,x,y).real).real
    return Bval, eps, dist2(f, fp), dist2(f, h), 2*sqrt(2)*eps
if __name__ == "__main__":
  print(" a      B_opt        eps        ||(D^1/2-1)A Omega||^2  ||(D^1/2-1)A' Omega||^2   bound 2sqrt2*eps")
  for a in [0.005, 0.01, 0.02, 0.05, 0.1]:
    Bv, eps, dA, dAp, bnd = run(a)
    print(f"{a:5.3f}  {Bv:.9f}  {eps:.3e}   {dA:.3e}               {dAp:.3e}                {bnd:.3e}")
