"""Check of Sorella's Drive note #08 (compact modular seeds) and of the Horodecki-optimal
settings for the SAME six Majorana vectors (f, f', h ; g, g', p)."""
import numpy as np
from math import pi, sqrt, cosh
from scipy.integrate import quad
from scipy.optimize import brentq

def bump(x):
    x = np.asarray(x, float); out = np.zeros_like(x); m = (x > 0) & (x < 1)
    out[m] = np.exp(-1.0/(x[m]*(1-x[m]))); return out
N0 = 1/sqrt(quad(lambda x: bump(x)**2, 0, 1, epsabs=1e-15, epsrel=1e-13, limit=200)[0])
mu2 = quad(lambda x: (x-0.5)**2*(N0*bump(x))**2, 0, 1, epsabs=1e-15, limit=200)[0]
I = lambda a: quad(lambda x: np.cosh(2*pi*a*(x-0.5))*(N0*bump(x))**2, 0, 1, epsabs=1e-15, limit=200)[0]

def GF(a, d):
    Ia = I(a)
    G = 2/sqrt(1 + 2*Ia*cosh(pi*a) + Ia**2)
    F = 2/sqrt(1 + 2*Ia*cosh(pi*(2*d+a)) + Ia**2)
    al, be = np.exp(-pi*a)*Ia, np.exp(pi*a)*Ia
    Tzz = -(3 + al + be - al*be)/(1 + al + be + al*be)
    return G, F, Tzz

# ---- direct grid construction of the six vectors (independent of the closed forms) ----
def grid_check(a, d, M=400001):
    L = d + a + 0.05
    w = np.linspace(-L, L, M); dw = w[1]-w[0]
    ip = lambda u, v: np.sum(np.conj(u)*v)*dw
    ba = N0*bump(w/a)/sqrt(a); q = N0*bump((w-d)/a)/sqrt(a)
    S  = lambda u: np.exp(pi*w)*np.conj(u[::-1])
    Sd = lambda u: np.exp(-pi*w)*np.conj(u[::-1])
    Al = lambda u: u + S(u); Bo = lambda u: 1j*(u + Sd(u))
    f, fp, h = Al(ba), Al(1j*ba), Al(q)
    g, gp, p = Bo(ba), Bo(1j*ba), Bo(q)
    vec = [f, fp, h, g, gp, p]; vec = [v/np.sqrt(ip(v, v).real) for v in vec]
    Gm = np.array([[ip(u, v) for v in vec] for u in vec])
    Re, Im = Gm.real, Gm.imag
    offRe = np.abs(Re - np.eye(6)).max()           # all six real-orthonormal? (dichotomy+locality)
    def corr(x, y, z, t):   # <(-i psi_x psi_y)(-i psi_z psi_t)> = Pf(Im) on (x,y,z,t)
        return Im[x,y]*Im[z,t] - Im[x,z]*Im[y,t] + Im[x,t]*Im[y,z]
    # Pauli triples: sigma_a = -i psi_b psi_c,  (b,c) = (1,2),(2,0),(0,1) within (f,fp,h) and (g,gp,p)
    cyc = [(1,2),(2,0),(0,1)]
    T = np.array([[corr(b, c, 3+e, 3+ff) for (e, ff) in cyc] for (b, c) in cyc])
    sv = np.linalg.svd(T, compute_uv=False)
    return offRe, T, 2*sqrt(sv[0]**2 + sv[1]**2)

if __name__ == "__main__":
    print(f"N0 = {N0:.4f}   mu2 = {mu2:.7f}")
    print(" a      G        F        Tzz       B_Sorella=2sqrt2 GF   B_opt=2sqrt(Tzz^2+(GF)^2)   [grid: offRe, B_opt]")
    for a in [0.001, 0.01, 0.04, 0.07, 0.10, 0.15, 0.1736, 0.25, 0.4]:
        G, F, Tzz = GF(a, a)
        Bs, Bo = 2*sqrt(2)*G*F, 2*sqrt(Tzz**2 + (G*F)**2)
        extra = ""
        if a >= 0.01:
            offRe, T, Bg = grid_check(a, a)
            extra = f"[{offRe:.1e}, {Bg:.6f}]"
        print(f"{a:6.4f} {G:.6f} {F:.6f} {Tzz:+.6f}   {Bs:.6f}              {Bo:.6f}          {extra}")
    acrit_S = brentq(lambda a: 2*sqrt(2)*np.prod(GF(a, a)[:2]) - 2, 0.05, 0.3)
    acrit_O = brentq(lambda a: GF(a, a)[2]**2 + np.prod(GF(a, a)[:2])**2 - 1, 0.05, 3.0)
    print(f"a_crit (Sorella settings) = {acrit_S:.5f};   a_crit (Horodecki-optimal, same vectors) = {acrit_O:.5f}")
    # small-a coefficients
    a = 1e-3; G, F, Tzz = GF(a, a)
    print("deficit/a^2  Sorella:", (2*sqrt(2) - 2*sqrt(2)*G*F)/a**2, "  optimal:", (2*sqrt(2) - 2*sqrt(Tzz**2+(G*F)**2))/a**2,
          "  (pi^2*(2mu2+5/4)*2sqrt2 =", pi**2*(2*mu2+1.25)*2*sqrt(2), ")")
    offRe, T, Bg = grid_check(0.1, 0.1); print("T matrix at a=d=0.1 (grid):\n", np.round(T, 6))
