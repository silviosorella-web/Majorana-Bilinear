"""Compact modular seeds (Secs. V-VI of the draft): Table I, critical widths a_c for the
planar and optimal settings, the example d = 2a, and the small-a expansion coefficients
of Eqs. (Bplanar-exp) and (Bopt-exp).  Uses the closed forms of sorella08_check.py."""
import numpy as np
from math import pi, sqrt
from scipy.optimize import brentq
from sorella08_check import GF, I, mu2, N0

print(f"N0 = {N0:.6f}   mu2 = {mu2:.7f}")
print(" a      G(a)      F(a,a)    T_zz       closed-form T_zz   B_planar  B_opt")
for a in [0.001, 0.01, 0.04, 0.10, 0.15, 0.20, 0.30, 0.40]:
    G, F, Tzz = GF(a, a)
    Tzz_closed = -1 + 0.5 * (I(a)**2 - 1) * G**2          # Eq. (T), second form
    print(f"{a:5.3f}  {G:.6f}  {F:.6f}  {Tzz:+.6f}  {Tzz_closed:+.6f}          "
          f"{2*sqrt(2)*G*F:.5f}   {2*sqrt(Tzz**2+(G*F)**2):.5f}")
ac_planar = brentq(lambda a: 2*sqrt(2)*np.prod(GF(a, a)[:2]) - 2, 0.05, 0.3)
ac_opt = brentq(lambda a: GF(a, a)[2]**2 + np.prod(GF(a, a)[:2])**2 - 1, 0.05, 3.0)
print(f"a_c (planar settings) = {ac_planar:.5f}    a_c (optimal settings) = {ac_opt:.5f}")
G, F, Tzz = GF(0.04, 0.08)
print(f"a = 0.04, d = 2a: planar {2*sqrt(2)*G*F:.4f}, optimal {2*sqrt(Tzz**2+(G*F)**2):.4f}")
a = 1e-3
for rho in (1, 2):
    G, F, Tzz = GF(a, rho * a)
    c_pl = (1 - G * F) / (pi * a)**2
    c_op = (1 - sqrt((Tzz**2 + (G * F)**2) / 2)) / (pi * a)**2
    print(f"rho = {rho}: planar coefficient {c_pl:.6f} (formula {2*mu2+1/8+(2*rho+1)**2/8:.6f}),"
          f" optimal {c_op:.6f} (formula {2*mu2+(1+(2*rho+1)**2)/16:.6f})")
print(f"deficit at contact: planar {2*sqrt(2)*pi**2*(2*mu2+5/4):.2f} a^2, optimal {2*sqrt(2)*pi**2*(2*mu2+5/8):.2f} a^2")
