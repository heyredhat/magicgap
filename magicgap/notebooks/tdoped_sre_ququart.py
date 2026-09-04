#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Linear stabilizer Renyi entropy (SRE) of the [[4,2,2]] logical space over
Clifford + T-doped states.

The logical space is 4-dimensional and is doped with T gates as TWO LOGICAL
QUBITS (random 2-qubit Cliffords interleaved with T gates).  For every doping
level k (number of injected T gates) we read off THREE linear SREs of the same
doped ensemble:

    M_2q = 1 - d  tr(Q_2qubit  A_k)   two logical qubits
    M_4q = 1 - dB tr(W         A_k)   four physical qubits  ([[4,2,2]] embedding)
    M_qq = 1 - d  tr(Q_ququart A_k)   the same states seen as a single ququart

and the two gaps
    Gap1 = M_4q - M_2q   (== 0, the code-embedding gap of Sec. VII E)
    Gap2 = M_2q - M_qq   (-> -2/35, the ququart magic gap).

Method
------
The k-fold T-doped Clifford ensemble is  C_t = C_k T C_{k-1} T ... T C_0 , with
each C_i uniform over the 2-qubit Clifford group.  The 4-copy operator standing
in for  E_{C_t}[(C_t |0><0| C_t^dag)^{⊗4}]  is

    A_k = Phi ( That Phi )^k ( |0><0|^{⊗4} ) ,      That(A) = T^{⊗4} A T^{†⊗4}

where Phi is the 2-qubit Clifford 4-fold twirl = orthogonal (Hilbert-Schmidt)
projector onto the commutant K.  The 2-qubit Clifford group is a unitary
3-design, so K = span{T_sigma, Q_2 T_sigma} (dimension 29) and Phi is evaluated
as the HS projection onto that span -- no group enumeration needed.

Key identity: W = V†⊗4 Q_B V⊗4 = (dS/dB) Q_2qubit = (1/4) Q_2qubit exactly, hence
M_4q = M_2q identically (Gap1 = 0).

Closed forms (rate 3/10, single mode because of the 3-design):
    M_2q = M_4q = (3/7)(1 - (3/10)^k)
    M_qq        = 17/35 - (13/70)(3/10)^k
    Gap2        = M_2q - M_qq = -2/35 - (17/70)(3/10)^k   ->  -2/35

Requires: numpy, matplotlib.
"""

import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- #
#  Generic helpers
# ----------------------------------------------------------------------------- #
def kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def kr4(C):
    """C^{⊗4}."""
    C2 = np.kron(C, C)
    return np.kron(C2, C2)

def perm_op(sigma, d):
    """Permutation operator T_sigma on 4 copies of C^d (sigma permutes factors)."""
    idx = list(itertools.product(range(d), repeat=4))
    pos = {t: n for n, t in enumerate(idx)}
    P = np.zeros((d**4, d**4), complex)
    sinv = [0]*4
    for j in range(4):
        sinv[sigma[j]] = j
    for t in idx:
        nt = tuple(t[sinv[j]] for j in range(4))
        P[pos[nt], pos[t]] = 1.0
    return P


# ----------------------------------------------------------------------------- #
#  Stabilizer-entropy operators Q  (Q = d^{-2} sum_a (D_a ⊗ D_a^†)^{⊗2})
# ----------------------------------------------------------------------------- #
def ququart_Q():
    """Single-ququart Z_4 Weyl-Heisenberg SRE operator Q_ququart (256x256)."""
    d = 4
    w = np.exp(2j*np.pi/d); tau = -np.exp(1j*np.pi/d)
    Z = np.diag([w**k for k in range(d)])
    X = np.zeros((d, d), complex)
    for k in range(d):
        X[(k+1) % d, k] = 1.0
    mp = np.linalg.matrix_power
    WHk = [(a1, a2) for a1 in range(d) for a2 in range(d)]
    Q = np.zeros((d**4, d**4), complex)
    for a1, a2 in WHk:
        Da = (tau**(a1*a2)) * mp(X, a1) @ mp(Z, a2)
        blk = np.kron(Da, Da.conj().T)
        Q += np.kron(blk, blk)
    return Q/d**2

def twoqubit_Q():
    """Two-qubit Pauli SRE operator Q_2qubit (256x256) and the 16 Paulis."""
    d = 4
    tau = -1j
    Z = np.diag([1, -1]).astype(complex); X = np.array([[0, 1], [1, 0]], complex)
    mp = np.linalg.matrix_power
    D1 = {(a, b): (tau**(a*b)) * mp(X, a) @ mp(Z, b) for a in range(2) for b in range(2)}
    P2 = [np.kron(D1[(a, b)], D1[(c, e)])
          for a in range(2) for b in range(2) for c in range(2) for e in range(2)]
    Q = np.zeros((d**4, d**4), complex)
    for Da in P2:
        blk = np.kron(Da, Da.conj().T)
        Q += np.kron(blk, blk)
    return Q/d**2


# ----------------------------------------------------------------------------- #
#  2-qubit Clifford twirl = HS projector onto span{T_sigma, Q_2 T_sigma}
#  (this span equals the 4th-moment commutant because the 2-qubit Clifford
#   group is a unitary 3-design; verified: dim = 29)
# ----------------------------------------------------------------------------- #
def twoqubit_twirl(Q2):
    d = 4
    S4 = list(itertools.permutations(range(4)))
    Ts = [perm_op(s, d) for s in S4]
    cand = Ts + [Q2 @ t for t in Ts]
    M = np.array([c.flatten() for c in cand]).T
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    r = int((s > 1e-8*s[0]).sum())
    E = [U[:, i].reshape(d**4, d**4) for i in range(r)]     # orthonormal basis of K
    def Phi(A):
        S = np.zeros((d**4, d**4), complex)
        for e in E:
            S += e * np.trace(e.conj().T @ A)
        return S
    return Phi, r


# ----------------------------------------------------------------------------- #
#  [[4,2,2]] code isometry and physical (big-space) SRE operator W = V†⊗4 Q_B V⊗4
# ----------------------------------------------------------------------------- #
def code_W():
    def ket(bits):
        v = np.zeros(16); v[int(bits, 2)] = 1.0; return v
    code = {0: (ket('0000')+ket('1111'))/np.sqrt(2),
            1: (ket('0011')+ket('1100'))/np.sqrt(2),
            2: (ket('0101')+ket('1010'))/np.sqrt(2),
            3: (ket('0110')+ket('1001'))/np.sqrt(2)}
    V = np.array([code[j] for j in range(4)]).T.astype(complex)   # 16 x 4
    tau = -1j
    Z = np.diag([1, -1]).astype(complex); X = np.array([[0, 1], [1, 0]], complex)
    mp = np.linalg.matrix_power
    D1 = [(tau**(a*b))*mp(X, a)@mp(Z, b) for a in range(2) for b in range(2)]
    W = np.zeros((256, 256), complex)
    for combo in itertools.product(range(4), repeat=4):
        Da = D1[combo[0]]
        for c in combo[1:]:
            Da = np.kron(Da, D1[c])
        va  = V.conj().T @ Da @ V
        vad = V.conj().T @ Da.conj().T @ V
        W += np.kron(np.kron(va, vad), np.kron(va, vad))
    return W/16**2, V


# ----------------------------------------------------------------------------- #
#  T-doped iteration:  A_k = Phi (That Phi)^k (|0><0|^{⊗4})
# ----------------------------------------------------------------------------- #
def doped_curves(Phi, Q_2qubit, Q_ququart, W, T, kmax):
    d = 4
    ket0 = np.zeros(d); ket0[0] = 1.0
    r1 = np.outer(ket0, ket0)
    rho0 = kron(r1, r1, r1, r1)
    T4 = kr4(T); T4d = T4.conj().T
    A = Phi(rho0)
    M2q, M4q, Mqq = [], [], []
    for _ in range(kmax+1):
        M2q.append((1 - d *np.trace(Q_2qubit  @ A)).real)   # two logical qubits
        M4q.append((1 - 16*np.trace(W         @ A)).real)   # four physical qubits
        Mqq.append((1 - d *np.trace(Q_ququart @ A)).real)   # ququart view
        A = Phi(T4 @ A @ T4d)
    return np.array(M2q), np.array(M4q), np.array(Mqq)


# ----------------------------------------------------------------------------- #
#  Main
# ----------------------------------------------------------------------------- #
def main(kmax=30):
    # ququart T gate (a genuine single-ququart, non-Clifford, diagonal gate),
    # here injected as a (diagonal, entangling) 2-qubit gate on the logical qubits
    T = np.diag([1, np.exp(1j*np.pi/4), 1, np.exp(3j*np.pi/4)])

    Q2 = twoqubit_Q()
    Q4 = ququart_Q()
    W, V = code_W()
    assert np.abs(W - 0.25*Q2).max() < 1e-12, "expected W = (1/4) Q_2qubit"
    Phi, r = twoqubit_twirl(Q2)
    print(f"two-qubit 4th-moment commutant dim = {r}  (twirl = HS projection)")

    M2q, M4q, Mqq = doped_curves(Phi, Q2, Q4, W, T, kmax)
    k = np.arange(kmax+1)
    print(f"Haar limits: M_2q=M_4q->3/7={3/7:.4f}, M_qq->17/35={17/35:.4f}, "
          f"gap M_2q-M_qq->-2/35={-2/35:.4f}")
    print(f"max |M_4q - M_2q| = {np.abs(M4q-M2q).max():.1e}   (4 physical == 2 logical)")

    # ---- figure (manuscript template: markers+lines, dashed zero, math labels) ---
    plt.rcParams.update({'font.size': 11, 'axes.linewidth': 0.8,
                         'mathtext.fontset': 'cm',
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.top': True, 'ytick.right': True})
    C0, C1, C3 = '#1f77b4', '#ff7f0e', '#d62728'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # panel (a): the two "views" of the same 2-qubit-doped states
    ax1.plot(k, M2q, 's-', color=C3, ms=3.9, lw=1.3, label=r'$M_{2\mathrm{q}}=M_{4\mathrm{q}}$')
    ax1.plot(k, Mqq, 'o-', color=C0, ms=4,   lw=1.3, label=r'$M_{\mathrm{qq}}$')
    ax1.axhline(3/7,   ls=':', color=C3, lw=0.9)
    ax1.axhline(17/35, ls=':', color=C0, lw=0.9)
    ax1.text(kmax, 3/7-0.016, r'$\frac{3}{7}$',   color=C3, ha='right', va='top',    fontsize=10)
    ax1.text(kmax, 0.522,     r'$\frac{17}{35}$', color=C0, ha='right', va='center', fontsize=10)
    ax1.set_xlabel(r'number of $T$ gates $\,t$')
    ax1.set_ylabel(r'$M_2$')
    ax1.set_xlim(-0.6, kmax+0.6); ax1.set_ylim(-0.02, 0.55)
    ax1.legend(frameon=False, fontsize=9.5, loc='center right')
    ax1.text(-0.12, 1.02, '(a)', transform=ax1.transAxes, va='bottom', ha='left', fontsize=12)

    # panel (b): the two gaps
    ax2.axhline(0, ls='--', color='k', lw=1.0)
    ax2.plot(k, M2q-Mqq, 'o-', color=C0, ms=4, lw=1.3, label=r'$M_{4\mathrm{q}}-M_{\mathrm{qq}}$')
    ax2.plot(k, M4q-M2q, '^-', color=C1, ms=4, lw=1.3, label=r'$M_{4\mathrm{q}}-M_{2\mathrm{q}}\equiv 0$')
    ax2.axhline(-2/35, ls=':', color=C0, lw=0.9)
    ax2.text(kmax, -2/35+0.004, r'$-\frac{2}{35}$', color=C0, ha='right', va='bottom', fontsize=10)
    ax2.set_xlabel(r'number of $T$ gates $\,t$')
    ax2.set_ylabel(r'$\Delta M_{\mathcal{C}_t}$')
    ax2.set_xlim(-0.6, kmax+0.6); ax2.set_ylim(-0.32, 0.04)
    ax2.legend(frameon=False, fontsize=9.5, loc='center right')
    ax2.text(-0.12, 1.02, '(b)', transform=ax2.transAxes, va='bottom', ha='left', fontsize=12)

    fig.tight_layout()
    fig.savefig('tdoped_gap.png', dpi=300, bbox_inches='tight')
    print("saved tdoped_gap.png")


if __name__ == '__main__':
    matplotlib.use('Agg')
    main(kmax=30)
