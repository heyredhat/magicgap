import numpy as np
import scipy as sc
from itertools import product, permutations

from .utils import bv, rand_ket, kron

def d_j(d):
    return (d-1)/2

def j_d(j):
    return int(2*j+1)

def spin_bv(j, m):
    return np.eye(int(2*j+1))[int(j-m)]

def m_values(j):
    return [m for m in np.arange(j, -j-1, -1)]

def spin_matrices(j):
    Jp = np.diag([np.sqrt((j-m)*(j+m+1)) for m in np.arange(j-1, -j-1, -1)], k=1)
    return (Jp + Jp.conj().T)/2, (Jp - Jp.conj().T)/(2j), np.diag(m_values(j))

####################################################################################################

# Borrowed from qutip.utilities

def _factorial_prod(N, arr):
    arr[:int(N)] += 1

def _factorial_div(N, arr):
    arr[:int(N)] -= 1

def _to_long(arr):
    prod = 1
    for i, v in enumerate(arr):
        prod *= (i+1)**int(v)
    return prod

def clebsch_gordan(j1, j2, j3, m1, m2, m3):
    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    c_factor = np.zeros((int(j1 + j2 + j3 + 1)), np.int32)
    _factorial_prod(j3 + j1 - j2, c_factor)
    _factorial_prod(j3 - j1 + j2, c_factor)
    _factorial_prod(j1 + j2 - j3, c_factor)
    _factorial_prod(j3 + m3, c_factor)
    _factorial_prod(j3 - m3, c_factor)
    _factorial_div(j1 + j2 + j3 + 1, c_factor)
    _factorial_div(j1 - m1, c_factor)
    _factorial_div(j1 + m1, c_factor)
    _factorial_div(j2 - m2, c_factor)
    _factorial_div(j2 + m2, c_factor)
    C = np.sqrt((2.0 * j3 + 1.0)*_to_long(c_factor))

    s_factors = np.zeros(((vmax + 1 - vmin), (int(j1 + j2 + j3))), np.int32)
    sign = (-1) ** (vmin + j2 + m2)
    for i,v in enumerate(range(vmin, vmax + 1)):
        factor = s_factors[i,:]
        _factorial_prod(j2 + j3 + m1 - v, factor)
        _factorial_prod(j1 - m1 + v, factor)
        _factorial_div(j3 - j1 + j2 - v, factor)
        _factorial_div(j3 + m3 - v, factor)
        _factorial_div(v + j1 - j2 - m3, factor)
        _factorial_div(v, factor)
    common_denominator = -np.min(s_factors, axis=0)
    numerators = s_factors + common_denominator
    S = sum([(-1)**i * _to_long(vec) for i,vec in enumerate(numerators)]) * \
        sign / _to_long(common_denominator)
    return C * S

####################################################################################################

def J_values(j1, j2):
    return np.arange(abs(j1-j2), j1+j2+1, 1)

def recoupled_spin_basis_(basis1, basis2):
    j1, j2 = d_j(basis1.shape[0]), d_j(basis2.shape[0])
    return dict([(J, [np.array([sum([clebsch_gordan(j1, j2, J, m1, m2, M)*\
                                    np.kron(basis1[int(j1-m1)], basis2[int(j2-m2)])\
                                        for m2 in m_values(j2) for m1 in m_values(j1)])
                                            for M in m_values(J)])]) for J in J_values(j1, j2)])

def recoupled_spin_basis(*j_values):
    sectors = recoupled_spin_basis_(np.eye(j_d(j_values[0])), np.eye(j_d(j_values[1])))
    for j in j_values[2:]:
        new_sectors = {}
        for J, bases in sectors.items():
            for basis in bases:
                for J_, bases in recoupled_spin_basis_(basis, np.eye(j_d(j))).items():
                    if J_ not in new_sectors:
                        new_sectors[J_] = []
                    new_sectors[J_].extend(bases)
        sectors = new_sectors
    return sectors


def recoupled_spin_sectors(*j_values):
    return [(float(k), len(v)) for k,v in recoupled_spin_basis(*j_values).items()]

####################################################################################################

def plane_sphere(z):
    if z == np.inf:
        return np.array([0,0,-1])
    else:
        return np.array([2*z.real, 2*z.imag, 1-z.real**2-z.imag**2])/(1+z.real**2+z.imag**2)
    
def sphere_plane(xyz):
    x, y, z = xyz
    if z == -1:
        return np.inf
    return (x+1j*y)/(1+z)

def majorana_polynomial(ket):
    j = (len(ket)-1)/2
    return np.polynomial.Polynomial(\
                    [(-1)**(j-m)*\
                     np.sqrt(sc.special.binom(2*j, j-m))*\
                     spin_bv(j, m).conj() @ ket\
                         for m in np.arange(-j, j+1)])

def majorana_roots(ket):
    n = len(ket) - 1
    r = majorana_polynomial(ket).roots()
    return np.concatenate([r, np.repeat(np.inf, n-len(r))])

def majorana_stars(ket):
    return np.array([plane_sphere(r) for r in majorana_roots(ket)])

def stars_qubits(stars):
    roots = [sphere_plane(s) for s in stars]
    qubits = [np.array([0,1], dtype=np.complex128) if r == np.inf else np.array([1, r]) for r in roots]
    return np.array([qubit/np.linalg.norm(qubit) for qubit in qubits])

def sym_qubit_basis(j):
    sym_qubit_states = {}
    for idx in product([0,1], repeat=int(2*j)):
        total = sum(idx)
        if total not in sym_qubit_states:
            sym_qubit_states[total] = []
        sym_qubit_states[total].append(bv("".join([str(_) for _ in idx]), d=2))
    return np.array([sum(sym_qubit_states[i])/np.sqrt(len(sym_qubit_states[i])) for i in range(int(2*j+1))])

def perm_sym_product(kets):
    psi = sum([kron(*kets[perm, :]) for perm in permutations(list(range(len(kets))))])
    return psi/np.linalg.norm(psi)

def test_sym_qubits(j):
    d = j_d(j)
    B = sym_qubit_basis(j)
    ket = rand_ket(d)
    sym_qubits = B.conj().T @ ket
    sep_qubits = stars_qubits(majorana_stars(ket))
    sym_qubits2 = perm_sym_product(sep_qubits)
    return np.allclose(sym_qubits/sym_qubits[0], sym_qubits2/sym_qubits2[0])