import math
import numpy as np
import scipy as sc
from functools import reduce

def kron(*A):
    return reduce(np.kron, A)

def tensor_power(A, t):
    return kron(*[A]*t)

def upgrade(O, j, dims):
    return kron(*[O if i == j else np.eye(d) for i, d in enumerate(dims)])

def bv(s, d=2):
    return kron(*[np.eye(d)[int(_)] for _ in s])

####################################################################################################

def rand_ket(d):
    ket = np.random.randn(d) + 1j*np.random.randn(d)
    return ket/np.linalg.norm(ket)

def rand_kets(d, n):
    R = np.random.randn(d, n) + 1j*np.random.randn(d, n)
    R = R/np.linalg.norm(R, axis=0)
    return R.T

def rand_dm(d, r=1):
    A = np.random.randn(d,r) + 1j*np.random.randn(d,r)
    A = A @ A.conj().T
    return A/A.trace()

def rand_herm(d):
    A = np.random.randn(d,d) + 1j*np.random.randn(d,d)
    return A + A.conj().T

def rand_unitary(d):
    return sc.stats.unitary_group.rvs(d)

def rand_basis(n, d):
    return rand_unitary(d)[:n, :]

####################################################################################################

def permutation_operators(d, t):
    return np.array([sum([np.outer(kron(*[np.eye(d)[i] for i in np.array(prod)[np.array(perm)]]),\
                          kron(*[np.eye(d)[i] for i in prod]))\
                            for prod in product(np.arange(d), repeat=t)])
                                    for perm in permutations(np.arange(t))])

def symmetric_projector(d, t):
    return sum(permutation_operators(d, t))/math.factorial(t)

def perm_sym_product(kets):
    psi = sum([kron(*kets[perm, :]) for perm in permutations(list(range(len(kets))))])
    return psi/np.linalg.norm(psi)

def haar_moment(d, t):
    Pi = symmetric_projector(d,t)
    return Pi/Pi.trace()
