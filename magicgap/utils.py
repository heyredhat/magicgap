import json
import math
import numpy as np
import scipy as sc
from functools import reduce
from itertools import permutations, product

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

def rand_unitary(d, n=1):
    return sc.stats.unitary_group.rvs(d, size=n)

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

def haar_moment(d, t):
    Pi = symmetric_projector(d,t)
    return Pi/Pi.trace()

####################################################################################################

def sample_mean(f, M=10):
    samples = [f() for r in range(M)]
    return (np.mean(samples), np.std(samples))

####################################################################################################

def save_data(filename, data):
    with open(filename+".json", "w") as f:
        json.dump(data, f)

def keys_to_int(x):
  return {int(k) if k.isdigit() else k: v for k, v in x.items()}

def load_data(filename):
    with open(filename+".json", "r") as f:
        data = json.load(f, object_hook=keys_to_int)
    return data

####################################################################################################

def is_pow2(d):
    n = int(np.log2(d))
    return 2**n == d

####################################################################################################

def paulis():
    return np.array([[0,1],[1,0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1,0],[0,-1]])

####################################################################################################

def factors(n):
    if n == 0:
        return []
    n = abs(n)
    factors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(factors)