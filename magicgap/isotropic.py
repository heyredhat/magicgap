import numpy as np

import jax
import jax.numpy as jp
from jax import lax

from functools import partial
from itertools import combinations


from .stabilizer_entropy import symplectic_product, sum_symp_indices
def mod_decompose(a, d):
    a = np.asarray(a)
    x = a % d
    y = (a - x)/d
    return x,y

@partial(jax.jit, static_argnums=(1,))
def chi_sq(chi, d):
    d_bar = d if d % 2 != 0 else 2*d
    tau = -jp.exp(jp.pi * 1j / d) 
    ndims = len(chi.shape)

    grid = jp.stack(jp.meshgrid(*[jp.arange(d)] * ndims, indexing='ij'), axis=-1) 
    all_indices = grid.reshape(-1, ndims) 

    def symplectic_batch(a, b):
        a = a.reshape(-1, 2)
        b = b.reshape(-1, 2)
        return jp.sum(a[:, 1] * b[:, 0] - a[:, 0] * b[:, 1])

    def compute_single(a):
        def compute_contrib(b):
            a_minus_b = (a - b) % d_bar
            sp = symplectic_batch(b, a)
            phase = tau ** sp
            return chi[tuple(b)] * chi[tuple(a_minus_b)] * phase
        return jp.sum(jax.vmap(compute_contrib)(all_indices))
    result = jax.vmap(compute_single)(all_indices) 
    return result.reshape(*[d]*ndims)

def chi_from_isotropic_set(d, n, V, expanded=False):
    S = len(V)
    chi = np.zeros([d, d]*n)
    chi[tuple(np.asarray(V).T)] = 1/S
    if expanded:
        return expand_chi(chi)
    else:
        return chi

def find_isotropic_sets(d, n, d_s, m=1):
    d_b = d**n
    S = int(d_b/d_s)
    indices = list(np.ndindex(*[d, d]*n))
    sets = []
    for V in combinations(indices, S):
        chi = chi_from_isotropic_set(d, n, V, expanded=False)
        chi_expanded = chi_from_isotropic_set(d, n, V, expanded=True)
        if np.allclose(chi, chi_sq(chi_expanded, d)):
            sets.append(V)
            if len(sets) == m:
                return sets

def isotropy_projector(D, d, n, V):
    tau = -jp.exp(1j * jp.pi / d)
    chi = np.zeros([d, d]*n)
    chi[tuple(np.asarray(V).T)] = 1/len(V)
    Pi = sum([D[a]*chi[a] for a in np.ndindex(*[d, d]*n)])
    return Pi

def test_isotropy(d, V):
	one = np.allclose(np.array([[symplectic_product(a,b) % d for b in V] for a in V]), 0)
	V_ = [np.asarray(sum_symp_indices([a,b],d)).tolist() for b in V for a in V];
	V_ = {tuple(x) for x in V_}
	two = {tuple(x) for x in V} == V_
	return one and two

def term5(d, n, V):
    A = [a for a in np.ndindex(*[d,d]*n) if\
            tuple((2*np.asarray(a)) % d) in V and
            np.all([(symplectic_product(np.asarray(a), v) % d) == 0 for v in V])]
    return len(A)/len(V)**2