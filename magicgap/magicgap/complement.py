import numpy as np
import scipy as sc

import jax
import jax.numpy as jp
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from .stabilizer_entropy import flatten_if_needed
from .utils import rand_kets, rand_ket

def min_magic_over_complement(D, B, ket_s, R=100, return_ket_c=False):
    D = flatten_if_needed(D)
    d_s, d_b = B.shape
    B_c = np.linalg.qr(B.T, mode='complete')[0].T[d_s:]
    d_c = B_c.shape[0]
    ket_can = B.T @ ket_s

    @jax.jit
    def obj(V):
        V = V.reshape(2, d_c)
        ket_c = (V[0] + 1j*V[1]) @ B_c
        ket_b = ket_can + ket_c
        ket_b = ket_b/jp.linalg.norm(ket_b)
        se = 1 - (1/d_b)*jp.sum(abs(jp.einsum("j, ijk, k", ket_b.conj(), D, ket_b))**4)
        return se**2
    jac = jax.jit(jax.jacrev(obj))
    results = [sc.optimize.minimize(obj, np.random.randn(2*d_c), jac=jac,\
                                    tol=1e-16, options={"disp": False, "maxiter": 10000}) for _ in range(R)]
    result = results[np.argmin([res.fun for res in results])]
    V = result.x.reshape(2, d_c)
    ket_c = (V[0] + 1j*V[1]) @ B_c
    ket_b = ket_can + ket_c
    if return_ket_c:
        return ket_c
    return np.array(ket_b/jp.linalg.norm(ket_b))

def avg_magic_subspace_mc_with_optimal_support(D, B, M=500, R=5):
    d_s, d_b = B.shape
    K = np.array([min_magic_over_complement(D, B, rand_ket(d_s), R=R) for _ in range(M)])
    samples = 1 - np.sum(abs(np.einsum('id,jde,ie->ij', K.conj(), flatten_if_needed(D), K))**4, axis=1)/d_b
    return np.mean(samples), np.std(samples)

##########################################################################################

def min_avg_magic_subspace_support(D, B, M=750, R=10, scale_sampling=True):
    D = flatten_if_needed(D)
    d_s, d_b = B.shape
    B_c = np.linalg.qr(B.T, mode='complete')[0].T[d_s:]
    d_c = B_c.shape[0]
    if scale_sampling:
        M = M*d_s
        K = rand_kets(d_s, M)

    @jax.jit
    def obj(V):
        V = V.reshape(2, d_c)
        ket_c = (V[0] + 1j*V[1]) @ B_c
        ket_c = ket_c
        L = (K @ B + ket_c)/jp.sqrt((1 + ket_c.conj() @ ket_c))
        ase = jp.mean(1-jp.sum(abs(jp.einsum('id,jde,ie->ij', L.conj(), D, L))**4, axis=1)/d_b)
        return ase**2
    jac = jax.jit(jax.jacrev(obj))
    results = [sc.optimize.minimize(obj, np.random.randn(2*d_c), jac=jac,\
                                    tol=1e-16, options={"disp": False, "maxiter": 10000}) for _ in range(R)]
    result = results[np.argmin([res.fun for res in results])]
    V = result.x.reshape(2, d_c)
    ket_c = (V[0] + 1j*V[1]) @ B_c
    return ket_c

def avg_magic_subspace_mc_with_support(D, B, ket_c, M=750):
    d_s, d_b = B.shape
    K = rand_kets(d_s, M)
    L = (K @ B + ket_c)/np.sqrt((1 + ket_c.conj() @ ket_c))
    samples = 1-jp.sum(abs(jp.einsum('id,jde,ie->ij', L.conj(), flatten_if_needed(D), L))**4, axis=1)/d_b
    return np.mean(samples), np.std(samples)

