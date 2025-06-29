import numpy as np
import scipy as sc

import jax
import jax.numpy as jp
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from .stabilizer_entropy import flatten_if_needed
from .utils import rand_kets, rand_ket

def min_magic_over_complement(D, B, ket_s, R=5):
    D = flatten_if_needed(D)
    d_s, d_b = B.shape
    B_c = np.linalg.qr(B.T, mode='complete')[0].T[d_s:]
    d_c = B_c.shape[0]
    ket_can = ket_s @ B

    @jax.jit
    def obj(V):
        V = V.reshape(2, d_c)
        ket_c = V[0] + 1j*V[1]
        ket_c = (ket_c/jp.linalg.norm(ket_c)) @ B_c
        ket_b = ket_can + ket_c
        ket_b = ket_b/jp.linalg.norm(ket_b)
        se = 1 - (1/d_b)*jp.sum(abs(jp.einsum("j, ijk, k", ket_b.conj(), D, ket_b))**4)
        return se**2

    jac = jax.jit(jax.jacrev(obj))
    results = [sc.optimize.minimize(obj, np.random.randn(2*d_c), jac=jac,\
                                    tol=1e-16, options={"disp": False, "maxiter": 10000}) for _ in range(R)]
    result = results[np.argmin([res.fun for res in results])]
    V = result.x.reshape(2, d_c)
    ket_c = V[0] + 1j*V[1]
    ket_c = (ket_c/jp.linalg.norm(ket_c)) @ B_c
    ket_b = ket_can + ket_c
    ket_b = ket_b/jp.linalg.norm(ket_b)
    return np.array(ket_b)

def avg_magic_subspace_mc_with_optimal_support(D, B, M=500, R=5):
    d_s, d_b = B.shape
    K = np.array([min_magic_over_complement(D, B, rand_ket(d_s), R=R) for _ in range(M)])
    samples = 1 - np.sum(abs(np.einsum('id,jde,ie->ij', K.conj(), flatten_if_needed(D), K))**4, axis=1)/d_b
    return np.mean(samples), np.std(samples)

##########################################################################################

def min_ase_support_vector(D, B, M=750, R=10, scale_sampling=True):
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
        ket_c = V[0] + 1j*V[1]
        ket_c = (ket_c/jp.linalg.norm(ket_c)) @ B_c
        L = (K @ B + ket_c)/jp.sqrt((1 + ket_c.conj() @ ket_c))
        ase = jp.mean(1-jp.sum(abs(jp.einsum('id,jde,ie->ij', L.conj(), D, L))**4, axis=1)/d_b)
        return ase**2
    jac = jax.jit(jax.jacrev(obj))
    results = [sc.optimize.minimize(obj, np.random.randn(2*d_c), jac=jac,\
                                    tol=1e-16, options={"disp": False, "maxiter": 10000}) for _ in range(R)]
    result = results[np.argmin([res.fun for res in results])]
    V = result.x.reshape(2, d_c)
    ket_c = V[0] + 1j*V[1]
    ket_c = (ket_c/jp.linalg.norm(ket_c)) @ B_c
    return np.array(ket_c)

def avg_magic_subspace_mc_with_support_vector(D, B, ket_c, M=750):
    d_s, d_b = B.shape
    K = rand_kets(d_s, M)
    L = (K @ B + ket_c)/np.sqrt((1 + ket_c.conj() @ ket_c))
    samples = 1-jp.sum(abs(jp.einsum('id,jde,ie->ij', L.conj(), flatten_if_needed(D), L))**4, axis=1)/d_b
    return np.mean(samples), np.std(samples)

def extremize_subspace_magic_mc_with_support_vector(D, d_s, dir, R=1, M=750, scale_sampling=True):
    D = flatten_if_needed(D)
    d_b = D[0].shape[0]
    s = 1 if dir == "min" else -1
    if scale_sampling:
        M = M*d_s
    K = rand_kets(d_s, M)
    d_c = d_b - d_s
    
    @jax.jit
    def obj(V):
        V1, V2 = V[:2*d_b*d_s], V[2*d_b*d_s:]
        V1 = V1.reshape(2, d_s, d_b)
        B = V1[0] + 1j*V1[1]
        U, Sig, V = jp.linalg.svd(B) # not differentiable
        B, B_c = V[:d_s], V[d_s:]
        V2 = V2.reshape(2, d_c)
        ket_c = V2[0] + 1j*V2[1]
        ket_c = (ket_c/jp.linalg.norm(ket_c)) @ B_c
        L = (K @ B + ket_c)/jp.sqrt((1 + ket_c.conj() @ ket_c))
        ase = jp.mean(1-jp.sum(abs(jp.einsum('id,jde,ie->ij', L.conj(), D, L))**4, axis=1)/d_b)
        return s*ase**2

    results = [sc.optimize.minimize(obj, np.random.randn(2*d_b*d_s+2*d_c),\
                                         jac=jax.jit(jax.jacrev(obj)),\
                                         tol=1e-26, options={"disp": False, "maxiter": 10000})
                    for r in range(R)]
    if dir == "min":
        result = results[np.argmin([r.fun for r in results])]
    elif dir == "max":
        result = results[np.argmax([r.fun for r in results])]
    V = result.x
    V1, V2 = V[:2*d_b*d_s], V[2*d_b*d_s:]
    V1 = V1.reshape(2, d_s, d_b)
    B = V1[0] + 1j*V1[1]
    U = jp.linalg.qr(B.T, mode="complete")[0].T
    B, B_c = U[:d_s], jp.squeeze(U[d_s:])
    ket_c = V2[0] + 1j*V2[1]
    ket_c = (ket_c/jp.linalg.norm(ket_c)) @ B_c
    return np.array(B), np.array(ket_c)