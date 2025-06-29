import numpy as np
import scipy as sc

import jax
import jax.numpy as jp
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from .utils import rand_unitary

def clifford_entropy(D, U, alpha=2):
    D = flatten_if_needed(D)
    d = D[0].shape[0]
    return (1 - sum([abs((a.conj().T @ U @ b @ U.conj().T).trace())**(2*alpha) for b in D for a in D])/d**(2*(alpha+1)))/(alpha-1)

def clifford_entropy_vec(D, U_batch):
    D = flatten_if_needed(D)
    d = D.shape[-1]
    return 1 - np.sum(abs(np.einsum("axy, ixz, bzq, iyq->iab", D.conj(), U_batch, D, U_batch.conj()))**4, axis=(1,2))/d**6

def avg_clifford_entropy_mc(D, M=750):
    D = flatten_if_needed(D)
    d = D.shape[-1]
    samples = clifford_entropy_vec(D, rand_unitary(d, M))
    return np.mean(samples), np.std(samples)

def avg_clifford_entropy_subspace_mc(D, B, M=750):
    D = flatten_if_needed(D)
    d_s, d_b = B.shape
    Pi_C = np.eye(d_b) - B.T @ B.conj()
    samples = clifford_entropy_vec(D, B.T @ rand_unitary(d_s, M) @ B.conj() + Pi_C)
    return np.mean(samples), np.std(samples)

def lift_unitary(U, B):
    d_s, d_b = B.shape
    Pi_C = np.eye(d_b) - B.T @ B.conj()
    return B.T @ U @ B.conj() + Pi_C 

def wh_entropy(D_b, D_s, B):
    D_b = flatten_if_needed(D_b)
    D_s = flatten_if_need(D_s)
    d_s, d_b = B.shape
    chi = np.array([[(a.conj().T @ lift_unitary(b, B)).trace() for b in D_s] for a in D_b])
    return 1 - np.sum(abs(chi)**4)/d_b**6

def extremize_wh_entropy(D_b, D_s, dir, R=1):
    D_b = flatten_if_needed(D_b)
    D_s = flatten_if_need(D_s)

    s = 1 if dir == "min" else -1
    d_b = D_b.shape[-1]
    d_s = D_s.shape[-1]

    @jax.jit
    def obj(V):
        V = V.reshape(2, d_s, d_b)
        B = V[0] + 1j*V[1]
        B = jp.linalg.qr(B.T)[0].T
        D_s_lift = B.T @ D_s @ B.conj() + jp.eye(d_b) - B.T @ B.conj()
        whe = 1 - jp.sum(abs(jp.einsum("aij, bij", D_b.conj(), D_s_lift))**4)/d_b**6
        return s*whe**2

    jac = jax.jit(jax.jacrev(obj))
    results = [sc.optimize.minimize(obj, np.random.randn(2*d_b*d_s),\
                                         jac=jac,\
                                         tol=1e-26, options={"disp": False, "maxiter": 10000})
                    for r in range(R)]
    if dir == "min":
        result = results[np.argmin([r.fun for r in results])]
    elif dir == "max":
        result = results[np.argmax([r.fun for r in results])]
    V = result.x.reshape(2, d_s, d_b)
    B = V[0] + 1j*V[1]
    return np.array(jp.linalg.qr(B.T)[0].T), np.sqrt(abs(result.fun))

def min_unitary_completion(D, B, R=10):
    D = flatten_if_needed(D)
    d_s, d_b = B.shape
    d_c = d_b - d_s
    @jax.jit
    def obj(V):
        V = V.reshape(2, d_c, d_b)
        C = V[0] + 1j*V[1]
        U = jp.vstack([B, C])
        unitarity = jp.linalg.norm(U @ U.conj().T - jp.eye(d_b))
        ce = 1 - jp.sum(abs(jp.einsum("axy, xz, bzq, yq->ab", D.conj(), U, D, U.conj()))**4)/d_b**6
        return unitarity + ce**2
    jac = jax.jit(jax.jacrev(obj))
    results = [sc.optimize.minimize(obj, np.random.randn(2*d_b*d_c), jac=jac,\
                                         tol=1e-26, options={"disp": False, "maxiter": 10000})
                        for r in range(R)]
    result = results[np.argmin([r.fun for r in results])]
    V = result.x.reshape(2, d_c, d_b)
    C = V[0] + 1j*V[1]
    return np.array(jp.vstack([B, C]))