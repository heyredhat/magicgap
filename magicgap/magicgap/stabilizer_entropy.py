import numpy as np
import scipy as sc
from functools import partial, reduce
from itertools import product

import jax
import jax.numpy as jp
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from .utils import kron, rand_kets, rand_basis, haar_moment, tensor_power

def qudit_wh_operators(d, matrix=False, expanded=False):
    d_bar = d if d % 2 != 0 else 2 * d
    w = np.exp(2 * np.pi * 1j / d)
    tau = -np.exp(1j * np.pi / d)
    Z = np.diag([w ** i for i in range(d)])
    F = np.array([[w ** (i * j) for j in range(d)] for i in range(d)]) / np.sqrt(d)
    X = F.conj().T @ Z @ F
    up_to = d_bar if expanded else d
    D = np.array([[tau ** (q * p) * np.linalg.matrix_power(X, q) @ np.linalg.matrix_power(Z, p) for p in range(up_to)] for q in range(up_to)])
    return D if matrix else D.reshape(up_to * up_to, d, d)

def wh_operators(*dims, matrix=False, expanded=False):
    import itertools
    tensors = [qudit_wh_operators(d, matrix=True, expanded=expanded) for d in dims]
    sym_dims = [(d if d % 2 != 0 else 2 * d) if expanded else d for d in dims for _ in range(2)]
    all_indices = list(itertools.product(*[range(s) for s in sym_dims]))
    total_d = np.prod(dims)
    D = np.zeros(sym_dims + [total_d, total_d], dtype=np.complex128)
    for idx in all_indices:
        ridx = np.array(idx).reshape(-1, 2)
        kron_factor = reduce(np.kron, [tensors[i][tuple(r)] for i, r in enumerate(ridx)])
        D[idx] = kron_factor
    return D if matrix else D.reshape(-1, total_d, total_d)

@jax.jit
def sum_symp_indices(ind, d_bar):
    return tuple(jp.sum(jp.stack(jp.asarray(ind)), axis=0) % d_bar)

@jax.jit
def symplectic_product(a, b):
    a_reshaped = jp.asarray(a).reshape((-1, 2))
    b_reshaped = jp.asarray(b).reshape((-1, 2))
    return jp.sum(a_reshaped[:, 1] * b_reshaped[:, 0] - a_reshaped[:, 0] * b_reshaped[:, 1])

def construct_Q(D):
    d = D.shape[-1]
    Q = sum([kron(O, O.conj().T, O, O.conj().T) for O in D])/d**2
    return Q

####################################################################################################

def renyi_stabilizer_entropy(D, ket, alpha):
    d = ket.shape[0]
    chi = np.array([abs(ket.conj() @ O @ ket)**2 for O in D])/d
    return (1/(1-alpha))*np.log2(np.sum(chi**(2*alpha))) - np.log2(d)

def max_renyi_stabilizer_entropy(d, alpha=2):
    return (1/(1-alpha))*np.log2((1+(d-1)*(d+1)**(1-alpha))/d)

def linear_stabilizer_entropy(D, ket):
    d = ket.shape[0]
    chi = np.array([abs(ket.conj() @ O @ ket)**2 for O in D])/d
    return 1 - d*sum(chi**2)

####################################################################################################

def avg_magic_exact(D):
    d = D[0].shape[0]
    return (1 - d*(construct_Q(D) @ haar_moment(d, 4)).trace()).real

def avg_magic_mc(D, M=750):
    d = D[0].shape[0]
    K = rand_kets(d, M)
    samples = 1 - np.sum(abs(np.einsum('id,jde,ie->ij', K.conj(), D, K))**4, axis=1)/d
    return (np.mean(samples), np.std(samples))

def avg_magic_analytic(d, nqubits=False):
    if nqubits:
        return 1 - 4/(d+3)
    if d % 2 != 0:
        return 1 - 3/(d+2)
    return 1 - 3*(d+2)/((d+1)*(d+3))

####################################################################################################

def avg_magic_subspace_naive(D, B):
    d_b = D[0].shape[0]
    d_s = B.shape[0]
    B4 = tensor_power(B, 4)
    return 1 - d_b*(construct_Q(D) @ B4.conj().T @ haar_moment(d_s, 4) @ B4).trace().real

def avg_magic_subspace_mc(D, O, M=750, projector=False):
    d_b = D[0].shape[0]
    if projector:
        L, V = np.linalg.eigh(O)
        B = np.array([V[:,i] for i, l in enumerate(L) if np.isclose(l, 1)])
        d_s = B.shape[0]
    else:
        B = O
    d_s = B.shape[0]
    K = rand_kets(d_s, M) @ B
    samples = 1 - np.sum(abs(np.einsum('id,jde,ie->ij', K.conj(), D, K))**4, axis=1)/d_b
    return (np.mean(samples), np.std(samples))

####################################################################################################

def avg_magic_avg_subspace_mc(D_b, d_s, M=250, R=250):
    d_b = D_b[0].shape[0]
    means = []
    stds = []
    for i in range(R):
        mean, std = avg_magic_subspace_mc(D_b, rand_basis(d_s, d_b), M=M)
        means.append(mean)
        stds.append(std)
    means, stds = np.array(means), np.array(stds)
    weights = 1 / stds**2
    weighted_mean = np.sum(weights * means) / np.sum(weights)
    weighted_std = np.sqrt(1 / np.sum(weights))
    return weighted_mean, weighted_std

####################################################################################################

def extremize_subspace_magic_mc(D, d_small, dir, R=1, S=10, M=750):
    d_big = D[0].shape[0]
    s = 1 if dir == "min" else -1
    Ks = [rand_kets(d_small, M) for _ in range(S)]
    
    @jax.jit
    def obj(V):
        V = V.reshape(2, d_small, d_big)
        C = V[0] + 1j*V[1]
        C = jp.linalg.qr(C.conj().T)[0].conj().T
        samples = []
        for K in Ks:
            L = K @ C
            samples.append(jp.mean(1-jp.sum(abs(jp.einsum('id,jde,ie->ij', L.conj(), D, L))**4, axis=1)/d_big))
        return s*jp.mean(jp.array(samples))**2

    results = [sc.optimize.minimize(obj, np.random.randn(2*d_big*d_small),\
                                         jac=jax.jit(jax.jacrev(obj)),\
                                         tol=1e-26, options={"disp": False, "maxiter": 10000})
                    for r in range(R)]
    if dir == "min":
        result = results[np.argmin([r.fun for r in results])]
    elif dir == "max":
        result = results[np.argmax([r.fun for r in results])]
    V = result.x.reshape(2, d_small, d_big)
    C =  V[0] + 1j*V[1]
    return np.array(jp.linalg.qr(C.conj().T)[0].conj().T), np.sqrt(abs(result.fun))
