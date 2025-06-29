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

def qudit_wh_operators(d, matrix=True, expanded=False):
    d_bar = d if d % 2 != 0 else 2 * d
    w = np.exp(2 * np.pi * 1j / d)
    tau = -np.exp(1j * np.pi / d)
    Z = np.diag([w ** i for i in range(d)])
    F = np.array([[w ** (i * j) for j in range(d)] for i in range(d)]) / np.sqrt(d)
    X = F.conj().T @ Z @ F
    up_to = d_bar if expanded else d
    D = np.array([[tau ** (q * p) * np.linalg.matrix_power(X, q) @ np.linalg.matrix_power(Z, p) for p in range(up_to)] for q in range(up_to)])
    return D if matrix else D.reshape(up_to * up_to, d, d)

def wh_operators(*dims, matrix=True, expanded=False):
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

def flatten_if_needed(D):
    return D if D.ndim == 3 else D.reshape(-1, D.shape[-1], D.shape[-1])

@jax.jit
def sum_symp_indices(ind, d_bar):
    return tuple(jp.sum(jp.stack(jp.asarray(ind)), axis=0) % d_bar)

@jax.jit
def symplectic_product(a, b):
    a_reshaped = jp.asarray(a).reshape((-1, 2))
    b_reshaped = jp.asarray(b).reshape((-1, 2))
    return jp.sum(a_reshaped[:, 1] * b_reshaped[:, 0] - a_reshaped[:, 0] * b_reshaped[:, 1])

def construct_chi(D, X):
    d_b = D.shape[-1]
    return np.einsum("...jk,jk->...", D.conj(), X) / d_b

def chi_func_scalar(chi_base, d, c):
    c = jp.asarray(c)
    a = c % d
    b = (c - a) / d
    return jax.lax.cond(
        d % 2 != 0,
        lambda: chi_base[tuple(a)],
        lambda: (-1.0)**symplectic_product(a, b) * chi_base[tuple(a)],
    )

@jax.jit
def expand_chi(base):
    d = base.shape[0]
    n = base.ndim // 2
    d_bar = d if d % 2 != 0 else 2 * d
    c_ranges = [jp.arange(d_bar)] * (2 * n)
    c_grid = jp.stack(jp.meshgrid(*c_ranges, indexing='ij'), axis=-1).reshape(-1, 2 * n)
    vectorized_chi_func = jax.vmap(lambda c: chi_func_scalar(base, d, c), in_axes=0)
    chi_flat = vectorized_chi_func(c_grid)
    target_shape = [d_bar, d_bar] * n
    return chi_flat.reshape(target_shape)

def construct_Q(D):
    D = flatten_if_needed(D)
    d = D.shape[-1]
    Q = sum([kron(O, O.conj().T, O, O.conj().T) for O in D])/d**2
    return Q

####################################################################################################

def renyi_stabilizer_entropy(D, ket, alpha):
    D = flatten_if_needed(D)
    d = ket.shape[0]
    chi = np.array([abs(ket.conj() @ O @ ket)**2 for O in D])/d
    return (1/(1-alpha))*np.log2(np.sum(chi**(2*alpha))) - np.log2(d)

def max_renyi_stabilizer_entropy(d, alpha=2):
    return (1/(1-alpha))*np.log2((1+(d-1)*(d+1)**(1-alpha))/d)

def linear_stabilizer_entropy(D, ket):
    D = flatten_if_needed(D)
    d = ket.shape[0]
    chi = np.array([abs(ket.conj() @ O @ ket)**2 for O in D])/d
    return 1 - d*sum(chi**2)

####################################################################################################

def avg_magic_exact(D):
    D = flatten_if_needed(D)
    d = D[0].shape[0]
    return (1 - d*(construct_Q(D) @ haar_moment(d, 4)).trace()).real

def avg_magic_mc(D, M=750):
    D = flatten_if_needed(D)
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
    D = flatten_if_needed(D)
    d_b = D[0].shape[0]
    d_s = B.shape[0]
    B4 = tensor_power(B, 4)
    return 1 - d_b*(construct_Q(D) @ B4.T @ haar_moment(d_s, 4) @ B4.conj()).trace().real

def avg_magic_subspace_mc(D, O, M=750, projector=False):
    D = flatten_if_needed(D)
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

def extremize_subspace_magic_mc(D, d_s, dir, R=1, M=750, scale_sampling=True):
    D = flatten_if_needed(D)
    d_b = D[0].shape[0]
    s = 1 if dir == "min" else -1
    if scale_sampling:
        M = M*d_s
    K = rand_kets(d_s, M)
    
    @jax.jit
    def obj(V):
        V = V.reshape(2, d_s, d_b)
        B = V[0] + 1j*V[1]
        B = jp.linalg.qr(B.T)[0].T
        L = K @ B
        ase = jp.mean(1-jp.sum(abs(jp.einsum('id,jde,ie->ij', L.conj(), D, L))**4, axis=1)/d_b)
        return s*ase**2

    results = [sc.optimize.minimize(obj, np.random.randn(2*d_b*d_s),\
                                         jac=jax.jit(jax.jacrev(obj)),\
                                         tol=1e-26, options={"disp": False, "maxiter": 10000})
                    for r in range(R)]
    if dir == "min":
        result = results[np.argmin([r.fun for r in results])]
    elif dir == "max":
        result = results[np.argmax([r.fun for r in results])]
    V = result.x.reshape(2, d_s, d_b)
    B =  V[0] + 1j*V[1]
    return np.array(jp.linalg.qr(B.T)[0].T), np.sqrt(abs(result.fun))
