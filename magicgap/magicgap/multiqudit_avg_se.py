import numpy as np
import scipy as sc
import jax
import jax.numpy as jp
from jax import lax
from functools import reduce, partial

from .utils import kron
from .stabilizer_entropy import sum_symp_indices, symplectic_product, expand_chi

@partial(jax.jit, static_argnums=(1, 2))
def decode_multiqudit_index(i, d, n):
    base = d ** jp.arange(2 * n - 1, -1, -1)
    return ((i[:, None] // base) % d).astype(jp.int32)

@jax.jit
def multiqudit_term1(chi_trunc, d_b):
    chi_abs2 = jp.abs(chi_trunc) ** 2
    return 3 * d_b ** 2 * jp.sum(chi_abs2 ** 2)

@partial(jax.jit, static_argnums=(2, 3))
def multiqudit_term2(chi, d_b, d, n):
    total = d ** (2 * n)
    idx = jp.arange(total)
    coords = decode_multiqudit_index(idx, d, n)
    chi_vals = jp.abs(jax.vmap(lambda x: chi[tuple(x)])(coords)) ** 2
    sp_matrix = jax.vmap(lambda a: jax.vmap(lambda b: symplectic_product(a, b))(coords))(coords)
    phase = jp.exp(2j * jp.pi * sp_matrix / d)
    contrib = chi_vals[:, None] * chi_vals[None, :] * phase
    return 6 * d_b * jp.sum(contrib)

@partial(jax.jit, static_argnums=(2, 3, 4))
def multiqudit_term3(chi, d_b, d, n, d_bar):
    total = d ** (2 * n)
    idx = jp.arange(total)
    coords = decode_multiqudit_index(idx, d, n)
    chi_vals = jax.vmap(lambda a: chi[tuple(a)])(coords)

    def body(i, acc):
        a = coords[i]
        chi_a2 = chi_vals[i] ** 2
        def inner_body(j, inner_acc):
            b = coords[j]
            s_idx = sum_symp_indices([b, 2 * a], d_bar)
            return inner_acc + chi_a2 * chi[tuple(b)] * jp.conj(chi[s_idx])
        inner_sum = lax.fori_loop(0, total, inner_body, 0.0 + 0.0j)
        return acc + inner_sum

    total_sum = lax.fori_loop(0, total, body, 0.0 + 0.0j)
    return 6 * d_b * total_sum

@partial(jax.jit, static_argnums=(1, 2, 3))
def multiqudit_term4(chi, d, n, d_bar, tau):
    total = d ** (2 * n)
    idx = jp.arange(total)
    coords = decode_multiqudit_index(idx, d, n)

    def body(i, acc):
        a = coords[i]
        chi_a = chi[tuple(a)]
        def inner_body(j, acc2):
            b = coords[j]
            chi_b = chi[tuple(b)]
            def inner_most(k, acc3):
                c = coords[k]
                chi_c = chi[tuple(c)]
                idx_sum = sum_symp_indices([a, b, c], d_bar)
                phase = tau ** (symplectic_product(a, b) - symplectic_product(a, c) - symplectic_product(c, b))
                return acc3 + chi_a * chi_b * chi_c * jp.conj(chi[idx_sum]) * phase
            inner_sum = lax.fori_loop(0, total, inner_most, 0.0 + 0.0j)
            return acc2 + inner_sum
        mid_sum = lax.fori_loop(0, total, inner_body, 0.0 + 0.0j)
        return acc + mid_sum

    total_sum = lax.fori_loop(0, total, body, 0.0 + 0.0j)
    return 8 * total_sum

@partial(jax.jit, static_argnums=(1, 2, 3))
def multiqudit_term5(chi, d, n, d_bar):
    total = d ** (2 * n)
    idx = jp.arange(total)
    coords = decode_multiqudit_index(idx, d, n)

    def body(i, acc):
        a = coords[i]
        def inner_body(j, inner_acc):
            b = coords[j]
            s_idx = sum_symp_indices([b, -2 * a], d_bar)
            return inner_acc + chi[tuple(b)] * jp.conj(chi[s_idx])
        inner_sum = lax.fori_loop(0, total, inner_body, 0.0 + 0.0j)
        return acc + jp.abs(inner_sum) ** 2

    total_sum = lax.fori_loop(0, total, body, 0.0)
    return total_sum

def avg_magic_subspace_multiqudit(D, O, d, projector=False, return_terms=False):
    d_bar = d if d % 2 != 0 else 2 * d
    w = jp.exp(2 * jp.pi * 1j / d)
    tau = -jp.exp(1j * jp.pi / d)

    n = (D.ndim - 2) // 2
    d_b = d ** n

    if projector:
        d_s = sum(np.isclose(np.linalg.eigvals(O), 1))
        Pi = O
    else:
        d_s = O.shape[0]
        Pi = O.T @ O.conj()

    chi_trunc = jp.einsum("...jk,jk->...", D.conj(), Pi) / d_b
    chi = expand_chi(chi_trunc)

    term1 = multiqudit_term1(chi_trunc, d_b)
    term2 = multiqudit_term2(chi, d_b, d, n)
    term3 = multiqudit_term3(chi, d_b, d, n, d_bar)
    term4 = multiqudit_term4(chi, d, n, d_bar, tau)
    term5 = multiqudit_term5(chi, d, n, d_bar)
    terms = np.array([term1, term2, term3, term4, term5]).real

    if return_terms:
        return terms

    norm = d_b / (d_s * (d_s + 1) * (d_s + 2) * (d_s + 3))
    return float(1 - norm * sum(terms).real)

def avg_magic_subspace_multiqudit_term5(D, O, d, projector=False):
    n = (D.ndim - 2) // 2
    d_b = d ** n

    if projector:
        d_s = sum(np.isclose(np.linalg.eigvals(O), 1))
        Pi = O
    else:
        d_s = O.shape[0]
        Pi = O.T @ O.conj()

    chi = expand_chi(jp.einsum("...jk,jk->...", D.conj(), Pi) / d_b)
    return multiqudit_term5(chi, d, n, d_bar)

def extremize_subspace_magic_multiqudit(D, d, d_s, dir, R=1):
    s = 1 if dir == "min" else -1
    d_bar = d if d % 2 != 0 else 2 * d
    w = jp.exp(2 * jp.pi * 1j / d)
    tau = -jp.exp(1j * jp.pi / d)
    n = (D.ndim - 2) // 2
    d_b = d ** n

    @jax.jit
    def obj(V):
        V = V.reshape(2, d_s, d_b)
        B = V[0] + 1j*V[1]
        B = jp.linalg.qr(B.T)[0].T
        Pi = B.T @ B.conj()
        chi_trunc = jp.einsum("...jk,jk->...", D.conj(), Pi) / d_b
        chi = expand_chi(chi_trunc)

        term1 = multiqudit_term1(chi_trunc, d_b)
        term2 = multiqudit_term2(chi, d_b, d, n)
        term3 = multiqudit_term3(chi, d_b, d, n, d_bar)
        term4 = multiqudit_term4(chi, d, n, d_bar, tau)
        term5 = multiqudit_term5(chi, d, n, d_bar)
        norm = d_b / (d_s * (d_s + 1) * (d_s + 2) * (d_s + 3))
        se = 1 - norm * (term1 + term2 + term3 + term4 + term5).real
        return s*se**2
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
    B =  V[0] + 1j*V[1]
    return np.array(jp.linalg.qr(B.T)[0].T), np.sqrt(abs(result.fun))
