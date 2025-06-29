import numpy as np
import scipy as sc
from functools import partial

import jax
import jax.numpy as jp
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from .stabilizer_entropy import expand_chi

@partial(jax.jit, static_argnums=(1,))
def qudit_term1(chi_abs2, d):
    return 3 * d ** 2 * jp.sum(chi_abs2 ** 2)

@partial(jax.jit, static_argnums=(1,))
def qudit_term2(chi_abs2, d):
    w = jp.exp(2 * jp.pi * 1j / d)
    coords = jp.arange(d)
    a1, a2 = jp.meshgrid(coords, coords, indexing='ij')
    b1, b2 = jp.meshgrid(coords, coords, indexing='ij')
    sympl = (a2[..., None, None] * b1[None, None, ...] - a1[..., None, None] * b2[None, None, ...]) % d
    chi_flat2 = chi_abs2.reshape(-1).astype(jp.complex128)
    phase = w ** sympl.reshape(d * d, d * d)
    return 6 * d * jp.einsum("i,ij,j->", chi_flat2, phase, chi_flat2)

@partial(jax.jit, static_argnums=(2,))
def qudit_term3(chi, chi_trunc, d):
    coords = jp.arange(d)

    def outer_body(carry, a1):
        def inner_body(inner_carry, a2):
            chi_shifted = jp.roll(jp.roll(chi, shift=-2*a1, axis=0), shift=-2*a2, axis=1)
            val = chi[a1, a2]**2 * jp.sum(chi_trunc * chi_shifted[:d, :d].conj())
            return inner_carry + val, None
        inner_sum, _ = jax.lax.scan(inner_body, 0 + 0j, coords)
        return carry + inner_sum, None

    total_sum, _ = jax.lax.scan(outer_body, 0 + 0j, coords)
    return 6 * d * total_sum

@partial(jax.jit, static_argnums=(1,))
def qudit_term4(chi, d):
    d_bar = d if d % 2 != 0 else 2 * d
    tau = -jp.exp(1j * jp.pi / d)
    coords = jp.arange(d)
    coords_pairs = jp.array(jp.meshgrid(coords, coords, indexing='ij')).reshape(2, -1).T  # shape (d^2, 2)

    def inner_body(acc, x):
        C1, C2, A1, A2, B1, B2 = x
        idx1 = (A1 + B1 + C1) % d_bar
        idx2 = (A2 + B2 + C2) % d_bar
        exponent = ((A1 * ((B2 - C2) % d_bar) - A2 * ((B1 - C1) % d_bar)) - (C1 * B2 - C2 * B1)) % (2 * d)
        val = (chi[A1, A2] * chi[B1, B2] * chi[C1, C2] *
               jp.conj(chi[idx1, idx2]) * tau**exponent)
        return acc + val, None

    def middle_body(acc, B):
        B1, B2 = B
        A1, A2 = acc[1]
        n = coords_pairs.shape[0]  
        C1C2 = coords_pairs
        inner_inputs = jp.column_stack([
            C1C2[:, 0], C1C2[:, 1],
            jp.full((n,), A1), jp.full((n,), A2),
            jp.full((n,), B1), jp.full((n,), B2)
        ])
        total_C, _ = jax.lax.scan(inner_body, 0 + 0j, inner_inputs)
        return (acc[0] + total_C, (A1, A2)), None

    def outer_body(acc, A):
        A1, A2 = A
        total_B, _ = jax.lax.scan(middle_body, (0 + 0j, (A1, A2)), coords_pairs)
        return acc + total_B[0], None

    total_sum, _ = jax.lax.scan(outer_body, 0 + 0j, coords_pairs)
    return 8 * total_sum

@partial(jax.jit, static_argnums=(2,))
def qudit_term5(chi, chi_trunc, d):
    coords = jp.arange(d)

    def outer_body(carry, a1):
        def inner_body(inner_carry, a2):
            shifted = jp.roll(jp.roll(chi.conj(), shift=-2*a1, axis=0), shift=-2*a2, axis=1)
            prod = chi_trunc * shifted[:d, :d]
            val = jp.abs(jp.sum(prod)) ** 2
            return inner_carry + val, None
        inner_sum, _ = jax.lax.scan(inner_body, 0.0, coords)
        return carry + inner_sum, None

    total_sum, _ = jax.lax.scan(outer_body, 0.0, coords)
    return total_sum

def avg_magic_subspace_qudit(D, O, projector=False, return_terms=False):
    d = D[0, 0].shape[0]

    if projector:
        Pi = O
        d_s = sum(np.isclose(np.linalg.eigvals(Pi), 1))
    else:
        Pi = O.T @ O.conj()
        d_s = O.shape[0]

    chi_trunc = jp.einsum("ijkl,lk->ij", jp.transpose(D, (0, 1, 3, 2)).conj(), Pi) / d
    chi = expand_chi(chi_trunc)
    chi_abs2 = jp.abs(chi_trunc) ** 2

    term1 = qudit_term1(chi_abs2, d)
    term2 = qudit_term2(chi_abs2, d)
    term3 = qudit_term3(chi, chi_trunc, d)
    term4 = qudit_term4(chi, d)
    term5 = qudit_term5(chi, chi_trunc, d)
    terms = np.array([term1, term2, term3, term4, term5]).real

    if return_terms:
        return terms

    result = 1 - (d / (d_s * (d_s + 1) * (d_s + 2) * (d_s + 3))) * sum(terms)
    return float(result.astype(float))

def extremize_subspace_magic_qudit(D, d_s, dir, R=1):
    s = 1 if dir == "min" else -1
    d = D[0, 0].shape[0]
    d_bar = d if d % 2 != 0 else 2 * d

    @jax.jit
    def obj(V):
        V = V.reshape(2, d_s, d)
        B = V[0] + 1j*V[1]
        B = jp.linalg.qr(B.T)[0].T
        Pi = B.T @ B.conj()
        chi_trunc = jp.einsum("ijkl,lk->ij", jp.transpose(D, (0, 1, 3, 2)).conj(), B.conj().T @ B) / d
        chi = expand_chi(chi_trunc)
        chi_abs2 = jp.abs(chi_trunc) ** 2
        term1 = qudit_term1(chi_abs2, d)
        term2 = qudit_term2(chi_abs2, d)
        term3 = qudit_term3(chi, chi_trunc, d)
        term4 = qudit_term4(chi, d)
        term5 = qudit_term5(chi, chi_trunc, d)
        se =  1 - (d / (d_s * (d_s + 1) * (d_s + 2) * (d_s + 3))) * (term1 + term2 + term3 + term4 + term5).real
        return s*se**2
    jac = jax.jit(jax.jacrev(obj))

    results = [sc.optimize.minimize(obj, np.random.randn(2*d*d_s),\
                                         jac=jac,\
                                         tol=1e-26, options={"disp": False, "maxiter": 10000})
                    for r in range(R)]
    if dir == "min":
        result = results[np.argmin([r.fun for r in results])]
    elif dir == "max":
        result = results[np.argmax([r.fun for r in results])]
    V = result.x.reshape(2, d_s, d)
    B =  V[0] + 1j*V[1]
    return np.array(jp.linalg.qr(B.T)[0].T), np.sqrt(abs(result.fun))