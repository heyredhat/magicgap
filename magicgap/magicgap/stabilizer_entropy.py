import numpy as np
from functools import partial
from itertools import product

import jax
import jax.numpy as jp
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

from .utils import kron, rand_kets, haar_moment

def qudit_wh_operators(d, matrix=False, expanded=False):
    d_bar = d if d % 2 != 0 else 2*d
    w = np.exp(2*np.pi*1j/d)
    tau = -np.exp(1j*np.pi/d)
    Z = np.diag([w**i for i in range(d)])
    F = lambda d: np.array([[w**(i*j) for j in range(d)] for i in range(d)])/np.sqrt(d)
    X = F(d).conj().T @ Z @ F(d)
    up_to = d_bar if expanded else d
    D = np.array([[tau**(q*p) * np.linalg.matrix_power(X, q) @ np.linalg.matrix_power(Z, p) for p in range(up_to)] for q in range(up_to)])
    return D if matrix else D.reshape(d**2, d, d)

def wh_operators(*dims):
    return np.array([kron(*_) for _ in product(*[list(qudit_wh_operators(d)) for d in dims])])

def symplectic_product(a, b):
    return a[1]*b[0] - a[0]*b[1]

def test_wh_operators(d):
    d_bar = d if d % 2 != 0 else 2*d
    D = qudit_wh_operators(d, matrix=True, expanded=True)
    w = np.exp(2*np.pi*1j/d)
    tau = (-np.exp(1j*np.pi/d))

    one = np.all(np.array([np.allclose(D[i,j].conj().T, D[(-i)%d_bar, (-j)%d_bar]) for j in range(d) for i in range(d)]))

    A = np.array([[D[i,j]@D[k,l] for l in range(d) for k in range(d)] for j in range(d) for i in range(d)]);
    B = np.array([[tau**symplectic_product([i,j], [k,l])*D[(i+k)%d_bar, (j+l)%d_bar] for l in range(d) for k in range(d)] for j in range(d) for i in range(d)]);
    two = np.allclose(A,B)

    A = np.array([[D[(i+d*k) % d_bar, (j+d*l) % d_bar] for l in range(d) for k in range(d)] for j in range(d) for i in range(d)]);
    B = np.array([[D[i,j] if d % 2 != 0 else (-1)**symplectic_product([i,j], [k,l])*D[i,j] for l in range(d) for k in range(d)] for j in range(d) for i in range(d)]);
    three = np.allclose(A,B)

    return one and two and three

def construct_Q(D):
    d = D.shape[-1]
    Q = sum([kron(O, O.conj().T, O, O.conj().T) for O in D])/d**2
    return Q

####################################################################################################

def renyi_stabilizer_entropy(alpha, ket, D):
    d = ket.shape[0]
    chi = np.array([abs(ket.conj() @ O @ ket)**2 for O in D])/d
    return (1/(1-alpha))*np.log2(np.sum(chi**(2*alpha))) - np.log2(d)

def max_renyi_stabilizer_entropy(d, alpha=2):
    return (1/(1-alpha))*np.log2((1+(d-1)*(d+1)**(1-alpha))/d)

def linear_stabilizer_entropy(ket, D):
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

def avg_magic_analytic(D):
    return 1 - (3/(d+2) if d % 2 != 0 else 3*(d+2)/((d+1)*(d+3)))

####################################################################################################

def avg_magic_subspace_naive(D, B):
    d_b = D[0].shape[0]
    d_s = B.shape[0]
    B4 = tensor_power(B, 4)
    return 1 - d_b*(construct_Q(D) @ B4.conj().T @ haar_moment(d_s, 4) @ B4).trace().real

def avg_magic_subspace_mc(D, B, M=750):
    d_b = D[0].shape[0]
    d_s = B.shape[0]
    K = rand_kets(d_s, M) @ B
    samples = 1 - np.sum(abs(np.einsum('id,jde,ie->ij', K.conj(), D, K))**4, axis=1)/d_b
    return (np.mean(samples), np.std(samples))

@partial(jax.jit, static_argnums=(2,))
def term3_scan(chi, chi_trunc, d):
    """
    Computes term3 using nested lax.scan loops:
    term3 = 6 * d * sum_{a1,a2} chi[a1,a2]^2 * sum_{x,y} chi_trunc[x,y] * conj(chi shifted by -2a1,-2a2)[x,y]
    """
    coords = jp.arange(d)

    def outer_body(carry, a1):
        def inner_body(inner_carry, a2):
            # Shift chi by (-2a1, -2a2) in axes 0 and 1 respectively
            chi_shifted = jp.roll(jp.roll(chi, shift=-2*a1, axis=0), shift=-2*a2, axis=1)
            val = chi[a1, a2]**2 * jp.sum(chi_trunc * chi_shifted[:d, :d].conj())
            return inner_carry + val, None

        inner_sum, _ = jax.lax.scan(inner_body, 0 + 0j, coords)
        return carry + inner_sum, None

    total_sum, _ = jax.lax.scan(outer_body, 0 + 0j, coords)
    return 6 * d * total_sum

@partial(jax.jit, static_argnums=(1,))
def term4_scan(chi, d):
    """
    Computes term4 using nested lax.scan loops with batching over coordinate pairs:
    term4 = 8 * sum_{A,B,C} chi[A] * chi[B] * chi[C] * conj(chi[(A+B+C) mod d_bar]) * tau^{exponent}

    The iteration is over triples (A, B, C) where each is a pair of coords (a1,a2).
    Uses nested scans over these pairs without materializing the full 6D tensor.
    """
    d_bar = d if d % 2 != 0 else 2 * d
    tau = -jp.exp(1j * jp.pi / d)
    coords = jp.arange(d)

    # Precompute all coordinate pairs (for A, B, C)
    coords_pairs = jp.array(jp.meshgrid(coords, coords, indexing='ij')).reshape(2, -1).T  # shape (d^2, 2)

    def inner_body(acc, x):
        """
        Inner loop over C1, C2 given fixed A1,A2 and B1,B2.
        Accumulates sum of chi products with phase factor.
        """
        C1, C2, A1, A2, B1, B2 = x
        idx1 = (A1 + B1 + C1) % d_bar
        idx2 = (A2 + B2 + C2) % d_bar

        exponent = ((A1 * ((B2 - C2) % d_bar) - A2 * ((B1 - C1) % d_bar)) - (C1 * B2 - C2 * B1)) % (2 * d)
        val = (chi[A1, A2] * chi[B1, B2] * chi[C1, C2] *
               jp.conj(chi[idx1, idx2]) * tau**exponent)
        return acc + val, None

    def middle_body(acc, B):
        """
        Middle loop over B1,B2 given fixed A1,A2.
        Runs scan over all C pairs.
        """
        B1, B2 = B
        A1, A2 = acc[1]

        n = coords_pairs.shape[0]  # d^2
        C1C2 = coords_pairs

        # Prepare inputs for inner scan: for each C pair combine with fixed A and B
        inner_inputs = jp.column_stack([
            C1C2[:, 0], C1C2[:, 1],
            jp.full((n,), A1), jp.full((n,), A2),
            jp.full((n,), B1), jp.full((n,), B2)
        ])

        total_C, _ = jax.lax.scan(inner_body, 0 + 0j, inner_inputs)
        return (acc[0] + total_C, (A1, A2)), None

    def outer_body(acc, A):
        """
        Outer loop over A1,A2.
        Runs scan over all B pairs.
        """
        A1, A2 = A
        total_B, _ = jax.lax.scan(middle_body, (0 + 0j, (A1, A2)), coords_pairs)
        return acc + total_B[0], None

    total_sum, _ = jax.lax.scan(outer_body, 0 + 0j, coords_pairs)
    return 8 * total_sum

@partial(jax.jit, static_argnums=(2,))
def term5_scan(chi, chi_trunc, d):
    """
    Computes term5 using nested lax.scan loops:
    term5 = sum_{a1,a2} |sum_{x,y} chi_trunc[x,y] * shifted_chi_conj[x,y]|^2
    where shifted_chi_conj is chi conjugate rolled by (-2a1, -2a2)
    """
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

@jax.jit
def avg_magic_subspace_exact(D, B):
    """
    Compute the average magic of a subspace exactly.

    Arguments:
    - D: Weyl-Heisenberg displacement operators, shape (d_bar,d_bar,d,d)
    - B: Orthonormal basis for the subspace, shape (d_s, d)

    Returns:
    - Average magic value (real scalar)
    """
    d = D[0, 0].shape[0]
    d_s = B.shape[0]
    d_bar = d if d % 2 != 0 else 2 * d

    w = jp.exp(2 * jp.pi * 1j / d)
    tau = -jp.exp(1j * jp.pi / d)
    coords = jp.arange(d)

    # Compute characteristic function chi
    # shape: (d_bar, d_bar) with complex entries
    chi = jp.einsum("ijkl,lk->ij", jp.transpose(D, (0, 1, 3, 2)).conj(), B.conj().T @ B) / d
    chi_trunc = chi[:d, :d]

    # Term 1: sum of squared absolute values of chi truncated
    chi_abs2 = jp.abs(chi_trunc) ** 2
    term1 = 3 * d ** 2 * jp.sum(chi_abs2 ** 2)

    # Term 2: double sum with symplectic form phase factor
    a1, a2 = jp.meshgrid(coords, coords, indexing='ij')
    b1, b2 = jp.meshgrid(coords, coords, indexing='ij')
    sympl = (a2[..., None, None] * b1[None, None, ...] - a1[..., None, None] * b2[None, None, ...]) % d
    chi_flat2 = chi_abs2.reshape(-1)
    phase = w ** sympl.reshape(d * d, d * d)
    term2 = 6 * d * jp.einsum("i,ij,j->", chi_flat2, phase, chi_flat2)

    # Terms 3,4,5 computed by efficient nested scans
    term3 = term3_scan(chi, chi_trunc, d)
    term4 = term4_scan(chi, d)
    term5 = term5_scan(chi, chi_trunc, d)

    # Combine all terms and normalize
    result = 1 - (d / (d_s * (d_s + 1) * (d_s + 2) * (d_s + 3))) * (term1 + term2 + term3 + term4 + term5).real
    return result

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

def extremize_subspace_magic_mc(D, d_s, dir, M=750, R=1):
    d_b = D[0].shape[0]
    s = 1 if dir == "min" else -1
    K = rand_kets(d_small, M)
    
    @jax.jit
    def obj(V):
        V = V.reshape(2, d_small, d_big)
        B = V[0] + 1j*V[1]
        B = jp.linalg.qr(B.conj().T)[0].conj().T
        L = K @ B
        avg_se = jp.mean(1-jp.sum(abs(jp.einsum('id,jde,ie->ij', L.conj(), D, L))**4, axis=1))/d_b
        return s*avg_se**2

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
    return (jp.linalg.qr(B.conj().T)[0].conj().T, result)
