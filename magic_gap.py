import math
import numpy as np
import scipy as sc
np.set_printoptions(precision=3, suppress=True)

from sympy import S
from sympy.physics.quantum.cg import CG

import polyhedrec as pr
from functools import reduce
from itertools import product, permutations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import jax
import jax.numpy as jp
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

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

def rand_unitary(d):
    return sc.stats.unitary_group.rvs(d)

def rand_basis(n, d):
    return rand_unitary(d)[:n, :]

def kron(*A):
    return reduce(np.kron, A)

def tensor_power(A, t):
    return kron(*[A]*t)

def upgrade(O, j, dims):
    return kron(*[O if i == j else np.eye(d) for i, d in enumerate(dims)])

def b(s, d=2):
    return kron(*[np.eye(d)[int(_)] for _ in s])

####################################################################################################

def d_j(d):
    return (d-1)/2

def j_d(j):
    return int(2*j+1)

def spin_basis_vector(j, m):
    return np.eye(int(2*j+1))[int(j-m)]

def m_values(j):
    return [m for m in np.arange(j, -j-1, -1)]

def spin_matrices(j):
    Jp = np.diag([np.sqrt((j-m)*(j+m+1)) for m in np.arange(j-1, -j-1, -1)], k=1)
    return (Jp + Jp.conj().T)/2, (Jp - Jp.conj().T)/(2j), np.diag(m_values(j))

####################################################################################################

# From qutip
def _factorial_prod(N, arr):
    arr[:int(N)] += 1

def _factorial_div(N, arr):
    arr[:int(N)] -= 1

def _to_long(arr):
    prod = 1
    for i, v in enumerate(arr):
        prod *= (i+1)**int(v)
    return prod

def clebsch_gordan(j1, j2, j3, m1, m2, m3):
    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    c_factor = np.zeros((int(j1 + j2 + j3 + 1)), np.int32)
    _factorial_prod(j3 + j1 - j2, c_factor)
    _factorial_prod(j3 - j1 + j2, c_factor)
    _factorial_prod(j1 + j2 - j3, c_factor)
    _factorial_prod(j3 + m3, c_factor)
    _factorial_prod(j3 - m3, c_factor)
    _factorial_div(j1 + j2 + j3 + 1, c_factor)
    _factorial_div(j1 - m1, c_factor)
    _factorial_div(j1 + m1, c_factor)
    _factorial_div(j2 - m2, c_factor)
    _factorial_div(j2 + m2, c_factor)
    C = np.sqrt((2.0 * j3 + 1.0)*_to_long(c_factor))

    s_factors = np.zeros(((vmax + 1 - vmin), (int(j1 + j2 + j3))), np.int32)
    sign = (-1) ** (vmin + j2 + m2)
    for i,v in enumerate(range(vmin, vmax + 1)):
        factor = s_factors[i,:]
        _factorial_prod(j2 + j3 + m1 - v, factor)
        _factorial_prod(j1 - m1 + v, factor)
        _factorial_div(j3 - j1 + j2 - v, factor)
        _factorial_div(j3 + m3 - v, factor)
        _factorial_div(v + j1 - j2 - m3, factor)
        _factorial_div(v, factor)
    common_denominator = -np.min(s_factors, axis=0)
    numerators = s_factors + common_denominator
    S = sum([(-1)**i * _to_long(vec) for i,vec in enumerate(numerators)]) * \
        sign / _to_long(common_denominator)
    return C * S

####################################################################################################

def J_values(j1, j2):
    return np.arange(abs(j1-j2), j1+j2+1, 1)

def recoupled_basis_(basis1, basis2):
    j1, j2 = d_j(basis1.shape[0]), d_j(basis2.shape[0])
    return dict([(J, [np.array([sum([clebsch_gordan(j1, j2, J, m1, m2, M)*\
                                    np.kron(basis1[int(j1-m1)], basis2[int(j2-m2)])\
                                        for m2 in m_values(j2) for m1 in m_values(j1)])
                                            for M in m_values(J)])]) for J in J_values(j1, j2)])

def recoupled_basis(*j_values):
    sectors = recoupled_basis_(np.eye(j_d(j_values[0])), np.eye(j_d(j_values[1])))
    for j in j_values[2:]:
        new_sectors = {}
        for J, bases in sectors.items():
            for basis in bases:
                for J_, bases in recoupled_basis_(basis, np.eye(j_d(j))).items():
                    if J_ not in new_sectors:
                        new_sectors[J_] = []
                    new_sectors[J_].extend(bases)
        sectors = new_sectors
    return sectors

def intertwiner_basis(*j_values):
    return np.concatenate(recoupled_basis(*j_values)[0])

####################################################################################################

def flux_operators(j_values):
    d_values = [j_d(j) for j in j_values]
    return [[upgrade(O, i, d_values) for O in spin_matrices(j)] for i, j in enumerate(j_values)]

def angle_operators(j_values, flux_ops, B=None):
    angle_ops = {}
    for a in range(len(j_values)):
        for b in range(len(j_values)):
            if a <= b:
                angle_ops[(a,b)] = sum([flux_ops[a][i] @ flux_ops[b][i] for i in range(3)])
    if type(B) != type(None):
        angle_ops = dict([(idx, B @ O @ B.conj().T) for idx, O in angle_ops.items()])
    return angle_ops

def tet_volume_operator(angle_ops):
    gamma = 1
    prefactor = (np.sqrt(2)/3)**2*(8*np.pi*gamma)**3
    X = angle_ops[(0,1)]
    Y = angle_ops[(0,2)]
    return (-1j)*(X @ Y - Y @ X)

####################################################################################################

def expected_gram(n_faces, angle_ops, rho):
    G = np.zeros((n_faces, n_faces), dtype=np.complex128)
    for i in range(n_faces):
        for j in range(n_faces):
            if i <= j:
                G[i,j] = (angle_ops[(i,j)] @ rho).trace()
                G[j,i] = G[i,j]
    return G.real      

def vecs3D_from_gram(G):
    U, S, V_ = np.linalg.svd(G)
    return U[:,:3] * np.sqrt(S[:3]) 

def construct_poly(R):
    areas = np.linalg.norm(R, axis=1)
    unit_normals = list((R.T/areas).T)
    return pr.reconstruct(unit_normals, areas)

def plot_poly(poly):
    vertices = np.array(poly.vertices)
    vertices = vertices - np.sum(vertices, axis=0)/len(vertices)
    vertex_adjacency = poly.v_adjacency_matrix
    faces = [face.vertices for face in poly.faces]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='k')
    edges = []
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            if vertex_adjacency[i, j]:
                edges.append([vertices[i], vertices[j]])
    edge_collection = Line3DCollection(edges, colors='k', linewidths=1)
    ax.add_collection3d(edge_collection)
    if faces:
        face_vertices = [[vertices[idx] for idx in face] for face in faces]
        face_collection = Poly3DCollection(face_vertices, facecolors='blue', edgecolors='k', alpha=0.5)
        ax.add_collection3d(face_collection)
    ax.set_box_aspect([1,1,1]) 
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    plt.show()

####################################################################################################

def qudit_wh_operators(d, matrix=False):
    w = np.exp(2*np.pi*1j/d)
    tau = -np.exp(1j*np.pi/d)
    Z = np.diag([w**i for i in range(d)])
    F = lambda d: np.array([[w**(i*j) for j in range(d)] for i in range(d)])/np.sqrt(d)
    X = F(d).conj().T @ Z @ F(d)
    if matrix:
        return np.array([[tau**(q*p) * np.linalg.matrix_power(X, q) @ np.linalg.matrix_power(Z, p) for p in range(d)] for q in range(d)])
    else:
        return np.array([tau**(q*p) * np.linalg.matrix_power(X, q) @ np.linalg.matrix_power(Z, p) for p in range(d) for q in range(d)])

def wh_operators(*dims):
    return np.array([kron(*_) for _ in product(*[list(qudit_wh_operators(d)) for d in dims])])

def construct_Q(D):
    d = D.shape[-1]
    Q = sum([kron(O, O.conj().T, O, O.conj().T) for O in D])/d**2
    return Q

####################################################################################################

def renyi2_stabilizer_entropy(ket, D):
    d = ket.shape[0]
    chi = np.array([abs(ket.conj() @ O @ ket)**2 for O in D])/d
    return -np.log2(sum(chi**2)) - np.log2(d)

def renyi2_stabilizer_entropy_(ket, Q):
    d = ket.shape[0]
    ket4 = kron(*[ket]*4)
    return -np.log2(ket4.conj() @ Q @ ket4) - np.log2(d)

def max_renyi_stabilizer_entropy(d, alpha=2):
    return (1/(1-alpha))*np.log2((1+(d-1)*(d+1)**(1-alpha))/d)

def linear_stabilizer_entropy(ket, D):
    d = ket.shape[0]
    chi = np.array([abs(ket.conj() @ O @ ket)**2 for O in D])/d
    return 1 - d*sum(chi**2)

def linear_stabilizer_entropy_(ket, Q):
    d = ket.shape[0]
    ket4 = kron(*[ket]*4)
    return 1 - d*ket4.conj() @ Q @ ket4

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

def plane_sphere(z):
    if z == np.inf:
        return np.array([0,0,-1])
    else:
        return np.array([2*z.real, 2*z.imag, 1-z.real**2 - z.imag**2])/(1+z.real**2+z.imag**2)
    
def sphere_plane(xyz):
    x, y, z = xyz
    if z == -1:
        return np.inf
    return (x+1j*y)/(1+z)

def majorana_polynomial(ket):
    j = (len(ket)-1)/2
    return np.polynomial.Polynomial(\
                    [(-1)**(j-m)*\
                     np.sqrt(sc.special.binom(2*j, j-m))*\
                     spin_basis_vector(j, m).conj() @ ket\
                         for m in np.arange(-j, j+1)])

def majorana_roots(ket):
    n = len(ket) - 1
    r = majorana_polynomial(ket).roots()
    return np.concatenate([r, np.repeat(np.inf, n-len(r))])

def majorana_stars(ket):
    return np.array([plane_sphere(r) for r in majorana_roots(ket)])

def stars_qubits(stars):
    roots = [sphere_plane(s) for s in stars]
    qubits = [np.array([0,1], dtype=np.complex128) if r == np.inf else np.array([1, r]) for r in roots]
    return np.array([qubit/np.linalg.norm(qubit) for qubit in qubits])

def perm_sym_product(kets):
    psi = sum([kron(*kets[perm, :]) for perm in permutations(list(range(len(kets))))])
    return psi/np.linalg.norm(psi)

def sym_qubit_basis(j):
    sym_qubit_states = {}
    for idx in product([0,1], repeat=int(2*j)):
        total = sum(idx)
        if total not in sym_qubit_states:
            sym_qubit_states[total] = []
        sym_qubit_states[total].append(b("".join([str(_) for _ in idx]), d=2))
    return np.array([sum(sym_qubit_states[i])/np.sqrt(len(sym_qubit_states[i])) for i in range(int(2*j+1))])

####################################################################################################

def prime_mubs(d, kets=False):
    w = np.exp(2*np.pi*1j/d)
    D = qudit_wh_operators(d, matrix=True)
    mubs = np.array([[sum([w**(-j*r)*D[(m*r)%d, r] for r in range(d)])/d for j in range(d)] for m in range(d)]+\
                    [[sum([w**(-j*r)*D[r,0] for r in range(d)])/d for j in range(d)]])
    if not kets:
        return mubs
    else:
        states = []
        for mub in mubs:
            for state in mub:
                L, V = np.linalg.eigh(state)
                states.append([v for i, v in enumerate(V.T) if np.isclose(L[i], 1)][0])
        return np.array(states); states

####################################################################################################

def avg_magic(D, exact=False, M=750, R=10):
    d = D[0].shape[0]
    if exact:
        return ((1 - d*(construct_Q(D) @ haar_moment(d, 4)).trace()).real, 0)
    else:
        samples = []
        for r in range(R):
            K = rand_kets(d, M)
            samples.append(np.mean(1-np.sum(abs(np.einsum('id,jde,ie->ij', K.conj(), D, K))**4, axis=1)/d))
        return (np.mean(samples), np.std(samples))

def avg_magic_subspace(D, B, exact=False, M=750, R=5):
    d_big = D[0].shape[0]
    d_small = B.shape[0]
    if exact:
        B4 = tensor_power(B, 4)
        return ((1 - d_big*(construct_Q(D) @ B4.conj().T @ haar_moment(d_small, 4) @ B4).trace().real), 0)
    else:
        samples = []
        for r in range(R):
            K = rand_kets(d_small, M) @ B
            samples.append(np.mean(1-np.sum(abs(np.einsum('id,jde,ie->ij', K.conj(), D, K))**4, axis=1)/d_big))
        return (np.mean(samples), np.std(samples))

def magic_gap(D_big, D_small, B, big_exact=True, small_exact=True, M=750, R=5):
    d_small = D_small[0].shape[0]
    d_big = D_big[0].shape[0]

    if d_small > 5:
         small_exact = False
    if d_big > 5:
        big_exact = False

    avg_small_magic, avg_small_magic_std = avg_magic(D_small, exact=small_exact, M=M, R=R)
    avg_big_magic, avg_big_magic_std = avg_magic_subspace(D_big, B, exact=big_exact, M=M, R=R)

    avg_magic_gap = avg_big_magic - avg_small_magic
    avg_magic_gap_std = np.sqrt(avg_big_magic_std**2 + avg_small_magic_std**2)
    
    return {"d_small": d_small, "d_big": d_big, "avg_magic_gap": avg_magic_gap, "avg_magic_gap_std": avg_magic_gap_std,\
            "avg_small_magic": avg_small_magic, "avg_small_magic_std": avg_small_magic_std,\
            "avg_big_magic": avg_big_magic, "avg_big_magic_std": avg_big_magic_std,\
            "big_exact": big_exact, "small_exact": small_exact, "M": M, "R": R, "B": B, "D_small": D_small, "D_big": D_big}

def avg_magic_gap(D_big, D_small, M=5000, R=5000):
    d_small = D_small[0].shape[0]
    d_big = D_big[0].shape[0]

    avg_small_magic = np.mean([jit_linear_stabilizer_entropy(rand_ket(d_small), D_small) for i in range(M)])

    samples = []
    for i in range(R): 
        B = rand_basis(d_small, d_big)
        samples.append([np.mean([jit_linear_stabilizer_entropy(B.conj().T @ rand_ket(d_small), D_big) for j in range(M)])])
    avg_big_magic = np.mean(samples)
    avg_big_magic_std = np.std(samples)

    avg_magic_gap = avg_big_magic - avg_small_magic
    return {"d_small": d_small, "d_big": d_big, "avg_magic_gap": avg_magic_gap, 
            "avg_small_magic": avg_small_magic, 
            "avg_big_magic": avg_big_magic, "avg_big_magic_std": avg_big_magic_std,\
            "M": M, "R": R, "D_small": D_small, "D_big": D_big}

def extremize_subspace_magic(D, d_small, dir, exact=False, R=1, S=10, M=750):
    if exact:
        return extremize_subspace_magic_exact(D, d_small, dir, R=R)
    else:
        return extremize_subspace_magic_approx(D, d_small, dir, R=R, S=S, M=M)

def extremize_subspace_magic_exact(D, d_small, dir, R=1):
    d_big = D[0].shape[0]
    Q_big = construct_Q(D)
    M4_small = haar_moment(d_small, 4)
    s = 1 if dir == "min" else -1

    @jax.jit
    def obj(V):
        V = V.reshape(2, d_small, d_big)
        C = V[0] + 1j*V[1]
        C = jp.linalg.qr(C.conj().T)[0].conj().T
        C4 = jp.kron(C, jp.kron(C, jp.kron(C, C)))
        avg_magic_big = (1 - d_big*(Q_big @ C4.conj().T @ M4_small @ C4).trace()).real 
        return s*avg_magic_big**2

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
    return (jp.linalg.qr(C.conj().T)[0].conj().T, result)

def extremize_subspace_magic_approx(D, d_small, dir, R=1, S=10, M=750):
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
    return (jp.linalg.qr(C.conj().T)[0].conj().T, result)