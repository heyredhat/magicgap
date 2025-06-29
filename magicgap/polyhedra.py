import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from .utils import upgrade
from .spin import recoupled_spin_basis, spin_matrices, j_d
from .polyhedrec import *

def intertwiner_basis(*j_values):
    return np.concatenate(recoupled_spin_basis(*j_values)[0])

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

def tetrahedron_volume_operator(angle_ops):
    gamma = 1
    prefactor = (np.sqrt(2)/3)**2*(8*np.pi*gamma)**3
    X = angle_ops[(0,1)]
    Y = angle_ops[(0,2)]
    return (-1j)*(X @ Y - Y @ X)

####################################################################################################

def expected_polyhedron_gram(n_faces, angle_ops, rho):
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

def construct_polyhedron(R):
    areas = np.linalg.norm(R, axis=1)
    unit_normals = list((R.T/areas).T)
    return reconstruct(unit_normals, areas)

####################################################################################################

def plot_poly(poly, file=None):
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
    if file != None:
    	plt.savefig(file)