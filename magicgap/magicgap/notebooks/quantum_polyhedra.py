from magicgap import *

runs = [(1/2, 4), (1,4), (1/2, 6)]
results = []
for run in runs:
    local_j, n_faces = run
    B = intertwiner_basis(*[local_j]*n_faces)
    result = sample_mean(lambda : avg_magic_subspace_mc(wh_operators(*[j_d(local_j)]*n_faces), B, M=1000)[0], M=20)
    results.append(result)
    save_data("quantum_polyhedra", {"runs": runs, "results": results})