import sys

from magicgap import *

R = 25
results = []
for n in range(2, 6):
    d_b = 2**n
    D = wh_operators(*[2]*n, matrix=True, expanded=True)    
    for d_s in range(2, d_b):
        B_min, avg_se_min = extremize_subspace_magic_multiqudit(D, 2, d_s, "min", R=R)
        B_max, avg_se_max = extremize_subspace_magic_multiqudit(D, 2, d_s, "max", R=R)
        results.append({"d_b": d_b, "d_s": d_s, "avg_se_min": avg_se_min, "avg_se_max": avg_se_max})
        save_data("extremal_nqubit_magic", results)
        np.savez("data/extremal_nqubit_magic/%d_%d.npy" % (d_b, d_s), B_min=B_min, B_max=B_max)