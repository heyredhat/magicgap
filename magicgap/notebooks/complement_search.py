from magicgap import *

def analyze_complement(extrema, base, n, d, d_s, M=750, T=100):
    D = wh_operators(*[d]*n)
    min_ase = extrema[n][d][d_s]["min"]
    with np.load(base+extrema[n][d][d_s]["min_file"]) as data:
        B = data["Bmin"]
    min_ase_optimal_supp = np.array(avg_magic_subspace_mc_with_optimal_support(D, B, M=M, R=10))
    ket_c = min_ase_support_vector(D, B, M=M, R=10)
    min_ase_fixed_supp = np.array(avg_magic_subspace_mc_with_support_vector(D, B, ket_c, M=M))

    diffs = []
    for _ in range(T):
        B = rand_basis(d_s, d**n)
        ase = avg_magic_subspace_multiqudit(D, B, d)
        ket_c = min_ase_support_vector(D, B, M=M, R=10)
        supp_ase, supp_ase_std = avg_magic_subspace_mc_with_support_vector(D, B, ket_c, M=M)
        diff = (supp_ase - ase)/ase
        diffs.append(diff)
    avg_ase_diff_fixed_supp = np.array(np.mean(diffs), np.std(diffs))
    np.savez("complement_data/n%d_d%d_ds%d" % (n, d, d_s), n=n, d=d, d_s=d_s, min_ase=min_ase, min_ase_optimal_supp=min_ase_optimal_supp,\
                    min_ase_fixed_supp=min_ase_fixed_supp, avg_ase_diff_fixed_supp=avg_ase_diff_fixed_supp)

base = "extremize_magic/data/"
extrema = load_data(base+"extremal_magic_data")

n = 1
for d in [4,5,6]:
    for d_s in range(2, d):
        analyze_complement(extrema, base, n, d, d_s, M=750)

d = 2
for n in [2,3]:
    for d_s in range(2, d**n):
        print("%d,%d" % (n, d_s))
        analyze_complement(extrema, base, n, d, d_s, M=750)