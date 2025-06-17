import uuid
import argparse

from magicgap import *

base = "/hpcstor6/scratch01/m/matthew.weiss001/magicgap/"

parser = argparse.ArgumentParser(prog="ExtremizeASE")
parser.add_argument("-d", type=int, required=True)
parser.add_argument("-n", type=int, required=True)
parser.add_argument("-d_s", type=int, required=True)
parser.add_argument("-R", type=int, required=False, default=1)
parser.add_argument("-M", type=int, required=False, default=750)
parser.add_argument("-mc", action="store_true")
params = vars(parser.parse_args())
globals().update(params)

D = wh_operators(*[d]*n)

if mc:
    Bmin, avg_se_min = extremize_subspace_magic_mc(D, d_s, "min", R=R, M=M)
    Bmax, avg_se_max = extremize_subspace_magic_mc(D, d_s, "max", R=R, M=M)
else:
    if n == 1:
        Bmin, avg_se_min = extremize_subspace_magic_qudit(D, d_s, "min", R=R)
        Bmax, avg_se_max = extremize_subspace_magic_qudit(D, d_s, "max", R=R)
    else:
        Bmin, avg_se_min = extremize_subspace_magic_multiqudit(D, d, d_s, "min", R=R)
        Bmax, avg_se_max = extremize_subspace_magic_multiqudit(D, d, d_s, "max", R=R)

unique = str(uuid.uuid4())
filename = base+"%d_%d_%d-%s.npz" % (d, n, d_s, unique)
result = np.array([d, n, d_s, R, 1 if mc else 0, M, avg_se_min, avg_se_max])
np.savez(filename, result=result, Bmin=Bmin, Bmax=Bmax)