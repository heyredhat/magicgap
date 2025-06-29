import os
import shutil
from pathlib import Path

from magicgap import *

repository = {}
data_path = "/hpcstor6/scratch01/m/matthew.weiss001/magicgap"
destination_path = "data"
os.makedirs("data", exist_ok=True)
search_path = Path(data_path)
for file_path in search_path.glob('*.npz'):
    try:
        with np.load(file_path) as data:
            d, n, d_s, R, mc, M, avg_se_min, avg_se_max = data["result"]
            d, n, d_s = int(d), int(n), int(d_s)
            if n not in repository:
                repository[n] = {}
            if d not in repository[n]:
                repository[n][d] = {}
            if d_s not in repository[n][d]:
                repository[n][d][d_s] = {"min": [], "max": [], "filename": []}
            repository[n][d][d_s]["min"].append(avg_se_min)
            repository[n][d][d_s]["max"].append(avg_se_max) 
            repository[n][d][d_s]["filename"].append(file_path.name)
    except Exception as e:
        print(f"Error processing file '{file_path.name}': {e}")

extrema = {}
for n in repository.keys():
    extrema[n] = {}
    for d in repository[n].keys():
        extrema[n][d] = {}
        D = wh_operators(*[d]*n)
        for d_s in repository[n][d].keys():
            extrema[n][d][d_s] = {}
            argmin = np.argmin(repository[n][d][d_s]["min"])
            argmax = np.argmax(repository[n][d][d_s]["max"])
            extrema[n][d][d_s]["min_file"] = repository[n][d][d_s]["filename"][argmin]
            extrema[n][d][d_s]["max_file"] = repository[n][d][d_s]["filename"][argmax]
            min_file = search_path / extrema[n][d][d_s]["min_file"]
            max_file = search_path / extrema[n][d][d_s]["max_file"] 
            shutil.copy(min_file, destination_path)
            shutil.copy(max_file, destination_path)
            with np.load(min_file) as data:
                Bmin = data["Bmin"]
            with np.load(max_file) as data:
                Bmax = data["Bmax"]
            extrema[n][d][d_s]["min"] = avg_magic_subspace_multiqudit(D, Bmin, d)
            extrema[n][d][d_s]["max"] = avg_magic_subspace_multiqudit(D, Bmax, d) 
save_data("extremal_magic_data", extrema)