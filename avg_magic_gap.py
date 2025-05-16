from magic_gap import *

import matplotlib.pyplot as plt
import pickle

def extract(d_big, d_small, data):
    for _ in data:
        if _["d_big"] == d_big and _["d_small"] == d_small:
            return _

if False:
    d_big_range = np.arange(3, 10)
    qudit_data = []
    for d_big in d_big_range:
        for d_small in range(2, d_big):
            D_big = qudit_wh_operators(d_big)
            D_small = qudit_wh_operators(d_small)
            magic_gap_data = avg_magic_gap(D_big, D_small, M=750, R=750)
            print("d_big: %d, d_small: %d => gap: %.5f" % (d_big, d_small, magic_gap_data["avg_magic_gap"]))
            qudit_data.append(magic_gap_data)
            with open("simple_qudit_avg_magic_gap.pkl", "wb") as f:  
                pickle.dump(qudit_data, f)

    for d_big in d_big_range:
        d_small_range = np.arange(2, d_big)
        avg_magic_gaps = []
        for d_small in d_small_range:
            data = extract(d_big, d_small, qudit_data)
            avg_magic_gaps.append(data["avg_magic_gap"])
        plt.plot(d_small_range, avg_magic_gaps, alpha=0.7, label='d_big: %d' % d_big)

    plt.legend()
    plt.title("(avg) simple qudit/qudit stabilizer subspace")
    plt.xlabel("d_small")
    plt.ylabel("linear stabilizer entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("avg_qudit_qudit_magic.png")
    plt.clf()

n_qubits_range = np.arange(2, 6)
n_qubit_data = []
for n_qubits in n_qubits_range:
    d_big = 2**n_qubits
    for d_small in range(2, d_big):
        D_big = wh_operators(*[2]*n_qubits)
        D_small = qudit_wh_operators(d_small)
        magic_gap_data = avg_magic_gap(D_big, D_small, M=750, R=750)
        print("d_big: %d, d_small: %d => gap: %.5f" % (d_big, d_small, magic_gap_data["avg_magic_gap"]))
        n_qubit_data.append(magic_gap_data)
        with open("simple_nqubit_avg_magic_gap.pkl", "wb") as f:  
            pickle.dump(n_qubit_data, f)
for n_qubits in n_qubits_range:
    d_big = 2**n_qubits
    d_small_range = np.arange(2, d_big)
    avg_magic_gaps = []
    for d_small in d_small_range:
        data = extract(d_big, d_small, n_qubit_data)
        avg_magic_gaps.append(data["avg_magic_gap"])
    plt.plot(d_small_range, avg_magic_gaps, alpha=0.7, label='d_big: %d' % d_big)

plt.legend()
plt.title("(avg) simple n-qubit/qudit stabilizer subspace")
plt.xlabel("d_small")
plt.ylabel("linear stabilizer entropy")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("avg_nqubit_qudit_magic.png")