import numpy as np

from .stabilizer_entropy import qudit_wh_operators

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
