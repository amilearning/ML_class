import numpy as np

def compute_entropy(y):
    pb  = np.bincount(y.flatten()) / len(y)
    return -np.sum([p*np.log(p) for p in pb if p > 0])
