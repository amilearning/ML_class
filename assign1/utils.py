import numpy as np

def compute_entropy(y):
    pb  = np.bincount(y)/len(y)
    return -np.sum([p*np.log2(p) for p in pb if p > 0])
