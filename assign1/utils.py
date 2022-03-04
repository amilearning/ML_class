import numpy as np

class WhichAlgorithm(Enum):
     # Decition Tree = 0, Random Forest = 1, SVM = 2, kNN = 3
     DCT = 0
     RFT = 1
     SVM = 2
     KNN = 3

def compute_entropy(y):
    pb  = np.bincount(y.flatten()) / len(y)
    return -np.sum([p*np.log(p) for p in pb if p > 0])

def unpickle_from_file(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def sample_data(X,y,n_data=100):
    if n_data >= len(X):
        n_data = len(X)-1
    idx = np.random.choice(len(X[:,0]),n_data,replace = False)
    X = X[idx,:]
    y = y.flatten()[idx]
    return X, y