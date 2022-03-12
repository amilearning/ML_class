import numpy as np
from enum import Enum
import copy
     
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
    idx = np.random.choice(np.arange(0,len(X[:,0])),int(n_data),replace = False)
    X = X[idx,:]
    y = y.flatten()[idx]
    return X, y

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

def euclidean_distance(x,y):    
    return np.sqrt(np.sum(x-y)**2)


class param_set:
    def __init__(self):
        self.dct_best_param = None 
        self.rft_best_param = None
        self.svm_best_param = None 
        self.knn_best_param = None
    
        self.param_set_ =[]
        
        algorithms      = [WhichAlgorithm.DCT, WhichAlgorithm.RFT, WhichAlgorithm.SVM, WhichAlgorithm.KNN]        
        
      
        
        Tree_max_depths = np.array([2, 4,  10])
        n_feats         = np.array([5, 10, 100])
        n_threshold     = np.array([5, 10, 20])
        # n_threshold     = np.array([10])
        n_trees         = np.array([1, 2, 5])


        kernel          = ['linear', 'poly', 'rbf']        
        degree          = np.array([1])
        gamma           = np.array([ 1])
        C_param         = np.array([0.1, 10])
        
        KMean_enbled    = ['True', 'False']        
        knn_param_k     = np.array([10,20])
        k_mean_max_iters= np.array([20,100])


        param_default = {}
        param_default["algorithms"]         = algorithms[0]
        param_default["Tree_max_depths"]    = Tree_max_depths[0]
        param_default["n_feats"]            = n_feats[0]
        param_default["n_threshold"]        = n_threshold[0]
        param_default["n_trees"]            = n_trees[0]
        param_default["kernel"]             = kernel[0]
        param_default["degree"]             = degree[0]
        param_default["gamma"]              = gamma[0]
        param_default["C_param"]            = C_param[0]        
        param_default["KMean_enbled"]       = KMean_enbled[0]
        param_default["knn_param_k"]        = knn_param_k[0]
        param_default["k_mean_max_iters"]   = k_mean_max_iters[0]
        
        ##################### For DCT ###################        
        for Tree_max_depth in Tree_max_depths:
            for n_feat in n_feats:
                for threshold in n_threshold:                    
                    param = copy.deepcopy(param_default)
                    param["algorithms"]      = algorithms[0]
                    param["Tree_max_depths"] = Tree_max_depth
                    param["n_feats"]         = n_feat
                    param["n_threshold"]     = threshold
                    self.param_set_.append(param)
                       

        # ##################### For RFT ###################        
        for Tree_max_depth in Tree_max_depths:
            for n_feat in n_feats:
                for threshold in n_threshold:                    
                    for n_tree in n_trees:    
                        param = copy.deepcopy(param_default)
                        param["algorithms"]         = algorithms[1]
                        param["Tree_max_depths"]    = Tree_max_depth
                        param["n_feats"]            = n_feat
                        param["n_threshold"]        = threshold
                        param["n_trees"]            = n_tree
                        self.param_set_.append(param)


        # ##################### For SVM ###################                
        for kernel_ in kernel:
            for degree_ in degree:                    
                for gamma_ in gamma:    
                    for C_param_ in C_param:    
                        param = copy.deepcopy(param_default)
                        param["algorithms"]  = algorithms[2]                        
                        param["kernel"]      = kernel_
                        param["degree"]      = degree_
                        param["gamma"]       = gamma_
                        param["C_param"]     = C_param_
                        self.param_set_.append(param)
                   
               
        ##################### For KNN ###################                
        for KMean_enbled_ in KMean_enbled:
            for knn_param_k_ in knn_param_k:                    
                for k_mean_max_iters_ in k_mean_max_iters:                        
                    param = copy.deepcopy(param_default)
                    param["algorithms"]         = algorithms[3] 
                    param["KMean_enbled"]       = KMean_enbled_
                    param["knn_param_k"]        = knn_param_k_
                    param["k_mean_max_iters"]   = k_mean_max_iters_
                    self.param_set_.append(param)
        
        self.algorithms = algorithms      
        self.Tree_max_depths = Tree_max_depths 
        self.n_feats = n_feats         
        self.n_threshold = n_threshold     
        self.n_trees = n_trees         
        self.kernel = kernel          
        self.degree = degree          
        self.gamma = gamma           
        self.C_param = C_param         
        self.KMean_enbled = KMean_enbled         
        self.k_mean_max_iters = k_mean_max_iters

        