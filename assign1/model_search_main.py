
from distutils.log import error
import sys
import numpy as np
import os
import pickle 
from collections import Counter
from knn import KNN 
from decision_tree import TreeNode, DCT
from random_foreset import RFT
from knn import KNN
from random import randrange
from sklearn import svm, metrics
from utils import *
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))    
from sklearn.model_selection import GridSearchCV
import copy 

###################### Param setting ############################
# Decition Tree = DCT , Random Forest = RFT, SVM = SVM, kNN = KNN
algorithm = WhichAlgorithm.SVM

Training = True    
k_fold_load = False 
n_data = 10000

## Param for DCT 
n_feats = 100
n_threshold = 10
Tree_max_depth = 20
## Param for RFT
n_trees = 3
## Param for SVM
kernel = 'linear'
svm_param_grid = [
              {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
             ]

## Param for KNN
knn_param_k = 100
n_random_sample = 99
k_mean_max_iters = 1000
KMean_enbled = False



test_with_train_data = False
###################### Param setting END ########################



param_file_name = os.path.join(__location__, 'trained_weigths/best_param.pkl')
dct_file_name = 'best_weights/dct_best.pkl'
svm_file_name = 'best_weights/svm_best.pkl'
rft_file_name = 'best_weights/rft_best.pkl'
knn_file_name = 'best_weights/knn_best.pkl'

def load_data():    
    x_data_set = list()
    y_data_set = list()

    file_dir1 =  os.path.join(__location__, 'cifar10/data_batch_1')
    file_dir2 =  os.path.join(__location__, 'cifar10/data_batch_2')
    file_dir3 =  os.path.join(__location__, 'cifar10/data_batch_3')
    file_dir4 =  os.path.join(__location__, 'cifar10/data_batch_4')
    file_dir5 =  os.path.join(__location__, 'cifar10/data_batch_5')
    
    for i in range(1,6):
        file_name = 'cifar10/data_batch_'+str(i)        
        file_dir = os.path.join(__location__, file_name)    
        data_batch_ = unpickle_from_file(file_dir)
        X_train_ = data_batch_[b'data']
        y_train_ =  np.array(data_batch_[b'labels'])
        # X_train_, y_train_ = sample_data(X_train_, y_train_,10)
        x_data_set.append(X_train_)
        y_data_set.append(y_train_)
        # if i is 1:  
        #     X_train = X_train_ 
        #     y_train = y_train_
        # else:
        #     X_train = np.row_stack((X_train,X_train_))
        #     y_train = np.hstack((y_train,y_train_))                
           
    test_file_dir =  os.path.join(__location__, 'cifar10/test_batch')    
    test_data = unpickle_from_file(test_file_dir)    
    X_test =  test_data[b'data']    
    y_test =  np.array(test_data[b'labels'])

    return x_data_set, y_data_set, X_test, y_test

def k_fold_data_load(numFolds = 5):
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = sample_data(X_train, y_train, len(y_train))
    foldSize = int(len(X_train)/numFolds)
    x_data_set = list()
    y_data_set = list()
    index = randrange(len(X_train))            
    fold = X_train[index,:]
    y_fold = np.array(y_train[index])
    for _ in range(numFolds):
        while len(fold) < foldSize:
            index = randrange(len(X_train))            
            fold = np.row_stack((fold,X_train[index,:]))
            y_fold = np.hstack((y_fold,y_train[index]))
        x_data_set.append(fold)
        y_data_set.append(y_fold)

    return x_data_set, y_data_set, X_test, y_test

def param_save(model_to_save,file_name):
    with open(file_name,'wb') as outp:            
            pickle.dump(model_to_save,outp,pickle.HIGHEST_PROTOCOL)
            print("param saved = " + file_name)

def model_save(model_to_save,file_name):
    trained_file_name =  os.path.join(__location__, file_name)
    with open(trained_file_name,'wb') as outp:            
            pickle.dump(model_to_save,outp,pickle.HIGHEST_PROTOCOL)
            print("Trained model saved= " + trained_file_name)
    
if __name__ == "__main__":    
    
###################### Load Data  ############################   
    X_train_set, y_train_set, X_test, y_test = load_data()
#>>>>>>>>>>>>>>>>>>>>> Load Data END   #<<<<<<<<<<<<<<<<<<<<

    param_set_ = param_set()
    dct_prev_best_acc = 0
    rft_prev_best_acc = 0
    svm_prev_best_acc = 0
    knn_prev_best_acc = 0
    param_count = 0
    for param in param_set_.param_set_:
        param_count+=1
        print('Currently, '+ str(param_count/len(param_set_.param_set_)*100) +'%' +' of parameter set are tested')
        ############################
        algorithm = param["algorithms"]
        ############################               
        if algorithm is WhichAlgorithm.DCT:
            acc_tmp = 0
            Tree_max_depth = param["Tree_max_depths"]
            n_feats = param["n_feats"]
            n_threshold = param["n_threshold"]
            for idx, X_train_  in enumerate(X_train_set):                
                y_train_ =  y_train_set[idx]
                train_model = DCT(Tree_max_depth=Tree_max_depth,n_feats = n_feats, n_threshold = n_threshold)
                train_model.fit(X_train_, y_train_)            
                y_pred = train_model.predict(X_test)
                acc = accuracy(y_test, y_pred)
                acc_tmp +=acc
            acc_result = acc_tmp/len(X_train_set)
            if acc_result >= dct_prev_best_acc:
                dct_prev_best_acc = acc
                param_set_.dct_best_param = copy.deepcopy(param)
                param_save(param_set_,param_file_name)                
                model_save(train_model,dct_file_name)
                

        elif algorithm is WhichAlgorithm.RFT:      
            acc_tmp = 0
            Tree_max_depth = param["Tree_max_depths"]
            n_feats = param["n_feats"]
            n_threshold = param["n_threshold"]
            n_trees = param["n_trees"]
            for idx, X_train_  in enumerate(X_train_set):                
                y_train_ =  y_train_set[idx]                     
                train_model = RFT(Tree_max_depth=Tree_max_depth,max_n_feats = n_feats, max_n_threshold = n_threshold, n_trees = n_trees)            
                train_model.fit(X_train_, y_train_)            
                y_pred = train_model.predict(X_test)
                acc = accuracy(y_test, y_pred)
                acc_tmp +=acc
            acc_result = acc_tmp/len(X_train_set)
            if acc_result >= rft_prev_best_acc:
                rft_prev_best_acc = acc
                param_set_.rft_best_param = copy.deepcopy(param)
                param_save(param_set_,param_file_name)                
                model_save(train_model,rft_file_name)
                

        elif algorithm is WhichAlgorithm.SVM:
            acc_tmp = 0
            kernel = param["kernel"]
            degree = param["degree"]
            gamma = param["gamma"]
            C_param = param["C_param"]
            for idx, X_train_  in enumerate(X_train_set):                
                y_train_ =  y_train_set[idx]                   
                train_model = svm.SVC(C = C_param, kernel = kernel, degree = degree, gamma = gamma)            
                train_model.fit(X_train_, y_train_)  
                y_pred = train_model.predict(X_test)
                acc = accuracy(y_test, y_pred)
                acc_tmp +=acc
            acc_result = acc_tmp/len(X_train_set)
            if acc_result >= svm_prev_best_acc:
                svm_prev_best_acc = acc
                param_set_.svm_best_param = copy.deepcopy(param)
                param_save(param_set_,param_file_name)                
                model_save(train_model,svm_file_name)
                

        elif algorithm is WhichAlgorithm.KNN: 
            acc_tmp = 0
            KMean_enbled = param["KMean_enbled"]
            knn_param_k = param["knn_param_k"]            
            k_mean_max_iters = param["k_mean_max_iters"]
            for idx, X_train_  in enumerate(X_train_set):                
                y_train_ =  y_train_set[idx]                                         
                train_model = KNN(KMean_enbled = KMean_enbled, k = knn_param_k, n_random_sample = None, k_mean_max_iters = k_mean_max_iters)            
                train_model.fit(X_train_, y_train_) 
                if KMean_enbled:
                    cluster_label = train_model.kmean_clustering()
                y_pred = train_model.predict(X_test)
                acc = accuracy(y_test, y_pred)
                acc_tmp +=acc
            acc_result = acc_tmp/len(X_train_set)
            if acc_result >= knn_prev_best_acc:
                knn_prev_best_acc = acc
                param_set_.knn_best_param = copy.deepcopy(param)
                param_save(param_set_,param_file_name)                
                model_save(train_model,knn_file_name)

#>>>>>>>>>>>>>>>>>>>>> Train Model END  #<<<<<<<<<<<<<<<<<<<<
    
    print("Done")