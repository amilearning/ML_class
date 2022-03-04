
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
from sklearn import svm
from utils import *
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))    


###################### Param setting ############################
# Decition Tree = DCT , Random Forest = RFT, SVM = SVM, kNN = KNN
algorithm = WhichAlgorithm.SVM

Training = True    
k_fold_load = False 
n_data = 100

## Param for DCT 
n_feats = 100
n_threshold = 10
Tree_max_depth = 20
## Param for RFT
n_trees = 3
## Param for SVM

## Param for KNN
knn_param_k = 100
n_random_sample = 99
k_mean_max_iters = 1000
KMean_enbled = False


test_with_train_data = True
###################### Param setting END ########################


if algorithm is WhichAlgorithm.DCT:
    trained_file_name = 'trained_weigths/dct_'+ str(n_data) + '_' + str(n_feats) + '_' + str(n_threshold) + '.pkl'    
elif algorithm is WhichAlgorithm.RFT:
    trained_file_name = 'trained_weigths/rft_'+ str(n_data) + '_' + str(n_feats) + '_' + str(n_threshold) + '.pkl'    
elif algorithm is WhichAlgorithm.SVM:
    trained_file_name = 'trained_weigths/svm_'+ str(n_data) + '_' + str(n_feats) + '_' + str(n_threshold) + '.pkl'    
elif algorithm is WhichAlgorithm.KNN:
    trained_file_name = 'trained_weigths/knn_'+ str(n_data) + '_' + str(n_feats) + '_' + str(n_threshold) + '.pkl'    
else:
    print("Pls select the algorithm to run")

trained_file_name =  os.path.join(__location__, trained_file_name)



def load_data():
    
    file_dir1 =  os.path.join(__location__, 'cifar10/data_batch_1')
    file_dir2 =  os.path.join(__location__, 'cifar10/data_batch_2')
    file_dir3 =  os.path.join(__location__, 'cifar10/data_batch_3')
    file_dir4 =  os.path.join(__location__, 'cifar10/data_batch_4')
    file_dir5 =  os.path.join(__location__, 'cifar10/data_batch_5')
    
    for i in range(1,2):
        file_name = 'cifar10/data_batch_'+str(i)        
        file_dir = os.path.join(__location__, file_name)    
        data_batch_ = unpickle_from_file(file_dir)
        X_train_ = data_batch_[b'data']
        y_train_ =  np.array(data_batch_[b'labels'])
        if i is 1:  
            X_train = X_train_ 
            y_train = y_train_
        else:
            X_train = np.row_stack((X_train,X_train_))
            y_train = np.hstack((y_train,y_train_))
                
           
    test_file_dir =  os.path.join(__location__, 'cifar10/test_batch')    
    test_data = unpickle_from_file(test_file_dir)    
    X_test =  test_data[b'data']    
    y_test =  np.array(test_data[b'labels'])

    

    return X_train, y_train, X_test, y_test

def k_fold_data_load(numFolds = 2):
    X_train, y_train, X_test, y_test = load_data()
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


    
if __name__ == "__main__":    
    
    
###################### Load Data  ############################
    if k_fold_load:
        X_train_set, y_train_set, X_test, y_test = k_fold_data_load(numFolds = 5)
    else:
        X_train_, y_train_, X_test_, y_test_ = load_data()
    
 
    X_train_, y_train_ = sample_data(X_train_, y_train_, n_data)
#>>>>>>>>>>>>>>>>>>>>> Load Data END   #<<<<<<<<<<<<<<<<<<<<


###################### Train Model  ############################
    if Training:
        if algorithm is WhichAlgorithm.DCT:
            train_model = DCT(Tree_max_depth=Tree_max_depth,n_feats = n_feats, n_threshold = n_threshold)
            train_model.train(X_train_, y_train_)            
        elif algorithm is WhichAlgorithm.RFT:            
            train_model = RFT(Tree_max_depth=Tree_max_depth,max_n_feats = n_feats, max_n_threshold = n_threshold, n_trees = 3)            
        elif algorithm is WhichAlgorithm.SVM:
            print("SVM is not available yet")
        elif algorithm is WhichAlgorithm.KNN:                        
            train_model = KNN(KMean_enbled = KMean_enbled, k = knn_param_k,n_random_sample = n_random_sample, k_mean_max_iters = k_mean_max_iters)            
            
        
        train_model.train(X_train_, y_train_)
        if algorithm is WhichAlgorithm.KNN and KMean_enbled:
            cluster_label = train_model.kmean_clustering()
        
        with open(trained_file_name,'wb') as outp:            
            pickle.dump(train_model,outp,pickle.HIGHEST_PROTOCOL)
            print("Trained model saved= " + trained_file_name)
    
#>>>>>>>>>>>>>>>>>>>>> Train Model END  #<<<<<<<<<<<<<<<<<<<<
    

###################### Test Model ############################
    if not Training:
        print("No more Training, Loading trained model = " + trained_file_name)
        try:
            with open(trained_file_name,'rb') as inp:
                train_model = pickle.load(inp)
        except EnvironmentError:
            print ('Trained model NOT found!!, you can start train model instead by setting (Training = True)')
            sys.exit()

    if test_with_train_data:
        y_pred = train_model.predict(X_train_)
        acc = accuracy(y_train_, y_pred)
    else:
        y_pred = train_model.predict(X_test_)
        acc = accuracy(y_test_, y_pred)

#>>>>>>>>>>>>>>>>>>>>> Test Model END  #<<<<<<<<<<<<<<<<<<<<
    



    print("Accuracy:", acc)