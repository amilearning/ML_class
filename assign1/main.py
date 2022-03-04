
from distutils.log import error
import sys 
import numpy as np
import os
import pickle 
from collections import Counter 
from dct import TreeNode, DCT
from random import randrange

###################### Param setting ############################
DCT_Training = False    
k_fold_load = False 
n_data = 10000
n_feats = 20
n_threshold = 5
Tree_max_depth = 20
dct_file_name = 'dct_'+ str(n_data) + '_' + str(n_feats) + '_' + str(n_threshold) + '.pkl'    
test_with_train_data = False
###################### Param setting END ########################

def unpickle_from_file(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def shrink_data(X,y,n_data=100):
    if n_data >= len(X):
        n_data = len(X)-1
    idx = np.random.choice(len(X[:,0]),n_data,replace = False)
    X = X[idx,:]
    y = y.flatten()[idx]
    return X, y

def load_data():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))    
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
    
    # from sklearn import datasets
    # from sklearn.model_selection import train_test_split



    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # data = datasets.load_breast_cancer()
    # X, y = data.data, data.target

    # X_train_, X_test_, y_train_, y_test_ = train_test_split(
    #     X, y, test_size=0.2, random_state=1234
    # )
    
###################### Load Data  ############################
    if k_fold_load:
        X_train_set, y_train_set, X_test, y_test = k_fold_data_load(numFolds = 5)
    else:
        X_train_, y_train_, X_test_, y_test_ = load_data()
    
 
    X_train_, y_train_ = shrink_data(X_train_, y_train_, n_data)
###################### Load Data END ############################
    


###################### Train Data  ############################
    if DCT_Training:
        dct_ = DCT(Tree_max_depth=Tree_max_depth,n_feats = n_feats, n_threshold = n_threshold)
        dct_.train(X_train_, y_train_)
        ## Save Data 
        
        with open(dct_file_name,'wb') as outp:            
            pickle.dump(dct_,outp,pickle.HIGHEST_PROTOCOL)
            print("Trained model saved= " + dct_file_name)
    
###################### Train Data END  ############################
    

###################### Test  ############################
    if not DCT_Training:
        print("No more Training, Loading trained model = " + dct_file_name)
        with open(dct_file_name,'rb') as inp:
            dct_ = pickle.load(inp)
    if test_with_train_data:
        y_pred = dct_.predict(X_train_)
        acc = accuracy(y_train_, y_pred)
    else:
        y_pred = dct_.predict(X_test_)
        acc = accuracy(y_test_, y_pred)
###################### Test END ############################
    



    print("Accuracy:", acc)