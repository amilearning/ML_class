
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
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

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
        X_train_, y_train_ = sample_data(X_train_, y_train_,1000)
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

def k_rfold_data_load(X_train_set_,y_train_set_, numFolds = 5):
    foldSize = int(len(X_train_set_)) - int(len(X_train_set_)/numFolds)
    X_train_set=[]
    y_train_set=[]  
    for _ in range(numFolds):
        X_train, y_train = sample_data(X_train_set_, y_train_set_, foldSize)
        X_train_set.append(X_train)
        y_train_set.append(y_train)
    return X_train_set, y_train_set

def param_save(model_to_save,file_name):
    with open(file_name,'wb') as outp:            
            pickle.dump(model_to_save,outp,pickle.HIGHEST_PROTOCOL)
            print("param saved = " + file_name)

def model_save(model_to_save,file_name):
    trained_file_name =  os.path.join(__location__, file_name)
    with open(trained_file_name,'wb') as outp:            
            pickle.dump(model_to_save,outp,pickle.HIGHEST_PROTOCOL)
            print("Trained model saved= " + trained_file_name)

def merge_data(X_train_set_, y_train_set_):
    X_train_set = np.vstack(X_train_set_)
    y_train_set = np.hstack(y_train_set_)
    return X_train_set, y_train_set

if __name__ == "__main__":    
    
###################### Load Data  ############################   
    X_train_set_, y_train_set_, X_test, y_test = load_data()   
    X_train_set, y_train_set = merge_data(X_train_set_, y_train_set_)
    X_train_set, y_train_set = k_rfold_data_load(X_train_set, y_train_set)
    print("Succefully load Data")
#>>>>>>>>>>>>>>>>>>>>> Load Data END   #<<<<<<<<<<<<<<<<<<<<

    param_set_ = param_set()
    dct_prev_best_acc = 0
    rft_prev_best_acc = 0
    svm_prev_best_acc = 0
    knn_prev_best_acc = 0
    param_count = 0
    acc_result = []
    train_acc_result = []

    acc_plot_x = []
    acc_plot_y = []
    acc_plot_z = []
    acc_plot_c = []
    acc_plot_train_c = []

    for param in param_set_.param_set_:
        param_count+=1
        print('Currently, '+ str(math.trunc(param_count/len(param_set_.param_set_)*100)) +'%' +' of parameter set are tested')
        ############################
        algorithm = param["algorithms"]
        ############################               
        if algorithm is WhichAlgorithm.DCT:
            acc_tmp = 0
            train_acc_tmp = 0
            Tree_max_depth = param["Tree_max_depths"]            
            n_feats = param["n_feats"]            
            n_threshold = param["n_threshold"]
            acc_plot_x.append(Tree_max_depth)
            acc_plot_y.append(n_feats)
            acc_plot_z.append(n_threshold)            
            for idx, X_train_  in enumerate(X_train_set):     
                y_train_ =  y_train_set[idx]
                train_model = DCT(Tree_max_depth=Tree_max_depth,n_feats = n_feats, n_threshold = n_threshold)
                train_model.fit(X_train_, y_train_)            
                ## Training data accuracy
                y_train_pred = train_model.predict(X_train_)
                train_acc = accuracy(y_train_pred, y_train_)
                ## Test data accuracy
                y_pred = train_model.predict(X_test)
                acc = accuracy(y_test, y_pred)
                acc_tmp +=acc
                train_acc_tmp +=train_acc
            acc_result = acc_tmp/len(X_train_set)
            train_acc_result = train_acc_tmp/len(X_train_set)
            acc_plot_c.append(acc_result)
            acc_plot_train_c.append(train_acc_result)
            
            if acc_result >= dct_prev_best_acc:
                dct_prev_best_acc = acc
                param_set_.dct_best_param = copy.deepcopy(param)
                param_save(param_set_,param_file_name)                
                model_save(train_model,dct_file_name)
                

        elif algorithm is WhichAlgorithm.RFT:      
            acc_tmp = 0
            train_acc_tmp = 0
            Tree_max_depth = param["Tree_max_depths"]
            n_feats = param["n_feats"]            
            n_trees = param["n_trees"]
            n_threshold = param['n_threshold']
            acc_plot_x.append(Tree_max_depth)
            acc_plot_y.append(n_feats)
            acc_plot_z.append(n_trees)  

            for idx, X_train_  in enumerate(X_train_set):                
                y_train_ =  y_train_set[idx]                     
                train_model = RFT(Tree_max_depth=Tree_max_depth,max_n_feats = n_feats, max_n_threshold = n_threshold, n_trees = n_trees)            
                train_model.fit(X_train_, y_train_)                            
                ## Training data accuracy
                y_train_pred = train_model.predict(X_train_)
                train_acc = accuracy(y_train_pred, y_train_)
                ## Test data accuracy
                y_pred = train_model.predict(X_test)
                acc = accuracy(y_test, y_pred)
                acc_tmp +=acc
                train_acc_tmp +=train_acc
            acc_result = acc_tmp/len(X_train_set)
            train_acc_result = train_acc_tmp/len(X_train_set)
            acc_plot_c.append(acc_result)
            acc_plot_train_c.append(train_acc_result)


            if acc_result >= rft_prev_best_acc:
                rft_prev_best_acc = acc
                param_set_.rft_best_param = copy.deepcopy(param)
                param_save(param_set_,param_file_name)                
                model_save(train_model,rft_file_name)
                

        elif algorithm is WhichAlgorithm.SVM:
            acc_tmp = 0
            train_acc_tmp = 0
            kernel = param["kernel"]
            degree = param["degree"]
            gamma = param["gamma"]
            C_param = param["C_param"]
            
            acc_plot_x.append(degree)
            acc_plot_y.append(gamma)
            acc_plot_z.append(C_param)  

            for idx, X_train_  in enumerate(X_train_set):                
                y_train_ =  y_train_set[idx]                   
                train_model = svm.SVC(C = C_param, kernel = kernel, degree = degree, gamma = gamma)            
                train_model.fit(X_train_, y_train_)                  
                ## Training data accuracy
                y_train_pred = train_model.predict(X_train_)
                train_acc = accuracy(y_train_pred, y_train_)
                ## Test data accuracy
                y_pred = train_model.predict(X_test)
                acc = accuracy(y_test, y_pred)
                acc_tmp +=acc
                train_acc_tmp +=train_acc
            acc_result = acc_tmp/len(X_train_set)
            train_acc_result = train_acc_tmp/len(X_train_set)
            acc_plot_c.append(acc_result)
            acc_plot_train_c.append(train_acc_result)


            if acc_result >= svm_prev_best_acc:
                svm_prev_best_acc = acc
                param_set_.svm_best_param = copy.deepcopy(param)
                param_save(param_set_,param_file_name)                
                model_save(train_model,svm_file_name)
                

        elif algorithm is WhichAlgorithm.KNN: 
            acc_tmp = 0
            train_acc_tmp = 0
            KMean_enbled = param["KMean_enbled"]
            knn_param_k = param["knn_param_k"]            
            k_mean_max_iters = param["k_mean_max_iters"]
            for idx, X_train_  in enumerate(X_train_set):                
                y_train_ =  y_train_set[idx]                                         
                train_model = KNN(KMean_enbled = KMean_enbled, k = knn_param_k, n_random_sample = None, k_mean_max_iters = k_mean_max_iters)            
                train_model.fit(X_train_, y_train_) 
                if KMean_enbled:
                    cluster_label = train_model.kmean_clustering()                            
                ## Training data accuracy
                y_train_pred = train_model.predict(X_train_)
                train_acc = accuracy(y_train_pred, y_train_)
                ## Test data accuracy
                y_pred = train_model.predict(X_test)
                acc = accuracy(y_test, y_pred)
                acc_tmp +=acc
                train_acc_tmp +=train_acc
            acc_result = acc_tmp/len(X_train_set)
            train_acc_result = train_acc_tmp/len(X_train_set)
            acc_plot_c.append(acc_result)
            acc_plot_train_c.append(train_acc_result)

            if acc_result >= knn_prev_best_acc:
                knn_prev_best_acc = acc
                param_set_.knn_best_param = copy.deepcopy(param)
                param_save(param_set_,param_file_name)                
                model_save(train_model,knn_file_name)

#>>>>>>>>>>>>>>>>>>>>> Train Model END  #<<<<<<<<<<<<<<<<<<<<

#            acc_plot_x.append(degree)
#             acc_plot_y.append(gamma)
#             acc_plot_z.append(C_param)  

#>>>>>>>>>>>>>>>>>>>>> PLOT START  #<<<<<<<<<<<<<<<<<<<<
    print("train acc" )
    print(acc_plot_train_c)
    print("test acc" )
    print(acc_plot_c)           

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')


    ax.set_xlabel('degree')
    ax.set_ylabel('gamma')
    ax.set_zlabel('C')
    img = ax.scatter(acc_plot_x, acc_plot_y, acc_plot_z, c=acc_plot_c)    
    fig_colorbar = fig.colorbar(img)    
    fig_colorbar.set_label('Test Accuracy')

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_xlabel('degree')
    ax2.set_ylabel('gamma')
    ax2.set_zlabel('C')
    img2 = ax2.scatter(acc_plot_x, acc_plot_y, acc_plot_z, c=acc_plot_train_c)
    fig2_colorbar = fig2.colorbar(img2)    
    fig2_colorbar.set_label('Train Accuracy')
    
    plt.show()

#>>>>>>>>>>>>>>>>>>>>> PLOT END  #<<<<<<<<<<<<<<<<<<<<



    print("Done")
    