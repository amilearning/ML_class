import sys
from tkinter.tix import Tree
from turtle import left 
import numpy as np
import os
import pickle 
from collections import Counter 
from decision_tree import DCT
from utils import *

class RFT:
    def __init__(self, Tree_max_depth=100, max_n_feats = None, max_n_threshold = 1, n_trees = 1):        
        
        self.Tree_max_depth = Tree_max_depth        
        self.tree_id = 0
        self.max_n_feats = max_n_feats
        self.max_n_threshold = max_n_threshold
        self.n_trees = n_trees
        self.trees = []        

    def train(self, X, y):        
        for _ in range(self.n_trees):
            # get random #features and #threshold 
            n_feats = np.random.np.random.randint(self.max_n_feats*0.5,self.max_n_feats)
            n_threshold = np.random.np.random.randint(self.max_n_threshold*0.5,self.max_n_threshold)            
            tree_ = DCT(Tree_max_depth=self.Tree_max_depth, n_feats = n_feats, n_threshold = n_threshold, tree_id = self.tree_id)            
            self.tree_id +=1
            sampled_X_train, sampled_y_train = sample_data(X,y,n_data=len(y)/5)
            tree_.train(sampled_X_train, sampled_y_train)
            self.trees.append(tree_)
    

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)        
        y_pred = [self.common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
    
    def common_label(self, y):
        counter = Counter(y.flatten())
        most_common = counter.most_common(1)[0][0]
        return most_common
        