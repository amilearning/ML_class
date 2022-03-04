from re import S
import sys
from tkinter.tix import Tree
from turtle import left 
import numpy as np
import os
import pickle 
from collections import Counter 
from utils import *

class KNN:
    def __init__(self, k = 3, n_random_sample = None):
        self.k = k
        self.n_random_sample = n_random_sample 

    def train(self, X, y):        
        self.x_sample = X 
        self.y_sample = y
        sample_idxs = np.random.choice(self.x_sample[:,0], self.n_random_sample, replace=False)
        self.x_sample = self.x_sample[sample_idxs,:]        
        self.y_sample = np.array(self.y_sample[sample_idxs])

        
    def predict(self, x):        
        distances= [euclidiean_distance(x,sample_x) for sample_x in self.x_sample]
        k_idx = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_sample[i] for i in k_idx]        
        return self.common_label(k_nearest_labels)

    def predict_set(self,X):
        pred = [self.predict(x) for x in X]
        return np.array(pred)
        
    
    def common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
        