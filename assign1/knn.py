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
    def __init__(self, KMean_enbled = True, k = 3, n_random_sample = None, k_mean_max_iters = None, cluster_size = 10):
        
        self.KMean_enbled = KMean_enbled
        self.k = k
        self.n_random_sample = n_random_sample 
        self.k_mean_max_iters = k_mean_max_iters
        self.cluster_size = cluster_size           
        self.clusters = [[] for _ in range(self.cluster_size)]        
        self.centroids = []
        self.cluster_y_label = []
        self.cluster_y_label_history = []
        

    def fit(self, X, y):        
        self.x_sample = X 
        self.y_sample = y                  
        # sample_idxs = np.random.choice(len(self.x_sample[:,0]), self.n_random_sample, replace=False)
        # self.x_sample = self.x_sample[sample_idxs,:]        
        # self.y_sample = np.array(self.y_sample[sample_idxs])
        
        self.n_samples, self.n_features = self.x_sample.shape
        if self.cluster_size is None:
            self.cluster_size = len(np.unique(self.y_sample))    
        
        

    def kmean_clustering(self):
        
        random_sample_idxs = np.random.choice(self.n_samples, self.cluster_size, replace=False)
        self.centroids = [self.x_sample[idx] for idx in random_sample_idxs]

        for iter_num in range(self.k_mean_max_iters):            
            self.clusters = self.create_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters)
            
            self.cluster_y_label = [self.common_label(self.y_sample[self.clusters[i]]) for i in range(len(self.clusters))]
            n_y_label = len(np.unique(self.cluster_y_label))
            self.cluster_y_label_history.append(n_y_label)
            print('Found '+ str(n_y_label)+ ' unique labled clusteres')
            if  n_y_label > 9:                
                print(str(iter_num) + 'th Clustering processing done')                
                break
            else:
                print(str(iter_num) + 'th Clustering processing')
            
        return self.get_cluster_labels(self.clusters)

    

    def create_clusters(self, centroids):
        clusters = [[] for _ in range(self.cluster_size)]
        for idx, sample in enumerate(self.x_sample):
            centroid_idx = self.closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def closest_centroid(self, sample, centroids):        
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def get_centroids(self, clusters):        
        centroids = np.zeros((self.cluster_size, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x_sample[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def get_cluster_labels(self, clusters):        
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels


    def predict(self, x):  
        
        if self.KMean_enbled:
            distances_to_clusters= [euclidean_distance(x,sample_x) for sample_x in self.centroids]
            nearest_cluster_idx = np.argsort(distances_to_clusters)[0]    
            return self.cluster_y_label[nearest_cluster_idx]
        else:
            distances= [euclidean_distance(x,sample_x) for sample_x in self.x_sample]
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
        