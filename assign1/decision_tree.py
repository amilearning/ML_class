import sys
import numpy as np
import os
import pickle 
from collections import Counter 
from utils import compute_entropy


class TreeNode:
    def __init__(
        self, depth = None, Node_id = None, feature=None, threshold=None, left=None, right=None, label=None
    ):
        self.Node_id = Node_id
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
        self.depth = depth
        self.print_node()

    def check_leaf(self):
        return self.label is not None
    def print_node(self):
        msg_ = self.depth*'---|' + 'ID = ' + str(self.Node_id) 
        if self.check_leaf():
            msg_= msg_+', label = ' + str(self.label)
        else:
            msg_ = msg_+', feature = ' + str(self.feature) + ', threshold = '+ str(self.threshold)
        print(msg_)
    

class DCT:
    def __init__(self, Tree_max_depth=100, n_feats = None, n_threshold = 1, tree_id = 0):        
        
        self.Tree_max_depth = Tree_max_depth
        self.feature_dim = None
        self.root_node = None
        self.Node_id = 0
        self.n_feats = n_feats
        self.n_threshold = n_threshold

    def fit(self, X, y):
        self.feature_dim =  X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root_node = self.build_tree(X, y)
    
    def is_leaf(self,tree_depth = None, n_samples = None, n_labels = None):
        if (tree_depth >= self.Tree_max_depth or n_labels < 2  or n_samples < 2):
            return True
        return False

    def build_tree(self, X, y, tree_depth=0):        
        n_samples, node_feature_dim = X.shape     
        n_labels = len(np.unique(y))   
        
        if (self.is_leaf(tree_depth,n_samples,n_labels)):            
            leaf_value = self.common_label(y)
            self.Node_id +=1            
            return TreeNode(depth = tree_depth, Node_id = self.Node_id, label=leaf_value)
        
        # Continue building
        feat_idxs = np.random.choice(node_feature_dim, self.feature_dim, replace=False)
        best_feat, best_thresh = self.entropy_based_criteria(X, y, feat_idxs)
        
        left_idxs = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
        right_idxs = np.argwhere(X[:, best_feat] > best_thresh).flatten()
        
        if len(left_idxs) ==0 or len(right_idxs) ==0:
            leaf_value = self.common_label(y)
            self.Node_id +=1            
            return TreeNode(depth = tree_depth, Node_id = self.Node_id, label=leaf_value)
    
        left = self.build_tree(X[left_idxs, :], y[left_idxs], tree_depth + 1)
        right = self.build_tree(X[right_idxs, :], y[right_idxs], tree_depth + 1)
        
        self.Node_id +=1        
        return TreeNode(tree_depth, self.Node_id, best_feat, best_thresh, left, right)

   


    def entropy_based_criteria(self, X, y, feat_idxs):
        max_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            threshold_candidates = np.unique(X_column)
            if len(threshold_candidates) >= self.n_threshold:
                threshold_candidates = np.random.choice(threshold_candidates,self.n_threshold,replace = False)                         
            for threshold in threshold_candidates:
                gain = self.information_gain(y, X_column, threshold)
                if gain > max_gain:
                    max_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def information_gain(self, y, X_column, split_thresh):        
        
        parent_entropy = compute_entropy(y)
        
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = compute_entropy(y[left_idxs]), compute_entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        # Entropy of parent - weighted average of Entropy of childrens
        ig = parent_entropy - child_entropy

        
        return ig

    

    def traverse_tree(self, x, node):
        if node.check_leaf():
            return node.label

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def common_label(self, y):
        counter = Counter(y.flatten())
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root_node) for x in X])