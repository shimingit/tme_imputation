import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import random
import os
import csv
import pandas as pd

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def load_data(dataset, time_steps, train_ratio=0.7):
    
    feat_file = os.path.join(dataset, 'MDD_RPPA_Level3_preprocessed_2020-9.xlsx')
    feat_df = pd.read_excel(feat_file, sheet_name='MDD_RPPA_Level3_annotated').set_index('Protein')
    feat_df.columns = [c.split('_')[0]+'_'+c.split('_')[1] for c in feat_df.columns]
    feat_df = feat_df.loc[:,~feat_df.columns.str.startswith('Ctrl')]
    feat_df = feat_df.apply(pd.to_numeric, errors='ignore')
    feat_df = feat_df.groupby(feat_df.columns, axis=1, sort=False).mean()
    
    feats = feat_df.values
    feats = feats.reshape([feats.shape[0],6,5]).transpose([2, 0, 1])
    feat_list = []
    for i in range(time_steps):
        feat_list.append(feats[i])
#         feat_list.append(sp.coo_matrix(feats[i]))
    
    protein_names = feat_df.index.tolist()
    
    network_file = os.path.join(dataset, 'string_interactions.tsv')
    net_df = pd.read_csv(network_file,sep='\t')
    
    adj = np.zeros([len(protein_names),len(protein_names)])
    links = []
    no_links = []
    for index, row in net_df.iterrows():
        try:
            v1_idx = protein_names.index(row['node1'])
            v2_idx = protein_names.index(row['node2'])
            score = float(row['combined_score'])
            
            adj[v1_idx,v2_idx] = score
            adj[v2_idx,v1_idx] = score
            
            links.append((row['node1'], row['node2']))
        except:
            no_links.append((row['node1'], row['node2']))
    
    adj_list = []
    for i in range(time_steps):
        adj_list.append(sp.coo_matrix(adj))
    
#     print('num_links:', len(links), 'num_no_links:', len(no_links), adj)
    
    # split the train/val/test sets
    train_idx = np.array(random.sample(range(len(protein_names)), int(len(protein_names)*train_ratio))) 
    temp_indexes = np.setdiff1d(np.array(range(len(protein_names))), train_idx)
    val_idx = np.array(random.sample(list(temp_indexes), int(len(temp_indexes)*0.2)))
    test_idx = np.setdiff1d(temp_indexes, val_idx)
    
    return adj_list, feat_list, train_idx, val_idx, test_idx

load_data('datasets/rppa', 2)

def build_train_samples_imputation(embeds_list, feats, train_idx, val_idx, 
                                    test_idx, time_steps):
    points = []
    for i in range(time_steps-1): 
        points.append(embeds_list[i])
    point_seq = tf.concat([tf.expand_dims(p, 1) for p in points], 1)
    
    ps_x_train = tf.gather(point_seq, train_idx)
    ps_y_train = tf.gather(feats[time_steps-1], train_idx)
    ps_x_val = tf.gather(point_seq, val_idx)
    ps_y_val = tf.gather(feats[time_steps-1], val_idx)
    ps_x_test = tf.gather(point_seq, test_idx)
    ps_y_test = tf.gather(feats[time_steps-1], test_idx)
    
    return ps_x_train, ps_y_train, ps_x_val, ps_y_val, ps_x_test, ps_y_test
    
def const_train_samples(embeds_list, feats, train_idx, val_idx, test_idx,
                        time_steps, window_size, base_mat=None):
    if base_mat == None:
        base_mat = tf.zeros_like(embeds_list[0])
    
    ps_x_trains = []
    ps_y_trains = []
    ps_x_vals = []
    ps_y_vals = []
    ps_x_tests = []
    ps_y_tests = []
    for i in range(1, time_steps):
        points = []
        for j in range(window_size):
            s = i - window_size + j
            if s < 0:
                points.append(base_mat)
            else:
                points.append(embeds_list[j])
                
        point_seq = tf.concat([tf.expand_dims(p, 1) for p in points], 1)
        
        ps_x_trains.append(tf.gather(point_seq, train_idx[i]))
        ps_y_trains.append(tf.gather(feats[i], train_idx[i]))
        
        if len(test_idx[i]) != 0 & len(val_idx[i]) != 0:
            ps_x_vals.append(tf.gather(point_seq, val_idx[i]))
            ps_y_vals.append(tf.gather(feats, val_idx[i]))
            ps_x_tests.append(tf.gather(point_seq, test_idx[i]))
            ps_y_tests.append(tf.gather(feats, test_idx[i]))
        
    ps_x_train = tf.concat(ps_x_trains, 1)   
    ps_y_train = tf.concat(ps_y_trains, 1)   
    ps_x_val = tf.concat(ps_x_vals, 1)   
    ps_y_val = tf.concat(ps_y_vals, 1)   
    ps_x_test = tf.concat(ps_x_tests, 1)   
    ps_y_test = tf.concat(ps_y_tests, 1)   
        
    return ps_x_train, ps_y_train, ps_x_val, ps_y_val, ps_x_test, ps_y_test