"""
implementation of fast SaWL algorithm
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import numpy as np
import copy
import pdb
from data_util.pre_algorithms import preprocess_graphs


class Fast_SaWL(object):
     
    def __init__(self, path, dataset, iters):
        self.path = path
        self.dset_name = dataset
        self.iters = iters
        
                
    def compute_SaWL_psi(self, pre_data):
        """
        input: preprocessed graphs; output: feature mappings of graphs 
        
        """
        dset_psi_all_dims_iters = []
        dset_subgraph_nodes_iters = []

        dset_raw_node_neighbors, dset_subgraph_nodes, dset_id_labels, dset_max_labels = pre_data[0], pre_data[1], pre_data[2], pre_data[3]        
        dset_subgraph_nodes_iters.append(dset_subgraph_nodes)
        
        for l_dim in range(len(dset_id_labels)):     #each label dimension; 
            dset_psi_all_iters = []
            
            for iter in range(self.iters):      #each iter  
                print(f'iter_{iter}')                 
                dset_id_multiset_iter, multi_list_iter, dset_subgraph_update = self.multiset_determinate(dset_raw_node_neighbors, 
                                                                                                         dset_id_labels[l_dim], dset_subgraph_nodes_iters[iter])   
                dset_subgraph_nodes_iters.append(dset_subgraph_update)
                
                dset_id_label_iter_updated, dset_max_labels_updated = self.relabels(multi_list_iter, dset_id_labels[l_dim],
                                                                                    dset_max_labels[l_dim], dset_id_multiset_iter)

                dset_graphs_psi_iter = self.counting_psi(dset_id_label_iter_updated, dset_max_labels_updated,
                                                         dset_subgraph_nodes_iters)
            
                dset_max_labels[l_dim] = dset_max_labels_updated
                dset_id_labels[l_dim] = dset_id_label_iter_updated
                dset_psi_all_iters.append(dset_graphs_psi_iter)
            # pdb.set_trace()    
            dset_psi_all_iters = torch.cat(dset_psi_all_iters, -1)
            dset_psi_all_dims_iters.append(dset_psi_all_iters)
        
        dset_psi_all_dims_iters = torch.cat(dset_psi_all_dims_iters, -1)
        dset_max_labels = torch.tensor(dset_max_labels)
        
        return dset_psi_all_dims_iters, dset_max_labels
    
    
    def multiset_determinate(self, dset_graph_node_neighs, dset_graph_id_labels, dset_subgraph_nodes):
        """label multiset determination and subgraph identities record"""
        
        dset_graph_id_multiset = []
        multi_list_iter = []
        dset_subgraph_update = []
        
        for i in range(len(dset_graph_node_neighs)):  #each graph
            node_neigh = dset_graph_node_neighs[i]  
            subgraph_nodes = dset_subgraph_nodes[i]   # nodes in subgraphs 
            id_label = dset_graph_id_labels[i]   
            id_multiset = {}
            multi_list = []
            subgraph_update = {}
                     
            for key in node_neigh:
                tmp_ = sorted(id_label[x] for x in node_neigh[key])
                tmp = str(id_label[key]) + ',' + " ".join(str(tmp_[i]) for i in range(len(tmp_)))
                id_multiset[key] = tmp
                multi_list.append(tmp)
                
                tmp_id = []
                for item in subgraph_nodes[key]:
                    tmp_id.extend(node_neigh[item])
                subgraph_update[key] = list(set(tmp_id))   #record id 
                  
            dset_graph_id_multiset.append(id_multiset)
            multi_list_iter.extend(multi_list)
            dset_subgraph_update.append(subgraph_update)
            
        return dset_graph_id_multiset, multi_list_iter, dset_subgraph_update
    
    
    def relabels(self, multilist, id_labels,  max_label, id_multisets):
        """ relabel node's label in each graph """
        
        multi_set = sorted(set(multilist))
        compress_dict = {}
        max_label_ = max_label[-1]  
        dset_id_labels_updated = []
        
        for item in multi_set:
            compress_dict[item] = max_label_
            max_label_ += 1
        max_label.append(max_label_)
  
        for i in range(len(id_labels)):  #each graph
            id_label = id_labels[i]
            id_multiset = id_multisets[i]
            id_label_updated = {}
        
            for key in id_label.keys():
                var = id_multiset[key]
                id_label_updated[key] = compress_dict[var]
            dset_id_labels_updated.append(id_label_updated)
        
        return dset_id_labels_updated, max_label
    
    def counting_psi(self, id_label, max_label, dset_subgraph_iters):
        """ calculate full graph's feature mapping """
        
        graph_psi_all = []
        for i in range(len(id_label)): # each graph
            graph_psi = np.zeros((1, max_label[-1] - max_label[-2]), dtype=np.float32)
            for j in range(len(id_label[i])):
                tmp = len(dset_subgraph_iters[-1][i][j]) - len(dset_subgraph_iters[-2][i][j])  
                graph_psi[0][id_label[i][j] - max_label[-2]] += tmp
            graph_psi_all.append(graph_psi)
        graph_psi_all = np.concatenate(graph_psi_all, 0)
               
        return torch.from_numpy(graph_psi_all)


def load(path):       
    if path is None:
        return None
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as handle:
        attn = pickle.load(handle)
    return attn

def save(attn_enc, path, dir):
    dir = path + dir
    if path is None:
        return
    if not os.path.isfile(path):
        os.makedirs(path)
    with open(dir, 'wb') as handle:
        pickle.dump(attn_enc, handle)
     

    
if __name__ == '__main__':
    dset_name = 'Mutagenicity'        
    # dset_name = 'ogbg-molhiv'
    iters = 2
    dset_path = '../dataset/'
    sawl_path = f'../dataset/fast_sawl_pre/{dset_name}/'
    dir_psi =  'sawl_psi_iters_{}.pkl'.format(iters)
    path_max_label = 'sawl_max_label_iters_{}.pkl'.format(iters)
    
     
    if os.path.exists(sawl_path + dir_psi):
        saved_phi_list = load(sawl_path + dir_psi)
        saved_max_labels = load(sawl_path + path_max_label)
        print("Existing")
    else:
        pre_dset = preprocess_graphs(dset_name, dset_path)
        # dset_raw_node_neighbors, dset_id_labels, dset_max_labels = preprocess_graphs(dset)
        sawl_encoder = Fast_SaWL(path=sawl_path, dataset=dset_name, iters=iters)
        dset_psi, _ = sawl_encoder.compute_SaWL_psi(pre_dset)
        save(dset_psi, sawl_path, dir_psi)
        print("Saved")
    
        
    