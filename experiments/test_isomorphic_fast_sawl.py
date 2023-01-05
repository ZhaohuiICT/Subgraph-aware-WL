""" 
Test if two graphs are isomorphic
"""

import torch 
import pdb
from torch_geometric.data import Data


class Fast_SaWL(object):
     
    def __init__(self, max_iters):
        self.max_iters = max_iters
        
                
    def test_isomorphic(self, dset):
        """
        input: list of two graphs; 
        output: isomorphic or not 
        
        """
        dset_raw_node_neighbors, dset_subgraph_nodes, dset_id_labels, dset_max_labels = self.preprocess_graphs(dset)        
        dset_subgraph_nodes_iters = []
        dset_subgraph_nodes_iters.append(dset_subgraph_nodes)
    
            
        for iter in range(self.max_iters):      #each iter  
              
            dset_id_multiset_iter, multi_list_iter, dset_subgraph_update = self.multiset_determinate(dset_raw_node_neighbors, dset_id_labels, dset_subgraph_nodes_iters[iter])   
            dset_subgraph_nodes_iters.append(dset_subgraph_update)
            
            dset_id_label_iter_updated = self.relabels(multi_list_iter, dset_id_labels, dset_max_labels, dset_id_multiset_iter)

            graphs_list = self.decide_isomorphic(dset_id_label_iter_updated, dset_subgraph_nodes_iters[iter])    
            
            if graphs_list[0] != graphs_list[1]:
                print(f'Graph G and H are determined non-isomorphic in {iter+1}-th iter')
                print(f'Terminating condition: {graphs_list[0]} != {graphs_list[1]}')
                break
            
            else:
                print(f'Graph G and H cannot be decided non-isomorphic in {iter+1}-th iter')  
        
        
    
    def preprocess_graphs(self, dset):
        """preprocess graphs in the list"""
        
        dset_graph_node_neighbors = []
        dset_graph_subgraph_nodes = []
        dset_graph_node_id_labels = []
        dset_graph_max_labels = [0]
            
        for i, graph in enumerate(dset):

            graph_node_neigh, graph_subgraph_nodes, graph_id_label, max_label = self.pre_graph(graph)
            dset_graph_node_neighbors.append(graph_node_neigh)
            dset_graph_subgraph_nodes.append(graph_subgraph_nodes)
        
            dset_graph_node_id_labels.append(graph_id_label)
            if max_label > dset_graph_max_labels[0]:
                dset_graph_max_labels[0] = max_label
                
        return dset_graph_node_neighbors, dset_graph_subgraph_nodes, dset_graph_node_id_labels, dset_graph_max_labels
                
    def pre_graph(self, graph):
        """prepare neighbors of each node in the given graph"""
        node_neigh = {}
        subgraph_nodes = {}
        row = graph.edge_index[0].detach().tolist()
        col = graph.edge_index[1].detach().tolist()
        for i in range(len(row)):
            var = row[i]
            if var not in node_neigh.keys():
                node_neigh[var] = []
                subgraph_nodes[var] =[var]
            node_neigh[var].append(col[i])
            subgraph_nodes[var].append(col[i])
        
        id_label = {}
        for id in range(len(graph.x)):
            id_label[id] = graph.x[id].item()
        
        return node_neigh, subgraph_nodes, id_label, max(id_label.values())  
    
    def multiset_determinate(self, dset_graph_node_neighs, dset_graph_id_labels, dset_subgraph_nodes):
        """label multiset determination and subgraph identities record"""
        
        dset_graph_id_multiset = []
        multi_list_iter = []
        dset_subgraph_update = []
        
        for i in range(len(dset_graph_node_neighs)):  #each graph
            node_neigh = dset_graph_node_neighs[i]  #{}
            subgraph_nodes = dset_subgraph_nodes[i]   # nodes in subgraphs 
            id_label = dset_graph_id_labels[i]   #{}
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
        max_label_ = max_label[-1]  #check max_label
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

        return dset_id_labels_updated
    
    def decide_isomorphic(self, id_label, dset_subgraph):
        graphs_list = []
            
        for i in range(len(id_label)):
            graph_set = set()
            for j in range(len(id_label[0])):
                tmp = (id_label[i][j], len(dset_subgraph[i][j]))
                graph_set.add(tmp)
            graphs_list.append(graph_set)
            # pdb.set_trace()
        return graphs_list
        

    
if __name__ == '__main__':
    
    max_iters = 3
    graph_G = Data(x=torch.tensor([0,0,1,1,0,0]), edge_index=torch.tensor([[0,0,1,1,2,2,2,3,3,3,4,4,5,5],
                                                                           [1,2,0,3,0,3,4,1,2,5,2,5,3,4]]))
    graph_H = Data(x=torch.tensor([0,0,1,1,0,0]),edge_index=torch.tensor([[0,0,1,1,2,2,2,3,3,3,4,4,5,5],
                                                                          [1,2,0,2,0,1,3,2,4,5,3,5,3,4]]))
    dset = [graph_G, graph_H]
    sawl_encoder = Fast_SaWL(max_iters=max_iters)
    sawl_encoder.test_isomorphic(dset)
    
    
    
    

