"""
data preprocessing for fast SaWL algorithm
"""
from torch_geometric import datasets
from ogb.graphproppred import PygGraphPropPredDataset
import pdb

TU = ['NCI1', 'NCI109', 'Mutagenicity', 'MUTAG']
dset_onehot = ['NCI1', 'NCI109', 'Mutagenicity', 'MUTAG']

def preprocess_graphs(dset_name, dset_path):
    """preprocess all graphs in the dataset"""
    if 'ogb' in dset_name:
        dset = PygGraphPropPredDataset(name=dset_name, root=dset_path)
        dim = dset[0].x.shape[-1]
    
    if dset_name in TU:
        dset = datasets.TUDataset(dset_path, dset_name)
        dim = 1
    
    dset_graph_node_neighbors = []
    dset_graph_subgraph_nodes = []
    dset_graph_node_id_labels = [[] for _ in range(dim)]
    dset_graph_max_labels = [[0] for _ in range(dim)]
        
    for i, graph in enumerate(dset):
        graph_node_neigh, graph_subgraph_nodes = pre_graph_node_neighbor(graph)
        dset_graph_node_neighbors.append(graph_node_neigh)
        dset_graph_subgraph_nodes.append(graph_subgraph_nodes)
        # pdb.set_trace()
    
        for jth in range(dim):
            graph_node_id_label_jth, max_label_jth = pre_graph_id_label(dset_name, graph, jth)
            dset_graph_node_id_labels[jth].append(graph_node_id_label_jth)

            if max_label_jth > dset_graph_max_labels[jth][0]:
                dset_graph_max_labels[jth][0] = max_label_jth
            
    return dset_graph_node_neighbors, dset_graph_subgraph_nodes, dset_graph_node_id_labels, dset_graph_max_labels
            
def pre_graph_node_neighbor(graph):
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
        
    for j in range(len(graph.x)):
        if j not in node_neigh.keys():  # neighbors of isolated node is isolated node itself
            node_neigh[j] = [j]
            subgraph_nodes[j] = [j]
        else:
            subgraph_nodes[j].extend(node_neigh[j])
    
    return node_neigh, subgraph_nodes

def pre_graph_id_label(dset_name, graph, jth):
    id_label = {}
        
    for id in range(len(graph.x)):
        if dset_name in dset_onehot:
            id_label[id] =  graph.x[id].argmax().item()
        else:   #ogb
            id_label[id] = graph.x[id][jth].item()
    return id_label, max(id_label.values())

