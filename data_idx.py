from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.data import Data
import numpy as np
import os

def my_inc(self, key, value, *args, **kwargs):
    if key == 'subgraph_edge_index':
        return self.num_subgraph_nodes
    if key == 'subgraph_node_index':
        return self.num_nodes
    if key == 'subgraph_indicator':
        return self.num_nodes
    if 'index' in key:
        return self.num_nodes
    else:
        return 0


class SubgraphDataset(object):
    def __init__(self, dataset, k_hop=2, use_subgraph_edge_attr=False):
        
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.k_hop = k_hop
        self.use_subgraph_edge_attr = use_subgraph_edge_attr
        self.subembedding_list = None
        Data.__inc__ = my_inc
        self.extract_subgraph()
        
    def extract_subgraph(self):
        print(f"extract {self.k_hop} hops subgraph")

        self.subgraph_node_index = []
        self.subgraph_edge_index = []
        self.subgraph_indicator_index = []
        if self.use_subgraph_edge_attr :
            self.subgraph_edge_attr = []

        for i in range(len(self.dataset)):
            graph = self.dataset[i]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicator = []
            edge_start = 0
            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(node_idx, self.k_hop, graph.edge_index, True)
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index + edge_start)
                indicator.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask])
                edge_start += len(sub_nodes)
        
            self.subgraph_node_index.append(torch.cat(node_indices))
            self.subgraph_edge_index.append(torch.cat(edge_indices, dim=1))
            self.subgraph_indicator_index.append(torch.cat(indicator))
            if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                self.subgraph_edge_attr.append(torch.cat(edge_attributes))
        print("End")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        
        data = self.dataset[index]

        # if self.n_features == 1:
        #     data.x = data.x.squeeze(-1)

        # if not isinstance(data.y, list):
        #     data.y = data.y.view(-1)
        #         # data.y = data.y.view(data.y.shape[0], -1)
        # n = data.num_nodes
        data.idx = []
        data.idx.append(index)
        data.subgraph_node_index = self.subgraph_node_index[index]
        data.subgraph_edge_index = self.subgraph_edge_index[index]
        data.num_subgraph_nodes = len(self.subgraph_node_index[index])
        if self.use_subgraph_edge_attr and data.edge_attr is not None:
            data.subgraph_edge_attr = self.subgraph_edge_attr[index]
        data.subgraph_indicator = self.subgraph_indicator_index[index].type(torch.LongTensor)

        if self.subembedding_list is not None and len(self.subembedding_list) == len(self.dataset):
            data.subembedding = self.subembedding_list[index]

        return data

                
class GraphDataset(object):
    def __init__(self, dataset, nb_heads=1, degree=False):
        """a pytorch geometric dataset as input
        """
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.pe_list = None
        self.lap_pe_list = None

        self.deg_list = None
        if degree:
            self.compute_degree()
        self.nb_heads = nb_heads



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.pe_list is not None and len(self.pe_list) == len(self.dataset):
            data.pe = self.pe_list[index]
        if self.lap_pe_list is not None and len(self.lap_pe_list) == len(self.dataset):
            data.lap_pe = self.lap_pe_list[index]
        if self.deg_list is not None and len(self.deg_list) == len(self.dataset):
            data.deg = self.deg_list[index]
        return data
    
    def compute_degree(self):
        self.deg_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.deg_list.append(deg)





    def collate_fn(self):
        def collate(batch):
            batch = list(batch)
            nb_heads = self.nb_heads
            max_len = max(len(g.x) for g in batch)
            
            padded_x = torch.zeros((len(batch), max_len, self.n_features))
            adj_matrices = torch.zeros((len(batch), max_len, max_len))

            mask = torch.zeros((len(batch), max_len), dtype=bool)
            labels = []
            adj_matrices = torch.zeros((len(batch), max_len, max_len))
            # TODO: check if position encoding matrix is sparse
            # if it's the case, use a huge sparse matrix
            # else use a dense tensor
            pos_enc = None
            use_pe = hasattr(batch[0], 'pe') and batch[0].pe is not None
            if use_pe:
                if not batch[0].pe.is_sparse:
                    pos_enc = torch.zeros((len(batch), nb_heads, max_len, max_len))
                else:
                    print("Not implemented yet!")

            lap_pos_enc = None
            use_lap_pe = hasattr(batch[0], 'lap_pe') and batch[0].lap_pe is not None
            if use_lap_pe:
                lap_pe_dim = batch[0].lap_pe.shape[-1]
                lap_pos_enc = torch.zeros((len(batch), max_len, lap_pe_dim))
            
            deg = None
            use_degree = hasattr(batch[0], 'degree') and batch[0].degree is not None
            if use_degree:
                deg = torch.zeros((len(batch), max_len))


            for i, g in enumerate(batch):
                labels.append(g.y)
                g_len = len(g.x)

                padded_x[i, :g_len, :] = g.x
                adj = utils.to_dense_adj(g.edge_index, max_num_nodes=max_len)[0]
                adj_matrices[i, :g_len, :g_len] = adj[:g_len, :g_len]

                adj = utils.to_dense_adj(g.edge_index, max_num_nodes=max_len)[0]
                adj_matrices[i, :g_len, :g_len] = adj[:g_len, :g_len]

                mask[i, g_len:] = True
                if use_pe:
                    pos_enc[i, :nb_heads, :g_len, :g_len] = g.pe
                if use_lap_pe:
                    lap_pos_enc[i, :g_len, :g.lap_pe.shape[-1]] = g.lap_pe
                if use_degree:
                    deg[i, :g_len] = g.degree


        
            return padded_x, adj_matrices, mask, pos_enc, lap_pos_enc, deg, default_collate(labels)
        return collate


