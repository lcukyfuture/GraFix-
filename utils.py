from WL_gpu import WL
import torch
import pickle
import os
import math
import numpy as np
from torch_geometric import utils
from torch_geometric.data import Data
from grakel import graph_from_networkx
from grakel.kernels import ShortestPath, WeisfeilerLehman, RandomWalkLabeled, GraphletSampling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_kernel_CPU(dataset, data, kernel_type, num_hops, WL_k, GL_k):
    print(kernel_type)
    if kernel_type == 'SP':
        gk = ShortestPath(n_jobs = 8, normalize=True, with_labels = True)
    elif kernel_type == 'RW':
        gk = RandomWalkLabeled(n_jobs=32,normalize=True, method_type="fast", kernel_type="geometric", p=2)
    elif kernel_type == 'WL':
        gk = WeisfeilerLehman(n_jobs = 32, normalize=True, n_iter=WL_k)
    elif kernel_type == 'WLSP':
        gk = WeisfeilerLehman(n_jobs = 32, normalize=True, base_graph_kernel=ShortestPath)
    elif kernel_type == 'GL':
        gk = GraphletSampling(n_jobs=32, normalize=True, k=GL_k)
    if dataset == 'AIDS':
        edge_index, _ = utils.add_self_loops(data.edge_index, num_nodes = data.num_nodes)
    else:
        edge_index = data.edge_index
    nodes_indices = 0
    subgraph_networkx=[]

    for node_index in range(nodes_indices + data.num_nodes):
        sub_node, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(node_idx=node_index, 
                                                                    num_hops=num_hops, 
                                                                    edge_index=edge_index, 
                                                                    relabel_nodes=True)
        if sub_node.size(0) == 1:
            self_loop = torch.tensor([[0], [0]], dtype=torch.long)
            sub_edge_index = torch.cat([sub_edge_index, self_loop], dim=1)
        x = data.x[sub_node]
        label = np.argmax(x, axis=1).tolist()
        subdata = Data(x, edge_index=sub_edge_index)
        subdata['label'] = label
        subdata_networkx = utils.to_networkx(subdata, node_attrs=['label'])
        # nx.draw(subdata_networkx)
        # plt.show()
        subgraph_networkx.append(subdata_networkx)

    nodes_indices = nodes_indices + data.num_nodes
    grakel_data = graph_from_networkx(subgraph_networkx, node_labels_tag='label')
    kernel_out = gk.fit_transform(grakel_data)
    kernel_out = np.nan_to_num(kernel_out)
    # sub_kernels.append(torch.from_numpy(kernel_out).to(device))
    sub_kernels=[]
    sub_kernels.append(kernel_out)
    return sub_kernels

def compute_kernel_for_batch(batch_data, device, iteration=3):
    # print(batch_data)
    sub_kernels = []
    X = batch_data.x[batch_data.subgraph_node_index].argmax(-1).to(device)
    E = batch_data.subgraph_edge_index.to(device)
    B = batch_data.subgraph_indicator.to(device)
    # print(iteration)
    wl = WL(iteration)
    wl.fit((X,E,B))
    kernel_out = wl.transform((X,E,B))
    num_nodes = torch.diff(batch_data.ptr).tolist()
    start_idx = 0
    del X, E, B, batch_data
    torch.cuda.empty_cache() 
    for num in num_nodes:
        sub_kernels.append(kernel_out[start_idx:start_idx+num, start_idx:start_idx+num])
        start_idx += num
    # print(sub_kernels[0])
    return sub_kernels


def save_kernel(kernel, cache_path):
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))
    with open(cache_path, 'wb') as f:
        pickle.dump(kernel, f)

def load_kernel(cache_path):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        kernel = pickle.load(f)
    return kernel

    

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])



class KernelCacheControl:
    def __init__(self, cache_dir, hop, wl):
        self.cache_dir = cache_dir
        self.hop = hop
        self.wl = wl
        self.kernel_cache = {}
        self._load_cache()

    def _get_cache_filename(self):
        return os.path.join(self.cache_dir, f"kernel_cache_hop_{self.hop}_wl_{self.wl}.pkl")

    def _load_cache(self):
        cache_file = self._get_cache_filename()
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.kernel_cache = pickle.load(f)

    def save_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = self._get_cache_filename()
        with open(cache_file, 'wb') as f:
            pickle.dump(self.kernel_cache, f)

    def save_kernel_to_cache(self, batch_data, kernels):
        for i, kernel in enumerate(kernels):
            graph_idx = batch_data[i].idx[0]
            self.kernel_cache[graph_idx] = kernel
        self.save_cache()

    def load_kernel_cache(self, batch_data):
        kernels = []
        for i in range(len(batch_data)):
            graph_idx = batch_data[i].idx[0].item()
            kernel = self.kernel_cache.get(graph_idx, None)
            kernels.append(kernel)
        return kernels


def lambda_lr_schedule(epoch):
    warmup_epochs = 10
    total_epochs = 100
    base_lr = 1e-3  # 假设的初始学习率
    max_lr = 1e-2   # 预热阶段结束时的最大学习率
    final_lr = 1e-4 # 训练结束时的最终学习率
    
    if epoch < warmup_epochs:
        lr = (max_lr - base_lr) / warmup_epochs * epoch + base_lr
    else:
        # 例如使用余弦退火调整学习率
        decay = (1 + math.cos((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * math.pi)) / 2
        lr = (max_lr - final_lr) * decay + final_lr
    return lr / base_lr

class PositionEncoding(object):
    def __init__(self, savepath=None, zero_diag=False):
        self.savepath = savepath
        self.zero_diag = zero_diag

    def apply_to(self, dataset, split='train'):
        saved_pos_enc = self.load(split)
        all_pe = []
        dataset.pe_list = []
        for i, g in enumerate(dataset):
            if saved_pos_enc is None:
                pe = self.compute_pe(g)
                all_pe.append(pe)
            else:
                pe = saved_pos_enc[i]
            if self.zero_diag:
                pe = pe.clone()
                pe.diagonal()[:] = 0
            dataset.pe_list.append(pe)

        self.save(all_pe, split)

        return dataset

    def save(self, pos_enc, split):
        if self.savepath is None:
            return
        if not os.path.isfile(self.savepath + "." + split):
            with open(self.savepath + "." + split, 'wb') as handle:
                pickle.dump(pos_enc, handle)

    def load(self, split):
        if self.savepath is None:
            return None
        if not os.path.isfile(self.savepath + "." + split):
            return None
        with open(self.savepath + "." + split, 'rb') as handle:
            pos_enc = pickle.load(handle)
        return pos_enc

    def compute_pe(self, graph):
        pass

class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = utils.get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()

    def apply_to(self, dataset):
        dataset.lap_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            dataset.lap_pe_list.append(pe)
        return dataset