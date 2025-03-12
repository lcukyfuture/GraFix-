
import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.aggr import Aggregation
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from torch_geometric.data import Data, Batch
from einops import rearrange
import torch.nn.functional as F
from timeit import default_timer as timer

import sys
sys.path.append('./src')


import torch
import warnings

from torch import nn

class SimplifiedAttention(nn.Module):
    def __init__(self, embed_dim, dropout_p=0.0, num_heads=1):
        super(SimplifiedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.num_heads = num_heads

        self.in_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.Tensor(embed_dim))
        self.out_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.out_proj_bias = nn.Parameter(torch.Tensor(embed_dim))
        # self.in_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,bias=False)
        # self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0)
        nn.init.xavier_uniform_(self.out_proj_weight)
        nn.init.constant_(self.out_proj_bias, 0)

    def forward(self, value, attn_output_weights, key_padding_mask=None, need_weights=None):
        tgt_len, bsz, embed_dim = value.size()
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch."
        assert attn_output_weights.size(1) == self.num_heads

        v_proj = F.linear(value, self.in_proj_weight, self.in_proj_bias).view(tgt_len, bsz, self.num_heads, -1)
        #[num_node, bsz, num_heads, dim]

        v_proj = v_proj.permute(1, 2, 0, 3)  #[bsz, num_heads, num_node, dim]
        

        attn_output = torch.einsum("bhij,bhjd->bhid", attn_output_weights, v_proj) 
        # print(timer()-t1)
        #[bsz, num_heads, num_nodes, dim]

        attn_output = attn_output.permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)
        #[num_node, bsz, dim]
        attn_output = F.linear(attn_output, self.out_proj_weight, self.out_proj_bias)
        
        if need_weights:
            # Optionally return the attention weights in addition to the output
            return attn_output, attn_output_weights
        else:
            return attn_output, None



class DiffTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_norm=True, nb_heads=1):
        super().__init__(d_model, nhead=nb_heads,  # nhead is set to 1 as it's unused in SimplifiedAttention
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.n_heads = nb_heads

        self.self_attn = SimplifiedAttention(d_model, num_heads=self.n_heads)
        self.self_attn.batch_first = False  
        self.self_attn._qkv_same_embed_dim = True  
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        self.scaling = None

    def forward(self, src, pe, degree=None, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, pe, key_padding_mask = src_key_padding_mask, need_weights=False)
        if degree is not None:
            src2 = degree.transpose(0, 1).contiguous().unsqueeze(-1) * src2 
        src = src + self.dropout1(src2)

        if self.batch_norm:
            bsz = src.shape[1]
            src = src.view(-1, src.shape[-1])
        src = self.norm1(src)
        # print(self.norm1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.batch_norm:
            src = src.view(-1, bsz, src.shape[-1])
        
        return src

