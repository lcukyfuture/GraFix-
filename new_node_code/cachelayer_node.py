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
    def __init__(self, embed_dim, dropout_p=0.0, num_heads=1, gate_mode="symmetric"):
        super(SimplifiedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        valid_gate_modes = {"none", "symmetric", "asymmetric"}
        if gate_mode not in valid_gate_modes:
            raise ValueError(f"Invalid gate_mode '{gate_mode}'. Expected one of {sorted(valid_gate_modes)}.")
        self.gate_mode = gate_mode

        self.in_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.Tensor(embed_dim))
        self.out_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.out_proj_bias = nn.Parameter(torch.Tensor(embed_dim))
        
        if self.gate_mode == "symmetric":
            self.gate = nn.Linear(embed_dim, num_heads)
        elif self.gate_mode == "asymmetric":
            self.gate_src = nn.Linear(embed_dim, num_heads)
            self.gate_dst = nn.Linear(embed_dim, num_heads)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0)
        nn.init.xavier_uniform_(self.out_proj_weight)
        nn.init.constant_(self.out_proj_bias, 0)
        if self.gate_mode == "symmetric":
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 0)
        elif self.gate_mode == "asymmetric":
            nn.init.xavier_uniform_(self.gate_src.weight)
            nn.init.constant_(self.gate_src.bias, 0)
            nn.init.xavier_uniform_(self.gate_dst.weight)
            nn.init.constant_(self.gate_dst.bias, 0)

    def forward(
        self,
        value,
        attn_output_weights,
        key_padding_mask=None,
        need_weights=None,
        need_gate_modulation=False,
        need_pre_norm_attention=False,
    ):
        tgt_len, bsz, embed_dim = value.size()
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch."
        assert attn_output_weights.size(1) == self.num_heads

        # 1. Compute Value projection (unchanged)
        v_proj = F.linear(value, self.in_proj_weight, self.in_proj_bias).view(tgt_len, bsz, self.num_heads, -1)
        v_proj = v_proj.permute(1, 2, 0, 3)  #[bsz, num_heads, num_node, dim]

        if self.gate_mode == "symmetric":
            h = value.permute(1, 0, 2)
            g = self.gate(h)
            gate_score = g.unsqueeze(2) + g.unsqueeze(1)
            gate_score = gate_score.permute(0, 3, 1, 2)
            modulation = torch.sigmoid(gate_score)
            pre_norm_weights = attn_output_weights * modulation
            modulated_weights = pre_norm_weights / (pre_norm_weights.sum(dim=-1, keepdim=True) + 1e-6)
        elif self.gate_mode == "asymmetric":
            h = value.permute(1, 0, 2)
            g_src = self.gate_src(h)
            g_dst = self.gate_dst(h)
            gate_score = g_src.unsqueeze(2) + g_dst.unsqueeze(1)
            gate_score = gate_score.permute(0, 3, 1, 2)
            modulation = torch.sigmoid(gate_score)
            pre_norm_weights = attn_output_weights * modulation
            modulated_weights = pre_norm_weights / (pre_norm_weights.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            modulation = torch.ones_like(attn_output_weights)
            pre_norm_weights = attn_output_weights * modulation
            modulated_weights = pre_norm_weights / (pre_norm_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 2. Aggregate using modulated weights
        attn_output = torch.einsum("bhij,bhjd->bhid", modulated_weights, v_proj) 
        #[bsz, num_heads, num_nodes, dim]

        attn_output = attn_output.permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)
        #[num_node, bsz, dim]
        attn_output = F.linear(attn_output, self.out_proj_weight, self.out_proj_bias)
        
        if need_gate_modulation:
            if need_weights and need_pre_norm_attention:
                return attn_output, modulated_weights, modulation, pre_norm_weights
            if need_weights:
                return attn_output, modulated_weights, modulation
            if need_pre_norm_attention:
                return attn_output, modulation, pre_norm_weights
            return attn_output, modulation
        if need_weights and need_pre_norm_attention:
            return attn_output, modulated_weights, pre_norm_weights
        if need_weights:
            return attn_output, modulated_weights
        if need_pre_norm_attention:
            return attn_output, pre_norm_weights
        else:
            return attn_output, None



class DiffTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_norm=True, nb_heads=1, gate_mode="symmetric"):
        super().__init__(d_model, nhead=nb_heads,  # nhead is set to 1 as it's unused in SimplifiedAttention
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.n_heads = nb_heads

        self.self_attn = SimplifiedAttention(d_model, num_heads=self.n_heads, gate_mode=gate_mode)
        self.self_attn.batch_first = False  
        self.self_attn._qkv_same_embed_dim = True  
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        self.scaling = None

    def forward(
        self,
        src,
        pe,
        degree=None,
        src_mask=None,
        src_key_padding_mask=None,
        return_attention=False,
        return_gate_modulation=False,
        return_pre_norm_attention=False,
    ):
        attention_outputs = self.self_attn(
            src,
            pe,
            key_padding_mask=src_key_padding_mask,
            need_weights=return_attention,
            need_gate_modulation=return_gate_modulation,
            need_pre_norm_attention=return_pre_norm_attention,
        )
        src2 = attention_outputs[0]
        output_idx = 1
        attn = None
        gate_modulation = None
        pre_norm_attention = None
        if return_attention:
            attn = attention_outputs[output_idx]
            output_idx += 1
        if return_gate_modulation:
            gate_modulation = attention_outputs[output_idx]
            output_idx += 1
        if return_pre_norm_attention:
            pre_norm_attention = attention_outputs[output_idx]
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

        outputs = [src]
        if return_attention:
            outputs.append(attn)
        if return_gate_modulation:
            outputs.append(gate_modulation)
        if return_pre_norm_attention:
            outputs.append(pre_norm_attention)
        if len(outputs) > 1:
            return tuple(outputs)
        return src
