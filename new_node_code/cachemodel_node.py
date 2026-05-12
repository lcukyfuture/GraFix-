# filepath: /home/zhanglingfeng/ICPR/Final_code_new_table/cachemodel_node.py
import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.nn import DenseGraphConv, DenseGINConv, DenseSAGEConv, DenseGATConv
from cachelayer_node import DiffTransformerEncoderLayer


class DiffTransformerEncoder(nn.TransformerEncoder):
    def forward(
        self,
        src,
        pe,
        degree=None,
        mask=None,
        src_key_padding_mask=None,
        return_attention=False,
        return_gate_modulation=False,
        return_pre_norm_attention=False,
    ):
        output = src
        last_attention = None
        last_gate_modulation = None
        last_pre_norm_attention = None
        for mod in self.layers:
            layer_output = mod(
                output,
                pe=pe,
                degree=degree,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=return_attention,
                return_gate_modulation=return_gate_modulation,
                return_pre_norm_attention=return_pre_norm_attention,
            )
            if return_attention or return_gate_modulation or return_pre_norm_attention:
                output = layer_output[0]
                output_idx = 1
                if return_attention:
                    last_attention = layer_output[output_idx]
                    output_idx += 1
                if return_gate_modulation:
                    last_gate_modulation = layer_output[output_idx]
                    output_idx += 1
                if return_pre_norm_attention:
                    last_pre_norm_attention = layer_output[output_idx]
            else:
                output = layer_output
        if self.norm is not None:
            output = self.norm(output)

        outputs = [output]
        if return_attention:
            outputs.append(last_attention)
        if return_gate_modulation:
            outputs.append(last_gate_modulation)
        if return_pre_norm_attention:
            outputs.append(last_pre_norm_attention)
        if len(outputs) > 1:
            return tuple(outputs)
        return output


class GraphTransformerNode(nn.Module):
    """
    GraphTransformer for Node Classification
    
    Key differences from graph classification:
    1. No pooling layer - we need predictions for all nodes
    2. Output shape: (batch_size, num_nodes, num_classes)
    3. Each node gets its own prediction
    """
    def __init__(self, in_size, nb_class, d_model,
                 dim_feedforward=512, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos=False, lap_pos_dim=0, nb_heads=1, GNN=None, gate_mode="symmetric"):
        super(GraphTransformerNode, self).__init__()

        self.GNN = GNN
        self.lap_pos = lap_pos
        self.lap_pos_dim = lap_pos_dim
        self.nb_heads = nb_heads
        self.gate_mode = gate_mode
        
        if self.lap_pos and lap_pos_dim > 0:
            self.embedding_lap_pos = nn.Linear(lap_pos_dim, d_model)

        # Embedding layer for node features
        if GNN is None:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)
        else:
            # Currently using DenseGraphConv as default if GNN is specified
            # You can expand this logic to support other GNN types like in cachemodel.py if needed
            self.embedding = DenseGraphConv(in_channels=in_size,
                                            out_channels=d_model,
                                            bias=True)
        
        # Transformer encoder layers
        encoder_layer = DiffTransformerEncoderLayer(
                d_model, dim_feedforward, dropout, batch_norm=batch_norm, nb_heads=nb_heads, gate_mode=gate_mode)
        self.encoder = DiffTransformerEncoder(encoder_layer, nb_layers)
        
        # Classifier for each node (no pooling)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
        )

    def forward(
        self,
        x,
        edge_index,
        masks,
        pe,
        lap_pe=None,
        degree=None,
        return_attention=False,
        return_gate_modulation=False,
        return_pre_norm_attention=False,
    ):
        """
        Args:
            x: Node features (batch_size, num_nodes, in_size)
            edge_index: Adjacency matrix or edge index (batch_size, num_nodes, num_nodes) for Dense GNN
            masks: Attention masks (batch_size, num_heads, num_nodes, num_nodes)
            pe: Positional encoding / kernel matrix (batch_size, num_heads, num_nodes, num_nodes)
            lap_pe: Laplacian positional encoding (batch_size, num_nodes, lap_pos_dim)
            degree: Degree matrix/vector if needed by encoder
        
        Returns:
            output: Node predictions (batch_size, num_nodes, nb_class)
        """
        
        # Embed node features
        if self.GNN is None:
            # Permute to transformer convention: (num_nodes, batch_size, in_size)
            x = x.permute(1, 0, 2)
            output = self.embedding(x)  # (num_nodes, batch_size, d_model)
        else:
            # GNN expects (batch_size, num_nodes, in_size) and adj
            output = self.embedding(x, edge_index)
            # Permute to transformer convention after GNN
            output = output.permute(1, 0, 2) # (num_nodes, batch_size, d_model)
        
        # Add Laplacian PE if available
        if self.lap_pos and self.lap_pos_dim > 0 and lap_pe is not None:
            lap_pe = lap_pe.transpose(0, 1)  # (num_nodes, batch_size, lap_pos_dim)
            lap_pe = self.embedding_lap_pos(lap_pe)
            output = output + lap_pe
        
        # ============================================================
        # Fuse Adjacency Matrix into PE (Kernel) to incorporate local connectivity
        # ============================================================
        # if self.gate_mode != 'none' and edge_index is not None:
        #     # edge_index is passed as Dense Adjacency Matrix: [Batch, N, N]
        #     adj = edge_index.clone()
            
        #     # 1. Add self-loops to ensure nodes attend to themselves
        #     B, N, _ = adj.shape
        #     diag_idx = torch.arange(N, device=adj.device)
        #     adj[:, diag_idx, diag_idx] = 1.0
            
        #     # 2. Row Normalization (D^-1 A) to prevent value explosion
        #     # Similar to GCN aggregation: mean of neighbors
        #     deg_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        #     adj_norm = adj / deg_sum
            
        #     # 3. Expand dimensions to match PE: [Batch, 1, N, N]
        #     adj_expanded = adj_norm.unsqueeze(1)
            
        #     # 4. Fuse: Combine local adjacency with global structural kernel
        #     # We add the normalized adjacency to the kernel matrix.
        #     # This forces the attention mechanism to prioritize local neighbors (homophily),
        #     # while still retaining global structural information from the kernel.
        #     # The factor 0.5 is a hyperparameter to balance local vs global info.
        #     pe = adj_expanded + 0.5 * pe

        # Apply transformer encoder
        # Note: src_key_padding_mask logic might need adjustment depending on how masks are passed
        # In cachemodel.py, masks are passed as src_key_padding_mask. 
        # Assuming masks here is for attention bias or padding mask.
        last_attention = None
        last_gate_modulation = None
        last_pre_norm_attention = None
        if return_attention or return_gate_modulation or return_pre_norm_attention:
            encoder_output = self.encoder(
                output,
                pe,
                degree=degree,
                src_key_padding_mask=masks,
                return_attention=return_attention,
                return_gate_modulation=return_gate_modulation,
                return_pre_norm_attention=return_pre_norm_attention,
            )
            output = encoder_output[0]
            output_idx = 1
            if return_attention:
                last_attention = encoder_output[output_idx]
                output_idx += 1
            if return_gate_modulation:
                last_gate_modulation = encoder_output[output_idx]
                output_idx += 1
            if return_pre_norm_attention:
                last_pre_norm_attention = encoder_output[output_idx]
        else:
            output = self.encoder(output, pe, degree=degree, src_key_padding_mask=masks)
        # output: (num_nodes, batch_size, d_model)
        
        # Permute back to (batch_size, num_nodes, d_model)
        output = output.permute(1, 0, 2)
        
        # Classify each node
        output = self.classifier(output)  # (batch_size, num_nodes, nb_class)
        
        outputs = [output]
        if return_attention:
            outputs.append(last_attention)
        if return_gate_modulation:
            outputs.append(last_gate_modulation)
        if return_pre_norm_attention:
            outputs.append(last_pre_norm_attention)
        if len(outputs) > 1:
            return tuple(outputs)
        return output
