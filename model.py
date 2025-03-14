import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.nn import DenseGraphConv, DenseGINConv, DenseSAGEConv, DenseGATConv
from layer import DiffTransformerEncoderLayer
from einops import repeat
from scipy.cluster.vq import kmeans2
#k-means clustering to extract features from vectors each layer with more layers. 
from timeit import default_timer as timer



class DiffTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, pe, degree=None, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, pe=pe, degree=degree, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model,
                 dim_feedforward=512, dropout=0.1, nb_layers=4,
                 batch_norm=False, lap_pos=False, lap_pos_dim=0, nb_heads=1, GNN=None):
        super(GraphTransformer, self).__init__()
        
        self.GNN=GNN
        self.lap_pos = lap_pos
        self.lap_pos_dim = lap_pos_dim
        if self.lap_pos and lap_pos_dim > 0:
            self.embedding_lap_pos = nn.Linear(lap_pos_dim, d_model)

        if GNN is None:
            self.embedding = nn.Linear(in_features=in_size,
                                    out_features=d_model,
                                    bias=False)
        else:
            self.embedding = DenseGraphConv(in_channels=in_size,
                                            out_channels=d_model,
                                            bias=True)
        encoder_layer = DiffTransformerEncoderLayer(
                d_model, dim_feedforward, dropout, batch_norm=batch_norm, nb_heads=nb_heads)
        self.encoder = DiffTransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalAvg1D()
        #self.classifier = nn.Linear(in_features=d_model,
        #                            out_features=nb_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
            )

    def forward(self, x, edge_index, masks, pe, lap_pe=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        # st = timer()
        
        # t1 = timer()
        if self.GNN is None:
            x = x.permute(1, 0, 2)
            output = self.embedding(x)
        else:
            output = self.embedding(x, edge_index)
            output = output.permute(1, 0, 2)
        # print(timer()-t1)
        if self.lap_pos and self.lap_pos_dim > 0:
            lap_pe = lap_pe.transpose(0,1)
            lap_pe = self.embedding_lap_pos(lap_pe)
            output = output + lap_pe
        
        output = self.encoder(output, pe, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        output = self.pooling(output, masks)
        # print(timer()-t1)
        # et = timer()
        # print(et-st)
        # we only do mean pooling for now.
        return self.classifier(output)


class GlobalAvg1D(nn.Module):
    def __init__(self):
        super(GlobalAvg1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = (~mask).float().unsqueeze(-1)
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1)

