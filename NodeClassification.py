import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric import datasets
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree
import torch.nn.functional as F
from cachemodel_node import GraphTransformerNode
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import csv
import time 
import argparse
import copy
from utils import save_kernel, load_kernel, count_parameters, LapEncoding, compute_node_kernel_CPU
from tqdm import tqdm
import pickle

def load_args():
    parser = argparse.ArgumentParser(description='Graph Kernel Transformer Node Classification', 
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='PubMed', 
                       choices=['Cora', 'CiteSeer', 'PubMed', 'Cornell'],
                       help='Dataset to use')
    parser.add_argument('--num-layers', type=int, default=1, help="number of layers")
    parser.add_argument('--hop', type=int, default=2, help='Hop for subgraph extraction')
    parser.add_argument('--numheads', type=int, default=1, help='Number of heads')
    parser.add_argument('--isgnn', type=bool, default=True, help='if use GNN as embedding layer')
    parser.add_argument('--lappe', type=bool, default=False, help='use laplacian PE')
    parser.add_argument('--lap-dim', type=int, default=2, help='dimension for laplacian PE')
    parser.add_argument('--kernels', nargs='+', default=['WL', 'SP', 'RW', 'GL'], 
                       help='Kernel types for each head, e.g., "WL SP RW"')
    parser.add_argument('--kernel', type=str, default='WL', 
                       choices=['SP', 'WL', 'RW','GL'],
                       help='Kernel type')
    parser.add_argument('--GL_k', type=int, default=3, 
                       help='The dimension of given Graphlets')
    parser.add_argument('--dim_hidden', type=int, default=64, 
                       help="hidden dimension of Transformer")
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch_size')
    parser.add_argument('--dropout', type=float, default=0.5, help='drop out rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--patience', type=int, default=300, help='patience for early stopping')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], 
                       help='optimizer type')
    parser.add_argument('--outdir', type=str, default='', help='output path')
    parser.add_argument('--wl', type=int, default=3, help='WL iteration')
    parser.add_argument('--batch-norm', action='store_true', 
                       help='use batch norm instead of layer norm')
    parser.add_argument('--ablation-no-kernel', action='store_true',
                       help='Ablation study: Set kernel matrix to all ones to discard structural prior')
    parser.add_argument('--original-model', action='store_true',
                       help='Use the original model without adjacency fusion and gating mechanism')
    args = parser.parse_args()
    
    if args.outdir != '':
        outdir = args.outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = os.path.join(outdir, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        kernel_names = "_".join(args.kernels[:args.numheads])
        outdir = os.path.join(outdir, 
                            f'{args.isgnn}_{args.numheads}_{args.lappe}_{kernel_names}_{args.dim_hidden}_'
                            f'{args.wl}_{args.GL_k}_{args.num_layers}l_{args.hop}h_'
                            f'{args.dropout}_{args.lr}_{args.batch_size}')
        if not os.path.exists(outdir):
            os.makedirs(outdir)        
        args.outdir = outdir

    return args


class NodeSubgraphDataset(Dataset):
    """Dataset for node classification with subgraph kernels"""
    def __init__(self, data, node_indices, all_node_features, pe_matrix, adj, deg, lap_list=None, nb_heads=1):
        self.data = data
        self.node_indices = node_indices  # Indices for train/val/test split
        self.all_node_features = all_node_features  # All node features (num_nodes, feature_dim)
        self.pe_matrix = pe_matrix  # PE matrix for all nodes (num_heads, num_nodes, num_nodes)
        self.adj = adj
        self.deg = deg
        self.lap_list = lap_list  # LAP for all nodes if available
        self.nb_heads = nb_heads
        
    def __len__(self):
        return 1  # Only one graph, return entire graph each time
    
    def __getitem__(self, idx):
        # Return all nodes' data, but only labels for train/val/test nodes
        labels = self.data.y[self.node_indices]
        return self.all_node_features, self.pe_matrix, self.adj, self.deg, self.lap_list, labels, self.node_indices
    
    def collate_fn(self):
        def collate(batch):
            # batch contains only one item (the entire graph)
            all_features, pe_matrix, adj, deg, lap_list, labels, node_indices = batch[0]
            
            # all_features: (num_nodes, feature_dim)
            # pe_matrix: (num_heads, num_nodes, num_nodes) or (num_nodes, num_nodes)
            # labels: (num_train/val/test_nodes,)
            # node_indices: indices of train/val/test nodes
            
            # Unsqueeze to add batch dimension
            all_features = all_features.unsqueeze(0)  # (1, num_nodes, feature_dim)
            adj = adj.unsqueeze(0) # (1, num_nodes, num_nodes)
            
            if self.nb_heads == 1:
                # (1, 1, num_nodes, num_nodes)
                pe_matrix = pe_matrix.unsqueeze(0).unsqueeze(0)
            else:
                # (1, num_heads, num_nodes, num_nodes)
                pe_matrix = pe_matrix.unsqueeze(0)
            
            # Create mask (all ones)
            mask = torch.ones_like(pe_matrix)
            
            # LAP
            if lap_list is not None:
                lap_tensor = lap_list.unsqueeze(0)  # (1, num_nodes, lap_dim)
            else:
                lap_tensor = None
            
            if deg is not None:
                deg = deg.unsqueeze(0)
            
            return all_features, adj, mask, pe_matrix, lap_tensor, deg, labels, torch.tensor(node_indices)
        
        return collate


def extract_node_subgraphs(data, hop=2):
    """Extract k-hop subgraph for each node"""
    from torch_geometric.utils import k_hop_subgraph
    
    num_nodes = data.num_nodes
    node_subgraphs = []
    
    for node_idx in tqdm(range(num_nodes), desc=f"Extracting {hop}-hop subgraphs"):
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, hop, data.edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        
        # Create subgraph data
        subgraph_data = type(data)()
        subgraph_data.x = data.x[subset]
        subgraph_data.edge_index = edge_index
        subgraph_data.num_nodes = len(subset)
        
        node_subgraphs.append(subgraph_data)
    
    return node_subgraphs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    train_corr = 0.0
    total_samples = 0
    start_time = timer()
    
    for all_features, adj, mask, pe_matrix, lap, deg, labels, node_indices in loader:
        all_features = all_features.to(device)
        adj = adj.to(device)
        mask = mask.to(device)
        pe_matrix = pe_matrix.to(device)
        if lap is not None:
            lap = lap.to(device)
        if deg is not None:
            deg = deg.to(device)
        labels = labels.to(device)
        node_indices = node_indices.to(device)
        
        optimizer.zero_grad()
        # Forward pass for all nodes
        out = model(all_features, adj, mask, pe_matrix, lap, deg)  # (1, num_nodes, num_classes)
        
        # Extract predictions for train nodes only
        out = out.squeeze(0)[node_indices]  # (num_train_nodes, num_classes)
        
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        train_pred = out.data.argmax(dim=1)
        total_loss += loss.item() * len(labels)
        train_corr += torch.sum(train_pred == labels).item()
        total_samples += len(labels)
    
    end_time = timer()
    epoch_time = end_time - start_time
    train_avg_loss = total_loss / total_samples
    train_avg_corr = train_corr / total_samples
    
    return train_avg_loss, train_avg_corr, epoch_time


def val(loader, model, criterion):
    model.eval()
    val_loss = 0
    val_nums = 0
    corr = 0
    
    with torch.no_grad():
        for all_features, adj, mask, pe_matrix, lap, deg, labels, node_indices in loader:
            all_features = all_features.to(device)
            adj = adj.to(device)
            mask = mask.to(device)
            pe_matrix = pe_matrix.to(device)
            if lap is not None:
                lap = lap.to(device)
            if deg is not None:
                deg = deg.to(device)
            labels = labels.to(device)
            node_indices = node_indices.to(device)
            
            # Forward pass for all nodes
            out = model(all_features, adj, mask, pe_matrix, lap, deg)  # (1, num_nodes, num_classes)
            
            # Extract predictions for val/test nodes only
            out = out.squeeze(0)[node_indices]  # (num_val_nodes, num_classes)
            
            loss = criterion(out, labels)
            val_loss += loss.item() * len(labels)
            val_nums += len(labels)
            
            pred = out.argmax(dim=-1)
            corr += int((pred == labels).sum())
    
    val_avg_loss = val_loss / val_nums
    val_avg_corr = corr / val_nums  # Changed from len(loader.dataset)
    val_avg_loss = round(val_avg_loss, 3)
    
    return val_avg_loss, val_avg_corr


def plot_curve(train_loss_list, test_loss_list, train_acc_list, test_acc_list, outdir):
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(test_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.suptitle(f'Node Classification: Loss And Accuracy Curves')
    plt.savefig(os.path.join(outdir, f'curves.png'))
    plt.close()


def main():
    args = load_args()
    torch.manual_seed(42)
    np.random.seed(42)

    
    # Load dataset
    data_path = '../dataset/'
    if args.dataset == 'Cora':
        dataset = datasets.Planetoid(root=data_path, name='Cora')
    elif args.dataset == 'CiteSeer':
        dataset = datasets.Planetoid(root=data_path, name='CiteSeer')
    elif args.dataset == 'PubMed':
        dataset = datasets.Planetoid(root=data_path, name='PubMed')
    elif args.dataset == 'Cornell':
        dataset = datasets.WebKB(root=data_path, name='Cornell')
    
    data = dataset[0]
    num_classes = dataset.num_classes
    
    # Handle multiple masks (e.g., in WebKB datasets like Cornell)
    if hasattr(data, 'train_mask') and data.train_mask.dim() > 1:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {num_classes}")
    
    # [Mod] Feature Clustering for WL Kernel (Only for original model as requested)
    if args.original_model:
        print("\n!!! Applying Feature Clustering for Kernel Computation (Original Model) !!!")
        print("This discretizes high-dimensional features to help WL kernel learn meaningful similarities.")
        from sklearn.cluster import MiniBatchKMeans
        
        # Adjust clusters based on dataset characteristics
        if args.dataset == 'PubMed':
            n_clusters = 500  # Larger dataset, more clusters
        elif args.dataset in ['Cora', 'CiteSeer']:
            n_clusters = 128  # Medium size, balance granularity
        else:
            n_clusters = 50   # Small datasets like Cornell
            
        print(f"Clustering features into {n_clusters} clusters...")
        
        # CPU clustering - normalize features before clustering to improve KMeans
        from sklearn.preprocessing import normalize
        features_np = data.x.cpu().numpy()
        features_norm = normalize(features_np, axis=1)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init='auto')
        clusters = kmeans.fit_predict(features_norm)
        
        # Convert to One-Hot Tensor
        cluster_features = torch.zeros(data.num_nodes, n_clusters)
        cluster_features.scatter_(1, torch.tensor(clusters).unsqueeze(1), 1.0)
        
        # Create a temporary data object with clustered features
        data_for_kernel = copy.copy(data)
        data_for_kernel.x = cluster_features
    else:
        data_for_kernel = data

    # Extract node subgraphs
    print(f"Extracting {args.hop}-hop subgraphs for each node...")
    node_subgraphs_for_kernel = extract_node_subgraphs(data_for_kernel, hop=args.hop)

    # Compute kernels for all node subgraphs
    if not os.path.exists("cache/pe_node/{}".format(args.dataset)):
        try:
            os.makedirs("cache/pe_node/{}".format(args.dataset))
        except Exception:
            pass
    
    all_kernel_results = []
    for head in range(args.numheads):
        kernel_type = args.kernels[head]
        wl = args.wl if kernel_type == 'WL' else None
        gl = args.GL_k if kernel_type == 'GL' else None
        
        # Append _clustered to cache filename if using clustered features
        # Include n_clusters in suffix to avoid cache conflicts
        if args.original_model:
            # Re-calculate n_clusters for cache path consistency
            if args.dataset == 'PubMed':
                n_clusters = 500
            elif args.dataset in ['Cora', 'CiteSeer']:
                n_clusters = 128
            else:
                n_clusters = 50
            cache_suffix = f"_clustered_{n_clusters}"
        else:
            cache_suffix = ""
            
        kernel_cache_path = 'cache/pe_node/{}/{}_{}_{}_{}{}.pkl'.format(
            args.dataset, kernel_type, wl, gl, args.hop, cache_suffix)
        
        node_kernels = load_kernel(kernel_cache_path)
        
        if node_kernels is None:
            print(f"\n=== Computing {kernel_type} kernel (head {head+1}/{args.numheads}) ===")
            # Use the utility function to compute kernel matrix
            # Use node_subgraphs_for_kernel (which might have clustered features)
            node_kernels = compute_node_kernel_CPU(node_subgraphs_for_kernel, kernel_type, wl, gl)
            save_kernel(node_kernels, kernel_cache_path)
        else:
            print(f"Loaded cached {kernel_type} kernel (head {head+1}/{args.numheads})")
            if args.original_model:
                print("(Note: This kernel was computed using clustered features)")
        
        all_kernel_results.append(node_kernels)
    
    # Prepare PE matrix for all nodes
    if args.numheads == 1:
        pe_matrix = torch.tensor(all_kernel_results[0], dtype=torch.float)
    else:
        pe_matrix = torch.stack([torch.tensor(all_kernel_results[h], dtype=torch.float) 
                                for h in range(args.numheads)])

    # Ablation study: Discard structural prior
    if args.ablation_no_kernel:
        print("!!! ABLATION STUDY: Discarding structural prior (Setting Kernel = 1) !!!")
        pe_matrix = torch.ones_like(pe_matrix)

    # [Mod] Normalize Kernel Matrix
    # This step ensures kernel values are within a reasonable range [0, 1]
    # It helps with numerical stability and prevents gradient issues.
    print("Normalizing Kernel Matrix...")
    if pe_matrix.dim() == 3: # (num_heads, num_nodes, num_nodes)
        # Max normalization per head
        max_val = pe_matrix.flatten(start_dim=-2).max(dim=-1)[0].view(-1, 1, 1)
        pe_matrix = pe_matrix / (max_val + 1e-6)
    else: # (num_nodes, num_nodes)
        pe_matrix = pe_matrix / (pe_matrix.max() + 1e-6)

    # All node features
    all_node_features = data.x
    
    # Laplacian PE (optional)
    lap_matrix = None
    if args.lappe and args.lap_dim > 0:
        lap_pos_encoder = LapEncoding(args.lap_dim, normalization='sym')
        # Compute LAP for the entire graph
        lap_matrix = lap_pos_encoder.compute_pe(data)
    
    # Compute adj and degree
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    deg = degree(data.edge_index[0], data.num_nodes)

    # Split nodes into train/val/test
    train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    val_indices = data.val_mask.nonzero(as_tuple=True)[0].tolist()
    test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    
    # Create datasets
    train_dataset = NodeSubgraphDataset(data, train_indices, all_node_features, 
                                       pe_matrix, adj, deg, lap_matrix, args.numheads)
    val_dataset = NodeSubgraphDataset(data, val_indices, all_node_features, 
                                     pe_matrix, adj, deg, lap_matrix, args.numheads)
    test_dataset = NodeSubgraphDataset(data, test_indices, all_node_features, 
                                      pe_matrix, adj, deg, lap_matrix, args.numheads)
    
    # Create dataloaders (batch_size=1 since we process the whole graph at once)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, 
                             collate_fn=train_dataset.collate_fn())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           collate_fn=val_dataset.collate_fn())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                            collate_fn=test_dataset.collate_fn())
    
    # Initialize model - use node feature dimension
    input_size = data.num_features
    model = GraphTransformerNode(in_size=input_size,  # Use node feature dimension
                            nb_class=num_classes,
                            d_model=args.dim_hidden,
                            dim_feedforward=2*args.dim_hidden,
                            dropout=args.dropout,
                            nb_layers=args.num_layers,
                            batch_norm=args.batch_norm,
                            lap_pos=args.lappe,
                            lap_pos_dim=args.lap_dim,
                            nb_heads=args.numheads,
                            GNN=args.isgnn,
                            use_original_model=args.original_model).to(device)
    
    print("Total number of parameters: {}".format(count_parameters(model)))
    
    # Training setup
    if args.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                factor=0.5, patience=20, 
                                                                min_lr=1e-5)
    
    # CSV logging
    csv_file = open(args.outdir + '/results.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 
                        'Val Loss', 'Val Accuracy', 'Best Epoch', 'Best Accuracy'])
    
    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    
    start_time = time.time()
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}/{args.epochs}, LR: {optimizer.param_groups[0]["lr"]}')
        
        train_loss, train_acc, _ = train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = val(val_loader, model, criterion)
        lr_scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weight = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, '
              f'Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, Best loss: {best_loss:.4f}')
        csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, best_epoch, best_loss])
    
    end_time = time.time()
    
    # Test
    print(f'Best epoch: {best_epoch}')
    print(f'Best val loss: {best_loss:.4f}')
    model.load_state_dict(best_weight)
    test_loss, test_acc = val(test_loader, model, criterion)
    print(f'Test acc: {test_acc:.4f}')
    
    csv_writer.writerow(['Test', test_loss, test_acc])
    csv_writer.writerow(['Total Time', end_time - start_time])
    csv_file.close()
    
    plot_curve(train_loss_list, val_loss_list, train_acc_list, val_acc_list, args.outdir)
    
    print(f'Total training time: {end_time - start_time:.2f}s')


if __name__ == "__main__":
    main()
