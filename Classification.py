import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch

import torch.nn as nn
from torch.autograd import profiler
# from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader
from torch_geometric import datasets
import torch.nn.functional as F
from model import GraphTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from timeit import default_timer as timer
from torch_geometric.transforms import OneHotDegree
from torch.nn.utils.rnn import pad_sequence

import csv
import time 
import argparse
import copy
from data_idx2 import SubgraphDataset, GraphDataset
from utils import compute_kernel_for_batch, save_kernel, load_kernel, count_parameters, compute_kernel_CPU, LapEncoding
import pickle


def load_args():
    parser = argparse.ArgumentParser(description='Graph Kernel Transformer Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='MUTAG', choices=['MUTAG', 'PATTERN', 'PROTEINS','NCI1', 'PTC_MR', 'ogbg-molhiv', 'IMDB-BINARY','AIDS', 'ENZYMES'],
                        help='Dataset to use')
    parser.add_argument('--num-layers', type=int, default=3, help="number of layers")
    parser.add_argument('--hop', type=int, default=2, help='Hop for subgraph extraction')
    parser.add_argument('--numheads', type=int, default=2, help='Number of heads')
    parser.add_argument('--isgnn', type=bool, default=True, help='if use GNN as embedding layer')
    parser.add_argument('--lappe', type=bool, default=False, help='if use laplacian PE')
    parser.add_argument('--lap-dim', type=int, default=2, help='dimension for laplacian PE')
    parser.add_argument('--kernels', nargs='+', default=['WL', 'RW'], help='Kernel types for each head, e.g., "WL SP RW"')
    parser.add_argument('--kernel', type=str, default='WL_GPU', choices=['SP', 'WL', 'WLSP', 'RW','GL', 'WL_GPU'],
                        help='Kernel type')
    parser.add_argument('--GL_k', type=str, default=5, help='The dimension of given Graphlets')
    parser.add_argument('--fold', type=int, default=1, help='The number of K folds')
    parser.add_argument('--same-attn', type=bool, default=True, help='Use the same ')
    parser.add_argument('--dim_hidden', type=int, default=64, help="hidden dimension of Transformer")
    parser.add_argument('--epochs', type=int, default=300,help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch_size')
    parser.add_argument('--dropout', type=float, default=0, help='drop out rate')
    parser.add_argument('--outdir', type=str, default='',help='output path')
    parser.add_argument('--wl', type=int, default=3, help='WL_GPU iteration')
    parser.add_argument('--batch-norm', action='store_true', help='use batch norm instead of layer norm')
    args = parser.parse_args()
    
    if args.outdir != '':
        outdir = args.outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = os.path.join(outdir, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = os.path.join(outdir,'fold_{}'.format(args.fold))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.numheads, args.kernels[0], args.kernels[1], args.kernels[2], args.kernels[3], args.dim_hidden, args.wl, args.GL_k, args.num_layers, args.hop, args.dropout, args.lr, args.batch_size)
        # if not os.path.exists(outdir):
        #     os.makedirs(outdir)        
        # args.outdir = outdir
                # Adjust file name based on number of heads and kernels


        ### new 12/10 2024
        kernel_names = "_".join(args.kernels[:args.numheads])
        outdir = os.path.join(outdir, f'{args.isgnn}_{args.numheads}_{args.lappe}_{kernel_names}_{args.dim_hidden}_{args.wl}_{args.GL_k}_{args.num_layers}l_{args.hop}h_{args.dropout}_{args.lr}_{args.batch_size}')
        if not os.path.exists(outdir):
            os.makedirs(outdir)        
        args.outdir = outdir

    return args

device = torch.device('cuda')


def train(loader, model, warm_up, criterion, optimizer, lr_scheduler, epoch): 
    model.train()
    total_loss = 0.0
    train_corr = 0.0
    strat_time = timer()
    mid_time = 0
    # with profiler.profile(use_cuda=True) as prof:

    for i, batch in enumerate(loader):
        
        data, edge_index, mask, pe, lap, deg, labels = batch


        
        # print("loadertime:", timer() - strat_time)
        labels = labels.view(-1)
        
        data = data.to(device)
        edge_index = [e.to(device) for e in edge_index]
        mask = mask.to(device)
        pe = [p.to(device) for p in pe]
        
        if lap is not None:
            lap = lap.to(device)
        label = labels.to(device)

        #finalizing padding and stacking of larger metrices
        max_len = data.shape[1]
        #pe is a list of nh x np x np
        pe = pad_sequence([torch.nn.functional.pad(p.transpose(0,1),[0,max_len-p.shape[-1]]) for p in pe], batch_first=True, padding_value=0).transpose(1,2)
        edge_index = pad_sequence([torch.nn.functional.pad(e,[0,max_len-e.shape[-1]]) for e in edge_index], batch_first=True, padding_value=0)
        
        optimizer.zero_grad()
        #add kernel to model

        out = model(data, edge_index, mask, pe, lap, deg)


        loss = criterion(out, label)


        # mid_1 = timer() 
        loss.backward()
        
        optimizer.step()
        # mid_2 = timer()

        # mid_time += (mid_2-mid_1)



        train_pred = out.data.argmax(dim=1)
        total_loss += loss.item()*len(data)
        train_corr += torch.sum(train_pred==label).item()
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    # print("mid time:", mid_time)
    end_time = timer()
    epoch_time = end_time - strat_time
    n_samples = len(loader.dataset)
    train_avg_loss = total_loss / n_samples
    train_avg_corr = train_corr / n_samples


    return train_avg_loss, train_avg_corr, epoch_time


def val(loader, model, criterion):
    model.eval()
    val_loss = 0
    val_nums = 0
    corr = 0
    with torch.no_grad():
        # for data, edge_index, mask, pe, lap, deg, labels in loader:
     for i, batch in enumerate(loader):
            
            data, edge_index, mask, pe, lap, deg, labels = batch

            labels = labels.view(-1)

            size = len(data)
            data = data.to(device)
            edge_index = [e.to(device) for e in edge_index]
            mask = mask.to(device)
            pe = [p.to(device) for p in pe]
        
            #finalizing padding and stacking of larger metrices
            max_len = data.shape[1]
            #pe is a list of nh x np x np
            pe = pad_sequence([torch.nn.functional.pad(p.transpose(0,1),[0,max_len-p.shape[-1]]) for p in pe], batch_first=True, padding_value=0).transpose(1,2)
            edge_index = pad_sequence([torch.nn.functional.pad(e,[0,max_len-e.shape[-1]]) for e in edge_index], batch_first=True, padding_value=0)
        
            if lap is not None:
                lap = lap.to(device)
            label = labels.to(device)
            out = model(data, edge_index, mask, pe, lap, deg)

            loss = criterion(out, label)
            val_loss += loss.item()*size
            val_nums += size

            pred = out.argmax(dim=-1)
            corr += int((pred == label).sum())
    val_avg_loss = val_loss / val_nums
    val_avg_corr = corr / len(loader.dataset)
    val_avg_loss = round(val_avg_loss, 3)
    return val_avg_loss, val_avg_corr

def plot_curve(train_loss_list, test_loss_list, train_acc_list, test_acc_list, fold):
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
    plt.suptitle(f'Loss And Accuacy Curves of Fold {fold}')
    # plt.savefig(f'{args.kernel}{args.num_layers}layer{args.hop}hops{args.dropout}dropout_figs/curves_fold_{fold}.png')
    plt.savefig(os.path.join(args.outdir, f'curves_fold_{fold}.png'))
    plt.show()
def print_layer_parameter_counts(model):
    print("Layer-wise parameter counts:")
    print("="*50)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            print(f"{name}: {param_count} parameters")
            total_params += param_count
    print("="*50)
    print(f"Total trainable parameters: {total_params}")

global args



def main():
    
    global args
    args = load_args()
    torch.manual_seed(44)
    np.random.seed(44)
    data_path = './dataset/TUDataset'
    dataset_name = args.dataset
    # torch.use_deterministic_algorithms(True)
    # dataset = datasets.TUDataset(data_path, dataset_name)
    
    if args.dataset == 'IMDB-BINARY':
        transform = OneHotDegree(max_degree=540)
        dataset = datasets.TUDataset(root=data_path, name='IMDB-BINARY', transform=transform)
    else:
        dataset = datasets.TUDataset(data_path, dataset_name)
        
    classes = dataset.num_classes 

    print(f"{args.num_layers}layers {args.hop}hops")

    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    # csv_file = open(f'{args.kernel}{args.num_layers}layer{args.hop}hops{args.dropout}dropout_figs/{args.kernel}{args.num_layers}layer{args.hop}hops_results.csv', 'w', newline='')
    csv_file = open(args.outdir + '/results.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy', 'Best Epoch','Best Accuracy'])


    idx_path = 'new_folds/{}/inner_folds/{}-{}-{}.txt'
    test_idx_path = 'new_folds/{}/test_idx-{}.txt'
    inner_idx = 1

    train_fold_idx = torch.from_numpy(np.loadtxt(
        idx_path.format(args.dataset, 'train_idx', args.fold, inner_idx)).astype(int)).long()
    val_fold_idx = torch.from_numpy(np.loadtxt(
        idx_path.format(args.dataset, 'val_idx', args.fold, inner_idx)).astype(int)).long()
    test_fold_idx = torch.from_numpy(np.loadtxt(
        test_idx_path.format(args.dataset, args.fold)).astype(int)).long()

    # SubDataset = SubgraphDataset(dataset, k_hop = args.hop)
    print("Length of dataset:", len(dataset))

    if not os.path.exists("cache/pe/{}".format(args.dataset)):
        try:
            os.makedirs("cache/pe/{}".format(args.dataset))
        except Exception:
            pass
    
   
    all_kernel_results = []
    for head in range(args.numheads):
        kernel_type = args.kernels[head]
        wl = args.wl if kernel_type=='WL' else None
        gl = args.GL_k if kernel_type == 'GL' else None
        kernel_cache_path = 'cache/pe/{}/{}_{}_{}_{}.pkl'.format(
            args.dataset, kernel_type, wl, gl, args.hop)
        print(kernel_cache_path)
        Subgraph_kernels = load_kernel(kernel_cache_path)

        if Subgraph_kernels is None:
            print("compute {} kernel".format(kernel_type))
            Subgraph_kernels=[]
            for data in dataset:
                Subgraph_kernel = compute_kernel_CPU(args.dataset, data, kernel_type, args.hop, wl, gl)
                Subgraph_kernels.extend(Subgraph_kernel)
            save_kernel(Subgraph_kernels, kernel_cache_path)
        all_kernel_results.append(Subgraph_kernels)
    all_kernel_results = [list(head_kernels) for head_kernels in zip(*all_kernel_results)]

    
    # else:
    #     kernel_cache_path = 'cache/pe/{}/{}_{}_{}.pkl'.format(
    #         args.dataset, args.kernel, args.wl, args.hop)
    #     Subgraph_kernels = load_kernel(kernel_cache_path)

    # if Subgraph_kernels is None:
    #     Subgraph_kernels = []
    #     if args.kernel == 'WL_GPU':
    #         SubdDataset = SubgraphDataset(dataset, k_hop = args.hop)
    #         print("Length of dataset:", len(dataset))
    #         print("compute subgraph kernel...")
    #         SubDataloader = PyGDataLoader(SubdDataset, batch_size=1, shuffle=False)
    #         for data in SubDataloader:
    #             Subgraph_kernel = compute_kernel_for_batch(data, device, args.wl)
    #             Subgraph_kernels.extend(Subgraph_kernel)
    # else:
    #     for data in dataset: 
    #         Subgraph_kernel = compute_kernel_CPU(data, args.kernel, args.hop, args.wl)
    #         Subgraph_kernels.extend(Subgraph_kernel)
    # save_kernel(Subgraph_kernels, kernel_cache_path)

    # print('subgraph kernel:',Subgraph_kernel)
    train_fold_idx = train_fold_idx.tolist()
    val_fold_idx = val_fold_idx.tolist()
    test_fold_idx = test_fold_idx.tolist()
    train_dataset = GraphDataset(dataset[train_fold_idx], nb_heads=args.numheads, degree=True)
    
    print(len(train_dataset))

    val_dataset = GraphDataset(dataset[val_fold_idx], nb_heads=args.numheads, degree=True)
    test_dataset = GraphDataset(dataset[test_fold_idx], nb_heads=args.numheads, degree=True)
    
    # train_dataset.pe_list = [Subgraph_kernels[i] for i in train_fold_idx]
    # val_dataset.pe_list = [Subgraph_kernels[i] for i in val_fold_idx]
    # test_dataset.pe_list = [Subgraph_kernels[i] for i in test_fold_idx]
    # train_dataset.pe_list = [all_kernel_results[i] for i in train_fold_idx]
    # val_dataset.pe_list = [all_kernel_results[i] for i in val_fold_idx]
    # test_dataset.pe_list = [all_kernel_results[i] for i in test_fold_idx]
    print(len(all_kernel_results))
    if args.numheads == 1:
        train_dataset.pe_list = [torch.tensor(all_kernel_results[i]) for i in train_fold_idx]
        val_dataset.pe_list = [torch.tensor(all_kernel_results[i]) for i in val_fold_idx]
        test_dataset.pe_list = [torch.tensor(all_kernel_results[i]) for i in test_fold_idx]

    train_dataset.pe_list = [torch.stack([torch.tensor(head) for head in all_kernel_results[i]]) for i in train_fold_idx]
    val_dataset.pe_list = [torch.stack([torch.tensor(head) for head in all_kernel_results[i]]) for i in val_fold_idx]
    test_dataset.pe_list = [torch.stack([torch.tensor(head) for head in all_kernel_results[i]]) for i in test_fold_idx]

    if args.lappe and args.lap_dim > 0:
        lap_pos_encoder = LapEncoding(args.lap_dim, normalization='sym')
        lap_pos_encoder.apply_to(train_dataset)
        lap_pos_encoder.apply_to(val_dataset)
        lap_pos_encoder.apply_to(test_dataset)
    # print(train_dataset[0])

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=train_dataset.collate_fn())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn())



    best_acc = 0
    best_epoch = 9999
    input_size = dataset.num_node_features
    nb_class = dataset.num_classes


    # train_dataset = dataset[train_fold_idx]
    # val_dataset = dataset[val_fold_idx]
    # test_dataset = dataset[test_fold_idx]
    model = GraphTransformer(in_size=input_size,
                            nb_class=nb_class,
                            d_model=args.dim_hidden,
                            dim_feedforward=2*args.dim_hidden,
                            dropout=args.dropout,
                            nb_layers=args.num_layers,
                            batch_norm=False,
                            lap_pos = args.lappe,
                            lap_pos_dim = args.lap_dim,
                            nb_heads = args.numheads,
                            GNN = args.isgnn,
                            ).to(device)
    
    print("Total number of parameters: {}".format(count_parameters(model)))
    print_layer_parameter_counts(model)
    # print("Total number of parameters: {}".format(count_parameters(model)))
    # for name, param in model.named_parameters():
    #     print(name, param.data)
    # print(model)
    # print(model.parameters)
    warm_up = 100
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr , weight_decay = weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
    # lr_steps = lr / (warm_up * len(train_dataloader))
    # def warmup_lr_scheduler(s):
    #     lr = s * lr_steps
    #     return lr
    # lr_steps = (args.lr - 1e-6) / args.warmup
    # decay_factor = args.lr * args.warmup ** .5
    # def lr_scheduler(s):
    #     if s < args.warmup:
    #         lr = 1e-6 + s * lr_steps
    #     else:
    #         lr = decay_factor * s ** -.5
    #     return lr

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    best_loss = float('inf')
    patience_counter = 0
    Allstart_time = time.time()
    epoch_time_list = []
    for epoch in range(args.epochs):

        print(f'Epoch: {epoch}/{args.epochs}, LR: {optimizer.param_groups[0]["lr"]}')
        # train_loss, train_acc, epoch_time = train(train_dataloader, model, warm_up, criterion, optimizer, warmup_lr_scheduler, epoch)
        train_loss, train_acc, epoch_time = train(train_loader, model, warm_up, criterion, optimizer, lr_scheduler, epoch)
        epoch_time_list.append(epoch_time)
        val_loss, val_acc = val(val_loader, model, criterion)
        lr_scheduler.step()
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weight = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 200:
                break

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f'epoch: {epoch:03d}, Train loss: {train_loss:.4f}, val loss:{val_loss:.4f}, Train acc: {train_acc:.4f}, val acc : {val_acc:.4f}, Best loss: {best_loss:.4f}, Epoch time: {epoch_time}')
        csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, best_epoch, best_loss])

    print(f'Best epoch: {best_epoch}')
    print(f'Best val loss for fold {args.fold}: {best_loss:.4f}')
    model.load_state_dict(best_weight)
    test_loss, test_acc = val(test_loader, model, criterion)
    print(f'Test acc for fold {args.fold}: {test_acc:.4f}')
    csv_writer.writerow([test_loss, test_acc])
    plot_curve(train_loss_list, val_loss_list, train_acc_list, val_acc_list, args.fold)

    Allend_time = time.time()
    Gap_time = Allend_time - Allstart_time
    print(f'Time: {Gap_time}')
    csv_writer.writerow([Gap_time])
    mean_epoch_time = np.mean(epoch_time_list)
    std_epoch_time = np.std(epoch_time_list)/np.sqrt(epoch)
    print(f'epoch_time: {mean_epoch_time:.4f} +/-{std_epoch_time:.4f}')
    
if __name__ == "__main__":
    # args = load_args()
    # dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    
    main()