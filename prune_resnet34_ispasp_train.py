import argparse
import os
import math
from copy import deepcopy

import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from fvcore.nn import FlopCountAnalysis

from lib.data import get_dataset_ft
from lib.utils import accuracy, AverageMeter

from models_cifar10.resnet import resnet34 as cifar10_resnet34

RANDOM_SEED = 0
RNG = npr.default_rng(seed=RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class PrunedBasicBlock(nn.Module):
    def __init__(self, inplanes, midplanes, outplanes, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, midplanes)
        self.bn1 = norm_layer(midplanes)
        #self.relu = nn.ReLU()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(midplanes, outplanes)
        self.bn2 = norm_layer(outplanes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon = 0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def get_flops_and_params(model):
    model = model.eval()
    with torch.no_grad():
        # flop count
        inp = torch.zeros(1, 3, 224, 224)
        flops = FlopCountAnalysis(model.cpu(), inp).total()

        # param count
        params = sum(p.numel() for p in model.parameters())
        return flops, params


def residual_objective(mat):
    return 0.5 * torch.mean(mat**2)


def compute_grad_and_bn_step(og_hidden, og_output, pruned_block, optimizer, bs, num_batches, device):
    grad_W = None

    for i in range(0, og_hidden.shape[0], bs):
        mb_hid = og_hidden[i:i + bs, :]
        mb_hid = mb_hid.to(device)
        mb_og_out = og_output[i:i + bs, :]
        mb_og_out = mb_og_out.to(device)
        
        pruned_block.conv2.weight.requires_grad = True # track gradient on layer weights to determine importance
        pruned_block.bn2.bias.requires_grad = True # track gradient on batch norm parameters for later update
        pruned_block.bn2.weight.requires_grad = True # track gradient on batch norm parameters for later update

        pruned_output = pruned_block.conv2(mb_hid)
        pruned_output = pruned_block.bn2(pruned_output)
        
        residual = residual_objective(mb_og_out.detach() - pruned_output) 
        residual.backward()

        tmp_imp = pruned_block.conv2.weight.grad.detach().cpu()
        
        with torch.no_grad():
            if grad_W is None:
                grad_W = tmp_imp
            else:
                grad_W += tmp_imp

        optimizer.step() # take a step of SGD over the batch norm params

        mb_hid.grad = None
        pruned_block.zero_grad()
        optimizer.zero_grad()

    grad_W = grad_W / num_batches
    return grad_W


def select_and_threshold(pruned_block, Q_t, dW, pruned_indices, pruned_channels, lr, t, num_iter, device):
    with torch.no_grad():
        dW = dW.to(device)
        
        importance_with_S = torch.norm(dW, p=2, dim=(0, 2, 3)) # calculate importance of each input channel to conv2
        importance_sans_S = deepcopy(importance_with_S)
        for index in pruned_indices:
            importance_sans_S[index] = 0

        imp_top_idxs = torch.argsort(importance_sans_S, descending=True)[:pruned_channels] # find s best columns of dW
        imp_top_idxs = set(imp_top_idxs.cpu().tolist())
        
        D_t = imp_top_idxs.union(pruned_indices) # unite best columns of grad with previous active set
        
        # zero out all columns of dW except those in D_t           
        dW_at_D_t = torch.zeros_like(dW) 
        Q_t_at_D_t = torch.zeros_like(Q_t)
        for index in range(dW_at_D_t.shape[1]):
            if index in D_t:
                dW_at_D_t[:, index, :, :] = deepcopy(dW[:, index, :, :])
                Q_t_at_D_t[:, index, :, :] = deepcopy(Q_t[:, index, :, :])

        # update Q_t by gradient descent focused on D_t
        Q_t_at_D_t = Q_t_at_D_t - lr * dW_at_D_t

        for index in range(Q_t.shape[1]):
            if index in D_t:
                Q_t[:, index, :, :] = Q_t_at_D_t[:, index, :, :]

        # find most important neurons in Q_t
        Q_t_imp = torch.norm(Q_t, p=2, dim=(0, 2, 3))
        new_pruned_indices_tensor = torch.argsort(Q_t_imp, descending=True)[:pruned_channels]
        new_pruned_indices = set(new_pruned_indices_tensor.cpu().tolist())

        if new_pruned_indices == pruned_indices:
            print('no change in pruned indices')
        else:
            print('pruned indices changed')
            print(f'new indices: {new_pruned_indices.difference(pruned_indices)}')
            print(f'lost indices: {pruned_indices.difference(new_pruned_indices)}')
        
        pruned_indices = set(new_pruned_indices_tensor.cpu().tolist())
        pruned_indexer = torch.LongTensor(sorted(list(pruned_indices))).to(device)

    return Q_t, pruned_indices, pruned_indexer


def prune_basic_block(args, input_data, og_block, lr=0.01, ratio=0.5, num_iter=20, extra_sgd_epochs=10, bs=128, verbose=False):
    if verbose:
        print(f'\nPruning Settings: ratio {ratio}, iters {num_iter}, data ex {input_data.shape[0]}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    og_block = og_block.to(device)

    # compute size of pruned block and create it
    in_channels = og_block.conv1.in_channels
    out_channels = og_block.conv1.out_channels
    assert in_channels == out_channels
    pruned_channels = int(og_block.conv1.out_channels * ratio)
    pruned_block = PrunedBasicBlock(in_channels, out_channels, out_channels) # for now, pruned_block will be full-sized but will have zero-ed out pruned channels
    pruned_block = pruned_block.to(device)

    # initialize layers of pruned block to match og_block
    pruned_block.conv1.weight.data = deepcopy(og_block.conv1.weight.data)
    pruned_block.bn1.weight.data = deepcopy(og_block.bn1.weight.data)
    pruned_block.bn1.bias.data = deepcopy(og_block.bn1.bias.data)
    pruned_block.conv2.weight.data = deepcopy(og_block.conv2.weight.data)
    pruned_block.bn2.weight.data = deepcopy(og_block.bn2.weight.data)
    pruned_block.bn2.bias.data = deepcopy(og_block.bn2.bias.data)
    
    Q_t = deepcopy(pruned_block.conv2.weight.data) # use Q to track weights

    # compute dense hidden representation
    og_hidden = []
    og_output = []
    for i in range(0, input_data.shape[0], bs):
        mb_input = input_data[i: i + bs, :]
        mb_input = mb_input.to(device)
        with torch.no_grad():
            mb_hid = og_block.conv1(mb_input)
            mb_hid = og_block.bn1(mb_hid)
            mb_hid = og_block.relu(mb_hid)
            mb_out = og_block.conv2(mb_hid) 
            mb_out = og_block.bn2(mb_out)

            og_hidden.append(mb_hid.cpu())
            og_output.append(mb_out.cpu())
    with torch.no_grad():
        og_hidden = torch.cat(og_hidden, dim=0)
        og_output = torch.cat(og_output, dim=0)

    optimizer = torch.optim.SGD(pruned_block.bn2.parameters(), args.bn_lr, momentum=args.momentum, nesterov=True)

    # main pruning loop
    random_indices = RNG.choice(int(pruned_block.conv2.weight.data.shape[1]), pruned_channels, replace=False)
    pruned_indices = set(random_indices)

    pruned_indexer = torch.LongTensor(sorted(list(pruned_indices))).to(device) 

    pruned_block.train()
    with torch.no_grad():
        for index in range(pruned_block.conv2.weight.data.shape[1]):
            if index not in pruned_indices:
                pruned_block.conv2.weight.data[:, index, :, :] = torch.zeros_like(pruned_block.conv2.weight.data[:, index, :, :]) # truncate W_t to top s neurons

    for t in range(num_iter):
        # compute importance with automatic differentiation (chunked into mini-batches)
        dW = compute_grad_and_bn_step(og_hidden, og_output, pruned_block, optimizer, bs, args.num_cs_batches, device)
        Q_t, pruned_indices, pruned_indexer = select_and_threshold(pruned_block, Q_t, dW, pruned_indices, pruned_channels, lr, t, num_iter, device)
        
        # update pruned_block with new pruned indices and new weights for conv2
        with torch.no_grad():
            pruned_block.conv2.weight.data = deepcopy(Q_t)
            for index in range(Q_t.shape[1]):
                if index not in pruned_indices:
                    pruned_block.conv2.weight.data[:, index, :, :] = torch.zeros_like(pruned_block.conv2.weight.data[:, index, :, :]) # truncate W_t to top s neurons

        # take extra sgd steps over the pruned indices
        for e in range(extra_sgd_epochs):
            dW = compute_grad_and_bn_step(og_hidden, og_output, pruned_block, optimizer, bs, args.num_cs_batches, device)

            with torch.no_grad():
                dW = dW.to(device)
                dW_at_S_t = torch.zeros_like(dW) 
                for index in range(dW_at_S_t.shape[1]):
                    if index in pruned_indices:
                        dW_at_S_t[:, index, :, :] = dW[:, index, :, :]
                
                Q_t = Q_t - lr * dW_at_S_t
                pruned_block.conv2.weight.data[:, pruned_indexer, :, :] = Q_t[:, pruned_indexer, :, :]

    # create a new pruned block that only contains the pruned indices
    final_pruned_block = PrunedBasicBlock(in_channels, pruned_channels, out_channels)
    final_pruned_block = final_pruned_block.to(device)

    final_pruned_block.conv1.weight.data = pruned_block.conv1.weight.data[pruned_indexer, :]
    final_pruned_block.bn1.weight.data = pruned_block.bn1.weight.data[pruned_indexer]
    final_pruned_block.bn1.bias.data = pruned_block.bn1.bias.data[pruned_indexer]

    final_pruned_block.conv2.weight.data = pruned_block.conv2.weight.data[:, pruned_indexer, :, :]
    final_pruned_block.bn2.weight.data = pruned_block.bn2.weight.data
    final_pruned_block.bn2.bias.data = pruned_block.bn2.bias.data

    return final_pruned_block


def run_ft(args, model, ft_load, test_load, epochs, lr, criterion=None, optimizer=None, use_lr_sched=False, verbose=False):
    if verbose:
        print(f'\nRunning FT for {epochs} epoch(s), lr {lr}, sched {use_lr_sched}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if criterion is None:
        criterion = CrossEntropyLabelSmooth(args.num_classes).cuda()
    
    if optimizer is None and epochs > 0:
        no_wd_params, wd_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if ".bn" in name or '.bias' in name:
                    no_wd_params.append(param)
                else:
                    wd_params.append(param)
        no_wd_params = nn.ParameterList(no_wd_params)
        wd_params = nn.ParameterList(wd_params)
        optimizer = torch.optim.SGD([
                                    {'params': no_wd_params, 'weight_decay':0.},
                                    {'params': wd_params, 'weight_decay': args.wd},
                                ], lr, momentum=args.momentum, nesterov=True)
    
    tloss, tprec1 = validate(test_load, model, criterion, verbose=True)
    
    if verbose:
        print(f'Epoch 0 Test Loss/Acc: {tloss:.2f}/{tprec1:.2f}')
    
    model = model.to(device)

    test_accs = [tprec1]
    test_losses = [tloss]
    trn_losses = []
    trn_accs = []
    for e in range(epochs):
        if verbose:
            print(f'\nRunning FT Epoch {e+1}/{epochs}')

        # use cosine learning rate schedule
        if use_lr_sched:
            new_lr = 0.5 * lr * (1 + math.cos(math.pi * e / epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            if verbose:
                print(f'Changing LR to: {new_lr}')

        losses = AverageMeter()
        accs = AverageMeter()
        model = model.to(device)
        model.train()
        for i, (inputs, targets) in enumerate(ft_load): 
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            accs.update(prec1.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
        trn_losses.append(losses.avg)
        trn_accs.append(accs.avg)
        tloss, tprec1 = validate(test_load, model, criterion)
        if verbose:
            print(f'Epoch {e+1} Test Loss/Acc: {tloss:.2f}/{tprec1:.2f}')
        test_accs.append(tprec1)
        test_losses.append(tloss)

    metrics = {
        'trn_accs': trn_accs,
        'trn_losses': trn_losses,
        'test_accs': test_accs,
        'test_losses': test_losses,
    }
    return model, metrics


def validate(val_loader, model, criterion, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
    if verbose:
        pruneflops, pruneparams = get_flops_and_params(model)
        print("Prune Flops, Prune Params")
        print(pruneflops, pruneparams)
        print(f'Test Acc.: {top1.avg:.4f}')
        model = model.to(device)
    return losses.avg, top1.avg
     

def prune_rn34():
    parser = argparse.ArgumentParser(description='i-SpaSP prune for rn34')
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--exp-name', type=str, default='ispasp_rn34_prune_00')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--data-path', type=str, default='/data')
    parser.add_argument('--dataset', type=str, default='CIFAR-10')
    parser.add_argument('--pruned-path', type=str, default=None)

    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-cs-batches', type=int, default=5)
    parser.add_argument('--num-cs-iter', type=int, default=20)

    parser.add_argument('--layer1-ratio', type=float, default=0.4)
    parser.add_argument('--layer2-ratio', type=float, default=0.4)
    parser.add_argument('--layer3-ratio', type=float, default=0.4)
    parser.add_argument('--layer4-ratio', type=float, default=1.0)

    parser.add_argument('--layer1-lr', type=float, default=0.1)
    parser.add_argument('--layer2-lr', type=float, default=0.01)
    parser.add_argument('--layer3-lr', type=float, default=0.01)
    parser.add_argument('--layer4-lr', type=float, default=0.01)
    parser.add_argument('--bn-lr', type=float, default=0.01)

    parser.add_argument('--extra-epochs', type=int, default=1)

    parser.add_argument('--block-ft-epochs', type=int, default=1)
    parser.add_argument('--block-ft-lr', type=float, default=1e-2)

    parser.add_argument('--prune-ft-lr', type=float, default=1e-2)
    parser.add_argument('--prune-ft-epochs', type=int, default=90)

    parser.add_argument('--use-lr-sched', action='store_true', default=False)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--prune-last-layer', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.pruned_path is None:
        # prune each layer of ResNet34 with i-SpaSP based on pre-defined ratios
        # for each group of convolutional blocks within the network
        prune_perf_mets = {
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': [],
        }

        prune_load, val_load, _ = get_dataset_ft(args.dataset, args.batch_size,
                    args.workers, args.data_path)
        
        if args.dataset == 'CIFAR-10':
            pruned_model = cifar10_resnet34(pretrained=True)
            args.num_classes = 10
        
        else:
            pruned_model = torchvision.models.resnet34(pretrained=True)
            args.num_classes = 10

        pruned_model = pruned_model.to(device)    

        # validate on test set before pruning
        criterion = CrossEntropyLabelSmooth(args.num_classes).cuda()
        tloss, tprec1 = validate(val_load, pruned_model, criterion, verbose=True)            
        print(f'Pre-pruning: loss = {tloss}, acc = {tprec1}')

        # prune layer1
        if args.layer1_ratio < 1.0:
            print("\npruning layer 1")
            if args.prune_last_layer:
                max_layers = len(pruned_model.layer1)
            else:
                max_layers = len(pruned_model.layer1) - 1 
            for i in range(1, max_layers):
                with torch.no_grad():
                    pruned_model.eval()
                    full_prune_data = []
                    data_iter = iter(prune_load)
                    for b in range(args.num_cs_batches):
                        data_in = next(data_iter)[0].to(device)
                        tmp_prune_data = pruned_model.maxpool(pruned_model.relu(pruned_model.bn1(pruned_model.conv1(data_in))))
                        for lidx in range(i):
                            tmp_prune_data = pruned_model.layer1[lidx](tmp_prune_data) 
                        full_prune_data.append(tmp_prune_data.cpu())
                    full_prune_data = torch.cat(full_prune_data, dim=0)

                assert pruned_model.layer1[i].downsample is None

                print(f"\npruning layer 1, sublayer {i}")
                pruned_model.layer1[i] = prune_basic_block(args, full_prune_data, pruned_model.layer1[i],
                        lr=args.layer1_lr, ratio=args.layer1_ratio, num_iter=args.num_cs_iter, extra_sgd_epochs=args.extra_epochs, bs=args.batch_size,
                        verbose=args.verbose).to(device)

                criterion = CrossEntropyLabelSmooth(args.num_classes).cuda()
                tloss, tprec1 = validate(val_load, pruned_model, criterion, verbose=True)            
                #print(f'After pruning layer 1, sublayer {i}: loss = {tloss}, acc = {tprec1}')

                if args.block_ft_epochs > 0:
                    pruned_model, tmp_met = run_ft(
                            args, pruned_model, prune_load, val_load, args.block_ft_epochs, args.block_ft_lr,
                            use_lr_sched=False, verbose=args.verbose)
                    prune_perf_mets['layer1'].append(tmp_met)
        else:
            print('No pruning on layer1')
         
        # prune layer2
        if args.layer2_ratio < 1.0:
            print("\npruning layer 2")
            if args.prune_last_layer:
                max_layers = len(pruned_model.layer2)
            else:
                max_layers = len(pruned_model.layer2) - 1
            for i in range(1, max_layers):
                with torch.no_grad():
                    pruned_model.eval()
                    full_prune_data = []
                    data_iter = iter(prune_load)
                    for b in range(args.num_cs_batches):
                        data_in = next(data_iter)[0].to(device)

                        # everything before layer being pruned
                        tmp_prune_data = pruned_model.layer1(
                                pruned_model.maxpool(pruned_model.relu(pruned_model.bn1(
                                pruned_model.conv1(data_in)))))
                        for lidx in range(i):
                            tmp_prune_data = pruned_model.layer2[lidx](tmp_prune_data) 
                        full_prune_data.append(tmp_prune_data.cpu())
                    full_prune_data = torch.cat(full_prune_data, dim=0)

                # prune a residual block
                assert pruned_model.layer2[i].downsample is None
                print(f"\npruning layer 2, sublayer {i}")
                pruned_model.layer2[i] = prune_basic_block(args, full_prune_data, pruned_model.layer2[i],
                        lr=args.layer2_lr, ratio=args.layer2_ratio, num_iter=args.num_cs_iter, extra_sgd_epochs=args.extra_epochs, bs=args.batch_size,
                        verbose=args.verbose).to(device)
                
                criterion = CrossEntropyLabelSmooth(args.num_classes).cuda()
                tloss, tprec1 = validate(val_load, pruned_model, criterion, verbose=True)            
                #print(f'After pruning layer 2, sublayer {i}: loss = {tloss}, acc = {tprec1}')

                # fine-tune the pruned block
                if args.block_ft_epochs > 0:
                    pruned_model, tmp_met = run_ft(
                            args, pruned_model, prune_load, val_load, args.block_ft_epochs, args.block_ft_lr,
                            use_lr_sched=False, verbose=args.verbose)
                    prune_perf_mets['layer2'].append(tmp_met)
        else:
            print('No pruning on layer2')
    
        # prune layer3
        if args.layer3_ratio < 1.0:
            print("\npruning layer 3")
            if args.prune_last_layer:
                max_layers = len(pruned_model.layer3)
            else:
                max_layers = len(pruned_model.layer3) - 1
            for i in range(1, max_layers):
                with torch.no_grad():
                    pruned_model.eval()
                    full_prune_data = []
                    data_iter = iter(prune_load)
                    for b in range(args.num_cs_batches):
                        data_in = next(data_iter)[0].to(device)

                        # everything before layer being pruned
                        tmp_prune_data = pruned_model.layer2(pruned_model.layer1(
                                pruned_model.maxpool(pruned_model.relu(pruned_model.bn1(
                                pruned_model.conv1(data_in))))))
                        for lidx in range(i):
                            tmp_prune_data = pruned_model.layer3[lidx](tmp_prune_data) 
                        full_prune_data.append(tmp_prune_data.cpu())
                    full_prune_data = torch.cat(full_prune_data, dim=0)

                # prune a residual block
                assert pruned_model.layer3[i].downsample is None
                print(f"\npruning layer 3, sublayer {i}")
                pruned_model.layer3[i] = prune_basic_block(args, full_prune_data, pruned_model.layer3[i],
                        lr=args.layer3_lr, ratio=args.layer3_ratio, num_iter=args.num_cs_iter, extra_sgd_epochs=args.extra_epochs, bs=args.batch_size,
                        verbose=args.verbose).to(device) 

                criterion = CrossEntropyLabelSmooth(args.num_classes).cuda()
                tloss, tprec1 = validate(val_load, pruned_model, criterion, verbose=True)            
                #print(f'After pruning layer 3, sublayer {i}: loss = {tloss}, acc = {tprec1}')

                if args.block_ft_epochs > 0:
                    pruned_model, tmp_met = run_ft(
                            args, pruned_model, prune_load, val_load, args.block_ft_epochs, args.block_ft_lr,
                            use_lr_sched=False, verbose=args.verbose)
                    prune_perf_mets['layer3'].append(tmp_met)
        else:
            print('No pruning on layer3')
    
        # prune layer4
        if args.layer4_ratio < 1.0:
            print("\npruning layer 4")
            if args.prune_last_layer:
                max_layers = len(pruned_model.layer4)
            else:
                max_layers = len(pruned_model.layer4) - 1
            for i in range(1, max_layers):
                with torch.no_grad():
                    pruned_model.eval()
                    full_prune_data = []
                    data_iter = iter(prune_load)
                    for b in range(args.num_cs_batches):
                        data_in = next(data_iter)[0].to(device)

                        # everything before layer being pruned
                        tmp_prune_data = pruned_model.layer3(pruned_model.layer2(
                                pruned_model.layer1(pruned_model.maxpool(pruned_model.relu(
                                pruned_model.bn1(pruned_model.conv1(data_in)))))))
                        for lidx in range(i):
                            tmp_prune_data = pruned_model.layer4[lidx](tmp_prune_data) 
                        full_prune_data.append(tmp_prune_data.cpu())
                    full_prune_data = torch.cat(full_prune_data, dim=0)

                # prune a residual block
                assert pruned_model.layer4[i].downsample is None
                print(f"\npruning layer 4, sublayer {i}")
                pruned_model.layer4[i] = prune_basic_block(args, full_prune_data, pruned_model.layer4[i],
                        lr=args.layer4_lr, ratio=args.layer4_ratio, num_iter=args.num_cs_iter, extra_sgd_epochs=args.extra_epochs, bs=args.batch_size,
                        verbose=args.verbose).to(device) 
                
                criterion = CrossEntropyLabelSmooth(args.num_classes).cuda()
                tloss, tprec1 = validate(val_load, pruned_model, criterion, verbose=True)            
                #print(f'After pruning layer 4, sublayer {i}: loss = {tloss}, acc = {tprec1}')

                if args.block_ft_epochs > 0:
                    pruned_model, tmp_met = run_ft(
                            args, pruned_model, prune_load, val_load, args.block_ft_epochs, args.block_ft_lr,
                            use_lr_sched=False, verbose=args.verbose)
                    prune_perf_mets['layer4'].append(tmp_met)
        else:
            print('No pruning on layer4')
    
        pruneflops, pruneparams = get_flops_and_params(pruned_model)
        
        pre_ft_results = {
            'model': pruned_model.cpu(),
            'prune_perf_mets': prune_perf_mets,
            'args': args,
        }
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(pre_ft_results, os.path.join(args.save_dir, f'{args.exp_name}_no_ft.pth'))
    else:
        # load pruned model from a checkpoint
        pre_ft_results = torch.load(args.pruned_path)
        pruned_model = pre_ft_results['model']
        prune_perf_mets = pre_ft_results['prune_perf_mets']
    
    if args.prune_ft_epochs > 0:
        criterion = CrossEntropyLabelSmooth(args.num_classes).cuda()
        no_wd_params, wd_params = [], []
        for name, param in pruned_model.named_parameters():
            if param.requires_grad:
                if ".bn" in name or '.bias' in name:
                    no_wd_params.append(param)
                else:
                    wd_params.append(param)
        no_wd_params = nn.ParameterList(no_wd_params)
        wd_params = nn.ParameterList(wd_params)
        optimizer = torch.optim.SGD([
	        {'params': no_wd_params, 'weight_decay':0.},
		    {'params': wd_params, 'weight_decay': args.wd},
        ], args.prune_ft_lr, momentum=args.momentum, nesterov=True)

        # run final fine tuning and find metrics
        pruned_model, metrics = run_ft(args, pruned_model, prune_load, val_load, args.prune_ft_epochs, args.prune_ft_lr,
            criterion=criterion, optimizer=optimizer, use_lr_sched=args.use_lr_sched, verbose=True)
        pruneflops, pruneparams = get_flops_and_params(pruned_model)
        prune_data = {
            'prune_flops': pruneflops,
            'prune_params': pruneparams,
        }

        # save the results
        all_results = {
            'model': pruned_model.cpu(),
            'perf_mets': metrics,
            'prune_mets': prune_data,
            'args': args,
        }
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(all_results, os.path.join(args.save_dir, f'{args.exp_name}.pth'))


if __name__=='__main__':
    prune_rn34()
