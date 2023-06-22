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

from lib.data import get_dataset_ft
from lib.utils import accuracy, AverageMeter

RANDOM_SEED = 0
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RNG = npr.default_rng(seed=RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class MLP(nn.Module):
    def __init__(self, d_in=3072, d_hid=256, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid, bias=False)
        self.activation = nn.ReLU() # TODO: change activation
        self.fc2 = nn.Linear(d_hid, d_out, bias=False)
        
    def forward(self, x):
        out = self.fc1(x) # fc1 == W0
        out = self.activation(out)
        out = self.fc2(out) # fc2 == W1
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


def residual_objective(mat):
    return 0.5 * torch.mean(mat**2)


# compute the gradient of the loss wrt the weights of the second layer
def compute_grad(pruned_model, og_hidden, og_out, batch_size, num_batches, device):
    grad_W = None

    for i in range(0, og_hidden.shape[0], batch_size):
        mb_hid = og_hidden[i:i + batch_size, :]
        mb_hid = mb_hid.to(device)
        og_output = og_out[i:i + batch_size, :].to(device)
        
        pruned_model.fc2.weight.requires_grad = True # track gradient on layer weights to determine importance

        pruned_output = pruned_model.fc2(mb_hid)
        
        residual = residual_objective(og_output.detach() - pruned_output) 
        residual.backward()

        with torch.no_grad():
            U = og_output
            H_dense = mb_hid
            W = pruned_model.fc2.weight.data
            dW = -1 / torch.numel(U) * (U - H_dense @ W.T).T @ H_dense

        with torch.no_grad():
            if grad_W is None:
                grad_W = pruned_model.fc2.weight.grad.detach().cpu()
            else:
                grad_W += pruned_model.fc2.weight.grad.detach().cpu()

        mb_hid.grad = None
        #og_model.zero_grad()
        pruned_model.zero_grad()
    
    grad_W = grad_W / num_batches
    return grad_W


# find most important neurons, make a gradient step, then threshold
def select_and_threshold(args, pruned_model, Q_t, grad_W, pruned_indices, d_hid_pruned, t, num_iter, device):
    with torch.no_grad():
        grad_W = grad_W.to(device)

        importance_with_S = torch.norm(grad_W, p=2, dim=0) # calculate importance of each hidden neuron
        importance_sans_S = deepcopy(importance_with_S)
        for index in pruned_indices:
            importance_sans_S[index] = 0

        imp_top_idxs = torch.argsort(importance_sans_S, descending=True)[:d_hid_pruned] # find s best columns of grad_W
        imp_top_idxs = set(imp_top_idxs.cpu().tolist())
        
        D_t = imp_top_idxs.union(pruned_indices) # unite best columns of grad with previous active set

        # zero out all columns of grad_W except those in D_t 
        dW_at_D_t = torch.zeros_like(grad_W) 
        Q_t_at_D_t = torch.zeros_like(Q_t)
        for index in range(dW_at_D_t.shape[1]):
            if index in D_t:
                dW_at_D_t[:, index] = grad_W[:, index]
                Q_t_at_D_t[:, index] = Q_t[:, index]

        # update W_t by gradient descent focused on D_t
        Q_t_at_D_t = Q_t_at_D_t - args.lr * dW_at_D_t

        if args.use_lr_sched:
            args.lr = 0.5 * args.lr * (1 + math.cos(math.pi * t / num_iter))
            print(f'new lr: {args.lr}')

        for index in range(Q_t.shape[1]):
            if index in D_t:
                Q_t[:, index] = Q_t_at_D_t[:, index]

        # find most important neurons in Q_t
        Q_t_imp = torch.norm(Q_t, p=2, dim=0)
        new_pruned_indices_tensor = torch.argsort(Q_t_imp, descending=True)[:d_hid_pruned]            
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


def prune_hidden_neurons(args, input_data, og_model, ratio=0.5, num_iter=20, bs=128, gd_iters=0, verbose=False, val_data=None):
    if verbose:
        print(f'\nPruning Settings: ratio {ratio}, iters {num_iter}, data ex {input_data.shape[0]}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    og_model = og_model.to(device)

    # create pruned_model
    d_in = og_model.fc1.in_features
    d_hid = og_model.fc1.out_features
    d_out = og_model.fc2.out_features

    d_hid_pruned = int(d_hid * ratio)
    print(f'num hidden neurons before pruning: {d_hid}')
    print(f'num hidden neurons after pruning: {d_hid_pruned}')

    pruned_model = MLP(d_in, d_hid, d_out) # for now, pruned_model will be full-sized but will have zero-ed out pruned channels

    # initialize layers of pruned model to match og_model
    pruned_model.fc1.weight.data = deepcopy(og_model.fc1.weight.data)
    pruned_model.fc2.weight.data = deepcopy(og_model.fc2.weight.data)

    pruned_model = pruned_model.to(device)        

    Q_t = deepcopy(pruned_model.fc2.weight.data) # use Q to track weights

    # compute dense hidden representation
    og_hidden = []
    og_output = []

    input_data = input_data.flatten(start_dim=1)
    for i in range(0, input_data.shape[0], bs):
        mb_input = input_data[i: i + bs, :]
        mb_input = mb_input.to(device)
        with torch.no_grad():
            mb_hid = og_model.fc1(mb_input)
            mb_hid = og_model.activation(mb_hid)
            mb_out = og_model.fc2(mb_hid)
            og_hidden.append(mb_hid.cpu())
            og_output.append(mb_out.cpu())
    with torch.no_grad():
        og_hidden = torch.cat(og_hidden, dim=0)
        og_output = torch.cat(og_output, dim=0)

    # choose random pruned indices to start
    random_indices = RNG.choice(int(pruned_model.fc2.weight.data.shape[1]), d_hid_pruned, replace=False)
    pruned_indices = set(random_indices)
    pruned_indexer = torch.LongTensor(sorted(list(pruned_indices))).to(device) 

    # set randomly chosen neurons to 0
    with torch.no_grad():
        for index in range(d_hid):
            if index not in pruned_indices:
                pruned_model.fc2.weight.data[:, index] = torch.zeros_like(pruned_model.fc2.weight.data[:, index]) # truncate W_t to top s neurons

    for t in range(num_iter):
        # compute validation accuracy of current model
        if val_data is not None:
            with torch.no_grad():
                criterion = CrossEntropyLabelSmooth(10).cuda()                                                                                                                 
                tloss, tprec1 = validate(val_data, pruned_model, criterion)            
                print(f'Pruning, epoch {t}: loss = {tloss}, acc = {tprec1}')

        grad_W = compute_grad(pruned_model, og_hidden, og_output, bs, args.num_cs_batches, device)
        Q_t, pruned_indices, pruned_indexer = select_and_threshold(args, pruned_model, Q_t, grad_W, pruned_indices, d_hid_pruned, t, num_iter, device)
        
        # update pruned_block with new pruned indices and new weights for conv2
        with torch.no_grad(): 
            pruned_model.fc2.weight.data = deepcopy(Q_t)
            for index in range(Q_t.shape[1]):
                if index not in pruned_indices:
                    pruned_model.fc2.weight.data[:, index] = torch.zeros_like(pruned_model.fc2.weight.data[:, index]) # truncate W_t to top s neurons

        for e in range(gd_iters):
            grad_W = compute_grad(pruned_model, og_hidden, og_output, bs, args.num_cs_batches, device)
            grad_W = grad_W.to(device)

            # zero out grads corresponding to pruned neurons
            dW_at_S_t = torch.zeros_like(grad_W) 
            for index in range(dW_at_S_t.shape[1]):
                if index in pruned_indices:
                    dW_at_S_t[:, index] = grad_W[:, index]

            # make a gradient step
            Q_t = Q_t - args.ft_lr * dW_at_S_t

            # update pruned_model
            pruned_model.fc2.weight.data[:, pruned_indexer] = Q_t[:, pruned_indexer]

    # create a new pruned block that only contains the pruned indices
    final_pruned_block = MLP(d_in, d_hid_pruned, d_out)
    final_pruned_block = final_pruned_block.to(device)

    final_pruned_block.fc1.weight.data =  pruned_model.fc1.weight.data[pruned_indexer, :]
    final_pruned_block.fc2.weight.data = pruned_model.fc2.weight.data[:, pruned_indexer]

    return final_pruned_block


def run_ft(args, model, epochs, lr, criterion=None, optimizer=None, use_lr_sched=False, verbose=False):
    if verbose:
        print(f'\n\nRunning FT for {epochs} epoch(s), lr {lr}, sched {use_lr_sched}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ft_load, test_load, n_class = get_dataset_ft(args.dataset, args.batch_size,
            args.workers, args.data_path)
    
    if criterion is None:
        criterion = CrossEntropyLabelSmooth(10).cuda()
    
    if optimizer is None:
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

    test_accs = [tprec1]
    test_losses = [tloss]
    trn_losses = []
    trn_accs = []
    for e in range(epochs):
        if verbose:
            print(f'Running FT Epoch {e+1}/{epochs}')

        # use cosine learning rate schedule
        if use_lr_sched:
            new_lr = 0.5 * lr * (1 + math.cos(math.pi * e / epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            if verbose:
                print(f'\n\nChanging LR to: {new_lr}\n\n')

        losses = AverageMeter()
        accs = AverageMeter()
        model = model.to(device)
        model.train()
        for i, (inputs, targets) in enumerate(ft_load): 
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.flatten(start_dim=1)

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
            print(f'\n\nEpoch {e+1} Test Loss/Acc: {tloss:.2f}/{tprec1:.2f}\n')
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
            inputs = inputs.flatten(start_dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
    if verbose:
        print(f'Test Acc.: {top1.avg:.4f}')
    return losses.avg, top1.avg
     

def prune_mlp():
    print(f'random seed: {RANDOM_SEED}')
    parser = argparse.ArgumentParser(description='prune + train for mlp')
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--exp-name', type=str, default='prutra_mlp_00')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--data-path', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-cs-batches', type=int, default=5)
    parser.add_argument('--num-cs-iter', type=int, default=20)
    parser.add_argument('--pruning-ratio', type=float, default=0.4)
    parser.add_argument('--ft-lr', type=float, default=1e-2)
    parser.add_argument('--ft-epochs', type=int, default=90)
    parser.add_argument('--use-lr-sched', action='store_true', default=False)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--pruned-path', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='CIFAR-10')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--model-path', type=str, default='./models_cifar10/state_dicts/cifar_net.pth')
    parser.add_argument('--extra-epochs', type=int, default=0)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

    if args.pruned_path is None:
        # prune the hidden layer of a 2-layer MLP with Pruning+Training based on pre-defined ratios

        prune_load, val_load, _ = get_dataset_ft(args.dataset, args.batch_size,
                    args.workers, args.data_path)
        
        assert args.dataset == 'CIFAR-10'

        model = MLP(d_hid=256)
        state_dict = torch.load(SCRIPT_DIR + "/models_cifar10/state_dicts/" + args.model_path)
        model.load_state_dict(state_dict)
        model = model.to(device)    
        
        criterion = CrossEntropyLabelSmooth(10).cuda()                                                                                                                 
        tloss, tprec1 = validate(val_load, model, criterion)            
        print(f'Pre-pruning: loss = {tloss}, acc = {tprec1}')

        # prune hidden layer
        if args.pruning_ratio < 1.0:
            with torch.no_grad():
                full_prune_data = []
                data_iter = iter(prune_load)
                for b in range(args.num_cs_batches):
                    data_in = next(data_iter)[0].to(device)
                    tmp_prune_data = data_in
                    full_prune_data.append(tmp_prune_data.cpu())
                full_prune_data = torch.cat(full_prune_data, dim=0)

            pruned_model = prune_hidden_neurons(args, full_prune_data, model,
                    args.pruning_ratio, args.num_cs_iter, args.batch_size, gd_iters=args.extra_epochs,
                    verbose=args.verbose, val_data=val_load).to(device)
            
        else:
            print('No pruning done, using full model')
            
        pre_ft_results = {
            'model': pruned_model.cpu(),
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

    if args.ft_epochs > 0:
        criterion = CrossEntropyLabelSmooth(10).cuda()
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
	        {'params': no_wd_params, 'weight_decay': 0.},
		    {'params': wd_params, 'weight_decay': args.wd},
        ], args.ft_lr, momentum=args.momentum, nesterov=True)

        # run final fine tuning and find metrics
        pruned_model, metrics = run_ft(args, pruned_model, args.ft_epochs, args.ft_lr,
            criterion=criterion, optimizer=optimizer, use_lr_sched=args.use_lr_sched, verbose=args.verbose)

        # save the results
        all_results = {
            'model': pruned_model.cpu(),
            'perf_mets': metrics,
            'args': args,
        }
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(all_results, os.path.join(args.save_dir, f'{args.exp_name}.pth'))


if __name__=='__main__':
    prune_mlp()
