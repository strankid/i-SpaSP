# run i-SpaSP pruning on resnet34 

import os

GPU = 1
SAVE_DIR = './results/'
EXP_NUMS = []
DATA_PATH = '/data/cifar10' # put path to dataset here
RANDOM_SEEDS = [0, 1, 2, 3, 4]
BATCH_SIZE = 64
CS_BATCHES = 10
CS_ITER = 50
RATIO = 0.3
EXTRA_EPOCHS = [0, 1, 5, 10]
PRUNE_FT_ES = 0
PRUNE_FT_LR = 3.3e-6
LR = 3.3e-6
PRUNED_PATH = None # path to pruned model checkpoint
USE_ADAPTIVE_LR = True
VERBOSE = False 
MOMENTUM = 0.9
DATASET = 'CIFAR-10'
DATASET_NAME = 'cifar10'
MODEL_FILE = 'cifar_net_30e.pth'

exp_num = 0
for extra_epoch in EXTRA_EPOCHS:
    for seed in RANDOM_SEEDS:
        EXP_NAME = f'prune_and_train_mlp_acc_{DATASET_NAME}_{exp_num}'
        print(EXP_NAME)
        command = (
            'CUDA_LAUNCH_BLOCKING=1 '
            f'CUDA_VISIBLE_DEVICES={GPU} python prune_hidden_network_acc.py --save-dir {SAVE_DIR} '
            f'--exp-name {EXP_NAME} --data-path {DATA_PATH} --batch-size {BATCH_SIZE} '
            f'--num-cs-batches {CS_BATCHES} --num-cs-iter {CS_ITER} --pruning-ratio {RATIO} '
            f'--extra-epochs {extra_epoch} --ft-epochs {PRUNE_FT_ES} --ft-lr {PRUNE_FT_LR} '
            f'--dataset {DATASET} --lr {LR} --momentum {MOMENTUM} --model-path {MODEL_FILE} --random-seed {seed} ')

        if PRUNED_PATH is not None:
            command += f' --pruned-path {PRUNED_PATH}'
        if USE_ADAPTIVE_LR:
            command += ' --use-adaptive-lr'
        if VERBOSE:
            command += ' --verbose'
        os.system(command) 
        exp_num += 1