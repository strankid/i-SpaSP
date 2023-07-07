# run i-SpaSP pruning on resnet34 

import os

GPU = 1
SAVE_DIR = './results/'
EXP_NUM = 'x'
DATA_PATH = '/data/cifar10' # put path to dataset here
RANDOM_SEED = 2
BATCH_SIZE = 64
CS_BATCHES = 10
CS_ITER = 50
RATIO = 0.3
EXTRA_EPOCHS = 0
PRUNE_FT_ES = 0
PRUNE_FT_LR = 0.05
LR = 0.05
MOMENTUM = 0.9
PRUNED_PATH = None # path to pruned model checkpoint
USE_ADAPTIVE_LR = True
VERBOSE = False 
DATASET = 'CIFAR-10'
DATASET_NAME = 'cifar10'
MODEL_FILE = 'cifar_net_30e.pth'


EXP_NAME = f'prune_and_train_mlp_acc_{DATASET_NAME}_{EXP_NUM}'
command = (
    'CUDA_LAUNCH_BLOCKING=1 '
    f'CUDA_VISIBLE_DEVICES={GPU} python prune_hidden.py --save-dir {SAVE_DIR} '
    f'--exp-name {EXP_NAME} --data-path {DATA_PATH} --batch-size {BATCH_SIZE} '
    f'--num-cs-batches {CS_BATCHES} --num-cs-iter {CS_ITER} --pruning-ratio {RATIO} '
    f'--extra-epochs {EXTRA_EPOCHS} --ft-epochs {PRUNE_FT_ES} --ft-lr {PRUNE_FT_LR} '
    f'--dataset {DATASET} --lr {LR} --momentum {MOMENTUM} --model-path {MODEL_FILE} --random-seed {RANDOM_SEED} ')

if PRUNED_PATH is not None:
    command += f' --pruned-path {PRUNED_PATH}'
if USE_ADAPTIVE_LR:
    command += ' --use-adaptive-lr'
if VERBOSE:
    command += ' --verbose'
os.system(command) 
