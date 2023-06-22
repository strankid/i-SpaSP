# run i-SpaSP pruning on resnet34 

import os

GPU = 1
SAVE_DIR = './results/'
EXP_NUM = '00'
DATA_PATH = '/data/cifar10' # put path to dataset here
BATCH_SIZE = 64
CS_BATCHES = 50
CS_ITER = 200
RATIO = 0.3
EXTRA_EPOCHS = 1
PRUNE_FT_ES = 0
PRUNE_FT_LR = 0.05
LR = 0.05
PRUNED_PATH = None # path to pruned model checkpoint
USE_LR_SCHED = True
VERBOSE = False 
DATASET = 'CIFAR-10'
DATASET_NAME = 'cifar10'
MODEL_FILE = 'cifar_net_30e.pth'


EXP_NAME = f'prune_and_train_mlp_{DATASET_NAME}_{EXP_NUM}'
command = (
    'CUDA_LAUNCH_BLOCKING=1 '
    f'CUDA_VISIBLE_DEVICES={GPU} python prune_hidden_network.py --save-dir {SAVE_DIR} '
    f'--exp-name {EXP_NAME} --data-path {DATA_PATH} --batch-size {BATCH_SIZE} '
    f'--num-cs-batches {CS_BATCHES} --num-cs-iter {CS_ITER} --pruning-ratio {RATIO} '
    f'--extra-epochs {EXTRA_EPOCHS} --ft-epochs {PRUNE_FT_ES} --ft-lr {PRUNE_FT_LR} '
    f'--dataset {DATASET} --lr {LR} --model-path {MODEL_FILE}')

if PRUNED_PATH is not None:
    command += f' --pruned-path {PRUNED_PATH}'
if USE_LR_SCHED:
    command += ' --use-lr-sched'
if VERBOSE:
    command += ' --verbose'
os.system(command) 
