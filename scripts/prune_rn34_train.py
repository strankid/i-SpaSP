# run i-SpaSP pruning on resnet34 

import os

GPU = 1
SAVE_DIR = './results/'
EXP_NUM = '00'
DATA_PATH = '/data/cifar10' # put path to dataset here
BATCH_SIZE = 256
CS_BATCHES = 20
CS_ITER = 20
RATIOS = [0.4, 0.4, 0.4, 1.0]

EXTRA_EPOCHS = 10
LRS = [0.1, 0.01, 0.01, 0.01]
BN_LR = 0.01

BLOCK_FT_ES = 0
BLOCK_FT_LR = 0.01

PRUNE_FT_ES = 0
PRUNE_FT_LR = 0.01

PRUNED_PATH = None # path to pruned model checkpoint
USE_LR_SCHED = True
PRUNE_LAST_LAYER = False
VERBOSE = False 
DATASET = 'CIFAR-10'
DATASET_NAME = 'cifar10'


EXP_NAME = f'ispasp_train_rn34_{DATASET_NAME}_{EXP_NUM}'
command = (
    'CUDA_LAUNCH_BLOCKING=1 '
    f'CUDA_VISIBLE_DEVICES={GPU} python prune_resnet34_ispasp_train.py --save-dir {SAVE_DIR} '
    f'--exp-name {EXP_NAME} --data-path {DATA_PATH} --batch-size {BATCH_SIZE} '
    f'--num-cs-batches {CS_BATCHES} --num-cs-iter {CS_ITER} --layer1-ratio {RATIOS[0]} '
    f'--layer2-ratio {RATIOS[1]} --layer3-ratio {RATIOS[2]} --layer4-ratio {RATIOS[3]} '
    f'--extra-epochs {EXTRA_EPOCHS} --block-ft-epochs {BLOCK_FT_ES} --block-ft-lr {BLOCK_FT_LR} '
    f'--layer1-lr {LRS[0]} --layer2-lr {LRS[1]} --layer3-lr {LRS[2]} --layer4-lr {LRS[3]} '
    f'--bn-lr {BN_LR} --prune-ft-epochs {PRUNE_FT_ES} --prune-ft-lr {PRUNE_FT_LR} '
    f'--dataset {DATASET} ')

if PRUNE_LAST_LAYER:
    command += f' --prune-last-layer'
if PRUNED_PATH is not None:
    command += f' --pruned-path {PRUNED_PATH}'
if USE_LR_SCHED:
    command += ' --use-lr-sched'
if VERBOSE:
    command += ' --verbose'
os.system(command) 

