import logging
import os
import random
import sys
import copy
import math
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
import time
import json

import datasets
import numpy as np
from datasets import load_dataset, load_metric

from thop import profile

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BertTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions
)
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.trainer_utils import SchedulerType
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import evaluate
import torch

import torch.nn as nn

import code

check_min_version("4.16.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)

## TRAINING ARGUMENT CLASSES

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    language: str = field(
        default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
    )
    train_language: Optional[str] = field(
        default=None, metadata={"help": "Train language if it is different from the evaluation language."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

## PRUNING AGORITHM STEPS

## STEP 1: COMPUTE GRADIENT


def compute_grads(args, dense_out_lin, pruned_out_lin, H, weight_indexer, device='cpu'):
  pruned_out_lin.zero_grad()
  dense_out_lin.zero_grad()
  H.grad = None

  dense_out_lin = dense_out_lin.to(device)
  U = dense_out_lin(H)

  if weight_indexer is not None:
    with torch.no_grad():
      U_pruned = pruned_out_lin(H[:, :, weight_indexer])
    loss = residual_objective(U - U_pruned.detach())
  else:
    loss = residual_objective(U)

  loss.backward()

  dW = dense_out_lin.weight.grad

  # maybe instead of using attn grad, should aggregate parameter grads? either for weights only or for all
  pruned_out_lin.zero_grad()
  dense_out_lin.zero_grad()
  H.grad = None
  U.grad = None

  return dW, loss.item()

## STEP 2: FIND BEST S COLUMNS OF GRAD OUTSIDE S AND MERGE WITH S

@torch.no_grad()
def find_and_merge(dW, S, n_heads_og, dim_per_head, n_heads_to_keep, device='cpu'):
  dW_by_head = dW.view(dW.shape[0], n_heads_og, dim_per_head).to(device)
  importance = torch.norm(dW_by_head, p=2, dim=(0, 2))
  # importance = torch.sum(dW_by_head, dim=(0, 2))

  for index in S:
    importance[index] = 0
  imp_top_idxs = torch.argsort(importance, descending=True)[:n_heads_to_keep]
  imp_top_idxs = set(imp_top_idxs.tolist())

  D = S.union(imp_top_idxs)
  # print(f'top importance: {imp_top_idxs}, S: {S}, D: {D}')

  return D

## STEP 3: UPDATE PARAMETERS BY GRADIENT DESCENT FOCUSED ON D

@torch.no_grad()
def update_step(args, Q, dW_dense, weight_indexer, H, device=None):
  if weight_indexer is not None:
    dW_pruned = -1 * dW_dense[:, weight_indexer] # because gradient wrt pruned weights at S_t should be -1 * gradient wrt dense weights

    eta = args.eta
    if args.eta == 'adaptive':
      eta = torch.norm(dW_pruned) ** 2 / torch.norm(H[:, :, weight_indexer] @ dW_pruned.T) ** 2
    Q[:, weight_indexer] = Q[:, weight_indexer] - eta * dW_pruned

  return Q

## STEP 4: TRUNCATE Q TO BE S SPARSE

@torch.no_grad()
def truncate(args, Q, pruned_out_lin, h, D, n_heads_og, dim_per_head, n_heads_to_keep, dW, device='cpu'):
  if args.trunc_strategy == 'magnitude':
    Q_by_head = Q.view(Q.shape[0], n_heads_og, dim_per_head).to(device)
    imp = torch.norm(Q_by_head, dim=(0, 2)) # get the importance of each head

  elif args.trunc_strategy == 'attn_weights':
    imp = h # get the importance of each head

  elif args.trunc_strategy == 'dense_grad':
    dW_by_head = dW.view(dW.shape[0], n_heads_og, dim_per_head)
    imp = torch.norm(dW_by_head, p=2, dim=(0, 2))

  else:
    raise NotImplementedError()

  if args.choose_from_D:
    for index in range(imp.shape[0]):
      if index not in D:
        imp[index] = 0

  imp_top_idxs = torch.argsort(imp, descending=True)[:n_heads_to_keep]
  S = set(imp_top_idxs.cpu().tolist())

  S_indexer = get_weight_indexer(S, dim_per_head)
  pruned_out_lin.weight.data = Q[:, S_indexer]

  return pruned_out_lin, S, S_indexer

## STEP 5: DEBIAS

def debias(args, pruned_out_lin, dense_out_lin, Q, H, S_indexer, iters, device='cpu'):
  H_pruned = H[:, :, S_indexer]

  for i in range(iters):
    # print(f'debias iter {i}')
    pruned_out_lin.zero_grad()
    dense_out_lin.zero_grad()

    if args.debias_dense:
      U_dense = dense_out_lin(H)
      with torch.no_grad():
        U_pruned = pruned_out_lin(H_pruned)
    else:
      with torch.no_grad():
        U_dense = dense_out_lin(H)
      U_pruned = pruned_out_lin(H_pruned)

    if args.debias_dense:
      loss = residual_objective(U_dense - U_pruned.detach())
    else:
      loss = residual_objective(U_dense.detach() - U_pruned)

    loss.backward()

    with torch.no_grad():
      if args.debias_dense:
        dW = dense_out_lin.weight.grad
        if args.eta == 'adaptive':
          eta = torch.norm(dW) ** 2 / torch.norm(H @ dW.T) ** 2
        else:
          eta = args.eta

        Q = Q - eta * dW
        pruned_out_lin.weight.data = Q[:, S_indexer]

      else:
        dW_pruned = pruned_out_lin.weight.grad

        if args.eta == 'adaptive':
          eta = torch.norm(dW_pruned) ** 2 / torch.norm(H_pruned @ dW_pruned.T) ** 2
        else:
          eta = args.eta

        # print(f'debias grad: {dW_pruned}')
        # print(f'debias weight: {pruned_out_lin.weight.data}')

        pruned_out_lin.weight.data = pruned_out_lin.weight.data - eta * dW_pruned
        Q[:, S_indexer] = pruned_out_lin.weight.data

  return pruned_out_lin, Q

## UTILS 

## ISPASP

def heads_to_indexer(head_list, head_dim):
    indices = []
    for i in head_list:
        indices.extend(list(range(head_dim*i, head_dim*(i + 1))))
    indices = sorted(indices)
    return torch.LongTensor(indices)

def get_bert_embeddings(model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iids = data['input_ids'].to(device)
    ttids = data['token_type_ids'].to(device)
    att_mask = data['attention_mask'].to(device)
    ext_att_mask = model.bert.get_extended_attention_mask(att_mask, iids.size(), device)
    #head_mask = model.bert.get_head_mask(None, model.bert.config.num_hidden_layers)
    embedding_output = model.bert.embeddings(input_ids=iids, token_type_ids=ttids)
    return embedding_output, ext_att_mask

def residual_objective_ispasp(mat):
    return 0.5 * torch.sum(mat**2)


## GENERAL

class dotdict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, dotdict(value))
            else:
                setattr(self, key, value)

def load_pretrn(ckpt_path, model):
    print(f'\n\nLoading model from: {ckpt_path}\n\n')
    #CONFIG_NAME = 'config.json'
    WEIGHTS_NAME = 'pytorch_model.bin'
    #cfg_path = os.path.join(ckpt_path, CONFIG_NAME)
    weight_path = os.path.join(ckpt_path, WEIGHTS_NAME)
    assert os.path.exists(weight_path)
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict
    return model

def get_transformer_arguments(model, data, device='cpu'):
  model = model.to(device)
  input_ids = data['input_ids'].to(device)
  with torch.no_grad():
    embeddings = model.bert.embeddings(input_ids)

  input_shape = input_ids.size()

  attention_mask = data['attention_mask']

  if attention_mask is None:
      attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

  return embeddings, attention_mask.view(attention_mask.shape[0], 1, 1, attention_mask.shape[1])

## PRUNING 

# separate heads
def separate_heads(x, bs, n_heads, dim_per_head):
  return x.view(bs, -1, n_heads, dim_per_head).transpose(1, 2)

def get_weight_indexer(head_list, head_dim):
  indices = []
  for i in head_list:
      indices.extend(list(range(head_dim*i, head_dim*(i + 1))))
  indices = sorted(indices)
  return torch.LongTensor(indices)

def residual_objective(mat):
    return 0.5 * torch.norm(mat) # using torch.sum requires small stepsize (order of 1e-5)

def get_bert_embeddings(model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iids = data['input_ids'].to(device)
    ttids = data['token_type_ids'].to(device)
    att_mask = data['attention_mask'].to(device)
    ext_att_mask = model.bert.get_extended_attention_mask(att_mask, iids.size(), device)
    #head_mask = model.bert.get_head_mask(None, model.bert.config.num_hidden_layers)
    embedding_output = model.bert.embeddings(input_ids=iids, token_type_ids=ttids)
    return embedding_output, ext_att_mask


## DATA

def preprocess_function(examples, tokenizer):
    # Tokenize the texts
    inputs = tokenizer(
      examples['question'],
      examples["context"],
      max_length=512,
      truncation="only_second",
      stride=128,
      padding="max_length",
      return_tensors='pt',
      return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

## FINETUNE

def train_and_eval(args, model, train_dict, val_arguments, epochs, save_path=None, save_model=False, lr=5e-5, lr_decay=True, device='cpu'):
    model.eval()
    val_arguments['model'] = model
    metrics_before_ft = validate(args, **val_arguments, device=device)
    print(f'before finetuning: {metrics_before_ft}')

    if epochs > 0:
      train_dict['args'].learning_rate = lr
      if lr_decay:
          train_dict['args'].lr_scheduler_type = SchedulerType.LINEAR
      else:
          train_dict['args'].lr_scheduler_type = SchedulerType.CONSTANT
      train_dict['args'].num_train_epochs = float(epochs)
      trainer = Trainer(model=model, **train_dict)
      model.train()
      train_result = trainer.train(resume_from_checkpoint=None)
      metrics = train_result.metrics
      max_train_samples = len(train_dict['train_dataset'])
      metrics["train_samples"] = max_train_samples
      if save_path is not None:
          trainer.output_dir = save_path
          trainer.run_name = save_path
          if save_model:
              trainer.save_model()
          trainer.log_metrics("train", metrics)
          trainer.save_metrics("train", metrics)
          trainer.save_state()

    model.eval()
    val_arguments['model'] = model
    metrics_after_ft = validate(args, **val_arguments, device=device)
    print(f'after finetuning: {metrics_after_ft}')
    return model, (metrics_before_ft, metrics_after_ft)

## VALIDATE

def validate(args, model, ids, inputs, answers, metric, tokenizer, device='cpu'):
  model = model.to(device)
  inputs = inputs.to(device)
  with torch.no_grad():
    start_sp = time.time()
    outputs_sp = model(**inputs)
    end_sp = time.time()
    if args.verbose:
      print(f'done predicting using sparse model. time elapsed = {end_sp - start_sp}s')

  num_examples = len(ids)
  preds = []
  refs = []

  for i in range(num_examples):
    answer_start_index_sp = torch.argmax(outputs_sp.start_logits[i])
    answer_end_index_sp = torch.argmax(outputs_sp.end_logits[i])
    predict_answer_tokens_sp = inputs.input_ids[i, answer_start_index_sp : answer_end_index_sp + 1]
    pred_sp = tokenizer.decode(predict_answer_tokens_sp)

    pred = {'id': ids[i], 'prediction_text': pred_sp}
    preds.append(pred)
    ref = {'answers': answers[i], 'id': ids[i]}
    refs.append(ref)

  results = metric.compute(predictions=preds, references=refs)
  return results

def get_val_arguments(args, val_ds, model, tokenizer):
      val_arguments = {}
      start_idx = torch.randint(low=0, high=len(val_ds['id']) - args.val_size, size=(1,)).item()

      val_ids = val_ds['id'][start_idx : start_idx + args.val_size]
      val_questions =  val_ds['question'][start_idx : start_idx + args.val_size]
      val_texts =  val_ds['context'][start_idx : start_idx + args.val_size]
      val_answers =  val_ds['answers'][start_idx : start_idx + args.val_size]

      val_inputs = tokenizer(
            val_questions,
            val_texts,
            max_length=512,
            truncation="only_second",
            stride=128,
            padding="max_length",
            return_tensors='pt'
      )

      val_arguments['ids'] = val_ids
      val_arguments['inputs'] = val_inputs
      val_arguments['answers'] = val_answers
      val_arguments['model'] = deepcopy(model)
      val_arguments['metric'] = evaluate.load(args.dataset_name)
      val_arguments['tokenizer'] = tokenizer

      return val_arguments

## COMPARISONS

## ISPASP

def prune_layer_ispasp(blay, hidden_states_list, mask_list, num_iter=10, prune_ratio=0.5, layer_id=None):
    if layer_id is not None:
      print(f'\nPruning layer {layer_id}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def shape(x):
      """separate heads"""
      return x.view(x.shape[0], -1, blay.attention.self.num_attention_heads, blay.attention.self.attention_head_size).transpose(1, 2)

    def unshape(x):
      """group heads"""
      return x.transpose(1, 2).contiguous().view(x.shape[0], -1, blay.attention.self.num_attention_heads * blay.attention.self.attention_head_size)

    # pruning will occur a the level of the self attention module
    sout = blay.attention.output.dense

    # generate the linear layer used to do pruning stuff
    num_prune_heads = int(blay.attention.self.num_attention_heads*prune_ratio)
    # print(f'Num prune heads: {num_prune_heads}')
    pdense = torch.nn.Linear(
            in_features=int(blay.attention.self.attention_head_size*num_prune_heads),
            out_features=blay.attention.self.all_head_size, bias=True)

    # generate self attention output for each head
    # this acts as the hidden representation used as reference for pruning
    og_size = None
    satt_out = []
    hidden_states = torch.cat(hidden_states_list, dim=0)
    attn_mask = torch.cat(mask_list, dim=0)
    with torch.no_grad():
      attn_mask = attn_mask.to(device)
      hidden_states = hidden_states.to(device)

      _, weights = blay.attention(hidden_states, attn_mask, head_mask=None, output_attentions=True)

      v = shape(blay.attention.self.value(hidden_states))
      context = torch.matmul(weights, v)

      context = context.permute(0, 2, 1, 3).contiguous()
      # satt_out = unshape(context)

      if og_size is None:
          og_size = context.size() # store to resize importances later
      new_shape = context.size()[:-2] + (blay.attention.self.all_head_size,)
      context = context.view(*new_shape) # [32, 128, 768], THIS IS THE OUTPUT OF SELF ATTN
      satt_out.append(context.cpu())

    batch_size = og_size[0]
    with torch.no_grad():
        satt_out = torch.cat(satt_out, dim=0) # [N, 128, 768]
        aggh_shape = (satt_out.size()[0],) + og_size[1:]
        agg_hidden = satt_out.view(*aggh_shape) # [N, 128, 12, 64]
        agg_hidden = torch.sum(agg_hidden, dim=(0, 1, 3))

    # main pruning loop
    pruned_indices = set([])
    weight_indexer = None
    for t in range(num_iter):
        importance = None
        for i in range(0, satt_out.shape[0], batch_size):
            # track gradient on input to the MLP
            so = satt_out[i: i + batch_size, :]
            so = so.to(device)
            so.requires_grad = True

            # compute dense output while tracking gradient
            dense_out = sout(so)
            #out = sout.LayerNorm(out + data), we don't use residual/layernorm here

            # compute the pruning residual
            if len(pruned_indices) > 0:
                with torch.no_grad():
                    pruned_out = pdense(so[:, :, weight_indexer])
                residual = residual_objective(dense_out - pruned_out)
                residual.backward()
            else:
                residual = residual_objective(dense_out)
                residual.backward()

            # compute importance using gradient on the input
            tmp_imp = so.grad.detach().cpu().view(*og_size)
            with torch.no_grad():
                tmp_imp = torch.sum(tmp_imp, dim=(0, 1, 3))
                if importance is None:
                    importance = tmp_imp
                else:
                    importance += tmp_imp

            so.grad = None
            sout.zero_grad()
            pdense.zero_grad()

        # find most important attention heads, merge with previous active set, then threshold
        with torch.no_grad():
            imp_idxs = torch.argsort(importance, descending=True)[:2*num_prune_heads]
            tmp_imp_heads = set(imp_idxs.cpu().tolist())
            bigger_set = tmp_imp_heads.union(pruned_indices)
            indexer = torch.LongTensor(sorted(list(bigger_set)))
            hidden_sizes = agg_hidden[indexer]
            # hidden_sizes = agg_hidden
            new_pruned_indices = torch.argsort(hidden_sizes, descending=True)[:num_prune_heads]
            new_pruned_indices = set(indexer[new_pruned_indices].cpu().tolist())
            # new_pruned_indices = set(new_pruned_indices.cpu().tolist())
            pruned_indices = new_pruned_indices

            # copy weights into the new model
            weight_indexer = heads_to_indexer(pruned_indices, blay.attention.self.attention_head_size)
            pdense.weight.data = sout.weight.data[:, weight_indexer]
            pdense.bias.data = sout.bias.data

    print(f'Post-pruning: S = {pruned_indices}')

    heads_to_prune = [x for x in range(blay.attention.self.num_attention_heads) if not x in pruned_indices]
    blay.attention.prune_heads(heads_to_prune)

    if layer_id is not None:
      print(f'Done pruning layer {layer_id}')
    return blay

## RANDOM
def bert_layer_random_prune(blay, prune_ratio=0.5):
    satt = blay.attention
    num_heads_to_prune = int((1 - prune_ratio)*satt.self.num_attention_heads)
    heads_to_prune = list(range(satt.self.num_attention_heads))
    random.shuffle(heads_to_prune)
    heads_to_prune = heads_to_prune[:num_heads_to_prune]
    blay.attention.prune_heads(heads_to_prune)
    return blay


## GLOBAL MASKING

def prune_global_masking(model, dl, num_batches=10, ratio=0.5):
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_layers = len(model.bert.encoder.layer)
    num_heads = model.bert.encoder.layer[0].attention.self.num_attention_heads
    final_num_heads = int(ratio * num_layers * num_heads)
    mask_tensor = torch.ones(int(num_heads*num_layers))
    mask_tensor = mask_tensor.to(device)
    mask_tensor.requires_grad = True
    for b in range(num_batches):
        data_in = next(iter(dl))
        labels = data_in['labels']
        labels = labels.to(device)
        hidden_rep, ext_att_mask = get_bert_embeddings(model, data_in)

        # pass data thru each layer of the model
        for i in range(num_layers):
            blay = model.bert.encoder.layer[i]
            satt = blay.attention.self
            sout = blay.attention.output

            query_out = satt.transpose_for_scores(satt.query(hidden_rep))
            key_out = satt.transpose_for_scores(satt.key(hidden_rep))
            value_out = satt.transpose_for_scores(satt.value(hidden_rep))

            att_sc = torch.matmul(query_out, key_out.transpose(-1, -2)) # [32, 12, 128, 128], has all heads separated
            att_sc = att_sc / math.sqrt(satt.attention_head_size)
            att_sc += ext_att_mask # contains a bunch of negative infinities to remove stuff in softmax computation
            att_prob = torch.nn.functional.softmax(att_sc, dim=-1)

            ctxt = torch.matmul(att_prob, value_out) # [32, 12, 128, 64] still separated btwn heads
            ctxt = ctxt.permute(0, 2, 1, 3).contiguous() # [32, 128, 12, 64]
            sense_mask = mask_tensor[num_heads*i: num_heads*(i + 1)]

            ctxt = ctxt * sense_mask[None, None, :, None] # add mask into the forward pass

            new_shape = ctxt.size()[:-2] + (satt.all_head_size,)
            ctxt = ctxt.view(*new_shape) # [32, 128, 768], THIS IS THE OUTPUT OF SELF ATTN

            out = sout.dense(ctxt)
            out = sout.LayerNorm(out + hidden_rep)

            hidden_rep = apply_chunking_to_forward(blay.feed_forward_chunk,
                    blay.chunk_size_feed_forward, blay.seq_len_dim, out)

        output = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_rep,
                past_key_values=None, hidden_states=None, attentions=None,
                cross_attentions=None)
        pooled_output = model.bert.pooler(output[0])
        pooled_output_wrapper = BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=output[0], pooler_output=pooled_output,
                past_key_values=None, hidden_states=None, attentions=None,
                cross_attentions=None)
        pool_out = pooled_output_wrapper[1]
        logits = model.classifier(pool_out)
        loss = loss_fn(logits.view(-1, model.num_labels), labels.view(-1))
        loss.backward()

    agg_grad = mask_tensor.grad.detach().cpu()
    with torch.no_grad():
        agg_grad = torch.abs(agg_grad)
    imp_idxs = set(torch.argsort(agg_grad, descending=True)[:final_num_heads].cpu().tolist())
    for l in range(num_layers):
        heads_to_prune = []
        for hi in range(num_heads):
             curr_idx = num_heads * l + hi
             if not curr_idx in imp_idxs:
                 heads_to_prune.append(hi)
        assert len(heads_to_prune) < num_heads
        model.bert.encoder.layer[l].attention.prune_heads(heads_to_prune)

    return model

## PRUNE MODEL

def prune_bert(args, model, dl, num_batches, num_iter, ratios, training_dict, val_arguments, per_layer_epochs, final_epochs, prune_type='ispasp++'):
    # performs i-SpaSP pruning on the BERT model for squad
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_input = next(iter(dl))
    test_input_ids = test_input['input_ids'][0].to(device)

    input_shape = test_input_ids.size()
    test_attention_mask = test_input['attention_mask'][0]
    if test_attention_mask is None:
        test_attention_mask = torch.ones(input_shape, device=device)

    # pre_flops, pre_params = profile(model, inputs=(test_input_ids, test_attention_mask), verbose=False)
    # print(f'before, flops: {pre_flops}, params: {pre_params}')

    n_params_before = model.num_parameters(only_trainable=True)
    print(f'Number of parameters before pruning: {n_params_before}')

    # get the output dir
    save_base = training_dict['args'].output_dir

    transformer = model.bert.encoder
    layers = transformer.layer

    all_metrics = {}
    metrics = validate(args, **val_arguments, device=device)
    print(f'Pre-pruning metrics = {metrics}')
    all_metrics['pre-pruning'] = metrics

    # start passing thru each layer of encoder
    if prune_type in ['random', 'ispasp', 'ispasp++']:
        for i, rat in enumerate(ratios):
            if i < args.min_layer or i > args.max_layer:
              continue
            if prune_type == 'ispasp++':
                data_list = []
                mask_list = []
                for b in range(num_batches):
                    data_in = next(iter(dl))
                    with torch.no_grad():
                        model.eval()
                        hidden_rep, ext_att_mask = get_transformer_arguments(model, data_in, device)
                        hidden_rep, ext_att_mask = hidden_rep.to(device), ext_att_mask.to(device)
                        # hidden_rep, ext_att_mask = get_bert_embeddings(model, data_in)
                        for j in range(i):
                            hidden_rep = layers[j](hidden_rep, ext_att_mask)[0]
                            #hidden_rep = bert_layer_forward(model.bert.encoder.layer[j],
                            #        hidden_rep, ext_att_mask)
                        data_list.append(hidden_rep.detach().cpu())
                        mask_list.append(ext_att_mask.detach().cpu())
                layers[i].attention = prune_attn_layer(args, layers[i].attention, data_list, mask_list, val_arguments, prune_ratio=rat, layer_id=i, device=device)
            elif prune_type == 'ispasp':
                data_list = []
                mask_list = []
                for b in range(num_batches):
                    data_in = next(iter(dl))
                    with torch.no_grad():
                        model.eval()
                        hidden_rep, ext_att_mask = get_transformer_arguments(model, data_in, device)
                        hidden_rep, ext_att_mask = hidden_rep.to(device), ext_att_mask.to(device)
                        # hidden_rep, ext_att_mask = get_bert_embeddings(model, data_in)
                        for j in range(i):
                            hidden_rep = layers[j](hidden_rep, ext_att_mask)[0]
                            #hidden_rep = bert_layer_forward(model.bert.encoder.layer[j],
                            #        hidden_rep, ext_att_mask)
                        data_list.append(hidden_rep.detach().cpu())
                        mask_list.append(ext_att_mask.detach().cpu())
                layers[i] = prune_layer_ispasp(layers[i], data_list, mask_list, num_iter=args.iters, prune_ratio=rat, layer_id=i)
                # layers[i].attention = prune_attn_layer(args, layers[i].attention, data_list, mask_list, val_arguments, prune_ratio=rat, layer_id=i, device=device)
            elif prune_type == 'random':
                layers[i] = bert_layer_random_prune(
                        layers[i], prune_ratio=rat)
            else:
                raise NotImplementedError()
            output_dir = os.path.join(save_base, f'prune_layer_{i}/')
            model, layer_metrics = train_and_eval(args, model, training_dict, val_arguments, per_layer_epochs,
                    save_path=output_dir, save_model=False, lr=5e-5, lr_decay=True, device=device)
            all_metrics[f'layer {i}, before ft'] = layer_metrics[0]
            all_metrics[f'layer {i}, after ft'] = layer_metrics[1]

        model, final_metrics = train_and_eval(args, model, training_dict, val_arguments, final_epochs,
                save_path=save_base, save_model=True, lr=5e-5, lr_decay=True, device=device)
        all_metrics['final, before ft'] = final_metrics[0]
        all_metrics['final, after ft'] = final_metrics[1]

    elif prune_type == 'masking':
        model = prune_global_masking(model, dl, num_batches=num_batches, ratio=ratios[0])
        model = train_and_eval(args, model, training_dict, val_arguments, final_epochs,
                save_path=save_base, save_model=True, lr=5e-5, lr_decay=True, device=device)
    else:
        raise NotImplementedError()

    # post_flops, post_params = profile(model, inputs=(test_input_ids, test_attention_mask), verbose=False)
    # print(f'after, flops: {post_flops}, params: {post_params}')
    # print(f'saved, flops: {pre_flops - post_flops}, params: {pre_params - post_params}')


    n_params_after = model.num_parameters()
    print(f'Number of parameters after pruning: {n_params_after}')
    return all_metrics

## PRUNE LAYER

def prune_attn_layer(args, layer, hidden_states_list, mask_list, val_arguments, prune_ratio=0.5, layer_id=None, device='cpu'):
  if layer_id is not None:
    print(f'\nPruning layer {layer_id}')

  n_heads_to_keep = int(layer.self.num_attention_heads * prune_ratio)

  def shape(x):
    """separate heads"""
    return x.view(x.shape[0], -1, layer.self.num_attention_heads, layer.self.attention_head_size).transpose(1, 2)

  def unshape(x):
      """group heads"""
      return x.transpose(1, 2).contiguous().view(x.shape[0], -1, layer.self.num_attention_heads * layer.self.attention_head_size)


  dense_out_lin = layer.output.dense.to(device)
  pruned_out_lin = nn.Linear(in_features = layer.self.attention_head_size * n_heads_to_keep,
                             out_features = layer.self.all_head_size,
                             bias = True).to(device)
  Q = deepcopy(dense_out_lin.weight.data)

  S = set([]) # heads to keep
  S_indexer = None

  layer = layer.to(device)

  hidden_states = torch.cat(hidden_states_list, dim=0)
  attn_mask = torch.cat(mask_list, dim=0)
  with torch.no_grad():
      attn_mask = attn_mask.to(device)
      hidden_states = hidden_states.to(device)

      _, weights = layer(hidden_states, attn_mask, head_mask=None, output_attentions=True)

      v = shape(layer.self.value(hidden_states))
      context = torch.matmul(weights, v)

      H = unshape(context)
      h = torch.norm(context, p=2, dim=(0, 2, 3))

  for t in range(args.iters):
    if args.validate_iter:
      val_arguments['model'].bert.encoder.layer[layer_id].attention.output.dense = pruned_out_lin # might not work with dimensions
      acc = validate(args, **val_arguments, device=device)
      pruned_out_lin = pruned_out_lin.to(device)

    dW, loss = compute_grads(args, dense_out_lin, pruned_out_lin, H, S_indexer, device=device)

    if args.validate_iter:
      print(f'Iteration {t}: S = {S} | Loss = {loss} | Exact Match = {acc["exact_match"]} | F1 = {acc["f1"]}')
    elif args.iter_verbose:
      print(f'Iteration {t}: S = {S} | Loss = {loss}')

    D = find_and_merge(dW, S, layer.self.num_attention_heads, layer.self.attention_head_size, n_heads_to_keep, device=device)
    Q = update_step(args, Q, dW, S_indexer, H, device=device)
    pruned_out_lin, S, S_indexer = truncate(args, Q, pruned_out_lin, h, D, layer.self.num_attention_heads, layer.self.attention_head_size, n_heads_to_keep, dW, device=device)
    pruned_out_lin, Q = debias(args, pruned_out_lin, dense_out_lin, Q, H, S_indexer, args.debias_iters, device=device)

  print(f'Post-pruning: S = {S}')
  all_heads = set(range(layer.self.num_attention_heads))

  layer.prune_heads(all_heads.difference(S))
  # layer.out_lin = pruned_out_lin

  if layer_id is not None:
    print(f'Done pruning layer {layer_id}')

  return layer

## MAIN

def main(args):
    model_args = ModelArguments(tokenizer_name=args.tokenizer_name,
                                model_name_or_path=args.model_name)

    data_args = DataTrainingArguments()

    training_args = TrainingArguments(output_dir=args.output_dir,
                                      do_train=True,
                                      do_eval=False)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None

    set_seed(args.random_seed)

    if training_args.do_train:
        train_dataset = load_dataset(
            args.dataset_name, split="train", cache_dir=model_args.cache_dir
        )
        train_dataset = train_dataset.shuffle(seed=args.random_seed)

    if training_args.do_train:
        val_dataset = load_dataset(
            args.dataset_name, split="validation", cache_dir=model_args.cache_dir
        )
        val_dataset = val_dataset.shuffle(seed=args.random_seed)


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        # do_lower_case=model_args.do_lower_case,
        # cache_dir=model_args.cache_dir,
        # use_fast=model_args.use_fast_tokenizer,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    def preprocess_w_tokenizer(examples):
      return preprocess_function(examples, tokenizer)

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.select(range(args.train_size))
        train_dataset = train_dataset.map(
            preprocess_w_tokenizer,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.validate_iter or args.validate_layer:
      val_arguments = get_val_arguments(args, val_dataset, model, tokenizer)

    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # prune the heads of each layer
    train_dl = trainer.get_train_dataloader()

    training_dict = {
        'args': training_args,
        'train_dataset': train_dataset,
        'tokenizer': tokenizer,
        'data_collator': data_collator,
    }

    if not args.iterative:
        ratios = [args.total_prune_ratio for x in range(6)]
        result_metrics = prune_bert(args, model, train_dl, args.num_batches, args.iters, ratios,
                training_dict, val_arguments, per_layer_epochs=args.epochs_per_layer, final_epochs=args.final_epochs, prune_type=args.prune_type)
    else:
        result_metrics = []
        ratio_to_elim = 1.0 - args.total_prune_ratio
        prune_iters = int(ratio_to_elim / args.ratio_per_iter)
        print(f'\nRunning {prune_iters} pruning iterations\n')
        for pt in range(prune_iters):
            curr_ratio = 1.0 - (args.ratio_per_iter * (pt + 1))
            prev_ratio = float(int((1.0 - args.ratio_per_iter * pt)*12.0) / 12.0)
            new_ratio = curr_ratio / prev_ratio
            ratios = [new_ratio for x in range(6)]
            result_metrics.append(prune_bert(args, model, train_dl, args.num_batches, args.iters, ratios,
                    training_dict, val_arguments, per_layer_epochs=args.epochs_per_layer, final_epochs=args.final_epochs, prune_type=args.prune_type))

    return result_metrics


## RUN

args = {
    'output_dir': '/content',
    'results_dir': '/content/results/',
    'exp_num': 0,

    'tokenizer_name': 'deepset/bert-base-cased-squad2',
    'model_name': 'deepset/bert-base-cased-squad2',
    'dataset_name': 'squad',

    'random_seed': 0,

    'train_size': 1000,
    'val_size': 25,

    'prune_type': 'ispasp++',

    'final_epochs': 0,
    'epochs_per_layer': 0,
    'num_batches': 5,
    'iters': 10,
    'total_prune_ratio': 0.3,
    'iterative': False,
    'ratio_per_iter': 0.1,
    'min_layer': -1,
    'max_layer': 100,

    'validate_iter': False,
    'validate_layer': True,
    'iter_verbose': True,
    'verbose': False,

    'dense_update': False,
    'maintain_Q': False,
    'eta': 5e-5, #'adaptive', #5e-5,
    'debias_iters': 0,
    'debias_dense': False,
    'trunc_strategy': 'dense_grad', # magnitude dense_grad attn_weights
    'choose_from_D': False
}
d_args = dotdict(args)

run_mets = main(d_args)