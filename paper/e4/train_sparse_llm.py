import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, get_scheduler, default_data_collator
import math
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from datasets import load_from_disk
import os
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
import itertools
from datetime import datetime
from l0_layers import L0Linear
from typing import Dict, List, Optional, Tuple, Type
from humancompatible.train.dual_optim import ALM, PBM
from humancompatible.train.dual_optim import MoreauEnvelope
from torch.optim import Adam
import argparse
import ast
import csv
import hashlib
import json
import time

def load_code_ds_data():
    """
    TODO: swap for OpenWebText, FineWeb or FineWeb-Edu sample
    """

    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

    raw_datasets = DatasetDict(
        {
            "train": ds_train,  
            "valid": ds_valid,  
        }
    )

    return raw_datasets

def tokenize_curry(tokenizer, context_length):

    def tokenize(element):
        """
        Tokenize function that gets rid of texts that are smaller than context_length
        """
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    return tokenize


def tokenize_curry_lamma(tokenizer, context_length):
    """
    tokenize function that adds a EOS token at the end of each string.
    """

    def tokenize(element):
        """
        Tokenize function that gets rid of texts that are smaller than context_length
        """
        outputs = tokenizer(element["content"])
        return { "input_ids": [ids + [tokenizer.eos_token_id] for ids in outputs["input_ids"]]}

    def group(element):
        """
        Flattens the text and groups them into batches of context size
        """
        
        ids = list(itertools.chain.from_iterable(element["input_ids"]))
        n = (len(ids) // context_length) * context_length
        return {"input_ids": [ids[i:i + context_length] for i in range(0, n, context_length)]}

    return tokenize, group



def get_load_tokenized_code_ds_data(context_length = 128, 
                                    tokenizer=AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer"),
                                    save_dir="data/codeparrot_tokenized_ctx128", lamma=True):

    # create data directory
    os.makedirs("data", exist_ok=True)

    # check if the data exists - if so, load it 
    try: 
        tokenized_datasets = load_from_disk(save_dir)

        return tokenized_datasets

    except:
        print("Data not found, loading the raw dataset + tokenizing it...")
    
    # load the code ds data
    raw_datasets = load_code_ds_data()

    # tokenized datasets
    if lamma == True:  # if using lamma, also group the texts
        tokenized_datasets = raw_datasets.map(tokenize_curry_lamma(tokenizer, context_length)[0],
                                            batched=True, remove_columns=raw_datasets["train"].column_names,
                                            batch_size=500, num_proc=4)
        # group texts into chucks of window size
        tokenized_datasets = tokenized_datasets.map(tokenize_curry_lamma(tokenizer, context_length)[1],
                                            batched=True, batch_size=500, num_proc=4)
        
    else: 
        tokenized_datasets = raw_datasets.map(tokenize_curry(tokenizer, context_length),
                                            batched=True, remove_columns=raw_datasets["train"].column_names,
                                            batch_size=500, num_proc=4)

    # save the tokenized datasets
    tokenized_datasets.save_to_disk(save_dir)    

    return tokenized_datasets


def loss_fn(inputs, logits):
    
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return loss.mean()

def get_grouped_params(model):
    """
    get parameters for gates and network; also returns the sparsifiable modules
    """

    named = list(model.named_parameters())   
    params_dict = {"gates": [p for n, p in named if "log_alpha" in n],
                   "net":   [p for n, p in named if "log_alpha" not in n]}
    sparse_layers_dict = {"sparse_layers": [m for m in model.modules() if isinstance(m, L0Linear)]}
    return params_dict, sparse_layers_dict


def compute_constraints(sparse_layers_dict, per_layer=True):
    layers = sparse_layers_dict["sparse_layers"]
    if per_layer:
        dens = torch.stack([m.regularization()["exp_l0"] for m in layers])
    else:
        dens = torch.stack([m.regularization()["exp_l0"] for m in layers]).mean()
    return dens

def evaluate(model, eval_dataloader):
    model.eval()
    loss = torch.tensor(0.0)

    # evaluate the model losses on eval dataset
    for step, batch in enumerate(eval_dataloader):

        # eval mode
        with torch.no_grad():
            input_ids = batch["input_ids"].to("cuda")
        
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids).logits
                loss_cur = loss_fn(input_ids, logits)
        
                loss += loss_cur.item() / len(eval_dataloader)

        # TODO: remove this
        if (step + 1) % 100 == 0:
            break

    # compute the perplexity
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    model.train() # back to train

    # return both the test loss and perplexity
    return loss.item(), perplexity.item()

def probability_of_1_of_z(sparse_layers_dict):
    """
    Computes probability if 1 in z
    """

    with torch.no_grad():

        weights_z_all = []
        # sample weights_z
        for sparse_layer in sparse_layers_dict["sparse_layers"]:
            weights_z_all.append(sparse_layer.sample_gates()[0])

        # flatten and compute the probability if 1 
        weights_z_all = torch.cat(weights_z_all)
        probability = ((weights_z_all == 1).sum() ) / weights_z_all.nelement()

    return probability

def toy_params(method="spbm"):
    """
    TODO: this function is just a toy, later this will be passed as a parameter to the main
    and this function will be removed
    """

    params = {
        # ---------------- model / data
        "model_name": "meta-llama/Llama-3.2-1B",
        "data_save_dir": "data/codeparrot_tokenized_lamma32_1B_1024",
        "context_length": 1024,             # changing this requires re-tokenizing the dataset

        # ---------------- training
        "method": method,
        "num_epochs": 1,
        "batch_size": 16,                   # needs H200
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": False,    # True saves memory, ~30% slower
        "primal_lr": 5e-5,                  # lr of the network weights
        "weight_decay": 0.01,               # network weights only, never the gates
        "seed": 42,
        "eval_steps": 1_250,                # in optimizer updates
        "device": "cuda",
        "verbose": True,
    }

    if method == "unconstrained":
        return params

    # ---------------- sparsity problem (shared by ALL constrained methods)
    params.update({
        "candidates_to_sparsify": ["down_proj"],
        "init_mean": 1.0,                   # gate log_alpha at init: ~95% of z == 1, eval gates exactly 1
        "target_density": 0.7,              # constraint: expected DENSITY <= 0.7, i.e. >= 30% sparsity
        "per_layer": True,                  # True: one constraint per layer, False: one global constraint
        "primal_lr_gates": 0.1,            # lr of the gates (log_alpha)
    })

    # ---------------- method-specific (keys match the constructor arguments exactly)
    if method == "spbm":
        params["moreau"] = {                # MoreauEnvelope(optimizer_primal, **params["moreau"])
            "mu": 1.0,
            "beta": 0.5,
        }
        params["spbm"] = {                   # PBM(m=..., device=..., **params["spbm"])
            "penalty_mult": 0.1,
            "penalty_update": "dimin_adapt",
            "pbf": "quadratic_logarithmic",
            "init_duals": 0.001,
            "init_penalties": 1.0,
            "dual_range": (1e-4, 1000.0),
            "gamma": 0.1,
            "penalty_range": (1e-1, 1.0)
        }
    elif method == "alm":
        params["alm"] = {         
            "lr": 1e-1,         
            "init_duals": 0.001,    
            "dual_range": (-100.0, 100.0),
            "is_ineq": True,
            "penalty": 1.0

        }
    else:
        raise ValueError(f"unknown method: {method}")

    return params

def create_primal_dual_optimizer(model, params, sparse_layers_dict):
    """
    Creates dual and primal optimizers based on the given params
    """

    if params['method'] == 'unconstrained':
        optimizer_primal = AdamW([ # IMPORTANT: the network weights shuold be on the first position - important for weight clipping!
                {"params": model.parameters(),   "lr": params['primal_lr'], "weight_decay": 0.01},  # params of the network
            ])
        return optimizer_primal, None


    if params['method'] == 'spbm':

        # create a primal optimizer
        params_dict, sparse_layers_dict = get_grouped_params(model)
        optimizer_primal = AdamW([ # IMPORTANT: the network weights shuold be on the first position - important for weight clipping!
                        {"params": params_dict["net"],   "lr": params['primal_lr'], "weight_decay": 0.01},  # params of the network
                        {"params": params_dict["gates"], "lr": params['primal_lr_gates'], "weight_decay": 0.0},   # params of the gates
                    ])

        # create a dual optimizer
        n_constraints = len(sparse_layers_dict["sparse_layers"]) if params["per_layer"] else 1
        optimizer_primal = MoreauEnvelope(optimizer_primal, **params["moreau"])
        optimizer_dual = PBM(m=n_constraints, device=params["device"], **params["spbm"])

        return optimizer_primal, optimizer_dual

    if params['method'] == 'alm':

        # create a primal optimizer
        params_dict, sparse_layers_dict = get_grouped_params(model)
        optimizer_primal = AdamW([ # IMPORTANT: the network weights shuold be on the first position - important for weight clipping!
                        {"params": params_dict["net"],   "lr": params['primal_lr'], "weight_decay": 0.01},  # params of the network
                        {"params": params_dict["gates"], "lr": params['primal_lr_gates'], "weight_decay": 0.0},   # params of the gates
                    ])

        # create a dual optimizer
        n_constraints = len(sparse_layers_dict["sparse_layers"]) if params["per_layer"] else 1
        optimizer_dual = ALM(m=n_constraints, device=params["device"], **params["alm"])

        return optimizer_primal, optimizer_dual

        
        # TODO
    if params['method'] == 'ghost':
        pass
        # TODO

def inner_train_loop(model, train_dataloader, eval_dataloader,
                     num_epochs, primal_optimizer, dual_optimizer, params,
                     method='unconstrained',
                     gradient_accumulation_steps = 8, eval_steps = 5_000,
                     output_dir="saved_models/gpt2-code",
                     verbose=True,
                     sparse_layers_dict = [],
                     model_size_no_sparsity=0,
                     metrics_path="metrics.csv"):
    """
    Inner training loop for training sparse llms.
    method: Options ["unconstrained", "ssl-alm", "spbm", "sqp"]
    """

    # create the directory for models
    os.makedirs(output_dir, exist_ok=True)

    # one row per optimizer update, flushed each row so a killed job keeps its curve
    metrics_file = open(metrics_path, "w", newline="")
    metrics = csv.DictWriter(metrics_file, fieldnames=["update", "time_s", "loss", "constraint_max_viol",
                                                        "constraint_mean_viol", "network_size", "val_loss", "perplexity"])
    metrics.writeheader()
    train_time, t0 = 0.0, time.perf_counter()   # train_time excludes evaluation
    loss_sum, n_micro = 0.0, 0

    # define number of steps
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_epochs * num_update_steps_per_epoch

    # setup lr scheduler
    lr_scheduler_primal = get_scheduler(
        name="linear",
        optimizer=primal_optimizer,
        num_warmup_steps=1_000,
        num_training_steps=num_training_steps // gradient_accumulation_steps,
    )

    # train the model
    model.train()
    completed_steps = 0
    for epoch in range(num_epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=num_training_steps
        ):
            
            input_ids = batch["input_ids"].to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids).logits
                loss = loss_fn(input_ids, logits)
            loss_sum, n_micro = loss_sum + loss.detach(), n_micro + 1

            # print the status
            if step % 100 == 0:

                # compute the number of active constraints and print it 
                if method != "unconstrained":
                    with torch.no_grad():
                        layers = sparse_layers_dict["sparse_layers"]
                        ep_np = sum(m.expected_active_parameters() for m in layers).item()
                        total_sparse = sum(m.num_sparsifiable_params for m in layers)
                        constraints = compute_constraints(sparse_layers_dict, per_layer=True)
                        active = model_size_no_sparsity - total_sparse + ep_np
                log = {
                    "samples": (epoch * len(train_dataloader) + step) * train_dataloader.batch_size,
                    "updates": completed_steps,
                    "loss/train": f"{loss.item():.4f}",
                    }
                if method != "unconstrained":
                    log["max constraint"] = f"{constraints.max().item():.3f}"
                    log["params"] = f"{active:,.0f} ({100 * active / model_size_no_sparsity:.2f}%)"

                # write the output
                tqdm.write(" | ".join(f"{k}: {v}" for k, v in log.items()))
                
                if verbose:
                    gb = 1e9
                    tqdm.write(f"VRAM | allocated: {torch.cuda.memory_allocated() / gb:.2f} GB"
                            f" | reserved: {torch.cuda.memory_reserved() / gb:.2f} GB"
                            f" | peak: {torch.cuda.max_memory_allocated() / gb:.2f} GB"
                            f" | total: {torch.cuda.get_device_properties(0).total_memory / gb:.2f} GB")

            # update loop for unconstrained case
            if method == "unconstrained":
                
                # accumulate gradients
                loss = loss / gradient_accumulation_steps
                loss.backward()

                # update once accumulation is done
                if step % gradient_accumulation_steps == 0:

                    # clip the grads
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # step the weights    
                    primal_optimizer.step()
                    primal_optimizer.zero_grad()

                    # step the scheduler + info
                    lr_scheduler_primal.step()
                    completed_steps += 1

            # update loop for constrained case
            elif method == "spbm" or method == "alm":

                if step % (gradient_accumulation_steps - 1) != 0: # still accumulating for gradient_accumulation_steps - 1 steps - last step is performed in the else statement

                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    
                # update once accumulation is done
                else:  

                    # compute the constraints
                    constraints = compute_constraints(sparse_layers_dict, per_layer=params['per_layer']) - params['target_density']

                    # compute the lagrangian 
                    lagrangian = dual_optimizer.forward_update(loss / gradient_accumulation_steps, constraints )

                    # accumulate gradients
                    lagrangian.backward()

                    print(dual_optimizer.duals)
                    print(constraints)    

                    # clip the weights
                    torch.nn.utils.clip_grad_norm_(primal_optimizer.param_groups[0]['params'], 1.0) # IMPORTANT: here we assume that network weights are on the first position

                    # step the weights
                    primal_optimizer.step()
                    primal_optimizer.zero_grad()

                    # clip the gate means
                    for sparse_layer in sparse_layers_dict["sparse_layers"]:
                        sparse_layer.clamp_parameters() 

                    # step the scheduler + info
                    lr_scheduler_primal.step()
                    completed_steps += 1

            # ---------------- evaluate the model + log the step
            # if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            if step % 10 == 0: # TODO: remove this later
                row = {"update": completed_steps, "loss": (loss_sum / n_micro).item()}
                torch.cuda.synchronize()
                now = time.perf_counter()
                train_time = train_time + now - t0
                row["time_s"] = train_time
                if method != "unconstrained":   # here constraints = density - target, <= 0 is feasible
                    g = compute_constraints(sparse_layers_dict, per_layer=params['per_layer']) - params['target_density']
                    row["constraint_max_viol"] = g.max().item()
                    row["constraint_mean_viol"] = g.mean().item()

                    # log the size of the network
                    with torch.no_grad():
                        layers = sparse_layers_dict["sparse_layers"]
                        ep_np = sum(m.expected_active_parameters() for m in layers).item()
                        total_sparse = sum(m.num_sparsifiable_params for m in layers)
                        constraints = compute_constraints(sparse_layers_dict, per_layer=True)
                        active = model_size_no_sparsity - total_sparse + ep_np
                    row["network_size"] = 100 * active / model_size_no_sparsity
                        
                else:  # for unconstrained - constraint is just 1
                    row["constraint_max_viol"] = 1.0
                    row["constraint_mean_viol"] = 1.0
                    row["network_size"] = 100.0

                # compute the validation loss
                eval_loss, perplexity = evaluate(model, eval_dataloader)
                t0 = time.perf_counter() # restart time - do not include evaluate in the timing
                row["val_loss"] = eval_loss
                row["perplexity"] = perplexity
                metrics.writerow(row)
                metrics_file.flush()
                loss_sum, n_micro = 0.0, 0

            # TODO: remove this
            if step % 100 == 0:
                break


    # final evaluation, so every run has a val loss at the end
    metrics_file.close()


def set_sparse_model(model, init_L0_w_pretrained=True, candidates=['down_proj'], init_mean=4.5):
    """
    Creates a L0 linear layer in the candidates mlps
    init_L0_w_pretrained: sets the weights of the linear sparse layer copied in the original.
    """

    # sparsify each layer of the model
    for layer in model.model.layers:

        # sparsify each candidate in the layer
        for candidate in candidates:

            # get the model candidate
            mlp_old = getattr(layer.mlp, candidate)

            # create a new sparse l0 layer
            mlp_sparse = L0Linear(in_features=mlp_old.in_features, out_features=mlp_old.out_features, 
                                  use_bias=False, sparsity_type='structured')

            # copy the data from the pretrained layer
            if init_L0_w_pretrained:
                mlp_sparse.weight.data.copy_(mlp_old.weight.data)
                mlp_sparse.weight_log_alpha.data += init_mean
                
            # swap the mlp in the layer
            setattr(layer.mlp, candidate, mlp_sparse)
    


def apply_overrides(params, overrides):
    """
    Applies "key=value" overrides in place; dotted keys reach nested dicts (spbm.penalty_mult=0.5)
    """
    for item in overrides:
        key, value = item.split("=", 1)
        *path, last = key.split(".")
        d = params
        for k in path:
            d = d[k]
        if last not in d:   # a typo in a grid must fail, not silently run the default
            raise KeyError(f"unknown param {key!r}")
        try:
            d[last] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            d[last] = value  # plain string, e.g. spbm.pbf=quadratic_logarithmic
    return params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="unconstrained", choices=["unconstrained", "spbm", "alm"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a toy_params entry, e.g. --set primal_lr=1e-4 --set spbm.penalty_mult=0.5")
    return parser.parse_args()


def train_sprase_llm_code():

    args = parse_args()
    params = apply_overrides(toy_params(method=args.method), args.set)
    params['seed'] = args.seed

    # set a seed
    torch.manual_seed(params['seed'])

    # load a corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])

    # define context length here
    tokenized_datasets = get_load_tokenized_code_ds_data(context_length = params['context_length'], 
                                tokenizer=tokenizer,
                                save_dir=params['data_save_dir'])

    # create dataloaders
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=params['batch_size'], 
                                  shuffle=True, collate_fn=default_data_collator, num_workers=2, pin_memory=True)
    eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=params['batch_size'], 
                                 collate_fn=default_data_collator, num_workers=2, pin_memory=True)

    # load the model
    model = AutoModelForCausalLM.from_pretrained(params['model_name'], use_cache= not params['gradient_checkpointing'])

    if params['gradient_checkpointing']:
        model.gradient_checkpointing_enable()

    # print the model size
    model_size = sum(t.numel() for t in model.parameters())

    # create a sprase model - in the constrained case
    if params['method'] != 'unconstrained':
        set_sparse_model(model, candidates=params['candidates_to_sparsify'], init_mean=params['init_mean'])        

    if params['verbose']:
        print("Model Details: ")
        print(f"Lamma size: {model_size/1000**2:.1f}M parameters")
        print(model)

    # move the model to gpu
    model.to(params['device'])

    # create an optimizer
    params_dict, sparse_layers_dict = get_grouped_params(model)
    primal_optimizer, dual_optimizer = create_primal_dual_optimizer(model, params, sparse_layers_dict)

    # define name; the hash covers every param but the seed, so seeds of one grid cell share it
    cfg_hash = hashlib.md5(json.dumps({k: v for k, v in params.items() if k != 'seed'},
                                      sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_name = f"{params['model_name'].split('/')[-1]}_{params['method']}_seed{params['seed']}_{cfg_hash}"
    output_dir = os.path.join("saved_models", run_name)
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join("results", "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    inner_train_loop(model, train_dataloader, eval_dataloader,
                     params['num_epochs'], primal_optimizer, dual_optimizer, params,
                     method=params['method'],
                     gradient_accumulation_steps=params['gradient_accumulation_steps'], 
                     eval_steps = params['eval_steps'],
                     output_dir=output_dir,
                     verbose=params['verbose'],
                     sparse_layers_dict = sparse_layers_dict,
                     model_size_no_sparsity= sum(p.numel() for p in params_dict["net"]),
                     metrics_path=os.path.join(run_dir, "metrics.csv"))

    print('Training DONE.')


if __name__ == "__main__":

    # train_llm_trainer()   
    # train_lamma()
    train_sprase_llm_code()

