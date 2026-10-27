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


def compute_constraints(sparse_layers, per_layer=True):
    layers = sparse_layers["sparse_layers"]
    if per_layer:
        dens = torch.stack([m.regularization()["exp_l0"] for m in layers])
    else:
        dens = torch.stack([m.regularization()["exp_l0"] for m in layers]).mean()
    return dens

def evaluate(model, eval_dataloader):
    model.eval()
    losses = []

    # evaluate the model losses on eval dataset
    for step, batch in enumerate(eval_dataloader):

        # eval mode
        with torch.no_grad():
            input_ids = batch["input_ids"].to("cuda")
        
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids).logits
                loss = loss_fn(input_ids, logits)

            losses.append(loss.item())

            # # TODO: remove
            # print(step)
            # if step == 100:
            #     break

    loss = torch.mean(torch.tensor(losses))

    # compute the perplexity
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    # return both the test loss and perplexity
    return loss.item(), perplexity.item()

def inner_train_loop(model, train_dataloader, eval_dataloader,
                     num_epochs, optimizer_primal,
                     problem_type='unconstrained',
                     gradient_accumulation_steps = 8, eval_steps = 5_000,
                     output_dir="saved_models/gpt2-code",
                     verbose=True,
                     sparse_layers_dict = [],
                     eps_threshold=0.7, 
                     model_size_no_sparsity=0):
    """
    Inner training loop for training sparse llms. 
    problem_type: Options ["unconstrained", "ssl-alm", "spbm", "sqp"]
    """

    # create the directory for models
    os.makedirs(output_dir, exist_ok=True)

    # define number of steps
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_epochs * num_update_steps_per_epoch

    # setup lr scheduler
    lr_scheduler_primal = get_scheduler(
        name="linear",
        optimizer=optimizer_primal,
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

            # print the status
            if step % 100 == 0:

                # compute the number of active constraints and print it 
                with torch.no_grad():
                    layers = sparse_layers_dict["sparse_layers"]
                    ep_np = sum(m.expected_active_parameters() for m in layers).item()
                    total_sparse = sum(m.num_sparsifiable_params for m in layers)
                    constraints = compute_constraints(sparse_layers_dict, per_layer=True)

                active = model_size_no_sparsity - total_sparse + ep_np
                print({
                        "num samples": (epoch * step + step) * train_dataloader.batch_size,
                        "number of update steps": completed_steps,
                        "loss/train": loss.item(),
                        "num params": f"{active:,.0f} ({100 * active / model_size_no_sparsity:.2f}%)",
                        "max constraint": f"{constraints.max().item():.3f}"
                })

                if verbose:
                    print("VRAM status: ")
                    print(f"allocated:      {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"reserved:       {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                    print(f"peak:           {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                    print(f"total_available:{torch.cuda.get_device_properties(0).total_memory / 1e9, "GB"}")

            if problem_type == "unconstrained":
                
                # accumulate gradients
                loss = loss / gradient_accumulation_steps
                loss.backward()

                # update once accumulation is done
                if step % gradient_accumulation_steps == 0:

                    # step the weights
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_primal.step()
                    optimizer_primal.zero_grad()

                    # step the scheduler + info
                    lr_scheduler_primal.step()
                    completed_steps += 1

            elif problem_type == "spbm":

                
                # step the weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_primal.step()
                optimizer_primal.zero_grad()

                # step the scheduler + info
                lr_scheduler_primal.step()
                completed_steps += 1

                # constraints = compute_constraints(sparse_layers_dict)

                # print(constraints)
                # exit()


            # ---------------- Evaluate the model + save the model
            if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            # if (step % 10 ) == 0:
                print('evaluating')
                eval_loss, perplexity = evaluate(model, eval_dataloader)
                print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train() # back to train

                # save the model
                model.save_pretrained(output_dir)


def set_sparse_model(model, init_L0_w_pretrained=True, candidates=['down_proj'], init_mean=0.0):
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
    


def train_sprase_llm_code():

    # define hyperparams here 
    num_epochs = 1
    context_length = 1024 # changing this would need a recomputation of the dataset
    gradient_accumulation_steps = 4
    batch_size = 16                     # needs H200
    # problem_type='unconstrained'
    problem_type='spbm'
    model_name = "meta-llama/Llama-3.2-1B"
    data_save_dir =  "data/codeparrot_tokenized_lamma32_1B_1024"
    gradient_checkpointing = False # to save memory
    seed = 42
    device = 'cuda'
    verbose = True
    candidates_to_sparsify =['down_proj'] # what layer to sparsify
    eps_threshold = 0.7   # constraint threshold

    # set a seed
    torch.manual_seed(seed)

    # load a corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # define context length here
    tokenized_datasets = get_load_tokenized_code_ds_data(context_length = context_length, 
                                tokenizer=tokenizer,
                                save_dir=data_save_dir)

    # create dataloaders
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, 
                                  shuffle=True, collate_fn=default_data_collator, num_workers=2, pin_memory=True)
    eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=batch_size, 
                                 collate_fn=default_data_collator, num_workers=2, pin_memory=True)

    # load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache= not gradient_checkpointing)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # print the model size
    model_size = sum(t.numel() for t in model.parameters())

    # create a sprase model - in the constrained case
    if problem_type != 'unconstrained':
        set_sparse_model(model, candidates=candidates_to_sparsify)        

    if verbose:
        print("Model Details: ")
        print(f"Lamma size: {model_size/1000**2:.1f}M parameters")
        print(model)

    # move the model to gpu
    model.to(device)

    # create an optimizer
    params_dict, sparse_layers_dict = get_grouped_params(model)
    optimizer = AdamW([
                    {"params": params_dict["net"],   "lr": 5e-4, "weight_decay": 0.01},  # params of the network
                    {"params": params_dict["gates"], "lr": 5e-4, "weight_decay": 0.0},   # params of the gates
                ])

    # define name
    run_name = (f"{model_name.split('/')[-1]}_{problem_type}_bs{batch_size}"
                f"_ctx{context_length}_seed{seed}")
    output_dir = os.path.join("saved_models", run_name)
    os.makedirs(output_dir, exist_ok=True)

    inner_train_loop(model, train_dataloader, eval_dataloader,
                     num_epochs, optimizer,
                     problem_type=problem_type,
                     gradient_accumulation_steps=gradient_accumulation_steps, 
                     eval_steps = 5_000,
                     output_dir=output_dir,
                     verbose=verbose,
                     sparse_layers_dict = sparse_layers_dict,
                     eps_threshold=eps_threshold,
                     model_size_no_sparsity= sum(p.numel() for p in params_dict["net"]) )

    print('Training DONE.')


if __name__ == "__main__":

    # train_llm_trainer()   
    # train_lamma()
    train_sprase_llm_code()

