import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, get_scheduler, default_data_collator
import math
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, AutoModelForCausalLM
from datasets import load_from_disk
import os
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm

def load_code_ds_data():

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

def get_load_tokenized_code_ds_data(context_length = 128, 
                                    tokenizer="huggingface-course/code-search-net-tokenizer",
                                    save_dir="data/codeparrot_tokenized_ctx128"):

    # create data directory
    os.makedirs("data", exist_ok=True)

    # check if the data exists - if so, load it 
    try: 
        tokenized_datasets = load_from_disk(save_dir)

        # load a tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        return tokenized_datasets, tokenizer

    except:
        print("Data not found, loading the raw dataset + tokenizing it...")
    
    # load the code ds data
    raw_datasets = load_code_ds_data()

    # load a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # tokenized datasets
    tokenized_datasets = raw_datasets.map(tokenize_curry(tokenizer, context_length),
                                        batched=True, remove_columns=raw_datasets["train"].column_names,
                                        batch_size=500, num_proc=4)

    # save the tokenized datasets
    tokenized_datasets.save_to_disk(save_dir)    

    return tokenized_datasets, tokenizer


def loss_fn(inputs, logits):
    
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return loss.mean()

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"], weight_decay = 0.1):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []

    # evaluate the model losses on eval dataset
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))

    # compute the perplexity
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    # return both the test loss and perplexity
    return loss.item(), perplexity.item()

def inner_train_loop(model, train_dataloader, eval_dataloader, accelerator,
                     num_epochs, optimizer_primal,
                     problem_type='unconstrained',
                     gradient_accumulation_steps = 8, eval_steps = 5_000,
                     output_dir="saved_models"):
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

            logits = model(batch["input_ids"]).logits
            loss = loss_fn(batch["input_ids"], logits)

            # print the status
            if step % 100 == 0:
                accelerator.print(
                    {
                        "samples": (epoch * step + step) * train_dataloader.batch_sampler.batch_size,
                        "number of update steps": completed_steps,
                        "loss/train": loss.item(),
                    }
                )

            if problem_type == "unconstrained":
                
                # accumulate gradients
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)

                # update once accumulation is done
                if step % gradient_accumulation_steps == 0:

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_primal.step()
                    lr_scheduler_primal.step()
                    optimizer_primal.zero_grad()
                    completed_steps += 1

            # ---------------- Evaluate the model + save the model
            if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
                accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train() # back to train

                # save the model
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)



def train_sprase_llm_code():

    # define hyperparams here 
    num_epochs = 1
    context_length = 128
    gradient_accumulation_steps = 8
    problem_type='unconstrained'
    tokenizer_name = "huggingface-course/code-search-net-tokenizer"
    data_save_dir =  "data/codeparrot_tokenized_ctx128"
    seed = 42

    # set a seed
    torch.manual_seed(seed)

    # define context length here
    tokenized_datasets, tokenizer = get_load_tokenized_code_ds_data(context_length = context_length, 
                                tokenizer=tokenizer_name,
                                save_dir=data_save_dir)

    # create dataloaders
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, 
                                  shuffle=True, collate_fn=default_data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32, 
                                 collate_fn=default_data_collator)

    # load the model
    model_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # load a fresh config
    model = GPT2LMHeadModel(model_config)
    # model_size = sum(t.numel() for t in model.parameters())
    # print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    # create an optimizer
    optimizer = AdamW(get_grouped_params(model), lr=5e-4)

    # create an accelerator
    accelerator = Accelerator(mixed_precision='fp16')
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    inner_train_loop(model, train_dataloader, eval_dataloader, accelerator,
                     num_epochs, optimizer,
                     problem_type=problem_type,
                     gradient_accumulation_steps=gradient_accumulation_steps, 
                     eval_steps = 5_000,
                     output_dir="saved_models")

    print('Training DONE.')


if __name__ == "__main__":

    # train_llm_trainer()   
    # train_lamma()
    train_sprase_llm_code()

