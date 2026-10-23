import transformers
from datasets import load_dataset
from datasets import ClassLabel
import random
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import math


def tokenizer_curry(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    return tokenize_function

def group_texts(examples, block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def train_lamma():

    # load data here
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # define the models here
    model_checkpoint = "gpt2"
    tokenizer_checkpoint = "sgugger/gpt2-like-tokenizer"

    # load the tokenizer here
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

    # tokenize the data + remove the text
    tokenized_datasets = datasets.map(tokenizer_curry(tokenizer), batched=True, num_proc=16, remove_columns=["text"])

    # group the data based in the window grid + create labels
    lm_datasets = tokenized_datasets.map(
                        group_texts,
                        batched=True,
                        batch_size=1000,
                        num_proc=16,    
                    )

    print( tokenizer.decode(lm_datasets["train"][1]["input_ids"]) )

    # load the model here
    config = AutoConfig.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_config(config)

    print(config)
    # print(model)

    training_args = TrainingArguments(
        f"{model_checkpoint}-wikitext2",
        eval_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")




if __name__ == "__main__":

    train_lamma()
