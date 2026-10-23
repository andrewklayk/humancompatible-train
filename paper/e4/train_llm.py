import transformers
from datasets import load_dataset

def train_lamma():

    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

    print(datasets)


if __name__ == "__main__":

    train_lamma()
