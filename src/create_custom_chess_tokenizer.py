import dask
dask.config.set({"dataframe.query-planning": True})
import dask.dataframe as dd
import sys
import yaml

from pathlib import Path
from tokenizers import decoders, pre_tokenizers, processors, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

from utils.data_utils import Struct

def train_hf_tokenizer(config: Struct):
    """
    Rather than letting BPE train a tokenizer, we are just going to manually create a tokenizer,
    where the tokenizer is every possible move in our dataset.

    To do so, we'll just iterate through our dataset and maintain a set of each unique move.
    Then, the vocabularly will be the set of unique moves.
    """
    print(f"Data dir: {config.raw_dataset_path}")
    print("Loading dataset from disk")

    # train_dataset_path = Path(config.raw_dataset_path) / "train" / "*.parquet"

    # # Only load in train set, as that's all the tokenizer needs.
    # dataset = dd.read_parquet(path=train_dataset_path,
    #                           columns=[str(config.dataset_feature)]).compute()

    dataset_path = "/home/jo288/nobackup/autodelete/cGPT/dataset/raw/processed_games.csv"
    unique_moves = set()

    # Iterate through dataset and add each unique move to the set
    with open(dataset_path, 'r') as f:
        for line in f:
            moves = line.strip().split()
            for move in moves:
                unique_moves.add(move)
    
    # load in vocab dictionary
    vocab = {"<pad>": 0, "<bos>": 1, "<unk>": 2, "<eos>": 3}
    for i, move in enumerate(unique_moves):
        vocab[move] = i + 4

    print(f"Vocab size: {len(vocab)}")
    print(vocab)

    wordlevel = WordLevel(vocab=vocab, unk_token="<unk>")

    tokenizer = Tokenizer(wordlevel)

    # trainer = WordLevelTrainer(special_tokens=["<pad>", "<bos>", "<unk>", "<eos>"])

    # tokenizer.train_from_iterator(
    #     iter(dataset[config.dataset_feature]),
    #     trainer=trainer)

    # Wrap tokenizer with transformers library
    tokenizer = PreTrainedTokenizerFast(
        model_max_length=config.max_sequence_embeddings,
        padding_side="right",
        truncation_side="right",
        bos_token="<bos>",
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        tokenizer_object=tokenizer)

    # tokenizer = Tokenizer(wordlevel)

    # trainer = WordLevelTrainer(special_tokens=["<pad>", "<bos>", "<unk>", "<eos>"])

    # tokenizer.train_from_iterator(
    #     iter(dataset[config.dataset_feature]),
    #     trainer=trainer)

    # # Save tokenizer to file
    tokenizer_save_path = Path(config.tokenizer_path)
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path)
    print('Finished!')

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert args dict to object
    config = Struct(**config)

    train_hf_tokenizer(config)

if __name__ == '__main__':
    main()
