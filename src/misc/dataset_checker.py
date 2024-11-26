from utils.data_utils import Struct
import sys
import yaml
from sp_tokenizer.tokenizer import Tokenizer
from lightning.dataset import DataModule
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

def main():
    """
    Loads a dataset and prints the memory usage of the dataframe.
    This is outdated, but useful for understanding the structure of the dataset.
    """
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    # tokenizer = Tokenizer(model_path=config.tokenizer_path)
    # config.vocab_size = tokenizer.n_words
    # config.pad_id = tokenizer.pad_id

    # validation_dataset = DataModule(config, tokenizer)
    
    # validation_dataset.setup('test')

    # dataloader = validation_dataset.test_dataloader()

    # i = 0 
    # for batch in dataloader:
    #     print('batch: ', batch)
    #     i+= 1
    #     if i == 10:
    #         break
    data_dir = Path("/home/jo288/nobackup/autodelete/cGPT/dataset/tokenized/train/")
    full_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in data_dir.glob('*.parquet')
    )
    print(full_df.head())
    # print number of rows in df
    print(full_df.shape)

    # Graph distribution of length of rows
    



if __name__ == "__main__":
    main()
    