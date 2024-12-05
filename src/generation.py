import os
import sys
from typing import List
import yaml

import torch
from transformers import PreTrainedTokenizerFast as HFTokenizer
from transformers import GenerationConfig

from lightning.model import Model
from sp_tokenizer.tokenizer import Tokenizer as SPTokenizer
from utils.data_utils import Struct

import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

def generate(
    model,
    tokenizer,
    prompt: str,
    max_gen_len: int,
    temperature: float = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
) -> List[str]:
    """
    This generation script is for generating text from a prompt using a trained model.
    """
    
    if isinstance(tokenizer, SPTokenizer):
        prompt_tokens = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False)).reshape(1,-1)
    elif isinstance(tokenizer, HFTokenizer):
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")

    print('prompt_tokens: ', prompt_tokens)

    generate_tokens = model.generate(prompt_tokens.to(device=device),
                                    max_new_tokens=4,
                                    do_sample=False,
                                    top_k=10,
                                    pad_token_id=tokenizer.pad_id,
                                    eos_token_id=tokenizer.eos_id,
                                    bos_token_id=tokenizer.bos_id,
                                    output_scores=True,
                                    return_dict_in_generate=True)
    
    # Create a list of greedy predictions, starting from the highest probabilities and working down
    greedy_predictions = [] # list of token ids

    scores_original = []
    scores_sorted = []

    for score in generate_tokens.scores:
        # transform tensor to array and sort
        score = score.cpu().detach().tolist()[0]
        score = [round(s, 4) for s in score]
        scores_original.append(score)
        sorted_score = score.copy()
        sorted_score.sort()
        scores_sorted.append(sorted_score)

    print('scores_original: ', scores_original)
    print('scores_sorted: ', scores_sorted)

    # Just do the top 100 greedy predictions
    for _ in range(500):
        # Get index of max values in each scores
        prediction = []
        for index, score in enumerate(scores_sorted):
            max_val = score[-1]
            max_index = scores_original[index].index(max_val)

            prediction.append(max_index)

        greedy_predictions.append(prediction)

        # Drop the min difference
        min_difference = float('inf')
        min_index = None

        for index, score in enumerate(scores_sorted):
            max_val = score[-1]
            next_max_val = score[-2]
            difference = max_val - next_max_val

            if difference < min_difference:
                min_difference = difference
                min_index = index

        scores_sorted[min_index].pop(-1)

    # print out shape of scores
    # print(len(generate_tokens.scores)) # 6
    # print('generate_tokens.scores.shape[0]: ', generate_tokens.scores[0].shape) # 1, 700

    print('greedy_predictions: ', greedy_predictions)
    generate_tokens = greedy_predictions[0]

    if isinstance(tokenizer, SPTokenizer):
        decoded = tokenizer.decode(generate_tokens)
    elif isinstance(tokenizer, HFTokenizer):
        decoded = tokenizer.decode(generate_tokens[0], skip_special_tokens=True)

    return decoded

def generation(config):
    print('Beginning Inference')
    
    if config.tokenizer_type == 'hf':
        tokenizer = HFTokenizer.from_pretrained(config.tokenizer_path)
        config.pad_id = tokenizer.pad_token_id
    elif config.tokenizer_type == 'sp':
        tokenizer = SPTokenizer(config.tokenizer_path) 
        config.vocab_size = tokenizer.n_words
        config.pad_id = tokenizer.pad_id

    # Build model class
    model = Model(tokenizer=tokenizer, config=config)

    # Load checkpoint
    checkpoint_path=config.checkpoint_path

    print(f"Using checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])

    model = model.model

    model.cuda()
    model.eval()
    
    with open(config.generation_path, 'r') as f:
        prompt = f.read()

    decoded = generate(model,
                        tokenizer,
                        prompt,
                        max_gen_len = config.max_gen_len,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        repetition_penalty=config.repetition_penalty,)

    print('decoded: ', decoded)

    print('\nNo errors!\n')

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert args dict to object
    config = Struct(**config)

    generation(config)

if __name__ == "__main__":
    main()
