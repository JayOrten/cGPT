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

    # # TODO: get .generate() working, you will want it's functionality,
    # # but it's clearly not doing something right.
    # generate_ids = model(input_ids=prompt_tokens.to(device))
    #                             #   temperature=temperature, 
    #                             #   top_p=top_p, 
    #                             #   repetition_penalty=repetition_penalty, 
    #                             #   do_sample=True)

    # print('generate_ids.logits: ', generate_ids.logits)

    # probs = torch.softmax(generate_ids.logits, dim=2)

    # # Print probs dimensions
    # print('probs.shape: ', probs.shape)

    # print('probs: ', probs)

    # # plot probabilities to file
    # plt.plot(probs[0, 13, :].detach().cpu().numpy())
    # plt.savefig('probs2.png')
    # plt.close

    # generate_tokens = torch.argmax(probs, 2).detach().cpu().tolist()

    # ---
    # generate_tokens = model.generate(prompt_tokens.to(device=device), max_new_tokens=10, return_dict_in_generate=True, output_scores=True)
    # transition_scores = model.compute_transition_scores(
    #     generate_tokens.sequences, generate_tokens.scores, normalize_logits=True
    # )
    # print('transition_scores: ', transition_scores)
    # # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
    # # encoder-decoder models, like BART or T5.
    # input_length = prompt_tokens.shape[1]
    # generated_tokens = generate_tokens.sequences[:, input_length:]
    # print('generated_tokens: ', generated_tokens)
    # for tok, score in zip(generated_tokens[0], transition_scores[0]):
    #     print('tok: ', tok)
    #     print('score: ', score)
    #     print(f"| {tok.cpu().item()} | {tokenizer.decode(tok.cpu().item()):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")
    # ---
    generate_tokens = model.generate(prompt_tokens.to(device=device),
                                     max_new_tokens=20)
    
    generate_tokens = generate_tokens.tolist()
    print('generate_tokens: ', generate_tokens)

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
