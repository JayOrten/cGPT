import os
import torch
import chess.svg
import chess.pgn
import random
import string
import sys
import yaml
import matplotlib.pyplot as plt
import time

from lightning.model import Model
from sp_tokenizer.tokenizer import Tokenizer as SPTokenizer
from transformers import PreTrainedTokenizerFast as HFTokenizer
from utils.data_utils import Struct

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

class ChessInference:
    def __init__(self, config: Struct):
        self.config = config
        # Load tokenizer
        if config.tokenizer_type == 'hf':
            self.tokenizer = HFTokenizer.from_pretrained(config.tokenizer_path, padding_size='left')
            config.pad_id = self.tokenizer.pad_token_id
        elif config.tokenizer_type == 'sp':
            self.tokenizer = SPTokenizer(config.tokenizer_path)
            config.vocab_size = self.tokenizer.n_words
            config.pad_id = self.tokenizer.pad_id
        else:
            raise ValueError(f"Tokenizer type '{config.tokenizer_type}' not recognized. Must be 'hf' or 'sp'.")


    def load_model(self):
        # Build model class
        model = Model(tokenizer=self.tokenizer, config=self.config)

        # Load checkpoint
        checkpoint_path=self.config.checkpoint_path

        print(f"Using checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['state_dict'])

        model = model.model

        model.cuda()
        model.eval()

        return model


    def run_inference(self):
        board = chess.Board()
        num_moves = 0
        model_1_incorrect = []
        model_2_incorrect = []

        model_1 = self.load_model()
        model_2 = self.load_model()

        current_sequence = ['e4']
        board.push_san('e4')
        while not board.is_game_over():
            # Model_1
            current_sequence, board, num_incorrect = self.make_move(model_1, current_sequence, board)
            model_1_incorrect.append(num_incorrect)
            num_moves += 1

            if board.is_game_over():
                break
            if num_moves >= 500:
                print('MAX MOVES REACHED')
                break

            # Model_2
            current_sequence, board, num_incorrect = self.make_move(model_2, current_sequence, board)
            model_2_incorrect.append(num_incorrect)
            num_moves += 1

            if num_moves >= 500:
                print('MAX MOVES REACHED')
                break

        print('Game over')
        print('Final board:')
        print(board)
        
        print('Outcome: ')
        print(chess.Board.outcome(board))
        #print('Winner: ')
        #print(chess.Board.outcome(board).winner())

        # Export to pgn
        game = chess.pgn.Game()
        game = game.from_board(board=board)

        name = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=7))
        # print(game, file=open("./data/games/" + name + ".pgn", "w"), end="\n\n")

        # Run analysis
        self.analyze(num_moves, model_1_incorrect, model_2_incorrect)

        return model_1_incorrect, model_2_incorrect


    def make_move(self, model, current_sequence, board):
        # First, check if greedy move is legal
        k = 1
        while True:
            print('k: ', k)
            move = self.gpt_move(model, current_sequence, do_sample=True, top_k=k)
            if not self.is_illegal(board, move):
                break
            k += 1
            if k > 700:
                move = board.san(random.choice(list(board.legal_moves)))
                print('Selected random move: ', move)
                break

        current_sequence.append(move)
        board.push_san(move)
        print('---')
        print(board)

        return current_sequence, board, k

    def gpt_move(self, model, current_sequence, do_sample=True, top_k=1):
        new_sequence = list(current_sequence)
        current_sequence_length = len(current_sequence)
        string_sequence= ' '.join(new_sequence)

        # Tokenize prompt
        if isinstance(self.tokenizer, SPTokenizer):
            prompt_tokens = torch.tensor(self.tokenizer.encode(string_sequence, bos=True, eos=False)).reshape(1,-1)
            pad_id = self.tokenizer.pad_id
            eos_id = self.tokenizer.eos_id
            bos_id = self.tokenizer.bos_id
        elif isinstance(self.tokenizer, HFTokenizer):
            prompt_tokens = self.tokenizer.encode(string_sequence, return_tensors="pt")
            pad_id = self.tokenizer.pad_token_id
            eos_id = self.tokenizer.eos_token_id
            bos_id = self.tokenizer.bos_token_id

        # Return top 10 beams
        generate_ids = model.generate(input_ids=prompt_tokens.to(device), 
                                    max_new_tokens=6, 
                                    # temperature=self.config.temperature, 
                                    # top_k=5, 
                                    # repetition_penalty=self.config.repetition_penalty, 
                                    do_sample=do_sample,
                                    top_k=top_k,
                                    pad_token_id=pad_id,
                                    eos_token_id=eos_id,
                                    bos_token_id=bos_id)
        
        generate_tokens = generate_ids.tolist()

        if isinstance(self.tokenizer, SPTokenizer):
            decoded = self.tokenizer.decode(generate_tokens)
        elif isinstance(self.tokenizer, HFTokenizer):
            decoded = self.tokenizer.decode(generate_tokens[0], skip_special_tokens=True)

        # If it predicts padding as the next token, just get the last token in the list
        if len(decoded[0].split(' ')) <= current_sequence_length:
            return decoded[0].split(' ')[-1]
        else:
            return decoded[0].split(' ')[current_sequence_length]

    def is_illegal(self, board, move):
        try:
            if board.parse_san(move) in board.legal_moves:
                return False
            else:
                return True
        except:
            return True
        
    def analyze(self, num_moves, model_1_incorrect, model_2_incorrect):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(num_moves//2), model_1_incorrect[:num_moves//2], label="model_1")
        ax.plot(range(num_moves//2), model_2_incorrect[:num_moves//2], label="model_2")
        plt.legend()
        
        date_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        # Save to unique generated filename based on time stamp
        plt.savefig(f'/home/jo288/nobackup/autodelete/cGPT/figures/{date_time_str}.png')


# For running multiple games for analysis
def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert args dict to object
    config = Struct(**config)

    model_1_incorrect_games = []
    model_2_incorrect_games = []

    for _ in range(3):
        ci = ChessInference(config)
        m1, m2 = ci.run_inference()
        model_1_incorrect_games.append(m1)
        model_2_incorrect_games.append(m2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for game in model_1_incorrect_games:
        ax.plot(range(len(game)), game)
    for game in model_2_incorrect_games:
        ax.plot(range(len(game)), game)
    
    # Save plot
    date_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'/home/jo288/nobackup/autodelete/cGPT/figures/{date_time_str}_full.png')


if __name__ == "__main__":
    main()
