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
            if num_moves >= 150:
                print('MAX MOVES REACHED')
                break

            # Model_2
            current_sequence, board, num_incorrect = self.make_move(model_2, current_sequence, board)
            model_2_incorrect.append(num_incorrect)
            num_moves += 1

            if num_moves >= 100:
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
        num_incorrect = 0
        while True:
            move= self.gpt_move(model, current_sequence)
            print('Computer move: ', move)
            if self.is_illegal(board, move):
                # print('Computer selected illegal move.')
                num_incorrect += 1
    
                if num_incorrect >= 10:
                    move = board.san(random.choice(list(board.legal_moves)))
                    print('Selected random move: ', move)
                else:
                    continue
            current_sequence.append(move)
            board.push_san(move)
            print(board)
            break

        return current_sequence, board, num_incorrect

    def gpt_move(self, model, current_sequence):
        new_sequence = list(current_sequence)
        current_sequence_length = len(current_sequence)
        # print('new_sequence: ', new_sequence)
        string_sequence= ' '.join(new_sequence)
        # print('string_sequence: ', string_sequence)

        # Tokenize prompt
        if isinstance(self.tokenizer, SPTokenizer):
            prompt_tokens = torch.tensor(self.tokenizer.encode(string_sequence, bos=True, eos=False)).reshape(1,-1)
        elif isinstance(self.tokenizer, HFTokenizer):
            prompt_tokens = self.tokenizer.encode(string_sequence, return_tensors="pt")

        # print('prompt_tokens: ', prompt_tokens)

        generate_ids = model.generate(input_ids=prompt_tokens.to(device), 
                                    max_new_tokens=6, 
                                    num_beams=10,
                                    # temperature=self.config.temperature, 
                                    # top_p=self.config.top_p, 
                                    # repetition_penalty=self.config.repetition_penalty, 
                                    do_sample=True)
        # print('generate_ids: ', generate_ids)
        generate_tokens = generate_ids.tolist()
        # print(generate_tokens)

        if isinstance(self.tokenizer, SPTokenizer):
            decoded = self.tokenizer.decode(generate_tokens)
        elif isinstance(self.tokenizer, HFTokenizer):
            decoded = self.tokenizer.decode(generate_tokens[0], skip_special_tokens=True)

        # print('decoded: ', decoded)

        # print('decoded.split: ', decoded[0].split(' ')[current_sequence_length])

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
        # Save to unique generated filename based on time stamp
        plt.savefig(f'/home/jo288/nobackup/autodelete/cGPT/figures/{time.time()}.png')


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

    for _ in range(10):
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
    plt.savefig(f'/home/jo288/nobackup/autodelete/cGPT/figures/{time.time()}_full.png')


if __name__ == "__main__":
    main()
