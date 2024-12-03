### 11/26/24

To check:
- Is the huggingface model ignoring pads properly? May need to add eos/bos tokens to sequences
- when generating, it appears that it's learning to generate many pad tokens, is it properly ignoring the pads during training?

### 11/26/24

Loss appears to be working: much better to not use built in HuggingFace function.

To test:
- gpt2
- learning rate + warmup

### 11/29/24

Model stopped before full epoch. Restarted with more generous early stopping to see if loss will drop any lower.

### 12/2/24

Restarted a couple of times. Final validation loss: 1.87
Test generation, works well. Using temperature actually messes the model up in this application; we always want the highest probability moves.

Tested with beam search vs not sampling: if the model predicts a wrong move, it usually sticks with it across multiple predictions. It could just be messing up one token though, so we should encourage it to sample another top probability until we find a move that works.

### 12/3/24

Trained new tokenizer with custom, predefined vocabularly of moves. Extracting from dataset results in a vocab of ~15,000. In reality, there are probably more than 30,000 possible board moves in the notation system, but it seems like have of them are extremely uncommon in actual games. Will be interesting to experiment with.

TODO:
- Test several learning rates on a larger model
- test the two different vocabs
- Fix the generation method
- Create api.
