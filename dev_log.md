### 11/26/24

To check:
- Is the huggingface model ignoring pads properly? May need to add eos/bos tokens to sequences
- when generating, it appears that it's learning to generate many pad tokens, is it properly ignoring the pads during training?

### 11/26/24

Loss appears to be working: much better to not use built in HuggingFace function.

To test:
- gpt2
- a different model size?
- learning rate + warmup

### 11/29/24

Model stopped before full epoch. Restarted with more generous early stopping to see if loss will drop any lower.