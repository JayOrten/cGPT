### 11/26/24

To check:
- Is the huggingface model ignoring pads properly? May need to add eos/bos tokens to sequences
- when generating, it appears that it's learning to generate many pad tokens, is it properly ignoring the pads during training?