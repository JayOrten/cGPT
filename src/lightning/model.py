from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from transformers import (
    LlamaForCausalLM as LanguageModel, 
    LlamaConfig as HFConfig
)

from sacrebleu.metrics import BLEU, CHRF
from torchmetrics import Accuracy

# Use a lower precision for better performance
torch.set_float32_matmul_precision('medium')

class Model(LightningModule):
    def __init__(self,
                 tokenizer, 
                 config: dict = None):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        # Load model here
        if config.from_pretrained is not True:
            # * Configure necessary HF model parameters here
            model_config = HFConfig(
                vocab_size = config.vocab_size,
                max_position_embeddings = config.max_sequence_embeddings,
                hidden_size=config.dim,
                num_hidden_layers=config.n_layers,
                num_attention_heads=config.n_heads,
                rms_norm_eps=config.norm_eps,
                pad_token_id=config.pad_id
            )
            self.model = LanguageModel(model_config)
        elif config.from_pretrained is True and config.model_name is not None:
            self.model = LanguageModel.from_pretrained(config.model_name)
        else:
            raise ValueError("Must provide model_name if from_pretrained is True")
        
        self.validation_step_outputs = [] # Used for saving predictions throughout training

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch)

        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch)

        perplexity = torch.exp(loss)
        self.log('val_loss', 
                 loss, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 sync_dist=True)
        
        self.log('val_perplexity', 
                 perplexity, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 sync_dist=True)
            
        return loss
    
    def on_validation_epoch_end(self) -> None:
        if self.config.save_predictions_during_training == True:
            initial_input = self.tokenizer.encode("e4", return_tensors='pt')
            # Generate a test prediction and save output
            # This is useful for debugging and checking model performance
            generate_tokens = self.model.generate(
                initial_input,
                do_sample=True,
                max_length=100,
                pad_token_id=self.config.pad_id,
            )

            decoded = self.tokenizer.decode(generate_tokens[0].tolist())

            # Save out to file
            dir_path = Path(self.config.default_root_dir)
            file_path = dir_path / 'validation_predictions.txt'

            # Check if the directory exists. If not, create it
            dir_path.mkdir(parents=True, exist_ok=True)

            # Check if the file exists. If not, create it and append the outputs
            with file_path.open('a', encoding="utf-8") as f:
                f.write("Step " + str(self.global_step) + "tokens : ")
                f.write(str(generate_tokens) + '\n')
                f.write("Step " + str(self.global_step) + "generation : ")
                f.write(str(decoded) + '\n')
    
    def on_test_start(self,):
        # Create data structures to store predictions
        self.y_trues = []
        self.y_hats = []

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.

        Log/save any metrics you want to test here.
        """
        x, x_mask, y_true = batch

        output_ids = self.model.generate(input_ids=x, 
                                    attention_mask=x_mask,
                                    num_beams=5,
                                    min_length=0,
                                    max_new_tokens=self.config.max_gen_len)
        
        self.y_trues += self.tokenizer.batch_decode(y_true.tolist())
        self.y_hats += self.tokenizer.batch_decode(output_ids.tolist())
    
    def on_test_epoch_end(self):
        """
        Configure any metrics/output you want to save at the end of testing here.
        """
        # Save predictions
        dir_path = Path(self.config.default_root_dir)
        targets_path = dir_path / 'test_targets.txt'

        # Check if the directory exists. If not, create it
        dir_path.mkdir(parents=True, exist_ok=True)

        # Check if the file exists. If not, create it and append the outputs
        with targets_path.open('a', encoding="utf-8") as f:
            for item in self.y_trues:
                f.write(item + '\n')

        bleu = BLEU()
        chrf = CHRF()

        # # Get bleu score
        bleu = bleu.corpus_score(self.y_hats, [self.y_trues])

        # # Get chrf score
        chrf = chrf.corpus_score(self.y_hats, [self.y_trues])

        # Get accuracy
        accuracy = Accuracy()
        acc = accuracy(self.y_hats, self.y_trues)

        self.log('chrf', 
                 chrf.score, 
                 logger=True, 
                 sync_dist=True)
        self.log('bleu', 
                 bleu.score,
                 logger=True, 
                 sync_dist=True)
        self.log('accuracy', 
                 acc, 
                 logger=True, 
                 sync_dist=True)

        scores = ['chrf: ' + str(chrf.score), 'bleu: ' + str(bleu.score), 'accuracy: ' + str(acc)]
        print(scores)

    def step(self, batch):
        """
        Perform a forward pass and calculate loss.

        The behaviour at this step depends on the model being used.
        For instance, some HuggingFace models, such as Llama, will compute the shifted labels internally,
        meaning labels should be equivalent to input_ids, after which the loss is calculated internally.

        It's also possible to not pass labels, and compute the loss here instead.
        """
        x, x_mask, y_true = batch

        output = self.model(input_ids=x, 
                            attention_mask=x_mask)
        
        y_hat = output.logits

        print(y_hat.shape)
        print(y_true.shape)

        # Compute cross entropy loss
        loss = torch.nn.functional.cross_entropy(y_hat.permute(0,2,1), y_true, ignore_index=self.config.pad_id)

        return loss, y_hat
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer], [lr_scheduler]
    
    def monitor_gpu_memory(self):
        """
        Monitor GPU memory usage. Useful for debugging, checking GPU utilization.
        """
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    def is_illegal(self, board, move):
        try:
            if board.parse_san(move) in board.legal_moves:
                return False
            else:
                return True
        except:
            return True
