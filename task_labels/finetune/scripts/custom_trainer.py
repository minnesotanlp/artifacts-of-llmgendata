from transformers import Trainer, Seq2SeqTrainer, GenerationConfig, DisjunctiveConstraint 
import numpy as np
import torch
loss_fct = torch.nn.CrossEntropyLoss()
from transformers import T5Tokenizer, T5ForConditionalGeneration
class CustomSeq2SeqTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = torch.moveaxis(outputs.logits, 2, 1)
        loss = loss_fct(logits, labels)
        # TODO: make it so that the order doesn't matter
        # TODO: make it use the ordinal characteristic
        # 
        return (loss, outputs) if return_outputs else loss
