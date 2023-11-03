from transformers import Trainer, Seq2SeqTrainer, GenerationConfig, DisjunctiveConstraint 
import numpy as np
import torch
loss_fct = torch.nn.CrossEntropyLoss()
softmax = torch.nn.Softmax()
log_softmax = torch.nn.LogSoftmax()
import torch.nn.functional as F
ALLOWED_IDS = [209, 204, 220, 314, 305]
from transformers import T5Tokenizer, T5ForConditionalGeneration
'''
def restrict_decode_vocab(a, b):
    return [209, 204, 220, 314, 305]

def get_dist_matrix(num_labels):
    # Create the matrix using a list comprehension
    matrix = [[abs(i - j) for j in range(num_labels)] for i in range(num_labels)]

    # Print the matrix
    for row in matrix:
        print(" ".join(map(str, row)))
'''
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = len(ALLOWED_IDS)
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = torch.moveaxis(outputs.logits, 2, 1)

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
