from transformers import Trainer, Seq2SeqTrainer, GenerationConfig, DisjunctiveConstraint 
import numpy as np
import torch
import random
loss_fct = torch.nn.CrossEntropyLoss()
from transformers import T5Tokenizer, T5ForConditionalGeneration

def jaccard_loss(prediction, gold_label):
    # Convert prediction and gold_label to sets of tokens
    prediction_tokens = list(prediction)
    gold_tokens = list(gold_label)
    _pred = prediction_tokens.cpu().numpy()
    _label = gold_tokens.cpu().numpy()
    intersection = np.intersect1d(_pred, _label)
    union = np.union1d(_pred, _label)

    # Calculate the Jaccard similarity
    jaccard_similarity = intersection / union if union > 0 else 0.0

    # Calculate the negative log Jaccard loss
    jaccard_loss = -torch.log(torch.tensor(jaccard_similarity))

    return jaccard_loss

class CustomSeq2SeqTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits.detach(), dim = 2)
        logits = torch.moveaxis(outputs.logits, 2, 1)
        j_losses = []
        for i in range(len(preds)):
            pred = torch.sort(preds[i])[0]
            # remove -100 since we don't want that to be considered
            label = labels[i]
            label = label[label != -100]
            label = torch.sort(label)[0]
            if len(pred) > len(label):
                # if the length is different, get the intersection and difference, get whatever is missing from the difference
                # get intersection
                _pred = pred.cpu().numpy()
                _label = label.cpu().numpy()
                intersection = np.intersect1d(_pred, _label)
                difference = np.setdiff1d(_pred, _label)
                temp = np.concatenate([intersection, random.choices(difference, k=len(label)-len(intersection))])
                temp = np.sort(temp)
                pred = torch.tensor(temp, device = pred.device, dtype=torch.float64, requires_grad = True).long()
            j_losses.append(jaccard_loss(pred, label))
        loss = torch.mean(torch.stack(j_losses))
        return (loss, outputs) if return_outputs else loss

        loss = loss_fct(logits, labels)
        print(loss)
        raise Exception("stop")
        
        # TODO: make it use the ordinal characteristic

        return (loss, outputs) if return_outputs else loss
