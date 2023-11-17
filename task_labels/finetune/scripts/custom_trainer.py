from transformers import Trainer, Seq2SeqTrainer, GenerationConfig, DisjunctiveConstraint 
import numpy as np
import torch
import random
loss_fct = torch.nn.CrossEntropyLoss()
mse_loss = torch.nn.MSELoss()
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
        loss = loss_fct(logits, labels)
        # TODO: make it so that the order doesn't matter
        # TODO: make it use the ordinal characteristic
        # 
        return (loss, outputs) if return_outputs else loss
        print("LOGIT DIM", logits.shape)
        print("FIRST DECODE")
        print(self.tokenizer.batch_decode(preds, skip_special_tokens=True))
        for i in range(len(preds)):
            pred = torch.sort(preds[i])[0]
            # remove -100 since we don't want that to be considered
            label = labels[i]
            label = label[label != -100]
            label = torch.sort(label)[0]
            _pred = pred.cpu().numpy()
            _label = label.cpu().numpy()
            intersection = np.intersect1d(_pred, _label)
            if len(pred) > len(label):
                difference = np.setdiff1d(_pred, _label)
                temp = np.concatenate([intersection, random.choices(difference, k=len(label)-len(intersection))])
                #temp = np.sort(temp)
                pred = torch.tensor(temp, device = pred.device, dtype=torch.float64, requires_grad = True).long()
            pred = torch.sort(pred)[0]
            test = self.tokenizer.decode(pred, return_tensors="pt", skip_special_tokens=True)
            print(test)
            
            print(label)
            print(pred)
            print(mse_loss(pred.float(), label.float()))
            raise Exception("stop")
            union = np.union1d(_pred, _label)
            # Calculate the Jaccard similarity
            jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0

            # Calculate the negative log Jaccard loss
            jaccard_loss = -torch.log(torch.tensor(jaccard_similarity))
            j_losses.append(jaccard_loss)
        loss = torch.mean(torch.stack(j_losses)).to('cuda')
        loss.requires_grad_()
        print("THIS IS LOSS", loss)
        return (loss, outputs) if return_outputs else loss

        print(loss)
        raise Exception("stop")
        
        # TODO: make it use the ordinal characteristic

        return (loss, outputs) if return_outputs else loss
