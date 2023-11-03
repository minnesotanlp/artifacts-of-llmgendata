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
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs.keys())
        print("=====================")
        gen_cfg = GenerationConfig.from_model_config(model.config)
        print("gen_cfg", gen_cfg)

        inputs = model.generate(inputs["input_ids"], generation_config=gen_cfg)
         

        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )

        print("output logits.shape", outputs['logits'].shape)
        print("input labels.shape", inputs['labels'].shape)
        #output logits.shape torch.Size([128, 3125])
        #input labels.shape torch.Size([128, 7])

        loss = torch.nn.BCEWithLogitsLoss()(outputs['logits'],
                                         inputs['labels'])
        return (loss, outputs) if return_outputs else loss
'''
def restrict_decode_vocab(a, b):
    #from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
    #self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    #tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
    #print("BASEEEEE", tokenizer("1 2 3 4 5"))
    #tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large")
    #print("LARGEEEEEEE")
    #print(tokenizer("1 2 3 4 5"))
    return [209, 204, 220, 314, 305]

def get_dist_matrix(num_labels):
    # Create the matrix using a list comprehension
    matrix = [[abs(i - j) for j in range(num_labels)] for i in range(num_labels)]

    # Print the matrix
    for row in matrix:
        print(" ".join(map(str, row)))

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        '''
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
        '''
        ##############################################################
        ##############################################################
        ##############################################################
        num_classes = len(ALLOWED_IDS)
        dist_matrix = get_dist_matrix(num_classes)
        labels = inputs["labels"]
        num_annots = 5
        ##
        kwargs = {
            "input_ids": inputs["input_ids"],  #[32, 38]
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": inputs["input_ids"],
            "return_dict_in_generate": True,
            #"num_beams": 1, # MESS WITH THIS LATERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
            #"prefix_allowed_tokens_fn": restrict_decode_vocab,
            "max_new_tokens": num_annots + 1,
            "min_new_tokens": num_annots + 1,
            "output_scores": True,
            #"temperature": 1.0,
            #"do_sample": False,
        }
        #outputs = self.model.generate(**kwargs)
        #generated_tokens = []
        #for instance_i in range(len(outputs.sequences)):
        #    generated_tokens.append(outputs.sequences[instance_i][inputs["input_ids"][instance_i].shape[0]:])
        #generated_tokens = torch.stack(generated_tokens, requires_grad=True) # token ids
        #output_scores = torch.stack(outputs.scores, dim=2)
        outputs = model(**inputs)
        #print("22 output_scores", output_scores, output_scores.shape)
        #output_scores = torch.stack(outputs.logits, dim=2)
        #print("output_scores", output_scores, output_scores.shape)
        # replace all -inf with 0

        #print("11 output_scores", output_scores, output_scores.shape)
        #output_scores = torch.stack(outputs.scores, dim=0)
        #print("00 output_scores", output_scores, output_scores.shape)
        #output_scores = softmax(output_scores)
        logits = torch.moveaxis(outputs.logits, 2, 1)

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
        print(loss)
        # Expand dims to match the desired shape
        #output_scores = outputs.scores[0].unsqueeze(1).expand(batch_size, seq_length, vocab_size)
        #print("output_scores", output_scores.shape)
        #print("labels", labels.shape)
        #labels = labels.unsqueeze(1).expand(batch_size, seq_length)
        
        #outputs = model(**kwargs)
        #print(".............", outputs)
        #outputs = model(**inputs)
        #logits = outputs.scores[0]
        #probs = F.softmax(logits,dim=1)
        #prob_argmax = torch.argmax(probs,dim=2)
        # convert labels to one shot
        #one_hot_labels = torch.Tensor(np.eye(len(logits[0]))[labels.cpu().numpy()]).to(logits.device).requires_grad_(True)
        #print(logits.shape)
        #print(labels.shape)
        #raise Exception("stop")
        #print("LOSS", loss)
        #raise Exception("stop")
        losses = []
        for instance_i in range(len(logits)):
            print("PROBS", probs[instance_i])
            print(probs[instance_i].shape)
            print("LABELS", labels[instance_i])
            print(labels[instance_i].shape)
            losses.append(loss_fct(probs[instance_i], labels[instance_i]))
        loss = torch.mean(torch.stack(losses))
        #loss = torch.nn.BCEWithLogitsLoss()(prob_argmax, labels)
        return (loss, outputs) if return_outputs else loss
        ########## TEMP SOLUTION
        sequences = self.tokenizer.batch_decode(inputs["input_ids"])
        loss = self.compute_metrics([inputs['input_ids'], labels])
        for instance_i in range(len(labels)):
            print("WHAT IS THIS", sequences[instance_i])
            raise Exception("stop")
            #for token_i in range(len(labels[instance_i])):
            #    if labels[instance_i][token_i] not in ALLOWED_IDS:
            #        probas[instance_i][token_i] = 0
        print("LABELS", labels)
        print(self.tokenizer.batch_decode(labels))
        print("PROB_ARGMAX", prob_argmax)
        print(self.tokenizer.batch_decode(prob_argmax))
        # sort them both
        # remove intersection
        # pair most similar differences
        # if pred is string, diff = 6. multiply this by 1-pred
        raise Exception("stop")
        
        print("////////////////////", probas)
        print("PROBAS", probas)
        print("=================================")
        print("labels", labels)
        print("labels[k].item()", labels[0].item())
        print("num_classes*[labels[k].item()]", num_classes*[labels[0].item()])
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*abs(distances_tensor)**2
        loss = torch.sum(err,axis=1).mean()
        raise Exception("stop")
        return (loss, outputs) if return_outputs else loss

        #################

        version = 2
        labels = inputs.pop("labels") #[32, 9]
        sequences = outputs['sequences']
        
        logits = outputs['scores'][0]
        # only logits for [209, 204, 220, 314, 305] are valid because of restrict_decode_vocab
        count = 0
        for l in logits[0]:
            if l < -200:
                count += 1
        #    print(l)
        print("COUNT", count)
        print("ONEEEEE", logits[0])
        print("LEN", len(logits[0]))
        print(logits[0][209])
        print(logits[0][204])
        print(logits[0][220])
        print(logits[0][314])
        print(logits[0][305])
        print(logits[0][209] + logits[0][204] + logits[0][220] + logits[0][314] + logits[0][305])
        raise Exception("stop")

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("LABELS", labels)
        print("LOGITS", logits)
        raise Exception("stop")
        loss = self.compute_metrics([sequences, labels, inputs])
        return torch.Tensor(loss['losses']).to('cuda:0').requires_grad_(True)

        if version == 1:
            print(logits[0][0])
            logits = torch.stack(list(logits), dim=1)
            #for instance_i in range(len(logits)):
            #    for token_i in range(len(logits[instance_i])):
            #        logits[instance_i][token_i] = softmax(logits[instance_i][token_i])
            num_classes = logits.shape[-1]
            labels = torch.nn.functional.pad(labels, (0, logits.shape[1]  - labels.shape[1]))#, value=-100)
            one_hot_labels = torch.Tensor(np.eye(num_classes)[labels.cpu().numpy()]).to(logits.device).requires_grad_(True)
            loss = loss_fct(logits, one_hot_labels)
        elif version == 2: 
            preds = sequences
            print('type(self', type(self))
            print('self', self)
            print('self.eval_dataset', self.eval_dataset)
            print('self.compute_metrics', self.compute_metrics)
            print('type(model.module)', type(model.module))
            #print('model', model)
            print('type(self.model)', type(self.model))

            preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            print("DECODED PRED__", decoded_preds)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            print("DECODED LABELS__", decoded_labels)
            print("DID IT AT LEAST COME HERE")
            losses = []
            assert len(decoded_preds) == len(decoded_labels)
            for i in range(len(decoded_preds)):
                s1 = decoded_preds[i]
                s2 = decoded_labels[i]
                max_distance = max_edit_distance(s1, s2)
                if len(s2.split()) < num_labels:
                    #print(11111111111, s2.split(), num_labels)
                    continue
                #if s1 contains characters other than digits and spaces, use edit distance
                elif s1 == '':
                    losses.append(1.0)
                    #print(333333333333)
                    continue
                elif not s1.replace(" ", "").isdigit():
                    #print(555555555555, nltk.edit_distance(s1, s2)/max_distance)
                    #print(nltk.edit_distance(s1, s2))
                    #print(max_distance)
                    #print('')
                    losses.append(nltk.edit_distance(s1, s2)/max_distance)
                    continue
                #############################
                # Calculate length penalty (MSE loss)
                length_penalty = len(s1) - len(s2)
                print(444444444444)
                s1_digits = [int(s) for s in s1.split() if s.isdigit()]
                s2_digits = [int(s) for s in s2.split() if s.isdigit()]
                # character penalty
                if s1_digits == s2_digits:
                    character_penalty = 0
                else:
                    character_penalty = abs(len(s1_digits) - len(s2_digits))

                print("OR HEREEEEEEEEEEEEEE")
                # Calculate ordinal number penalty (MSE loss for the values in lists)
                ordinal_penalty = abs(sum(s1_digits) - sum(s2_digits)) 

                print("length_penalty", length_penalty)
                print("character_penalty", character_penalty)
                print("ordinal_penalty", ordinal_penalty)
                raise Exception("stop")

                # Combine the penalties with their respective weights
                length_penalty_weight = 0.01
                character_penalty_weight = 0.01
                ordinal_penalty_weight = 0.01
                total_loss = (
                    length_penalty_weight * length_penalty +
                    character_penalty_weight * character_penalty +
                    ordinal_penalty_weight * ordinal_penalty
                )
                losses.append(min(total_loss, 1))
                '''
                if len(s2.split()) < num_labels:
                    continue
                #if s1 contains characters other than digits and spaces, use edit distance
                elif s1 == '':
                    losses.append(1.0)
                    continue
                elif not s1.replace(" ", "").isdigit():
                    losses.append((len(s2)+nltk.edit_distance(s1, s2))/max(losses))
                    continue

                #if s1 contains only digits and spaces, use wasserstein distance 
                s1_digits = [int(s) for s in s1.split() if s.isdigit()]
                s2_digits = [int(s) for s in s2.split() if s.isdigit()]
                dist = scipy.stats.wasserstein_distance(s1_digits, s2_digits)
                #diff = sum(abs(s) for s in s1_digits) - sum(abs(s) for s in s2_digits)
                #overlap = set(s1_digits).intersection(set(s2_digits))
                #losses.append(-nltk.edit_distance(s1, s2)/10 - diff + len(overlap))
                losses.append((dist+nltk.edit_distance(s1, s2))/max(losses))
                '''

            # Some simple post-processing
            #######################BEFORE AND AFTER ARE THE SAME #########################
            print("losses", losses)
            return {'losses': losses}
            #model.compute_metrics(eval_preds, label_pad_token_id=-100)





    def label_smoothing_loss(self, logits, labels):
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = -lprobs.sum()
        return loss



