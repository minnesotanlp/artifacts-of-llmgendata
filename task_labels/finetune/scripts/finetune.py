import evaluate
import os
os.environ["WANDB_PROJECT"] = "artifacts"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
import torch
import random
from random import randrange        
import json
from pandas import read_csv
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import numpy as np
from huggingface_hub import HfFolder
from transformers import (
    pipeline,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    EarlyStoppingCallback,
    GenerationConfig
    )
import nltk
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from transformers import T5Tokenizer, T5ForConditionalGeneration
accelerator = Accelerator()
import math
import scipy
SEED = 42
random.seed(SEED)
import utils
#DATA [['model_name', 'dataset_name', 'text_ind', 'text', 'prompt', 'params', 'human_agg', 'model_annots'],
# intramodel [['dataset_name', 'text_ind', 'text', 'prompt', 'params', 'human_annots', 'model_annots']
# intermodel************ DATA [['model_name', 'dataset_name', 'text_ind', 'text', 'prompt', 'human_annots', 'model_annots']
from custom_trainer import CustomTrainer, CustomSeq2SeqTrainer
# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
BATCH_SIZE = utils.get_batch_size()
LR = 1e-4

def calc_num_outcomes(num_labels):
    # all possible combinations (ignore order) of num_labels)
    return math.factorial(5)/(math.factorial(5-num_labels)*math.factorial(num_labels))

class CustomT5Model(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("DO I GET HEREEEEEE")
        #self.allowed_token_ids = torch.tensor([209, 204, 220, 314, 305]).cuda()

    def forward_nah(self, *args, **kwargs):
        print("DO I GET HEREEEEEEi22222222222")
        outputs = super().forward(*args, **kwargs)
        logits = outputs.logits
        mask = torch.full_like(logits, -1e9)
        print("BEFOREEEE mask", mask.shape)
        #print(mask[:, :, self.allowed_token_ids])
        #mask[:, :, self.allowed_token_ids] = 0
        print("AFTERRRRR  mask", mask.shape)
        #print(mask[:, :, self.allowed_token_ids])
        logits = logits + mask
        return torch.nn.functional.log_softmax(logits, dim=-1)

class MyT5DecoderModule(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.allowed_token_ids = torch.tensor([209, 204, 220, 314, 305])

    def forward(self, *args, **kwargs):
        original_logits = self.base_model(*args, **kwargs).logits
        mask = torch.ones_like(original_logits) * -1e9 
        print("BEFOREEEE mask", mask.shape)
        mask[:,:,self.allowed_token_ids] = 0
        print("AFTERRRRR  mask", mask.shape)
        return original_logits + mask


def debug_dc():
    return DataCollatorWithPadding(tokenizer=self.tokenizer)#, pad_to_multiple_of=8)

class Model:
    def __init__(self, model_name, num_labels=0):
        self.model_name = model_name
        if "roberta" in model_name:
            from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=calc_num_outcomes(num_labels))
        elif "t5" in model_name:
            from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
                #task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            #base_model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model = CustomT5Model.from_pretrained(model_name)
            #self.model = T5ForConditionalGeneration(MyT5DecoderModule(base_model))
            #self.model = get_peft_model(self.model, peft_config)
            #self.model.print_trainable_parameters()
            self.model.to(accelerator.device)
        #elif "t5" in model_name:
        #    from transformers import T5ForConditionalGeneration, AutoTokenizer
        #    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        #    self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            raise ValueError("Model model_name not supported")
        self.num_labels = num_labels

    def __repr__(self):
        return self.model_name

    def set_tokenized_dataset(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def set_training_var(self, repository_id, compute_metrics):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

        is_roberta = "roberta" in self.model_name
        if is_roberta:
            from transformers import Trainer, TrainingArguments
            trainer = CustomTrainer
            training_args = TrainingArguments
        elif "t5" in self.model_name:
            from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
            #trainer = Seq2SeqTrainer
            trainer = CustomSeq2SeqTrainer
            training_args = Seq2SeqTrainingArguments
        print(accelerator.num_processes)
        #generation_config = GenerationConfig.from_pretrained(self.model_name)
        #generation_config.max_new_tokens = 5
        #generation_config.min_new_tokens = 5
        # Define training args
        self.training_args = training_args(
            output_dir=repository_id,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            predict_with_generate=True,
            fp16=True,############3
            fp16_full_eval=True,#########
            dataloader_num_workers=accelerator.num_processes,
            learning_rate=LR,
            num_train_epochs=30,
            logging_dir=f"{repository_id}/logs",
            logging_strategy="steps",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="train_loss",
            greater_is_better=False,
            report_to="wandb",
            push_to_hub=False,
            include_inputs_for_metrics=True,
            hub_strategy="every_save",
            hub_model_id=repository_id,
            hub_token=HfFolder.get_token(),
            #generation_config=generation_config,
        )
        # Create Trainer instance
        if not is_roberta:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                label_pad_token_id=label_pad_token_id,
                #pad_to_multiple_of=8
            )
        else:
            from transformers import DataCollatorWithPadding
            #data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)#, pad_to_multiple_of=8)
            print('self.tokenized_dataset["train"]', self.tokenized_dataset["train"].shape)
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                #label_pad_token_id=-100,
                #pad_to_multiple_of=8
            )
            print('self.tokenized_dataset["val"]', self.tokenized_dataset["val"].shape)
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
        '''
        "return_dict_in_generate": True,
            "num_beams": 1, # MESS WITH THIS LATERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
            "prefix_allowed_tokens_fn": restrict_decode_vocab,
            #"constraints": [
            #    DisjunctiveConstraint([[209, 204, 220, 314, 305]]),
            #],
            "max_new_tokens": 5,
            "min_new_tokens": 5,
            "output_scores": True,
            #"temperature": 1.0,
            "do_sample": False,
        '''
        self.trainer = trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            optimizers=(optimizer, scheduler),
            data_collator=data_collator,
            callbacks=[early_stopping],
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["val"],
            compute_metrics=compute_metrics,
        )
        
    def tbd(self):
        # throwing in here for now
        self.trainer.train()
        self.trainer.evaluate()
        self.model.save_pretrained("output_dir") 
        self.tokenizer.save_pretrained(repository_id)
        self.trainer.create_model_card()
        self.trainer.push_to_hub()

        
def restrict_decode_vocab(a, b):
    return [209, 204, 220, 314, 305]

def max_edit_distance(target, output):
    # Step 1: Find the maximum length difference
    length_difference = abs(len(target) - len(output))

    # Step 2: Character Replacement
    char_difference = 0
    for t, o in zip(target, output):
        if t != o:
            char_difference += 1

    # Step 3: Calculate Maximum Edit Distance
    max_distance = length_difference + char_difference
    return max_distance

def main(filename, model_id, dataset_name, remove_columns, col_for_num_labels=[]):
    #model = Model("google/t5-v1_1-base")
    #model = Model("google/flan-t5-small")
    if dataset_name in ['SChem5Labels', 'Sentiment']:
        num_labels = 5
    elif dataset_name in ['SBIC']:
        num_labels = 3
    elif dataset_name in ['ghc']:
        num_labels = 2
    else:
        raise Exception("dataset_name not supported or not entered")
    model = Model(model_id, num_labels=pow(num_labels, 5))
    model.model.generation_config.max_new_tokens = 5
    model.model.generation_config.min_new_tokens = 5
    model.model.generation_config.num_beams = 1
    model.model.generation_config.do_sample = False
    #model.model.generation_config.return_dict_in_generate = True
    model.model.generation_config.prefix_allowed_tokens_fn = "lambda a, b: [209, 204, 220, 314, 305]"
    #model.model.generation_config.prefix_allowed_tokens_fn = restrict_decode_vocab
    #tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, model.tokenizer, col_for_num_labels, remove_columns=remove_columns, mode='sorted')
    #tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, model.tokenizer, col_for_num_labels, remove_columns=remove_columns, mode='frequency')
    tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, model.tokenizer, col_for_num_labels, remove_columns=remove_columns, mode='shuffle')

    if 'intra' in filename: 
        repository_id = f"{model_id.replace('/','-')}-intra_model"
    else:
        repository_id = f"{model_id.replace('/','-')}-inter_model"

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print("PREDS", preds)
        print("LABELS", labels)
        #print("INPUTS", inputs)
        if type(preds) != np.ndarray:
            pass
            #inputs = inputs['input_ids']
        else:
            preds = torch.from_numpy(preds) 
            #preds = ([pred[inputs.shape[0]:] for pred in preds])
            #https://github.com/huggingface/transformers/issues/17117
            #preds = ([pred[inputs.shape[1]:] for pred in preds])
            labels = torch.from_numpy(labels)
        # Replace -100 in the labels as we can't decode them.
        ######preds = np.where(preds != -100, preds, model.tokenizer.pad_token_id)
        #[input_ids.shape[0]:]i
        #labels = np.where(labels != -100, labels, model.tokenizer.pad_token_id)
        decoded_preds = model.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = model.tokenizer.batch_decode(labels, skip_special_tokens=True)
        for i in range(len(decoded_preds)):
            print("decoded_preds", decoded_preds[i])
            print("decoded_labels", decoded_labels[i])
            print("")
            break
        losses = []
        assert len(decoded_preds) == len(decoded_labels)
        for i in range(len(decoded_preds)):
            s1 = decoded_preds[i]
            s2 = decoded_labels[i]
            max_distance = max_edit_distance(s1, s2)
            if len(s2.split()) < num_labels:
                continue
            #if s1 contains characters other than digits and spaces, use edit distance
            elif s1 == '':
                losses.append(10.0)
                continue
            elif not s1.replace(" ", "").isdigit():
                losses.append(nltk.edit_distance(s1, s2)/max_distance)
                continue
            # Calculate length penalty
            length_penalty = abs(len(s1) - len(s2))
            s1_digits = [int(s) for s in s1.split() if s.isdigit()]
            s2_digits = [int(s) for s in s2.split() if s.isdigit()]
            # character penalty: check number of numbers
            if s1_digits == s2_digits:
                character_penalty = 0
            else:
                character_penalty = abs(len(s1_digits) - len(s2_digits))

            # Calculate ordinal number penalty 
            ordinal_penalty = abs(sum(s1_digits) - sum(s2_digits)) 

            # overlap penalty
            overlap_penalty = 0
            for elem in s1_digits + s2_digits:
                if elem in s1_digits and elem in s2_digits:
                    continue
                else:
                    overlap_penalty += 1

            #print("length_penalty", length_penalty)
            #print("character_penalty", character_penalty)
            #print("ordinal_penalty", ordinal_penalty)

            # Combine the penalties with their respective weights
            length_penalty_weight = 0.2
            character_penalty_weight = 0.2
            ordinal_penalty_weight = 0.2
            overlap_penalty_weight = 0.2
            total_loss = (
                length_penalty_weight * length_penalty +
                character_penalty_weight * character_penalty +
                ordinal_penalty_weight * ordinal_penalty +
                overlap_penalty_weight * overlap_penalty
            )
            #losses.append(total_loss)
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
        print("losses", losses, "mean", np.mean(losses))
        return {'losses': losses, 'train_loss': np.mean(losses)}
        #model.compute_metrics(eval_preds, label_pad_token_id=-100)

    model.set_tokenized_dataset(tokenized_dataset)
    model.set_training_var(repository_id, compute_metrics)
    model.trainer.train()
    #model.trainer.evaluate()
    #model.model.save_pretrained("output_dir") 
    #model.tokenizer.save_pretrained(repository_id)
    #model.trainer.create_model_card()
    #model.trainer.push_to_hub()


# Hugging Face repository id
col_for_num_labels = "human_annots"
#def main(filename, remove_columns, repository_id, col_for_num_labels):
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SChem5Labels",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots")
'''
main(filename = '../data/intermodel_data.csv', 
     remove_columns = ['model_name', 'dataset_name', 'text_ind', 'prompt', 'human_annots'],
     repository_id = f"{MODEL_ID.split('/')[1]}-inter_model",
     col_for_num_labels = "model_annots")
main(filename = '../data/intramodel_data.csv', 
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'model_annots'],
     repository_id = f"{MODEL_ID.split('/')[1]}-intra_human",
     col_for_num_labels = "human_annots")
main(filename = '../data/intermodel_data.csv', 
     remove_columns = ['model_name', 'dataset_name', 'text_ind', 'prompt', 'model_annots'],
     repository_id = f"{MODEL_ID.split('/')[1]}-inter_human",
     col_for_num_labels = "human_annots")
'''
# load model and tokenizer from huggingface hub with pipeline
#summarizer = pipeline("summarization", model="philschmid/flan-t5-base-samsum", device=0)

# select a random test sample
#sample = dataset['test'][randrange(len(dataset["test"]))]
#print(f"dialogue: \n{sample['dialogue']}\n---------------")

# summarize dialogue
#res = summarizer(sample["dialogue"])

#print(f"flan-t5-base summary:\n{res[0]['summary_text']}")

#data_df = read_csv('../data/data.csv')
#data_df = data_df.fillna(-1)
# create barplot of integers inside human_annots column based on frequency
#data_df['human_agg'].value_counts().plot(kind='bar')

