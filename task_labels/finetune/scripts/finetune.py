import os
os.environ["WANDB_PROJECT"] = "artifacts"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
torch.cuda.empty_cache()
import random
from random import randrange        
import json
from pandas import read_csv
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import numpy as np
from huggingface_hub import HfFolder
import transformers
from transformers import (
    pipeline,
    AdamW,
    Adafactor,
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
from custom_trainer import CustomSeq2SeqTrainer
# we want to ignore tokenizer pad token in the loss
# Data collator
BATCH_SIZE = -1
LR = 1e-4
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
global_num_labels = 0
global_num_annots = 0

#def restrict_decode_vocab(a, b):
#    return [209, 204, 220, 314, 305]

class Model:
    def __init__(self, model_name, num_labels=0, num_annots=0):
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_annots = num_annots
        if "t5" in model_name:
            from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
            from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
            import transformers
            print(transformers.__file__)
            my_config = {}
            my_config['max_new_tokens'] = global_num_annots + 1
            my_config['min_new_tokens'] = global_num_annots + 1
            #my_config["prefix_allowed_tokens_fn"] = restrict_decode_vocab #not json serializable
            my_config['renormalize_logits'] = True
            my_config['return_dict_in_generate'] = True
            #my_config['bos_token_id'] = 0
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            peft_config = LoraConfig(
                #task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.generation_config = GenerationConfig.from_dict(my_config)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            self.model.to(accelerator.device)
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
        #optimizer = Adafactor(self.model.parameters(), relative_step=False, warmup_init=False, lr=LR)
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=10000)

        is_roberta = "roberta" in self.model_name
        if is_roberta:
            from transformers import Trainer, TrainingArguments
            trainer = CustomTrainer
            training_args = TrainingArguments
        elif "t5" in self.model_name:
            from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
            #trainer = Seq2SeqTrainer
            #training_args = Seq2SeqTrainingArguments
            trainer = CustomSeq2SeqTrainer
            training_args = Seq2SeqTrainingArguments
        # Define training args
        self.training_args = training_args(
            output_dir=repository_id,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            predict_with_generate=True,
            generation_config=self.model.generation_config,
            fp16=True,############3
            fp16_full_eval=True,#########
            dataloader_num_workers=accelerator.num_processes,
            learning_rate=LR,
            num_train_epochs=200,
            logging_dir=f"{repository_id}/logs",
            logging_strategy="steps",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to="wandb",
            push_to_hub=False,
            include_inputs_for_metrics=True,
            hub_strategy="every_save",
            hub_model_id=repository_id,
            hub_token=HfFolder.get_token(),
            #generation_config=generation_config,
        )
        self.training_args._n_gpu = 2
        # Create Trainer instance
        if not is_roberta:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                #label_pad_token_id=label_pad_token_id,
                #pad_to_multiple_of=8
            )
        else:
            from transformers import DataCollatorWithPadding
            #data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)#, pad_to_multiple_of=8)
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                #label_pad_token_id=-100,
                #pad_to_multiple_of=num_annots
            )
        early_stopping = EarlyStoppingCallback(early_stopping_patience=5)
        #"constraints": [
        #    DisjunctiveConstraint([[209, 204, 220, 314, 305]]),
        self.trainer = trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            optimizers=(optimizer, scheduler),
            data_collator=data_collator,
            callbacks=[early_stopping],
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["val"],
            #compute_metrics=compute_metrics,
        )
        
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

def main(filename, model_id, dataset_name, remove_columns, col_for_num_labels=[], dataset_mode='sorted', target_col='model_annots_str'):
    global global_num_labels, global_num_annots, BATCH_SIZE
    global_num_labels = utils.get_num_labels(dataset_name)
    global_num_annots = utils.get_num_annots(dataset_name)
    BATCH_SIZE = utils.get_batch_size(dataset_name)
    model = Model(model_id, num_labels=global_num_labels, num_annots=global_num_annots)   
    # contains labels and string-versions of annotations
    tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, model.tokenizer, col_for_num_labels, remove_columns=remove_columns, mode=dataset_mode, target_col=target_col)
    if 'intra' in filename: 
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-intra_model-{dataset_mode}-{target_col}"
    else:
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-inter_model-{dataset_mode}-{target_col}"
    def compute_metrics(eval_preds):
        if type(eval_preds) == transformers.trainer_utils.EvalPrediction:
            preds = eval_preds.predictions
            labels = eval_preds.label_ids
        elif type(preds) == np.ndarray:
            preds = torch.from_numpy(preds) 
            labels = torch.from_numpy(labels)
        else:
            preds, labels = eval_preds
        # Replace -100 in the labels as we can't decode them.
        preds = np.where(preds != -100, preds, model.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, model.tokenizer.pad_token_id)
        decoded_preds = model.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = model.tokenizer.batch_decode(labels, skip_special_tokens=True)
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
            elif not s1.isdigit():
                losses.append(nltk.edit_distance(s1, s2)/max_distance)
                print(losses[-1])
                raise Exception('STOP')
                continue
            # Calculate length penalty
            length_penalty = abs(len(s1) - len(s2))
            s1_digits = [int(s) for s in s1 if s.isdigit()]
            s2_digits = [int(s) for s in s2 if s.isdigit()]
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
        return {'losses': losses, 'train_loss': np.mean(losses), 'eval_train_loss': np.mean(losses)}

    model.set_tokenized_dataset(tokenized_dataset)
    model.set_training_var(repository_id, compute_metrics)
    model.trainer.train()
    model.trainer.evaluate(
        eval_dataset=tokenized_dataset["test"],
        #metric_key_prefix=""
    )
    model.model.save_pretrained(repository_id) 
    model.tokenizer.save_pretrained(repository_id)
    model.trainer.create_model_card()
    model.trainer.push_to_hub()


col_for_num_labels = "human_annots"
'''
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-small",#"roberta-base",
     dataset_name = "SChem5Labels",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'sorted')
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-base",#"roberta-base",
     dataset_name = "SChem5Labels",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'sorted')
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "ghc",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'sorted')
'''
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SChem5Labels",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'model_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'dataset-frequency')
     #target_col = "human_annots_str")
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "Sentiment",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'model_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'dataset-frequency')
     #target_col = "human_annots_str")
'''
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'model_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'dataset-frequency',
     target_col = "human_annots_str")
'''
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "ghc",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'model_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'dataset-frequency',
     target_col = "human_annots_str")
'''
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'shuffle')
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'shuffle',
     target_col = "human_annots_str")
main(filename = '../data/intermodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'frequency',
     target_col = "human_annots_str")
main(filename = '../data/intermodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'shuffle')
main(filename = '../data/intermodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'shuffle',
     target_col = "human_annots_str")
main(filename = '../data/intermodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'sorted')
main(filename = '../data/intermodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'sorted',
     target_col = "human_annots_str")
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'frequency',
     target_col = "human_annots_str")
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'frequency')
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-small",#"roberta-base",
     dataset_name = "Sentiment",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'sorted')
main(filename = '../data/intramodel_data.csv', 
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'model_annots'],
     repository_id = f"{MODEL_ID.split('/')[1]}-intra_human",
     col_for_num_labels = "human_annots")
main(filename = '../data/intermodel_data.csv', 
     remove_columns = ['model_name', 'dataset_name', 'text_ind', 'prompt', 'model_annots'],
     repository_id = f"{MODEL_ID.split('/')[1]}-inter_human",
     col_for_num_labels = "human_annots")
'''
