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
    )
import nltk
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
accelerator = Accelerator()
import math
SEED = 42
random.seed(SEED)
import utils
#DATA [['model_name', 'dataset_name', 'text_ind', 'text', 'prompt', 'params', 'human_agg', 'model_annots'],
# intramodel [['dataset_name', 'text_ind', 'text', 'prompt', 'params', 'human_annots', 'model_annots']
# intermodel************ DATA [['model_name', 'dataset_name', 'text_ind', 'text', 'prompt', 'human_annots', 'model_annots']
from custom_trainer import CustomTrainer
# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
BATCH_SIZE = utils.get_batch_size()

def calc_num_outcomes(num_labels):
    # all possible combinations (ignore order) of num_labels)
    return math.factorial(5)/(math.factorial(5-num_labels)*math.factorial(num_labels))



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
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
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
        is_roberta = "roberta" in self.model_name
        if is_roberta:
            from transformers import Trainer, TrainingArguments
            trainer = CustomTrainer
            training_args = TrainingArguments
        elif "t5" in self.model_name:
            from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
            trainer = Seq2SeqTrainer
            training_args = Seq2SeqTrainingArguments
        print(accelerator.num_processes)
        # Define training args
        self.training_args = training_args(
            output_dir=repository_id,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            predict_with_generate=True,
            fp16=True,############3
            fp16_full_eval=True,#########
            dataloader_num_workers=accelerator.num_processes,
            learning_rate=5e-4,
            num_train_epochs=50,
            logging_dir=f"{repository_id}/logs",
            logging_strategy="steps",
            logging_steps=500,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="edit_distance",
            report_to="wandb",
            push_to_hub=False,
            hub_strategy="every_save",
            hub_model_id=repository_id,
            hub_token=HfFolder.get_token(),
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
        self.trainer = trainer(
            model=self.model,
            args=self.training_args,
            data_collator=data_collator,
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
'''
def compute_metrics_(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    #labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

    edit_distances = []
    assert len(decoded_preds) == len(decoded_labels)
    for i in range(len(decoded_preds)):
        s1 = decoded_preds[i]
        s2 = decoded_labels[i]
        edit_distances.append(nltk.edit_distance(s1, s2))

    # Some simple post-processing
    #######################BEFORE AND AFTER ARE THE SAME #########################
    return {'edit_distance': np.mean(edit_distances)}
'''


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
    tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, model.tokenizer, col_for_num_labels, remove_columns=remove_columns)
    if 'intra' in filename: 
        repository_id = f"{model_id.replace('/','-')}-intra_model"
    else:
        repository_id = f"{model_id.replace('/','-')}-inter_model"

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print("==============", preds)
        preds = torch.from_numpy(preds) 
        labels = torch.from_numpy(labels)
        # Replace -100 in the labels as we can't decode them.
        preds = np.where(preds != -100, preds, model.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, model.tokenizer.pad_token_id)
        decoded_preds = model.tokenizer.batch_decode(preds, skip_special_tokens=True)
        print("DECODED PRED", decoded_preds)
        decoded_labels = model.tokenizer.batch_decode(labels, skip_special_tokens=True)
        print("DECODED LABELS", decoded_labels)

        edit_distances = []
        assert len(decoded_preds) == len(decoded_labels)
        for i in range(len(decoded_preds)):
            s1 = decoded_preds[i]
            if s1 == "":
                edit_distances.append(-10.0)
                continue
            s2 = decoded_labels[i]
            if len(s2.split()) < num_labels:
                continue
            # calculate number of overlapping digits
            # extract digits in s1 that are in s2
            s1_digits = [int(s) for s in s1.split() if s.isdigit()]
            s2_digits = [int(s) for s in s2.split() if s.isdigit()]

            diff = sum(abs(s) for s in s1_digits) - sum(abs(s) for s in s2_digits)

            overlap = set(s1_digits).intersection(set(s2_digits))

            edit_distances.append(-nltk.edit_distance(s1, s2)/10 - diff + len(overlap))


        # Some simple post-processing
        #######################BEFORE AND AFTER ARE THE SAME #########################
        print("edit_distances", edit_distances)
        return {'edit_distance': np.mean(edit_distances)}
        #model.compute_metrics(eval_preds, label_pad_token_id=-100)

    model.set_tokenized_dataset(tokenized_dataset)
    model.set_training_var(repository_id, compute_metrics)
    model.trainer.train()
    model.trainer.evaluate()
    model.model.save_pretrained("output_dir") 
    model.tokenizer.save_pretrained(repository_id)
    model.trainer.create_model_card()
    model.trainer.push_to_hub()


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

