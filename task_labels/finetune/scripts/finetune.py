import os
os.environ["WANDB_PROJECT"] = "artifacts"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
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
    get_cosine_with_hard_restarts_schedule_with_warmup,
    EarlyStoppingCallback,
    GenerationConfig,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM,
    AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM,
    Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    T5Tokenizer, T5ForConditionalGeneration,
    BitsAndBytesConfig,
    )
from deepspeed.runtime.utils import see_memory_usage
import nltk
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
accelerator = Accelerator()
from trl import SFTTrainer
import math
import scipy
SEED = 42
random.seed(SEED)
import utils
#DATA [['model_name', 'dataset_name', 'text_ind', 'text', 'prompt', 'params', 'human_agg', 'model_annots'],
# intramodel [['dataset_name', 'text_ind', 'text', 'prompt', 'params', 'human_annots', 'model_annots']
# intermodel************ DATA [['model_name', 'dataset_name', 'text_ind', 'text', 'prompt', 'human_annots', 'model_annots']
from custom_trainer import CustomSeq2SeqTrainer
from trl import SFTTrainer
# we want to ignore tokenizer pad token in the loss
# Data collator
num_warmup_steps = 10
BATCH_SIZE = -1
LR = 1e-4
NUM_CYCLES = 1
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
global_num_labels = 0
global_num_annots = 0


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
#def restrict_decode_vocab(a, b):
#    return [209, 204, 220, 314, 305]
# for cosign annealing LR
def calculate_max_iterations(dataset_size, batch_size, num_epochs, num_cycles):
    num_batches_per_epoch = dataset_size // batch_size
    max_iterations = num_epochs * num_batches_per_epoch * num_cycles
    return max_iterations

class CustomTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        preds = logits.argmax(-1)
        print("LABELS", labels)
        print("PREDICTIONS", preds)
        print("")
        return super().compute_loss(model, inputs, return_outputs)
    '''
        # compute custom loss (suppose one has 3 labels with different weights)
        #loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        #loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        #return (loss, outputs) if return_outputs else loss#def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    #    self.remove_unused_columns = False
    '''
    def create_optimizer_and_scheduler(self, num_training_steps):
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        num_warmup_steps = int(len(self.train_dataset) * 0.05) 
        self.lr_scheduler =  get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps, NUM_CYCLES)
        self._created_lr_scheduler = True
        return (self.optimizer, self.lr_scheduler)

class Model:
    def __init__(self, model_name, num_labels=0, num_annots=0):
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_annots = num_annots
        my_config = {}
        my_config['renormalize_logits'] = True
        #my_config['return_dict_in_generate'] = True
        # We shouldn't do this when there's no guarantee that 1 number/label = 1 token
        #my_config['max_new_tokens'] = global_num_annots + 1
        #my_config['min_new_tokens'] = global_num_annots + 1
        if "t5" in model_name:
            #my_config["prefix_allowed_tokens_fn"] = restrict_decode_vocab #not json serializable
            #my_config['bos_token_id'] = 0
            self.tokenizer = T5Tokenizer.from_pretrained(model_name,
                #quantization_config=bnb_config,
                device_map="auto",
                pad_token='0',
                eos_token='1',
                low_cpu_mem_usage=True,
            )
            #"decoder_start_token_id": 0,

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=64, lora_dropout=0.05
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    )
            self.model.generation_config = GenerationConfig.from_dict(my_config)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            #self.model.to(accelerator.device)

        elif "Mistral" in model_name: #bfloat16
            #my_config["prefix_allowed_tokens_fn"] = restrict_decode_vocab #not json serializable
            #my_config['bos_token_id'] = 0
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                padding_side="left",
                bos_token='1',
                eos_token='2',
                add_eos_token=True,
                add_bos_token=True)
            #self.tokenizer.pad_token = self.tokenizer.eos_token
            #self.tokenizer.padding_side = 'right'
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=64, lora_dropout=0.05,
            )
            self.model = MistralForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.pad_token_id =  self.tokenizer.unk_token_id
            self.model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.generation_config = GenerationConfig.from_dict(my_config)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            #self.model.to(accelerator.device) # NOT NEEDED IF USING DEEPSPEED
        elif "roberta" in model_name:
            raise ValueError("Not implemented")
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
        num_training_steps = int(len(self.tokenized_dataset["train"])/BATCH_SIZE) + 1
        #optimizer = Adafactor(self.model.parameters(), relative_step=False, warmup_init=False, lr=LR)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, NUM_CYCLES)
        self.model, optimizer, self.tokenized_dataset['train'], scheduler = accelerator.prepare(self.model, optimizer, self.tokenized_dataset["train"], scheduler)
        if "roberta" in self.model_name:
            trainer = CustomTrainer
            training_args = TrainingArguments
        elif "t5" in self.model_name:
            #trainer = Seq2SeqTrainer
            #training_args = Seq2SeqTrainingArguments
            trainer = CustomSeq2SeqTrainer
            training_args = Seq2SeqTrainingArguments
        elif "Mistral" in self.model_name:
            #trainer = CustomTrainer
            trainer = SFTTrainer
            training_args = TrainingArguments
        # Define training args
        self.training_args = training_args(
            output_dir=repository_id,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            #predict_with_generate=True, #comment out for sfttrainer
            #generation_config=self.model.generation_config, #commend out for sfttrainer
            #fp16=True,############3
            #fp16_full_eval=True,#########
            dataloader_num_workers=accelerator.num_processes,
            learning_rate=LR,
            num_train_epochs=50,
            logging_dir=f"{repository_id}/logs",
            logging_strategy="steps",
            logging_steps=10,
            evaluation_strategy="epoch",
            bf16=True if "Mistral" in self.model_name else False,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to="wandb",
            push_to_hub=True,
            include_inputs_for_metrics=True,
            hub_strategy="every_save",
            hub_model_id=repository_id,
            hub_token=HfFolder.get_token(),
            do_train=True,
            #remove_unused_columns = "Mistral" in self.model_name,
            #compute_loss=compute_metrics,
            #generation_config=generation_config,
        )
        self.training_args._n_gpu = 2
        if "t5" in self.model_name:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                #label_pad_token_id=-100,label_pad_token_id,
                pad_to_multiple_of=8 # not sure if necessary - certain GPUs just do better with multiples of 8
            )
        else:
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer, 
                pad_to_multiple_of=8
            )
            #data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
        #"constraints": [
        #    DisjunctiveConstraint([[209, 204, 220, 314, 305]]),
        self.trainer = trainer(
            model=self.model,
            #dataset_text_field="short_prompt",    ##### only for causalLM - really need to refactor this
            tokenizer=self.tokenizer,
            args=self.training_args,
            optimizers=(optimizer, scheduler),
            data_collator=data_collator,
            callbacks=[early_stopping],
            #train_dataset=self.tokenized_dataset["train"].select(range(10)),
            #eval_dataset=self.tokenized_dataset["val"].select(range(10)),
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["val"],
            #compute_metrics=compute_metrics,
            #packing=True, ###### also only for causal lm
        )
        ## get trainer's model
        #optim_scheduler = self.trainer.create_optimizer_and_scheduler(num_training_steps=10) #num_training_steps) ########################################################
        #optim_scheduler = self.trainer.create_optimizer_and_scheduler(model=trainer.model, ....)
        ## override the default optimizer
        #trainer.optimizer = optim_scheduler[0]
        #trainer.lr_scheduler = optim_scheduler[1]
        
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

def main(filename, model_id, dataset_name, remove_columns, col_for_num_labels, dataset_mode, target_col='model_annots_str'):
    global global_num_labels, global_num_annots, BATCH_SIZE
    global_num_labels = utils.get_num_labels(dataset_name)
    global_num_annots = utils.get_num_annots(dataset_name)
    BATCH_SIZE = utils.get_batch_size(dataset_name)
    model = Model(model_id, num_labels=global_num_labels, num_annots=global_num_annots)   
    tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, model.tokenizer, col_for_num_labels, remove_columns=remove_columns, mode=dataset_mode, target_col=target_col)
    # RERUN THINGS WITHOUT BELOW WHEN WE HAVE TIME
    tokenized_dataset["train"] = tokenized_dataset["train"].select(range(min(1000, len(tokenized_dataset["train"]))))
    tokenized_dataset["val"] = tokenized_dataset["val"].select(range(min(100, len(tokenized_dataset["val"]))))
    #tokenized_dataset["test"] = tokenized_dataset["test"].select(range(min(100, len(tokenized_dataset["test"]))))

    loss_type = "mse"
    if 'intra' in filename: 
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-intra-{dataset_mode}-{target_col.replace('_annots_str', '')}-cross-ent"
    else:
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-inter-{dataset_mode}-{target_col.replace('_annots_str', '')}-cross-ent"
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
            if len(s2.split()) < global_num_annots:
                print("I hope it's not coming here too often")
                print("s2", s2.split(), "global_num_annots", global_num_annots)
                continue
            #if s1 contains characters other than digits and spaces, use edit distance
            #if s1 == '':
            #    #losses.append(10.0)
            #    losses.append(max_distance)
            #    continue
            elif (not s1.isdigit()) or (len(s1) != len(s2)) or (s1 ==  ''):
                losses.append(nltk.edit_distance(s1, s2)/max_distance)
                continue

            s1_digits = [int(s) for s in s1 if s.isdigit()]
            s2_digits = [int(s) for s in s2 if s.isdigit()]

            if len(s1_digits) != len(s2_digits):
                losses.append(nltk.edit_distance(s1, s2))
                continue
            
            ### pairwise mse start
            pairwise_squared_diff = torch.square(np.subtract(np.array(s1_digits), np.array(s2_digits)))
            mse = torch.mean(pairwise_squared_diff)
            ### pairwise mse end
            losses.append(mse)
            '''
            # Calculate length penalty
            length_penalty = abs(len(s1) - len(s2))
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
            length_penalty_weight = 0.9
            character_penalty_weight = 0.9
            ordinal_penalty_weight = 0.9
            overlap_penalty_weight = 0.9
            #print("LENGTH PENALTY", length_penalty) 
            #print("CHARACTER PENALTY", character_penalty)
            #print("ORDINAL PENALTY", ordinal_penalty)
            
            total_loss = (
                length_penalty_weight * length_penalty +
                character_penalty_weight * character_penalty +
                ordinal_penalty_weight * ordinal_penalty# +
                #overlap_penalty_weight * overlap_penalty
            )
            losses.append(total_loss)
            #losses.append(min(total_loss, 1))
            '''
        return {'losses': losses, 'train_loss': np.mean(losses), 'eval_train_loss': np.mean(losses)}
    model.set_tokenized_dataset(tokenized_dataset)
    model.set_training_var(repository_id, compute_metrics)
    model.model.config.use_cache = False
    model.trainer.train()
    #model.trainer.evaluate(
    #    eval_dataset=tokenized_dataset["test"],
    #    #metric_key_prefix=""
    #)
    model.model.save_pretrained(f'{repository_id}.pt') 
    model.tokenizer.save_pretrained(f'{repository_id}.pt')
    model.trainer.create_model_card()
    model.trainer.push_to_hub()


col_for_num_labels = "human_annots"
model_id = "google/t5-v1_1-xl"
#model_id="mistralai/Mistral-7B-v0.1"
#model_id = "mistralai/Mistral-7B-Instruct-v0.1")

for dn in ['SChem5Labels', 'Sentiment', 'ghc', 'SBIC']:
    #for m in ['frequency', 'dataset-frequency']:
    for m in ['frequency']:
    #for m in ['dataset-frequency']:
        main(filename = '../data/intramodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
             col_for_num_labels = "model_annots",
             dataset_mode = m)
        #'''
        main(filename = '../data/intermodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
             col_for_num_labels = "human_annots",
             dataset_mode = m)
        #'''
        main(filename = '../data/intramodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
             col_for_num_labels = "model_annots",
             dataset_mode = m,
             target_col='human_annots_str')
        main(filename = '../data/intermodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
             col_for_num_labels = "human_annots",
             dataset_mode = m,
             target_col = "human_annots_str")
'''
for dn in ['ghc', 'SBIC', 'Sentiment']:
    #for m in ['frequency', 'dataset-frequency']:
    for m in ['dataset-frequency']:
        main(filename = '../data/intramodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
             col_for_num_labels = "model_annots",
             dataset_mode = m)
        main(filename = '../data/intermodel_data.csv', 
             model_id = "google/t5-v1_1-large",#"roberta-base",
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
             col_for_num_labels = "human_annots",
             dataset_mode = m)
        main(filename = '../data/intramodel_data.csv', 
             model_id = "google/t5-v1_1-large",
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
             col_for_num_labels = "model_annots",
             dataset_mode = m,
             target_col='human_annots_str')
        main(filename = '../data/intermodel_data.csv', 
             model_id = "google/t5-v1_1-large",#"roberta-base",
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
             col_for_num_labels = "human_annots",
             dataset_mode = m,
             target_col = "human_annots_str")
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'frequency',
     target_col='human_annots_str')
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",
     dataset_name = "Sentiment",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'frequency')
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",
     dataset_name = "Sentiment",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'frequency',
     target_col='human_annots_str')
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",
     dataset_name = "ghc",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'frequency')
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SChem5Labels",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'dataset-frequency')
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SChem5Labels",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'dataset-frequency')
main(filename = '../data/intramodel_data.csv', 
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
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SBIC",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'frequency')
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
main(filename = '../data/intermodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "ghc",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'shuffle')
main(filename = '../data/intermodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "ghc",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
     col_for_num_labels = "human_annots",
     dataset_mode = 'frequency')
'''
