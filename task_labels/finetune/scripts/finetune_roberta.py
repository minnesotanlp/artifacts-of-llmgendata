import os
os.environ["WANDB_PROJECT"] = "artifacts"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import torch
torch.cuda.empty_cache()
import random
from random import randrange        
import json
from copy import deepcopy
from pandas import read_csv
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from torch.utils.data import DataLoader
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
    BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer, RobertaConfig,
    )
import pickle
#from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import math
import json
import evaluate
SEED = 42
random.seed(SEED)
import utils
LR = 1e-5
NUM_CYCLES = 2
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
loss_fct = CrossEntropyLoss(ignore_index=-1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaModel
alpha = 0.8 # for the loss function 
import utils
accelerator = utils.get_accelerator()
from evaluate import load
from collections import Counter

class CustomValueDistanceLoss(nn.Module):
    def __init__(self):
        super(CustomValueDistanceLoss, self).__init__()

    def forward(self, y_true, y_pred):
        if y_true == -1 or y_pred == -1:
            return 0
        loss = 0.5 * torch.mean((y_true - y_pred)**2)
        return loss
val_dist_fct = CustomValueDistanceLoss()

class MultiTaskRobertaModel(nn.Module):
    def __init__(self, roberta_name, num_annots, num_classes):
        # both roberta and linear layers show up in model.named_parameters()
        torch.manual_seed(0)
        super(MultiTaskRobertaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name,).to(accelerator.device)
        self.config = self.roberta.config
        self.classifiers = nn.ModuleList([nn.Linear(self.roberta.config.hidden_size, num_classes).to(accelerator.device) for _ in range(num_annots)])

    def forward(self, input_ids, attention_mask, labels=[]):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token representation
        outputs = [classifier(last_hidden_state) for classifier in self.classifiers]
        return outputs

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        print('args', args)
        print('kwargs', kwargs)
        self.alpha = kwargs.pop('alpha')
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs.to(accelerator.device))
        loss = 0
        # sum up losses from all heads
        for i in range(len(labels)):
            for j in range(len(outputs)):
                ce = loss_fct(outputs[j][i], labels[i][j])
                pred_label = torch.argmax(outputs[j][i]).float()
                pred_label.requires_grad = True
                dist = val_dist_fct(labels[i][j], pred_label)
                loss += self.alpha * ce + (1-self.alpha) * dist 
        return (loss, outputs) if return_outputs else loss

def main(filename, model_id, dataset_name, remove_columns, dataset_mode, target_col, alpha=0.8):
    suffix = f'alpha{alpha}_whole_{LR}'
    global_num_labels = utils.get_num_labels(dataset_name)
    global_num_annots = utils.get_num_annots(dataset_name)

    BATCH_SIZE = 1#utils.get_batch_size(dataset_name)
    model = MultiTaskRobertaModel(model_id, global_num_annots, global_num_labels).to(accelerator.device)
    tokenizer = RobertaTokenizer.from_pretrained(model_id)

    tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, tokenizer, remove_columns=remove_columns, mode=dataset_mode, target_col=target_col)

    if 'intra' in filename: 
        # for the non-batch size stuff, we used a batch size of 5000, which worked horribly for the bigger models
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-intra-{dataset_mode}-{target_col.replace('_annots_str', '')}"
    else:
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-inter-{dataset_mode}-{target_col.replace('_annots_str', '')}"

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)
    num_training_steps = int(len(tokenized_dataset["train"])/BATCH_SIZE) + 1000
    num_warmup_steps = num_training_steps * 0.1
    #scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, NUM_CYCLES)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    training_args = TrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        include_inputs_for_metrics=True,
        hub_strategy="checkpoint",
        hub_model_id=f'{repository_id}_{suffix}',
        hub_token=HfFolder.get_token(),
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        alpha=alpha,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        optimizers=(optimizer, scheduler),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    try:
        trainer.create_model_card()
        trainer.push_to_hub()
    except Exception as e:
        trainer.save_model(f'{repository_id}_{suffix}.pt')
    p = []
    model = model.to(device)
    with torch.no_grad():
        for row in tokenized_dataset['test']:
            res = model(torch.tensor(row['input_ids']).to(device).unsqueeze(0), torch.tensor(row['attention_mask']).to(device).unsqueeze(0))
            p.append([el.cpu().numpy() for el in res])
    filename = f"results_new/{repository_id}_{suffix}.pkl" # with the same indices
    with open(filename, 'wb') as f:
        pickle.dump(p, f)

if __name__ == "__main__":
    use_bash = True 
    if use_bash: # if running using run.sh
        # get args
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_id", type=str, default="roberta-base")
        parser.add_argument("--dataset_name", type=str, default="SChem5Labels")
        parser.add_argument("--filename", type=str, default="../data/intermodel_data.csv")
        parser.add_argument("--dataset_mode", type=str, default="data-frequency")
        parser.add_argument("--target_col", type=str, default="model_annots")
        parser.add_argument("--alpha", type=float, default=0.8)
        args = parser.parse_args()
        if args.filename == "../data/intramodel_data.csv":
            args.remove_columns = ['index', 'dataset_name', 'text', 'text_ind', 'prompt', 'params'] #, 'human_annots'
        else:
            args.remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_name', 'text', 'index'] #model_annots
        main(args.filename, args.model_id, args.dataset_name, args.remove_columns, args.dataset_mode, args.target_col, args.alpha)
    else: # if running file directly
        for dataset_name in ['SBIC', 'ghc', 'SChem5Labels', 'Sentiment'][::-1]:
            for m in ['sorted', 'shuffle', 'frequency', 'data-frequency']:
                for target_col in ['human_annots', 'model_annots']:
                    main(filename = '../data/intermodel_data.csv',
                         model_id = 'roberta-base',
                         dataset_name = dataset_name,
                         remove_columns = ['dataset_name', 'text_ind', 'prompt'],#, 'model_annots'],
                         dataset_mode = m,
                         target_col = target_col)
                    if target_col == 'model_annots':
                        main(filename = '../data/intramodel_data.csv',
                             model_id = 'roberta-base',
                             dataset_name = dataset_name,
                             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params'],#, 'human_annots'],
                             dataset_mode = m,
                             target_col = target_col)

