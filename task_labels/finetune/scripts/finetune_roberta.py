import os
os.environ["WANDB_PROJECT"] = "artifacts"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    BitsAndBytesConfig,
    )
import pickle
#from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import math
import json
import evaluate
SEED = 42
random.seed(SEED)
import utils
LR = 1e-4
NUM_CYCLES = 2
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
loss_fct = CrossEntropyLoss(ignore_index=-1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaModel
alpha = 0.5 # for the loss function 
import utils
accelerator = utils.get_accelerator()
from evaluate import load

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
        self.device = accelerator.device

    def forward(self, input_ids, attention_mask, labels=[]):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token representation
        outputs = [classifier(last_hidden_state) for classifier in self.classifiers]
        return outputs


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs.to(accelerator.device))
        #print('*************************************')
        #for layer in model.module.classifiers.named_parameters():
        #    if 'weight' in layer[0]:
        #        print(torch.sum(layer[1]).item())
        loss = 0
        # sum up losses from all heads
        for i in range(len(labels)):
            for j in range(len(outputs)):
                ce = loss_fct(outputs[j][i], labels[i][j])
                pred_label = torch.argmax(outputs[j][i]).float()
                pred_label.requires_grad = True
                dist = val_dist_fct(labels[i][j], pred_label)
                #print("ce", ce, "dist", dist, "labels", labels[i][j], "pred_label", pred_label)
                loss += alpha * ce + (1-alpha) * dist 
        return (loss, outputs) if return_outputs else loss

def main(filename, model_id, dataset_name, remove_columns, col_for_num_labels, dataset_mode, target_col='model_annots_str'):
    global_num_labels = utils.get_num_labels(dataset_name)
    global_num_annots = utils.get_num_annots(dataset_name)

    BATCH_SIZE = utils.get_batch_size(dataset_name)
    model = MultiTaskRobertaModel(model_id, global_num_annots, global_num_labels).to(accelerator.device)
    tokenizer = RobertaTokenizer.from_pretrained(model_id)

    tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, tokenizer, col_for_num_labels, remove_columns=remove_columns, mode=dataset_mode, target_col=target_col)
    # take subset of data if training is larger than 5000
    if len(tokenized_dataset["train"]) > 5000:
        # choose random numbers between 0 and len(tokenized_dataset[split])
        train_ind = random.sample(range(len(tokenized_dataset["train"])), 5000)
        val_ind = random.sample(range(len(tokenized_dataset["val"])), 500)
        test_ind = random.sample(range(len(tokenized_dataset["test"])), 500)
        tokenized_dataset["train"] = tokenized_dataset["train"].select(train_ind)
        tokenized_dataset["val"] = tokenized_dataset["val"].select(val_ind)
        tokenized_dataset["test"] = tokenized_dataset["test"].select(test_ind)

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

    # WHEN DEBUGGING 
    #tokenized_dataset["train"] = tokenized_dataset["train"].select(range(min(100, len(tokenized_dataset["train"]))))
    #tokenized_dataset["val"] = tokenized_dataset["val"].select(range(min(10, len(tokenized_dataset["val"]))))
    #tokenized_dataset["test"] = tokenized_dataset["test"].select(range(min(10, len(tokenized_dataset["test"]))))

    if 'intra' in filename: 
        # for the non-batch size stuff, we used a batch size of 5000, which worked horribly for the bigger models
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-intra-{dataset_mode}-{target_col.replace('_annots_str', '')}"
    else:
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-inter-{dataset_mode}-{target_col.replace('_annots_str', '')}"

    training_args = TrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        #predict_with_generate=True, #comment out for sfttrainer
        #generation_config=self.model.generation_config, #commend out for sfttrainer
        #dataloader_num_workers=accelerator.num_processes,
        learning_rate=LR,
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=1,
        evaluation_strategy="epoch",
        num_train_epochs=5,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        #metric_for_best_model="f1",
        #greater_is_better=True,
        report_to="wandb",
        #push_to_hub=True,
        include_inputs_for_metrics=True,
        hub_strategy="checkpoint",
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
        #remove_unused_columns = False,#"Mistral" in self.model_name,
    )

    def compute_metrics(eval_preds):
        print("INSIDE COMPUTE METRICS")
        metric = evaluate.load("f1")
        labels = eval_preds.label_ids.flatten()
        labels2 = eval_preds.label_ids
        logits = eval_preds.predictions
        print(logits)
        predictions = np.argmax(logits, axis=0).flatten()
        if len(labels) != len(predictions):
            print("====len logits", len(logits), len(logits[0]), len(logits[0][0]))
            print("====len labels", len(labels2), labels2[0])
            print('labels', eval_preds.label_ids[0], 'predictions', np.argmax(logits, axis=0)[0])
            raise Exception("labels and predictions are not the same length")
        return metric.compute(predictions=predictions, references=labels, average="macro")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        optimizers=(optimizer, scheduler),
        #compute_metrics=compute_metrics,
    )
    trainer.train()
    try:
        trainer.create_model_card()
        trainer.push_to_hub()
    except Exception as e:
        trainer.save_model(f'{repository_id}.pt')
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Failed to push to hub", repository_id)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    p = trainer.predict(tokenized_dataset["test"])
    e = trainer.evaluate(tokenized_dataset["test"])
    filename = f"results/{repository_id}.pkl"
    print(p[1])
    print(e)

    with open(filename, 'wb') as f:
        pickle.dump([p, e], f)

if __name__ == "__main__":
    '''
    # get args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="roberta-base")
    parser.add_argument("--dataset_name", type=str, default="SChem5Labels")
    parser.add_argument("--filename", type=str, default="../data/intermodel_data.csv")
    parser.add_argument("--col_for_num_labels", type=str, default="model_annots")
    parser.add_argument("--dataset_mode", type=str, default="sorted")
    parser.add_argument("--target_col", type=str, default="model_annots")
    args = parser.parse_args()
    if args.filename == "../data/intramodel_data.csv":
        args.remove_columns = ['index', 'dataset_name', 'text', 'text_ind', 'prompt', 'params']
    else:
        args.remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_name', 'text', 'index']
    main(args.filename, args.model_id, args.dataset_name, args.remove_columns, args.col_for_num_labels, args.dataset_mode, args.target_col)
    '''
    for dataset_name in ['Sentiment']:#, 'SBIC', 'ghc', 'SChem5Labels']:
        #for m in ['dataset-frequency']:#, 'shuffle', 'sorted']:
        for m in ['frequency']:#, 'sorted']:
            #for m in ['shuffle']:#, 'sorted']:
            for target_col in ['human_annots', 'model_annots']:
                #first_order([dataset_name], 'minority')
                #first_order([dataset_name], 'all')
                #second_order([dataset_name])
                main(filename = '../data/intermodel_data.csv',
                     model_id = 'roberta-base',
                     dataset_name = dataset_name,
                     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
                     col_for_num_labels = "human_annots",
                     dataset_mode = m,
                     target_col = target_col)
                if target_col == 'model_annots':
                    main(filename = '../data/intramodel_data.csv',
                         model_id = 'roberta-base',
                         dataset_name = dataset_name,
                         remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
                         col_for_num_labels = "model_annots",
                         dataset_mode = m,
                         target_col = target_col)

    '''
    #for dn in ['SChem5Labels', 'Sentiment', 'SBIC', 'ghc']:
    #for m in ['frequency', 'dataset-frequency']:
    for m in ['sorted']:
    #for m in ['dataset-frequency']:
        main(filename = '../data/intramodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['index', 'dataset_name', 'text', 'text_ind', 'prompt', 'params', 'human_annots', 'model_annots'],
             col_for_num_labels = "model_annots",
             target_col='model_annots',
             dataset_mode = m)
        main(filename = '../data/intermodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
             col_for_num_labels = "human_annots",
             target_col='model_annots',
             dataset_mode = m)
        main(filename = '../data/intramodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
             col_for_num_labels = "model_annots",
             dataset_mode = m,
             target_col='human_annots')
        main(filename = '../data/intermodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
             col_for_num_labels = "human_annots",
             dataset_mode = m,
             target_col = "human_annots")
    '''
