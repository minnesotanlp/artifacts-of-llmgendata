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
mse_metric = evaluate.load("mse")
SEED = 42
random.seed(SEED)
import utils
LR = 1e-4
NUM_CYCLES = 2
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
loss_fct = CrossEntropyLoss(ignore_index=-1)
global_num_labels = 0
global_num_annots = 0
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
        print('*************************************')
        for layer in model.classifiers.named_parameters():
            if 'weight' in layer[0]:
                print(torch.sum(layer[1]).item())
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

def compute_metrics(eval_preds):
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main(filename, model_id, dataset_name, remove_columns, col_for_num_labels, dataset_mode, target_col='model_annots_str'):
    global global_num_labels, global_num_annots
    global_num_labels = utils.get_num_labels(dataset_name)
    global_num_annots = utils.get_num_annots(dataset_name)
    BATCH_SIZE = utils.get_batch_size(dataset_name)
    model = MultiTaskRobertaModel(model_id, global_num_annots, global_num_labels).to(accelerator.device)
    tokenizer = RobertaTokenizer.from_pretrained(model_id)

    # take subset of data
    tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, tokenizer, col_for_num_labels, remove_columns=remove_columns, mode=dataset_mode, target_col=target_col)
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
    tokenized_dataset["train"] = tokenized_dataset["train"].select(range(min(100, len(tokenized_dataset["train"]))))
    tokenized_dataset["val"] = tokenized_dataset["val"].select(range(min(10, len(tokenized_dataset["val"]))))
    tokenized_dataset["test"] = tokenized_dataset["test"].select(range(min(10, len(tokenized_dataset["test"]))))

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
        evaluation_strategy="steps",
        num_train_epochs=5,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        push_to_hub=True,
        include_inputs_for_metrics=True,
        hub_strategy="checkpoint",
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
        remove_unused_columns = False,#"Mistral" in self.model_name,
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        pad_to_multiple_of=8
    )
    #train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    #test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    #train_dataloader = DataLoader(
    #    tokenized_dataset,
    #    batch_size=training_args.per_device_train_batch_size,
    #    shuffle=True,
    #    num_workers=4,  # Adjust based on your system configuration
    #    #collate_fn=custom_collate_fn,  # Replace with your collate function if neede
    #)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    #trainer.save_model(f'{repository_id}.pt')
    trainer.create_model_card()
    trainer.push_to_hub()
    p = trainer.predict(tokenized_dataset["test"])
    filename = f"{repository_id}.txt"
    print(p[1])

    with open(pkl_filename, 'w') as f:
        f.writelines(p[1])
    return


    #print(trainer.evaluate(eval_dataset=tokenized_dataset["test"]))
    #print("-------PREDICT-----------")
    '''
    #'''
    #print(len(p['predictions']), len(p['predictions'][0]))
    pkl_filename = "test.pkl"
    #print(len(p['label_ids']), len(p['label_ids'][0]))
    with open(pkl_filename, 'rb') as f:
        print(f)
        p = pickle.load(f)

    correct = 0
    total = 0
    print(p)
    print('total', len(p)*len(p[0]))
    for inst in range(len(p)):
        for annot in range(len(p[inst])):
            total += 1
            if p[inst][annot] == tokenized_dataset["test"]["labels"][inst][annot]:
                correct += 1
            #print(p[inst][annot], tokenized_dataset["test"]["labels"][inst][annot])

    print('correct', correct)
    print('total', total)
    print('accuracy', correct/total)
    # Example data (you would replace these with your own datasets)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

    # Forward pass
    output1, output2 = model(input_ids, attention_mask)

    # Joint training with multiple losses
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    def set_training_var(self, repository_id, compute_metrics):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
        num_training_steps = 10000#int(len(self.tokenized_dataset["train"])/BATCH_SIZE) + 1
        #optimizer = Adafactor(self.model.parameters(), relative_step=False, warmup_init=False, lr=LR)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, NUM_CYCLES)
        if "roberta" in self.model_name:
            trainer = CustomTrainer
            training_args = TrainingArguments
        # Define training args
        #data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        #"constraints": [
        #    DisjunctiveConstraint([[209, 204, 220, 314, 305]]),
        self.trainer = trainer(
            model=self.model,
            #dataset_text_field="short_prompt",    ##### only for causalLM - really need to refactor this
            tokenizer=self.tokenizer,
            args=self.training_args,
            data_collator=data_collator,
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

    model.set_tokenized_dataset(tokenized_dataset)
    model.set_training_var(repository_id, compute_metrics)
    model.model.config.use_cache = False
    model.trainer.train()
    #model.trainer.evaluate(
    #    eval_dataset=tokenized_dataset["test"],
    #    #metric_key_prefix=""
    #)
    #'''
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
    for dataset_name in ['SChem5Labels', 'Sentiment', 'SBIC', 'ghc']:
        for m in ['frequency', 'dataset-frequency', 'shuffle', 'sorted']:
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
                main(filename = '../data/intramodel_data.csv',
                     model_id = 'roberta-base',
                     dataset_name = dataset_name,
                     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
                     col_for_num_labels = "model_annots",
                     dataset_mode = m,
                     target_col = target_col)
                raise Exception()

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
