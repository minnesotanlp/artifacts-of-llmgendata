import os
os.environ["WANDB_PROJECT"] = "artifacts"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import torch
torch.cuda.empty_cache()
import random
from random import randrange        
import json
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
from deepspeed.runtime.utils import see_memory_usage
#from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import math
import json
#import scipy
SEED = 42
random.seed(SEED)
import utils
BATCH_SIZE = -1
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
from accelerate import Accelerator, DistributedType, DeepSpeedPlugin
from accelerate.state import AcceleratorState
deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2)
accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)
accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
accelerator.state.deepspeed_plugin.num_gpus = 2
distributed_type = accelerator.state.distributed_type

class MultiTaskRobertaModel(nn.Module):
    def __init__(self, roberta_name, num_annots, num_classes):
        super(MultiTaskRobertaModel, self).__init__()

        self.roberta = RobertaModel.from_pretrained(roberta_name,
            #low_cpu_mem_usage=True
        ).to(accelerator.device)
        self.config = self.roberta.config
        #self.classifiers = [nn.Linear(self.roberta.config.hidden_size, num_classes) for _ in range(num_annots)]
        self.classifiers = [nn.Linear(self.roberta.config.hidden_size, num_classes).to(accelerator.device) for _ in range(num_annots)]

    def forward(self, input_ids, attention_mask, labels=[]):
        # Forward pass through RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token representation

        # Forward pass through classifiers
        outputs = [classifier(last_hidden_state) for classifier in self.classifiers]
        return outputs

class CustomTrainer(Trainer):
      def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = 0
        for i in range(len(labels)):
            for j in range(len(outputs)):
                loss += loss_fct(outputs[j][i], labels[i][j])

        return (loss, outputs) if return_outputs else loss



def main(filename, model_id, dataset_name, remove_columns, col_for_num_labels, dataset_mode, target_col='model_annots_str'):
    #from accelerate import Accelerator, DistributedType
    #from accelerate.state import AcceleratorState
    #accelerator = Accelerator()
    #accelerator_state = AcceleratorState()
    #accelerator_state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    #distributed_type = accelerator_state.__dict__['distributed_type']

    global global_num_labels, global_num_annots, BATCH_SIZE
    global_num_labels = utils.get_num_labels(dataset_name)
    global_num_annots = utils.get_num_annots(dataset_name)
    BATCH_SIZE = utils.get_batch_size(dataset_name)
    # Instantiate the model
    model = MultiTaskRobertaModel(model_id, global_num_annots, global_num_labels)
    #model = nn.DataParallel(model)
    tokenizer = RobertaTokenizer.from_pretrained(model_id)
    tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, tokenizer, col_for_num_labels, remove_columns=remove_columns, mode=dataset_mode, target_col=target_col)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_training_steps = int(len(tokenized_dataset["train"])/BATCH_SIZE) + 1000
    num_warmup_steps = num_training_steps * 0.1
    #scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, NUM_CYCLES)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    #optimizer = accelerator.prepare(optimizer)
    #scheduler = accelerator.prepare(scheduler)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    #model, optimizer,scheduler = accelerator.prepare(model, optimizer,scheduler)
    # RERUN THINGS WITHOUT BELOW WHEN WE HAVE TIME
    #tokenized_dataset["train"] = tokenized_dataset["train"].select(range(min(1000, len(tokenized_dataset["train"]))))
    #tokenized_dataset["val"] = tokenized_dataset["val"].select(range(min(100, len(tokenized_dataset["val"]))))
    #tokenized_dataset["test"] = tokenized_dataset["test"].select(range(min(100, len(tokenized_dataset["test"]))))

    if 'intra' in filename: 
        # for the non-batch size stuff, we used a batch size of 5000, which worked horribly for the bigger models
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-intra-{dataset_mode}-{target_col.replace('_annots_str', '')}-cross-ent"
    else:
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-inter-{dataset_mode}-{target_col.replace('_annots_str', '')}-cross-ent"

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    training_args = TrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        #predict_with_generate=True, #comment out for sfttrainer
        #generation_config=self.model.generation_config, #commend out for sfttrainer
        #bf16=True,############3
        #fp16_full_eval=True,#########
        #dataloader_num_workers=accelerator.num_processes,
        learning_rate=LR,
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=1,
        evaluation_strategy="steps",
        #save_steps=100,
        num_train_epochs=50,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        push_to_hub=True,
        include_inputs_for_metrics=True,
        hub_strategy="every_save",
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
        remove_unused_columns = False,#"Mistral" in self.model_name,
        #compute_loss=compute_metrics,
        #generation_config=generation_config,
    )
    #training_args._n_gpu = 2
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
    def compute_metrics(p):
        pass
        '''
        # p is a tuple containing predictions for each task
        predictions1, predictions2 = p.predictions

        # Assuming predictions are logits; convert them to class probabilities
        probabilities1 = torch.nn.functional.softmax(predictions1, dim=-1)
        probabilities2 = torch.nn.functional.softmax(predictions2, dim=-1)

        # Assuming you have labels for each task in the dataset
        labels1 = p.label_ids1
        labels2 = p.label_ids2

        # Calculate accuracy for each task
        correct1 = (torch.argmax(probabilities1, dim=-1) == labels1).float()
        accuracy1 = correct1.mean().item()

        correct2 = (torch.argmax(probabilities2, dim=-1) == labels2).float()
        accuracy2 = correct2.mean().item()

        # You can return any metrics you're interested in
        return {"accuracy1": accuracy1, "accuracy2": accuracy2}
        '''

    if distributed_type == DistributedType.DEEPSPEED:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            callbacks=[early_stopping],
        )
        #optim_scheduler = self.trainer.create_optimizer_and_scheduler(num_training_steps=10) #num_training_steps) ########################################################
        #trainer.optimizer = optim_scheduler[0]
        #trainer.lr_scheduler = optim_scheduler[1]
        trainer.optimizer = optimizer
        trainer.lr_scheduler = scheduler
        
    else:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            callbacks=[early_stopping],
            optimizers=(optimizer, scheduler),
            #compute_metrics=,
        )

    trainer.train()

    model.save_pretrained(f'{repository_id}.pt') 
    tokenizer.save_pretrained(f'{repository_id}.pt')
    trainer.create_model_card()
    trainer.push_to_hub()

    #print(trainer.evaluate(eval_dataset=tokenized_dataset["test"]))
    #print("-------PREDICT-----------")
    p = trainer.predict(tokenized_dataset["test"])

    pkl_filename = "test.pkl"

    with open(pkl_filename, 'wb') as f:
        pickle.dump(p[1], f)
    #print(len(p['predictions']), len(p['predictions'][0]))
    #print(len(p['label_ids']), len(p['label_ids'][0]))
    with open(pkl_filename, 'rb') as f:
        p = pickle.load(f)

    correct = 0
    total = 0
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
    '''
    # Example data (you would replace these with your own datasets)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

    # Forward pass
    output1, output2 = model(input_ids, attention_mask)

    # Joint training with multiple losses
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    # Example targets (you would replace these with your own targets)
    target1 = torch.tensor([0])
    target2 = torch.tensor([1])

    # Calculate losses
    loss1 = criterion1(output1, target1)
    loss2 = criterion2(output2, target2)

    # Total loss for joint training
    total_loss = loss1 + loss2

    # Backpropagation
    total_loss.backward()

    # Update model parameters
    optimizer.step()
    ''
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
        self.model, optimizer, self.tokenized_dataset['train'], scheduler = accelerator.prepare(self.model, optimizer, self.tokenized_dataset["train"], scheduler)
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
model_id = "roberta-large"
#for dn in ['SChem5Labels', 'Sentiment', 'SBIC']:
for dn in ['ghc']:
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
        '''
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
