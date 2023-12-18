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
                loss += self.alpha * ce + (1-self.alpha) * dist 
        return (loss, outputs) if return_outputs else loss
    '''
    def predict_risako(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)
    '''

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

    #print('repository_id-----------------', repository_id)
    #print('before tokenized_dataset', len(tokenized_dataset['train']), len(tokenized_dataset['val']), len(tokenized_dataset['test']))
    #for i in range(3):
    #    print(tokenized_dataset['train'][target_col][i])
    #print(sum(utils.flatten_recursive(tokenized_dataset['train'][target_col.replace('_str', '')])))

    #if os.path.exists(f"results_new/{repository_id}.pkl"):
    #    print('already done', repository_id)
    #    return
    max_size = 5000
    # take subset of data if training is larger than max_size 
    if False:# and len(tokenized_dataset["train"]) > max_size:
        # choose random numbers between 0 and len(tokenized_dataset[split])
        ind_filename = f'{dataset_name}_indices.pkl'
        if os.path.exists(ind_filename):
            with open(ind_filename, 'rb') as f:
                train_ind, val_ind, test_ind = pickle.load(f)
            tokenized_dataset["train"] = tokenized_dataset["train"].select(train_ind)
            tokenized_dataset["val"] = tokenized_dataset["val"].select(val_ind)
            tokenized_dataset["test"] = tokenized_dataset["test"].select(test_ind)
        else:
            random.seed(SEED)
            train_ind = random.sample(range(len(tokenized_dataset["train"])), max_size)
            val_ind = random.sample(range(len(tokenized_dataset["val"])), max_size*0.1)
            test_ind = random.sample(range(len(tokenized_dataset["test"])), max_size*0.1)
            with open(ind_filename, 'wb') as f:
                pickle.dump([train_ind, val_ind, test_ind], f)
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
        evaluation_strategy="epoch",
        num_train_epochs=3,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        #metric_for_best_model="f1",
        #greater_is_better=True,
        report_to="wandb",
        #push_to_hub=True,
        include_inputs_for_metrics=True,
        hub_strategy="checkpoint",
        hub_model_id=f'{repository_id}_{suffix}',
        hub_token=HfFolder.get_token(),
        #remove_unused_columns = False,#"Mistral" in self.model_name,
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
        trainer.push_to_hub()#repo_id=f'{repository_id}_{suffix}.pt')#, use_auth_token=HfFolder.get_token()
    except Exception as e:
        trainer.save_model(f'{repository_id}_{suffix}.pt')
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Failed to push to hub", repository_id)
        print(e)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    p = []
    model = model.to(device)
    with torch.no_grad():
        for row in tokenized_dataset['test']:
            res = model(torch.tensor(row['input_ids']).to(device).unsqueeze(0), torch.tensor(row['attention_mask']).to(device).unsqueeze(0))
            p.append([el.cpu().numpy() for el in res])
    filename = f"results_new/{repository_id}_{suffix}.pkl" # with the same indices
    with open(filename, 'wb') as f:
        pickle.dump(p, f)
    '''
    test_loader = DataLoader(tokenized_dataset['test'], batch_size=32)
    with torch.no_grad():
        res = []
        for batch in test_loader:
            outputs = model(torch.stack(batch['input_ids'], dim=0).to(device), torch.stack(batch['attention_mask'], dim=0).to(device))
            res += outputs

    filename = f"results_new/{repository_id}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(res, f)
    '''
    '''

    #print("before predict!!!!!!!!!!!!!!!!!!!!!!!!!!!!!AND HEREEEEE")
    #print(tokenized_dataset['test'][0])
    p = trainer.predict(test_dataset=tokenized_dataset["test"])
    #print('after predict`@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    #print(len(model.classifiers))
    #print(np.array(p.predictions).shape)
    #print(np.array(p.label_ids).shape)
    #print('............................................')
    e = trainer.evaluate(tokenized_dataset["test"])
    filename = f"results_new/{repository_id}_{suffix}.pkl" # with the same indices
    # orig: alpha=0.5
    # 2: alpha=0.8


    print(p[1])
    print(e)

    with open(filename, 'wb') as f:
        pickle.dump([p, e], f)
    '''

if __name__ == "__main__":
    use_bash = True 
    if use_bash:
        # get args
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_id", type=str, default="roberta-base")
        parser.add_argument("--dataset_name", type=str, default="SChem5Labels")
        parser.add_argument("--filename", type=str, default="../data/intermodel_data.csv")
        parser.add_argument("--dataset_mode", type=str, default="data-frequency")
        #parser.add_argument("--dataset_mode", type=str, default="sorted")
        parser.add_argument("--target_col", type=str, default="model_annots")
        parser.add_argument("--alpha", type=float, default=0.8)
        args = parser.parse_args()
        if args.filename == "../data/intramodel_data.csv":
            args.remove_columns = ['index', 'dataset_name', 'text', 'text_ind', 'prompt', 'params'] #, 'human_annots'
        else:
            args.remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_name', 'text', 'index'] #model_annots
        main(args.filename, args.model_id, args.dataset_name, args.remove_columns, args.dataset_mode, args.target_col, args.alpha)
    else:
        for dataset_name in ['SBIC', 'ghc', 'SChem5Labels', 'Sentiment'][::-1]:
        #for dataset_name in ['SBIC', 'ghc']:
        #for dataset_name in ['SChem5Labels']:
            #for m in ['frequency']:#, 'shuffle', 'sorted']:
            for m in ['sorted', 'shuffle', 'frequency', 'data-frequency']:
                #for m in ['shuffle']:#, 'sorted']:
                for target_col in ['human_annots', 'model_annots']:
                    #first_order([dataset_name], 'minority')
                    #first_order([dataset_name], 'all')
                    #second_order([dataset_name])

                    #if False:
                    if True:
                        main(filename = '../data/intermodel_data.csv',
                             model_id = 'roberta-base',
                             dataset_name = dataset_name,
                             remove_columns = ['dataset_name', 'text_ind', 'prompt'],#, 'model_annots'],
                             dataset_mode = m,
                             target_col = target_col)
                        raise Exception("stop")
                    if target_col == 'model_annots':
                        main(filename = '../data/intramodel_data.csv',
                             model_id = 'roberta-base',
                             dataset_name = dataset_name,
                             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params'],#, 'human_annots'],
                             dataset_mode = m,
                             target_col = target_col)

    ########################################################################
    ########################################################################
    ########################################################################
    ''' 
    #for dn in ['SChem5Labels', 'Sentiment', 'SBIC', 'ghc']:
    #for m in ['frequency', 'dataset-frequency']:
    for m in ['sorted']:
    #for m in ['dataset-frequency']:
        main(filename = '../data/intramodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['index', 'dataset_name', 'text', 'text_ind', 'prompt', 'params', 'human_annots', 'model_annots'],
             target_col='model_annots',
             dataset_mode = m)
        main(filename = '../data/intermodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
             target_col='model_annots',
             dataset_mode = m)
        main(filename = '../data/intramodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
             dataset_mode = m,
             target_col='human_annots')
        main(filename = '../data/intermodel_data.csv', 
             model_id = model_id,
             dataset_name = dn,
             remove_columns = ['dataset_name', 'text_ind', 'prompt', 'model_annots'],
             dataset_mode = m,
             target_col = "human_annots")
    '''
                        
