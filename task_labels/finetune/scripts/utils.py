import numpy as np
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator
import random
from collections import Counter
BATCH_SIZE = -1
RANDOM_SEED = 42
from accelerate import Accelerator, DistributedType, DeepSpeedPlugin
from accelerate.state import AcceleratorState

def get_batch_size(dataset_name):
    # 1 when model is xl+
    if dataset_name in ['SChem5Labels', 'Sentiment']:
        BATCH_SIZE = 32
    elif dataset_name in ['SBIC', 'ghc']:
        BATCH_SIZE = 32
    else:
        raise Exception("dataset_name not supported or not entered")
    return BATCH_SIZE

def get_deepspeed_plugin():
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=2)
    return deepspeed_plugin

def get_accelerator(deepspeed_plugin=None):
    #if not deepspeed_plugin:
    #    deepspeed_plugin = get_deepspeed_plugin()
    #print(deepspeed_plugin)
    #accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)
    accelerator = Accelerator()
    return accelerator
accelerator = get_accelerator() 


def seed_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def get_num_annots(dataset_name):
    if dataset_name in ['SChem5Labels']:
        num_annots = 5
    elif dataset_name in ['Sentiment']:
        num_annots = 4
    elif dataset_name in ['SBIC', 'ghc']:
        num_annots = 3
    else:
        raise Exception("dataset_name not supported or not entered")
    return num_annots

def get_num_labels(dataset_name):
    if dataset_name in ['SChem5Labels', 'Sentiment']:
        num_labels = 5
    elif dataset_name in ['SBIC']:
        num_labels = 3
    elif dataset_name in ['ghc']:
        num_labels = 2
    else:
        raise Exception("dataset_name not supported or not entered")
    # plus 1 for the "other" labels
    return num_labels# + 1

def str_to_lst(x):
    # assume string that looks like list of numbers e.g. "[1, 2, 3]"
    if type(x) == list:
        x = ''.join([str(el) for el in x])
    if x[0] == '[':
        x = x[1:-1]
    return list(x.replace(" ", "").replace("nan", "").replace(".", "").replace("'", "").replace('"', ""))       

def str_to_num_lst(x):
    x = str_to_lst(x)
    return [int(el) for el in x]

def flatten_recursive(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_recursive(item))
        else:
            result.append(item)
    return result

def format_dataset_roberta(filename, dataset_name, mode="sorted"):
    np.random.seed(RANDOM_SEED)
    num_annots = get_num_annots(dataset_name)
    df = pd.read_csv(filename)
    df = df[df['dataset_name'] == dataset_name]

    df.reset_index(inplace=True)
    # since nans are already removed here (still checking below just in case), we may have lists that are shorter than expected

    # convert string to list, make sure num annotations match
    for col in ['human_annots', 'model_annots']:
        for i in range(df[col].shape[0]):
            df[col][i] = str_to_lst(df[col][i])
            if len(df[col][i]) > num_annots:
                random.seed(RANDOM_SEED*i)
                idx = random.sample(range(len(df[col][i])), k=num_annots)
                df[col][i] = [x for i, x in enumerate(df[col][i]) if i in idx]

    if mode == "sorted":
        for col in ['human_annots', 'model_annots']:
            df[col] = df[col].apply(lambda x: sorted([int(el) for el in x]))
    elif "dataset-frequency" in mode:# [frequency, reverse_frequency]
        for col in ['human_annots', 'model_annots']:
            all_annots = [str_to_lst(row) for row in df[col]]
            all_annots = [item for row in all_annots for item in row]
            freq_dict = dict(Counter(all_annots))
            freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=(mode=="dataset-frequency")))
            for i in range(df[col].shape[0]):
                this_str = ''.join(df[col][i])
                row_freq_dict = dict(Counter([el for el in this_str]))
                new_str = ''.join([str(k)*row_freq_dict.get(k, 0) for k in freq_dict.keys()])
                df[col][i] = [int(el) for el in new_str]
    elif "frequency" in mode:# [frequency, reverse_frequency]
        # this is probably almost similar to shuffle when people are disagreeing
        for col in ['human_annots', 'model_annots']:
            for i in range(df[col].shape[0]):
                this_str = (df[col][i])
                freq_dict = dict(Counter([row for row in this_str]))
                freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=(mode=="frequency")))
                new_str = ''.join([str(k)*freq_dict[k] for k in freq_dict.keys()])
                df[col][i] = [int(el) for el in new_str]
    elif mode == "shuffle":
        for col in ['human_annots', 'model_annots']:
            for i in range(df[col].shape[0]):
                x = str_to_lst(df[col][i])
                random.shuffle(x)
                df[col][i] = [int(el) for el in x]
    # pad/truncate here since we always want the padding to come at the end
    for col in ['human_annots', 'model_annots']:
        for i in range(df[col].shape[0]):
            if len(df[col][i]) < num_annots:
                df[col][i] += [-1]*(num_annots - len(df[col][i]))
    return df

def split(df, dataset_name, grouping):
    suffix = dataset_name + "_" + grouping 
    if os.path.exists(f"train_data_{suffix}.pkl"):
        train_data = pd.read_pickle(f"train_data_{suffix}.pkl")
        val_data = pd.read_pickle(f"val_data_{suffix}.pkl")
        test_data = pd.read_pickle(f"test_data_{suffix}.pkl")
    else:
        # count rows in each df
        testing = False
        if testing:
            train_data = df.sample(frac=0.01, random_state=RANDOM_SEED*3)
            val_data = train_data
            #df.sample(frac=0.1, random_state=42)
            test_data = val_data
            #df.sample(frac=0.1, random_state=42)
        else:
            train_data = df.sample(frac=0.8, random_state=RANDOM_SEED)
            val_data = df.drop(train_data.index).sample(frac=0.5, random_state=RANDOM_SEED*2)
            test_data = df.drop(train_data.index).drop(val_data.index)
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        train_data.to_pickle(f"train_data_{suffix}.pkl")
        val_data.to_pickle(f"val_data_{suffix}.pkl")
        test_data.to_pickle(f"test_data_{suffix}.pkl")
        train_data['model_annots'] = train_data['model_annots'].astype(str)
        val_data['model_annots'] = val_data['model_annots'].astype(str)
        test_data['model_annots'] = test_data['model_annots'].astype(str)
    return DatasetDict({
        "train": Dataset.from_pandas(train_data),
        "val": Dataset.from_pandas(val_data),
        "test": Dataset.from_pandas(test_data)
    })

def get_data(filename, dataset_name, mode="sorted", model_id="roberta-base"):
    # returns DatasetDict with train, val, test
    model_df = format_dataset_roberta(filename, dataset_name, mode)
    grouping = 'inter' if 'inter' in filename else 'intra'
    dataset = split(model_df, dataset_name, grouping)
    #print(f"Train dataset size: {len(intra_dataset['train'])}")
    #print(f"Test dataset size: {len(inter_dataset['test'])}")
    return dataset

def get_dataloader(filename):
    # untested
    data = get_data(filename)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn = seed_init_fn)
    return dataloader

def get_tokenized_data(filename, dataset, tokenizer, remove_columns, mode="sorted", target_col="model_annots_str", model_id="roberta-base"):
    grouping = 'inter' if 'inter' in filename else 'intra'
    num_labels = get_num_labels(dataset) # -1 for the extra "other" label
    num_annots = get_num_annots(dataset)
    def preprocess_function(sample, target=target_col):
        inputs = sample['text']
        tokenized = tokenizer(inputs, truncation=True, padding=True)
        max_source_length = max([len(x) for x in tokenized["input_ids"]])
        model_inputs = tokenizer(inputs, truncation=True, padding=True, max_length=max_source_length)
        # convert all missing/invalid values to -1 here
        # sample[target] should be a list of labels
        if type(sample[target][0]) == str:
            sample[target] = [eval(row) for row in sample[target]]
        model_inputs['labels'] = np.array(sample[target]).astype(int).tolist()
        for row_i in range(len(sample[target])):
            for annot_i in range(len(sample[target][row_i])):
                if not (0 <= int(sample[target][row_i][annot_i]) and int(sample[target][row_i][annot_i]) < num_labels):
                    model_inputs['labels'][row_i][annot_i] = -1
            model_inputs['labels'][row_i] += [-1]*(num_annots - len(model_inputs['labels'][row_i]))
                #else:
                #    model_inputs[row_i][annot_i] = int(sample[target][row_i][annot_i])
        #raise Exception("roberta-base not supported yet")
        #model_inputs['labels'] = [(el if (0 <= el and el < num_labels) else -1 for el in sample[target]]
        #model_inputs['labels'] += [-1]*(num_annots - len(model_inputs['labels']))
        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).to(accelerator.device)
        return model_inputs
    data = get_data(filename, dataset, mode, model_id)
    # TODO: maybe pass in columns to remove
    tokenized_data = data.map(
            preprocess_function, 
            batched=True, 
            batch_size=BATCH_SIZE,
            remove_columns=remove_columns,
            )#num_proc=accelerator.num_processes, )
    return tokenized_data
