import numpy as np
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

def format_dataset_roberta(filename, dataset_name, mode="sorted"):
    np.random.seed(RANDOM_SEED)
    num_annots = get_num_annots(dataset_name)
    print("========================FORMAT DATASET ROBERTA============")
    print("dataset_name", dataset_name)
    print("num_annots", num_annots)
    print("num_labels", get_num_labels(dataset_name))
    df = pd.read_csv(filename)
    df = df[df['dataset_name'] == dataset_name]

    df.reset_index(inplace=True)
    # since nans are already removed here (still checking below just in case), we may have lists that are shorter than expected

    # random sample here so the new ordering gets reflected in the ordering
    for col in ['human_annots', 'model_annots']:
        for i in range(df[col].shape[0]):
            df[col][i] = str_to_lst(df[col][i])
            if len(df[col][i]) > num_annots:
                idx = random.sample(range(len(df[col][i])), k=num_annots)
                df[col][i] = [x for i, x in enumerate(df[col][i]) if i in idx]
    print("ORIGINAL", df['human_annots'][0])
    print("ORIGINAL", df['model_annots'][0])
    if mode == "sorted":
        for col in ['human_annots', 'model_annots']:
            #df[col] = df[col].apply(lambda x: sorted([i if i != 'nan' else -1 for i in np.fromstring(x[1:-1].replace('.',''), dtype=int, sep=' ')]))
            df[col] = df[col].apply(lambda x: sorted([int(el) for el in x]))
    elif "dataset-frequency" in mode:# [frequency, reverse_frequency]
        for col in ['human_annots', 'model_annots']:
            all_annots = np.array([str_to_lst(row) for row in df[col]]).flatten()
            all_annots = ''.join(all_annots)
            all_annots = [annot for annot in all_annots]
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

    # pad/truncate here since we always want the padding to come at the end
    for col in ['human_annots', 'model_annots']:
        for i in range(df[col].shape[0]):
            if len(df[col][i]) < num_annots:
                df[col][i] += [-1]*(num_annots - len(df[col][i]))
    return df

def format_dataset(filename, dataset_name, mode="sorted"):
    np.random.seed(RANDOM_SEED)
    df = pd.read_csv(filename)
    df = df[df['dataset_name'] == dataset_name]
    df.reset_index(inplace=True)
    # since nans are already removed here (still checking below just in case), we may have lists that are shorter than expected
    if True or mode == "sorted":
        for col in ['human_annots', 'model_annots']:
            df[col] = df[col].apply(lambda x: sorted([i if i != 'nan' else -1 for i in np.fromstring(x[1:-1].replace('.',''), dtype=int, sep=' ')]))
            df[f'{col}_str'] = df[col].apply(lambda x: " ".join([str(i) for i in x]))
            print('sorted', df[f'{col}_str'][0])
    if True or "dataset-frequency" in mode:# [frequency, reverse_frequency]
    #elif "dataset-frequency" in mode:# [frequency, reverse_frequency]
        for col in ['human_annots', 'model_annots']:
            all_annots = [row[1:-1].replace(" ", "").replace("nan", "").replace(".", "") for row in df[col]]
            all_annots = ''.join(all_annots)
            all_annots = [annot for annot in all_annots]
            freq_dict = dict(Counter(all_annots))
            freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=(mode=="dataset-frequency")))
            for i in range(df[col].shape[0]):
                this_str = df[col][i][1:-1].replace(" ", "").replace("nan", "").replace(".", "")
                # adding 1 of each label so it shows up in final string - won't mess up frequency order
                row_freq_dict = dict(Counter([el for el in this_str]))
                #row_freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=(mode=="frequency")))
                #print("row_freq_dict", row_freq_dict)
                new_str = ''.join([str(k)*row_freq_dict.get(k, 0) for k in freq_dict.keys()])
                #print("NEW_STR", new_str)
                df[col][i] = list(new_str)
            df[f'{col}_str'] = df[col].apply(lambda x: (' '.join(list(x))))
            print('dataset freq', df[f'{col}_str'][0])
    #elif "frequency" in mode:# [frequency, reverse_frequency]
    if True or "frequency" in mode:# [frequency, reverse_frequency]
        for col in ['human_annots', 'model_annots']:
            for i in range(df[col].shape[0]):
                this_str = df[col][i][1:-1].replace(" ", "").replace("nan", "").replace(".", "")
                # adding 1 of each label so it shows up in final string - won't mess up frequency order
                freq_dict = dict(Counter([row for row in this_str]))
                freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=(mode=="frequency")))
                new_str = ' '.join([str(k)*freq_dict[k] for k in freq_dict.keys()])
                df[col][i] = list(new_str)
            df[f'{col}_str'] = df[col].apply(lambda x: (' '.join(list(x))))
            print('frequency', df[f'{col}_str'][0])
    if True or mode == "shuffle":
    #elif mode == "shuffle":
        for col in ['human_annots', 'model_annots']:
            for i in range(df[col].shape[0]):
                x = df[col][i][1:-1].replace('.','').split()
                x = [el if el != 'nan' else -1 for el in x]
                random.shuffle(x)
                df[col][i] = [str(el) for el in x]
            df[f'{col}_str'] = df[col].apply(lambda x: (' '.join(list(x))))
            print('shuffle', df[f'{col}_str'][0])
    # remove last sentence fragment for multi-label tasks
    raise Exception()
    df['short_prompt'] = df['text'].apply(lambda x: 'Multi-label classification results: ' + x)
            #.replace("Sentence: ", "##### Sentence #####\n")\
            #.replace("Label options: ", "\n##### Labels options #####\n"))
    return df

def split(df, suffix=''):
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
    if model_id == "roberta-base":
        model_df = format_dataset_roberta(filename, dataset_name, mode)
    else:
        model_df = format_dataset(filename, dataset_name, mode)
    dataset = split(model_df, dataset_name)
    #print(f"Train dataset size: {len(intra_dataset['train'])}")
    #print(f"Test dataset size: {len(inter_dataset['test'])}")
    return dataset

def get_dataloader(filename):
    # untested
    data = get_data(filename)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn = seed_init_fn)
    return dataloader

def get_tokenized_data(filename, dataset, tokenizer, col_for_num_labels, remove_columns, mode="sorted", target_col="model_annots_str", model_id="roberta-base"):
    num_labels = get_num_labels(dataset) # -1 for the extra "other" label
    num_annots = get_num_annots(dataset)
    def preprocess_function(sample, target=target_col):
    #def preprocess_function(sample, padding="max_length", target="model_annots_str", max_target_length=32):
        if model_id != "roberta-base":
            inputs = []
            for i in range(len(sample[col_for_num_labels])):
                #prompt = f"##### Predict a set of annotations from {len(sample[col_for_num_labels][i])} different people #####\n" 
                #prompt += sample['short_prompt'][i]
                #prompt += "Answer: The set of annotations from {len(sample[col_for_num_labels][i])} different people is ["
                prompt = 'Multi-label classification results: '+sample['text'][i]
                inputs.append(prompt)
            model_inputs["short_prompt"] = inputs
        else:
            inputs = sample['text']
        tokenized = tokenizer(inputs, truncation=True, padding=True)
        max_source_length = max([len(x) for x in tokenized["input_ids"]])
        model_inputs = tokenizer(inputs, truncation=True, padding=True, max_length=max_source_length)
        if model_id == "roberta-base":
            # convert all missing/invalid values to -1 here
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
        else:
            labels = tokenizer(sample[target], truncation=True, padding=True, add_special_tokens=False)
            model_inputs["labels"] = labels["input_ids"]
        #model_inputs["label"] = labels["input_ids"] #did we need this for mistral
        #model_inputs["label_ids"] = labels["input_ids"] #same concern as above
        if 't5' in model_id:
            model_inputs["decoder_input_ids"] = model_inputs["input_ids"]
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
