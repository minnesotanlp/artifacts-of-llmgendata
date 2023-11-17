import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from accelerate import Accelerator
accelerator = Accelerator()
import nltk 
import random
from collections import Counter
BATCH_SIZE = -1
RANDOM_SEED = 42

def get_batch_size(dataset_name):
    if dataset_name in ['SChem5Labels', 'Sentiment']:
        BATCH_SIZE = 64
    elif dataset_name in ['SBIC', 'ghc']:
        BATCH_SIZE = 16
    else:
        raise Exception("dataset_name not supported or not entered")
    return BATCH_SIZE

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
    return num_labels + 1

def format_dataset(filename, dataset_name, mode="sorted"):
    np.random.seed(RANDOM_SEED)
    df = pd.read_csv(filename)
    df = df[df['dataset_name'] == dataset_name]
    df.reset_index(inplace=True)
    # since nans are already removed here (still checking below just in case), we may have lists that are shorter than expected
    if mode == "sorted":
        for col in ['human_annots', 'model_annots']:
            df[col] = df[col].apply(lambda x: sorted([i if i != 'nan' else -1 for i in np.fromstring(x[1:-1].replace('.',''), dtype=int, sep=' ')]))
            df[f'{col}_str'] = df[col].apply(lambda x: " ".join([str(i) for i in x]))
    elif "dataset-frequency" in mode:# [frequency, reverse_frequency]
        for col in ['human_annots', 'model_annots']:
            all_annots = [row[1:-1].replace(" ", "").replace("nan", "").replace(".", "") for row in df[col]]
            all_annots = ''.join(all_annots)
            all_annots = [annot for annot in all_annots]
            freq_dict = dict(Counter(all_annots))
            freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=(mode=="dataset-frequency")))
            print('=========', freq_dict)
            for i in range(df[col].shape[0]):
                this_str = df[col][i][1:-1].replace(" ", "").replace("nan", "").replace(".", "")
                # adding 1 of each label so it shows up in final string - won't mess up frequency order
                #this_str += ''.join([str(num) for num in range(get_num_labels(dataset_name))])
                row_freq_dict = dict(Counter([el for el in this_str]))
                #print("THIS STR", this_str)
                #print("freq_dict", freq_dict)
                #row_freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=(mode=="frequency")))
                #print("row_freq_dict", row_freq_dict)
                new_str = ''.join([str(k)*row_freq_dict.get(k, 0) for k in freq_dict.keys()])
                #print("NEW_STR", new_str)
                df[col][i] = list(new_str)
            print("GOT HERE?")
            df[f'{col}_str'] = df[col].apply(lambda x: (' '.join(list(x))))
            print(df[f'{col}_str'][:10])
    elif "frequency" in mode:# [frequency, reverse_frequency]
        for col in ['human_annots', 'model_annots']:
            for i in range(df[col].shape[0]):
                this_str = df[col][i][1:-1].replace(" ", "").replace("nan", "").replace(".", "")
                # adding 1 of each label so it shows up in final string - won't mess up frequency order
                #this_str += ''.join([str(num) for num in range(get_num_labels(dataset_name))])
                freq_dict = dict(Counter([row for row in this_str]))
                freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=(mode=="frequency")))
                new_str = ' '.join([str(k)*freq_dict[k] for k in freq_dict.keys()])
                df[col][i] = list(new_str)
            print("GOT HERE?")
            df[f'{col}_str'] = df[col].apply(lambda x: (' '.join(list(x))))
    elif mode == "shuffle":
        for col in ['human_annots', 'model_annots']:
            for i in range(df[col].shape[0]):
                x = df[col][i][1:-1].replace('.','').split()
                x = [el if el != 'nan' else -1 for el in x]
                random.shuffle(x)
                df[col][i] = [str(el) for el in x]
            df[f'{col}_str'] = df[col].apply(lambda x: (' '.join(list(x))))
    # remove last sentence fragment for multi-label tasks
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
    return DatasetDict({
        "train": Dataset.from_pandas(train_data),
        "val": Dataset.from_pandas(val_data),
        "test": Dataset.from_pandas(test_data)
    })

def get_data(filename, dataset_name, mode="sorted"):
    # returns DatasetDict with train, val, test
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

def get_tokenized_data(filename, dataset, tokenizer, col_for_num_labels, remove_columns, mode="sorted", target_col="model_annots_str"):
    def preprocess_function(sample, target=target_col):
    #def preprocess_function(sample, padding="max_length", target="model_annots_str", max_target_length=32):
        inputs = []
        for i in range(len(sample[col_for_num_labels])):
            #prompt = f"##### Predict a set of annotations from {len(sample[col_for_num_labels][i])} different people #####\n" 
            #prompt += sample['short_prompt'][i]
            #prompt += "Answer: The set of annotations from {len(sample[col_for_num_labels][i])} different people is ["
            prompt = 'Multi-label classification results: '+sample['text'][i]
            inputs.append(prompt)
        tokenized = tokenizer(inputs, truncation=True)
        max_source_length = max([len(x) for x in tokenized["input_ids"]])
        model_inputs = tokenizer(inputs, truncation=True)#, padding=True)
        labels = tokenizer(sample[target], truncation=True, add_special_tokens=False)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        #if padding == "max_length":
        #    labels["input_ids"] = [
        #        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #    ]
        model_inputs["short_prompt"] = inputs
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["decoder_input_ids"] = model_inputs["input_ids"]
        return model_inputs

    data = get_data(filename, dataset, mode)
    # TODO: maybe pass in columns to remove
    tokenized_data = data.map(
            preprocess_function, 
            batched=True, 
            batch_size=BATCH_SIZE,
            num_proc=accelerator.num_processes, remove_columns=remove_columns)
    return tokenized_data

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    edit_distances = []
    assert len(decoded_preds) == len(decoded_labels)
    for i in range(len(decoded_preds)):
        s1 = decoded_preds[i]
        s2 = decoded_labels[i]
        edit_distances.append(nltk.edit_distance(s1, s2))

    # Some simple post-processing
    #######################BEFORE AND AFTER ARE THE SAME #########################
    return {'edit_distance': np.mean(edit_distances)}

