import numpy as np
from pandas import read_csv
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from accelerate import Accelerator
accelerator = Accelerator()
import nltk 
import random
from collections import Counter
BATCH_SIZE = 16 
RANDOM_SEED = 42

def get_batch_size():
    return BATCH_SIZE

def seed_init_fn(x):
    seed = args.seed + x
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

def get_num_labels(dataset_name):
    if dataset_name in ['SChem5Labels', 'Sentiment']:
        num_labels = 5
    elif dataset_name in ['SBIC']:
        num_labels = 3
    elif dataset_name in ['ghc']:
        num_labels = 2
    else:
        raise Exception("dataset_name not supported or not entered")
    return num_labels

def format_dataset(filename, dataset_name, mode="sorted"):
    np.random.seed(RANDOM_SEED)
    df = read_csv(filename)
    df = df[df['dataset_name'] == dataset_name]
    # since nans are already removed here (still checking below just in case), we may have lists that are shorter than expected
    if mode == "sorted":
        for col in ['human_annots', 'model_annots']:
            df[col] = df[col].apply(lambda x: sorted([i if i != 'nan' else -1 for i in np.fromstring(x[1:-1].replace('.',''), dtype=int, sep=' ')]))
            df[f'{col}_str'] = df[col].apply(lambda x: ' '.join([str(i) for i in x]))
    elif "frequency" in mode:# [frequency, reverse_frequency]
        for col in ['human_annots', 'model_annots']:
            for i in range(df[col].shape[0]):
                this_str = df[col][i][1:-1].replace(" ", "").replace("nan", "").replace(".", "")
                # adding 1 of each label so it shows up in final string - won't mess up frequency order
                #this_str += ''.join([str(num) for num in range(get_num_labels(dataset_name))])
                freq_dict = dict(Counter([row for row in this_str]))
                freq_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=(mode=="frequency")))
                new_str = ''.join([str(k)*freq_dict[k] for k in freq_dict.keys()])
                df[col][i] = list(new_str)
            df[f'{col}_str'] = df[col].apply(lambda x: (' '.join(list(x))))
    elif mode == "shuffle":
            df[col] = df[col].apply(lambda x: np.random.shuffle([i if i != 'nan' else -1 for i in np.fromstring(x[1:-1].replace('.',''), dtype=int, sep=' ')]))
            df[f'{col}_str'] = df[col].apply(lambda x: ' '.join([str(i) for i in x]))

    # remove last sentence fragment for multi-label tasks
    df['short_prompt'] = df['prompt'].apply(lambda x: x[(x.index("Sentence: ")):].replace("Among the given options, I think the most appropriate option is (",""))
            #.replace("Sentence: ", "##### Sentence #####\n")\
            #.replace("Label options: ", "\n##### Labels options #####\n"))
    return df

def split(df, suffix=''):
    # count rows in each df
    '''
    train_data = df.sample(frac=0.01, random_state=42)
    val_data = train_data
    #df.sample(frac=0.1, random_state=42)
    test_data = val_data
    #df.sample(frac=0.1, random_state=42)
    '''
    train_data = df.sample(frac=0.8, random_state=42)
    val_data = df.drop(train_data.index).sample(frac=0.5, random_state=42)
    test_data = df.drop(train_data.index).drop(val_data.index)
    #'''     
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    train_data.to_pickle(f"train_data{suffix}.pkl")
    val_data.to_pickle(f"val_data{suffix}.pkl")
    test_data.to_pickle(f"test_data{suffix}.pkl")
    return DatasetDict({
        "train": Dataset.from_pandas(train_data),
        "val": Dataset.from_pandas(val_data),
        "test": Dataset.from_pandas(test_data)
    })

def get_data(filename, dataset_name):
    # returns DatasetDict with train, val, test
    model_df = format_dataset(filename, dataset_name)
    print("=====================+" + filename + "=====================")
    print(model_df)
    dataset = split(model_df) 
    #print(f"Train dataset size: {len(intra_dataset['train'])}")
    #print(f"Test dataset size: {len(inter_dataset['test'])}")
    return dataset

def get_dataloader(filename):
    # untested
    data = get_data(filename)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn = seed_init_fn)
    return dataloader

def get_tokenized_data(filename, dataset, tokenizer, col_for_num_labels, remove_columns):
    def preprocess_function(sample, target="model_annots_str"):
    #def preprocess_function(sample, padding="max_length", target="model_annots_str", max_target_length=32):
        # target is just labels so we can hardcode it as 32
        # add prefix to the input for t5
        inputs = []
        for i in range(len(sample[col_for_num_labels])):
            #prompt = f"##### Predict a set of annotations from {len(sample[col_for_num_labels][i])} different people #####\n" 
            #prompt += sample['short_prompt'][i]
            #prompt += "Answer: The set of annotations from {len(sample[col_for_num_labels][i])} different people is ["
            prompt = sample['text'][i]
            inputs.append(prompt)
        tokenized = tokenizer(inputs, truncation=True)
        max_source_length = max([len(x) for x in tokenized["input_ids"]])
        model_inputs = tokenizer(inputs, truncation=True, padding=True)
        labels = tokenizer(text_target=sample[target], truncation=True, padding=True)
        #model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        #labels = tokenizer(text_target=sample[target], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        #if padding == "max_length":
        #    labels["input_ids"] = [
        #        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #    ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data = get_data(filename, dataset)
    # TODO: maybe pass in columns to remove
    tokenized_data = data.map(
            preprocess_function, 
            batched=True, 
            batch_size=BATCH_SIZE,
            num_proc=accelerator.num_processes, remove_columns=remove_columns)
    return tokenized_data

def compute_metrics(eval_preds, tokenizer, label_pad_token_id=-100):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
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

