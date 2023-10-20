import numpy as np
from pandas import read_csv
from datasets import Dataset, DatasetDict

def format_dataset(filename):
    df = read_csv(filename)
    # since nans are already removed here (still checking below just in case), we may have lists that are shorter than expected
    for col in ['human_annots', 'model_annots']:
        df[col] = df[col].apply(lambda x: sorted([i if i != 'nan' else -1 for i in np.fromstring(x[1:-1].replace('.',''), dtype=int, sep=' ')]))
        df[f'{col}_str'] = df[col].apply(lambda x: ' '.join([str(i) for i in x]))
    df['short_prompt'] = df['prompt'].apply(lambda x: x[(x.index("Sentence: ")):])
            #.replace("Sentence: ", "##### Sentence #####\n")\
            #.replace("Label options: ", "\n##### Labels options #####\n"))
    return df

def split(df):
    # count rows in each df
    train_data = df.sample(frac=0.1, random_state=42)
    val_data = df.sample(frac=0.1, random_state=42)
    test_data = df.sample(frac=0.1, random_state=42)
    #train_data = df.sample(frac=0.8, random_state=42)
    #val_data = df.drop(train_data.index).sample(frac=0.5, random_state=42)
    #test_data = df.drop(train_data.index).drop(val_data.index)
    return DatasetDict({
        "train": Dataset.from_pandas(train_data),
        "val": Dataset.from_pandas(val_data),
        "test": Dataset.from_pandas(test_data)
    })

def get_data(filename):
    model_df = format_dataset(filename)
    print("=====================+" + filename + "=====================")
    print(model_df)
    dataset = split(model_df) 
    return dataset
