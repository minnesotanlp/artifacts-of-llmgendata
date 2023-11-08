import torch
torch.cuda.empty_cache()
import os
os.environ['VISIBLE_CUDA_DEVICES'] = '0,1'
import accelerate
import numpy as np
from torch.utils.data import DataLoader
from peft import PeftModel    
from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        AutoModelForSeq2SeqLM,
        AutoTokenizer, 
        StoppingCriteria, 
        StoppingCriteriaList, 
        TextIteratorStreamer)
import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import visualize
import pickle
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

dataset_name = 'ghc'
# Load the pre-trained model and tokenizer
#model_name = f"owanr/{dataset_name}-google-t5-v1_1-large-intra_model-sorted"
model_id = "google/t5-v1_1-large"
tokenizer = T5Tokenizer.from_pretrained(model_id)
acc = accelerate.Accelerator()
my_config = {}
#my_config['renormalize_logits'] = True
my_config['return_dict_in_generate'] = True
def restrict_decode_vocab(a, b):
    return [209, 204, 220, 314, 305]
my_config["prefix_allowed_tokens_fn"] = restrict_decode_vocab #not json serializable
my_config['max_new_tokens'] = 5
my_config['min_new_tokens'] = 5
#my_config['bos_token_id'] = 0

def dataset_hist(cat):
    with open(f'../data/test_data_{cat}_{dataset_name}.pkl', 'rb') as f:
        test_data = pickle.load(f)
    human_annots = []
    for ha in test_data['human_annots']:
        ha = ha[1:-1].replace("nan","").replace(".","").split()
        ha = [int(x) for x in ha]
        human_annots += ha
    model_annots = []
    for ma in test_data['model_annots']:
        ma = ma[1:-1].replace("nan","").replace(".","").split()
        ma = [int(x) for x in ma]
        model_annots += ma
    visualize.create_hist_from_lst(human_annots, num_labels=5, title=f'{dataset_name}_human_labels')
    visualize.create_hist_from_lst(model_annots, num_labels=5, title=f'{dataset_name}_model_labels')

# doesn't matter   
#dataset_hist('inter')
#dataset_hist('intra')

def create_hist():
    for cat in ['inter', 'intra']:
        #for dataset_mode in ['sorted', 'shuffle', 'dataset-frequency', 'frequency']:
        torch.cuda.empty_cache()
        for dataset_mode in ['dataset-frequency', 'frequency']:
            torch.cuda.empty_cache()
            for target_col in ['human_annots_str', 'model_annots_str']:
                hf_model = f"owanr/{dataset_name}-{model_id.replace('/','-')}-{cat}_model-{dataset_mode}-{target_col}"
                # check if file exsts
                title = f'{hf_model.replace("owanr/","")}'
                if os.path.exists(f"./png/{title}.png"):
                    print(f"Skipping {hf_model.replace('owanr/','')} since it exists already")
                    continue
                try:
                    model = T5ForConditionalGeneration.from_pretrained(hf_model, device_map="auto", load_in_8bit=True)
                    #model.to(acc.device)
                except Exception as e:
                    print(f"Failed to load {hf_model}")
                    print(e)
                    print("")
                    continue
                annots = []
                with open(f'../data/test_data_{cat}_{dataset_name}.pkl', 'rb') as f:
                    test_data = pickle.load(f)
                texts = test_data['text']

                inputs = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True, truncation=True)
                inputs['decoder_input_ids'] = inputs['input_ids'].clone()
                inputs = inputs.to(model.device)

                # Perform inference
                with torch.no_grad():
                    outputs = model.generate(**inputs, **my_config)

                # remove input portion of output
                output_ids = outputs[0][:, inputs['input_ids'].shape[-1]:]
                answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                for answer in answers:
                    annots += [int(x) for x in list(answer.replace("nan","").replace(".","").replace(' ',''))]

                # plot annots into simple histogram
                visualize.create_hist_from_lst(annots, num_labels=5, title=title)

create_hist()
