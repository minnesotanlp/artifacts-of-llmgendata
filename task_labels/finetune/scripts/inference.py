# location of accelerate/deepspeed config file
#/home/risako/.cache/huggingface/accelerate/default_config.yaml
import random
import os
#os.environ['VISIBLE_CUDA_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
from scipy.stats import kendalltau, spearmanr
import gc
import torch
from pynvml import *
nvmlInit()
torch.cuda.init()
import utils
def memory(msg):
    print(msg)
    '''
    handle = nvmlDeviceGetHandleByIndex(0)
    res = nvmlDeviceGetUtilizationRates(handle)
    print(f'nvml gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    handle = nvmlDeviceGetHandleByIndex(1)
    res = nvmlDeviceGetUtilizationRates(handle)
    print(f'nvml gpu: {res.gpu}%, gpu-mem: {res.memory}%')

    print(torch.cuda.memory_summary(device="cuda:0", abbreviated=True))
    print(torch.cuda.memory_summary(device=1, abbreviated=True))
    '''
memory("FIRST")
gc.collect()
torch.cuda.empty_cache()
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import rankdata


# Load the pre-trained model and tokenizer
#model_name = f"owanr/{dataset_name}-google-t5-v1_1-large-intra_model-sorted"
model_id = "google/t5-v1_1-large"
tokenizer = T5Tokenizer.from_pretrained(model_id)
acc = accelerate.Accelerator()
def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                   type(obj.data).__name__,
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "",
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)


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

def get_second_order_annots():
    #for dataset_name in ['SChem5Labels', 'Sentiment', 'ghc', 'SBIC']:
    for dataset_name in ['ghc']:
        num_annots = utils.get_num_annots(dataset_name)
        num_labels = utils.get_num_labels(dataset_name)
        my_config = {}
        #my_config['renormalize_logits'] = True
        my_config['return_dict_in_generate'] = True
        def restrict_decode_vocab(a, b):
            return [209, 204, 220, 314, 305][:num_labels-1]
        my_config["prefix_allowed_tokens_fn"] = restrict_decode_vocab #not json serializable
        my_config['max_new_tokens'] = num_annots
        my_config['min_new_tokens'] = num_annots
        my_config['bos_token_id'] = 0

        # get human labels here
        for cat in ['inter', 'intra']:
            # load pkl file
            pkl_file = f'../data/test_data_{cat}_{dataset_name}.pkl'
            if not os.path.exists(pkl_file):
                print(f"Skipping {pkl_file} since it doesn't exist")
                continue
            with open(pkl_file, 'rb') as f:
                test_data = pickle.load(f)
            random.seed(42)
            human_annots = []
            for ha in test_data['human_annots']:
                ha = ha[1:-1].replace("'","").replace("nan","").replace(".","").split()
                ha = random.choices([int(x) for x in ha], k=num_annots)
                #if len(ha) < num_annots:
                #    ha += [-1] * (num_annots - len(ha))
                human_annots += ha
            model_annots = []
            for ma in test_data['model_annots']:
                ma = ma[1:-1].replace("'","").replace("nan","").replace(".","").split()
                ma = random.choices([int(x) for x in ma], k=num_annots)
                #if len(ma) < num_annots:
                #    ma += [-1] * (num_annots - len(ma))
                model_annots += ma
            #for dataset_mode in ['sorted', 'shuffle', 'dataset-frequency', 'frequency']:
            # TODO: get human labels here
            for dataset_mode in ['dataset-frequency', 'frequency']:
                for target_col in ['human_annots_str', 'model_annots_str']:
                    print(dataset_name, cat, dataset_mode, target_col)
                    memory(target_col + dataset_mode + cat)
                    hf_model = f"owanr/{dataset_name}-{model_id.replace('/','-')}-{cat}_model-{dataset_mode}-{target_col}"
                    # check if file exsts
                    title = f'{hf_model.replace("owanr/","")}'
                    if False and os.path.exists(f"./png/{title}.png"):
                        print(f"Skipping {hf_model.replace('owanr/','')} since it exists already")
                        continue
                    try:
                        model = T5ForConditionalGeneration.from_pretrained(hf_model)#, device_map="auto", load_in_8bit=True)
                        model.to(acc.device)
                    except Exception as e:
                        print(f"Failed to load {hf_model}")
                        print(e)
                        print("")
                        continue
                    annots = []
                    with open(f'../data/test_data_{cat}_{dataset_name}.pkl', 'rb') as f:
                        test_data = pickle.load(f)
                    texts = test_data['text']
                    print("HOW MANY INSTANCES DO I HAVE", len(texts))
                    for text in texts:
                        text_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                        text_input['decoder_input_ids'] = text_input['input_ids'].clone()
                        text_input = text_input.to(model.device)
                        with torch.no_grad():
                            text_output = model.generate(**text_input, **my_config)
                        text_output_ids = text_output['sequences'][:, text_input['input_ids'].shape[-1]:]
                        answer = tokenizer.decode(text_output_ids[0], skip_special_tokens=True)
                        answer = [int(x) for x in list(answer.replace('nan','').replace('.','').replace(' ',''))]
                        annots += answer
                    '''
                    inputs = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True, truncation=True)
                    inputs['decoder_input_ids'] = inputs['input_ids'].clone()
                    inputs = inputs.to(model.device)

                    # Perform inference
                    with torch.no_grad():
                        memory('before gen' + target_col + dataset_mode + cat)
                        outputs = model.generate(**inputs, **my_config)
                        print(outputs['sequences'].shape)
                        return
                        memory('after gen' + target_col + dataset_mode + cat)

                    # remove input portion of output
                    output_ids = outputs[0][:, inputs['input_ids'].shape[-1]:]
                    answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    for answer in answers:
                        annots += [int(x) for x in list(answer.replace("nan","").replace(".","").replace(' ',''))]
                    '''

                    predicted_array = np.array(annots)
                    gold_array = np.array(human_annots) if target_col == 'human_annots_str' else np.array(model_annots)
                    # Calculate Kendall's Tau
                    try:
                        kendall_tau, _ = kendalltau(predicted_array, gold_array)
                        print(f"Kendall's Tau: {kendall_tau}")

                        # Calculate Spearman's Rank Correlation
                        spearman_corr, _ = spearmanr(predicted_array, gold_array)
                        print(f"Spearman's Rank Correlation: {spearman_corr}")
                    except Exception as e:
                        print(e)
                        print("")
                        continue
                    
                    title_pref = f'{dataset_name}_{cat}_{dataset_mode}_{target_col}'

                    with open(f'{title_pref}.pkl', 'wb') as f:
                        pickle.dump((gold_array, predicted_array, kendall_tau, spearman_corr), f)
                    #with open(f'{title_pref}.pkl', 'rb') as f:
                    #    gold_array, predicted_array, kendall_tau, spearman_corr = pickle.load(f)
def analyze_second_order_annots():
    for dataset_name in ['SChem5Labels', 'Sentiment', 'ghc', 'SBIC']:
        num_annots = utils.get_num_annots(dataset_name)
        num_labels = utils.get_num_labels(dataset_name)
        for cat in ['inter', 'intra']:
            random.seed(42)
            for dataset_mode in ['dataset-frequency', 'frequency']:
                # first check both files exist
                if not os.path.exists(f'old/{dataset_name}_{cat}_{dataset_mode}_human_annots_str.pkl') or not os.path.exists(f'old/{dataset_name}_{cat}_{dataset_mode}_model_annots_str.pkl'):
                    print(f"Skipping {dataset_name}_{cat}_{dataset_mode} since it's missing files")
                    continue
                for target_col in ['human_annots_str', 'model_annots_str']:
                    title_pref = f'{dataset_name}_{cat}_{dataset_mode}_{target_col}'
                    with open(f'old/{title_pref}.pkl', 'rb') as f:
                        # when kendall_tau and spearman_corr are nan, it means the annots are all the same for one or both
                        gold_array, predicted_array, kendall_tau, spearman_corr = pickle.load(f)
                    plt.figure(figsize=(8, 6))
                    plt.scatter(gold_array, predicted_array, color='blue', label='Data Points')
                    plt.plot(gold_array, predicted_array, color='red', linestyle='-', linewidth=2, label='Line of Best Fit')

                    # Add labels and title
                    plt.xlabel('Gold Labels')
                    plt.ylabel('Predicted Labels')

                    # Add a legend
                    plt.legend()
                    plt.savefig(f'png/{title_pref}ScatterPlotofGoldvsPredictedLabels.png')
                    plt.close()
                    continue
                    # 1. Distribution Comparison
                    plt.figure(figsize=(10, 6))
                    sns.histplot([gold_array, predicted_array], kde=True, bins=2, color=['blue', 'orange'], label=['Gold Labels', 'Predicted Labels'])
                    plt.xlabel('Labels')
                    plt.ylabel('Frequency')
                    plt.yticks([0,1])
                    plt.legend()
                    plt.savefig(f'png/{title_pref}DistributionComparison.png')
                    plt.close()

                    # 2. Error Analysis
                    errors = np.abs(predicted_array - gold_array)
                    plt.figure(figsize=(10, 6))
                    sns.histplot(errors, kde=True, bins=num_labels, color='red', label='Error')
                    plt.xlabel('Absolute Error')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.savefig(f'png/{title_pref}ErrorAnalysis.png')
                    plt.close()

                    # 3. Confusion Matrix
                    conf_matrix = confusion_matrix(gold_array, predicted_array)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(gold_array), yticklabels=np.unique(gold_array))
                    plt.xlabel('Predicted Labels')
                    plt.ylabel('True Labels')
                    plt.savefig(f'png/{title_pref}ConfusionMatrix.png')
                    plt.close()

                    # plot annots into simple histogram
                    num_labels = utils.get_num_labels(dataset_name)
                    #visualize.create_hist_from_lst(annots, num_labels=num_labels, title=title)

#get_second_order_annots()
analyze_second_order_annots()
