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
from numpy.polynomial.polynomial import Polynomial
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
model_id = "google/t5-v1_1-xl"
tokenizer = T5Tokenizer.from_pretrained(model_id)
acc = accelerate.Accelerator()

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

def clean_row(row, num_labels, num_annots):
    # very specific to the string from the pkl file
    row = row[1:-1].replace("'","").replace("nan","").replace(".","").split()
    row = [int(x) for x in row]
    row = [num for num in row if 0 <= num and num < num_labels]
    row = random.choices(row, k=num_annots)
    return row

def get_second_order_annots():
    for dataset_name in ['SChem5Labels', 'Sentiment']:#, 'ghc', 'SBIC']:
    #for dataset_name in ['ghc']:
        num_annots = utils.get_num_annots(dataset_name)
        num_labels = utils.get_num_labels(dataset_name)
        my_config = {}
        my_config['renormalize_logits'] = True
        my_config['return_dict_in_generate'] = True
        def restrict_decode_vocab(a, b):
            return [209, 204, 220, 314, 305][:num_labels-1]
        my_config["prefix_allowed_tokens_fn"] = restrict_decode_vocab #not json serializable
        my_config['max_new_tokens'] = 10
        my_config['min_new_tokens'] = num_annots
        #my_config['bos_token_id'] = 0

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
                human_annots += clean_row(ha, num_labels, num_annots)
            model_annots = []
            for ma in test_data['model_annots']:
                model_annots += clean_row(ma, num_labels, num_annots)
            #for dataset_mode in ['sorted', 'shuffle', 'dataset-frequency', 'frequency']:
            # TODO: get human labels here
            #for dataset_mode in ['dataset-frequency', 'frequency']:
            for dataset_mode in ['frequency']:
                for target_col in ['human_annots_str', 'model_annots_str']:
                    print(dataset_name, cat, dataset_mode, target_col)
                    memory(target_col + dataset_mode + cat)
                    hf_model = f"owanr/{dataset_name}-{model_id.replace('/','-')}-{cat}-{dataset_mode}-{target_col.replace('_annots_str','')}-cross-ent"
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
                    print(f"Predicted: {predicted_array}")
                    print(f"Gold: {gold_array}")
                    try:
                        kendall_tau, _ = kendalltau(predicted_array, gold_array)
                        print(f"Kendall's Tau: {kendall_tau}")

                        # Calculate Spearman's Rank Correlation
                        spearman_corr, _ = spearmanr(predicted_array, gold_array)
                        print(f"Spearman's Rank Correlation: {spearman_corr}")
                    except Exception as e:
                        print(e)
                        print("")
                    
                    title_pref = f'{dataset_name}_{cat}_{dataset_mode}_{target_col}'

                    with open(f'{title_pref}.pkl', 'wb') as f:
                        pickle.dump((gold_array, predicted_array), f)
                        #pickle.dump((gold_array, predicted_array, kendall_tau, spearman_corr), f)
                    #with open(f'{title_pref}.pkl', 'rb') as f:
                    #    gold_array, predicted_array, kendall_tau, spearman_corr = pickle.load(f)

shapes = ['o', 'x']
colors = ['red', 'blue']
colors = {
    'inter': {
        'human_annots_str': 'red',
        'model_annots_str': 'blue'
    },
    'intra': {
        'human_annots_str': 'orange',
        'model_annots_str': 'green'
    }}
def analyze_second_order_annots(plot_type='hist'):
    for dataset_name in ['SChem5Labels', 'Sentiment', 'ghc', 'SBIC']:
        num_annots = utils.get_num_annots(dataset_name)
        num_labels = utils.get_num_labels(dataset_name)
        for dataset_mode in ['dataset-frequency', 'frequency']:
            random.seed(42)

            # Latex table
            print(f'{dataset_name} {dataset_mode}***')
            print('\\begin\\{table\\}[h]\\n')
            print('\\begin\\{tabular\\}\\{|l|l|l|l|l|\\}\\n')
            print('\\hline\\n')
            print('& \\multicolumn\\{2\\}\\{|c|\\}\\{\\{\\textbf\\{Inter\\}\\}\\} & \\multicolumn\\{2\\}\\{|c|\\}\\{\\{\\textbf\\{Intra\\}\\}\\} \\\\\\n')
            print('& \\textbf\\{Human\\} & \\textbf\\{Model\\} & \\textbf\\{Human\\} & \\textbf\\{Model\\} \\\\\\n')
            print('\\hline\\n')
            temp = {}
            for ci, cat in enumerate(['inter', 'intra']):
                # first check both files exist
                if not os.path.exists(f'old/{dataset_name}_{cat}_{dataset_mode}_human_annots_str.pkl') or not os.path.exists(f'old/{dataset_name}_{cat}_{dataset_mode}_model_annots_str.pkl'):
                    print(f"Skipping {dataset_name}_{cat}_{dataset_mode} since it's missing files")
                    continue
                temp[cat] = {"human_annots_str": {}, "model_annots_str": {}}
                for ti, target_col in enumerate(['human_annots_str', 'model_annots_str']):
                    title_pref = f'{dataset_name}_{cat}_{dataset_mode}_{target_col}'
                    #with open(f'old/{title_pref}.pkl', 'rb') as f:
                    with open(f'{title_pref}.pkl', 'rb') as f:
                        # when kendall_tau and spearman_corr are nan, it means the annots are all the same for one or both
                        # TODO: MAKE SURE THERE ARE NO INVALID ANNOTATIONS
                        content = pickle.load(f)
                        print(len(content), "-------------------")
                        gold_array, predicted_array, kendall_tau, spearman_corr = pickle.load(f)
                    # 1d because the ideal line is 1d
                    corr2 = Polynomial.fit(gold_array, predicted_array, 1)
                    temp[cat][target_col]['Slope'] = corr2.convert().coef[1]
                    temp[cat][target_col]['Intercept'] = corr2.convert().coef[0]
                    temp[cat][target_col]['Kendall'] = kendall_tau
                    temp[cat][target_col]['Spearman'] = spearman_corr
            #for row_header in ['Slope', 'Intercept', 'Kendall', 'Spearman']:
            #    print(f'{row_header} & {temp["inter"]["human_annots_str"][row_header]:.3f} & {temp["inter"]["model_annots_str"][row_header]:.3f} & {temp["intra"]["human_annots_str"][row_header]:.3f} & {temp["intra"]["model_annots_str"][row_header]:.3f} \\\\\n')
            print('\\hline\\n')
            print('\\end{tabular}\\n')
            print('\\end{table}\\n\\n')

            if plot_type == 'hist':
                # distribution comparison
                for ci, cat in enumerate(['inter', 'intra']):
                    ax = plt.figure(figsize=(8, 6))
                    # first check both files exist
                    #if not os.path.exists(f'old/{dataset_name}_{cat}_{dataset_mode}_human_annots_str.pkl') or not os.path.exists(f'old/{dataset_name}_{cat}_{dataset_mode}_model_annots_str.pkl'):
                    if not os.path.exists(f'{dataset_name}_{cat}_{dataset_mode}_human_annots_str.pkl') or not os.path.exists(f'{dataset_name}_{cat}_{dataset_mode}_model_annots_str.pkl'):
                        print(f"Skipping {dataset_name}_{cat}_{dataset_mode} since it's missing files")
                        continue
                    for ti, target_col in enumerate(['human_annots_str', 'model_annots_str']):
                        title_pref = f'{dataset_name}_{cat}_{dataset_mode}_{target_col}'
                        with open(f'{title_pref}.pkl', 'rb') as f:
                            # when kendall_tau and spearman_corr are nan, it means the annots are all the same for one or both
                            content = pickle.load(f)
                            if len(content) == 2:
                                gold_array, predicted_array = content
                            else:
                                gold_array, predicted_array, kendall_tau, spearman_corr = pickle.load(f)

                        print(f"Gold: {gold_array}", len(gold_array), gold_array[:10])
                        print(f"Predicted: {predicted_array}", len(predicted_array), gold_array[:10])


                        # Create histograms for each sequence
                        hist1, bins = np.histogram(gold_array, bins=range(utils.get_num_labels(dataset_name)), density=True)
                        hist2, _ = np.histogram(predicted_array, bins=range(utils.get_num_labels(dataset_name)), density=True)

                        # Plotting
                        width = 0.35  # the width of the bars
                        fig, ax = plt.subplots()
                        rects1 = ax.bar(np.array(bins[:-1]), hist1, width, label='Gold')
                        rects2 = ax.bar(np.array(bins[:-1]) + width, hist2, width, label='Predicted')

                        # Add some text for labels, title, and legend
                        ax.set_xlabel('Labels')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Distribution of Sequences for'+dataset_name)
                        ax.set_xticks(np.array(bins[:-1]) + width / 2)
                        ax.set_xticklabels([str(i) for i in range(utils.get_num_labels(dataset_name)-1)])
                        ax.legend()


                        #sns.histplot([gold_array, predicted_array], kde=False, bins=num_labels, color=['red', 'blue'][:2], label=['Gold Labels', 'Predicted Labels'])
                        #sns.histplot([gold_array, predicted_array], kde=False, bins=num_labels, color=['red', 'orange', 'green', 'blue', 'purple', 'pink'][:2], label=['Gold Labels', 'Predicted Labels'])
                        #plt.xlabel('Labels')
                        #plt.ylabel('Frequency')
                        #ax.yticks([0,1])

                    # Add a legend
                    ax.legend()
                    plt.savefig(f'png/delete_{title_pref}_DistributionComparison.png')
                    plt.close()
            '''
            elif plot_type == 'error':
                    # 2. Error Analysis
                    errors = np.abs(predicted_array - gold_array)
                    plt.figure(figsize=(10, 6))
                    sns.histplot(errors, kde=True, bins=num_labels, color='red', label='Error')
                    plt.xlabel('Absolute Error')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.savefig(f'png/{title_pref}ErrorAnalysis.png')
                    plt.close()
            elif plot_type == 'confusion':
                ##################################################
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
            '''

#get_second_order_annots()
analyze_second_order_annots('hist')
