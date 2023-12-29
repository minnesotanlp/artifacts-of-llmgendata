import utils
import ast
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from collections import Counter
import re
import json
import torch
import time
from scipy.special import softmax
modes = ['frequency', 'data-frequency', 'sorted', 'shuffle']
sources = ['human', 'inter']
dataset_names = ['Sentiment', 'SChem5Labels', 'ghc', 'SBIC']


def create_hist_from_lst(lst, num_labels=5, title="Histogram of Label Annotations"):
    f = plt.figure()
    plt.hist(lst, bins=num_labels)
    plt.title(title)
    y_ticks = list(range(num_labels))
    plt.yticks(y_ticks)

    plt.xlabel("Annotation")
    plt.ylabel("Frequency")
    plt.savefig(f"./png/{title}.png")
    #plt.close(f)
    #plt.show()

def before_after_line_plots_orig(res, suffix):
    linewidth = 3
    color_lst = [plt.cm.Set1(0), plt.cm.Set1(1), plt.cm.Set1(2), plt.cm.Set1(3), plt.cm.Set1(4), plt.cm.Set1(5)]
    line_lst = ['-', '--', '-.', ':']
    for dataset_name in res:
        num_labels = utils.get_num_labels(dataset_name)
        for m in modes:
            for sources in [['human', 'inter']]:#, ['human', 'intra']]:  
                x = list(range(num_labels))
                plt.figure()
                fig, ax = plt.subplots()
                plot_ind = 0
                # creates 4 lines
                for source in sources: 
                    #print(f'{dataset_name}-{m}-{source}-gold', res[dataset_name][m][source]['gold'])
                    print(f'***** {dataset_name}-{m}-{source}-pred', res[dataset_name][m][source]['pred'])
                    y = [res[dataset_name][m][source]['gold'].get(str(i),0) for i in range(num_labels)]
                    ax.plot(x, y, color=color_lst[plot_ind], marker=',', linestyle=line_lst[0], label=f'{source if source=="human" else "model"}-gold', linewidth=linewidth)
                    y = [res[dataset_name][m][source]['pred'].get(str(i),0) for i in range(num_labels)]
                    ax.plot(x, y, color=color_lst[plot_ind], marker=',', linestyle=line_lst[1], label=f'{source if source=="human" else "model"}-pred', linewidth=linewidth)
                    plot_ind += 1
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                ax.legend(fontsize=18)
                # Add labels and a title
                ax.xaxis.label.set_size(18)
                ax.yaxis.label.set_size(18)
                ax.set_xlabel('Labels')
                ax.set_ylabel('Counts')
                plt.tight_layout()
                # set xticks
                x_ticks = list(range(num_labels))
                ax.set_xticks(x_ticks)
                ax.set_title(f'{dataset_name}')
                ax.title.set_size(20)
                # Display the plot
                filename = f"{dataset_name}-{m}-{source}{suffix}-before-after.png"
                print('saving', filename)
                plt.savefig(filename, bbox_inches='tight', dpi=300) 
                plt.figure().clear()
                plt.close()

def before_after_line_plots(res, suffix):
    include_human = False
    if include_human:
        source_lst = ['human', 'inter']
    else:
        source_lst = ['inter']
    linewidth = 3
    color_lst = []
    for i in range(10):
        color_lst.append(plt.cm.Set1(i))
    #color_lst = [plt.cm.Set1(0), plt.cm.Set1(1), plt.cm.Set1(2), plt.cm.Set1(3), plt.cm.Set1(4), plt.cm.Set1(6)]
    line_lst = ['-', '--', '-.', '-']
    print('datasets', res.keys())
    for dataset_name in res:
        num_labels = utils.get_num_labels(dataset_name)
        x = list(range(num_labels))
        if len(modes) == 2:
            plt.figure()
        else:
            plt.figure(figsize=(20,8))
        fig, ax = plt.subplots()
        plot_ind = 0
        if include_human:
            y = [res[dataset_name]['sorted']['human']['gold'].get(str(i),0) for i in range(num_labels)]
            ax.plot(x, y, color='black', marker=',', linestyle=line_lst[-1], label=f'human (H)', linewidth=linewidth)
        y = [res[dataset_name]['sorted']['inter']['gold'].get(str(i),0) for i in range(num_labels)]
        ax.plot(x, y, color='black', marker=',', linestyle=line_lst[-2], label=f'model (M)', linewidth=linewidth)
        for k, source in enumerate(source_lst):
            for m in modes:
                if m != 'sorted' and source == 'human':
                    continue
                y = [res[dataset_name][m][source]['pred'].get(str(i),0) for i in range(num_labels)]
                print(source, m, y)
                print('-'*10)
                ax.plot(x, y, color=color_lst[plot_ind], marker=',', linestyle=line_lst[1], label=f'{"H" if source=="human" else "M"} pred. ({m})', linewidth=linewidth)
                plot_ind += 1
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if len(modes) == 2:
            ax.legend(fontsize=18)
        else:
            ax.legend(fontsize=18, loc='center left', bbox_to_anchor=(1, 0.5))
        # Add labels and a title
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ax.set_xlabel('Labels')
        ax.set_ylabel('Counts')
        #plt.tight_layout()
        # set xticks
        x_ticks = list(range(num_labels))
        ax.set_xticks(x_ticks)
        ax.set_title(f'{dataset_name}')
        ax.title.set_size(20)
        # Display the plot
        filename = f"{dataset_name}-{suffix}-"
        print('????', filename)
        if not include_human:
            filename += "model_only-"
        if len(modes) == 4:
            filename += f"all_modes-before-after.png"
        else:
            filename += f"before-after.png"
        print('saving', filename)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.figure().clear()
        plt.close()

def counter_to_sorted_dict(counter):
    d = {}
    for key in counter.keys():
        d[int(key)] = int(counter[key])
        
    d = dict(sorted(d.items()))
    return d


def main(suffix, flatten):
    # TODO: clean up code
    res = {} 
    gold = {}

    for dataset_name in dataset_names:
        res[dataset_name] = {}
        gold[dataset_name] = {}
        num_labels = utils.get_num_labels(dataset_name)
        for m in modes: 
            start_time = time.time()
            res[dataset_name][m] = {} 
            gold[dataset_name][m] = {} # we get gold labels earlier so we want to store it here

            for source in source_lst:
                res[dataset_name][m][source] = {}
                gold[dataset_name][m][source] = {}
                for t in ['pred', 'gold']:
                    res[dataset_name][m][source][t] = []
                gold_filename = '../data/intramodel_data.csv' if source == 'intra' else '../data/intermodel_data.csv'
                test_data = utils.get_data(gold_filename, dataset_name=dataset_name, mode=m, model_id="roberta-base")['test']
                gold[dataset_name][m][source]['gold'] = [row['human_annots'] if source=='human' else row['model_annots'] for row in test_data]
        for m in modes: 
            for source in source_lst:
                filename = f'results_new/{dataset_name}-roberta-base-{"intra" if source == "intra" else "inter"}-{m}-{"human" if source == "human" else "model"}_annots{suffix}.pkl'
                with open(filename, 'rb') as f:
                    logits = pickle.load(f)
                labels = np.array(gold[dataset_name][m][source]['gold']).flatten().tolist()
                predictions = np.argmax(logits, axis=-1).flatten().tolist()
                if flatten:
                    if type(predictions) != list:
                        res[dataset_name][m][source]['pred'] = predictions.tolist()
                    else:
                        res[dataset_name][m][source]['pred'] = predictions
                    res[dataset_name][m][source]['gold'] = labels#.tolist()
                else:
                    res[dataset_name][m][source]['pred'] = counter_to_sorted_dict(Counter(predictions))
                    res[dataset_name][m][source]['gold'] = counter_to_sorted_dict(Counter(labels))
    # write to file here
    write_to = 'res_test'
    if flatten:
        write_to += '_flatten'
    write_to += suffix
    write_to += '.json'
    with open(write_to, 'w') as f:
        json.dump(res, f, indent=4)

def ratio_of_largest_values(dictionary):
    # Extract values from the dictionary
    values = list(dictionary.values())

    # Check if there are at least two values in the dictionary
    if len(values) < 2:
        raise ValueError("Dictionary must have at least two values")

    # Sort the values in descending order
    sorted_values = sorted(values, reverse=True)

    # Calculate the ratio of the largest value over the next largest value
    ratio = sorted_values[0] / sorted_values[1]

    return ratio

def get_gold(res):
    for dataset_name in res.keys():
        for mode in res[dataset_name].keys():
            for source in res[dataset_name][mode].keys():
                ratio = ratio_of_largest_values(res[dataset_name][mode][source]['gold'])
                print(dataset_name, mode, source, ratio)

if __name__ == '__main__':
    '''
    for flatten in [False]:
        for suffix in ["_alpha0.8_whole_1e-05"]:
            main(suffix, flatten=flatten)
    '''
    for suffix in ['_alpha0.8_whole_1e-05']:
        write_to = f'res{suffix}'
        with open(write_to+'.json') as f:
            res = json.load(f)
            get_gold(res)
        #before_after_line_plots(res, suffix)
