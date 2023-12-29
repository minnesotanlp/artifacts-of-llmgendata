import utils
import ast
from pandas import read_csv
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

def before_after_line_plots_orig(res, suffix, top_n):
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
                    print(f'***** {dataset_name}-{m}-{source}-pred-{top_n}', res[dataset_name][m][source]['pred'])
                    y = [res[dataset_name][m][source]['gold'].get(str(i),0) for i in range(num_labels)]
                    ax.plot(x, y, color=color_lst[plot_ind], marker=',', linestyle=line_lst[0], label=f'{source if source=="human" else "model"}-gold', linewidth=linewidth)
                    y = [res[dataset_name][m][source]['pred'].get(str(i),0) for i in range(num_labels)]
                    ax.plot(x, y, color=color_lst[plot_ind], marker=',', linestyle=line_lst[1], label=f'{source if source=="human" else "model"}-pred', linewidth=linewidth)
                    #if source != 'human':
                    #    if m == 'sorted':
                    #        y_max = max(y) 
                    #        print('predictions', y)
                    #        print('max', y_max)
                    #    if m == 'shuffle':
                    #        print('added', y_max, f'to ***** {dataset_name}-{m}-{source}-pred-{top_n}')
                    #        # add horizontal line at y_max
                    #        ax.axhline(y=y_max, color=color_lst[plot_ind+1], linestyle=line_lst[-1], label=f'{source if source=="human" else "model"}-sorted-prediction-max', linewidth=linewidth)
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
                if top_n == 1:
                    filename = f"{dataset_name}-{m}-{source}{suffix}-before-after.png"
                else:
                    filename = f"{dataset_name}-{m}-{source}{suffix}-before-after-top{top_n}.png"
                print('saving', filename)
                plt.savefig(filename, bbox_inches='tight', dpi=300) 
                plt.figure().clear()
                plt.close()

def before_after_line_plots(res, suffix, top_n):
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
        if top_n == 1:
            if len(modes) == 4:
                filename += f"all_modes-before-after.png"
            else:
                filename += f"before-after.png"
        else:
            filename += f"before-after-top{top_n}.png"
        print('saving', filename)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.figure().clear()
        plt.close()

def show_trend(x, y, title):
    # Sample data
    x = np.arange(1, 11)
    y = np.array([3, 5, 8, 12, 9, 6, 4, 7, 10, 14])

    # Threshold value to determine colors
    threshold = 8

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot bars with two colors based on the threshold
    for xi, yi in zip(x, y):
        if yi >= threshold:
            # Bar above threshold: top part is green, bottom part is red
            ax.bar(xi, yi - threshold, bottom=threshold, color='green', edgecolor='black')
            ax.bar(xi, threshold, color='red', edgecolor='black')
        else:
            # Bar below threshold: top part is red, bottom part is green
            ax.bar(xi, threshold - yi, bottom=yi, color='red', edgecolor='black')
            ax.bar(xi, yi, color='green', edgecolor='black')

    # Add a line plot to show the trend
    ax.plot(x, y, label='Trend Line', color='blue', marker='o')

    # Add labels and legend
    ax.set_xlabel('X-axis Label')
    ax.set_ylabel('Y-axis Label')
    ax.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
    ax.legend()

    # Show the plot
    plt.savefig(f"./png/{title}.png")

def counter_to_sorted_dict(counter):
    d = {}
    for key in counter.keys():
        d[int(key)] = int(counter[key])
        
    d = dict(sorted(d.items()))
    return d


def main(suffix, top_n, flatten):
    # TODO: clean up code
    res = {} 
    gold = {}

    for dataset_name in ['Sentiment', 'SChem5Labels', 'ghc', 'SBIC']:
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
                try:
                    with open(filename, 'rb') as f:
                        logits = pickle.load(f)
                        #print(f'loaded {filename}')
                        #print(logits[:10])
                except:
                    print('!!!!!!!!!!!!!!!!!!!!!')
                    print("ERROR", filename)
                    continue
                labels = np.array(gold[dataset_name][m][source]['gold']).flatten().tolist()
                if top_n == 1:
                    predictions = np.argmax(logits, axis=-1).flatten().tolist()
                else:
                    predictions = np.array([])
                    for logit_row in logits:
                        for annot_row in logit_row:
                            # choose one according to percentage
                            annot_row = softmax(annot_row)
                            # get the indices of the ones over 0.5
                            chosen = np.argwhere(annot_row > 0.4).flatten()
                            if len(chosen) == 0:
                                chosen = np.argwhere(annot_row == max(annot_row)).flatten()
                            predictions = np.append(predictions, chosen)
                            # ********************
                            # choose one randomly
                            #chosen = np.argpartition(annot_row[0], -top_n, axis=-1)[::-1][:top_n]
                            #predictions = np.append(predictions, [int(random.choice(chosen))])
                #predictions = np.random.choice(predictions, size=len(labels), replace=False).flatten().tolist()
                if flatten:
                    if type(predictions) != list:
                        res[dataset_name][m][source]['pred'] = predictions.tolist()
                    else:
                        res[dataset_name][m][source]['pred'] = predictions
                    res[dataset_name][m][source]['gold'] = labels#.tolist()
                else:
                    res[dataset_name][m][source]['pred'] = counter_to_sorted_dict(Counter(predictions))
                    res[dataset_name][m][source]['gold'] = counter_to_sorted_dict(Counter(labels))

                #print(Counter(utils.flatten_recursive(gold_inter[dataset_name]['model'])))
                #print("")
                #for i in range(len(predictions)):
                #    if predictions[i] == -1 or inter_labels[i] == -1:
                #        continue
            ''' not using
                for i, row in enumerate(gold_inter[dataset_name][source]):
                    row = row.replace('nan', '-1').replace('.', '')
                    row = re.sub(r'(\d)\s', r'\1, ', row)
                    gold_inter[dataset_name][source][i] = eval(row)
                gold_inter[dataset_name][source] = [int(item) for row in gold_inter[dataset_name][source] for item in row]
           ''' 
    # write to file here
    write_to = 'res'
    if flatten:
        write_to += '_flatten'
    write_to += suffix
    if top_n != 1:
        write_to += f'_top{top_n}'
    write_to += '.json'
    with open(write_to, 'w') as f:
        json.dump(res, f, indent=4)

def bar_plot(res, mode, suffix=''): 
    '''
    Uses the non-flattened res
    '''
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    categories = list(res.keys())
    bar_width = 0.25
    hatches = [ '*', 'o', '.']

    for i, xy in enumerate([(0,0),(0,1),(1,0),(1,1)][:len(categories)]):
        dataset_name = categories[i]
        labels = list(range(utils.get_num_labels(dataset_name)))
        bar = {}
        bar_gold = {}
        diff = {}
        colors = {}
        for k, source in enumerate(['human', 'inter']):
            bar[source] = res[dataset_name][mode][source]
            bar_gold[source] = [bar[source]['gold'].get(str(j),0) for j in range(len(labels))]

            maj = np.argmax(np.array([bar[source]['gold'].get(str(j),0) for j in range(len(labels))]))
            diff[source] = [(bar[source]['pred'].get(str(j),0)-bar[source]['gold'].get(str(j),0))/bar[source]['gold'].get(str(j),0) for j in range(len(labels))]
            colors[source] = ['white' if i != maj else plt.cm.Set1(0) for i in range(len(labels))]
            print(dataset_name, mode, source, 'maj', maj, suffix)
            print('diff', [round(el, 2) for el in diff[source]])
            print('')

    '''
            #axs[xy[0], xy[1]].bar(labels, diff, color=colors, hatch='/')
            # Add the bar plots with slight overlap
            if k == 0:
                axs[xy[0], xy[1]].bar(np.arange(len(labels)) - bar_width * 0.5, diff[source], color=plt.cm.Set2(0), width=bar_width, align='center', label="Human", edgecolor = colors[source], linewidth=2)
            elif k == 1:
                axs[xy[0], xy[1]].bar(np.arange(len(labels)), diff[source], color=plt.cm.Set2(2), width=bar_width, align='center', label="Inter", edgecolor = colors[source], linewidth=2)
            else:
                axs[xy[0], xy[1]].bar(np.arange(len(labels)) + bar_width * 0.5, diff[source], color=plt.cm.Set2(4), width=bar_width, align='center', label="Intra", edgecolor = colors[source], linewidth=2)
        axs[xy[0], xy[1]].set_title(dataset_name)
        axs[xy[0], xy[1]].set_xticks(labels)
        axs[xy[0], xy[1]].set_xticklabels(labels)
        axs[xy[0], xy[1]].set_xticks(np.arange(len(labels)) + 0.5, minor=True)

        #axs[xy[0], xy[1]].axhline(y = 0, linestyle = '-')
        axs[xy[0], xy[1]].xaxis.set_tick_params(which="minor", size=0)
        axs[xy[0], xy[1]].xaxis.set_tick_params(which="minor", width=0)

        # Add a legend
        axs[xy[0], xy[1]].legend(["Human", "Inter", "Intra"], loc='upper left' )

    plt.tight_layout()
    plt.savefig(f"{mode}-diff-{suffix}.png")
    '''

def density_plot(suffix, top_n=2, flatten=False):
    write_to = 'res'
    if flatten:
        write_to += '_flatten'
    write_to += suffix
    if top_n != 1:
        write_to += f'_top{top_n}'
    write_to += '.json'
    with open(write_to) as f:
        res = json.load(f)
    for dataset_name in ['Sentiment', 'SChem5Labels']:
        num_labels = utils.get_num_labels(dataset_name)
        labels_lst = list(range(num_labels+1))
        for m in ['sorted', 'shuffle']:
            print(res[dataset_name][m]['human']['gold'][0])
            data_HO = res[dataset_name][m]['human']['gold']
            data_HP = res[dataset_name][m]['human']['pred']
            data_MO = res[dataset_name][m]['inter']['gold']
            data_MP = res[dataset_name][m]['inter']['pred']
            plt.figure(figsize=(10,8))
            plt.hist(data_HO, bins=labels_lst, density=False, alpha=0.5, color='blue', label='Human Original')
            plt.hist(data_HP, bins=labels_lst, density=False, alpha=0.5, color='blue', linestyle='dashed', histtype='step', label='Human Predicted')
            plt.hist(data_MO, bins=labels_lst, density=False, alpha=0.5, color='red', label='Machine Original')
            plt.hist(data_MP, bins=labels_lst, density=False, alpha=0.5, color='red', linestyle='dashed', histtype='step', label='Machine Predicted')

            # Adding labels and title
            plt.xlabel('Annotation')
            plt.ylabel('Density')
            plt.xticks(list(range(num_labels+1)))
            #plt.title('Density Plot of Annotations by Type')

            # Adding a legend
            plt.legend()
            if top_n == 1:
                plt.savefig(f"{dataset_name}-{m}-density-{suffix}.png")
            else:
                filename = f"{dataset_name}-{m}-density-{suffix}-top{top_n}"
                plt.savefig(f"{filename}.png")
            plt.figure().clear()
            plt.close()

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
        for top_n in [1]:
            #for suffix in ["_alpha0.5_whole_1e-05", "_alpha0.8_whole_1e-05"]:
            for suffix in ["_alpha0.8_whole_1e-05"]:
                main(suffix, top_n=top_n, flatten=flatten)
                #density_plot(suffix, top_n=top_n, flatten=flatten)
    '''
    for top_n in [1]:
        for suffix in ['_alpha0.8_whole_1e-05']:
            write_to = f'res{suffix}'
            with open(write_to+'.json') as f:
                res = json.load(f)
                get_gold(res)
                #print('keys', res['Sentiment']['shuffle']['human']['gold'])
                #exit()
            #if 'ghc' in res:
            #    del res['ghc']
            #if 'SBIC' in res:
            #    del res['SBIC']
            #before_after_line_plots(res, suffix, top_n)
            #bar_plot(res, 'sorted', suffix)
            #bar_plot(res, 'shuffle', suffix)
    #'''
    #'''
    #with open(f'res{suffix}.json') as f:
    #    res = json.load(f)
    #bar_plot(res, 'data-frequency', suffix+'_top2')
    #bar_plot(res, 'frequency', suffix+'_top2')
    #bar_plot(res, 'shuffle', suffix+'_top2')
    #'''
