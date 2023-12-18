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

def before_after_line_plots(res, suffix, full, top_n):
    color_lst = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'gray', 'olive', 'cyan']
    line_lst = ['-', '--', '-.', ':']
    for dataset_name in res:
        num_labels = utils.get_num_labels(dataset_name)
        for m in res[dataset_name]:
            for sources in [['human', 'inter']]:  
                x = list(range(num_labels))
                plt.figure()
                fig, ax = plt.subplots()
                plot_ind = 0
                for source in sources: 
                    #print(f'{dataset_name}-{m}-{source}-gold', res[dataset_name][m][source]['gold'])
                    print(f'{dataset_name}-{m}-{source}-pred-{full}-{top_n}', res[dataset_name][m][source]['pred'])
                    continue
                    y = [res[dataset_name][m][source]['gold'].get(str(i),0) for i in range(num_labels)]
                    ax.plot(x, y, color=color_lst[plot_ind], marker=',', linestyle=line_lst[0], label=f'{source if source=="human" else "model"}-gold')
                    y = [res[dataset_name][m][source]['pred'].get(str(i),0) for i in range(num_labels)]
                    ax.plot(x, y, color=color_lst[plot_ind], marker=',', linestyle=line_lst[1], label=f'{source if source=="human" else "model"}-prediction')
                    plot_ind += 1
                '''
                ax.legend()
                # Add labels and a title
                ax.set_xlabel('Labels')
                ax.set_ylabel('Counts')
                # set xticks
                x_ticks = list(range(num_labels))
                ax.set_xticks(x_ticks)
                ax.set_title(f'{dataset_name}-{m}')
                # Display the plot
                if top_n == 1:
                    plt.savefig(f"{dataset_name}-{m}-{suffix}-before-after.png") 
                elif full:
                    plt.savefig(f"{dataset_name}-{m}-{suffix}-before-after-full.png")
                else:
                    plt.savefig(f"{dataset_name}-{m}-{suffix}-before-after-top{top_n}.png")
                plt.figure().clear()
                plt.close()
                '''

def movement_heatmap(mat):
    for dataset_name in mat:
        num_labels = utils.get_num_labels(dataset_name)
        for m in mat[dataset_name]:
            for source in mat[dataset_name][m]:
                #mat[dataset_name][m][source] = np.array(mat[dataset_name][m][source])
                heatmap_data = np.full((num_labels, num_labels), np.nan)
                #heatmap_data = np.zeros((num_labels, num_labels))
                #for i in range(num_labels): #gold/from
                #    for j in range(num_labels): #pred/to
                #        heatmap_data[i][j] = np.nan

                for i in range(num_labels): #gold/from
                    for j in range(num_labels): #pred/to
                        if i != j:
                            heatmap_data[i][j] = ((i-j)/abs(i-j))*mat[dataset_name][m][source][i][j] 
                # remove all edge rows or columns will all nan
                print('before', heatmap_data)
                heatmap_data = heatmap_data[~np.isnan(heatmap_data).all(axis=1)]
                print('after', heatmap_data)

                plt.figure(figsize=(8, 6))
                #sns.heatmap(np.isnan(heatmap_data), cmap='Greys', cbar=False, linewidths=.5)#, mask=np.isnan(heatmap_data), annot=False)
                sns.heatmap(heatmap_data, annot=False, linewidths=.5, square=True, cbar_kws={'label': 'Color Values'}, cmap='coolwarm', center=0)

                # Adding labels and title
                plt.xlabel('Gold Labels')
                plt.ylabel('Pred Labels')

                # Displaying the plot
                #plt.savefig(f'{dataset_name}-{m}-{source}-label-movement-heatmap.png')
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

def get_gold(grouping):
    res = {}
    m = 'frequency'
    for dataset_name in ['Sentiment', 'SBIC', 'ghc', 'SChem5Labels']:
        res[dataset_name] = {}
        df = read_csv(f'../data/{grouping}model_data.csv')
        df = df[df['dataset_name'] == dataset_name]
        res[dataset_name]['human'] = [utils.str_to_num_lst(row) for row in df['human_annots']]
        res[dataset_name]['model'] = [utils.str_to_num_lst(row) for row in df['model_annots']]
    return res

def counter_to_sorted_dict(counter):
    d = {}
    for key in counter.keys():
        d[int(key)] = int(counter[key])
        
    d = dict(sorted(d.items()))
    return d


def main(suffix, top_n=2, full=False, flatten=False):
    # TODO: clean up code
    res = {} 
    switch = {}
    gold = {}
    #gold_inter = get_gold('inter')
    #gold_intra = get_gold('intra')

    #modes = ['frequency', 'data-frequency', 'sorted', 'shuffle']
    modes = ['sorted', 'shuffle']

    #temp = [item for row in list(gold_inter['SBIC']['human']) for item in row]
    #for dataset_name in ['SChem5Labels']:#, 'SBIC', 'ghc', 'SChem5Labels']:
    #for dataset_name in ['SChem5Labels']:
    for dataset_name in ['Sentiment', 'SChem5Labels']:
        max_ct = 0
        min_ct = 1000000
        res[dataset_name] = {}
        gold[dataset_name] = {}
        switch[dataset_name] = {}
        num_labels = utils.get_num_labels(dataset_name)
        for m in modes: 
            start_time = time.time()
            switch[dataset_name][m] = {} 
            res[dataset_name][m] = {} 
            gold[dataset_name][m] = {} # we get gold labels earlier so we want to store it here

            for source in ['human', 'inter']:
                switch[dataset_name][m][source] = {l: {l: 0 for l in range(num_labels)} for l in range(num_labels)}
                res[dataset_name][m][source] = {}
                gold[dataset_name][m][source] = {}
                for t in ['pred', 'gold']:
                    res[dataset_name][m][source][t] = []
                if source != 'inter':
                    gold_filename = '../data/intramodel_data.csv' if source == 'intra' else '../data/intermodel_data.csv'
                    test_data = utils.get_data(gold_filename, dataset_name=dataset_name, mode=m, model_id="roberta-base")['test']
                gold[dataset_name][m][source]['gold'] = [row['human_annots'] if source=='human' else row['model_annots'] for row in test_data]
                #gold
                #human_gold = [row['human_annots'] for row in test_data]
                #inter_gold = [row['model_annots'] for row in test_data]
                #intra_gold = [row['model_annots'] for row in test_data]
                #print("INTER GOLD", inter_gold)
                #print("INTRA GOLD", intra_gold)
                #return
        for m in modes: 
            for source in ['human', 'inter']:
                filename = f'results_new/unused/{dataset_name}-roberta-base-{"intra" if source == "intra" else "inter"}-{m}-{"human" if source == "human" else "model"}_annots{suffix}.pkl'
                try:
                    with open(filename, 'rb') as f:
                        logits = pickle.load(f)
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
                            chosen = np.argpartition(annot_row[0], -top_n, axis=-1)[::-1][:top_n]
                            # choose one randomly
                            if full:
                                predictions = np.concatenate((predictions, chosen))
                            else:
                                predictions = np.append(predictions, [int(random.choice(chosen))])
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
                #    switch[dataset_name][m]['inter'][inter_labels[i]][predictions[i]] += 1
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
        if full:
            write_to += '_full'
        else:
            write_to += f'_top{top_n}'
    write_to += '.json'
    with open(write_to, 'w') as f:
        json.dump(res, f, indent=4)
    #print(json.dumps(switch, indent=4))

def bar_plot(res, mode, suffix=''): 
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    categories = list(res.keys())
    bar_width = 0.25
    hatches = [ '*', 'o', '.']

    for i, xy in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        dataset_name = categories[i]
        labels = list(range(utils.get_num_labels(dataset_name)))
        bar = {}
        bar_gold = {}
        diff = {}
        colors = {}
        for k, source in enumerate(['human', 'inter', 'intra']):
            bar[source] = res[dataset_name][mode][source]
            bar_gold[source] = [bar[source]['gold'].get(str(j),0) for j in range(len(labels))]

            maj = np.argmax(np.array([bar[source]['gold'].get(str(j),0) for j in range(len(labels))]))
            diff[source] = [(bar[source]['pred'].get(str(j),0)-bar[source]['gold'].get(str(j),0))/bar[source]['gold'].get(str(j),0) for j in range(len(labels))]
            colors[source] = ['white' if i != maj else plt.cm.Set1(0) for i in range(len(labels))]
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


def density_plot(suffix, top_n=2, full=False, flatten=False):
    write_to = 'res'
    if flatten:
        write_to += '_flatten'
    write_to += suffix
    if top_n != 1:
        if full:
            write_to += '_full'
        else:
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
                filename = f"{dataset_name}-{m}-density-{suffix}-"
                if full:
                    filename += 'full'
                else:
                    filename += f'top{top_n}'
                plt.savefig(f"{filename}.png")
            plt.figure().clear()
            plt.close()

if __name__ == '__main__':
    #'''
    for top_n in [1, 2]:
        for suffix in ['_alpha0.8_whole_1e-05']:
            for full in [True, False]:
                if top_n == 1 and full:
                    continue
                write_to = f'res{suffix}'
                if top_n != 1:
                    if full:
                        write_to += '_full'
                    else:
                        write_to += f'_top{top_n}'
                write_to += '.json'
                with open(write_to) as f:
                    res = json.load(f)
                if 'ghc' in res:
                    del res['ghc']
                if 'SBIC' in res:
                    del res['SBIC']
                before_after_line_plots(res, suffix, full, top_n)
    '''
    for flatten in [False]:
        for full in [True, False]:
            for top_n in [1, 2]:
                #for suffix in ["_alpha0.0_whole_1e-05", "_alpha0.5_whole_1e-05", "_alpha0.8_whole_1e-05"]:
                for suffix in ["_alpha0.8_whole_1e-05"]:
                    main(suffix, top_n=top_n, full=full, flatten=flatten)
                    #density_plot(suffix, top_n=top_n, full=full, flatten=flatten)
    '''
    #with open(f'res{suffix}_full.json') as f:
    #    res = json.load(f)
    #bar_plot(res, 'data-frequency', suffix+'_top2')
    #bar_plot(res, 'frequency', suffix+'_top2')
    #bar_plot(res, 'sorted', suffix+'_top2')
    #bar_plot(res, 'shuffle', suffix+'_top2')
    #'''
'''
# Set the x axis labels and tick positions
'''        
