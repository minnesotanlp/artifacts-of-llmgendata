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

def before_after_line_plots(res, group='inter'):
    color_lst = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'gray', 'olive', 'cyan']
    line_lst = ['-', '--', '-.', ':']
    for dataset_name in res:
        num_labels = utils.get_num_labels(dataset_name)
        for m in res[dataset_name]:
            for sources in [['human', 'inter'], ['human', 'intra']]:  
                x = list(range(num_labels))
                fig, ax = plt.subplots()
                plot_ind = 0
                for source in sources: 
                    y = [res[dataset_name][m][source]['gold'].get(i,0) for i in range(num_labels)]
                    ax.plot(x, y, color=color_lst[plot_ind], marker=',', linestyle=line_lst[0], label=f'{source if source=="human" else "model"}-gold')
                    y = [res[dataset_name][m][source]['pred'].get(i,0) for i in range(num_labels)]
                    ax.plot(x, y, color=color_lst[plot_ind], marker=',', linestyle=line_lst[1], label=f'{source if source=="human" else "model"}-pred')
                    plot_ind += 1
                ax.legend()

                # Add labels and a title
                ax.set_xlabel('Labels')
                ax.set_ylabel('Counts')
                # set xticks
                x_ticks = list(range(num_labels))
                ax.set_xticks(x_ticks)
                ax.set_title(f'{dataset_name}-{m}')
                # Display the plot
                plt.savefig(f"{dataset_name}-{m}-{group}-before-after.png")
                plt.figure().clear()
                plt.close()

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


def main():
    # TODO: clean up code
    res = {} 
    switch = {}
    gold_inter = get_gold('inter')
    gold_intra = get_gold('intra')
    suffix = "_alpha0.8_whole"

    modes = ['frequency', 'dataset-frequency', 'sorted', 'shuffle']
    #temp = [item for row in list(gold_inter['SBIC']['human']) for item in row]
    for dataset_name in ['SChem5Labels']:#, 'SBIC', 'ghc', 'SChem5Labels']:
    #for dataset_name in ['Sentiment']:#, 'SBIC', 'ghc', 'SChem5Labels']:
        max_ct = 0
        min_ct = 1000000
        res[dataset_name] = {}
        switch[dataset_name] = {}
        num_labels = utils.get_num_labels(dataset_name)
        for m in modes: 
            switch[dataset_name][m] = {} 
            res[dataset_name][m] = {}
            for source in ['human', 'inter', 'intra']:
                switch[dataset_name][m][source] = {l: {l: 0 for l in range(num_labels)} for l in range(num_labels)}
                res[dataset_name][m][source] = {}
                for t in ['pred', 'gold']:
                    res[dataset_name][m][source][t] = []

            #for source in ['human', 'model']:

            ''' not using
                for i, row in enumerate(gold_inter[dataset_name][source]):
                    row = row.replace('nan', '-1').replace('.', '')
                    row = re.sub(r'(\d)\s', r'\1, ', row)
                    gold_inter[dataset_name][source][i] = eval(row)
                gold_inter[dataset_name][source] = [int(item) for row in gold_inter[dataset_name][source] for item in row]
           ''' 

            #for m in modes: 
            print(dataset_name, m, "annots", utils.get_num_annots(dataset_name), "labels", utils.get_num_labels(dataset_name), '========================================')
            # get human data
            #human_inds = f'{dataset_name}-roberta-base-inter-dataset-{m}-human_annots-indices.pkl' 
            human_data = f'results_new/{dataset_name}-roberta-base-inter-{m}-human_annots{suffix}.pkl'
            with open(human_data, 'rb') as f:
                human_data = pickle.load(f)
                human_data = human_data[0] # eval stuff at index 1
            print('num rows',len(human_data[0]))
            print('num cols',len(human_data[0][0]))
            return
            logits = np.array(human_data[0]).transpose() #prediction output
            predictions = np.argmax(logits, axis=0).flatten()
            human_labels = human_data[1][:,1:].flatten() #label ids
            res[dataset_name][m]['human']['pred'] = counter_to_sorted_dict(Counter(predictions)) 
            res[dataset_name][m]['human']['gold'] = counter_to_sorted_dict(Counter(human_labels)) 
            print(res[dataset_name][m]['human']['gold'])
            print(Counter(utils.flatten_recursive(gold_inter[dataset_name]['human'])))
            print(Counter(utils.flatten_recursive(gold_intra[dataset_name]['human'])))
            print("")
            s = sum(int(predictions[i]!=human_labels[i]) for i in range(len(predictions)))
            for i in range(len(predictions)):
                if predictions[i] == -1 or human_labels[i] == -1:
                    continue
                switch[dataset_name][m]['human'][human_labels[i]][predictions[i]] += 1
            
            inter_data = f'results_new/{dataset_name}-roberta-base-inter-{m}-model_annots{suffix}.pkl'
            with open(inter_data, 'rb') as f:
                inter_data = pickle.load(f)[0]
            logits = np.array(inter_data[0]).transpose()
            predictions = np.argmax(logits, axis=0).flatten()
            inter_labels = inter_data[1][:,1:].flatten()
            res[dataset_name][m]['inter']['pred'] = counter_to_sorted_dict(Counter(predictions))
            res[dataset_name][m]['inter']['gold'] = counter_to_sorted_dict(Counter(inter_labels))
            print(res[dataset_name][m]['inter']['gold'])
            print(Counter(utils.flatten_recursive(gold_inter[dataset_name]['model'])))
            print("")
            for i in range(len(predictions)):
                if predictions[i] == -1 or inter_labels[i] == -1:
                    continue
                switch[dataset_name][m]['inter'][inter_labels[i]][predictions[i]] += 1

            #intra_inds = f'{dataset_name}-roberta-base-intra-dataset-{m}-model_annots-indices.pkl' 
            intra_data = f'results_new/{dataset_name}-roberta-base-intra-{m}-model_annots{suffix}.pkl'
            with open(intra_data, 'rb') as f:
                intra_data = pickle.load(f)[0]
            logits = np.array(intra_data[0]).transpose()
            predictions = np.argmax(logits, axis=0).flatten()
            intra_labels = intra_data[1][:,1:].flatten()
            res[dataset_name][m]['intra']['pred'] = counter_to_sorted_dict(Counter(predictions))
            res[dataset_name][m]['intra']['gold'] = counter_to_sorted_dict(Counter(intra_labels))
            print(res[dataset_name][m]['intra']['gold'])
            print(Counter(utils.flatten_recursive(gold_intra[dataset_name]['model'])))
            print("")
            for i in range(len(predictions)):
                if predictions[i] == -1 or intra_labels[i] == -1:
                    continue
                switch[dataset_name][m]['intra'][intra_labels[i]][predictions[i]] += 1
    #movement_heatmap(switch)
    #before_after_line_plots(res)
    #print(json.dumps(switch, indent=4))

if __name__ == '__main__':
    main()
