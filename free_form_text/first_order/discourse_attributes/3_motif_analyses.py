import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from common_utils import load_pickle_file, save_pickle_file

random.seed(42)


def compute_difference_distribution(machines, humans, distribution_type='norm_by_edges', motif_size=3):
    if distribution_type == 'raw':
        distribution_type = f'motif_m{motif_size}_hist_raw'
    elif distribution_type == 'norm_by_edges':
        distribution_type = f'motif_m{motif_size}_hist_norm_by_edges'
    elif distribution_type == 'norm_by_motifs':
        distribution_type = f'motif_m{motif_size}_hist_norm_by_motifs'
    else:
        raise ValueError(f'Unknown distribution_type: {distribution_type}')

    machine_dists = [sample[distribution_type] for sample in machines]
    human_dists = [sample[distribution_type] for sample in humans]

    machine_distribution = np.mean(machine_dists, axis=0)
    human_distribution = np.mean(human_dists, axis=0)

    difference_distribution = machine_distribution - human_distribution
    return difference_distribution


def plot_difference_distribution(df_dists, save_path, selected_labels=None):
    plt.rcParams['figure.figsize'] = [40, 9]

    # fig, axs = plt.subplots(ncols=3, nrows=1)
    fig, axs = plt.subplots(len(df_dists.columns))

    # df_dists = df_dists.loc[:45]

    xticks_length = len(df_dists)
    labels = np.arange(0, xticks_length, 1)
    colors = ['dimgrey'] * xticks_length

    if selected_labels is not None and isinstance(selected_labels, list):
        labels = [''] * xticks_length
        for label in selected_labels:
            labels[label] = str(label)
            colors[label] = 'orangered'

    for i, col_name in enumerate(df_dists.columns):
        sns.barplot(data=df_dists[col_name], ax=axs[i], palette=colors)
        axs[i].set_xticks(ticks=np.arange(0, xticks_length, 1), labels=labels, rotation=20)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    fig.savefig(save_path)
    plt.clf()


def visualize_graph_motifs(motifs, save_path, selected_motif_indices=None, show_edge_label=True):
    assert isinstance(motifs, list)

    motifs_indices = list(range(len(motifs)))

    if isinstance(selected_motif_indices, list) and len(selected_motif_indices) > 0:
        motifs = [motifs[i] for i in selected_motif_indices]
        motifs_indices = selected_motif_indices


    n_graphs = len(motifs)
    print("[info] amount of graphs: ", n_graphs)
    column = 3
    if(n_graphs >= 100):
        column = 20
    row = math.ceil(n_graphs/column)
    print(f"[info] will provide a {column} x {row} figure")

    figsize = (column*4, row*3)
    plt.clf()
    fig = plt.figure(figsize=figsize)
    for index, g in enumerate(motifs):
        _col, _row = index%column, int(index/column)
        print(f"    img position: {_col}, {_row}")
        ax = fig.add_subplot(row, column, index+1)
        # pos = nx.spring_layout(g)
        pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
        nx.draw(g, pos=pos, ax=ax, node_size=25)
        if show_edge_label:
            edge_labels=dict([((u, v), d['label_0']) for u, v, d in g.edges(data=True)])
            nx.draw_networkx_edge_labels(g, pos=pos, ax=ax, edge_labels=edge_labels)
        ax.set_title(f"G{motifs_indices[index]}", fontsize=13)
    # fig.suptitle(f'Distinctive Network Motifs', fontsize=16)
    plt.tight_layout()

    plt.savefig(save_path)


def is_focus_domain(source, focus_domains):
    for domain in focus_domains:
        if source[:len(domain)] == f'{domain}':
            return domain
    return False


def extract_interesting_motif_indices(difference_distributions, domain, num_std=2):
    diff_dist = difference_distributions[domain]['machine_mfidf-human_mfidf']
    abs_diff_dist = np.absolute(diff_dist)
    # median = np.median(abs_diff_dist)  # median is 0
    mean = np.mean(abs_diff_dist)
    std = np.std(abs_diff_dist)
    indices = np.argwhere(abs_diff_dist >= mean + (std * num_std))
    indices = np.squeeze(indices)
    return indices.tolist()


if __name__ == '__main__':
    corpora_dir = 'data/corpora/raw'
    plot_dir = 'data/plots'
    dataset_names = ['HC3.train.discourse_added.networkx_added.motifs_added.motifs_M6-M9_added.pkl',
                     'HC3.valid.discourse_added.networkx_added.motifs_added.motifs_M6-M9_added.pkl',
                     'DeepfakeTextDetect.train.discourse_added.networkx_added.motifs_added.motifs_M6-M9_added.pkl',
                     'DeepfakeTextDetect.validation.discourse_added.networkx_added.motifs_added.motifs_M6-M9_added.pkl']
    motif_size_to_path = {3: 'data/motifs/M_3_HC3-DeepfakeTextDetect.pkl',
                          6: 'data/motifs/M_6_HC3-DeepfakeTextDetect.pkl',
                          9: 'data/motifs/M_9-triangular_HC3-DeepfakeTextDetect.pkl'}

    # sources_to_focus = ['yelp', 'wp', 'cmv', 'xsum', 'sci_gen']
    sources_to_focus = ['cmv', 'yelp', 'xsum', 'tldr', 'eli5', 'wp', 'roct', 'hswag', 'squad', 'sci_gen',
                        'finance', 'medicine', 'open_qa', 'reddit_eli5', 'wiki_csai']

    motif_sizes = [3, 6, 9]
    interesting_indices = {}
    for motif_size in motif_sizes:
        assert motif_size in [3, 6, 9]
        num_motifs_mapping = {3: 69, 6: 592, 9: 2394}
        num_motifs = num_motifs_mapping[motif_size]
        motifs = load_pickle_file(motif_size_to_path[motif_size])

        humans = {src: [] for src in sources_to_focus}
        machines = {src: [] for src in sources_to_focus}

        for d_name in dataset_names:
            dataset_path = os.path.join(corpora_dir, d_name)
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)

            for sample in dataset:
                domain = is_focus_domain(sample['src'], sources_to_focus)
                if domain is False:
                    continue

                authorship_tag = sample['src'].split('_')[-1]
                if authorship_tag == 'human':
                    humans[domain].append(sample)
                else:
                    machines[domain].append(sample)

        difference_distributions = {}
        for domain in sources_to_focus:
            random.shuffle(humans[domain])
            random.shuffle(machines[domain])

            min_len = min(len(humans[domain]), len(machines[domain]))
            humans[domain] = humans[domain][:min_len]
            machines[domain] = machines[domain][:min_len]
            print(f'no. of samples for {domain}: {min_len}')

            diff_dist = compute_difference_distribution(machines=machines[domain], humans=humans[domain],
                                                        distribution_type='norm_by_edges', motif_size=motif_size)
            difference_distributions[domain] = diff_dist

        df_difference_distributions = pd.DataFrame.from_dict(difference_distributions)
        print(df_difference_distributions)
        save_path = os.path.join(plot_dir, f'motif_m{motif_size}_diff_dist-norm_by_edges.pdf')
        plot_difference_distribution(df_difference_distributions, save_path=save_path,
                                     selected_labels=[0, 1, 5, 7, 8, 10, 18, 19, 22, 27, 28, 33, 34, 41])

        motifs_path = 'data/motifs/M_3_HC3-DeepfakeTextDetect.pkl'
        with open(motifs_path, 'rb') as f:
            motifs = pickle.load(f)
        save_path = os.path.join(plot_dir, 'all_motifs_of_size_three.pdf')
        # visualize_graph_motifs(motifs, save_path=save_path, selected_motif_indices=[1, 7, 18, 19, 33, 34], show_edge_label=True)
        visualize_graph_motifs(motifs, save_path=save_path, selected_motif_indices='all', show_edge_label=True)

        all_indices = set()
        for domain in difference_distributions:
            indices = extract_interesting_motif_indices(difference_distributions, domain, num_std=1)
            all_indices.update(indices)

        interesting_indices[motif_size] = sorted(list(all_indices))
        print(f'{motif_size}: {len(interesting_indices[motif_size])}')

        save_path = os.path.join(plot_dir, f'interesting_motifs_m{motif_size}.pdf')
        visualize_graph_motifs(motifs=motifs, save_path=save_path, selected_motif_indices=interesting_indices)
    save_pickle_file(interesting_indices, 'data/motifs/selected_motif_indices_one-std_HC3-DeepfakeTextDetect.pkl')