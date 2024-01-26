import math
import random
random.seed(42)
import os
import pickle
from collections import Counter
from copy import deepcopy
from itertools import combinations, permutations
from multiprocessing import Manager, Pool

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import Progress, track
from tqdm.contrib.concurrent import process_map


def has_bidirectional_edges(G):
    for u in G.nodes:
        for v in G.nodes:
            if u == v:
                continue
            if G.has_edge(u, v) and G.has_edge(v, u):
                return True
    return False


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def visualize_graph_motifs(graphs_dict, n_nodes, domain, show_edge_label=False):
    '''
    :param graphs_dict: dict: { n_edges: list( nx.Graph() )}
    :return:
    '''
    flattened_graphs_dict = []
    if isinstance(graphs_dict, dict):
        for key in graphs_dict:
            for g in graphs_dict[key]:
                flattened_graphs_dict.append(g)
    elif isinstance(graphs_dict, list):
        flattened_graphs_dict = graphs_dict

    n_graphs = len(flattened_graphs_dict)
    print("[info] amount of graphs: ", n_graphs)
    column = 8 # (8 graphs in a line)
    if(n_graphs >= 100):
        column = 20
    row = math.ceil(n_graphs/column)
    print(f"[info] will provide a {column} x {row} figure")

    n_nodes = flattened_graphs_dict[0].number_of_nodes()

    figsize = (column*3, row*2)
    plt.clf()
    fig = plt.figure(figsize=figsize)
    for index, g in enumerate(flattened_graphs_dict):
        _col, _row = index%column, int(index/column)
        print(f"    img position: {_col}, {_row}")
        ax = fig.add_subplot(row, column, index+1)
        pos = nx.spring_layout(g)
        if show_edge_label:
            edge_labels=dict([((u, v), d['label_0']) for u, v, d in g.edges(data=True)])
            nx.draw_networkx_edge_labels(g, pos=pos, ax=ax, edge_labels=edge_labels)
        ax.set_title(f"G{index}", fontsize=8)
    fig.suptitle(f'All graph motifs with {n_nodes} nodes', fontsize=16)
    plt.tight_layout()

    plt.savefig(f'{domain}_{n_nodes}_motifs.png')


def save_graph_dict(n_nodes, domain, graphs):
    flattened_graphs_dict = graphs
    file_name_flattened = f"data/edge_labeled_from_corpus/M_{domain}_{n_nodes}_flattened.pkl"
    with open(file_name_flattened, "wb") as fh:
        pickle.dump(flattened_graphs_dict, fh)


def load_dataset(domain, corpus_dir='/nvme_pool/zaemyung/Development/human_vs_machine_texts/season_1/discourse_parsed'):
    path = os.path.join(corpus_dir, f'final_graph_and_edus_{domain}.pkl')
    with open(path, 'rb') as f:
        networkx_graphs = pickle.load(f)

    chatgpt_graphs = networkx_graphs['chatgpt']
    human_graphs = networkx_graphs['human']
    assert len(chatgpt_graphs) == len(human_graphs)

    return {'chatgpt': chatgpt_graphs, 'human': human_graphs}


def is_isomorphic_multiple(graphs, candidate_graph, ignore_index=None):
    for i, motif in enumerate(graphs):
        if isinstance(ignore_index, int) and ignore_index == i:
            continue
        DiGM = nx.algorithms.isomorphism.DiGraphMatcher(motif, candidate_graph, edge_match=lambda e1, e2: e1['label_0'] == e2['label_0'])
        if DiGM.is_isomorphic():
            return True
    return False


def contains_isolates(G):
    for n, d in G.degree():
        if d == 0:
            return True
    return False


def extract_subgraphs_all(graphs, node_size):
    unique_subgraphs = []

    for G in track(graphs, total=len(graphs)):
        for SG in (G.subgraph(s).copy() for s in combinations(G, node_size)):
        # for SG in (G.subgraph(s).copy() for s in nx.weakly_connected_components(G)):
            # print(SG.nodes(), SG.edges())

            # if len(list(nx.isolates(SG))) > 0 or has_bidirectional_edges(SG):
            if contains_isolates(SG):
                continue

            if is_isomorphic_multiple(unique_subgraphs, SG):
                continue

            unique_subgraphs.append(SG)

            # plt.clf()
            # pos = nx.spring_layout(SG, seed=24)
            # nx.draw(SG, pos)
            # nx.draw_networkx_labels(SG, pos)
            # edge_labels=dict([((u, v), d['label_0']) for u, v, d in SG.edges(data=True)])
            # nx.draw_networkx_edge_labels(SG, pos, edge_labels=edge_labels)
            # plt.savefig('subgraph.png')
            # input('next')

    return unique_subgraphs


def extract_subgraphs(G):
    node_size = 3
    subgraphs = {}
    for SG in (G.subgraph(s).copy() for s in combinations(G, node_size)):
        # if len(list(nx.isolates(SG))) > 0 or has_bidirectional_edges(SG):
        if contains_isolates(SG):
            continue

        sg_hash = nx.weisfeiler_lehman_graph_hash(SG, edge_attr='label_0')
        subgraphs[sg_hash] = SG

    return subgraphs


if __name__ == '__main__':
    hc3_datasets = {
        'finance': load_dataset('finance'),
        'medicine': load_dataset('medicine'),
        'open_qa': load_dataset('open_qa'),
        'reddit_eli5': load_dataset('reddit_eli5'),
        'wiki_csai': load_dataset('wiki_csai')
    }

    motif_size = 3

    for domain, graphs in hc3_datasets.items():
        print(domain)
        chatgpt_graphs = [g['graph'] for g in graphs['chatgpt']]
        human_graphs = [g['graph'] for g in graphs['human']]

        unique_subgraphs = extract_subgraphs_all(chatgpt_graphs + human_graphs, motif_size)
        save_graph_dict(motif_size, domain, unique_subgraphs)
        visualize_graph_motifs(unique_subgraphs, motif_size, domain, show_edge_label=True)


    splits = ['test_ood_gpt', 'test_ood_gpt_para', 'test']
    # splits = ['train', 'validation']
    # splits = ['test_ood_gpt']

    all_samples = []

    for split in splits:
        dataset_path = f'../data/DeepfakeTextDetect.{split}.discourse_added.networkx_added.pkl'
        if os.path.isfile(dataset_path):
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f'Could not find: {dataset_path}')
        print(len(dataset))

        all_samples.extend(dataset)

    # sample
    sources = set()
    for sample in all_samples:
        sources.add(sample['src'])

    print(sources, len(sources))

    num_samples_per_source = 50
    random.shuffle(all_samples)

    sampled_graphs = {k: [] for k in sources}

    for sample in all_samples:
        src = sample['src']
        graph = sample['graph']
        if len(sampled_graphs[src]) < num_samples_per_source:
            sampled_graphs[src].append(graph)

    all_graphs = []

    for graphs in sampled_graphs.values():
        all_graphs.extend(graphs)

    all_subgraphs_dict = {}
    all_subgraphs = process_map(extract_subgraphs, all_graphs, max_workers=64, chunksize=1)
    for subgraph_dict in all_subgraphs:
        for h, g in subgraph_dict.items():
            all_subgraphs_dict[h] = g

    # filter isomorphic ones once again
    unique_subgraphs = []
    for i, candidate in track(enumerate(all_subgraphs_dict.values()), total=len(all_subgraphs_dict)):
        if len(unique_subgraphs) < 1:
            unique_subgraphs.append(candidate)
            continue

        is_iso_flag = False
        for motif in unique_subgraphs:
            DiGM = nx.algorithms.isomorphism.DiGraphMatcher(motif, candidate, edge_match=lambda e1, e2: e1['label_0'] == e2['label_0'])
            if DiGM.is_isomorphic():
                is_iso_flag = True
                break
        if not is_iso_flag:
            unique_subgraphs.append(candidate)

    print(len(unique_subgraphs))

    save_paths = ['/space4/zaemyung/Development/human_vs_machine_texts/season_2/motifs/data/edge_labeled_from_corpus/M_deepfaketext_3_flattened.pkl',
                  '/space4/zaemyung/Development/human_vs_machine_texts/season_2/motifs/data/edge_labeled_from_corpus/M_deepfaketext_3_flattened_2.pkl',
                  '/space4/zaemyung/Development/human_vs_machine_texts/season_2/motifs/data/edge_labeled_from_corpus/M_all_3_flattened.pkl']
    all_motifs = []
    for path in save_paths:
        with open(path, 'rb') as f:
            motifs = pickle.load(f)
            all_motifs.extend(motifs)

    # filter isomorphic ones once again
    unique_subgraphs = []
    for i, candidate in track(enumerate(all_motifs), total=len(all_motifs)):
        if len(unique_subgraphs) < 1:
            unique_subgraphs.append(candidate)
            continue

        is_iso_flag = False
        for motif in unique_subgraphs:
            DiGM = nx.algorithms.isomorphism.DiGraphMatcher(motif, candidate, edge_match=lambda e1, e2: e1['label_0'] == e2['label_0'])
            if DiGM.is_isomorphic():
                is_iso_flag = True
                break
        if not is_iso_flag:
            unique_subgraphs.append(candidate)

    for i, subgraph in enumerate(unique_subgraphs):
        if is_isomorphic_multiple(unique_subgraphs, subgraph, i):
            print('Fail')
    print('Success')

    save_path = '/space4/zaemyung/Development/human_vs_machine_texts/season_2/motifs/data/edge_labeled_from_corpus/M_all-combined_3_flattened.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(unique_subgraphs, f)

    print(len(unique_subgraphs))