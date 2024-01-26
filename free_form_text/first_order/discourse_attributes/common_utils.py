import json
import os
import pickle
import sys
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import regex as re
from rich.progress import track

# git clone git@github.com:zaemyung/DMRST_Parser.git
DMRST_PARSER_DIR_PATH = '/space4/zaemyung/Development/DMRST_Parser'
if not os.path.isdir(DMRST_PARSER_DIR_PATH):
    DMRST_PARSER_DIR_PATH = '/Users/zaemyung/Development/DMRST_Parser'
# download the model from following the link in `https://github.com/zaemyung/DMRST_Parser/tree/main/depth_mode/Savings`
DMRST_PARSER_MODEL_PATH = os.path.join(DMRST_PARSER_DIR_PATH, 'depth_mode/Savings/multi_all_checkpoint.torchsave')
sys.path.append(DMRST_PARSER_DIR_PATH)

from MUL_main_Infer import DiscourseParser


def load_pickle_file(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle_file(things, path: str):
    with open(path, 'wb') as f:
        pickle.dump(things, f)


def load_jsonl_file(path: str):
    results = []
    with open(path, 'r') as f:
        for json_line in f:
            results.append(json.loads(json_line))
    return results


def chunks(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def draw_graph(G, save_path):
    plt.clf()
    fig = plt.figure(figsize=(20,20))
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, node_size=25)
    edge_labels=dict([((u, v), d['label_0']) for u, v, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.savefig(save_path)
    plt.clf()


def run_discourse_parser(texts: List[str], batch_size: int = 20, parser_model_path: str = DMRST_PARSER_MODEL_PATH, return_filtered_indices=False):
    if isinstance(texts, str):
        texts = [texts]
    assert isinstance(texts, list)

    discourse_parser = DiscourseParser(model_path=parser_model_path)
    tokens, segments, parsed = discourse_parser.parse(texts, batch_size=batch_size)
    assert len(tokens) == len(segments) == len(parsed) == len(texts)

    results = []
    filtered_indices = []

    for i, text in enumerate(texts):
        if parsed[i][0] == 'NONE':
            print(f'{i}-th sample is filtered.')
            filtered_indices.append(i)
            continue

        edus = {}
        last_end = 0
        for edu_i, edu_end in enumerate(segments[i], start=1):
            edu = ''.join(tokens[i][last_end:edu_end+1]).replace('▁', ' ')
            edus[f'span_{edu_i}-{edu_i}'] = edu
            # print(f'{edu_i}:{edu}')
            last_end = edu_end + 1

        results.append({
            'text': text,
            'tokenized': tokens[i],
            'segments': segments[i],
            'edus': edus,
            'parsed': parsed[i][0],
        })

    if return_filtered_indices:
        return results, filtered_indices
    return results


def create_graph_from_const_format(format_string: str, save_graph_plot: bool = False):
    spans = format_string.strip().split(' ')
    rgx_span = r'\((\d+):(.+)=(.+):(\d+),(\d+):(.+)=(.+):(\d+)\)'
    edges = []
    nodes_types = {}
    edu_indices = set()
    for i, span in enumerate(spans, start=1):
        m_span = re.match(rgx_span, span)
        # if m_span is None:
        #     print(sample_index)
        #     print(format_string)
        #     print(span)
        assert m_span is not None
        left_most_edu_index = int(m_span.group(1))
        right_most_edu_index = int(m_span.group(8))

        left_type = m_span.group(2)
        left_relation = m_span.group(3)
        left_end_edu_index = int(m_span.group(4))

        right_start_edu_index = int(m_span.group(5))
        right_type = m_span.group(6)
        right_relation = m_span.group(7)

        node_label = f'span_{left_most_edu_index}-{right_most_edu_index}'
        left_node_label = f'span_{left_most_edu_index}-{left_end_edu_index}'
        right_node_label = f'span_{right_start_edu_index}-{right_most_edu_index}'

        edu_indices.add(left_most_edu_index)
        edu_indices.add(right_most_edu_index)
        edu_indices.add(left_end_edu_index)
        edu_indices.add(right_start_edu_index)

        edges.append((left_node_label, node_label, '/'))
        edges.append((right_node_label, node_label, '/'))

        left_node_type = left_type
        right_node_type = right_type

        if left_relation != 'span':
            edges.append((left_node_label, right_node_label, left_relation))
        if right_relation != 'span':
            edges.append((right_node_label, left_node_label, right_relation))

        if i == 1:
            root_node_label = node_label
            nodes_types[root_node_label] = 'root'

        nodes_types[left_node_label] = left_node_type
        nodes_types[right_node_label] = right_node_type

    G = nx.DiGraph()
    for u, v, label in edges:
        # G.add_edge(u, v, label=label)
        G.add_edge(u, v, label_0=label)

    nx.set_node_attributes(G, nodes_types, 'label_0')

    assert root_node_label == f'span_1-{max(edu_indices)}'

    if save_graph_plot:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', root=root_node_label)
        pos = {node: (-x, -y) for (node, (x, y)) in pos.items()}

        nodes_shapes = set(['root', 'Nucleus', 'Satellite'])
        shape_mapping = {
            'root': '*',
            'Nucleus': '8',
            'Satellite': 'o'
        }
        plt.figure(figsize=(12,10))
        print(G.nodes(data=True))
        print(G.edges(data=True))

        for shape in nodes_shapes:
            nx.draw_networkx(
            G,
            pos,
            with_labels=True,
            # with_labels=False,
            node_shape = shape_mapping[shape],
            node_size=1000,
            font_size=12,
            nodelist = [
                node[0] for node in filter(lambda x: x[1]['label_0'] == shape, G.nodes(data=True))
            ],
            arrowsize=20
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label_0'))
        plt.savefig('graph.png')

    return G


def add_discourse_parsed_result(data_split_set, output_path=None):
    texts = [sample['text'] for sample in data_split_set]
    texts_discourse_added, filtered_indices = run_discourse_parser(texts, return_filtered_indices=True)
    data_split_set = [sample for index, sample in enumerate(data_split_set) if index not in set(filtered_indices)]
    assert len(data_split_set) == len(texts_discourse_added)
    if len(filtered_indices) > 0:
        print(f'filtered {len(filtered_indices)} samples due to length.')

    if output_path is None:
        return data_split_set

    with open(output_path, 'w') as f:
        for sample, discourse_result in track(zip(data_split_set, texts_discourse_added), total=len(texts_discourse_added)):
            assert discourse_result['text'] == sample['text']
            sample['tokenized'] = discourse_result['tokenized']
            sample['segments'] = discourse_result['segments']
            sample['edus'] = discourse_result['edus']
            sample['parsed'] = discourse_result['parsed']
            f.write(f'{json.dumps(sample, indent=None)}\n')

    return data_split_set


def add_networkx_graphs(dataset, output_path=None):
    if isinstance(dataset, str):
        assert '.jsonl' in dataset
        dataset = load_jsonl_file(dataset)

    for sample in track(dataset, total=len(dataset)):
        parsed_discourse = sample['parsed']
        graph = create_graph_from_const_format(parsed_discourse, save_graph_plot=False)
        sample['graph'] = graph

    if output_path is None:
        return dataset

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset


def is_motif_present(G, motif):
    DiGM = nx.algorithms.isomorphism.DiGraphMatcher(G, motif, edge_match=lambda e1, e2: e1['label_0'] == e2['label_0'])
    if next(DiGM.subgraph_isomorphisms_iter(), 'EMPTY') == 'EMPTY':
        return False
    return True


def count_motif_from_graph(G, motif, normalize=True):
    count = 0
    DiGM = nx.algorithms.isomorphism.DiGraphMatcher(G, motif, edge_match=lambda e1, e2: e1['label_0'] == e2['label_0'])

    for subgraph in DiGM.subgraph_isomorphisms_iter():
        count += 1
    if normalize:
        count /= G.number_of_edges()
    return count


def calc_motif_distribution(G, graph_motifs):
    # graph_motifs = list(graph_motifs.values())

    hist = np.zeros(len(graph_motifs), dtype=int)
    for index, motif in enumerate(graph_motifs):
        if(motif.number_of_nodes() == 1):
            hist[index] = G.number_of_nodes()
            continue
        if(motif.number_of_nodes() == 2):
            hist[index] = G.number_of_edges()
            continue

        if(index % 10 == 0):
            pass

        DiGM = nx.algorithms.isomorphism.DiGraphMatcher(G, motif, edge_match=lambda e1, e2: e1['label_0'] == e2['label_0'])
        for subgraph in DiGM.subgraph_isomorphisms_iter():
            hist[index] += 1

    norm_hist_by_edges = hist / G.number_of_edges()
    num_of_motifs = np.sum(hist)
    norm_hist_by_motifs = hist / num_of_motifs if num_of_motifs > 0 else hist
    return hist, norm_hist_by_edges, norm_hist_by_motifs
