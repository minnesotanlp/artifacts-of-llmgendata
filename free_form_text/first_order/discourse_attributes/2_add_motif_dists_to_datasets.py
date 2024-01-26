import multiprocessing as mp
import os
import pickle

from common_utils import calc_motif_distribution, load_pickle_file, save_pickle_file
from tqdm.contrib.concurrent import process_map


def add_motif_dist_to_sample(sample):
    graph = sample['graph']
    sample['motif_m3_hist_raw'] = sample['motif_hist_raw']
    sample['motif_m3_hist_norm_by_edges'] = sample['motif_hist_norm_by_edges']
    sample['motif_m3_hist_norm_by_motifs'] = sample['motif_hist_norm_by_motifs']

    del sample['motif_hist_raw']
    del sample['motif_hist_norm_by_edges']
    del sample['motif_hist_norm_by_motifs']

    hist, norm_hist_by_edges, norm_hist_by_motifs = calc_motif_distribution(graph, motifs_m6)
    sample['motif_m6_hist_raw'] = hist
    sample['motif_m6_hist_norm_by_edges'] = norm_hist_by_edges
    sample['motif_m6_hist_norm_by_motifs'] = norm_hist_by_motifs

    hist, norm_hist_by_edges, norm_hist_by_motifs = calc_motif_distribution(graph, motifs_m9)
    sample['motif_m9_hist_raw'] = hist
    sample['motif_m9_hist_norm_by_edges'] = norm_hist_by_edges
    sample['motif_m9_hist_norm_by_motifs'] = norm_hist_by_motifs
    return sample


if __name__ == '__main__':
    CORPORA_DIR = 'data/corpora/raw'
    MOTIFS_DIR = 'data/motifs'

    motifs_m3 = load_pickle_file(os.path.join(MOTIFS_DIR, 'M_3_HC3-DeepfakeTextDetect.pkl'))
    motifs_m6 = load_pickle_file(os.path.join(MOTIFS_DIR, 'M_6_HC3-DeepfakeTextDetect.pkl'))
    motifs_m9 = load_pickle_file(os.path.join(MOTIFS_DIR, 'M_9-triangular_HC3-DeepfakeTextDetect.pkl'))
    print(f'motifs_M3 len = {len(motifs_m3)}')
    print(f'motifs_M6 len = {len(motifs_m6)}')
    print(f'motifs_M9 len = {len(motifs_m9)}')

    # hc3_names = [
    #     'HC3.train.discourse_added.networkx_added.motifs_added.pkl',
    #     'HC3.valid.discourse_added.networkx_added.motifs_added.pkl',
    #     'HC3.test.discourse_added.networkx_added.motifs_added.pkl',
    # ]

    # splits = ['test_ood_gpt', 'test_ood_gpt_para', 'train', 'validation', 'test']
    splits = ['test', 'validation', 'train']
    deepfake_names = [f'DeepfakeTextDetect.{split}.discourse_added.networkx_added.motifs_added.pkl' for split in splits]

    # dataset_names = hc3_names + deepfake_names
    dataset_names = deepfake_names

    for name in dataset_names:
        print(name)
        dataset_path = os.path.join(CORPORA_DIR, name)
        assert os.path.isfile(dataset_path)
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        print(len(dataset))

        dataset = process_map(add_motif_dist_to_sample, dataset, max_workers=64, chunksize=1)
        # results = []
        # for sample in dataset:
        #     results.append(add_motif_dist_to_sample(sample))
        # print(len(results))

        save_path = os.path.join(CORPORA_DIR, f'{name[:-4]}.motifs_M6-M9_added.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
