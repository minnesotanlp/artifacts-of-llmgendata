import os
import pickle

from common_utils import add_discourse_parsed_result, add_networkx_graphs
from datasets import load_dataset


if __name__ == '__main__':
    # 0 for machine-generated; 1 for human-written
    all_dataset = load_dataset('yaful/DeepfakeTextDetect')

    # splits = ['test_ood_gpt', 'test_ood_gpt_para', 'train', 'validation', 'test']
    splits = ['test_ood_gpt']

    MIN_LEN = 6
    SAVE_DIR = '/space4/zaemyung/Development/fifty_shades_of_human_writers/data/corpora/raw_dir'

    for split in splits:
        print(split)
        dataset = [sample for sample in all_dataset[split] if len(sample['text']) >= MIN_LEN]
        print(len(dataset))

        discourse_added_save_path = os.path.join(SAVE_DIR, f'DeepfakeTextDetect.{split}.discourse_added.jsonl')
        dataset = add_discourse_parsed_result(dataset, output_path=discourse_added_save_path)
        print(len(dataset))

        networkx_added_save_path = os.path.join(SAVE_DIR, f'DeepfakeTextDetect.{split}.discourse_added.networkx_added.pkl')
        dataset = add_networkx_graphs(dataset=discourse_added_save_path, output_path=networkx_added_save_path)
        print(len(dataset))
