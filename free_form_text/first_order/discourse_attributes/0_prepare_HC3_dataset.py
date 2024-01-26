import os
import pickle
import random
from copy import deepcopy

random.seed(42)

from common_utils import add_discourse_parsed_result, add_networkx_graphs
from datasets import load_dataset

if __name__ == '__main__':
    hc3_datasets_raw = {
        'finance': load_dataset('Hello-SimpleAI/HC3', 'finance'),
        'medicine': load_dataset('Hello-SimpleAI/HC3', 'medicine'),
        'open_qa': load_dataset('Hello-SimpleAI/HC3', 'open_qa'),
        'reddit_eli5': load_dataset('Hello-SimpleAI/HC3', 'reddit_eli5'),
        'wiki_csai': load_dataset('Hello-SimpleAI/HC3', 'wiki_csai')
    }

    # dict_keys(['id', 'question', 'human_answers', 'chatgpt_answers'])
    # print(hc3_datasets_raw['finance']['train'][0].keys())

    MIN_LEN = 6

    hc3_datasets = {}
    for domain in hc3_datasets_raw.keys():
        hc3_datasets[domain] = []
        for sample in hc3_datasets_raw[domain]['train']:
            del sample['id']

            try:
                human_answer = sample['human_answers'][0].strip()
                chatgpt_answer = sample['chatgpt_answers'][0].strip()
            except IndexError:
                continue

            if len(human_answer) < MIN_LEN or len(chatgpt_answer) < MIN_LEN:
                continue

            human_sample = deepcopy(sample)
            chatgpt_sample = deepcopy(sample)

            human_sample['text'] = human_answer
            chatgpt_sample['text'] = chatgpt_answer
            del human_sample['human_answers']
            del human_sample['chatgpt_answers']
            del chatgpt_sample['human_answers']
            del chatgpt_sample['chatgpt_answers']

            human_sample['label'] = 1
            chatgpt_sample['label'] = 0
            human_sample['src'] = f'{domain}_human'
            chatgpt_sample['src'] = f'{domain}_chatgpt'
            hc3_datasets[domain].append(human_sample)
            hc3_datasets[domain].append(chatgpt_sample)
        # print(domain, len(hc3_datasets[domain]))

    SAVE_DIR = '/space4/zaemyung/Development/fifty_shades_of_human_writers/data/corpora/raw_dir'

    for domain in hc3_datasets.keys():
        dataset = hc3_datasets[domain]
        discourse_added_save_path = os.path.join(SAVE_DIR, f'HC3.{domain}.discourse_added.jsonl')
        dataset = add_discourse_parsed_result(dataset, output_path=discourse_added_save_path)

        networkx_added_save_path = os.path.join(SAVE_DIR, f'HC3.{domain}.discourse_added.networkx_added.pkl')
        dataset = add_networkx_graphs(dataset=discourse_added_save_path, output_path=networkx_added_save_path)

    VALID_SIZE_PER_DOMAIN = 200
    TEST_SIZE_PER_DOMAIN = 400

    all_train = []
    all_valid = []
    all_test = []
    for domain in hc3_datasets.keys():
        networkx_added_save_path = os.path.join(SAVE_DIR, f'HC3.{domain}.discourse_added.networkx_added.pkl')
        with open(networkx_added_save_path, 'rb') as f:
            dataset = pickle.load(f)

        random.shuffle(dataset)

        all_valid.extend(dataset[:VALID_SIZE_PER_DOMAIN])
        all_test.extend(dataset[VALID_SIZE_PER_DOMAIN:VALID_SIZE_PER_DOMAIN + TEST_SIZE_PER_DOMAIN])
        all_train.extend(dataset[VALID_SIZE_PER_DOMAIN + TEST_SIZE_PER_DOMAIN:])

    random.shuffle(all_train)
    random.shuffle(all_valid)
    random.shuffle(all_test)

    print(len(all_train), len(all_valid), len(all_test))

    train_save_path = os.path.join(SAVE_DIR, f'HC3.train.discourse_added.networkx_added.pkl')
    with open(train_save_path, 'wb') as f:
        pickle.dump(all_train, f)

    valid_save_path = os.path.join(SAVE_DIR, f'HC3.valid.discourse_added.networkx_added.pkl')
    with open(valid_save_path, 'wb') as f:
        pickle.dump(all_valid, f)

    test_save_path = os.path.join(SAVE_DIR, f'HC3.test.discourse_added.networkx_added.pkl')
    with open(test_save_path, 'wb') as f:
        pickle.dump(all_test, f)
