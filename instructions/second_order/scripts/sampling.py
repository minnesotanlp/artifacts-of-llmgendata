import gc, json, os, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from jsonargparse import CLI, ArgumentParser
import numpy as np
import hdbscan
import random
import torch

from tqdm import tqdm
from rouge import Rouge

from datasets import load_dataset

import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

pd.options.mode.chained_assignment = None

TOTAL_PATH = Path('/corpora/active-instruction-tuning/datasets/dedup')
SAMPLED_PATH = Path('/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled')
# VALIDATION_PATH = Path('/corpora/InstructTune/active-instruction-tuning/datasets/validation/validation_10000.parquet.gzip')
INTER_DUPLICATED_PATH = Path('/corpora/active-instruction-tuning/datasets/grouped/duplicated_final.pickle')
LLAMA_PATH = "/corpora/InstructTune/llama/llama_2_hf/llama-2-7b/7bweights"
MODEL_NAME = 'all-mpnet-base-v2'
SAMPLE_INSTANCES = 10000

DATA_PATH = DATA_PATH = '../datasets/fine-tuning_data/'
ROUGE_N_SAMPLES = 1000

class Sampler:
    def __init__(self, args):
        self.sample_type = args.sample_type
        self.n_instances = args.n_instances
        self.random_state = args.random_state
        self.extract_validation = args.extract_validation
        self.data_set = args.data_set
        self.ftype = args.ftype
        self._set_seed(self.random_state)
    
    def load_sni(self):
        #load super natural instructions from huggingface, this is separate from the sni loaded in load_datasets,
        #where we load sni from a different locally downloaded dataset
        saved_path = Path(os.path.join(DATA_PATH, self.data_set))
        saved_file = os.path.join(saved_path, 'data.json')

        dataset = load_dataset("Muennighoff/natural-instructions")
        sampled_results = pd.DataFrame(dataset['train'])
        sampled_results.rename(columns={'definition': 'instruction', 'inputs': 'input', 'targets': 'output'}, inplace=True)
        sampled_results['categories'] = sampled_results['task_name'].str.extract(r'_(\w+)')
        sampled_results['source'] = 'sni'
        if not os.path.exists(saved_file):
            saved_path.mkdir(parents=True, exist_ok=True)
            sampled_results.to_csv(saved_file, index=False, escapechar="\\")
            prompts = (sampled_results['instruction'] + " " + sampled_results['input']).tolist()
            gts = sampled_results['output'].tolist()
            prompt_file = os.path.join(saved_path, 'prompts.json')
            gts_file = os.path.join(saved_path, 'gts.json')
            with open(prompt_file, 'w') as f:
                json.dump(prompts, f)
            with open(gts_file, 'w') as f:
                json.dump(gts, f)
        else:
            sampled_results = pd.read_csv(saved_file)

            if all(column in sampled_results.columns for column in ['instruction', 'input', 'output', 'source']):
                print("DataFrame loaded successfully with the desired columns.")
            else:
                print("Some columns are missing from the loaded DataFrame.")
            print(sampled_results.head())
            print(len(sampled_results))
            print(sampled_results.columns)
            print(sampled_results.iloc[0])
        
        return sampled_results

    def load_si(self):
        #Load self instruct dataset from the downloaded json file
        saved_path = Path(os.path.join(DATA_PATH, self.data_set))
        saved_file = os.path.join(saved_path, 'data.json')
        data_rows = []  
        si_file = "/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/si_data.json"
        if not os.path.exists(saved_file):
            with open(si_file, 'r') as f:
                for line in f:
                    json_data = json.loads(line)
                    json_data['source'] = 'self_instruct'
                    data_rows.append(json_data)
            sampled_results = pd.DataFrame(data_rows)
            saved_path.mkdir(parents=True, exist_ok=True)
            sampled_results.to_csv(saved_file, index=False, escapechar="\\")
            prompts = (sampled_results['instruction'] + " " + sampled_results['input']).tolist()
            gts = sampled_results['output'].tolist()
            prompt_file = os.path.join(saved_path, 'prompts.json')
            gts_file = os.path.join(saved_path, 'gts.json')
            with open(prompt_file, 'w') as f:
                json.dump(prompts, f)
            with open(gts_file, 'w') as f:
                json.dump(gts, f)
        else:
            sampled_results = pd.read_csv(saved_file)

            if all(column in sampled_results.columns for column in ['instruction', 'input', 'output', 'source']):
                print("DataFrame loaded successfully with the desired columns.")
            else:
                print("Some columns are missing from the loaded DataFrame.")
        
        return sampled_results

    def load_dataset(self, extract_validation: False):

        saved_path = Path(os.path.join(DATA_PATH, self.data_set))
        saved_file = os.path.join(saved_path, 'data.json')
        
        if not os.path.exists(saved_file):
            # if not extract_validation:
            #     validation = pd.read_parquet(VALIDATION_PATH)
            inter_duplicated = pd.read_pickle(INTER_DUPLICATED_PATH)
            inter_duplicated.reset_index(names=['sentences'], inplace=True)
            inter_duplicated['source_list'] = inter_duplicated['source'].str.split(',')
            inter_duplicated['instruction'] = ''
            inter_duplicated['input'] = ''
            inter_duplicated['output'] = ''
            
            def exclude_inter_duplicated(inter_duplicated: pd.DataFrame, df_name: str, df: pd.DataFrame):
                # Exclude duplications
                inter_duplicated_indices = inter_duplicated['source_list'].apply(lambda x: df_name in x)
                inter_duplicated_sentences = inter_duplicated.loc[inter_duplicated['source_list'].apply(lambda x: df_name in x), 'sentences']
                print(f'duplicated sentences length: {len(inter_duplicated_sentences)}')
                df['sentences'] = df['instruction'] + df['input'] + df['output']
                if df_name == 'flan2021':
                    df.drop_duplicates(subset='sentences', inplace=True)
                print(f'before exclusion: {df.shape}')
                
                # Add seperated instruction, input, and output to the inter_duplicated
                inter_duplicated = inter_duplicated.merge(df[['instruction', 'input', 'output', 'sentences']], how='left', on='sentences')
                inter_duplicated['instruction_x'] = np.where(inter_duplicated_indices, inter_duplicated['instruction_y'], inter_duplicated['instruction_x'])
                inter_duplicated['input_x'] = np.where(inter_duplicated_indices, inter_duplicated['input_y'], inter_duplicated['input_x'])
                inter_duplicated['output_x'] = np.where(inter_duplicated_indices, inter_duplicated['output_y'], inter_duplicated['output_x'])
                inter_duplicated.drop(columns=['instruction_y', 'input_y', 'output_y'], inplace=True)
                inter_duplicated.rename(columns={'instruction_x': 'instruction', 'input_x': 'input', 'output_x': 'output'}, inplace=True)
                
                df = df.loc[~df['sentences'].isin(inter_duplicated_sentences)]
                print(f'after exclusion: {df.shape}')
                
                df.drop(columns=['sentences'], inplace=True)
                
                return inter_duplicated, df
            
            concat_list = []
            path_list = sorted(list(TOTAL_PATH.iterdir()))
            count = 0
            for p in path_list:
                if p.name.endswith('.parquet.gzip'):
                    print(f'{p} will be processed.')
                    tmp = pd.read_parquet(p)
                    
                    df_name = tmp.loc[0, 'source']
                    if self.data_set == 'all':
                        inter_duplicated, tmp = exclude_inter_duplicated(inter_duplicated=inter_duplicated, df_name=df_name, df=tmp)
                    
                    concat_list.append(tmp.copy())
                elif p.name.endswith('.pickle'):
                    print(f'{p} will be processed.')
                    tmp = pd.read_pickle(p)
                    
                    df_name = tmp.loc[0, 'source']
                    if p.name != 'xp3_dedup.pickle':
                        if self.data_set == 'all':
                            inter_duplicated, tmp = exclude_inter_duplicated(inter_duplicated=inter_duplicated, df_name=df_name, df=tmp)
                    
                    concat_list.append(tmp.copy())
            
            # Finally, add the inter_duplicated
            inter_duplicated = inter_duplicated[['instruction', 'input', 'output', 'source']]
            concat_list.append(inter_duplicated)

            del tmp
            gc.collect()
            
            result = pd.concat(concat_list, axis=0, ignore_index=True)
            sampled_results = result[result['source']==self.data_set]
            saved_path.mkdir(parents=True, exist_ok=True)
            sampled_results.to_csv(saved_file, index=False, escapechar="\\")
            prompts = (sampled_results['instruction'] + " " + sampled_results['input']).tolist()
            gts = sampled_results['output'].tolist()
            prompt_file = os.path.join(saved_path, 'prompts.json')
            gts_file = os.path.join(saved_path, 'gts.json')
            with open(prompt_file, 'w') as f:
                json.dump(prompts, f)
            with open(gts_file, 'w') as f:
                json.dump(gts, f)
        else:
            sampled_results = pd.read_csv(saved_file)

            if all(column in sampled_results.columns for column in ['instruction', 'input', 'output', 'source']):
                print("DataFrame loaded successfully with the desired columns.")
            else:
                print("Some columns are missing from the loaded DataFrame.")
        
        return sampled_results

    def save_sampled(self, sampled: pd.DataFrame, sampling_type: str, n_instances: int, extract_validation: bool = False):
        if extract_validation:
            sampled.to_parquet(f'datasets/validation/validation_{sampling_type}_{n_instances}.parquet.gzip', compression='gzip')
        else:
            path = SAMPLED_PATH.joinpath(f'{self.data_set}/{self.sample_type}/{self.n_instances}/{self.random_state}')
            if not path.exists():
                path.mkdir(parents=True)
                
            sampled.to_parquet(path.joinpath(f'sampled_{sampling_type}_{n_instances}.parquet.gzip'), compression='gzip')
            print(f'Save sampled dataset in {path}.')

    def random_sampling(self, n_instances: int, random_state: int, extract_validation: bool):

        if self.data_set == 'self_instruct':
            total = self.load_si()
        elif self.data_set == 'sni':
            total = self.load_sni()
        else:
            total = self.load_dataset(extract_validation=extract_validation)
        sampled = total.sample(
            n=n_instances,
            random_state=random_state,
        )
        print('len of RS:', len(sampled))
        print(sampled[0:10])
        
        return sampled

    def _sample_k(self, tmp, k):
        tmp = tmp.sample(n = k, random_state=RANDOM_STATE)
        return tmp
    
    def _set_seed(self, random_state):
        deterministic = True
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

    def __call__(self):
        assert self.sample_type in ['random']
        
        if self.sample_type == 'random':
            sampled = self.random_sampling(self.n_instances, random_state=self.random_state, extract_validation=self.extract_validation)
        
        self.save_sampled(sampled, self.sample_type, self.n_instances, self.extract_validation)

def get_args():
    parser = ArgumentParser(description="Sampler for active learning.")
    parser.add_argument('--sample_type', default='random', choices=['random'], help='Type of sampling algorithm to use.')
    parser.add_argument('--n_instances', type=int, default=10000, help='Number of instances to sample.')
    parser.add_argument('--data_set', default='dolly', help='Dataset to use for sampling.')
    parser.add_argument('--random_state', type=int, default=2023, help='Random state for reproducibility.')
    parser.add_argument('--extract_validation', type = bool, default = False, help='Whether to extract validation set.')
    parser.add_argument('--ftype', type=str, help='Type of infoverse single feature (all/perp/rouge/length)')
    
    return parser.parse_args()


if __name__ == '__main__':
    # test()
    args = get_args()
    sampler = Sampler(args)
    sampler()
