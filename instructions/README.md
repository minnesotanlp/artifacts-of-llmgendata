# First-Order Effects 
Our first-order experiments comprised manual annotations of LLM-generated instruction-input-output triples. The annotations can be found on our Huggingface [repo](https://huggingface.co/datasets/minnesotanlp/LLM-Artifacts) in the `instruction` split.

# Second-Order Effects 

## Overview

Enable instruction-tuning in Large Language Models.

## Setup

Install required packages. (`requirements.txt` includes all submodule packages.)

```bash
pip install -r requirements.txt
```
Unzip the datasets zip file. Place the unzipped file inside `instructions/second_order`.

You are all set.

## Datasets

- Cleaned Alpaca Dataset
- Dolly
- FLAN (v1)
- GPT-4-LLM
- Instructions in the Wild
- Unnatural Instructions

### Sampling

```bash
python scripts/sampling.py --sample_type random --n_instances 10000 --data_set dolly --random_state 2021
```

```
--sample_type: Sampling algorithm type - (random)

--data_set: A specific dataset or combined dataset of all to run sampling on - (all, dolly, cleaned_alpaca, self_instruct, sni, etc)`

--n_instances: No. of examples to sample from the dataset (eg: 10000)

--random_state: Random Seed Number for reproducibility (eg: 2021, 2022, 2023)
```

Sampled dataset will be saved under `datasets/sampled/{data_set}/{sampling_type}/{sample_size}/{random_state}`

## Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python llama2/scripts/finetune.py --data_path ./datasets/sampled/dolly/random/10000/2021/sampled_random_10000.parquet.gzip --sample_type random --data_set dolly --n_instances 10000 --random_state 2021 --llama_path /path/to/saved/finetuned/model/weights/ --model_path /path/to/original/model/weights/
```

```
--data_path: Path to the saved sampled data given by:

datasets/sampled/{data_set}/{sampling_type}/{n_instances}/{random_state}/sampled_{sampling_type}_{n_instances}.parquet.gzip

--model_path: Path to where the llama 2 HuggingFace converted weights are stored

--llama_path: Directory path where you want the different llama 2 finetuned model weights to be saved
```

## Inferences

Generate inferences from the fine-tuned model with the test dataset.

```bash
 CUDA_VISIBLE_DEVICES=0 python llama2/scripts/generate_inferences_finetuned.py --sample_type random --data_set dolly --test_set flan2021 --n_instances 10000 --det True --random_state 2021 --llama_path /path/to/saved/finetuned/model/weights/
```

```bash
--test_set: Testing data (flan2021/iw (instructions in the wild))

--det: Deterministic Sampling or not
```

## Evaluation
Compare inferences to the ground-truth by cosine similarities, rouge, and perplexity.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/metrics.py --data_set dolly --sample_type random --n_instances 10000
```
