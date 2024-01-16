## Prerequisites

To run the codes for preference results, run the following script on the terminal: 
  ```sh
  pip install -r requirements.txt
  ```

## First-order effects

In this folder, you will have two jupyter notebook files that contain codes to implement our analysis results for each dataset. 
For this analysis, you will need to load `data/dynasent2_pref_df_will_all_semantic.csv` for p2c analysis and `data/cobbler_sentences_df_roberta_entailment.csv` for CoBBLEr analysis. 

- `first_order/p2c_cleaned.ipynb`: p2c data analysis
- `first_order/cobbler_cleaned.ipynb`: CoBBLEr data analysis


## Second-order effects

In this folder, you will see `train.py`, which finetunes RoBERTa, generates distributions of predicted preferences by the model, and creates figures of the prediction made by human and LLMs. 
For this analysis, you will need to run `train.py` with specific arguments on the terminal. 

For example, if you want to create a training distribution of model finetuned on human preference data of p2c dataset, then the example code looks like: 

  ```python
python train.py --dataset p2c --preferences human
  ```

Here is the explanation of files that are needed to run the `train.py`: 

- `human_cobbler_preferences.csv`: a dataset of human preferences for each sentence pair in CoBBLEr data
- `gpt4_cobbler_preferences.csv`: a dataset of GPT-4 preferences for each sentence pair in CoBBLEr data
- `p2c_subjective_pref_preferences.csv`: a dataset of human preferences for sentence pairs in p2c data
- `p2c_generative_pref_preferences.csv`: a dataset of GPT-3 preferences for sentence pairs in p2c data

### Column Info

For CoBBLEr preference datasets, 
- `output1`, `output2`: the outputs of model 1 and model 2, respectively
- `target`: the corresponding preference label (either by human or LLM)
- `gold_labels`: a list of the names of model 1 and model 2, respectively

For p2c preference datasets, 
- `output1`, `output2`: the first sentence (*sentence 1*) and the second sentence (*sentence 2*), respectively
- `target`: the corresponding preference label (either by human or LLM)
