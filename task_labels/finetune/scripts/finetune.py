#get_ipython().system('pip install pytesseract datasets rouge-score nltk tensorboard py7zr --upgrade')
#get_ipython().system('sudo apt-get install git-lfs --yes')
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
import random
from random import randrange        
import json
from pandas import read_csv
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import numpy as np
from huggingface_hub import HfFolder
from transformers import (
        pipeline,
        Seq2SeqTrainer, 
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        AutoTokenizer, 
        AutoModelForSeq2SeqLM)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
accelerator = Accelerator()
#DATA [['model_name', 'dataset_name', 'text_ind', 'text', 'prompt', 'params', 'human_agg', 'model_annots'],
# intramodel [['dataset_name', 'text_ind', 'text', 'prompt', 'params', 'human_annots', 'model_annots']
# intermodel************ DATA [['model_name', 'dataset_name', 'text_ind', 'text', 'prompt', 'human_annots', 'model_annots'] 
SEED = 42
random.seed(SEED)
MODEL_ID = "google/flan-t5-small"
import utils
# code for loading csv into huggingface dataset

data_df = read_csv('../data/data.csv')
data_df = data_df.fillna(-1)

# create barplot of integers inside human_annots column based on frequency
data_df['human_agg'].value_counts().plot(kind='bar')

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

def main(filename):
    dataset = utils.get_data('../data/intramodel_data.csv')


tokenized_intra_dataset = utils.get_tokenized_data('../data/intramodel_data.csv', tokenizer, 'model_annots', remove_columns=['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'])
tokenized_inter_dataset = utils.get_tokenized_data('../data/intermodel_data.csv', tokenizer, 'model_annots', remove_columns=['model_name', 'dataset_name', 'text_ind', 'prompt', 'human_annots'])

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(accelerator.device)

# Metric
metric = evaluate.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    ######## NOT SURE IF WE NEED THIS ########
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    #######################BEFORE AND AFTER ARE THE SAME #########################
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    print("DECODED PREDS", decoded_preds)
    print("DECODED LABELS", decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    print("RESULTS", result)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    print("PREDS", prediction_lens)
    result["gen_len"] = np.mean(prediction_lens)
    print("GEN LEN", result["gen_len"])
    raise Exception("STOP")
    return result


# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


# Hugging Face repository id
repository_id = f"{MODEL_ID.split('/')[1]}-intra_model"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=True,
    fp16_full_eval=True,
    dataloader_num_workers=accelerator.num_processes,
    learning_rate=5e-5,
    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=False,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_intra_dataset["train"],
    eval_dataset=tokenized_intra_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()


trainer.evaluate()

model.save_pretrained("output_dir") 
# Save our tokenizer and create model card
tokenizer.save_pretrained(repository_id)
trainer.create_model_card()
# Push the results to the hub
trainer.push_to_hub()


# load model and tokenizer from huggingface hub with pipeline
#summarizer = pipeline("summarization", model="philschmid/flan-t5-base-samsum", device=0)

# select a random test sample
#sample = dataset['test'][randrange(len(dataset["test"]))]
#print(f"dialogue: \n{sample['dialogue']}\n---------------")

# summarize dialogue
#res = summarizer(sample["dialogue"])

#print(f"flan-t5-base summary:\n{res[0]['summary_text']}")


main('../data/intramodel_data.csv')
main('../data/intermodel_data.csv')
