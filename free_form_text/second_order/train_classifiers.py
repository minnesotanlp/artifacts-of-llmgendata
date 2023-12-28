import pandas as pd
from datasets import Dataset
from datasets import load_from_disk

human_dataset = load_from_disk("human_dataset")
machine_dataset = load_from_disk("machine_dataset")
machine_dataset['train'] = machine_dataset['train'].shuffle(seed=0).select(range(943)) # to match size of human 

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)
    
tokenized_machine = machine_dataset.map(preprocess_function, batched=True)
tokenized_human = human_dataset.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate

f1_metric = evaluate.load("f1")

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    f1_category = f1_metric.compute(predictions=predictions, references=labels, average=None)

    concatenated_results = {
        'f1_weighted': f1_weighted['f1'],
        'f1_category': f1_category['f1'].tolist()  # Convert numpy array to list
    }
    return concatenated_results

classes = ['social_support', 'conflict', 'trust', 'fun', 'similarity_identity', 'respect', 'knowledge',  'power']
id2label = {i: c for i, c in enumerate(classes)}
label2id = {c: i for i, c in enumerate(classes)}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=len(id2label), id2label=id2label, label2id=label2id
)

human_training_args = TrainingArguments(
    output_dir="human_text_model_f1",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    # push_to_hub=True,
)

human_trainer = Trainer(
    model=model,
    args=human_training_args,
    train_dataset=tokenized_human["train"],
    eval_dataset=tokenized_human["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

machine_training_args = TrainingArguments(
    output_dir="machine_text_model_f1",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    # push_to_hub=True,
)

machine_trainer = Trainer(
    model=model,
    args=machine_training_args,
    train_dataset=tokenized_machine["train"],
    eval_dataset=tokenized_machine["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

human_trainer.train()
machine_trainer.train()

all_human_results = human_trainer.evaluate(tokenized_human["test"])

all_machine_results = machine_trainer.evaluate(tokenized_human["test"])

print(f"Human text model (all): {all_human_results}\n")

print(f"Machine text model (all): {all_machine_results}\n")


