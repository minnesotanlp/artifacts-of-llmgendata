from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.nn import Softmax, Sigmoid
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Choose between CoBBLEr or P2C or MT-bench"
    )
    parser.add_argument(
        "--preferences",
        required=True,
        type=str,
        help="Choose between machine or human"
    )
    parser.add_argument("--num_epochs", default="5", type=int, help="num training epochs")
    parser.add_argument("--model_name", required=True, type=str, help="model name")
    parser.add_argument("--batch_size", default="8", type=int, help="batch size")
    args = parser.parse_args()
    # torch.cuda.set_device(args.local_rank)
    return args

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Define the maximum sequence length
max_length = 256 

# Load the dataset from the CSV file
def load_preference_dataset(csv_file_path):
    dataset = pd.read_csv(csv_file_path)
    return dataset

# Preprocess the dataset and convert it to PyTorch tensors
def preprocess_dataset(csv_file_path, dataset, tokenizer, max_length):
    input_ids = []
    attention_masks = []
    targets = []

    for _, row in dataset.iterrows():
        sentence1 = row['output1']
        sentence2 = row['output2']
        target = row['target']

        # Tokenize and pad the input sentences
        inputs1 = tokenizer(
            sentence1,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        inputs2 = tokenizer(
            sentence2,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids1 = inputs1['input_ids'].squeeze()
        attention_mask1 = inputs1['attention_mask'].squeeze()
        input_ids2 = inputs2['input_ids'].squeeze()
        attention_mask2 = inputs2['attention_mask'].squeeze()

        input_ids.append((input_ids1, input_ids2))
        attention_masks.append((attention_mask1, attention_mask2))
        targets.append(target)

    input_ids = torch.stack([torch.cat((ids1, ids2)) for ids1, ids2 in input_ids])
    attention_masks = torch.stack([torch.cat((mask1, mask2)) for mask1, mask2 in attention_masks])
    targets = torch.tensor(targets) #, dtype=torch.float32)

    return input_ids, attention_masks, targets
    
def prepare_dataset(tokenizer, csv_file_path):        
    dataset = load_preference_dataset(csv_file_path)
    shuffled_dataset = dataset.sample(frac=1, random_state=42)

    input_ids, attention_masks, targets = preprocess_dataset(csv_file_path, shuffled_dataset, tokenizer, max_length)

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(input_ids, attention_masks, targets)

    # Assuming you have the 'targets' tensor containing unique labels, e.g., -1 and 1
    unique_labels = torch.unique(targets)

    num_labels = len(unique_labels)
    print("NUM_LABELS: ", num_labels)
    
    return dataset, num_labels

def get_csv_file(data, data_type):
    if data == "cobbler":
        if data_type == "human":
            return "/home/ryankoo/artifacts-of-llm-preference/pairwise_preferences/human_cobbler_preferences.csv"
        else:
            return "/home/ryankoo/artifacts-of-llm-preference/pairwise_preferences/gpt4_cobbler_preferences.csv"
    elif data == "p2c":
        if data_type == "human":
            return "/home/ryankoo/artifacts-of-llm-preference/pairwise_preferences/p2c_subjective_pref_preferences.csv"
        else:
            return "/home/ryankoo/artifacts-of-llm-preference/pairwise_preferences/p2c_generative_pref_preferences.csv"
    elif data == "mtbench":
        if data_type == "human":
            return "/home/ryankoo/artifacts-of-llm-preference/pairwise_preferences/mtbench_human_model_preferences.csv"
        else:
            return "/home/ryankoo/artifacts-of-llm-preference/pairwise_preferences/mtbench_gpt4_model_preferences.csv"

def main():
    args = parse_args()
    csv_file_path = get_csv_file(args.dataset, args.preferences)
    


    # Load a pre-trained BERT model and tokenizer
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset, num_labels = prepare_dataset(tokenizer, csv_file_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    
    
    if args.preferences == "human":
        val_file_path = get_csv_file(args.dataset, "machine")
    else:
        val_file_path = get_csv_file(args.dataset, "human")
    opp_dataset, _ = prepare_dataset(tokenizer, val_file_path)
    _, opp_val_data = train_test_split(opp_dataset, test_size=0.8, shuffle=True)
    
    # Split the dataset into train and validation sets
    train_data, valid_data = train_test_split(dataset, test_size=0.2, shuffle=True)

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    
    # opposite validation
    opp_valid_dataloader = DataLoader(opp_val_data, batch_size=args.batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss() if num_labels == 1 else nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop
    num_epochs = args.num_epochs
    max_grad_norm = 1.0
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        total_loss = 0
        num_samples = 0
        
        for batch in progress_bar:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            
            if num_labels == 1:
                # Binary classification case
                loss = criterion(logits, labels.view(-1, 1))
            else:
                # Multiclass classification case
                loss = criterion(logits, labels) #type(torch.LongTensor))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * input_ids.size(0)
            num_samples += input_ids.size(0)
            average_loss = total_loss / num_samples
            progress_bar.set_postfix({'Loss': average_loss})

        # Validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            num_samples = 0
            all_predictions = []
            all_targets = []
            for batch in valid_dataloader:
                input_ids = batch[0].to(device)
                attention_masks = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = model(input_ids, attention_mask=attention_masks)
                logits = outputs.logits
                if num_labels == 1:
                    # Binary classification case
                    loss = criterion(logits, labels.view(-1, 1))
                else:
                    # Multiclass classification case
                    loss = criterion(logits, labels)
                total_loss += loss.item() * input_ids.size(0)
                num_samples += input_ids.size(0)
                
                if num_labels == 1:    
                    predictions = torch.where(logits > 0, 1.0, 0.0).view(-1).cpu().numpy()
                else:
                    probabilities = torch.softmax(logits, dim=1)
                    # Get the predicted labels based on the class with the highest probability
                    predictions = torch.argmax(probabilities, dim=1)
                    predictions = predictions.cpu().numpy()
                    
                targets = labels.view(-1).cpu().numpy()  # Convert labels to NumPy array
                all_predictions.extend(predictions)
                all_targets.extend(targets)

            average_loss = total_loss / num_samples
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_loss:.4f}')
            
            from sklearn.metrics import accuracy_score, f1_score
            
            # print(all_predictions)
            accuracy = accuracy_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions) if num_labels == 1 else f1_score(all_targets, all_predictions, average="micro")
            print(f'Validation Accuracy: {accuracy:.4f}')
            print(f'Validation F1 Score: {f1:.4f}')

    model.to(device)  # Move the model to the GPU if available
    model.eval()
    
    print("Evaluating on opposite annotation set now")
    ## validate on opposite annotation set
    with torch.no_grad():
        total_loss = 0
        num_samples = 0
        all_predictions = []
        all_targets = []
        for batch in opp_valid_dataloader:
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            if num_labels == 1:
                # Binary classification case
                loss = criterion(logits, labels.view(-1, 1))
            else:
                # Multiclass classification case
                loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            num_samples += input_ids.size(0)
            
            if num_labels == 1:    
                predictions = torch.where(logits > 0, 1.0, 0.0).view(-1).cpu().numpy()
            else:
                probabilities = torch.softmax(logits, dim=1)
                # Get the predicted labels based on the class with the highest probability
                predictions = torch.argmax(probabilities, dim=1)
                predictions = predictions.cpu().numpy()
                
            targets = labels.view(-1).cpu().numpy()  # Convert labels to NumPy array
            all_predictions.extend(predictions)
            all_targets.extend(targets)

        average_loss = total_loss / num_samples
        print(f'Validation Loss: {average_loss:.4f}')
        
        from sklearn.metrics import accuracy_score, f1_score
        
        # print(all_predictions)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions) if num_labels == 1 else f1_score(all_targets, all_predictions, average="micro")
        print(f'Opposite annotation | Validation Accuracy: {accuracy:.4f}')
        print(f'Opposite annotation | Validation F1 Score: {f1:.4f}')
    
    
    data_dict = {'probs': [], 'logodds': []}

    softmax = Softmax(dim=1)
    probabilities = []
    logodds = []

    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc='Inference'):
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            logodds.extend(logits.cpu().numpy())
            batch_probabilities = softmax(logits).cpu().numpy()
            probabilities.extend(batch_probabilities)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    class_index = 0
    logodds = np.array(logodds)
    # print(logodds)
    # print(logodds.shape)
    for class_index in range(len(logodds[0])):
        print("Plotting class index: ", class_index)
        class_probabilities = [p[class_index] for p in probabilities]
        class_logodds = logodds[:, class_index]
        data_dict['probs'].append(class_probabilities)
        data_dict['logodds'].append(class_logodds)
        plt.scatter(class_probabilities, class_logodds, label=f'Class {class_index}')
    print(len(data_dict['probs']))

    with open(f'/home/ryankoo/artifacts-of-llm-preference/results/{args.model_name}_{args.dataset}_{args.preferences}.pkl', 'wb') as pkl_file:
        pickle.dump(data_dict, pkl_file)
        
    plt.xlabel('Class Probabilities')
    plt.ylabel('Logits')
    plt.title('Logits vs. Class Probabilities')
    plt.legend()
    plt.show()
    plt.savefig(f"{args.model_name}_{args.dataset}_{args.preferences}")
    plt.savefig(f"{args.model_name}_{args.dataset}_{args.preferences}.pdf")

if __name__ == '__main__':
    main()