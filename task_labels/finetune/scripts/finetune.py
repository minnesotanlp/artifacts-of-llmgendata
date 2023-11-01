import evaluate
import os
os.environ["WANDB_PROJECT"] = "artifacts"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import random
from random import randrange        
import json
from pandas import read_csv
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import numpy as np
from huggingface_hub import HfFolder
import transformers
from transformers import (
    pipeline,
    AdamW,
    Adafactor,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    EarlyStoppingCallback,
    GenerationConfig
    )
import nltk
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from transformers import T5Tokenizer, T5ForConditionalGeneration
accelerator = Accelerator()
import math
import scipy
SEED = 42
random.seed(SEED)
import utils
#DATA [['model_name', 'dataset_name', 'text_ind', 'text', 'prompt', 'params', 'human_agg', 'model_annots'],
# intramodel [['dataset_name', 'text_ind', 'text', 'prompt', 'params', 'human_annots', 'model_annots']
# intermodel************ DATA [['model_name', 'dataset_name', 'text_ind', 'text', 'prompt', 'human_annots', 'model_annots']
from custom_trainer import CustomTrainer, CustomSeq2SeqTrainer
# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
BATCH_SIZE = utils.get_batch_size()
LR = 1e-4
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
global_num_labels = 0
global_num_annots = 0

def calc_num_outcomes(num_labels):
    # all possible combinations (ignore order) of num_labels)
    return math.factorial(5)/(math.factorial(5-num_labels)*math.factorial(num_labels))


class CustomT5Model(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global global_num_labels, global_num_annots
        config = args[0]
        # ahh so hacky
        self.tokenizer = T5Tokenizer.from_pretrained(config._name_or_path)
        self.lm_head = torch.nn.Linear(config.d_model, global_num_labels, bias=False)
        self.allowed_token_ids = torch.tensor([209, 204, 220, 314, 305]).cuda()
    
    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        ##outputs = super().forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, transformers.modeling_outputs.BaseModelOutput):
            encoder_outputs = transformers.modeling_outputs.BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        past = lm_logits
        lm_logits = torch.moveaxis(lm_logits, 2, 1)
        #preds = np.where(preds != -100, preds, model.tokenizer.pad_token_id)
        labels[labels == -100] = self.tokenizer.pad_token_id 
        #labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        #print("REPLACED LABELS", labels)
        self.allowed_token_ids = torch.tensor([209, 204, 220, 314, 305]).cuda()
        #print("========BATCH DECODE==============")
        #print(self.tokenizer.batch_decode(labels, return_tensors="pt", skip_special_tokens=True))
        # indices are the same as labels now
        temp = self.tokenizer.batch_decode(labels, return_tensors="pt", skip_special_tokens=True)
        #temp2 = self.tokenizer.batch_decode(lm_logits.argmax(-1), return_tensors="pt", skip_special_tokens=True)
        temp = [t for t in temp]
        # Ignore short labels for now (ones less than num_annots)
        labels = []
        for i in range(len(temp)):
            labels.append([(int(t) if int(t) < global_num_labels else global_num_labels-1) for t in temp[i]])
            if len(labels[-1]) < global_num_annots:
                # pad this array
                #labels[-1] = labels[-1] + [self.tokenizer.pad_token_id]*(global_num_annots-len(labels[-1]))
                labels[-1] = labels[-1] + [global_num_labels-1]*(global_num_annots-len(labels[-1]))
        labels = torch.tensor(labels).cuda()
        
        loss = None
        loss_fct = CrossEntropyLoss()#ignore_index=-100)
        #loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        debugging = False
        if not debugging:
            loss = loss_fct(lm_logits, labels)
        else:
            try:
                loss = loss_fct(lm_logits, labels)
            except Exception as e:
                loss = 1#loss_fct(torch.zeros_like(), labels)
                print(e)
                print("type", type(self.decoder))
                print("decoder_input_ids", decoder_input_ids)
                print(self.generation_config)
                print(self.lm_head)
                print("LM_LOGITS", lm_logits)
                print("LM_LOGITS SHAPE", lm_logits.shape)
                print("PAST SHAPE", past.shape)
                print("LABELS", labels)
                print("TYPE LABELS", type(labels))
                print("SHAPES", lm_logits.shape, labels.shape)
                print("DECODER OUTPUTS", decoder_outputs['last_hidden_state'])
                print("DECODER OUTPUTS", decoder_outputs['last_hidden_state'].shape)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return transformers.modeling_outputs.Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class Model:
    def __init__(self, model_name, num_labels=0, num_annots=0):
        self.model_name = model_name
        if "roberta" in model_name:
            from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=calc_num_outcomes(num_labels))
        elif "t5" in model_name:
            from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
            from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
            import transformers
            print(transformers.__file__)
            my_config = {}
            print("NUM ANNOTS", global_num_annots)
            my_config['max_new_tokens'] = global_num_annots
            my_config['min_new_tokens'] = global_num_annots
            #my_config['max_length'] = 300
            my_config['renormalize_logits'] = True
            #my_config['return_dict_in_generate'] = True
            #my_config['num_beams'] = 1
            #my_config['do_sample'] = False
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
                #task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            #self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model = CustomT5Model.from_pretrained(
                pretrained_model_name_or_path=model_name,
                ignore_mismatched_sizes=True
            )
            #self.model.generation_config = GenerationConfig.from_dict(my_config)
            #self.model = T5ForConditionalGeneration(MyT5DecoderModule(base_model))
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            self.model.to(accelerator.device)
        #elif "t5" in model_name:
        #    from transformers import T5ForConditionalGeneration, AutoTokenizer
        #    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        #    self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            raise ValueError("Model model_name not supported")
        self.num_labels = num_labels

    def __repr__(self):
        return self.model_name

    def set_tokenized_dataset(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def set_training_var(self, repository_id, compute_metrics):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
        #optimizer = Adafactor(self.model.parameters(), relative_step=False, warmup_init=False, lr=LR)
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=10000)

        is_roberta = "roberta" in self.model_name
        if is_roberta:
            from transformers import Trainer, TrainingArguments
            trainer = CustomTrainer
            training_args = TrainingArguments
        elif "t5" in self.model_name:
            from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
            trainer = Seq2SeqTrainer
            training_args = Seq2SeqTrainingArguments
            #trainer = CustomSeq2SeqTrainer
            #training_args = Seq2SeqTrainingArguments
        print(accelerator.num_processes)
        #generation_config = GenerationConfig.from_pretrained(self.model_name)
        #generation_config.max_new_tokens = num_annots
        #generation_config.min_new_tokens = num_annots
        # Define training args
        self.training_args = training_args(
            output_dir=repository_id,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            predict_with_generate=True,
            generation_config=self.model.generation_config,
            fp16=True,############3
            fp16_full_eval=True,#########
            dataloader_num_workers=accelerator.num_processes,
            learning_rate=LR,
            num_train_epochs=200,
            logging_dir=f"{repository_id}/logs",
            logging_strategy="steps",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="train_loss",
            greater_is_better=False,
            report_to="wandb",
            push_to_hub=False,
            include_inputs_for_metrics=True,
            hub_strategy="every_save",
            hub_model_id=repository_id,
            hub_token=HfFolder.get_token(),
            #generation_config=generation_config,
        )
        # Create Trainer instance
        if not is_roberta:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                #label_pad_token_id=label_pad_token_id,
                #pad_to_multiple_of=8
            )
        else:
            from transformers import DataCollatorWithPadding
            #data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)#, pad_to_multiple_of=8)
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                #label_pad_token_id=-100,
                #pad_to_multiple_of=8
            )
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
        '''
        "return_dict_in_generate": True,
            "num_beams": 1, # MESS WITH THIS LATERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
            "prefix_allowed_tokens_fn": restrict_decode_vocab,
            #"constraints": [
            #    DisjunctiveConstraint([[209, 204, 220, 314, 305]]),
            #],
            "max_new_tokens": 5,
            "min_new_tokens": 5,
            "output_scores": True,
            #"temperature": 1.0,
            "do_sample": False,
        '''
        self.trainer = trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            optimizers=(optimizer, scheduler),
            data_collator=data_collator,
            callbacks=[early_stopping],
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["val"],
            compute_metrics=compute_metrics,
        )
        
def restrict_decode_vocab(a, b):
    return [209, 204, 220, 314, 305]

def max_edit_distance(target, output):
    # Step 1: Find the maximum length difference
    length_difference = abs(len(target) - len(output))

    # Step 2: Character Replacement
    char_difference = 0
    for t, o in zip(target, output):
        if t != o:
            char_difference += 1

    # Step 3: Calculate Maximum Edit Distance
    max_distance = length_difference + char_difference
    return max_distance

def main(filename, model_id, dataset_name, remove_columns, col_for_num_labels=[], dataset_mode='sorted'):
    global global_num_labels, global_num_annots
    #model = Model("google/t5-v1_1-base")
    #model = Model("google/flan-t5-small")
    global_num_labels = utils.get_num_labels(dataset_name)
    global_num_annots = utils.get_num_annots(dataset_name)
    model = Model(model_id, num_labels=global_num_labels, num_annots=global_num_annots)   
    tokenized_dataset = utils.get_tokenized_data(filename, dataset_name, model.tokenizer, col_for_num_labels, remove_columns=remove_columns, mode=dataset_mode)
    if 'intra' in filename: 
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-intra_model"
    else:
        repository_id = f"{dataset_name}-{model_id.replace('/','-')}-inter_model"

    def compute_metrics(eval_preds):
        if type(eval_preds) == transformers.trainer_utils.EvalPrediction:
            preds = eval_preds.predictions
            labels = eval_preds.label_ids
        elif type(preds) == np.ndarray:
            preds = torch.from_numpy(preds) 
            labels = torch.from_numpy(labels)
        else:
            preds, labels = eval_preds
        # Replace -100 in the labels as we can't decode them.
        preds = np.where(preds != -100, preds, model.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, model.tokenizer.pad_token_id)
        decoded_preds = model.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = model.tokenizer.batch_decode(labels, skip_special_tokens=True)
        losses = []
        assert len(decoded_preds) == len(decoded_labels)
        for i in range(len(decoded_preds)):
            s1 = decoded_preds[i]
            s2 = decoded_labels[i]
            max_distance = max_edit_distance(s1, s2)
            if len(s2.split()) < num_labels:
                continue
            #if s1 contains characters other than digits and spaces, use edit distance
            elif s1 == '':
                losses.append(10.0)
                continue
            elif not s1.isdigit():
                losses.append(nltk.edit_distance(s1, s2)/max_distance)
                print(losses[-1])
                raise Exception('STOP')
                continue
            # Calculate length penalty
            length_penalty = abs(len(s1) - len(s2))
            s1_digits = [int(s) for s in s1 if s.isdigit()]
            s2_digits = [int(s) for s in s2 if s.isdigit()]
            # character penalty: check number of numbers
            if s1_digits == s2_digits:
                character_penalty = 0
            else:
                character_penalty = abs(len(s1_digits) - len(s2_digits))

            # Calculate ordinal number penalty 
            ordinal_penalty = abs(sum(s1_digits) - sum(s2_digits)) 

            # overlap penalty
            overlap_penalty = 0
            for elem in s1_digits + s2_digits:
                if elem in s1_digits and elem in s2_digits:
                    continue
                else:
                    overlap_penalty += 1

            #print("length_penalty", length_penalty)
            #print("character_penalty", character_penalty)
            #print("ordinal_penalty", ordinal_penalty)

            # Combine the penalties with their respective weights
            length_penalty_weight = 0.2
            character_penalty_weight = 0.2
            ordinal_penalty_weight = 0.2
            overlap_penalty_weight = 0.2
            total_loss = (
                length_penalty_weight * length_penalty +
                character_penalty_weight * character_penalty +
                ordinal_penalty_weight * ordinal_penalty +
                overlap_penalty_weight * overlap_penalty
            )
            #losses.append(total_loss)
            losses.append(min(total_loss, 1))
            '''
            if len(s2.split()) < num_labels:
                continue
            #if s1 contains characters other than digits and spaces, use edit distance
            elif s1 == '':
                losses.append(1.0)
                continue
            elif not s1.replace(" ", "").isdigit():
                losses.append((len(s2)+nltk.edit_distance(s1, s2))/max(losses))
                continue

            #if s1 contains only digits and spaces, use wasserstein distance 
            s1_digits = [int(s) for s in s1.split() if s.isdigit()]
            s2_digits = [int(s) for s in s2.split() if s.isdigit()]
            dist = scipy.stats.wasserstein_distance(s1_digits, s2_digits)
            #diff = sum(abs(s) for s in s1_digits) - sum(abs(s) for s in s2_digits)
            #overlap = set(s1_digits).intersection(set(s2_digits))
            #losses.append(-nltk.edit_distance(s1, s2)/10 - diff + len(overlap))
            losses.append((dist+nltk.edit_distance(s1, s2))/max(losses))
            '''

        # Some simple post-processing
        #######################BEFORE AND AFTER ARE THE SAME #########################
        print("losses", losses, "mean", np.mean(losses))
        return {'losses': losses, 'train_loss': np.mean(losses), 'eval_train_loss': np.mean(losses)}
        #model.compute_metrics(eval_preds, label_pad_token_id=-100)

    model.set_tokenized_dataset(tokenized_dataset)
    model.set_training_var(repository_id, compute_metrics)
    model.trainer.train()
    model.trainer.evaluate(
        eval_dataset=tokenized_dataset["test"],
    )
    model.model.save_pretrained(repository_id) 
    model.tokenizer.save_pretrained(repository_id)
    model.trainer.create_model_card()
    model.trainer.push_to_hub()


col_for_num_labels = "human_annots"
'''
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-small",#"roberta-base",
     dataset_name = "SChem5Labels",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'sorted')
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-base",#"roberta-base",
     dataset_name = "SChem5Labels",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'sorted')
'''
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-large",#"roberta-base",
     dataset_name = "SChem5Labels",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'sorted')
'''
main(filename = '../data/intramodel_data.csv', 
     model_id = "google/t5-v1_1-small",#"roberta-base",
     dataset_name = "Sentiment",
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'human_annots'],
     col_for_num_labels = "model_annots",
     dataset_mode = 'sorted')
main(filename = '../data/intermodel_data.csv', 
     remove_columns = ['model_name', 'dataset_name', 'text_ind', 'prompt', 'human_annots'],
     repository_id = f"{MODEL_ID.split('/')[1]}-inter_model",
     col_for_num_labels = "model_annots")
main(filename = '../data/intramodel_data.csv', 
     remove_columns = ['dataset_name', 'text_ind', 'prompt', 'params', 'model_annots'],
     repository_id = f"{MODEL_ID.split('/')[1]}-intra_human",
     col_for_num_labels = "human_annots")
main(filename = '../data/intermodel_data.csv', 
     remove_columns = ['model_name', 'dataset_name', 'text_ind', 'prompt', 'model_annots'],
     repository_id = f"{MODEL_ID.split('/')[1]}-inter_human",
     col_for_num_labels = "human_annots")
'''
