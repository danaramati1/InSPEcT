from src.constants import *
from datasets import load_dataset
import os
from peft import PromptTuningInit, PromptTuningConfig, TaskType
from transformers import default_data_collator, get_linear_schedule_with_warmup

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_prompt_tuning_config(model_name, num_tokens=NUM_TOKENS):
    conf = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM, # How to initialize the prompt tuning parameters.
        num_virtual_tokens=num_tokens,
        tokenizer_name_or_path=model_name           # tokenizer to use
    )

    return conf

def preprocess_generate_function(examples, tokenizer, text_column, label_column, task_prompt, one_word_instruction=False):
    instruction = "Please answer with one word only. " if one_word_instruction else ""
    inputs = [f"{instruction}{task_prompt}\n\n{text_column} : {x.strip()}\n\nAnswer: " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]

    # Tokenize the input text and leave the labels as text.
    model_inputs = tokenizer(inputs, padding=True)
    model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
    model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
    model_inputs["labels"] = tokenizer(targets, padding=True)
    model_inputs["labels"] = torch.tensor(model_inputs["labels"]["input_ids"])
    return model_inputs

def preprocess_function(examples, tokenizer, text_column, label_column, dataset, eval_split, max_length=MAX_LENGTH):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x.strip()} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    max_label_length = get_labels_max_length(dataset, tokenizer, label_column, eval_split)

    # Tokenize the input text and labels.
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)

    # pad the labels with the tokenizers pad_token_id.
    # Concatenate the input text and labels into the model_inputs.
    # Create a separate attention mask for labels and model_inputs.
    for i in range(batch_size):
        end_padding_length = max_label_length - len(labels["input_ids"][i])
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        input_suffix = label_input_ids
        model_inputs["input_ids"][i] = sample_input_ids + input_suffix + \
            [tokenizer.pad_token_id] * end_padding_length
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids + \
            [-100] * end_padding_length
        model_inputs["attention_mask"][i] = [1] * (len(model_inputs["input_ids"][i]) - end_padding_length) + \
            [0] * end_padding_length

    # pad the input ids, labels and attention_mask to the max_length 
    # and convert them to PyTorch tensors
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * \
            (max_length - len(sample_input_ids)) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + \
            model_inputs["attention_mask"][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids

        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_dataset_with_name(dataset_name, create_text_labels, eval_split, max_training_examples=MAX_EXAMPLES):
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    label_column = "label"

    if create_text_labels:
        #replace the Label value with the corresponding label text and store them in a text_label column
        if dataset_name == "CogComp/trec":
            classes = ["abbreviation", "entity", "description", "human", "location", "number"]
            label_column = "coarse_label"
        elif dataset_name == "fancyzhx/ag_news":
            classes = [k.replace("_", " ") for k in dataset["train"].features["label"].names]
            classes[3] = "Technology"
        elif dataset_name == "tasksource/subjectivity":
            classes = {"SUBJ":"subjective", "OBJ":"objective"}
            label_column = "Label"
        elif dataset_name == "SetFit/sst5":
            classes = ["terrible", "bad", "neutral", "good", "great"]
        else:
            classes = [k.replace("_", " ") for k in dataset["train"].features["label"].names]
        
        dataset['train'] = dataset['train'].select(range(max_training_examples)) if \
            len(dataset['train']) > max_training_examples else dataset['train']
        
        dataset[eval_split] = dataset[eval_split].shuffle().select(range(MAX_EVAL)) if \
            len(dataset[eval_split]) > MAX_EVAL else dataset[eval_split]

        return dataset.map(
            lambda x: {"text_label": [classes[label] for label in x[label_column]]},
            batched=True,
            num_proc=1,
        )
    
    return dataset


def get_labels_max_length(dataset, tokenizer, label_column, eval_split):
    label_lengths = [len(tokenizer.encode(l)) for l in dataset[eval_split][label_column]]
    return max(label_lengths)


def split_data(dataset, tokenizer, text_column, label_column, eval_split=EVAL_SPLIT, batch_size=BATCH_SIZE):
    train_dataset = dataset["train"].map(
        lambda d: preprocess_function(d, tokenizer, text_column, label_column, dataset, eval_split),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = dataset[eval_split].map(
        lambda d: preprocess_function(d, tokenizer, text_column, label_column, dataset, eval_split),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    # create data loader for train and evaluate
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

    return train_dataloader, eval_dataloader


def dataset_for_generation_eval(dataset, tokenizer, text_column, label_column, task_prompt, 
                                eval_split=EVAL_SPLIT, batch_size=BATCH_SIZE, one_word_instruction=False):
    eval_dataset = dataset[eval_split].map(
        lambda d: preprocess_generate_function(d, tokenizer, text_column, label_column, task_prompt, one_word_instruction=one_word_instruction),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    return eval_dataloader


def train(model, train_dataloader, eval_dataloader, num_tokens, saved_model_name,
          num_epochs=NUM_EPOCHS, lr=LR, batch_size=BATCH_SIZE, 
          eval_batch_interval=None, output_path="."):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    checkpoint_num = 0

    # save initial prompt
    eval_loss, eval_correct = eval(model, eval_dataloader, num_tokens)
    eval_accuracy = eval_correct / (len(eval_dataloader) * batch_size)
    save_checkpoint_params(model, checkpoint_num, eval_accuracy, saved_model_name, output_path)
    checkpoint_num += 1

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if eval_batch_interval is not None and (idx + 1) % eval_batch_interval == 0:
                eval_loss, eval_correct = eval(model, eval_dataloader, num_tokens)
                eval_accuracy = eval_correct / (len(eval_dataloader) * batch_size)
                save_checkpoint_params(model, checkpoint_num, eval_accuracy, saved_model_name, output_path)
                checkpoint_num += 1
                model.train()
        
        if eval_batch_interval is None:
            eval_loss, eval_correct = eval(model, eval_dataloader, num_tokens)
            eval_accuracy = eval_correct / (len(eval_dataloader) * batch_size)
            save_checkpoint_params(model, epoch + 1, eval_accuracy, saved_model_name, output_path)
            
        eval_accuracy = eval_correct / (len(eval_dataloader) * batch_size)
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=} {eval_accuracy=:.4f}")

    return model


def eval(model, eval_dataloader, num_tokens):
    model.eval()
    eval_loss = 0
    eval_correct = 0
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        top_tokens = torch.argmax(outputs.logits, dim=-1)[:,num_tokens-1:-1]
        is_prediction_correct = (
            (batch['labels'] == -100) |
            (batch['labels'] == top_tokens)
        ).all(dim=1)
        eval_correct += sum(is_prediction_correct)
    
    return eval_loss, eval_correct


def eval_text_generation(model, eval_dataloader, tokenizer, max_length=50):
    model.eval()
    eval_correct = 0
    total_examples = 0
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=batch['input_ids'].shape[1] + max_length
            )
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True) 
        labels_texts = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)       
        for generated, label in zip(generated_texts, labels_texts):
            generated = generated.split("\n\nAnswer: ", 1)[1].lower()
            label = label.lower()
            if label in generated:
                eval_correct += 1
        total_examples += len(labels_texts)
    return eval_correct / total_examples


def get_saved_model_name(model_name, dataset_name, num_epochs, lr=LR, num_tokens=NUM_TOKENS, index=None):
    model_name = model_name.split('/')[1]
    dataset_name = dataset_name.split('/')[1]
    prompt_index = f'_{index}' if index is not None else ''
    return f"{model_name}_{dataset_name}_lr{lr}_{num_epochs}_epochs_pt_n{num_tokens}{prompt_index}".replace("/", "_")


def save_checkpoint_params(model, epoch, accuracy, saved_model_name, output_path="."):
    checkpoint_dir = f'{output_path}/{saved_model_name}'
    if not os.path.exists(checkpoint_dir):
        try:
            os.mkdir(checkpoint_dir)
        except OSError as e:
            import sys
            sys.stderr.write(f"could not save checkpoints to {checkpoint_dir}, with error: {e}")
            return

    checkpoint_path = f'{checkpoint_dir}/epoch_{epoch:04d}_acc_{accuracy}.pt'
    trainable_params = [p for p in model.parameters() if p.requires_grad][0]
    torch.save(trainable_params, checkpoint_path)
