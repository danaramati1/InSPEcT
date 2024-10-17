import argparse
from src.constants import *
from src.prompt_tuning.prompt_tuning import get_prompt_tuning_config, \
    load_dataset_with_name, train, split_data, get_saved_model_name, \
    get_labels_max_length
from src.utils import get_peft_pt_model, get_tokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator


def preprocess_function(examples, tokenizer, label_column, dataset, eval_split="test", max_length=MAX_LENGTH):
    batch_size = len(examples['premise'])
    inputs = [f"premise : {p}, hypothesis : {h}, Label : " for p, h in zip(examples['premise'], examples['hypothesis'])]
    targets = [str(x) for x in examples[label_column]]
    max_label_length = get_labels_max_length(dataset, tokenizer, label_column, eval_split)

    # Tokenize the input text and labels.
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)

    # For each example in a batch, pad the labels with the tokenizers pad_token_id.
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

    # Loop through each example in the batch again to pad the input ids, labels,
    # and attention mask to the max_length and convert them to PyTorch tensors
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



def split_data(dataset, tokenizer, label_column, eval_split=EVAL_SPLIT, batch_size=BATCH_SIZE):
    train_dataset = dataset["train"].map(
        lambda d: preprocess_function(d, tokenizer, label_column, dataset),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = dataset[eval_split].map(
        lambda d: preprocess_function(d, tokenizer, label_column, dataset),
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


def train_snli_prompt(model_name, dataset_name, label_column, num_epochs, 
                      batch_size, num_tokens, lr, eval_split, create_text_labels, 
                      eval_every, max_training_examples, output_path, prompt_index):
    
    # load tokenizer
    tokenizer = get_tokenizer(model_name)

    # load dataset
    dataset = load_dataset_with_name(dataset_name, create_text_labels, eval_split, max_training_examples)
    train_dataloader, eval_dataloader = split_data(dataset, tokenizer, label_column, eval_split)
    
    # load peft model
    pt_config = get_prompt_tuning_config(model_name, num_tokens=num_tokens)
    model = get_peft_pt_model(model_name, pt_config)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    saved_model_name = get_saved_model_name(model_name, dataset_name, num_epochs, lr, num_tokens, prompt_index)

    if not "70B" in model_name:
        model.to(DEVICE)

    keywargs = dict()
    if eval_every is not None:
        keywargs['eval_batch_interval'] = eval_every
    if output_path is not None:
        keywargs['output_path'] = output_path
        
    # train
    model = train(model, train_dataloader, eval_dataloader, num_tokens,
                  saved_model_name, num_epochs, lr, batch_size, **keywargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='The model hf path', required=True, nargs='?')
    parser.add_argument('-d', '--dataset', type=str, help='The dataset hf path', required=True, nargs='?')
    parser.add_argument('-l', '--label_column', type=str, help='the label column in the dataset. required if create_text_labels set to False', required=False, nargs='?')
    parser.add_argument('-e', '--num_epochs', type=int, help='num of epochs', nargs='?')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', nargs='?')
    parser.add_argument('-n', '--num_tokens', type=int, help='num tokens', nargs='?')
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate', nargs='?')
    parser.add_argument('-es', '--eval_split', type=str, help='evaluation split name', nargs='?')
    parser.add_argument('-tl', '--text_labels', default=False, help='create text labels for dataset', action='store_true')
    parser.add_argument('-ee', '--eval_every', type=int, help='eval and checkpoint batches interval', nargs='?')
    parser.add_argument('-mt', '--max_trainig_examples', type=int, help='maximal training examples to use per epoch', nargs='?')
    parser.add_argument('-o', '--output_path', type=str, help='output directory path for checkpoints', nargs='?')
    parser.add_argument('-pi', '--prompt_index', type=int, help='add an index number to the prompt output directory', nargs='?')

    args = parser.parse_args()

    if not args.label_column and not args.create_text_labels:
        parser.error('--label_column is required when --create_text_labels is not set.')

    model_name = args.model
    dataset_name = args.dataset
    label_column = args.label_column if not args.text_labels else LABEL_CREATED_COLUMN
    num_epochs = args.num_epochs or NUM_EPOCHS
    batch_size = args.batch_size or BATCH_SIZE
    num_tokens = args.num_tokens or NUM_TOKENS
    lr = args.learning_rate or LR
    eval_split = args.eval_split or EVAL_SPLIT
    create_text_labels = args.text_labels
    eval_every = args.eval_every
    max_training_examples = args.max_trainig_examples
    output_path = args.output_path
    prompt_index = args.prompt_index

    train_snli_prompt(model_name, dataset_name, label_column, num_epochs, 
                      batch_size, num_tokens, lr, eval_split, create_text_labels, 
                      eval_every, max_training_examples, output_path, prompt_index)
