import argparse
from src.constants import *
from src.prompt_tuning.prompt_tuning import get_prompt_tuning_config, \
    load_dataset_with_name, train, split_data, get_saved_model_name
from src.utils import get_peft_pt_model, get_tokenizer


def train_prompt(model_name, dataset_name, text_column, label_column, num_epochs, 
                 batch_size, num_tokens, lr, eval_split, create_text_labels, 
                 eval_every, max_training_examples, output_path):
    
    # load tokenizer
    tokenizer = get_tokenizer(model_name)

    # load dataset
    dataset = load_dataset_with_name(dataset_name, create_text_labels, eval_split, max_training_examples)
    train_dataloader, eval_dataloader = split_data(dataset, tokenizer, text_column, label_column, eval_split)
    
    # load peft model
    pt_config = get_prompt_tuning_config(model_name, num_tokens=num_tokens)
    model = get_peft_pt_model(model_name, pt_config)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    saved_model_name = get_saved_model_name(model_name, dataset_name, num_epochs, lr, num_tokens)

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
    parser.add_argument('-t', '--text_column', type=str, help='the text column in the dataset', required=True, nargs='?')
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

    args = parser.parse_args()

    if not args.label_column and not args.create_text_labels:
        parser.error('--label_column is required when --create_text_labels is not set.')

    model_name = args.model
    dataset_name = args.dataset
    text_column = args.text_column
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

    train_prompt(model_name, dataset_name, text_column, label_column, num_epochs, 
                 batch_size, num_tokens, lr, eval_split, create_text_labels, 
                 eval_every, max_training_examples, output_path)
