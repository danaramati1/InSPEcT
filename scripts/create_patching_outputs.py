import argparse
import os
from src.patchscopes.layers import get_layers_combinations_for_model
from src.patchscopes.patch_layers import run_patching_and_save
from src.patchscopes.target_prompt import few_shot_demonstrations, create_few_shot_prompt
from src.utils import get_model, get_tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='The model name', required=True, nargs='?')
    parser.add_argument('-n', '--num_tokens', type=int, help='number of patched tokens', required=True, nargs='?')
    parser.add_argument('-d', '--dataset', type=str, help='the dataset name', required=True, nargs='?')
    parser.add_argument('-c', '--checkpoints_path', type=str, help='directory with soft prompts to run patching on', required=True, nargs='?')
    parser.add_argument('-tp', '--target_prompt', type=str, help='the target prompt', required=False, nargs='?')
    parser.add_argument('-tn', '--target_prompt_name', type=str, help='the target prompt name for results output', required=False, nargs='?')
    parser.add_argument('-t', '--target_prompt_type', type=str, help='the target prompt type to sample', required=True, nargs='?')
    parser.add_argument('-min', '--min_epoch', type=int, help='the epoch to start patching from', required=False, nargs='?')
    parser.add_argument('-max', '--max_epoch', type=int, help='the maximal epoch to patch', required=False, nargs='?')
    parser.add_argument('-j', '--jumps', type=int, help='the jump between epochs patchings', required=False, nargs='?')
    parser.add_argument('-i', '--index', type=int, help='sampled prompt index (an integer)', required=True, nargs='?')
    parser.add_argument('-f', '--first_epochs', type=int, help='how many first epochs to evaluate on without jumping', required=False, nargs='?')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory for results', required=False, nargs='?')

    args = parser.parse_args()
    model_path = args.model
    num_tokens = args.num_tokens
    target_prompt_type = args.target_prompt_type
    checkpoints_dir = args.checkpoints_path
    task_dataset = args.dataset
    min_epoch = args.min_epoch or 0
    max_epoch = args.max_epoch or len(os.listdir(checkpoints_dir)) - 1
    jumps = args.jumps or 1
    target_prompt_index = args.index or 3
    first_epochs = args.first_epochs or 20
    output_dir = args.output_dir

    examples = few_shot_demonstrations.get(target_prompt_type)
    if args.target_prompt is not None:
        target_name = args.target_prompt_name or 'custom'
        target_prompts = {f"target_{target_name}_{target_prompt_index}": args.target_prompt}
    else:
        target_name = target_prompt_type
        target_prompts = {f"target_{target_name}_{target_prompt_index}": create_few_shot_prompt(num_tokens, examples) for i in range(1)}

    print(target_prompts[f'target_{target_name}_{target_prompt_index}'])

    tokenizer = get_tokenizer(model_path)
    model = get_model(model_path, to_device=True)
    layers_combinations = get_layers_combinations_for_model(model_path)

    for f in os.listdir(checkpoints_dir):
        epoch = int(f.split('_')[1])
        if epoch < min_epoch or epoch > max_epoch or ((epoch - min_epoch) % jumps != 0 and epoch > first_epochs):
            continue

        soft_prompt_path = f'{checkpoints_dir}/{f}'
        for name, target_prompt in target_prompts.items():
            run_patching_and_save(
                model_path, 
                name, 
                soft_prompt_path, 
                num_tokens, 
                task_dataset, 
                model, 
                tokenizer, 
                target_prompt,
                layer_combinations=layers_combinations,
                output_dir=output_dir,
            )
