import csv
import os
from layers import all_small_layers
from prompt_patching import run_patchscopes_with_params
import torch
from src.utils import get_model, get_tokenizer


def run_patchscopes_for_layers_combinations(model, tokenizer, 
                                            soft_prompt, target_prompt, num_tokens,
                                            combinations=None, end_token=None, print_out=False):
    results = []
    combinations = combinations or all_small_layers
    for comb in combinations:
        for source_layer in range(comb.get("min_source"), comb.get("max_source") + 1):
            for target_layer in range(comb.get("min_target"), comb.get("max_target") + 1):
                patched_output = run_patchscopes_with_params(
                    model, 
                    tokenizer, 
                    soft_prompt, 
                    target_prompt, 
                    num_tokens, 
                    source_layer, 
                    target_layer,
                    end_token
                )

                if print_out:
                    print(f"results for {source_layer=} {target_layer=}:")
                    print(f"{patched_output=}")
                    print()

                results.append({
                    "source_layer": source_layer,
                    "target_layer": target_layer,
                    "output": patched_output
                })

    return results


def get_output_file_path(model_path, task_dataset, num_tokens, soft_prompt_path, 
                         target_prompt_name, output_dir=None):
    model = model_path.split('/')[-1]
    soft_prompt = soft_prompt_path.split('/')[-1][:-3]
    output_dir = output_dir or "./patching_output"
    output_path = f'{output_dir}/{model}/{task_dataset}/n{num_tokens}_{target_prompt_name}'
    os.makedirs(output_path, exist_ok=True)
    return f'{output_path}/{soft_prompt}.csv'


def write_results_to_csv(results, output_path):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["source_layer", "target_layer", "output"], escapechar='\\')
        writer.writeheader()
        writer.writerows(results)


def run_patching_and_save(model_path, target_prompt_name, soft_prompt_path, num_tokens, task_dataset, 
                          model=None, tokenizer=None, target_prompt=None, layer_combinations=None, output_dir=None):
    if tokenizer is None:
        tokenizer = get_tokenizer(model_path)

    if model is None:
        model = get_model(model_path, to_device=True)

    soft_prompt = torch.load(soft_prompt_path)
    end_token = tokenizer.encode('$')[0]
    
    output_path = get_output_file_path(
        model_path, task_dataset, num_tokens, 
        soft_prompt_path, target_prompt_name, output_dir)
    
    if os.path.exists(output_path):
        return

    results = run_patchscopes_for_layers_combinations(
        model, 
        tokenizer, 
        soft_prompt, 
        target_prompt, 
        num_tokens,
        end_token=end_token,
        combinations=layer_combinations
    )
    
    write_results_to_csv(results, output_path)
    