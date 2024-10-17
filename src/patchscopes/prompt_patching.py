import torch
from src.constants import DEVICE


def set_hs_patch_hooks(model, hs_patch_config, num_of_tokens):
    def patch_hs(name, position_hs):
        def hook(module, input, output):
            for position_, hs_ in position_hs:
                # (batch, sequence, hidden_state)
                output[0][0, position_ : position_ + num_of_tokens] = hs_
        return hook

    hooks = []
    for l in hs_patch_config:
        if model.config.model_type == 'llama':
            layer = model.model.layers[l]

        else:
            raise ValueError(f"Unknown model: {model.config.model_type}")

        hooks.append(layer.register_forward_hook(
            patch_hs(f"patch_hs_{l}", hs_patch_config[l])
        ))

    return hooks

def set_soft_prompt_patch_hook(model, soft_prompt, source_position, num_of_tokens):
    def patch_sp(name, soft_prompt):
        def hook(module, input, output):
            # (batch, sequence, hidden_state)
            if model.config.model_type == "llama":
                output[0][0, source_position : source_position + num_of_tokens] = soft_prompt
            
            else:
                raise ValueError(f"Unknown model: {model.config.model_type}")
        
        return hook
    
    if model.config.model_type == "llama":
        first_layer = model.model.layers[0]
    
    else:
        raise ValueError(f"Unknown model: {model.config.model_type}")
    
    hooks = [first_layer.register_forward_hook(
        patch_sp("patch_sp_0", soft_prompt)
    )]
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def build_soft_hs_cache(soft_prompt, model, tokenizer, num_of_tokens):
    layers_to_cache = list(range(model.config.num_hidden_layers+1))
    hs_cache = {}
    inp = tokenizer(" x" * num_of_tokens, return_tensors="pt").to(DEVICE)
    pos = inp['input_ids'].shape[1] - num_of_tokens
    patch_hooks = set_soft_prompt_patch_hook(model, soft_prompt, pos, num_of_tokens) 
    with torch.no_grad():
        output = model(**inp, output_hidden_states = True)
    remove_hooks(patch_hooks)

    for layer in layers_to_cache:
        if layer not in hs_cache:
            hs_cache[layer] = []
        hs_cache[layer].append(output["hidden_states"][layer][0][pos:])
    return hs_cache, inp

def generate_greedy_deterministic(hs_patch_config, inp, max_length, end_token, model, tokenizer, num_of_tokens, target_layer):
    input_ids = inp["input_ids"].detach().clone().to(DEVICE)
    with torch.no_grad():
        for _ in range(max_length):
            patch_hooks = set_hs_patch_hooks(model, hs_patch_config, num_of_tokens) 
            outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
            remove_hooks(patch_hooks)
            
            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            if next_token_id.item() == end_token:
                break

    generated_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    patched_pattern = (" x" * num_of_tokens).strip()
    return "".join(generated_text).split(patched_pattern)[-1]


def run_patchscopes_with_params(model, tokenizer, soft_prompt, target_prompt, num_tokens,
                                source_layer, target_layer, end_token=None):
    hs_cache, _ = build_soft_hs_cache(soft_prompt, model, tokenizer, num_tokens)
    target_inp = tokenizer(target_prompt, return_tensors="pt").to(DEVICE)
    target_position = target_inp["input_ids"].shape[1] - num_tokens # -1 for the ":" in the end
    end_token = end_token or tokenizer.encode(',')[0]

    # run model on the same prompt
    target_inp_copy = {}
    for _k, _v in target_inp.items():
        target_inp_copy[_k] = _v.detach().clone().to(DEVICE)

    hs_patch_config = {
        target_layer: [
            (target_position, hs_cache[source_layer+1][0])
        ]
    }

    return generate_greedy_deterministic(hs_patch_config, target_inp_copy, 60, end_token, model, tokenizer, num_tokens, target_layer)
