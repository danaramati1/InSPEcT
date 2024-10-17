all_small_layers = \
    [{"min_source": 2, "max_source": 30, "min_target":2, "max_target": 30}]

llama_best_scored_layers = [
    {"min_source": 2, "max_source": 26, "min_target":2, "max_target": 6},
    {"min_source": 2, "max_source": 30, "min_target":7, "max_target": 11},
    {"min_source": 2, "max_source": 30, "min_target":12, "max_target": 16},
    {"min_source": 7, "max_source": 26, "min_target":17, "max_target": 21},
    {"min_source": 17, "max_source": 26, "min_target":22, "max_target": 26},
    {"min_source": 22, "max_source": 30, "min_target":27, "max_target": 30},
]


def get_layers_combinations_for_model(model_path):
    if model_path in ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct"]:
        return llama_best_scored_layers
    
    return all_small_layers
