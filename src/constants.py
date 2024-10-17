import torch

#################
### Constants ###
#################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
EVAL_SPLIT = "validation"            # Evaluation dataset split
LABEL_CREATED_COLUMN = "text_label"  # The added column for label names in the dataloaders
MAX_EXAMPLES = 50_000                # Maximal number of training examples to use
MAX_EVAL = 2000                      # Maximal number of evaluation examples to use
MAX_LENGTH = 80                      # Default max length for each data example

# Training
BATCH_SIZE = 8                       # Default batch size
EVAL_INTERVAL = 782                  # Eval prompt every how many batches
LR = 8e-4                            # Default learning rate
NUM_EPOCHS = 8                       # Default num of tokens
NUM_TOKENS = 7                       # Prompt length  
