import random

few_shot_demonstrations = dict(
    classes_to_description=[
        "True or false: Is the information in this sentence correct?",
        "summer, winter, autumn or spring: Identify which season is decribed in this text",
        "joy, sadness, anger or fear: Identify the emotion expressed in this text",
        "Shakespeare or Marlowe: who is the author of a this text",
        "spam or not spam: Identify the type of this email",
        "science fiction, romance, thriller: Classify this passage from a book or movie into its genre",
        "left or right: Identify the political leaning of a text or author",
        "bug report, feature request, compliment: Categorize customer feedback into different types",
    ],
    description_and_classes=[
        "Identify the emotion expressed in this text: joy, sadness, anger, fear",
        "Is the information in this sentence correct?: True, False",
        "Classify this passage from a book or movie into its genre: science fiction, romance, thriller",
        "Determine who is the author of a given text: Shakespeare or Marlowe",
        "Identify which season is described in this text: summer, winter, autumn or spring",
        "Categorize customer feedback into different types: bug report, feature request, compliment",
        "Identify the type of this email: spam or not spam",
        "Identify the political leaning of a text or author: left or right"
    ]
)


def create_few_shot_prompt(num_of_tokens, examples, separator='|'):
    """Creates a few-shot prompt using randomly picked examples.

    Args:
        num_of_tokens: The number of tokens to use for the prompt.
        examples: A list of examples.
        separator: The separator character to use between examples.

    Returns:
        A few-shot prompt.
    """
    selected_examples = random.sample(examples, 3)
    separator = " " + separator + " "
    prompt = separator.join(selected_examples) + separator[:-1]  + " x" * num_of_tokens

    return prompt
