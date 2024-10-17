import string
import textstat
import re
import nltk
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
from src.task_classes import task_classes
from src.task_references import task_references

nltk.download('stopwords')


class PatchingScore:
    @classmethod
    def calculate_classes_occurence(cls, task_dataset, patched_output):
        classes = set(task_classes.get(task_dataset))
        words =  set(cls.__get_sentence_words(patched_output))
        classes_count = sum(1 for c in classes if c in words)
        return (classes_count / len(classes)) if classes else 0
    
    @classmethod
    def calculate_rouge_score(cls, task_dataset, patched_output,
                              remove_stopwords=False, remove_duplicates=False):
        references = task_references.get(task_dataset)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = [scorer.score(cls.__preprocess_text(patched_output,
                                                     remove_stopwords=remove_stopwords,
                                                     remove_duplicates=remove_duplicates),
                               cls.__preprocess_text(ref,
                                                     remove_stopwords=remove_stopwords,
                                                     remove_duplicates=remove_duplicates))
                  for ref in references]

        average_scores = {
            'rouge1': max(score['rouge1'].fmeasure for score in scores),
            'rouge2': max(score['rouge2'].fmeasure for score in scores),
            'rougeL': max(score['rougeL'].fmeasure for score in scores),
        }

        return average_scores
    
    @classmethod
    def calculate_fluency(text):
        # Flesch-Kincaid readability score
        readability_score = textstat.flesch_kincaid_grade(text)
        return readability_score

    @staticmethod
    def __get_sentence_words(sentence):
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        words = sentence.lower().split()
        return words
    
    @staticmethod
    def __preprocess_text(text, remove_stopwords=False, remove_duplicates=False):
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Convert to lowercase
        text = text.lower()

        if remove_duplicates:
          text = ' '.join(dict.fromkeys(text.split()))

        # Remove stopwords
        if remove_stopwords:
          stop_words = set(stopwords.words('english'))
          text = ' '.join(word for word in text.split() if word not in stop_words)

        return text
    
    @staticmethod
    def cut_string_at_first_occurrence(text):
        text = str(text)

        # Calculate 20% of the string length
        min_length = len(text) * 0.2

        # Initialize position for cutting
        cut_position = len(text)  # Default to full string length (no cut)
        start_position = 0

        # Iterate through the string to find each occurrence of the delimiters
        for i, char in enumerate(text):
            if char in ['\n', '$', '.', '|']:
                # Check if the current cut leaves more than 20% of the string
                if i >= min_length:
                    cut_position = i
                    break
                else: start_position = i + 1

        # Return the cut string or the full string if no valid cut was found
        return text[start_position:cut_position] if cut_position < len(text) else text
