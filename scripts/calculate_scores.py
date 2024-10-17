import argparse
import os
import pandas as pd
from src.scoring import PatchingScore
from tqdm import tqdm


def calculate_scores(input_path, subdir_df, task):
    epoch = input_path.split('_')[-3]
    accuracy = float(input_path.split('_')[-1][:-4])
    df = pd.read_csv(input_path)

    df['output'].fillna('', inplace=True)
    df['output'] = df['output'].str.strip()
    df['output'] = df['output'].astype(str)
    df['cut_output'] = df['output'].apply(PatchingScore.cut_string_at_first_occurrence)

    df['rouge1'] = df.apply(lambda row: PatchingScore.calculate_rouge_score(task, row['cut_output'])['rouge1'], axis=1)
    df['rouge2'] = df.apply(lambda row: PatchingScore.calculate_rouge_score(task, row['cut_output'])['rouge2'], axis=1)
    df['rougeL'] = df.apply(lambda row: PatchingScore.calculate_rouge_score(task, row['cut_output'])['rougeL'], axis=1)

    df['class_rate'] = df.apply(lambda row: PatchingScore.calculate_classes_occurence(task, row['cut_output']), axis=1)

    df['rouge1_no_stop'] = df.apply(lambda row: PatchingScore.calculate_rouge_score(task, row['cut_output'], remove_stopwords=True)['rouge1'], axis=1)
    df['rouge2_no_stop'] = df.apply(lambda row: PatchingScore.calculate_rouge_score(task, row['cut_output'], remove_stopwords=True)['rouge2'], axis=1)
    df['rougeL_no_stop'] = df.apply(lambda row: PatchingScore.calculate_rouge_score(task, row['cut_output'], remove_stopwords=True)['rougeL'], axis=1)

    best_rouge1_output = df.loc[df['rouge1_no_stop'] == df['rouge1_no_stop'].max(), 'cut_output'].iloc[0]
    best_class_rate_output = df.loc[df['class_rate'] == df['class_rate'].max(), 'cut_output'].iloc[0]

    subdir_df = subdir_df._append({
        'epoch': epoch,
        'accuracy': accuracy,
        'max_rouge1': df['rouge1_no_stop'].max(),
        'max_class_rate': df['class_rate'].max(),
        'best_rouge1_output': best_rouge1_output,
        'best_class_rate_output' : best_class_rate_output
    }, ignore_index=True)

    return df, subdir_df


def apply_score_calculation_and_save(input_dir, output_dir, task):
    """
    Applies a calculation function to each CSV file in a directory 
    and saves the output to a new directory with the same structure.
  
    Args:
      input_dir: The directory containing the CSV files.
      output_dir: The directory to save the output CSV files.
      calculation_function: The function to apply to each CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    for subdir, _, files in os.walk(input_dir):
        print(f'{subdir=}')
        relative_subdir = os.path.relpath(subdir, input_dir)
        output_subdir = os.path.join(output_dir, relative_subdir)
        
        print(f'{output_subdir=}')
        os.makedirs(output_subdir, exist_ok=True)
        subdir_df = pd.read_csv(f'{output_subdir}.csv') if \
            os.path.exists(f'{output_subdir}.csv') else pd.DataFrame(columns=['epoch', 'accuracy', 'max_rouge1', 'max_class_rate', 'best_rouge1_output', 'best_class_rate_output'])
        
        for file in tqdm(files):
            if file.endswith('.csv'):
                output_path = os.path.join(output_subdir, file)
                if os.path.exists(output_path):
                    continue

                print(f'{file=}')
                input_path = os.path.join(subdir, file)
                output_df, subdir_df = calculate_scores(input_path, subdir_df, task)
                output_df.to_csv(output_path, index=False)

        subdir_df.to_csv(f'{output_subdir}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='input directory', required=True, nargs='?')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory', required=True, nargs='?')
    parser.add_argument('-t', '--task', type=str, help='task name', required=True, nargs='?')

    args = parser.parse_args()

    apply_score_calculation_and_save(args.input_dir, args.output_dir, args.task)
