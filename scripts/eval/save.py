import argparse
import json
import pandas as pd
from pathlib import Path

class EvaluationTableGeneratorWithPandas:
    def __init__(self, eval_output_dir, score, score_type, output_dir, ignore_llama):
        self.eval_output_dir = eval_output_dir
        self.score = score
        self.score_type = score_type
        self.output_dir = output_dir
        self.ignore_llama = ignore_llama
        self.data = []

    def parse_filepath(self, filepath):
        try:
            parts = filepath.parts
            return {
                'model': parts[-4],
                'model_type': parts[-5],
                'task': parts[-3],
                'dataset': parts[-2]
            }
        except IndexError:
            print(filepath)
            exit(1)

    def read_and_extract_scores(self, filepath):
        with open(filepath, 'r') as file:
            return json.load(file)

    def collect_data(self):
        for filepath in Path(self.eval_output_dir).rglob('*.json'):
            if self.ignore_llama and 'llama' in str(filepath):
                continue

            score_metadata = self.parse_filepath(filepath)
            model = f"{score_metadata['model_type'].upper()}: {score_metadata['model'].upper()}"
            dataset = f"{score_metadata['dataset'].title()}-{score_metadata['task'].split('-')[0].upper()}"
            if "Rest" in dataset:
                dataset = dataset.replace("Rest", "Restaurant")
            task = score_metadata['task']
            scores = self.read_and_extract_scores(filepath)

            for score_type, scores_dict in scores.items():
                for score, value in scores_dict.items():
                    self.data.append({
                        'Task': task,
                        'Dataset': dataset,
                        'Model': model,
                        f"{score_type}-{score}": value * 100
                    })

    def format_title(self, task):
        title_task, title_score_type, title_score = self.format_title_components(task)
        return f"{title_task}: {title_score_type} {title_score}".replace(' ', '_')

    def generate_tables(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.data)
        for task in df['Task'].unique():
            task_df = df[df['Task'] == task].drop('Task', axis=1)
            # Ensuring unique column order for consistency across files
            column_order = ['Dataset', 'Model'] + [col for col in task_df.columns if col not in ['Dataset', 'Model']]

            if "-" in self.score_type:
                table_filepath = f"{self.output_dir}/combos/{self.score_type.count('-')+1}/{self.score_type}/{self.score}/{self.format_title(task)}.csv"
            else:
                table_filepath = f"{self.output_dir}/{self.score_type}/{self.score}/{self.format_title(task)}.csv"
                
            task_df.to_csv(table_filepath, index=False, columns=column_order)

def main():
    parser = argparse.ArgumentParser(description='Generate tables summarizing the performance metrics of multiple models across several datasets using pandas.')
    parser.add_argument('-s', '--score', type=str, default='f1-score', help='The score to compare. Default: f1-score')
    parser.add_argument('-st', '--score_type', type=str, default='exact', help='The type of score to compare. Default: exact')
    parser.add_argument('-e', "--eval_output_dir", type=str, default='eval_output', help='The directory containing the evaluation output files. Default: eval_output')
    parser.add_argument("-o", "--output_dir", type=str, default='eval_visualize/tables', help='The directory to save the tables. Default: tables')
    parser.add_argument('-i', '--ignore_llama', action='store_true', help='Ignore llama models in the comparison.')

    args = parser.parse_args()

    table_generator = EvaluationTableGeneratorWithPandas(args.eval_output_dir, args.score, args.score_type, args.output_dir, args.ignore_llama)
    table_generator.collect_data()
    table_generator.generate_tables()


if __name__ == "__main__":
    main()
