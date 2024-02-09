import argparse
import numpy as np
import json
from pathlib import Path
from pandas import DataFrame


class EvaluationComparer:
    def __init__(self, eval_output_dir, score, score_type, output_dir, ignore_llama):
        self.eval_output_dir = eval_output_dir
        self.score = score
        self.score_type = score_type
        self.output_dir = output_dir
        self.ignore_llama = ignore_llama
        self.scores_by_task = {}

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

    def read_and_extract_score(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
            if self.score_type not in data or self.score not in data[self.score_type]:
                return 0, False
            return data[self.score_type][self.score] * 100, True

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
            metric_score, is_valid = self.read_and_extract_score(filepath)

            if not is_valid:
                continue

            self.scores_by_task.setdefault(task, {}).setdefault(dataset, {})[model] = metric_score
            
    def generate_plots(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        for task, datasets_scores in self.scores_by_task.items():
            self.plot_scores(task, datasets_scores)

    def plot_scores(self, task, datasets_scores):
        datasets = sorted(datasets_scores.keys())
        models = sorted(set(model for scores in datasets_scores.values() for model in scores))
        plt.figure(figsize=(10, 6))
        self.plot_by_dataset(datasets, datasets_scores, models)
        self.format_plot(task, datasets)

        if "-" in self.score_type:
            chart_filepath = f"{self.output_dir}/combos/{self.score_type.count('-')+1}/{self.score_type}/{self.score}/{self.format_title(task)}.png"
        else:
            chart_filepath = f"{self.output_dir}/{self.score_type}/{self.score}/{self.format_title(task)}.png"


        Path(chart_filepath).parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(chart_filepath)
        plt.show()

    def plot_by_dataset(self, datasets, datasets_scores, models):
        if len(datasets) > 1:
            width = 0.8 / len(models)  # Adjust width based on number of models
            for i, model in enumerate(models):
                scores = [datasets_scores[dataset].get(model, 0) for dataset in datasets]
                positions = np.arange(len(datasets)) + i * width
                plt.bar(positions, scores, width=width, label=model)
            plt.xticks(np.arange(len(datasets)) + width * (len(models) - 1) / 2, datasets)
            self.xlabel = "Datasets"
        else:
            scores = [datasets_scores[datasets[0]][model] for model in models]
            for i, model in enumerate(models):
                plt.bar(models[i], scores[i], label=model)
            self.xlabel = f"Models [Dataset: {datasets[0]}]"

    def format_plot(self, task, datasets):
        title_task, title_score_type, title_score = self.format_title_components(task)
        plt.title(f"{title_task}: {title_score_type} {title_score}", fontsize=20)
        plt.xlabel(self.xlabel, fontsize=18)
        plt.ylabel(title_score, fontsize=18)
        plt.ylim(0, 100)
        if len(datasets) > 1:
            plt.legend(title='Models', loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.5, fontsize=16)
        plt.tight_layout()

    def format_title(self, task):
        title_task, title_score_type, title_score = self.format_title_components(task)
        return f"{title_task}: {title_score_type} {title_score}".replace(' ', '_')

    def format_title_components(self, task):
        parts = task.split('-')
        title_task = f"{parts[0].upper()}-{parts[1].title()}" if len(parts) > 1 else task.upper()
        title_score_type = self.score_type.title()
        title_score = self.score.title().replace("Avg", "Average").replace("Iou", "IoU")
        return title_task, title_score_type, title_score


def main():
    parser = argparse.ArgumentParser(description='Compare performance metrics of multiple models across several datasets.')
    parser.add_argument('-e', "--eval_output_dir", type=str, default='eval_output', help='The directory containing the evaluation output files. Default: eval_output')
    parser.add_argument('-s', '--score', type=str, default='f1-score', help='The score to compare. Default: f1-score')
    parser.add_argument('-st', '--score_type', type=str, default='exact', help='The type of score to compare. Default: exact')
    parser.add_argument("-o", "--output_dir", type=str, default='eval_visualize/charts', help='The directory to save the charts. Default: charts')
    parser.add_argument('-i', '--ignore_llama', action='store_true', help='Ignore llama models in the comparison.')

    args = parser.parse_args()

    comparer = EvaluationComparer(args.eval_output_dir, args.score, args.score_type, args.output_dir, args.ignore_llama)
    comparer.collect_data()
    comparer.generate_plots()


if __name__ == "__main__":
    main()