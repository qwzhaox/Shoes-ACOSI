import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from pattern.en import pluralize, singularize
from pathlib import Path

COLORS = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"]
PATTERNS = ["/", "-", "\\", "o", "+", "x", "*", "|"]

PARAM_1 = 2
PARAM_2 = 3
CONSTANT_1 = 0
CONSTANT_2 = 1

TASK_ORDERING = ["ACOS-Extract", "ACOSI-Extract", "ACOS-Extend"]
DATASET_ORDERING = ["Restaurant-ACOS", "Laptop-ACOS", "Shoes-ACOS", "Shoes-ACOSI"]
MODEL_ORDERING = ["MvP", "GEN-SCL-NAT", "GPT", "LLaMA"]

FONTSIZE = 16


### HELPER FUNCTIONS ###

def is_skip_dir(s, substrings):
    return any(substring in s for substring in substrings)


def parse_eval_filepath(filepath, eval_dir):

    try:
        parts = filepath.parts
        model_type = parts[-5] if len(parts) > 5 and parts[-5] != eval_dir else parts[-4]
        model = parts[-4]
        if "llama" in model_type:
            model_type = "llama" + model_type[model_type.find("llama") + 5:]
        if "tf-idf" in model_type or "random" in model_type:
            print(model_type)
            model = model + "-" + model_type[model_type.find("-") + 1:]
            model_type = model_type[:model_type.find("-")]
        return {
            'model_type': model_type,
            'model': model,
            'task': parts[-3],
            'dataset': parts[-2]
        }
    except IndexError:
        print(filepath)
        exit(1)


def init_data():
    data = {}
    data['model_type'] = []
    data['model'] = []
    data['dataset'] = []
    data['task'] = []
    data['term'] = []
    data['metric'] = []
    data['score'] = []
    return data


def init_metadata():
    metadata = {}
    metadata['dataset'] = []
    metadata['stat'] = []
    metadata['stat_type'] = []
    metadata['stat_desc'] = []
    metadata['value'] = []
    return metadata


def key_is_stat(key):
    return key in ['mean', 'median', 'stdev', 'min', 'max', 'count']


def parse_metadata(metadata, metadata_dict, dataset):
    for key, value in metadata_dict.items():
        for k, v in value.items():
            metadata['dataset'].append(dataset.split('-')[0])
            metadata['stat'].append(key.title() if key != 'ea/eo/ia/io' else key.upper())
            if key_is_stat(k):
                metadata['stat_type'].append(k.upper())
                metadata['stat_desc'].append('-')
            elif key == 'Splits':
                metadata['stat_type'].append("COUNT")
                metadata['stat_desc'].append(k.upper())
            else:
                metadata['stat_type'].append("COUNT")
                metadata['stat_desc'].append(k.upper())
            metadata['value'].append(v)
    return metadata


def init_mdtt_dict(df):
    mdtt_dict = {}
    mdtt_dict["model"] = sort_models_by_heading_order(df['model'].unique())
    mdtt_dict["dataset"] = df['dataset'].unique()
    mdtt_dict["task"] = df['task'].unique()
    mdtt_dict["term"] = df['term'].unique()
    return mdtt_dict


def get_fixed_mdtt_model(mdtt_dict, df):
    mdtt_dict["model"] = sort_models_by_heading_order(df["model"].unique())
    return mdtt_dict


def sort_models_by_heading_order(elements):
    priority_map = {heading: priority for priority, heading in enumerate(MODEL_ORDERING)}
    def get_heading(element):
        if element != "GEN-SCL-NAT":
            return element.split('-')[0]
        return element
    sorted_elements = sorted(elements, key=lambda x: priority_map.get(get_heading(x), float('inf')))

    return sorted_elements

### DF FUNCTIONS ###

def clean_model_names(df):
    df['model'] = df['model'].str.replace("LLAMA", "LLaMA")
    df['model'] = df['model'].str.replace("-LONG", "")
    df['model'] = df['model'].str.replace("GPT-", "")
    df['model'] = df['model'].str.replace("35", "3.5")
    df['model'] = df['model'].str.replace("-10", "")
    df['model'] = df['model'].str.replace("-5", "")
    df['model'] = df['model'].str.replace("-RANDOM", "-RAND")
    df['model'] = df['model'].str.replace("-TF-IDF", "-KNN")
    df['model_type'] = df['model_type'].replace("LLAMA", "LLaMA")
    df['model_type'] = df['model_type'].str.replace(r'^MVP-SEED-.*$', 'MvP', regex=True)
    return df


def get_filtered_df(df, column, val):
    filtered_df = df.query(f"{column} == '{val}'")
    return filtered_df


def get_double_filtered_df(df, col1, col2, val1, val2):
    filtered_df = get_filtered_df(df, col1, val1)
    filtered_df = get_filtered_df(filtered_df, col2, val2)
    return filtered_df


def combine_model_type_and_model(df, mdtt_dict):
    if df.empty:
        return df, mdtt_dict
    df["model"] = df.apply(lambda row: row["model_type"] + "-" + row["model"] if row["model_type"] not in row["model"] else row["model"], axis=1)
    df = df.drop(columns=["model_type"])
    return df, get_fixed_mdtt_model(mdtt_dict, df)


def do_reorder_columns(df, ordering, key):
    ordering = [item for item in ordering if item in df.columns.get_level_values(key).unique()]
    header_parts = [item for item in df.columns.get_level_values(key).unique() if item not in ordering]
    ordering = header_parts + ordering
    new_columns = df.columns[df.columns.get_level_values(key).isin(ordering)]
    ordered_new_columns = [col for item in ordering for col in new_columns if col == item or col[0] == item]
    df = df[ordered_new_columns]
    return df


def do_reorder_rows(df, ordering, key):
    df[key] = pd.Categorical(df[key], categories=ordering, ordered=True)
    df = df.sort_values(key)
    return df


def reorder_columns_and_rows(df, param1, param2, selected_terms):
    formatted_terms = [get_formatted_term(term) for term in selected_terms]

    if param1 == "term":
        df = do_reorder_columns(df, formatted_terms, "term")
    elif param1 == "task":
        df = do_reorder_columns(df, TASK_ORDERING, "task")
    elif param1 == "dataset":
        df = do_reorder_columns(df, DATASET_ORDERING, "dataset")
    elif param1 == "model":
        df = do_reorder_columns(df, MODEL_ORDERING, "model_type")

    if param2 == "term":
        df = do_reorder_rows(df, formatted_terms, "term")
    elif param2 == "task":
        df = do_reorder_rows(df, TASK_ORDERING, "task")
    elif param2 == "dataset":
        df = do_reorder_rows(df, DATASET_ORDERING, "dataset")
    elif param2 == "model":
        df = do_reorder_rows(df, MODEL_ORDERING, "model_type")

    return df


def reorganize_metadata(df):
    df = do_reorder_columns(df, ["Restaurant", "Laptop", "Shoes"], "dataset")
    df = do_reorder_rows(df, ["Splits", "Tokens/Review", "Tuples", "EA/EO/IA/IO", "Sentiment"], "stat")

    order = ['COUNT', 'MEAN', 'MEDIAN', 'STDEV', 'MIN', 'MAX']
    sorting_key = {stat_type: order.index(stat_type) for stat_type in order if stat_type in df['stat_type'].unique()}
    
    tokens_review = df[df['stat'] == 'Tokens/Review'].sort_values(by='stat_type', key=lambda x: x.map(sorting_key))
    tuples = df[df['stat'] == 'Tuples'].sort_values(by='stat_type', key=lambda x: x.map(sorting_key))
    rest = df[~df['stat'].isin(['Tokens/Review', 'Tuples', 'TOTAL', 'TRAIN', 'TEST', 'DEV'])]
    
    df = pd.concat([tokens_review, tuples, rest], ignore_index=True)

    top_stats_order = ['TOTAL', 'TRAIN', 'DEV', 'TEST']
    sorting_key = {stat_desc: top_stats_order.index(stat_desc) for stat_desc in top_stats_order if stat_desc in df['stat_desc'].unique()}

    splits = df[df['stat'] == 'Splits'].sort_values(by='stat_desc', key=lambda x: x.map(sorting_key))
    rest = df[~df['stat'].isin(['Splits', 'TOTAL', 'TRAIN',  'DEV', 'TEST'])]

    df = pd.concat([splits, rest], ignore_index=True)

    new_categories = df['stat_desc'].unique()
    df['stat'] = df['stat'].cat.add_categories(new_categories)
    
    df.loc[df['stat'] == 'Sentiment', 'stat'] = df['stat_desc']
    df.loc[df['stat'] == 'EA/EO/IA/IO', 'stat'] = df['stat_desc']
    df.loc[df['stat'] == 'Splits', 'stat'] = df['stat_desc']
    df = df.drop(columns=['stat_desc'])

    return df


def merge_mvp_seeds(df):
    condition = (df['model_type'] == 'MvP') 
    df_filtered = df[condition]
    df_merged = df_filtered.groupby(['model_type', 'model', 'dataset', 'task', 'term', 'metric'], as_index=False)['score'].mean()

    # Append the averaged rows back to the original DataFrame, excluding the rows that were averaged
    df_result = pd.concat([df[~condition], df_merged]).reset_index(drop=True)
    return df_result


### FORMATTING FUNCTIONS ###

def get_formatted_model(model_type, model):
    return model_type.upper(), model.upper()


def get_formatted_dataset(dataset, task):
    dataset = f"{dataset.title()}-{task.split('-')[0].upper()}"

    if "Rest" in dataset:
        dataset = dataset.replace("Rest", "Restaurant")

    return dataset


def get_formatted_task(task):
    return f"{task.split('-')[0].upper()}-{task.split('-')[1].title()}" if len(task.split('-')) > 1 else task.upper()


def get_formatted_term(term):
    return term.replace("_", " ").title()


def get_formatted_metric(metric):
    metric = metric.title().replace("Avg", "Average").replace("Iou", "IoU")
    return metric


def get_formatted_title(constant_val1, constant_val2, metric):
    return f"{constant_val1}: {constant_val2} {metric}".replace("_", " ")


class EvalVisualizer:
    def __init__(self, terms, eval_output_dir, output_dir, ordering):
        self.eval_output_dir = Path(eval_output_dir)
        self.output_dir = output_dir

        self.selected_terms = terms
        self.constant1 = ordering[CONSTANT_1]
        self.constant2 = ordering[CONSTANT_2]
        self.param1 = ordering[PARAM_1]
        self.param2 = ordering[PARAM_2]

        self.df = None
        self.df_metadata = None

    def collect_data(self, skip_dirs=[]):
        data = init_data()
        dataset_stat = init_metadata()
        
        for filepath in self.eval_output_dir.rglob('*.json'):
            print("Collecting data from", filepath)

            if is_skip_dir(str(filepath), skip_dirs):
                continue

            metadata = parse_eval_filepath(filepath, self.eval_output_dir.name)

            model_type, model = get_formatted_model(metadata['model_type'], metadata['model'])
            dataset = get_formatted_dataset(metadata['dataset'], metadata['task'])
            task = get_formatted_task(metadata['task'])

            score_dict, dataset_stat_dict = self.__read_and_extract_scores(filepath)
            dataset_stat = parse_metadata(dataset_stat, dataset_stat_dict, dataset)

            for term in score_dict.keys():
                for metric, score in score_dict[term].items():
                    data['model_type'].append(model_type)
                    data['model'].append(model)
                    data['dataset'].append(dataset)
                    data['task'].append(task)
                    data['term'].append(get_formatted_term(term))
                    data['metric'].append(metric)
                    data['score'].append(score * 100)

        self.df = pd.DataFrame(data)
        self.df = clean_model_names(self.df)
        self.df = merge_mvp_seeds(self.df)

        self.df_metadata = pd.DataFrame(dataset_stat)
        self.df_metadata = self.df_metadata.drop_duplicates()

        self.__metadata_to_csv()

    def generate_visuals(self, create_charts=True, create_tables=True, terms_file=None):

        mdtt_dict = init_mdtt_dict(self.df)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        for const_val1 in mdtt_dict[self.constant1]:
            for const_val2 in mdtt_dict[self.constant2]:
                filtered_df = get_double_filtered_df(self.df, self.constant1, self.constant2, const_val1, const_val2)
                if create_tables:
                    print("Creating tables for", const_val1, const_val2)
                    self.__scores_to_csv(filtered_df, const_val1, const_val2, terms_file)
                if create_charts:
                    print("Creating charts for", const_val1, const_val2)
                    filtered_df, mdtt_dict = combine_model_type_and_model(filtered_df, mdtt_dict)

                    self.__plot_scores(filtered_df, const_val1, const_val2, mdtt_dict)

    def generate_additional_visuals(self, create_charts=True, create_tables=True, terms_file=None):
        max_tokens = self.metadata['Tokens/Review']['MAX'].max()

    ### DATA COLLECTION FUNCTIONS ###

    def __read_and_extract_scores(self, filepath):
        with open(filepath, 'r') as file:
            score_json = json.load(file)

            score_dict = {}
            for term in self.selected_terms:
                if term not in score_json:
                    continue
                score_dict[term] = score_json[term]

            dataset_stat_dict = score_json['metadata']
                    
            return score_dict, dataset_stat_dict

    ### TABLE FUNCTIONS ###

    def __metadata_to_csv(self):
        df = self.df_metadata
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns

        df = df[~df['stat_desc'].str.contains('PRED ')]
        df = df[~df['stat_desc'].str.contains('TEST ')]
        df = df[~df['stat'].str.contains('Predicted')]
        df = df[~df['stat'].str.contains('Test')]

        df = df.pivot_table(index=['stat', 'stat_type', 'stat_desc'], columns='dataset', values='value').reset_index()
        df = reorganize_metadata(df)

        self.df_metadata = df

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.df_metadata.to_csv(f"{self.output_dir}/metadata.csv", index=False)
        

    def __scores_to_csv(self, df, const_val_1, const_val_2, terms_file):
        df = df.drop(columns=[self.constant1, self.constant2])
        df = df.query("metric != 'macro IoU'")
        df = df.query("metric != 'avg micro IoU'")
        df = df.query("metric != 'span avg micro IoU'")
        df = df.query("metric != 'span macro IoU'")

        if self.param2 == "model":
            df = df.pivot_table(index=['model_type', self.param2], columns=[self.param1, 'metric'], values='score').reset_index()
        elif self.param1 == "model":
            df = df.pivot_table(index=self.param2, columns=['model_type', self.param1, 'metric'], values='score').reset_index()
        else:
            df = df.pivot_table(index=self.param2, columns=[self.param1, 'metric'], values='score').reset_index()

        if df.empty:
            return

        nan_indices = df['model_type'].isna()
        df.loc[nan_indices, 'model_type'] = df.loc[nan_indices, 'model'].values

        df = reorder_columns_and_rows(df, self.param1, self.param2, self.selected_terms)

        table_filepath = self.__get_table_filepath(const_val_1, const_val_2, terms_file)
        Path(table_filepath).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(table_filepath, index=False)

    ### CHART FUNCTIONS ###

    def __plot_scores(self, df, const_val1, const_val2, mdtt_dict):
        param1_vals, param2_vals = mdtt_dict[self.param1], mdtt_dict[self.param2]

        plt.figure(figsize=(10, 6))

        for metric in df['metric'].unique():

            metric_df = get_filtered_df(df, 'metric', metric)
            param1_vals = self.__plot_by_param1(metric_df, param1_vals, param2_vals)

            legend_loc = self.__get_legend_loc(metric_df, param1_vals)
            self.__label_plot(metric, const_val1, const_val2, param1_vals, legend_loc)

            chart_filepath = self.__get_chart_filepath(metric, const_val1, const_val2)
            Path(chart_filepath).parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(chart_filepath)
            # plt.show()
            plt.clf()

        plt.close()
    

    def __plot_by_param1(self, df, param1_vals, param2_vals):
        all_scores = []
        for param2_val in param2_vals:
            scores, remove_idx = self.__get_scores(df, param2_val, param1_vals)
            all_scores.append(scores)
            if len(scores) == 0:
                continue
            param1_vals = np.delete(param1_vals, remove_idx)

        if len(param1_vals) > 1:
            width = 0.8 / len(param2_vals)  # Adjust width based on number of models
            for i, param2_val in enumerate(param2_vals):
                scores = all_scores[i]
                positions = np.arange(len(param1_vals)) + i * width
                plt.bar(positions, scores, width=width, label=param2_val, color=COLORS[i], hatch=PATTERNS[i])
            
            x_labels = deepcopy(param1_vals)
            if self.param1 == "term" and x_labels[-1] == "Implicit Indicator":
                x_labels[-1] = "Implict Ind."
            plt.xticks(np.arange(len(param1_vals)) + width * (len(param2_vals) - 1) / 2, x_labels, fontsize=FONTSIZE-2)
        else:
            for i, param2_val in enumerate(param2_vals):
                score = all_scores[i][0]
                if score is not None:
                    plt.bar(param2_vals[i], score, label=param2_val, color=COLORS[i], hatch=PATTERNS[i])
            # plt.xticks(fontsize=FONTSIZE-5)
            plt.gca().set_xticks([])

        return param1_vals

    def __get_scores(self, df,  param2_val, param1_vals):
        scores = []
        remove_idx = []
        for j, param1_val in enumerate(param1_vals):
            score = self.__get_score(df, param1_val, param2_val)
            if score is not None:
                scores.append(score)
            else:
                remove_idx.append(j)

        return scores, remove_idx

    def __get_score(self, df, param1_val, param2_val):
        filtered_df = get_double_filtered_df(df, self.param1, self.param2, param1_val, param2_val)
        if not df.empty and not filtered_df.empty:
            return filtered_df['score'].iloc[0]
        return None

    def __get_legend_loc(self, df, param1_vals):
        legend_upper_bound = 64
        legend_lower_bound = 40

        if len(param1_vals) > 1:
            filtered_df = get_filtered_df(df, self.param1, param1_vals[-1])
            last_param1_scores = list(filtered_df['score'].values)
            if len(param1_vals) > 3:
                filtered_df1 = get_filtered_df(df, self.param1, param1_vals[-2])
                last_param1_scores.extend(filtered_df1['score'].values)
            if all(score <= legend_upper_bound for score in last_param1_scores):
                return 'r'
            
            filtered_df2 = get_filtered_df(df, self.param1, param1_vals[0])
            first_param1_scores = list(filtered_df2['score'].values)
            if len(param1_vals) > 3:
                filtered_df3 = get_filtered_df(df, self.param1, param1_vals[1])
                first_param1_scores.extend(filtered_df3['score'].values)
            if all(score <= legend_upper_bound for score in first_param1_scores):
                return 'l'

            if all(score >= legend_lower_bound for score in last_param1_scores):
                return 'rb'

            if all(score >= legend_lower_bound for score in first_param1_scores):
                return 'lb'
            
        return None

    def __label_plot(self, metric, const_val1, const_val2, param1_vals, legend_loc):
        metric = get_formatted_metric(metric)
        plt.title(get_formatted_title(const_val1, const_val2, metric), fontsize=16)
        plt.ylabel(metric, fontsize=FONTSIZE)
        plt.ylim(0, 100)

        legend_label = pluralize(self.param2.title())
        xlabel = pluralize(self.param1.title())
        framealpha = 0.5

        if legend_loc == 'r':
            loc = 'upper right'
            bbox_to_anchor = (1, 1)
        elif legend_loc == 'l':
            loc = 'upper left'
            bbox_to_anchor = (0, 1)
        elif legend_loc == 'rb':
            loc = 'lower right'
            bbox_to_anchor = (1, 0)
            framealpha = 0.75
        elif legend_loc == 'lb':
            loc = 'lower left'
            bbox_to_anchor = (0, 0)
            framealpha = 0.75
        else:
            loc = 'upper left'
            bbox_to_anchor = (1, 1)
        
        if len(param1_vals) == 1:
            xlabel = f"{legend_label} [{singularize(xlabel)}: {param1_vals[0]}]"

        # if len(param1_vals) > 1:
        #     plt.legend(
        #         title=legend_label, 
        #         loc=loc, 
        #         bbox_to_anchor=bbox_to_anchor, 
        #         framealpha=framealpha, 
        #         fontsize=FONTSIZE-2,
        #         title_fontsize=FONTSIZE
        #     )

        plt.legend(
            title=legend_label, 
            loc=loc, 
            bbox_to_anchor=bbox_to_anchor, 
            framealpha=framealpha, 
            fontsize=FONTSIZE-2,
            title_fontsize=FONTSIZE
        )

        plt.xlabel(xlabel, fontsize=FONTSIZE)
        plt.tight_layout()

    ### FILEPATH FUNCTIONS ###

    def __get_chart_filepath(self, metric, const_val1, const_val2):
        chart_filepath = f"{self.output_dir}/charts/{self.__get_filepath(metric, const_val1, const_val2)}.png"
        return chart_filepath

    def __get_table_filepath(self, const_val1, const_val2, terms_file):
        filename = Path(terms_file).stem
        table_filepath = f"{self.output_dir}/tables/{self.__get_filepath(filename, const_val1, const_val2)}.csv"
        return table_filepath

    def __get_filepath(self, metric, const_val1, const_val2):
        const_val1_path, const_val2_path = const_val1, const_val2

        if const_val1 == "term" and "-" in const_val1:
            const_val1_path = f"combos/{const_val1.count('-')+1}/" + const_val1_path
        if const_val2 == "term" and "-" in const_val2:
            const_val2_path = f"combos/{const_val2.count('-')+1}/" + const_val2_path

        chart_type = f"{self.param1}_{self.param2}"
        metric = get_formatted_metric(metric)

        filepath = f"{chart_type}/{const_val1_path}/{const_val2_path}/{const_val1}_{const_val2}_{metric}".replace(" ", "_")
        return filepath


def main():
    parser = argparse.ArgumentParser(description='Compare performance metrics of multiple models across several datasets.')
    parser.add_argument('-f', '--terms_file', type=str, default="eval_visualize/terms/default_chart_terms.txt", help='File containing selected terms to compare.')
    parser.add_argument('-e', "--eval_output_dir", type=str, default='eval_output', help='The directory containing the evaluation output files. Default: eval_output')
    parser.add_argument("-o", "--output_dir", type=str, default='eval_visualize', help='The directory to save the charts. Default: charts')
    parser.add_argument('-c', '--create_charts', action='store_true', help='Create the charts.')
    parser.add_argument('-t', '--create_tables', action='store_true', help='Create the tables.')
    parser.add_argument('-s', '--skip_file', action='store_true', default="eval_visualize/skip_list.txt", help='Skip the dir if in this file.')
    parser.add_argument('-l', '--ordering', type=list, default=["task", "dataset", "term", "model"], help='The parameters and constants of the chart: [PARAM_1, PARAM_2, CONSTANT_1, CONSTANT2].')

    args = parser.parse_args()

    with open(args.terms_file, 'r') as file:
        terms = file.read().splitlines()

    visualizer = EvalVisualizer(terms, args.eval_output_dir, args.output_dir, args.ordering)

    with open(args.skip_file, 'r') as file:
        skip_dirs = file.read().splitlines()

    visualizer.collect_data(skip_dirs=skip_dirs)
    visualizer.generate_visuals(args.create_charts, args.create_tables, terms_file=args.terms_file)


if __name__ == "__main__":
    main()
