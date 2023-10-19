import pandas as pd
import argparse
import csv
import re
import json
from copy import deepcopy
from pathlib import Path
from string import punctuation

parser = argparse.ArgumentParser(
    description="Aggregate annotations and create a human-friendly output."
)
parser.add_argument(
    "-d", "--data_dir", type=str, default=".", help="Path to the data folder."
)
parser.add_argument(
    "-a",
    "--annotation_folder",
    default="annotation",
    type=str,
    help="Path to the annotation folder.",
)
parser.add_argument(
    "-c",
    "--curated_folder",
    type=str,
    default="curated_shoe_reviews",
    help="Path to the folder containing curated annotation files.",
)
parser.add_argument(
    "-t",
    "--tracking_file",
    type=str,
    default="annotation_tracking.xlsx",
    help="Path to the annotation tracking file (Excel).",
)
parser.add_argument(
    "-p",
    "--product_names",
    type=str,
    default="p_name.jsonl",
    help="Path to the product names file.",
)

parser.add_argument(
    "-o",
    "--output_file",
    type=str,
    default="results.json",
    help="Path to the output JSON file.",
)

parser.add_argument("-r", "--only_curated", action="store_true")

args = parser.parse_args()

# Process arguments
DATA_DIR = Path(args.data_dir)

ANNOTATION_FOLDER_PATH = DATA_DIR / Path(args.annotation_folder)
CURATED_FOLDER_PATH = DATA_DIR / Path(args.curated_folder)

TRACKING_FILE_PATH = DATA_DIR / Path(args.tracking_file)
PRODUCT_NAMES_FILE_PATH = DATA_DIR / Path(args.product_names)

OUTPUT_FILE_PATH = DATA_DIR / Path(args.output_file)

# Load tracking file
TRACKING_FILE_DF = pd.read_excel(TRACKING_FILE_PATH)

# Get engineer and designer names
ROLE_START_IDX, NUM_ANNOTATORS_PER_ROLE = 1, 6
DESIGNER_COL, ENGINEER_COL = 7, 8

DESIGNERS = TRACKING_FILE_DF.iloc[
    ROLE_START_IDX : ROLE_START_IDX + NUM_ANNOTATORS_PER_ROLE, DESIGNER_COL
].values.tolist()
ENGINEERS = TRACKING_FILE_DF.iloc[
    ROLE_START_IDX : ROLE_START_IDX + NUM_ANNOTATORS_PER_ROLE, ENGINEER_COL
].values.tolist()

# Load product names
PRODUCT_NAMES_DF = pd.read_json(path_or_buf=PRODUCT_NAMES_FILE_PATH, lines=True)
PRODUCT_NAMES_DF = PRODUCT_NAMES_DF[["text", "name", "p_name"]]
PRODUCT_NAMES_DF.drop_duplicates(subset="text", inplace=True)

# Constants
FIRST_REVIEW_IDX = 5

REVIEW_TEXT_INDICATOR = "#Text="
IMPLICT = "IMPLICIT"

ANNOT_WORD_IDX = 2
ANNOT_START = 3
ANNOT_END = 7


def get_annotation_filepath(folder_path, filename, search_pattern=False):
    annotation_file_path = folder_path / filename

    if not annotation_file_path.is_file():
        if search_pattern:
            for annotated_filename in folder_path.iterdir():
                if re.search(filename.stem, annotated_filename.stem):
                    annotation_file_path = folder_path / Path(annotated_filename)
                    return annotation_file_path

            raise FileNotFoundError(f"File {annotation_file_path} does not exist.")
        else:
            raise FileNotFoundError(f"File {annotation_file_path} does not exist.")

    return annotation_file_path


def get_annotation_tsv(folder_path, batch_name, search_pattern=False):
    filename = Path(f"{batch_name}.tsv")
    annotation_file = get_annotation_filepath(folder_path, filename, search_pattern)
    annotation_tsv_file = open(annotation_file, "r")
    annotation_tsv = csv.reader(annotation_tsv_file, delimiter="\t")

    return annotation_tsv, annotation_tsv_file


def get_curated_tsv_and_file(batch_name):
    try:
        curated_tsv, curated_tsv_file = get_annotation_tsv(
            CURATED_FOLDER_PATH, batch_name
        )
    except FileNotFoundError as error:
        if args.only_curated:
            raise error

        curated_tsv = None
        curated_tsv_file = None

    return curated_tsv, curated_tsv_file


def get_annotators_and_file_path(batch_name, batch_id):
    cur_annot_folder_path = ANNOTATION_FOLDER_PATH / Path(f"{batch_name}.txt")

    if cur_annot_folder_path.is_dir():
        assigned_annotators = TRACKING_FILE_DF.loc[
            TRACKING_FILE_DF["Batch"] == batch_id,
            ["Annotator1 Name", "Annotator2 Name"],
        ].values.flatten()

        assigned_annotators = [name.strip().lower() for name in assigned_annotators]

    else:
        raise NotADirectoryError(f"Folder {cur_annot_folder_path } does not exist.")

    return assigned_annotators, cur_annot_folder_path


def get_annotator_tsv_list(assigned_annotators, cur_annot_folder_path):
    annotator_tsv_list = []
    annotator_tsv_file_list = []
    for annotator in assigned_annotators:
        annotator_tsv, annotator_tsv_file = get_annotation_tsv(
            cur_annot_folder_path, annotator, search_pattern=True
        )
        annotator_tsv_list.append(annotator_tsv)
        annotator_tsv_file_list.append(annotator_tsv_file)
    return annotator_tsv_list, annotator_tsv_file_list


def process_annot_filedir_err(error, curated_tsv_exists):
    if not curated_tsv_exists:
        raise error

    return [], [], []


def get_annotator_tsvs_and_files(batch_name, batch_id, curated_tsv_exists=True):
    try:
        (
            annotators,
            cur_annot_folder_path,
        ) = get_annotators_and_file_path(batch_name, batch_id)

        annotator_tsv_list, annotator_tsv_file_list = get_annotator_tsv_list(
            annotators, cur_annot_folder_path
        )
    except FileNotFoundError as error:
        (
            annotators,
            annotator_tsv_list,
            annotator_tsv_file_list,
        ) = process_annot_filedir_err(error, curated_tsv_exists)
    except NotADirectoryError as error:
        (
            annotators,
            annotator_tsv_list,
            annotator_tsv_file_list,
        ) = process_annot_filedir_err(error, curated_tsv_exists)

    return annotators, annotator_tsv_list, annotator_tsv_file_list


def create_complete_list(curated, annotator_list):
    complete_list = [curated] if curated is not None else []
    complete_list.extend(annotator_list)
    return complete_list


def process_dataset_review(line):
    # remove #Text= and IMPLICIT from the review
    annot_str = line[0].replace(REVIEW_TEXT_INDICATOR, "").replace(IMPLICT, "")

    # make sure all punctation is separated by spaces
    punc = re.compile(r"([.,!?;:()[\]])")
    annot_str = punc.sub(" \\1 ", annot_str)

    # remove extra spaces
    annot_str = annot_str.strip()
    annot_str = " ".join(annot_str.split())
    return annot_str


def process_product_review(review):
    # remove all punctuation and spaces
    table = str.maketrans("", "", f"{punctuation} ")
    annot_str = review.translate(table).lower()
    return annot_str


def get_review(line):
    annot_str = process_dataset_review(line)
    return annot_str


def get_annot_dict_starter(review, batch_id, review_id):
    processed_annotations = {}
    product_review = deepcopy(review)
    product_review = process_product_review(product_review)

    PRODUCT_NAMES_DF["text"] = PRODUCT_NAMES_DF["text"].apply(process_product_review)

    name, p_name = PRODUCT_NAMES_DF.loc[
        PRODUCT_NAMES_DF["text"] == product_review,
        ["name", "p_name"],
    ].values.flatten()

    processed_annotations["review"] = review
    processed_annotations["name"] = name
    processed_annotations["p_name"] = p_name
    processed_annotations["global_metadata"] = {
        "batch_id": batch_id,
        "review_id": review_id,
    }
    processed_annotations["annotations"] = []

    return processed_annotations


def process_annotations():
    pass


def process_batch(filedir, review_id, proc_annots_list):
    batch_name = filedir.stem
    batch_id = int(batch_name.split("init_shoes_")[1])

    curated_tsv, curated_tsv_file = get_curated_tsv_and_file(batch_name)
    curated_tsv_exists = curated_tsv is not None

    (
        annotators,
        annotator_tsv_list,
        annotator_tsv_file_list,
    ) = get_annotator_tsvs_and_files(batch_name, batch_id, curated_tsv_exists)

    complete_annotation_tsv_list = create_complete_list(curated_tsv, annotator_tsv_list)
    complete_annotation_tsv_file_list = create_complete_list(
        curated_tsv_file, annotator_tsv_file_list
    )

    first_annotator_csv_idx = 0 if curated_tsv is None else 1

    last_review_id = review_id

    for csv_idx, annotation in enumerate(complete_annotation_tsv_list):
        batch_review_id = last_review_id

        cur_annots_dict = {}

        for line in annotation:
            if not line:
                continue
            elif len(line) == 1:
                if not REVIEW_TEXT_INDICATOR in line[0]:
                    continue

                batch_review_id += 1

                if csv_idx == 0:
                    review_id += 1
                    review = get_review(line)
                    cur_annots_dict = get_annot_dict_starter(
                        review, batch_id, review_id
                    )
                    proc_annots_list.append(cur_annots_dict)

                else:
                    cur_annots_dict = proc_annots_list[batch_review_id]
                    assert (
                        get_review(line) == proc_annots_list[batch_review_id]["review"]
                    )

                continue

            word = line[ANNOT_WORD_IDX]

    for tsv_file in complete_annotation_tsv_file_list:
        tsv_file.close()

    return review_id


def main(processed_annotations_list):
    review_id = -1

    if args.only_curated:
        filedir_names = CURATED_FOLDER_PATH.iterdir()

    else:
        filedir_names = ANNOTATION_FOLDER_PATH.iterdir()

    for filedir in filedir_names:
        try:
            review_id = process_batch(
                filedir, review_id, processed_annotations_list
            )
        except FileNotFoundError:
            print(f"Skipping {filedir}, could not find curated annotation file.")
        except NotADirectoryError:
            print(f"Skipping {filedir}, could not find annotation batch folder.")


if __name__ == "__main__":
    print("Processing annotations...")
    processed_annotations_list = []
    main(processed_annotations_list)

    print(f"Saving results to {OUTPUT_FILE_PATH}...")
    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump(processed_annotations_list, f, indent=4)
    print("Done.")
