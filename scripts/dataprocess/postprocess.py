import pandas as pd
import argparse
import re
import json
from copy import deepcopy
from pathlib import Path
from string import punctuation

parser = argparse.ArgumentParser(
    description="Aggregate annotations and create a human-friendly output."
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    default="data/raw_data",
    help="Path to the data folder.",
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
    default="curated",
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
    "-f",
    "--output_path",
    type=str,
    default="data/",
    help="Path to the output folder.",
)

parser.add_argument(
    "-o",
    "--output_file",
    type=str,
    default="dataset.json",
    help="Path to the output JSON file.",
)

parser.add_argument("-r", "--only_curated", action="store_true")

args = parser.parse_args()

# Process arguments
DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_path)

ANNOTATION_FOLDER_PATH = DATA_DIR / Path(args.annotation_folder)
CURATED_FOLDER_PATH = DATA_DIR / Path(args.curated_folder)
OLD_CURATED_FOLDER_PATH = DATA_DIR / Path(args.curated_folder + "_old")

TRACKING_FILE_PATH = DATA_DIR / Path(args.tracking_file)
PRODUCT_NAMES_FILE_PATH = DATA_DIR / Path(args.product_names)

OUTPUT_FILE_PATH = OUTPUT_DIR / Path(args.output_file)

# Load tracking file
TRACKING_FILE_DF = pd.read_excel(TRACKING_FILE_PATH)

# Get engineer and designer names
ROLE_START_IDX, NUM_ANNOTATORS_PER_ROLE = 1, 6
DESIGNER_COL, ENGINEER_COL = 7, 8

DESIGNERS = TRACKING_FILE_DF.iloc[
    ROLE_START_IDX : ROLE_START_IDX + NUM_ANNOTATORS_PER_ROLE, DESIGNER_COL
].values.tolist()
DESIGNERS = [name.strip().lower() for name in DESIGNERS]

ENGINEERS = TRACKING_FILE_DF.iloc[
    ROLE_START_IDX : ROLE_START_IDX + NUM_ANNOTATORS_PER_ROLE, ENGINEER_COL
].values.tolist()
ENGINEERS = [name.strip().lower() for name in ENGINEERS]

# Load product names
PRODUCT_NAMES_DF = pd.read_json(path_or_buf=PRODUCT_NAMES_FILE_PATH, lines=True)
PRODUCT_NAMES_DF = PRODUCT_NAMES_DF[["text", "name", "p_name"]]
PRODUCT_NAMES_DF.drop_duplicates(subset="text", inplace=True)

# Constants
FIRST_REVIEW_IDX = 5

REVIEW_TEXT_INDICATOR = "#Text="
IMPLICT = "IMPLICIT"

LINE_LENGTH = 7
ANNOT_ID_IDX = 0
ANNOT_WORD_IDX = 2
ANNOT_MENTION_TYPE_IDX = 3
ANNOT_CATEGORY_IDX = 4
ANNOT_POLARITY_IDX = 5
ANNOT_SRC_REL_ID_IDX = 6

EMPTY = "_"

ASPECT = "aspect"
OPINION = "opinion"
CATEGORY = "category"
SENTIMENT = "sentiment"
IMPLICT_EXPLICIT = "impl_expl"

MENTION_TYPE = "mention_type"
WORD_SPAN = "word_span"
ANNOT_ID = "annot_id"
REL_ID = "rel_id"
ONGOING = "ongoing"

SRC_REL_ID = "src_rel_id"
SRC_ANNOT_ID = "src_annot_id"
TARGET_REL_ID = "target_rel_id"
TARGET_ANNOT_ID = "target_annot_id"

DIRECT_INDIRECT_IDX = 0
MENTION_TYPE_IDX = 1


def get_annotation_filepath(folder_path, filename, search_pattern=False):
    annotation_file_path = folder_path / filename

    if not annotation_file_path.is_file():
        if search_pattern:
            for annotated_filename in folder_path.iterdir():
                fn_no_spaces = filename.stem.replace(" ", "")
                afn_no_spaces = annotated_filename.stem.replace(" ", "")
                if re.search(fn_no_spaces, afn_no_spaces):
                    annotation_file_path = folder_path / Path(annotated_filename)
                    return annotation_file_path

            raise FileNotFoundError(f"File {annotation_file_path} does not exist.")
        else:
            raise FileNotFoundError(f"File {annotation_file_path} does not exist.")

    return annotation_file_path


def get_annotation_tsv(folder_path, batch_name_or_annotator, search_pattern=False, is_curated=False):
    if is_curated:
        filename = Path(f"{batch_name_or_annotator}.txt/CURATION_USER.tsv")
    else:
        filename = Path(f"{batch_name_or_annotator}.tsv")
    annotation_file = get_annotation_filepath(folder_path, filename, search_pattern)
    annotation_tsv = []

    def is_not_empty_str(x):
        return x != ""

    with open(annotation_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        new_lines = list(filter(is_not_empty_str, line.split("\n")))
        for new_line in new_lines:
            new_line = list(filter(is_not_empty_str, new_line.split("\t")))
            annotation_tsv.append(new_line)

    return annotation_tsv


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
    for annotator in assigned_annotators:
        annotator_tsv = get_annotation_tsv(
            cur_annot_folder_path, annotator, search_pattern=True
        )
        annotator_tsv_list.append(annotator_tsv)
    return annotator_tsv_list


def process_annot_filedir_err(error, curated_tsv_exists):
    if not curated_tsv_exists:
        raise error

    return [], []


def get_curated_tsv(batch_name):
    try:
        curated_tsv = get_annotation_tsv(CURATED_FOLDER_PATH, batch_name, is_curated=True)
    except FileNotFoundError as error:
        if args.only_curated:
            raise error

        curated_tsv = None

    return curated_tsv


def get_annotator_tsvs(batch_name, batch_id, curated_tsv_exists=True):
    try:
        (
            annotators,
            cur_annot_folder_path,
        ) = get_annotators_and_file_path(batch_name, batch_id)

        annotator_tsv_list = get_annotator_tsv_list(annotators, cur_annot_folder_path)
    except FileNotFoundError as error:
        annotators, annotator_tsv_list = process_annot_filedir_err(
            error, curated_tsv_exists
        )
    except NotADirectoryError as error:
        annotators, annotator_tsv_list = process_annot_filedir_err(
            error, curated_tsv_exists
        )

    return annotators, annotator_tsv_list


def create_complete_list(curated, annotator_list):
    complete_list = [curated] if curated is not None else []
    complete_list.extend(annotator_list)
    return complete_list


############################################################### ANNOTATION PROCESSING REVIEW HELPERS ###############################################################


def clean_punctuation(words):
    punc = re.compile(f"[{re.escape(punctuation)}]")
    words = punc.sub(" \\g<0> ", words)

    # remove extra spaces
    words = words.strip()
    words = " ".join(words.split())
    return words


def process_dataset_review(review):
    # remove #Text= and IMPLICIT from the review
    annot_str = (
        review.replace(REVIEW_TEXT_INDICATOR, "").replace(IMPLICT, "").replace("`", "'")
    )

    annot_str = clean_punctuation(annot_str)

    return annot_str


def process_product_review(review):
    # remove all punctuation and spaces
    table = str.maketrans("", "", f"{punctuation} ")
    annot_str = review.translate(table).lower()
    return annot_str


############################################################### ANNOTATION PROCESSING ANNOTATION HELPERS ###############################################################


def set_spans_ongoing_false(ongoing_spans):
    for span in ongoing_spans:
        span[ONGOING] = False


def get_mention_type_rel_id(raw_mention_type):
    rel_id = re.findall(r"\[.*?\]", raw_mention_type)
    if not rel_id:
        rel_id = 0
    else:
        rel_id = int(rel_id[0].replace("[", "").replace("]", ""))
    return rel_id


def clean_mention_type(raw_mention_type):
    # def __clean_mention_type(self, raw_mention_type, rel_id):
    raw_mention_type = raw_mention_type.split(" ")
    direct_indirect = raw_mention_type[DIRECT_INDIRECT_IDX].lower()
    if ASPECT in raw_mention_type[MENTION_TYPE_IDX].lower():
        # if ASPECT in raw_mention_type[MENTION_TYPE_IDX].lower() or rel_id == 0:
        mention_type = ASPECT
    elif OPINION in raw_mention_type[MENTION_TYPE_IDX].lower():
        mention_type = OPINION

    return direct_indirect, mention_type


def get_new_span(annot_id, rel_id, direct_indirect, mention_type, word):
    new_span = {}

    new_span[ANNOT_ID] = annot_id
    new_span[REL_ID] = rel_id
    new_span[IMPLICT_EXPLICIT] = direct_indirect
    new_span[MENTION_TYPE] = mention_type
    new_span[WORD_SPAN] = [word]
    new_span[ONGOING] = True

    return new_span


def is_same_span(span, new_span):
    return (
        new_span[REL_ID] == span[REL_ID]
        and new_span[IMPLICT_EXPLICIT] == span[IMPLICT_EXPLICIT]
        and new_span[MENTION_TYPE] == span[MENTION_TYPE]
    )


def update_ongoing_spans_if_necessary(ongoing_spans, new_span, rel_id):
    if rel_id == 0:
        return

    for span in ongoing_spans:
        if is_same_span(span, new_span):
            span[WORD_SPAN].extend(new_span[WORD_SPAN])
            new_span[ONGOING] = False
            span[ONGOING] = True


def remove_finished_spans(ongoing_spans):
    for span in ongoing_spans:
        if not span[ONGOING]:
            ongoing_spans.remove(span)


def get_span_rel_id_annot_id(span):
    return span[REL_ID], span[ANNOT_ID]


def update_incomplete_annots(incomplete_annots, span):
    rel_id, annot_id = get_span_rel_id_annot_id(span)

    for incomplete_annot in incomplete_annots:
        if not incomplete_annot[TARGET_REL_ID] == rel_id:
            continue

        incomplete_annot[span[MENTION_TYPE]] = span[WORD_SPAN]
        incomplete_annot[f"{span[MENTION_TYPE]}_{IMPLICT_EXPLICIT}"] = span[
            IMPLICT_EXPLICIT
        ]
        incomplete_annot[TARGET_ANNOT_ID] = annot_id


def add_new_spans_to_ongoing_spans(
    ongoing_spans, new_spans, span_dict, incomplete_annots
):
    for span in new_spans:
        if not span[ONGOING]:
            continue

        update_incomplete_annots(incomplete_annots, span)

        ongoing_spans.append(span)

        rel_id, annot_id = get_span_rel_id_annot_id(span)
        span_dict[(annot_id, rel_id)] = span


def is_relation(line):
    return (
        line[ANNOT_CATEGORY_IDX] != EMPTY
        and line[ANNOT_POLARITY_IDX] != EMPTY
        and line[ANNOT_SRC_REL_ID_IDX] != EMPTY
    )


def get_incomplete_annots(categories, polarities, raw_src_rel_ids):
    categories = categories.split("|")
    polarities = polarities.split("|")
    raw_src_rel_ids = raw_src_rel_ids.split("|")

    assert len(categories) == len(polarities) == len(raw_src_rel_ids)

    incomplete_annots = []
    for category, polarity, raw_src_rel_id in zip(
        categories, polarities, raw_src_rel_ids
    ):
        src_annot_id, src_rel_id, target_rel_id = clean_src_rel_id(raw_src_rel_id)
        incomplete_annot = get_incomplete_annot(
            category, polarity, src_annot_id, src_rel_id, target_rel_id
        )
        incomplete_annots.append(incomplete_annot)

    return incomplete_annots


def get_incomplete_annot(category, polarity, src_annot_id, src_rel_id, target_rel_id):
    new_annot = {}
    new_annot[CATEGORY] = category
    new_annot[SENTIMENT] = polarity
    new_annot[SRC_ANNOT_ID] = src_annot_id
    new_annot[SRC_REL_ID] = src_rel_id
    new_annot[TARGET_REL_ID] = target_rel_id
    return new_annot


def clean_src_rel_id(raw_src_rel_id):
    raw_src_rel_id = raw_src_rel_id.split("[")
    src_annot_id = raw_src_rel_id[0]
    if len(raw_src_rel_id) == 1:
        src_rel_id = 0
        target_rel_id = 0
    else:
        raw_src_rel_id = raw_src_rel_id[1].replace("]", "")
        raw_src_rel_id = raw_src_rel_id.split("_")
        src_rel_id = int(raw_src_rel_id[0])
        target_rel_id = int(raw_src_rel_id[1])
    return src_annot_id, src_rel_id, target_rel_id


def get_missing_span(annot, span_dict):
    src_annot_id = annot[SRC_ANNOT_ID]
    src_rel_id = annot[SRC_REL_ID]
    return span_dict[(src_annot_id, src_rel_id)]


def process_new_annot(annot, missing_span, m_ie_key):
    annot[missing_span[MENTION_TYPE]] = missing_span[WORD_SPAN]
    annot[m_ie_key] = missing_span[IMPLICT_EXPLICIT]

    aspect = " ".join(annot[ASPECT])

    if aspect.lower() == "it":
        aspect = "IMPLICIT"

    annot_list = [
        aspect,
        annot[CATEGORY],
        annot[SENTIMENT],
        " ".join(annot[OPINION]),
        annot[f"{OPINION}_{IMPLICT_EXPLICIT}"],
    ]

    return annot_list


############################################################### ANNOTATION PROCESSING ###############################################################


class AnnotationProcessor:
    def __init__(
        self,
        complete_annotation_tsv_list,
        proc_annots_list,
        review_id,
        batch_id,
        annotators,
        curated_tsv_exists,
    ):
        self.complete_annotation_tsv_list = complete_annotation_tsv_list

        self.proc_annots_list = proc_annots_list

        self.review_id = review_id
        self.batch_id = batch_id
        self.LAST_REVIEW_ID = review_id

        self.annotators = annotators
        self.first_annotator_tsv_idx = 0 if not curated_tsv_exists else 1

    ########################## MAIN PROCESSING FUNCTION ##########################

    def process_annotations(self):
        for tsv_idx, annotation in enumerate(self.complete_annotation_tsv_list):
            self.review_id_batch = self.LAST_REVIEW_ID
            self.cur_annots_dict = {}
            self.is_first_annot_line = True
            self.skip_review = False

            is_first_tsv = tsv_idx == 0
            annotator_idx = tsv_idx - self.first_annotator_tsv_idx

            for line in annotation:
                is_annotation = self.__is_annotation(line, is_first_tsv)
                if not is_annotation:
                    self.ongoing_spans = []
                    continue

                if self.is_first_annot_line:
                    self.skip_review = False
                    self.__init_new_annot(annotator_idx)

                if not self.skip_review:
                    try:
                        self.__process_annot_line(line)
                    except IndexError:
                        print(f"SKIPPED: {line}")
                        print(f"Batch: {self.batch_id}\nReview: {self.review_id}\n")
                        self.skip_review = True
                        exit(1)
                        break

        return self.review_id

    ########################## TYPE DETERMINER ##########################

    def __is_annotation(self, line, is_first_tsv):
        if len(line) == 1:
            if not self.is_first_annot_line:
                self.__process_new_annots()
            if REVIEW_TEXT_INDICATOR in line[0]:
                self.__process_review(line[0], is_first_tsv)
            return False
        if len(line) < LINE_LENGTH:
            line.extend([EMPTY] * (LINE_LENGTH - len(line)))
        elif len(line) > LINE_LENGTH:
            new_line = new_line[:LINE_LENGTH]
        if line[ANNOT_MENTION_TYPE_IDX] == EMPTY:
            # print(f"SKIPPED: {line}")
            return False
        return True

    ########################## ANNOTATION FUNCTIONS ##########################

    def __init_new_annot(self, annotator_idx):
        self.annot_dict_for_cur_review = {}
        self.annot_dict_for_cur_review["metadata"] = self.__get_metadata(annotator_idx)
        self.annot_dict_for_cur_review["annotation"] = []
        self.cur_annots_dict["annotations"].append(self.annot_dict_for_cur_review)

        self.incomplete_annot_list = []

        self.span_dict = {}
        self.ongoing_spans = []

        self.is_first_annot_line = False

    def __get_metadata(self, annotator_idx):
        metadata = {}

        if self.__is_annotator(annotator_idx):
            metadata["name"] = self.annotators[annotator_idx]

            if metadata["name"] in DESIGNERS:
                metadata["role"] = "designer"
            elif metadata["name"] in ENGINEERS:
                metadata["role"] = "engineer"
            else:
                metadata["role"] = "unknown"

        else:
            metadata["name"] = "curator"
            metadata["role"] = "curator"

        return metadata

    def __is_annotator(self, annotator_idx):
        return annotator_idx >= 0

    def __process_annot_line(self, line):
        annot_id = line[ANNOT_ID_IDX]
        word = clean_punctuation(line[ANNOT_WORD_IDX])
        raw_mention_types = line[ANNOT_MENTION_TYPE_IDX]

        if is_relation(line):
            categories = line[ANNOT_CATEGORY_IDX]
            polarities = line[ANNOT_POLARITY_IDX]
            raw_src_rel_ids = line[ANNOT_SRC_REL_ID_IDX]
            incomplete_annots = get_incomplete_annots(
                categories, polarities, raw_src_rel_ids
            )
            self.__process_spans(raw_mention_types, word, annot_id, incomplete_annots)
            self.incomplete_annot_list.extend(incomplete_annots)
        else:
            self.__process_spans(raw_mention_types, word, annot_id)

    def __process_spans(self, raw_mention_types, word, annot_id, incomplete_annots=[]):
        set_spans_ongoing_false(self.ongoing_spans)
        new_spans = []

        raw_mention_types = raw_mention_types.split("|")
        for raw_mention_type in raw_mention_types:
            rel_id = get_mention_type_rel_id(raw_mention_type)
            direct_indirect, mention_type = clean_mention_type(raw_mention_type)
            # direct_indirect, mention_type = clean_mention_type(
            #     raw_mention_type, rel_id
            # )
            new_span = get_new_span(
                annot_id, rel_id, direct_indirect, mention_type, word
            )
            new_spans.append(new_span)

            update_ongoing_spans_if_necessary(self.ongoing_spans, new_span, rel_id)

        remove_finished_spans(self.ongoing_spans)
        add_new_spans_to_ongoing_spans(
            self.ongoing_spans, new_spans, self.span_dict, incomplete_annots
        )

    def __process_new_annots(self):
        try:
            for annot in self.incomplete_annot_list:
                missing_span = get_missing_span(annot, self.span_dict)
                m_ie_key = f"{missing_span[MENTION_TYPE]}_{IMPLICT_EXPLICIT}"

                if missing_span[MENTION_TYPE] in annot:
                    original_span = annot[missing_span[MENTION_TYPE]]
                    original_ie = annot[m_ie_key]
                    original_mention_type = missing_span[MENTION_TYPE]

                try:
                    annot_list = process_new_annot(annot, missing_span, m_ie_key)
                    self.annot_dict_for_cur_review["annotation"].append(annot_list)
                except KeyError:
                    self.__print_invalid_annot(
                        annot,
                        original_span,
                        missing_span,
                        original_ie,
                        original_mention_type,
                    )
                    raise KeyError
        except KeyError:
            self.annot_dict_for_cur_review["annotation"] = []
            return

    def __print_invalid_annot(
        self, annot, original_span, missing_span, original_ie, original_mention_type
    ):
        print("INVALID ANNOTATION: SKIPPED")
        print(
            f"Annotator: {self.annot_dict_for_cur_review['metadata']['name']}\n"
            f"Batch: {self.batch_id}\n"
            f"Target: [{annot[TARGET_ANNOT_ID]}, {original_ie} {original_mention_type} {annot[TARGET_REL_ID]}, {annot[CATEGORY]}, {annot[SENTIMENT]}]\n"
            f"Source: [{annot[SRC_ANNOT_ID]}, {missing_span[IMPLICT_EXPLICIT]} {missing_span[MENTION_TYPE]} {annot[SRC_REL_ID]}]\n"
        )

        print("Target span:", original_span)
        print("Source span:", missing_span[WORD_SPAN])
        print("\n")

    ########################## NON-ANNOTATION FUNCTIONS ##########################

    def __process_review(self, review, is_first_tsv):
        self.is_first_annot_line = True
        self.review_id_batch += 1

        if is_first_tsv:
            self.review_id += 1
            review = process_dataset_review(review)
            self.cur_annots_dict = self.__get_annot_dict_starter(review)
            self.proc_annots_list.append(self.cur_annots_dict)

        else:
            self.cur_annots_dict = self.proc_annots_list[self.review_id_batch]
            assert process_dataset_review(review) == self.cur_annots_dict["review"]

    def __get_annot_dict_starter(self, review):
        processed_annotations = {}

        product_review = deepcopy(review)
        product_review = process_product_review(product_review)

        PRODUCT_NAMES_DF["text"] = PRODUCT_NAMES_DF["text"].apply(
            process_product_review
        )

        names = PRODUCT_NAMES_DF.loc[
            PRODUCT_NAMES_DF["text"] == product_review,
            ["name", "p_name"],
        ].values.flatten()

        if len(names) == 0:
            name, p_name = "unknown", "unknown"
        elif len(names) == 1:
            name, p_name = names[0], names[0]
        else:
            name, p_name = names[0], names[1]

        processed_annotations["review"] = review
        processed_annotations["name"] = name
        processed_annotations["p_name"] = p_name
        processed_annotations["global_metadata"] = {
            "batch_id": self.batch_id,
            "review_id": self.review_id,
        }
        processed_annotations["annotations"] = []

        return processed_annotations


############################################################### ANNOTATION PROCESSING ###############################################################


def process_batch(filedir, review_id, proc_annots_list):
    batch_name = filedir.stem
    batch_id = int(batch_name.split("init_shoes_")[1])

    if not Path(OLD_CURATED_FOLDER_PATH / f"{batch_name}.tsv").is_file():
        return 0, False

    curated_tsv = get_curated_tsv(batch_name)
    curated_tsv_exists = curated_tsv is not None

    annotators, annotator_tsv_list = get_annotator_tsvs(
        batch_name, batch_id, curated_tsv_exists
    )

    complete_annotation_tsv_list = create_complete_list(curated_tsv, annotator_tsv_list)

    annotation_processor = AnnotationProcessor(
        complete_annotation_tsv_list,
        proc_annots_list,
        review_id,
        batch_id,
        annotators,
        curated_tsv_exists,
    )

    review_id = annotation_processor.process_annotations()

    return review_id, True


def main(processed_annotations_list):
    review_id = -1

    if args.only_curated:
        filedir_names = CURATED_FOLDER_PATH.iterdir()

    else:
        filedir_names = ANNOTATION_FOLDER_PATH.iterdir()

    for filedir in filedir_names:
        try:
            old_review_id = review_id
            review_id, is_valid_batch = process_batch(filedir, review_id, processed_annotations_list)
            if not is_valid_batch:
                review_id = old_review_id
                raise Exception
        except FileNotFoundError:
            print(f"Skipping {filedir}, could not find curated annotation file.")
            continue
        except NotADirectoryError:
            print(f"Skipping {filedir}, could not find annotation batch folder.")
            continue
        except:
            print(f"Skipping {filedir}, invalid batch name.")
            continue


if __name__ == "__main__":
    print("Processing annotations...")
    processed_annotations_list = []
    main(processed_annotations_list)

    print(f"Saving results to {OUTPUT_FILE_PATH}...")
    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump(processed_annotations_list, f, indent=4)
    print("Done.")
