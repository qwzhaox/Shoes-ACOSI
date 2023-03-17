"""
Aggregates annotations from 1) a folder containing per each batch, 2 annotators' annotation files and 2) a folder of curated annotation files, 1 per batch.
--> Need to have folders "annotation" and "curated_shoe_reviews" in the same folder as this script in order to run.
Dumps to a human-friendly output format.
"""
from cgitb import text
from collections import defaultdict
import itertools
import os
import csv
import json
import sys
import re
import pandas as pd

DATA_FOLDER = sys.argv[1] # annotation
file_out = sys.argv[2] # results.json
ANNO_TRACKING = sys.argv[3] # annotation_tracking.xlsx
p_names = sys.argv[4] # p_name.jsonl
CURATED_FOLDER = sys.argv[5] # curated_shoe_reviews
DATA_FOLDER_PATH = os.path.abspath(DATA_FOLDER)
df = pd.read_excel(ANNO_TRACKING)
anno1_col = df["Annotator1 Name"]
anno2_col = df["Annotator2 Name"]
EOS_STR = 'IMPLICIT'
designers = df.iloc[1:8, 7].values.tolist() 
engineers = df.iloc[1:8, 8].values.tolist() 
review_dict = {}


# HELPER FUNCTIONS

def find_product_name(review_text, file):
    for line in file:
        data = json.loads(line)
        #print(data)
        data_text_list = re.compile('\w+').findall(data["text"])
        review_text_list = re.compile('\w+').findall(review_text)

        if review_text_list == data_text_list:
            print("HERE")
            #print(data["p_name"])
            return (data["name"], data["p_name"])


def sort_key(s):
    s = s[:-4]
    match = re.search(r'\d+$', s)
    return int(match.group(0)) if match else 0


def find_first_row_of_next_review(reader, current_row, current_review_num):
    for idx, row in enumerate(itertools.islice(reader, current_row, None)):
        if len(row) > 1:
            if int(row[0].split("-")[0]) == (int(current_review_num) + 1):
                # is this correct?
                return current_row + idx


# start_row is the start of the next review.
# to_append is tuple: (category, sentimentPolarity, connection)
def add_to_correct_implicit_row(reader, start_row, to_append):
    for idx, row in enumerate(itertools.islice(reader, start_row, None)):
        if len(row) > 1:
            if EOS_STR in row[2]:
                row[4] += ('|' + to_append[0])
                row[5] += ('|' + to_append[1])
                row[6] += ('|' + to_append[2])
                return


# function to prepare to append 'connection' (list of lists) to 'processed' (after function returns). 
# run at last token of each review.
def process_dict(dict):
    connected = []
    for index in dict.keys():
        for group in dict[index]:
            word, type, category, sentimentPolarity, connection = group
            if connection != "_":
                thisIsAspect = False
                if "Aspect" in type:
                    thisIsAspect = True
                conn_split = connection.split("[")
                new_conn = []
                other_word = dict[conn_split[0]][0][0]  

                if thisIsAspect:
                    opinion_type =  type.lower().split()[0]
                    new_conn = [word, category, sentimentPolarity, other_word, opinion_type]
                else:
                    opinion_type =  type.lower().split()[0]
                    new_conn = [other_word, category, sentimentPolarity, word, opinion_type]
                connected.append(new_conn)   
                # endif 
    return connected


def standardize_length(mentionType, category, sentimentPolarity, connection):
    mentionType = mentionType.split("|")
    category = category.split("|")
    sentimentPolarity = sentimentPolarity.split("|")
    connection = connection.split("|")
    
    if len(category) < len(mentionType):
        category *= len(mentionType)
        sentimentPolarity *= len(mentionType)
        connection *= len(mentionType)

    return mentionType, category, sentimentPolarity, connection


def detect_and_clean_row_mistakes(connection_first, index_first, category, sentimentPolarity, connection, reader, idx, anno_row):
    if connection_first != '_' and index_first != connection_first:
        move_to_next_review_anno = category, sentimentPolarity, connection
        start_row_arg = find_first_row_of_next_review(reader, idx, index_first)
        add_to_correct_implicit_row(reader, start_row_arg, move_to_next_review_anno)
        for col in range(4, 8):
            anno_row[col] = "_"


def process_nondirect_aspect(mentionType, typeToIndex, activeType, reader, idx, tbd, word, index, category, sentimentPolarity, connection):
    for type in range(len(mentionType)):
        if mentionType[type] in typeToIndex and mentionType[type] == activeType and mentionType[type] != 'Direct Aspect' and idx != (len(reader) - 1) and reader[idx + 1]:
            tbd[typeToIndex[mentionType[type]]][type][0] += " " + word
        else:
            typeToIndex[mentionType[type]] = index 
            if index not in tbd:
                tbd[index] = []
            tbd[index].append(list((word, mentionType[type], category[type], sentimentPolarity[type], connection[type])))
            activeType = mentionType[type]
    return activeType


def process_final_token(index, tbd, mentionType, word, category, sentimentPolarity, connection):
    if index not in tbd:
        tbd[index] = []
    for type in range(len(mentionType)):
        if (list((word, mentionType[type], category[type], sentimentPolarity[type], connection[type])) not in tbd[index]):
            tbd[index].append(list((word, mentionType[type], category[type], sentimentPolarity[type], connection[type])))


def process_blank_mentionType(index, tbd, word, mentionType, category, sentimentPolarity, connection):
    if index not in tbd:
        tbd[index] = []
    tbd[index].append(list((word, mentionType, category, sentimentPolarity, connection)))


# PROCESSING ANNOTATION FILES

out_dicts_list = []
# process 2 specified annotation files in each subdirectory of the 'annotation' folder
for root, dirs, files in os.walk(DATA_FOLDER_PATH): 
    dirs.sort(key=sort_key)
    # each dir represents contains annotations from a batch
    for dir in dirs:
        # TODO: if there is a curated annotation file with the same name (excluding extension) as dir, process that curated file. 
        # Will need to establish new variables - anno3_file, reader3, tbd3, typeToIndex3, activeType3, etc.
        batch_num = dir[:-4]
        process_curated = False
        for filename in os.listdir(CURATED_FOLDER):
            if filename[:-4] == batch_num:
                process_curated = True
                break
        
        batch_num = int(re.search(r'\d+', dir).group(0))
        anno1_name = anno1_col[batch_num].lower()
        anno2_name = anno2_col[batch_num].lower()
        anno1_file = open(DATA_FOLDER+'/'+dir+'/'+anno1_name+'.tsv')
        anno2_file = open(DATA_FOLDER+'/'+dir+'/'+anno2_name+'.tsv')
        anno1_file_lines = anno1_file.read().replace('"', "'").split("\n")
        anno2_file_lines = anno2_file.read().replace('"', "'").split("\n")
        reader1 = list(csv.reader(anno1_file_lines, delimiter="\t"))
        reader2 = list(csv.reader(anno2_file_lines, delimiter="\t"))
        tbd1 = {}
        tbd2 = {}
        typeToIndex1 = {}
        typeToIndex2 = {}
        activeType1 = ""
        activeType2 = ""
        review_num = 0

        anno3_file = None
        anno3_file_lines = None
        reader3 = None
        tbd3 = {}
        typeToIndex3 = {}
        activeType3 = ""

        if process_curated:
            anno3_file = open(CURATED_FOLDER+'/'+filename, 'r')
            anno3_file_lines = anno3_file.read().replace('"', "'").split("\n")
            reader3 = list(csv.reader(anno3_file_lines, delimiter="\t"))
            print(anno3_file)

        # detect and clean mistakes in 2 annotators' files. some IMPLICIT tags were not linked to the correct review through the 
        # annotation GUI. 
        for idx, (anno1_row, anno2_row) in enumerate(zip(reader1, reader2)):
            if len(anno1_row) > 1: # len(anno2_row) would also be > 1 in this case
                index1, chars1, word1, mentionType1, category1, sentimentPolarity1, connection1 = anno1_row[0:7]
                index2, chars2, word2, mentionType2, category2, sentimentPolarity2, connection2 = anno2_row[0:7]
                index1_first = index1.split("-")[0]
                connection1_first = connection1.split("-")[0]
                index2_first = index2.split("-")[0]
                connection2_first = connection2.split("-")[0]
                detect_and_clean_row_mistakes(connection1_first, index1_first, category1, sentimentPolarity1, connection1, reader1, idx, anno1_row)
                detect_and_clean_row_mistakes(connection2_first, index2_first, category2, sentimentPolarity2, connection2, reader2, idx, anno2_row)

        curated_file_line = -1
        # line-by-line enumeration of the tsv output
        for idx, (anno1_row, anno2_row) in enumerate(zip(reader1, reader2)):
            curated_file_line += 1 # TODO: is this ok?
            anno3_row = None
            if process_curated:
                anno3_row = reader3[curated_file_line]
            # if we're dealing with an empty row, then ignore
            # if input files contain no errors, then anno1_row and anno2_row, which contain the review, should be the same.
            if len(anno1_row) == 0 or anno2_row[0][0] == "#":
                # the original sentence with IMPLICIT removed
                if len(anno1_row) != 0 and "#Text" in anno2_row[0]:
                    review_num += 1
                    text_seg = " ".join(anno1_row)[6:]
                    text_seg = text_seg.replace("IMPLICIT", " ")
                    text_seg = " ".join(text_seg.split())
                    review_dict["review"] = text_seg
                    product_name = ""
                    with open(p_names, 'r') as p_names_f:
                        product_names = find_product_name(review_dict["review"], p_names_f)
                    review_dict["name"] = product_names[0]
                    review_dict["p_name"] = product_names[1]
                    review_dict["global_metadata"] = {
                        "batch_id": batch_num,
                        "review_id": review_num
                    }

                    # initialize review_dict to contain 2 annotators' annotations
                    review_dict["annotations"] = [{}, {}]

                    review_dict["annotations"][0]["metadata"] = {}
                    review_dict["annotations"][0]["metadata"]["name"] = anno1_name
                    review_dict["annotations"][0]["metadata"]["annotator_experience"] = "Designer" if anno1_name in designers else "Engineer"
                    review_dict["annotations"][0]["annotation"] = []

                    review_dict["annotations"][1]["metadata"] = {}
                    review_dict["annotations"][1]["metadata"]["name"] = anno2_name
                    review_dict["annotations"][1]["metadata"]["annotator_experience"] = "Designer" if anno2_name in designers else "Engineer"
                    review_dict["annotations"][1]["annotation"] = []

                    # if matching curated annotation exists, add an entry onto review_dict.
                    if process_curated:
                        review_dict["annotations"].append({})
                        review_dict["annotations"][2]["metadata"] = {}
                        review_dict["annotations"][2]["metadata"]["name"] = "Curator"
                        review_dict["annotations"][2]["metadata"]["annotator_experience"] = "Curator"
                        review_dict["annotations"][2]["annotation"] = []

                continue

            index1, chars1, word1, mentionType1, category1, sentimentPolarity1, connection1 = anno1_row[0:7]
            index2, chars2, word2, mentionType2, category2, sentimentPolarity2, connection2 = anno2_row[0:7]
            # if matching curated annotation exists, parse its row
            index3, chars3, word3, mentionType3, category3, sentimentPolarity3, connection3 = (anno3_row[0:7] if process_curated else (None,)*7)
            
            ####################################################################################################            
            mentionType1, category1, sentimentPolarity1, connection1 = standardize_length(mentionType1, category1, sentimentPolarity1, connection1)
            mentionType2, category2, sentimentPolarity2, connection2 = standardize_length(mentionType2, category2, sentimentPolarity2, connection2)
            mentionType3, category3, sentimentPolarity3, connection3 = (standardize_length(mentionType3, category3, sentimentPolarity3, connection3) if process_curated else (None,)*4)
            ####################################################################################################
        
            ####################################################################################################
            activeType1 = process_nondirect_aspect(mentionType1, typeToIndex1, activeType1, reader1, idx, tbd1, word1, index1, category1, sentimentPolarity1, connection1)
            activeType2 = process_nondirect_aspect(mentionType2, typeToIndex2, activeType2, reader2, idx, tbd2, word2, index2, category2, sentimentPolarity2, connection2)
            activeType2 = (process_nondirect_aspect(mentionType3, typeToIndex3, activeType3, reader3, idx, tbd3, word3, index3, category3, sentimentPolarity3, connection3) if process_curated else None)
            ####################################################################################################

            #################################################################################################### 
            # if this is the final token in the review  
            if process_curated:
                if anno3_row[2] == EOS_STR and not reader3[idx + 1]:
                    activeType3 = ""
                    process_final_token(index3, tbd3, mentionType3, word3, category3, sentimentPolarity3, connection3)
                    review_dict["annotations"][2]["annotation"] = process_dict(tbd3)
                    tbd3 = {}
                    typeToIndex3 = {}

            if anno1_row[2] == EOS_STR and not reader1[idx + 1]:
                activeType1 = ""
                process_final_token(index1, tbd1, mentionType1, word1, category1, sentimentPolarity1, connection1)
                review_dict["annotations"][0]["annotation"] = process_dict(tbd1)
                tbd1 = {}
                typeToIndex1 = {}
            
            if anno2_row[2] == EOS_STR and not reader2[idx + 1]:
                activeType2 = ""
                process_final_token(index2, tbd2, mentionType2, word2, category2, sentimentPolarity2, connection2)
                review_dict["annotations"][1]["annotation"] = process_dict(tbd2)
                # NOTE: need to use review_dict.copy(), or else out_dicts_list will only contain the very last review repeated a bunch of times
                if review_dict["annotations"][0]["annotation"] and review_dict["annotations"][1]["annotation"]:
                    out_dicts_list.append(review_dict.copy())
                tbd2 = {}
                typeToIndex2 = {}
            ####################################################################################################

            ####################################################################################################
            if process_curated:
                if mentionType3 == "_":
                    activeType3 = ""
                    # if this is the final token in the review
                    if anno3_row[2] == EOS_STR and not reader3[idx + 1]:
                        process_blank_mentionType(index3, tbd3, word3, mentionType3, category3, sentimentPolarity3, connection3)
                        review_dict["annotations"][2]["annotation"] = process_dict(tbd3)
                        tbd3 = {}
                        typeToIndex3 = {}

            if mentionType1 == "_":
                activeType1 = ""
                if anno1_row[2] == EOS_STR and not reader1[idx + 1]:
                    process_blank_mentionType(index1, tbd1, word1, mentionType1, category1, sentimentPolarity1, connection1)
                    review_dict["annotations"][0]["annotation"] = process_dict(tbd1)
                    tbd1 = {}
                    typeToIndex1 = {}

            if mentionType2 == "_":
                activeType2 = ""
                if anno2_row[2] == EOS_STR and not reader2[idx + 1]:
                    process_blank_mentionType(index2, tbd2, word2, mentionType2, category2, sentimentPolarity2, connection2)
                    review_dict["annotations"][1]["annotation"] = process_dict(tbd2)
                    if review_dict["annotations"][0]["annotation"] and review_dict["annotations"][1]["annotation"]:
                        out_dicts_list.append(review_dict.copy())
                    tbd2 = {}
                    typeToIndex2 = {}
                continue
            ####################################################################################################
        # ENDFOR (traverse 2 annotators' files per batch + 1 curator's file if available)
    # ENDFOR (traverse 1 dir)
print(out_dicts_list)
with open(file_out, 'w') as ofile:
    ofile.write(json.dumps(out_dicts_list))
