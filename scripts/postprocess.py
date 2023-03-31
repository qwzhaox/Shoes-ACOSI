"""
Aggregates annotations from a folder of annotation files
Dumps to a human-friendly output format
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

DATA_FOLDER = sys.argv[1]
file_out = sys.argv[2]
ANNO_TRACKING = sys.argv[3]
p_names = sys.argv[4]
DATA_FOLDER_PATH = os.path.abspath(DATA_FOLDER)
df = pd.read_excel(ANNO_TRACKING)
anno1_col = df["Annotator1 Name"]
anno2_col = df["Annotator2 Name"]
EOS_STR = 'IMPLICIT'
designers = df.iloc[1:8, 7].values.tolist() 
engineers = df.iloc[1:8, 8].values.tolist() 
review_dict = {}


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


out_dicts_list = []
# process 2 specified annotation files in each subdirectory of the 'annotation' folder
for root, dirs, files in os.walk(DATA_FOLDER_PATH): 
    dirs.sort(key=sort_key)
    for dir in dirs:
        batch_num = dir[:-4]
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
        next_blank_indices = []
        review_num = 0
        # Correct mistakes in annotation files
        for idx, (anno1_row, anno2_row) in enumerate(zip(reader1, reader2)):
            if len(anno1_row) > 1: # len(anno2_row) would also be > 1 in this case
                index1, chars1, word1, mentionType1, category1, sentimentPolarity1, connection1 = anno1_row[0:7]
                index2, chars2, word2, mentionType2, category2, sentimentPolarity2, connection2 = anno2_row[0:7]
                index1_first = index1.split("-")[0]
                connection1_first = connection1.split("-")[0]
                index2_first = index2.split("-")[0]
                connection2_first = connection2.split("-")[0]
                if connection1_first != '_' and index1_first != connection1_first:
                    move_to_next_review_anno1 = category1, sentimentPolarity1, connection1
                    start_row_arg = find_first_row_of_next_review(reader1, idx, index1_first)
                    # find IMPLICIT in next review. append category1 at end of existing category1, and same for sentiment & connection.
                    add_to_correct_implicit_row(reader1, start_row_arg, move_to_next_review_anno1)
                    # change values in reader
                    for col in range(4, 8):
                        anno1_row[col] = "_"
                
                if connection2_first != '_' and index2_first != connection2_first:
                    move_to_next_review_anno2 = category2, sentimentPolarity2, connection2
                    start_row_arg = find_first_row_of_next_review(reader2, idx, index2_first)
                    add_to_correct_implicit_row(reader2, start_row_arg, move_to_next_review_anno2)
                    for col in range(4, 8):
                        anno2_row[col] = "_"

        # line-by-line enumeration of the tsv output
        for idx, (anno1_row, anno2_row) in enumerate(zip(reader1, reader2)):
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
                    review_dict["annotations"] = [{}, {}]

                    review_dict["annotations"][0]["metadata"] = {}
                    review_dict["annotations"][0]["metadata"]["name"] = anno1_name
                    review_dict["annotations"][0]["metadata"]["annotator_experience"] = "Designer" if anno1_name in designers else "Engineer"
                    review_dict["annotations"][0]["annotation"] = []

                    review_dict["annotations"][1]["metadata"] = {}
                    review_dict["annotations"][1]["metadata"]["name"] = anno2_name
                    review_dict["annotations"][1]["metadata"]["annotator_experience"] = "Designer" if anno2_name in designers else "Engineer"
                    review_dict["annotations"][1]["annotation"] = []
                continue

            index1, chars1, word1, mentionType1, category1, sentimentPolarity1, connection1 = anno1_row[0:7]
            index2, chars2, word2, mentionType2, category2, sentimentPolarity2, connection2 = anno2_row[0:7]

            ####################################################################################################
            mentionType1 = mentionType1.split("|")
            category1 = category1.split("|")
            sentimentPolarity1 = sentimentPolarity1.split("|")
            connection1 = connection1.split("|")

            mentionType2 = mentionType2.split("|")
            category2 = category2.split("|")
            sentimentPolarity2 = sentimentPolarity2.split("|")
            connection2 = connection2.split("|")
            
            if len(category1) < len(mentionType1):
                category1 *= len(mentionType1)
                sentimentPolarity1 *= len(mentionType1)
                connection1 *= len(mentionType1)

            if len(category2) < len(mentionType2):
                category2 *= len(mentionType2)
                sentimentPolarity2 *= len(mentionType2)
                connection2 *= len(mentionType2)
            ####################################################################################################
        
            ####################################################################################################
            for type in range(len(mentionType1)):
                if mentionType1[type] in typeToIndex1 and mentionType1[type] == activeType1 and mentionType1[type] != 'Direct Aspect' and idx != (len(reader1) - 1) and reader1[idx + 1]:
                    tbd1[typeToIndex1[mentionType1[type]]][type][0] += " " + word1
                else:
                    typeToIndex1[mentionType1[type]] = index1
                    if index1 not in tbd1:
                        tbd1[index1] = []
                    tbd1[index1].append(list((word1, mentionType1[type], category1[type], sentimentPolarity1[type], connection1[type])))
                    activeType1 = mentionType1[type]

            for type in range(len(mentionType2)):
                if mentionType2[type] in typeToIndex2 and mentionType2[type] == activeType2 and mentionType2[type] != 'Direct Aspect' and idx != (len(reader2) - 1) and reader2[idx + 1]:
                    tbd2[typeToIndex2[mentionType2[type]]][type][0] += " " + word2
                else:
                    typeToIndex2[mentionType2[type]] = index2
                    if index2 not in tbd2:
                        tbd2[index2] = []
                    tbd2[index2].append(list((word2, mentionType2[type], category2[type], sentimentPolarity2[type], connection2[type])))
                    activeType2 = mentionType2[type]
            ####################################################################################################

            #################################################################################################### 
            # if this is the final token in the review  
            if anno1_row[2] == EOS_STR and not reader1[idx + 1]:
                activeType1 = ""
                if index1 not in tbd1:
                    tbd1[index1] = []
                for type in range(len(mentionType1)):
                    if (list((word1, mentionType1[type], category1[type], sentimentPolarity1[type], connection1[type])) not in tbd1[index1]):
                        tbd1[index1].append(list((word1, mentionType1[type], category1[type], sentimentPolarity1[type], connection1[type])))
                review_dict["annotations"][0]["annotation"] = process_dict(tbd1)
                tbd1 = {}
                typeToIndex1 = {}
            
            if anno2_row[2] == EOS_STR and not reader2[idx + 1]:
                activeType2 = ""
                if index2 not in tbd2:
                    tbd2[index2] = []
                for type in range(len(mentionType2)):
                    if (list((word2, mentionType2[type], category2[type], sentimentPolarity2[type], connection2[type])) not in tbd2[index2]):
                        tbd2[index2].append(list((word2, mentionType2[type], category2[type], sentimentPolarity2[type], connection2[type])))
                review_dict["annotations"][1]["annotation"] = process_dict(tbd2)
                # NOTE: need to use review_dict.copy(), or else out_dicts_list will only contain the very last review repeated a bunch of times
                if review_dict["annotations"][0]["annotation"] and review_dict["annotations"][1]["annotation"]:
                    out_dicts_list.append(review_dict.copy())
                tbd2 = {}
                typeToIndex2 = {}
            ####################################################################################################

            ####################################################################################################
            if mentionType1 == "_":
                activeType1 = ""
                # if this is the final token in the review
                if anno1_row[2] == EOS_STR and not reader1[idx + 1]:
                    if index1 not in tbd1:
                        tbd1[index1] = []
                    tbd1[index1].append(list((word1, mentionType1, category1, sentimentPolarity1, connection1)))
                    review_dict["annotations"][0]["annotation"] = process_dict(tbd1)
                    tbd1 = {}
                    typeToIndex1 = {}

            if mentionType2 == "_":
                activeType2 = ""
                if anno2_row[2] == EOS_STR and not reader2[idx + 1]:
                    if index2 not in tbd2:
                        tbd2[index2] = []
                    tbd2[index2].append(list((word2, mentionType2, category2, sentimentPolarity2, connection2)))
                    review_dict["annotations"][1]["annotation"] = process_dict(tbd2)
                    if review_dict["annotations"][0]["annotation"] and review_dict["annotations"][1]["annotation"]:
                        out_dicts_list.append(review_dict.copy())
                    tbd2 = {}
                    typeToIndex2 = {}
                continue
            ####################################################################################################
        # ENDFOR (finish traversing 1 batch - 2 annotators' files per batch)
    # ENDFOR (finish traversing all batches)
print(out_dicts_list)
with open(file_out, 'w') as ofile:
    ofile.write(json.dumps(out_dicts_list))
