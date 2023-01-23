"""
Aggregates annotations from a folder of annotation files
Dumps to a human-friendly output format
"""
from cgitb import text
from collections import defaultdict
import os
import csv
import json

import sys
DATA_FOLDER = sys.argv[1]
ANNO_TRACKING = sys.argv[3]
EOS_STR = 'IMPLICIT'
aggregate = True

if aggregate == True:
    file_out = sys.argv[2]

# out_dicts_list is a list of review_dict dictionaries, one per REVIEW. each dictionary contains:
# "review": ""
# "global_metadata": {}
# "annotations": [] 
out_dicts_list = [] 
review_dict = {}

def process_dict(dict):
    connected = []
    for index in dict.keys():
        for group in dict[index]:
            #print(group)
            word, type, category, sentimentPolarity, connection = group
            if connection!="_":
                thisIsAspect=False
                if "Aspect" in type:
                    thisIsAspect=True
                conn_split = connection.split("[")
                
                other_word = dict[conn_split[0]][0][0]
                
                if thisIsAspect:
                    opinion_type =  type.lower().split()[0]
                    new_conn = [word, category, sentimentPolarity, other_word, opinion_type]
                else:
                    opinion_type =  type.lower().split()[0]
                    new_conn = [other_word, category, sentimentPolarity, word, opinion_type]
                connected.append(new_conn)
        
    return connected

# process each annotation file in the annotation folder
for filename in sorted(os.listdir(DATA_FOLDER)):
    file = open(DATA_FOLDER+'/'+filename, 'r')
    if not aggregate:
        file_out = open('shoes_training/'+filename[:-4]+".txt", 'w')
    #lines = file.read().replace('.','').replace('!','').replace('?','').splitlines()
    #lines = [line + "NULL\n" for line in lines]
    #file_out.writelines(lines)
    lines = file.read().replace('"', "'").split("\n")
    processed = []
    reader = list(csv.reader(lines, delimiter="\t"))
    current_list = []
    tbd = {}
    typeToIndex = {}
    activeType = ""
    original_text = []
    next_blank_indices = []

    # line-by-line enumeration of the tsv output
    for idx, row in enumerate(reader):
        # if we're dealing with an empty row, then ignore
        # 
        if len(row)==0 or row[0][0]=="#":
            # the original sentence with IMPLICIT removed
            if len(row)!=0 and "#Text" in row[0]:
                text_seg = " ".join(row)[6:]
                text_seg = text_seg.replace("IMPLICIT", " ")
                text_seg = " ".join(text_seg.split())
                original_text.append(text_seg)
            continue
        
        #print(row)
        try:
            index, chars, word, mentionType, category, sentimentPolarity, connection = row[0:7]
        except:
            import pdb
            pdb.set_trace()
        #import pdb
        #pdb.set_trace()
        # TODO active type meaning?
        # this seems to be the case where we end on a dead thing - need to clean up loose ends
        if mentionType=="_":
            activeType = ""
            # if this is the final token in the review
            if row[2]==EOS_STR and not reader[idx+1]:
                # ie, this row isn't needed for anything?
                if index not in tbd:
                    tbd[index] = []
                tbd[index].append(list((word, mentionType, category, sentimentPolarity, connection)))
               
                try:
                    processed.append(process_dict(tbd))
                except:
                    import pdb
                    pdb.set_trace()
                tbd = {}
                typeToIndex = {}
            continue
        #if filename == 'init_shoes_11.tsv' and idx == 640:
        #    import pdb
        #    pdb.set_trace()
        # these are cases where the current row contains a mention type
        mentionType = mentionType.split("|")
        category = category.split("|")
        sentimentPolarity = sentimentPolarity.split("|")
        connection = connection.split("|")
        
        if len(category)<len(mentionType):
            category*=len(mentionType)
            sentimentPolarity*=len(mentionType)
            connection*=len(mentionType)
    
        # typetoindex seems to find the start index of unique aspects/opinions
        for type in range(len(mentionType)):
            if mentionType[type] in typeToIndex and mentionType[type]==activeType and mentionType[type] != 'Direct Aspect' and idx != (len(reader) - 1) and reader[idx+1]:
                # extend this in cases where z
                tbd[typeToIndex[mentionType[type]]][type][0] += " " + word
            else:
                typeToIndex[mentionType[type]] = index
                if index not in tbd:
                    tbd[index] = []
                #print((word, mentionType[type], category[type], sentimentPolarity[type], connection[type]))
                tbd[index].append(list((word, mentionType[type], category[type], sentimentPolarity[type], connection[type])))
                activeType=mentionType[type]

        if row[2]==EOS_STR and not reader[idx+1]:
            activeType=""
            if index not in tbd:
                    tbd[index] = []
            for type in range(len(mentionType)):
                if (list((word, mentionType[type], category[type], sentimentPolarity[type], connection[type])) not in tbd[index]):
                    tbd[index].append(list((word, mentionType[type], category[type], sentimentPolarity[type], connection[type])))
        
            processed.append(process_dict(tbd))
            tbd = {}
            typeToIndex = {}
    
    #print(processed)

    for line in range(len(original_text)):
        # print(original_text[line], '\n')
        # print(processed[line])
        review_dict["review"] = original_text[line]
        review_dict["annotations"] = processed[line]
        out_dicts_list.append(review_dict)
        review_dict = {}
    
print(out_dicts_list)

with open(file_out, 'w') as ofile:
    ofile.write(json.dumps(out_dicts_list))
    