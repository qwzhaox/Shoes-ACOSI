import pandas as pd
from postprocess import clean_punctuation

num2sent = {"0": "negative", "1": "neutral", "2": "positive"}


def process(input_file, output_file):
    """
    convert the original data to unified format for MvP
    """
    wf = open(output_file, 'w')
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            cur_sent = line.split("####")[0].lower()
            quads = eval(line.split("####")[1])
            new_quads = []
            for quad in quads:
                c, a, o, s = quad
                a_start, a_end = a.split(',')
                a_start = int(a_start)
                a_end = int(a_end)
                if a_start == -1 or a_end == -1:
                    a = "NULL"
                else:
                    a = clean_punctuation(cur_sent[a_start:a_end])
                # c = c.replace('#', ' ').lower()
                c = c.lower()
                s = num2sent[s]
                o_start, o_end = o.split(',')
                o_start = int(o_start)
                o_end = int(o_end)
                if o_start == -1 or o_end == -1:
                    o = "NULL"
                else:
                    o = clean_punctuation(cur_sent[o_start:o_end])
                new_quads.append([a, c, s, o])
            wf.writelines(clean_punctuation(cur_sent) + '####')
            wf.writelines(str(new_quads))
            wf.write('\n')
    wf.close()


if __name__ == '__main__':
    postfix = ['train', 'test', 'dev']
    for p in postfix:
        process(f'./data/raw_data/ASQP-Datasets-main/en-Phone/{p}.txt', f'./data/asqp_dataset/phone/{p}.txt')

    # for p in postfix:
    #     process(f'./acos/laptop/laptop_quad_{p}.tsv', f'./acos/laptop/{p}.txt')

