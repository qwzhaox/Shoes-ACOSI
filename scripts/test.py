import string
import re

d = [{"thing": []}, {"thing": []}]

t = d[0]


for i in d:
    i["thing"].append("hi")

n = {}
n["a"] = d[0]["thing"]

for i in d:
    i["thing"].append("hi")

print(n)


l = ["a a", "b b", "c c"]

for i in l:
    i = i.split(" ")


raw_mention_type = "boo"
rel_id = re.findall(r"\[.*?\]", raw_mention_type)

print(rel_id)
