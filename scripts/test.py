import string

t = str.maketrans("", "", string.punctuation)
s = "hello,                     world!"
s = " ".join(s.split())
print(s.translate(t))
