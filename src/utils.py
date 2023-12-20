import nltk

def sent_tokenize(sentence):
    return nltk.sent_tokenize(sentence.strip())

def tokenize(sentence):
    return nltk.word_tokenize(sentence.strip())

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [i.strip() for i in f.readlines()]
        text = []
        for line in lines:
            text.extend(sent_tokenize(line))
