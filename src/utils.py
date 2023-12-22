import nltk

def sent_tokenize(sentence):
    return nltk.sent_tokenize(sentence.strip())
