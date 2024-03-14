from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy, nltk

Lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence.strip())

def lemmatize(word):
    return Lemmatizer.lemmatize(word.lower().strip())

def one_hot_encoding(tokenized_sentence, words):
    sentence_words = set([lemmatize(word) for word in tokenized_sentence])
    encoding = numpy.zeros(len(words), dtype=numpy.float32)
    
    for idx, w in enumerate(words):
        if w in sentence_words:
            encoding[idx] = 1

    return encoding

def stop_words(tokens):
    ignore_chars = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    filler_words = set(stopwords.words("english"))
    all_words = [lemmatize(words) for words in tokens if words not in ignore_chars]

    return [word for word in all_words if word not in filler_words]

# encoder: take a string, output a list of integers
def encode(text, stoi):
    return [stoi[c] for c in text]

# decoder: take a list of integers, output a string
def decode(tensor, itos):
    return ''.join([itos[i] for i in tensor])
