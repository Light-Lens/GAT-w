from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy, nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('nps_chat')
nltk.download('stopwords')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def ngrams(lst, n):
    ngrams = zip(*[lst[i:] for i in range(n)])
    return list(ngrams)

def tokenize(sentence):
    return nltk.word_tokenize(sentence.strip())

def sent_tokenize(sentence):
    return nltk.sent_tokenize(sentence.strip())

def stem(word):
    return stemmer.stem(word.lower().strip())

def lemmatize(word):
    return lemmatizer.lemmatize(word.lower().strip())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [lemmatize(word) for word in tokenized_sentence]
    bag = numpy.zeros(len(words), dtype=numpy.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag

def stop_words(tokens):
    ignore_words = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    filler_words = set(stopwords.words("english"))

    all_words = [stem(words) for words in tokens if words not in ignore_words]
    return [word for word in all_words if word not in filler_words]
