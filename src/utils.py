from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy, nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('nps_chat')
nltk.download('stopwords')

class Utils:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def ngrams(self, lst, n):
        ngrams = zip(*[lst[i:] for i in range(n)])
        return list(ngrams)

    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence.strip())

    def sent_tokenize(self, sentence):
        return nltk.sent_tokenize(sentence.strip())

    def stem(self, word):
        return self.stemmer.stem(word.lower().strip())

    def lemmatize(self, word):
        return self.lemmatizer.lemmatize(word.lower().strip())

    def bag_of_words(self, tokenized_sentence, words):
        sentence_words = [self.lemmatize(word) for word in tokenized_sentence]
        bag = numpy.zeros(len(words), dtype=numpy.float32)
        for idx, w in enumerate(words):
            if w in sentence_words:
                bag[idx] = 1

        return bag

    def stop_words(self, tokens):
        ignore_words = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        filler_words = set(stopwords.words("english"))

        all_words = [self.stem(words) for words in tokens if words not in ignore_words]
        return [word for word in all_words if word not in filler_words]
