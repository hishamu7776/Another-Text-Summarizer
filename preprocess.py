import re
import pandas as pd

from nltk import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize


class Preprocess:

    def __init__(self, path=None, file_type='text'):

        self.file_type = file_type
        self.path = path
        self.word_tokens = list()
        self.word_freq = dict()
        self.sentence_tokens = list()
        self.sentences = list()
        self.sentence_vectors = list()

        try:
            if self.file_type == 'dataframe':
                self.corpus = pd.read_csv(path)
            else:
                self.corpus = open(self.path, "r", encoding='utf-8')                
        except:
            print("Error**-Please check your file path or file type.")

    def read_article(self, column=None):
        if self.file_type == 'dataframe':
            if column:
                self.document = self.corpus[column]
            else:
                print(
                    "Mentioned data type is dataframe and column for desired document is not mentioned."
                )
                return
        elif self.file_type == 'text':
            self.document = self.corpus.read()
            self.corpus.close()
            return

        return self.document

    def sentence_split(self):
        self.sentences.extend(sent_tokenize(self.document))
        return self.sentences

    def sentence_filter(self, stopwords, occurance=2):
        PUNCTUATION_RE = re.compile('[^\w\s]')
        #SPACE_RE = re.compile('[\s]+')
        NEW_LINE = re.compile('\n')
        self.word_tokens = list()
        sentence_vectors_temp = list()
        #Cleaning all sentences
        for index, sentence in enumerate(self.sentences):
            sentence = sentence.lower()
            sentence = PUNCTUATION_RE.sub('', sentence)
            sentence = NEW_LINE.sub('', sentence)
            self.sentence_tokens.append(sentence)
            word_list = [
                word for word in sentence.split() if word not in stopwords
            ]
            #print(sentence)
            if len(sentence) > 0:
                sentence_vectors_temp.append(word_list)
                self.word_tokens.extend(word_list)

        self.word_freq = FreqDist(self.word_tokens)

        for word_list in sentence_vectors_temp:
            sentence_vector = [
                word for word in word_list if self.word_freq[word] >= occurance
            ]
            if len(sentence_vector) >= occurance:
                self.sentence_vectors.append(sentence_vector)

    @staticmethod
    def tokenize(string):
        tokens = word_tokenize(string)
        return tokens
