import re
import pandas as pd

from nltk import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize


class Preprocess:

    def __init__(self, document=None):
        if document:
            self.document = document
        else:
            print("No text document found to summarize")
            return
        self.word_tokens = list()
        self.word_freq = dict()
        self.sentence_tokens = list()
        self.sentences = list()
        self.sentence_vectors = list()
        self.sentence_vectors_indexes = list()

    def sentence_split(self):
        self.sentences.extend(sent_tokenize(self.document))
        return self.sentences

    def sentence_filter(self, stopwords, min_freq=2):
        PUNCTUATION_RE = re.compile('[^\w\s]')
        #SPACE_RE = re.compile('[\s]+')
        NEW_LINE = re.compile('\n')
        self.word_tokens = list()
        sentence_vectors_temp = list()
        index_list_temp = list()
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
                index_list_temp.append(index)
                self.word_tokens.extend(word_list)

        self.word_freq = FreqDist(self.word_tokens)

        for index, word_list in enumerate(sentence_vectors_temp):
            sentence_vector = [
                word for word in word_list if self.word_freq[word] >= min_freq
            ]
            if len(sentence_vector) >= min_freq:
                self.sentence_vectors.append(sentence_vector)
                self.sentence_vectors_indexes.append(index_list_temp[index])

    @staticmethod
    def tokenize(string):
        tokens = word_tokenize(string)
        return tokens
