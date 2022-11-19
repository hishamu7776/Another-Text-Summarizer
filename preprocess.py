import re
import pandas as pd

from nltk import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize


class Preprocess:

    def __init__(self, path=None, columns=None):
        self.corpus = pd.read_csv(path)
        self.document_column = columns[0]
        self.highlights_column = columns[1]

    def select_a_document(self, index=0):
        self.document = self.corpus[self.document_column][index]
        self.highlight = self.corpus[self.highlights_column], [index]
        return self.document, self.highlight

    def sentence_split(self):
        self.sentences = sent_tokenize(self.document)
        return self.sentences

    def sentence_filter(self, stopwords, occurance=2):
        PUNCTUATION_RE = re.compile('[^\w\s]')
        SPACE_RE = re.compile('[\s]+')
        NEW_LINE = re.compile('\n')
        self.word_tokens = list()
        #Cleaning all sentences
        new_sentence_list = list()
        for index, sentence in enumerate(self.sentences):
            sentence = sentence.lower()
            sentence = PUNCTUATION_RE.sub('', sentence)
            #print(sentence)
            sentence = NEW_LINE.sub('', sentence)
            #print(sentence)
            sentence = [
                word for word in sentence.split() if word not in stopwords
            ]
            #print(sentence)
            if len(sentence) > 0:
                new_sentence_list.append(sentence)
                self.word_tokens.extend(sentence)

        self.word_freq = FreqDist(self.word_tokens)

        for index, sentence in enumerate(new_sentence_list):
            for word, freq in self.word_freq.items():
                if freq < occurance and word in sentence:
                    sentence.remove(word)
                new_sentence_list[index] = sentence

        self.sentence_vectors = new_sentence_list
        return new_sentence_list

    @staticmethod
    def tokenize(string):
        tokens = word_tokenize(string)
        return tokens
