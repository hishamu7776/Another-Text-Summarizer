from multiprocessing.resource_sharer import stop
import re
import nltk

from turtle import clear
import pandas as pd
import numpy as np

from string import punctuation
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.probability import FreqDist
from textblob import TextBlob

from io import open
from conllu import parse_incr



class Preprocess:
    def __init__(self,path=None,columns=None):
        self.corpus = pd.read_csv(path)
        self.document_column = columns[0]
        self.highlights_column = columns[1]
    
    def select_a_document(self,index = 0):
        self.document = self.corpus[self.document_column][index]
        self.highlight = self.corpus[self.highlights_column],[index]
        return self.document,self.highlight
    
    def sentence_split(self):
        self.sentences = sent_tokenize(self.document)
        return self.sentences
    
    def sentence_filter(self,stopwords):
        PUNCTUATION_RE = re.compile('[^\w\s]')
        SPACE_RE = re.compile('[\s]+')
        NEW_LINE = re.compile('\n')
        self.word_tokens = list()
        #Cleaning all sentences
        new_sentence_list = list()
        for index,sentence in enumerate(self.sentences):
            sentence = sentence.lower()
            sentence = PUNCTUATION_RE.sub('',sentence)
            #print(sentence)
            sentence = NEW_LINE.sub('',sentence)
            #print(sentence)
            sentence = ' '.join([word for word in sentence.split() if word not in stopwords])
            #print(sentence)
            if len(sentence) > 0:
                new_sentence_list.append(sentence)
            word_token = self.tokenize(sentence)
            self.word_tokens.extend(word_token)
        self.word_freq=nltk.FreqDist(self.word_tokens)
        for word,freq in self.word_freq.items():
            if freq < 2:
                for index, sentence in enumerate(new_sentence_list):
                    sentence = sentence.strip()
                    new_sentence_list[index] = SPACE_RE.sub(" ",sentence.replace(word, ''))
        self.sentences = new_sentence_list
        return new_sentence_list
    
    @staticmethod
    def tokenize(string):
        tokens = word_tokenize(string)
        return tokens







