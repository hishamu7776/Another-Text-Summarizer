from multiprocessing.resource_sharer import stop
import re
from turtle import clear
import pandas as pd
import numpy as np

from string import punctuation
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.probability import FreqDist




class Preprocessing:
    def __init__(self):
        self.text = ""
    


    @staticmethod
    def read_data(path):
        corpus = pd.read_csv(path)
        return corpus
    
    @staticmethod
    def get_one_item(index = 0, column= "text", corpus = None):
        if corpus is None:
            print("Please specify the corpus.")
            return
        item = corpus[column][index]
        return item

    @staticmethod
    def prepare_text(doc,stopwords):
        
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        NEW_LINE = re.compile('\n')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        doc = doc.lower()            
        doc = NEW_LINE.sub(' ',doc)
        word_freq = {}
        #RULE BASED SENTENCE SPLITTING
        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', doc)
        texts = []
        
        #lower case and clear punctuation, remove stopwords
        for text in sentences:            
            text = REPLACE_BY_SPACE_RE.sub(" ",text)
            text = BAD_SYMBOLS_RE.sub("",text)
            sentence = ""
            for word in text.split():
                count = doc.count(word)
                word_freq[word] = count
                if word not in stopwords and count > 1:
                    sentence = sentence+" "+word
            #text = ' '.join([word for word in text.split() if word not in stopwords])            
            texts.append(sentence)
        return texts




    