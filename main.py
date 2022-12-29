import os
import sys
import nltk
import numpy as np

from nltk.corpus import stopwords
from preprocess import Preprocess
from summarizer import Summarizer
from helper import Helper

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

if __name__ == "__main__":
    arg_size = len(sys.argv)
    if arg_size < 5:
        print("Error**-Arguments required for this application.")
        print("Eg:- python main.py './dataset/football_article.txt' 'text' 7 'freq'")
    else:
        file_path, file_type, num_sentences, method, article_col, row_index, normalizer, occurance = Helper.get_arguments(sys.argv)        
    
        stopwords = set(stopwords.words('english'))
        
        
        preprocess = Preprocess(path=file_path, file_type=file_type)
        preprocess.read_article(column = article_col)
        preprocess.sentence_split()
        preprocess.sentence_filter(stopwords=stopwords, occurance=occurance)

        summarizer = Summarizer(summarizer=method, preprocess=preprocess, num_sentences = num_sentences)
        summary = summarizer.summarize()
    
