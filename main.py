import os
import sys
import nltk
import numpy as np

from nltk.corpus import stopwords
from preprocess import Preprocess
from summarizer import Summarizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

if __name__ == "__main__":
    arg_size = len(sys.argv)
    if arg_size < 5:
        print("Requires atleast 3 arguments. Following are example commands.")
    else:
        file_path = sys.argv[1] #Takes only csv or txt file. #../dailymail/dataset/test.csv
        file_type = sys.argv[2] #text #dataframe
        num_sentences = int(sys.argv[3]) #Number of sentences needed in summary.
        method = sys.argv[4] #freq #artex
        occurance = 2
        article_col = 0
        if file_type == 'csv' and method == 'freq':
            if arg_size == 7:
                article_col = sys.argv[5] #column name
                row_index = int(sys.argv[6]) #index of article.                
            else:
                print("Error**-Wrong number of argument for this type")
                print("Example:- python main.py 'filepath/name.csv' 'dataframe' num_sentences 'freq' 'col_name' index")
        elif file_type == 'csv' and method == 'artex':
            if arg_size == 9:
                article_col = sys.argv[5] #column name
                row_index = int(sys.argv[6]) #index of article.
                normalizer = sys.argv[7] #stemming #lemmatizer #ultra_1 #ultra_n 
                occurance = int(sys.argv[8]) #Keep words with n occurances
            else:
                print("Error**-Wrong number of argument for this type")
                print("Example:- python main.py 'filepath/name.csv' 'dataframe' num_sentences 'freq' 'col_name' index 'normalizer' occurances")
        elif file_type == 'text' and method == 'artex':
            if arg_size == 7:
                normalizer = sys.argv[5] #stemming #lemmatizer #ultra_1 #ultra_n 
                occurance = int(sys.argv[6]) #Keep words with n occurances            
            else:
                print("Error**-Wrong number of argument for this type")
                print("Example:- python main.py 'filepath/name.csv' 'dataframe' num_sentences 'freq' 'normalizer' occurances")

        
    
    stopwords = set(stopwords.words('english'))
    
    
    preprocess = Preprocess(path=file_path, file_type=file_type)
    preprocess.read_article(column = article_col)
    preprocess.sentence_split()
    preprocess.sentence_filter(stopwords=stopwords, occurance=occurance)
    summarizer = Summarizer(preprocess=preprocess, num_sentences = num_sentences)
    summary = summarizer.summarize()
    print(summary)
