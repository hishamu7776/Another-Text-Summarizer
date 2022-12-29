import itertools
import numpy as np
from nltk.stem import WordNetLemmatizer, PorterStemmer


class Helper:    
    @staticmethod
    def vectorize_word(sentences = None, word_normalizer = "stem", fix = 1):

        root = list()
        if word_normalizer == "stem":
            stemmer = PorterStemmer()
            for sentence in sentences:
                sentence = [stemmer.stem(word) for word in sentence]
                root.append(sentence)
        elif word_normalizer == "lemma":
            lemmatizer = WordNetLemmatizer()
            for sentence in sentences:
                sentence = [lemmatizer.lemmatize(word, pos="a") for word in sentence]
                root.append(sentence)
        elif word_normalizer == "ultra":
            for sentence in sentences:
                sentence = [word[:fix] for word in sentence]
                root.append(sentence)
        elif word_normalizer == "raw":
            root = sentences
        else:
            print(
                "Error**The assigned normalizer does not support or incorrect normalizer.")
            return
        vector = set(itertools.chain.from_iterable(root))
        matrix = np.zeros([len(root),len(vector)])
        for col, item in enumerate(vector):
            for row, line in enumerate(root):
                matrix[row, col] = line.count(item)
        return vector, matrix
        
        

    @staticmethod
    def get_arguments(args):
        arg_size = len(args) 
        file_path = args[1] #Takes only csv or txt file. #../dailymail/dataset/test.csv
        file_type = args[2] #text #dataframe
        num_sentences = int(args[3]) #Number of sentences needed in summary.
        method = args[4] #freq #artex
        article_col = 'article'
        row_index = 0
        word_normalizer = 'ultra_1'
        occurance = 2
        if file_type == 'csv' and method == 'freq':
            if arg_size == 7:
                article_col = args[5] #column name
                row_index = int(args[6]) #index of article.                
            else:
                print("Error**-Wrong number of argument for this type")
                print("Example:- python main.py 'filepath/name.csv' 'dataframe' num_sentences 'freq' 'col_name' index")
        elif file_type == 'csv' and method == 'artex':
            if arg_size == 9:
                article_col = args[5] #column name
                row_index = int(args[6]) #index of article.
                word_normalizer = args[7] #stemming #lemmatizer #ultra_1 #ultra_n 
                occurance = int(args[8]) #Keep words with n occurances
            else:
                print("Error**-Wrong number of argument for this type")
                print("Example:- python main.py 'dailymail/dataset/train.csv' 'dataframe' 7 'artex' 'article' 1 'ultra_1' 2")
        elif file_type == 'text' and method == 'artex':
            if arg_size == 7:
                word_normalizer = args[5] #stemming #lemmatizer #ultra_1 #ultra_n 
                occurance = int(args[6]) #Keep words with n occurances            
            else:
                print("Error**-Wrong number of argument for this type")
                print("Example:- python main.py 'filepath/name.csv' 'dataframe' num_sentences 'freq' 'lemma' occurances")


        return file_path, file_type, num_sentences, method, article_col, row_index, word_normalizer, occurance
    


        