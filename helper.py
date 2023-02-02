import itertools
import os
import numpy as np
from nltk.stem import WordNetLemmatizer, PorterStemmer
from evaluator import Evaluator
from plots import Plots

class Helper: 
    @staticmethod
    def read_data(path):
        text_path, summary_path = os.listdir(path)
        text_path = os.path.join(path,text_path)
        summary_path = os.path.join(path,summary_path)
        texts,summaries = [],[]
        for root, dirs, files in os.walk(path):
            if len(dirs) == 0:
                if text_path in root:
                    for file in files:
                        f = open(os.path.join(root,file), "r",encoding = 'ISO-8859-1')
                        texts.append(f.read())
                if summary_path in root:
                    for file in files:
                        f = open(os.path.join(root,file), "r",encoding = 'ISO-8859-1')
                        summaries.append(f.read())
        return texts, summaries

    @staticmethod
    def read_file(text_path, summary_path):
        text_file = open(text_path, "r",encoding = 'ISO-8859-1')
        text = text_file.read()
        summary_file = open(summary_path, "r",encoding = 'ISO-8859-1')
        summary = summary_file.read()
        return text, summary

    @staticmethod
    def vectorize_word(sentences = None, word_normalizer = "stem", fix = 1):
        Evaluator.mantal(sentences)
        root = list()
        if word_normalizer == "stem":
            TITLE = 'Stemming'
            stemmer = PorterStemmer()
            for sentence in sentences:
                sentence = [stemmer.stem(word) for word in sentence]
                root.append(sentence)
        elif word_normalizer == "lemma":
            TITLE = "Lemmatization"
            lemmatizer = WordNetLemmatizer()
            for sentence in sentences:
                sentence = [lemmatizer.lemmatize(word, pos="a") for word in sentence]
                root.append(sentence)
        elif word_normalizer == "ultra":
            TITLE = f"Ultra Stemming Fix {fix}"
            for sentence in sentences:
                sentence = [word[:fix] for word in sentence]
                root.append(sentence)
        elif word_normalizer == "raw":
            TITLE = "Raw Text"
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
        
        #matrix_density = Evaluator.matrix_density(matrix)
        #Plots.plot_density(matrix=matrix_density, x_tick=vector, y_tick=np.arange(0,matrix_density.shape[0],1),title=TITLE,labels=['density', 'units'])
        return vector, matrix
        
    


        