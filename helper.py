import itertools
import numpy as np
from nltk.stem import WordNetLemmatizer, PorterStemmer


class Helper:
    
    @staticmethod
    def normalize(sentences = None,type = "stemming", fix=1):
        base = []
        if type == "stemming":
            stemmer = PorterStemmer()
            for sentence in sentences:
                sentence = [stemmer.stem(word) for word in sentence]
                base.append(sentence)
        elif type == "lemmatizer":
            lemmatizer = WordNetLemmatizer()
            for sentence in sentences:
                sentence = [lemmatizer.lemmatize(word, pos="a") for word in sentence]
                base.append(sentence)
        elif type == "ultra-stemming":
            for sentence in sentences:
                sentence = [word[:fix] for word in sentence]
                base.append(sentence)
        else:
            print(
                "The assigned normalizer does not support or incorrect input.")
            return
        return base

    @staticmethod
    def vectorize(base):
        vector = set(itertools.chain.from_iterable(base))
        matrix = np.zeros([len(base),len(vector)])
        for col, item in enumerate(vector):
            for row, line in enumerate(base):
                matrix[row, col] = line.count(item)
        return vector, matrix
    


        