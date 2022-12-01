import numpy as np

class Artex:
    @staticmethod
    def summeraize(matrix = None):
        P,N = matrix.shape
        a = list() #Average pseudo vector
        b= list() #Average pseudo sentence vector
        sentence_weight = list()
        for i in range(P):
            a.append(np.sum(matrix[i])/len(matrix[i]))
        for j in range(N):
            word_count = 0
            for i in range(P):
                word_count += matrix[i][j]
            b.append(word_count/P)
        print(a)
        print(b)
        return