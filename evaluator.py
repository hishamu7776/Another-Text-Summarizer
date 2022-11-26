import numpy as np

class Evaluator:
    @staticmethod
    def matrix_density(matrix):
        P, N = matrix.shape
        size = P*N
        density_matrix = np.zeros([P,N])
        for idx,S in enumerate(matrix):
            for jdx, cw in enumerate(S):
                if cw > 0:
                    density_matrix[idx][jdx] = cw/size
        return density_matrix

    @staticmethod
    def compute_average_density(matrix):
        return np.sum(np.array(matrix).flatten())/np.prod(matrix.shape)

    @staticmethod
    def compute_volume(A = None, B = None):
        return np.prod(B.shape)*100/np.prod(A.shape)
             
    @staticmethod
    def word_length_distribution(matrix):
        word_lengths = dict()
        for arr in matrix:
            for word in arr:
                if len(word) in word_lengths.keys():
                    word_lengths[len(word)] = word_lengths[len(word)]+1
                else:
                    word_lengths[len(word)] = 1
        return word_lengths
    