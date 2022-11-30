import numpy as np
from scipy import stats
from plots import Plots

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

    @staticmethod
    def mantel_test(s_prime = None, s = None):
        _, N_ = s_prime.shape
        row, N = s.shape
        if True:
            A, B = Evaluator.create_square_matrix(s_prime,s)
            temp_A = A.copy()
            cor_coef, p_val = Evaluator.compute_pearson_correlation(A,B)
            cor_coef_list = list()
            cor_coef_list.append(cor_coef)
            p = 0
            for i in range(20):
                np.random.shuffle(temp_A)
                cor_coef, p_val = Evaluator.compute_pearson_correlation(temp_A,B)
                cor_coef_list.append(cor_coef)
            return max(cor_coef_list)

    @staticmethod
    def create_square_matrix(matrix1=None, matrix2 = None):
        if matrix1 is not None and matrix2 is not None:
            matrix1 = np.matmul(matrix1,matrix1.T)
            matrix2 = np.matmul(matrix2,matrix2.T)
            return matrix1,matrix2
        elif matrix1 is not None:
            matrix1 = np.matmul(matrix1,matrix1.T)
            return matrix1
        elif matrix2 is not None:
            matrix2 = np.matmul(matrix2,matrix2.T)
            return matrix2
        else:
            print("Wrong parameter")
            return

    @staticmethod
    def compute_pearson_correlation(A,B):
        
        N, M = A.shape
        A_vec = list(A[np.triu_indices(N, k = 1)])
        B_vec = list(B[np.triu_indices(N, k = 1)])
        P = len(A_vec)
        return stats.pearsonr(A_vec, B_vec)

    @staticmethod
    def compute_confidence(stats):
        return (np.array(stats)>0).sum()/len(stats)

    @staticmethod
    def cross_correlate_all(matrices=None,names=None):
        cor_mat = np.zeros([len(matrices),len(matrices)])
        for i in range(len(matrices)):
            for j in range(len(matrices)):
                cor_mat[i,j] = Evaluator.mantel_test(s_prime=matrices[i],s=matrices[j])
        Plots.plot_density(matrix=cor_mat, x_tick=names, y_tick= names, title="Correlation", labels=["methods","methods"], annot=True)
        return