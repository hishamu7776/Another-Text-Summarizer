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
    