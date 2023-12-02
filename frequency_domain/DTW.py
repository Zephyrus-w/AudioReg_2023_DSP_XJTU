import numpy as np

def dtw_distance(sequence1, sequence2):
    n, m = len(sequence1), len(sequence2)
    dtw_matrix = np.zeros((n+1, m+1))

    for i in range(1, n+1):
        dtw_matrix[i, 0] = np.inf
    for i in range(1, m+1):
        dtw_matrix[0, i] = np.inf

    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(sequence1[i-1] - sequence2[j-1])
            last_min = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[n, m]
