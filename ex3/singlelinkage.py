import numpy as np
import scipy.io as sio


def update_clusters(X, clusters, dist_matrix, curr_loop):
    m = X.shape[0]
    min_dist_idx = np.unravel_index(np.argmin(dist_matrix), (m, m))
    a = min_dist_idx[0]
    b = min_dist_idx[1]
    for i in range(m):
        if i != a and i != b:
            temp = min(dist_matrix[a][i], dist_matrix[b][i])
            dist_matrix[a][i] = temp
            dist_matrix[i][a] = temp

    # 'b' cluster merged into 'a'. Set dist from 'b' cluster to all other clusters to be infinity
    dist_matrix[b, :] = np.inf
    dist_matrix[:, b] = np.inf

    clusters[clusters == b] = a

    return clusters


def singlelinkage(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m = X.shape[0]
    dist_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(i):
            dist = np.linalg.norm(X[i] - X[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
        # fill diagonal with infinity values, that way clusters would not choose itself for combining
        dist_matrix[i, i] = np.inf

    clusters = np.array(range(m))
    for curr_loop in range(X.shape[0] - k):
        clusters = update_clusters(X, clusters, dist_matrix, curr_loop)

    return clusters.reshape((clusters.shape[0], 1))


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    data0 = data['train0']
    data1 = data['train1']
    X = np.concatenate((data0[np.random.choice(data0.shape[0], 30)], data1[np.random.choice(data1.shape[0], 30)]))
    m, d = X.shape

    # run single-linkage
    c = singlelinkage(X, k=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
