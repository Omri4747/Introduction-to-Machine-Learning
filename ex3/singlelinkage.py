import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def update_clusters(X, clusters, dist_matrix):
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
    for _ in range(X.shape[0] - k):
        clusters = update_clusters(X, clusters, dist_matrix)

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


def calc_error(c, k):
    correct = 0
    for i in range(k):
        size = c[c == i].shape[0]
        indexes = np.where(c == i)[0]
        labels_count = [np.where((j*30 <= indexes) & ((j+1)*30 > indexes))[0].shape[0] for j in range(10)]
        common = np.argmax(labels_count)
        correct += labels_count[common]
        percentage = labels_count[common] / sum(labels_count)
        print(f"cluster {i}: size={size}, common={common},percentage={percentage:.2f}")
    print(f"correct {correct} out of 1000. error of {1 - correct/300 :.2f}")


def run_on_random():
    data = np.load('mnist_all.npz')
    data_examples = None
    for i in range(10):
        curr_data = data[f"train{i}"]
        indices = np.random.choice(range(curr_data.shape[0]), 30)
        to_concat = curr_data[indices]
        if data_examples is not None:
            data_examples = np.concatenate((data_examples, to_concat))
        else:
            data_examples = to_concat
    X = data_examples
    m, d = X.shape

    # run K-means
    k = 6
    c = singlelinkage(X, k)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

    cluster_number = 0
    clusters = np.unique(c)
    for i in clusters:
        c[c == i] = cluster_number
        cluster_number += 1

    calc_error(c, k)


def ridge_regression(X, Y, l):
    m = X.shape[0]
    w = np.linalg.inv(X @ np.transpose(X) + l * np.eye(N=m, M=m)) @ X @ Y
    return w


def run_ridge_regression():
    data = sio.loadmat("regdata.mat")
    X = data['X']
    testX = data['Xtest']
    Y = data['Y']
    testY = data['Ytest']
    sizes = list(range(10, 101))
    opt_l = []
    for size in sizes:
        min_mean_squared_error = np.inf
        curr_opt_l = -1
        for l in range(31):
            w = ridge_regression(X[:, :size], Y[:size], l)
            curr_mean_squared_error = (np.linalg.norm(w) ** 2) * l + np.linalg.norm(np.transpose(testX) @ w - testY) ** 2
            if curr_mean_squared_error < min_mean_squared_error:
                min_mean_squared_error = curr_mean_squared_error
                curr_opt_l = l
        opt_l.append(curr_opt_l)
        print(f"for size {size}: optimal lambda is {opt_l[size-10]} with mean error {min_mean_squared_error}")

    plt.plot(sizes, opt_l)
    plt.xlabel("Sample Size")
    plt.ylabel("Lambda")
    plt.title("lambda as function of sample size")
    plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    # 1.d
    # run_on_random()
    # here you may add any code that uses the above functions to solve question 2

    run_ridge_regression()