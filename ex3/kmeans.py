import numpy as np


def random_centrodis(X, k):
    indexes = np.random.choice(range(X.shape[0]), size=k)
    # centroids = [X[i] for i in indexes]
    return X[indexes]


def update_clusters(X, k, centroids):
    distances = np.zeros((X.shape[0], k))
    for i in range(k):
        distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
    clusters = np.argmin(distances, axis=1)
    return clusters


def update_centroids(X, k, clusters):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_i = X[clusters == i]
        new_centroids[i] = np.mean(cluster_i, axis=0)

    return new_centroids


def kmeans(X, k, t):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    centroids: np.ndarray = random_centrodis(X, k)
    clusters = None
    for _ in range(t):
        prev = clusters
        clusters = update_clusters(X, k, centroids)
        centroids = update_centroids(X, k, clusters)
        if prev is None:
            continue
        is_same = prev == clusters
        if is_same.all():
            break

    return clusters.reshape((clusters.shape[0], 1))


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run K-means
    c = kmeans(X, k=10, t=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
