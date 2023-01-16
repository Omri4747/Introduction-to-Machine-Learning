import numpy as np


def random_centroids(X, k):
    indexes = np.random.choice(range(X.shape[0]), size=k)
    # centroids = [X[i] for i in indexes]
    return X[indexes]


def update_clusters(X, k, centroids: np.ndarray):
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
    centroids: np.ndarray = random_centroids(X, k)
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


def calc_error(c, k):
    correct = 0
    for i in range(k):
        size = c[c == i].shape[0]
        indexes = np.where(c == i)[0]
        labels_count = [np.where((j*100 <= indexes) & ((j+1)*100 > indexes))[0].shape[0] for j in range(10)]
        common = np.argmax(labels_count)
        correct += labels_count[common]
        percentage = labels_count[common] / sum(labels_count)
        print(f"cluster {i}: size={size}, common={common},percentage={percentage:.2f}")
    print(f"correct {correct} out of 1000. error of {1 - correct/1000 :.2f}")


def run_on_random():
    data = np.load('mnist_all.npz')
    data_sets = []
    data_examples = None
    for i in range(10):
        curr_data = data[f"train{i}"]
        indices = np.random.choice(range(curr_data.shape[0]), 100)
        to_concat = curr_data[indices]
        if data_examples is not None:
            data_examples = np.concatenate((data_examples, to_concat))
        else:
            data_examples = to_concat
    # indices = np.random.choice(data_examples.shape[0], 1000)
    # X = data_examples[indices]
    X = data_examples
    m, d = X.shape

    # run K-means
    k = 10
    c = kmeans(X, k, t=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

    calc_error(c, k)




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
    # simple_test()

    # 1.b
    run_on_random()
    # here you may add any code that uses the above functions to solve question 2
