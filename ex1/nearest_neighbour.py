import numpy as np
from numpy import ndarray
from scipy.spatial import distance
from matplotlib import pyplot as plt
import tqdm
from typing import List, TypeVar
import random

ExampleType = TypeVar("ExampleType")
LabelType = TypeVar("LabelType")


class KNearestNeighborClassifier:
    def __init__(self, examples: List[ExampleType], labels: np.ndarray, k: int):
        self.examples = examples
        self.labels = labels.astype(int)
        self.k = k

    def find_k_closest_labels(self, x: ExampleType) -> List[LabelType]:
        f"""

        :param x: data point, not necessary from {self.examples}
        :return: kth closest neighbors by euclidean distance from {self.examples}
        """
        # tuple (distance, index)
        distances = [(distance.euclidean(neighbor, x), index) for index, neighbor in enumerate(self.examples)]
        # sort by distance
        distances.sort(key=lambda a: a[0])
        # get k nearest distances
        k_distances = distances[:self.k]
        # get k nearest labels by the index of them
        k_neighbors_labels = [self.labels[a[1]] for a in k_distances]
        return k_neighbors_labels

    def find_k_nearest_neighbors_label(self, x: ExampleType) -> LabelType:
        k_neighbors_labels = np.array(self.find_k_closest_labels(x))
        return np.bincount(k_neighbors_labels).argmax()


def choose_other(curr, l: list):
    f"""
    chooses element from list {l} that is different than {curr}
    :param curr: 
    :param l: 
    :return: 
    """
    if curr in l:
        l = [elem for elem in l if elem != curr] # new list without the current element
    return random.choice(l)

def gensmallm_corrupted(x_list: List[ExampleType], y_list: List[LabelType], m: int, rate: float = 0.15):
    f"""
    generates a random corrupted with rate of {rate}
    :param x_list:
    :param y_list:
    :param m:
    :return:
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]
    rearranged_x = rearranged_x[:m]
    rearranged_y = rearranged_y[:m]
    for i in range(len(rearranged_y)):
        if random.uniform(1,100) > rate*100:
            continue
        label = rearranged_y[i]
        new_label = choose_other(label, y_list)
        rearranged_y[i] = new_label
    return rearranged_x, rearranged_y

def gensmallm(x_list: List[ExampleType], y_list: List[LabelType], m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]




def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """

    classifier = KNearestNeighborClassifier(x_train, y_train, k)
    return classifier


def predictknn(classifier: KNearestNeighborClassifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    labels = [classifier.find_k_nearest_neighbors_label(test) for test in x_test]
    return np.array([labels]).transpose()


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")
    for i in range(10):
        print(f"The {i}'th test sample was classified as {preds[i]}")


def run_knn_mnist_sample_size(k: int, training_sample_size: int, x_test, y_test, trains, labels) -> ndarray:
    x_train, y_train = gensmallm(trains, labels, training_sample_size)

    classifer = learnknn(k, x_train, y_train)

    preds = predictknn(classifer, x_test)
    y_test = y_test.astype(int).reshape(preds.shape)
    error = np.mean(y_test != preds)

    return error

def run_knn_mnist_sample_size_corrupted(k: int, training_sample_size: int, x_test, y_test, trains, labels) -> ndarray:
    x_train, y_train = gensmallm_corrupted(trains, labels, training_sample_size)

    classifer = learnknn(k, x_train, y_train)

    preds = predictknn(classifer, x_test)
    y_test = y_test.astype(int).reshape(preds.shape)
    error = np.mean(y_test != preds)

    return error

def run_knn_mnist_k1():
    """
    run knn with k=1
    :return:
    """
    data = np.load("mnist_all.npz")

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    to_train = [train2, train3, train5, train6]
    labels = [2, 3, 5, 6]

    test_length = test2.shape[0] + test3.shape[0] + test5.shape[0] + test6.shape[0]

    x_test, y_test = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], test_length)
    k = 1
    each_sample_repetition = 10
    sample_sizes = []
    avg_errors = []
    errors_per_run = {}
    min_errors = []
    max_errors = []
    for training_sample_size in range(1, 101, 10):
        sample_sizes.append(training_sample_size)
        errors_per_run[training_sample_size] = []
        for _ in range(each_sample_repetition):
            error = run_knn_mnist_sample_size(k, training_sample_size, x_test, y_test, to_train, labels)
            errors_per_run[training_sample_size].append(error)
        avg_errors.append(sum(errors_per_run[training_sample_size]) / each_sample_repetition)
        min_errors.append(min(errors_per_run[training_sample_size]))
        max_errors.append(max(errors_per_run[training_sample_size]))

    min_distance = [a - b for a, b in zip(avg_errors, min_errors)]
    max_distance = [b - a for a, b in zip(avg_errors, max_errors)]
    error_bar = np.array([min_distance, max_distance])
    plt.plot(sample_sizes, avg_errors, color='blue', marker="o")
    plt.errorbar(sample_sizes, avg_errors, yerr=error_bar, fmt="none", ecolor='red')
    plt.xlabel("Training Sample Size")
    plt.ylabel("Average Error")
    plt.title(f"Average Error of KNearestNeighbors with k = {k}")
    plt.show()


def run_knn_mnist(training_sample_size, k_min, k_max):
    data = np.load("mnist_all.npz")

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    to_train = [train2, train3, train5, train6]
    labels = [2, 3, 5, 6]

    test_length = test2.shape[0] + test3.shape[0] + test5.shape[0] + test6.shape[0]

    x_test, y_test = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], test_length)
    each_sample_repetition = 10
    avg_errors = []
    errors_per_run = {}
    min_errors = []
    max_errors = []

    for k in tqdm.tqdm(range(k_min, k_max+1)):
        errors_per_run[k] = []
        for _ in tqdm.tqdm(range(each_sample_repetition)):
            error = run_knn_mnist_sample_size(k, training_sample_size, x_test, y_test, to_train, labels)
            errors_per_run[k].append(error)
        avg_errors.append(sum(errors_per_run[k]) / each_sample_repetition)
        min_errors.append(min(errors_per_run[k]))
        max_errors.append(max(errors_per_run[k]))

    min_distance = [a - b for a, b in zip(avg_errors, min_errors)]
    max_distance = [b - a for a, b in zip(avg_errors, max_errors)]
    error_bar = np.array([min_distance, max_distance])
    ks = list(range(k_min, k_max+1))
    plt.plot(ks, avg_errors, color='blue', marker="o")
    plt.errorbar(ks, avg_errors, yerr=error_bar, fmt="none", ecolor='red')
    plt.xlabel("k")
    plt.ylabel("Average Error")
    plt.title(f"Average Error of KNearestNeighbors with Sample Size = {training_sample_size}")
    plt.show()


def run_knn_mnist_corrupted(training_sample_size, k_min, k_max):
    data = np.load("mnist_all.npz")

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    to_train = [train2, train3, train5, train6]
    labels = [2, 3, 5, 6]

    test_length = test2.shape[0] + test3.shape[0] + test5.shape[0] + test6.shape[0]

    x_test, y_test = gensmallm_corrupted([test2, test3, test5, test6], [2, 3, 5, 6], test_length)
    each_sample_repetition = 10
    avg_errors = []
    errors_per_run = {}
    min_errors = []
    max_errors = []

    for k in tqdm.tqdm(range(k_min, k_max+1)):
        errors_per_run[k] = []
        for _ in tqdm.tqdm(range(each_sample_repetition)):
            error = run_knn_mnist_sample_size(k, training_sample_size, x_test, y_test, to_train, labels)
            errors_per_run[k].append(error)
        avg_errors.append(sum(errors_per_run[k]) / each_sample_repetition)
        min_errors.append(min(errors_per_run[k]))
        max_errors.append(max(errors_per_run[k]))

    min_distance = [a - b for a, b in zip(avg_errors, min_errors)]
    max_distance = [b - a for a, b in zip(avg_errors, max_errors)]
    error_bar = np.array([min_distance, max_distance])
    ks = list(range(k_min, k_max+1))
    plt.plot(ks, avg_errors, color='blue', marker="o")
    plt.errorbar(ks, avg_errors, yerr=error_bar, fmt="none", ecolor='red')
    plt.xlabel("k")
    plt.ylabel("Average Error")
    plt.title(f"Average Error of KNearestNeighbors with Sample Size = {training_sample_size}")
    plt.show()

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()

    # 2.a
    # run_knn_mnist_k1()

    # 2.e
    # sample_size = 200
    # k_min = 1
    # k_max = 11
    # run_knn_mnist(sample_size, k_min, k_max)

    # 2.f
    sample_size = 200
    k_min = 1
    k_max = 11
    run_knn_mnist_corrupted(sample_size, k_min, k_max)
