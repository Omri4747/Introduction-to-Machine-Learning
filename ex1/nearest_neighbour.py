import numpy as np
from scipy.spatial import distance
from typing import List, TypeVar

ExampleType = TypeVar("ExampleType")
LabelType = TypeVar("LabelType")


class KNearestNeighbor:
    def __init__(self, examples: List[ExampleType], labels: List[LabelType], k: int):
        self.examples = examples
        self.labels = labels
        self.k = k
        self.example_label_dict = {}

    def find_k_closest_neighbor(self, x: ExampleType) -> List[ExampleType]:
        """

        :param x: data point, not necessary from examples
        :return: kth closest neighbors by euclidean distance
        """
        distances = []
        for neighbor in self.examples:
            curr_distance = distance.euclidean(neighbor, x)
            distances.append((curr_distance, neighbor))
        distances.sort(key=lambda a: a[0])
        k_distances = distances[:self.k]
        k_neighbors = [a[1] for a in k_distances]
        return k_neighbors

    def get_label(self, example: ExampleType) -> LabelType:
        f"""
        takes example that is in {self.examples} and returns its label
        :param example: {example} to check
        :return: the label of example
        """
        index = 0
        for optional in self.examples:
            if np.array_equal(optional, example):
                break
            index += 1
        if index == len(self.examples):
            raise ValueError("Was given a data point that is not in examples")
        return self.labels[index]

    def find_k_nearest_neighbors_label(self, x: ExampleType) -> LabelType:
        k_neighbors = self.find_k_closest_neighbor(x)
        labels_count = {}
        for example in k_neighbors:
            label = self.get_label(example)
            if label in labels_count.keys():
                labels_count[label] = labels_count[label] + 1
            else:
                labels_count[label] = 1
        return max(labels_count)


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


# todo: complete the following functions, you may add auxiliary functions or define class to help you


def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """

    classifier = KNearestNeighbor(x_train, y_train, k)
    return classifier


def predictknn(classifier: KNearestNeighbor, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    labels = []
    for test in x_test:
        label = classifier.find_k_nearest_neighbors_label(test)
        labels.append(label)
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


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
