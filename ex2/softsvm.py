import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
from tqdm import tqdm


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def calcA(m, d, trainX, trainy):
    block_1_A = np.zeros((m, d))
    block_2_A = np.eye(m)
    block_3_A = np.diag(trainy) @ trainX
    block_4_A = np.eye(m)
    A = np.block([[block_1_A, block_2_A], [block_3_A, block_4_A]])

    return matrix(A)


def calcH(l, m, d):
    H = spmatrix(2 * l, range(d), range(d), (d + m, d + m))
    return H


def calcu(m, d):
    d_zeros = np.zeros(d)
    m_1_m = np.ones(m) * (1 / m)
    u = np.concatenate((d_zeros, m_1_m))
    return matrix(u)


def calcv(m):
    v = np.concatenate((np.zeros(m), np.ones(m)))
    return matrix(v)


def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m, d = trainX.shape

    u = calcu(m, d)
    v = calcv(m)
    H = calcH(l, m, d)
    A = calcA(m, d, trainX, trainy)

    sol = solvers.qp(H, u, -A, -v)
    w_zeta = sol["x"]
    w = np.array(w_zeta[:d])

    return w


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


def calc_error(w, testX, testy):
    predictys = [int(np.sign(testX[i] @ w)) for i in range(testX.shape[0])]
    predictys = np.array(predictys)
    testy = testy.reshape((testy.shape[0], 1))
    predictys = predictys.reshape(predictys.shape[0], 1)
    total_errors = np.sum(predictys != testy)
    return total_errors / testy.shape[0]


def run_softsvm(trainX, testX, trainy, testy, m, d, l):
    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(l, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    train_error_percent = calc_error(w, _trainX, _trainy)
    test_error_percent = calc_error(w, testX, testy)

    return train_error_percent, test_error_percent


def plot_results(xs, errors, line_color, error_color=None, label=None):
    if error_color is None:
        error_color = line_color
    min_errors = [min(curr) for curr in errors]
    max_errors = [max(curr) for curr in errors]
    avg_errors = [sum(curr) / len(curr) for curr in errors]
    min_distance = [a - b for a, b in zip(avg_errors, min_errors)]
    max_distance = [b - a for a, b in zip(avg_errors, max_errors)]
    error_bar = np.array([min_distance, max_distance])
    plt.xscale('log')
    plt.plot(xs, avg_errors, color=line_color, marker="o", label=label)
    plt.errorbar(xs, avg_errors, yerr=error_bar, fmt="none", ecolor=error_color)
    plt.xlabel("Lambda")
    plt.ylabel("Average Error")
    # plt.show()


def small_sample(trainX, testX, trainy, testy, m, d, l):
    train_errors = []
    test_errors = []
    for _ in tqdm(range(10)):
        train_error, test_error = run_softsvm(trainX, testX, trainy, testy, m, d, l)
        train_errors.append(train_error)
        test_errors.append(test_error)
    return train_errors, test_errors


def calc_min_max_avg_plot_results(xs, errors):
    train_errors = [a[0] for a in errors]
    test_errors = [a[1] for a in errors]
    plot_results(xs, train_errors, 'blue', label="train result: m = 100")
    plot_results(xs, test_errors, 'green', label="test result: m = 100")


def k_cross_validation(num_of_folds):
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    lambda_options = [1, 10, 100]
    trainX_sets = np.split(trainX, num_of_folds)
    trainy_sets = np.split(trainy, num_of_folds)
    errors = []
    avg_errors = []
    for l in tqdm(lambda_options):
        for i in tqdm(range(num_of_folds)):
            if i != 0 and i != num_of_folds - 1:
                curr_trainX = np.concatenate((np.concatenate(trainX_sets[:i]), np.concatenate(trainX_sets[i + 1:])))
                curr_trainy = np.concatenate((np.concatenate(trainy_sets[:i]), np.concatenate(trainy_sets[i + 1:])))
            elif i == 0:
                curr_trainX = np.concatenate(trainX_sets[i + 1:])
                curr_trainy = np.concatenate(trainy_sets[i + 1:])
            else:
                curr_trainX = np.concatenate(trainX_sets[:i])
                curr_trainy = np.concatenate(trainy_sets[:i])
            curr_testy = trainy_sets[i]
            curr_testX = trainX_sets[i]

            w = softsvm(l, curr_trainX, curr_trainy)
            print(w)
            val_error = calc_error(w, curr_testX, curr_testy)
            errors.append(val_error)

        avg_error = sum(errors) / len(errors)
        avg_errors.append(avg_error)
        print(f"errors for l={l}: {errors}")
        errors.clear()

    min_error = min(avg_errors)
    index = avg_errors.index(min_error)
    for i, l in enumerate(lambda_options):
        print(f"l={l}, error={avg_errors[i]}")
    l = lambda_options[index]
    print(f"l={l}, are the best")
    w = softsvm(l, trainX, trainy)
    final_error = calc_error(w, testX, testy)
    print(f"final error is {final_error}")
    return w



def test_sample(sample_size):
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = sample_size
    d = trainX.shape[1]
    xs = []
    errors = []
    # small sample
    for n in tqdm(range(1, 11)):
        l = 10 ** n
        xs.append(l)
        train_errors, test_errors = small_sample(trainX, testX, trainy, testy, m, d, l)
        errors.append((train_errors, test_errors))

    calc_min_max_avg_plot_results(xs, errors)

    # large sample
    m = sample_size * 10
    ns = [1, 3, 5, 8]


    train_errors = []
    test_errors = []
    xs = []
    for n in ns:
        l = 10 ** n
        xs.append(l)
        train_error, test_error = run_softsvm(trainX, testX, trainy, testy, m, d, l)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plt.scatter(xs, train_errors, color='orange', label="train results: m = 1000")
    plt.scatter(xs, test_errors, color='purple', label="test results: m = 1000")
    plt.legend()
    plt.title("Two experiments, m = 100 and m = 1000")
    plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
    # test_sample(100)

    k_cross_validation(5)