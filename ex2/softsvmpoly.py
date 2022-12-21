import numpy
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
from tqdm import tqdm


def calc_gaussian_gram(trainX: np.array, sigma: float):
    m = trainX.shape[0]

    G = matrix(0., (m, m))
    for i in range(m):
        for j in range(m):
            G[i, j] = polynomial_kernel(trainX[i], trainX[j], sigma)

    return G

def polynomial_kernel(x1: np.array, x2: np.array, k):
    return (1 + numpy.inner(x1, x2)) ** k


class Psi:
    def __init__(self, k):
        self.k = k

    def x(self):
        

def calcG(trainX, k, l):
    m = trainX.shape[0]
    G = np.zeros((m, m))
    # g_tag = calc_gaussian_gram(trainX, k)
    for i in range(m):
        for j in range(m):
            g_i_j = polynomial_kernel(trainX[i], trainX[j], k)
            G[i][j] = g_i_j

            # G[j][i] = g_i_j

    return G


def calcu(m, d):
    # x = (1 / m) * (np.append(np.zeros(m, dtype=float), np.ones(m, dtype=float)))
    d_zeros = np.zeros(d)
    m_1_m = np.ones(m) * (1 / m)
    u = np.concatenate((d_zeros, m_1_m))
    return matrix(u)


def calcA(m, G, trainy):
    block_1_A = np.zeros((m, m))
    block_2_A = np.eye(m)
    block_3_A = np.diag(trainy) @ G
    block_4_A = np.eye(m)
    A = np.block([[block_1_A, block_2_A], [block_3_A, block_4_A]])

    return matrix(A)


def calcv(m):
    v = np.concatenate((np.zeros(m), np.ones(m)))
    return matrix(v)


def calcH(l, G, m):
    block_1_H = 2 * l * G
    block_2_H = np.zeros((m, m))
    block_3_H = np.zeros((m, m))
    block_4_H = np.zeros((m, m))
    # block_1_H += np.full(m, 1.e-5)
    H = np.block([[block_1_H, block_2_H], [block_3_H, block_4_H]])
    epsilon_eye = np.eye(2*m) * 1e-5
    H += epsilon_eye
    # H = H + helper
    # zeros = spmatrix(0., [], [], (m, m))
    # x = spdiag([(2 * l) * G, zeros])
    # # check for positive definite
    # helper_matrix = spmatrix(np.full(m, 1.e-5), range(m), range(m), (2 * m, 2 * m))
    # x = x + helper_matrix
    # H = 2 * l * G
    return matrix(H)


def predict(x, k, trainX, alpha):
    kernel_calculations = [polynomial_kernel(xj, x, k) for xj in trainX]
    kernel_calculations = np.array(kernel_calculations)
    a = alpha.reshape(alpha.shape[0])
    return int(np.sign(np.inner(a, kernel_calculations)))


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m = trainX.shape[0]
    G = calcG(trainX, k, l)
    A = calcA(m, G, trainy)
    H = calcH(l, G, m)
    u = calcu(m, m)
    v = calcv(m)

    sol = solvers.qp(H, u, -A, -v)
    alpha_zeta = sol["x"]
    alpha = np.array(alpha_zeta[:m])

    return alpha


def plot_points_from_dataset():
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    red_points = [x for i, x in enumerate(trainX) if trainy[i] == 1]
    blue_points = [x for i, x in enumerate(trainX) if trainy[i] == -1]
    red_xs = [x[0] for x in red_points]
    red_ys = [x[1] for x in red_points]
    blue_xs = [x[0] for x in blue_points]
    blue_ys = [x[1] for x in blue_points]
    plt.scatter(red_xs, red_ys, color="red", label="label = 1")
    plt.scatter(blue_xs, blue_ys, color="blue", label="label = -1")
    plt.legend()
    print("hi")


def calc_error(alpha, testX, testy, trainX, k):
    predictys = [predict(x, k, trainX, alpha) for x in testX]
    predictys = np.array(predictys)
    testy = testy.reshape((testy.shape[0], 1))
    predictys = predictys.reshape(predictys.shape[0], 1)
    total_errors = np.sum(predictys != testy)
    return total_errors / testy.shape[0]


def k_cross_validation(num_of_folds):
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    lambda_options = [1, 10, 100]
    k_options = [2, 5, 8]
    options = [(l, k) for l in lambda_options for k in k_options]
    trainX_sets = np.split(trainX, num_of_folds)
    trainy_sets = np.split(trainy, num_of_folds)
    errors = []
    avg_errors = []
    for l, k in tqdm(options):
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

            alpha = softsvmpoly(l, k, curr_trainX, curr_trainy)
            val_error = calc_error(alpha, curr_testX, curr_testy, trainX, k)
            errors.append(val_error)

        avg_error = sum(errors) / len(errors)
        avg_errors.append(avg_error)
        errors.clear()

    min_error = min(avg_errors)
    index = avg_errors.index(min_error)
    for i, (l, k) in enumerate(options):
        print(f"l={l}, k={k}, error={avg_errors[i]}")
    l, k = options[index]
    print(f"l={l}, k={k} are the best")
    alpha = softsvmpoly(l, k, trainX, trainy)
    final_error = calc_error(alpha, testX, testy, trainX, k)
    print(f"final error is {final_error}")
    return alpha


def plot_regions(alpha, k, trainX):
    grid = []
    for i in tqdm(range(-50, 50)):
        row = []
        for j in range(-50, 50):
            point = (j / 50., i / 50.)
            prediction = predict(point, k, trainX, alpha)
            if prediction == 1:
                row.append([255, 0, 0])
            else:
                row.append([0, 0, 255])
        grid.insert(0, row)

    plt.imshow(grid,  extent=[-1, 1, -1, 1])
    plt.title(f"for lambda=100 and k = {k}")
    plt.show()


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]
    # tests to make sure the output is of the intended class and shape

    w = run_svm_plot_regions()

    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


def run_svm_plot_regions():
    # load question 2 data
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    l = 100
    ks = [3, 5, 8]
    for k in ks:
        # run the softsvmpoly algorithm
        alpha = softsvmpoly(l, k, trainX, trainy)
        plot_regions(alpha, k, trainX)



def run_svm_calc_w():
    # load question 2 data
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    l = 1
    k = 5
    alpha = softsvmpoly(l, k, trainX, trainy)
    print("hi")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    # 4.a)
    # plot_points_from_dataset()
    # 4.c)
    # k_cross_validation(5)
    # here you may add any code that uses the above functions to solve question 4
    # 4.e )
    # run_svm_plot_regions()
    # 4.f)
    run_svm_calc_w()