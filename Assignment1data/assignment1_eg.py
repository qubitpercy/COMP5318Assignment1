import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy.optimize import minimize


def load_data():
    # train data
    with h5py.File('./Input/train/images_training.h5', 'r') as H:
        data_train = np.copy(H['datatrain'])
        data_train = data_train.reshape(data_train.shape[0], -1)
    with h5py.File('./Input/train/labels_training.h5', 'r') as H:
        label_train = np.copy(H['labeltrain'])
    # test data
    with h5py.File('./Input/test/images_testing.h5', 'r') as H:
        data_test = np.copy(H['datatest'])[:2000]
        data_test = data_test.reshape(data_test.shape[0], -1)
    with h5py.File('./Input/test/labels_testing_2000.h5', 'r') as H:
        label_test = np.copy(H['labeltest'])
    return data_train, label_train, data_test, label_test


def normalize_data(data, minimum, maximum):
    return (data - minimum) / (maximum - minimum)


def svd_reconstruction(x):
    n_components = 84
    U, s, Vt = np.linalg.svd(x, full_matrices=False)
    S = np.diag(s)
    x_reconstructed = U[0:U.shape[0], 0:n_components].dot(S[0:n_components, 0:n_components]).dot(
        Vt[0:n_components, 0:Vt.shape[1]])
    SSE = np.sum((x - x_reconstructed) ** 2)
    print(x.shape[1], ' ', x.shape[0])
    comp_ratio = (x.shape[1] * n_components + n_components + x.shape[0] * n_components) / (x.shape[1] * x.shape[0])

    print(s[0])
    pl.figure(figsize=(5, 5))
    pl.subplot(111)
    pl.plot(np.arange(len(s)), s)
    pl.grid()
    pl.title('Singular values distribution')
    pl.xlabel('n_components', fontsize=16)
    pl.ylabel('explained_variance', fontsize=16)
    pl.show()
    print('compression ratio is: ', comp_ratio)
    return x_reconstructed


def standardize_data(arr):
    rows, columns = arr.shape

    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)

    for column in range(columns):

        mean = np.mean(X[:, column])
        std = np.std(X[:, column])
        tempArray = np.empty(0)

        for element in X[:, column]:
            tempArray = np.append(tempArray, ((element - mean) / std))

        standardizedArray[:, column] = tempArray

    return standardizedArray

# https://blog.csdn.net/u012421852/article/details/80458350?utm_medium=distribute.pc_relevant.none-task-blog-utm_term-4&spm=1001.2101.3001.4242
class PCA():
    def __init__(self, X, n_components):
        self.X = X  # input
        self.n_components = n_components  # input
        self.centrX = []  # centralizaion
        self.C = []  # covariance
        self.U = []  # transfer matrix
        self.Z = []  # reduced dimension matrix

        self.centrX = self.centralized()
        self.C = self.calculate_covariance()
        self.U = self.U_matrix()
        self.Z = self.reduced_dimension()

    def centralized(self):
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        return centrX

    def calculate_covariance(self):
        total_sample = self.centrX.shape[0]
        covariance = np.dot(self.centrX.T, self.centrX) / (total_sample - 1)
        return covariance

    def U_matrix(self):
        """U matrix is data matrix reduce to k componets' transfer matrix"""
        eigen_value, eigen_vector = np.linalg.eig(self.C)
        index = np.argsort(-1 * eigen_value)
        UT = [eigen_vector[:, index[i]] for i in range(self.n_components)]
        U = np.transpose(UT)
        return U

    def reduced_dimension(self):
        """Z matrix is the PCA reduced dimension matrix"""
        Z = np.dot(self.X, self.U)
        return Z

    def comp_ratio(self):
        x = self.X
        n_components = self.n_components
        comp_ratio = (x.shape[1] * n_components + n_components + x.shape[0] * n_components) / (x.shape[1] * x.shape[0])
        return comp_ratio


PCA = PCA(data_train, 84)
data_PCA_self = PCA.reduced_dimension()
comp_ratio = PCA.comp_ratio()
print(data_PCA_self.shape[0])
print(data_PCA_self.shape[1])

print(data_PCA_self.shape)
print("comp_ratio:", comp_ratio)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lost(theta, x, y):
    m = x.shape[1]
    z = np.dot(x, theta.T)
    exp_z = sigmoid(z)

    the_lost = np.sum(np.multiply(y, np.log(exp_z)) + np.multiply(1 - y, np.log(1 - exp_z))) / (-m)
    # print(the_cost)

    # regularisation
    the_lambda = np.math.e ** -27
    the_lost = the_lost + 1 / 2 * the_lambda * theta.dot(theta.T)

    if np.isnan(the_lost):
        return np.inf
    return the_lost


def gradient_descent(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)  # (n, 785)
    y = np.matrix(y)  # (n, 1)

    z = np.dot(x, theta.T)  # ()
    exp_z = sigmoid(z)  # (n, 1)

    dloss = np.sum(np.multiply((exp_z - y), x), axis=0)  # (1, 785)

    return dloss


def train(x, y, num_labels):
    num_of_data = x.shape[0]
    num_of_param = x.shape[1]

    # build k classifiers, all_theta is a (10, 785) matrix
    all_theta = np.random.random((num_labels, num_of_param + 1))

    # insert an all one column as the first column
    x = np.insert(x, 0, values=np.ones(num_of_data), axis=1)

    # classify by y, separately train 10 classifiers
    for i in range(0, num_labels):
        theta = np.zeros(num_of_param + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = y_i.reshape(num_of_data, 1)

        # for improving speed performance, train 100 data each time
        size = 100
        for j in range(size, num_of_data + 1, size):
            y_j = y_i[j - size: j]
            x_j = x[j - size: j]
            # minimize a scalar function using a truncated Newton (TNC) algorithm
            theta_min = minimize(fun=lost, x0=theta, args=(x_j, y_j), method='TNC', jac=gradient_descent)
            # print(theta_min.success)
            theta = theta_min.x
        all_theta[i, :] = theta_min.x

    return all_theta


def predict(x, all_theta):
    num_of_data = x.shape[0]
    # insert an all one column as the first column, same with training process
    x = np.insert(x, 0, values=np.ones(num_of_data), axis=1)
    x = np.matrix(x)
    all_theta = np.matrix(all_theta)

    # calculate the possibility of every class
    possibilities = sigmoid(x.dot(all_theta.T))

    # choose the highest one
    result = np.argmax(possibilities, axis=1)

    return result


def show_pic(pic1, pic2):
    pic1 = pic1.reshape(28, 28)
    pic2 = pic2.reshape(28, 28)
    pl.figure(figsize=(15, 10))  # figsize=(15,10)
    pl.subplot(121)
    pl.imshow(pic1, cmap=pl.cm.gray)
    pl.title('Original image')
    pl.subplot(122)
    pl.imshow(pic2, cmap=pl.cm.gray)
    pl.title('Compressed image')
    pl.show()


if __name__ == "__main__":
    train_set_x, train_set_y, test_set_x, test_set_y = load_data()
    max_grey = np.max(train_set_x)
    min_grey = np.min(train_set_x)
    train_set_x = normalize_data(train_set_x, min_grey, max_grey)
    test_set_x = normalize_data(test_set_x, min_grey, max_grey)

    num = 30000
    train_set_x = train_set_x[:num]
    train_set_y = train_set_y[:num]
    print(train_set_x.shape)
    print(train_set_y.shape)
    print(test_set_x.shape)
    print(test_set_y.shape)

    pic_ori = train_set_x[0]
    train_set_x = svd_reconstruction(train_set_x)
    # test_set_x = svd_reconstruction(test_set_x)
    # show_pic(pic_ori, train_set_x[0])

    all_theta = train(train_set_x, train_set_y, 10)

    results = predict(test_set_x, all_theta)
    correct_rate = [1 if y_hat == y else 0 for (y_hat, y) in zip(results, test_set_y)]
    accuracy = (sum(correct_rate) / test_set_y.shape[0])
    print('accuracy = ', accuracy * 100, '%')

    # print(list(zip(test_set_y, y_pred)))
