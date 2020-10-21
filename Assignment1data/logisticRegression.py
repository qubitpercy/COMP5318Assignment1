import matplot as pl
from scipy.optimize import minimize
import matplotlib.pyplot as pl
from Assignment1data.PCA import *

from Assignment1data.getData import getData, splitData, normalize


# def normalize(x_train):
#     normalizer = (x_train - np.min(x_train)) / np.ptp(x_train)
#     return normalizer
#
#
# def svd_reconstruction(x):
#     n_components = 84
#     U, s, Vt = np.linalg.svd(x, full_matrices=False)
#     S = np.diag(s)
#     x_reconstructed = U[0:U.shape[0], 0:n_components].dot(S[0:n_components, 0:n_components]).dot(
#         Vt[0:n_components, 0:Vt.shape[1]])
#     SSE = np.sum((x - x_reconstructed) ** 2)
#     print(x.shape[1], ' ', x.shape[0])
#     comp_ratio = (x.shape[1] * n_components + n_components + x.shape[0] * n_components) / (x.shape[1] * x.shape[0])
#
#     print(s[0])
#     pl.figure(figsize=(15, 10))
#     pl.subplot(111)
#     pl.plot(np.arange(len(s)), s)
#     pl.grid()
#     pl.title('Singular values distribution')
#     pl.xlabel('n')
#     pl.show()
#
#     return x_reconstructed
#
#
# def sigmoid(e):
#     return 1 / (1 + np.exp(-e))
#
#
# def calculate_lost(theta, x, y):
#     m = x.shape[1]
#     z = np.dot(x, theta.T)
#     exp_z = sigmoid(z)
#
#     the_lost = np.sum(np.multiply(y, np.log(exp_z)) + np.multiply(1 - y, np.log(1 - exp_z))) / (-m)
#     # print(the_cost)
#
#     # regularisation
#     the_lambda = np.math.e ** -27
#     the_lost = the_lost + 1 / 2 * the_lambda * theta.dot(theta.T)
#
#     if np.isnan(the_lost):
#         return np.inf
#     return the_lost
#
#
# def gradient_descent(theta, x_train, y_label):
#     x_train = np.matrix(x_train)  # (n, 785)
#     y_label = np.matrix(y_label)  # (n, 1)
#     theta = np.matrix(theta)
#
#     m = np.dot(x_train, theta.T)  # ()
#     exp_m = sigmoid(m)  # (n, 1)
#     dataloss = np.sum(np.multiply((exp_m - y_label), x_train), axis=0)  # (1, 785)
#     return dataloss
#
#
# def train(x, y, num_labels):
#     num_of_data = x.shape[0]
#     num_of_param = x.shape[1]
#
#     # build k classifiers, all_theta is a (10, 785) matrix
#     all_theta = np.random.random((num_labels, num_of_param + 1))
#
#     # insert an all one column as the first column
#     x = np.insert(x, 0, values=np.ones(num_of_data), axis=1)
#
#     # classify by y, separately train 10 classifiers
#     for i in range(0, num_labels):
#         theta = np.zeros(num_of_param + 1)
#         y_i = np.array([1 if label == i else 0 for label in y])
#         y_i = y_i.reshape(num_of_data, 1)
#
#         # for improving speed performance, train 100 data each time
#         size = 100
#         for j in range(size, num_of_data + 1, size):
#             y_j = y_i[j - size: j]
#             x_j = x[j - size: j]
#             # minimize a scalar function using a truncated Newton (TNC) algorithm
#             theta_min = minimize(fun=calculate_lost, x0=theta, args=(x_j, y_j), method='TNC', jac=gradient_descent)
#             # print(theta_min.success)
#             theta = theta_min.x
#         all_theta[i, :] = theta_min.x
#
#     return all_theta
#
#
# def predict(x, all_theta):
#     num_of_data = x.shape[0]
#     # insert an all one column as the first column, same with training process
#     x = np.insert(x, 0, values=np.ones(num_of_data), axis=1)
#     x = np.matrix(x)
#     all_theta = np.matrix(all_theta)
#
#     # possibility of each class
#     possibilities = sigmoid(x.dot(all_theta.T))
#     # choose the highest one
#     result = np.argmax(possibilities, axis=1)
#     return result
#
#
# def show_pic(pic1, pic2):
#     pic1 = pic1.reshape(28, 28)
#     pic2 = pic2.reshape(28, 28)
#     pl.figure(figsize=(15, 10))  # figsize=(15,10)
#     pl.subplot(121)
#     pl.imshow(pic1, cmap=pl.cm.gray)
#     pl.title('Original image')
#     pl.subplot(122)
#     pl.imshow(pic2, cmap=pl.cm.gray)
#     pl.title('Compressed image')
#     pl.show()


class Logistic_Regression:
    def __init__(self):
        self.x_train = None
        self.y_label = None

    def sigmoid(self,e):
        return 1 / (1 + np.exp(-e))

    def calculate_lost(self, theta, x_train, y_label):
        m = x_train.shape[1]
        z = np.dot(x_train, theta.T)
        exp_z = self.sigmoid(z)
        np.seterr(divide='ignore', invalid='ignore')
        lost = np.sum(np.multiply(y_label, np.log(exp_z)) + np.multiply(1 - y_label, np.log(1 - exp_z))) / (-m)
        # print(the_cost)
        # regularisation
        lmd = np.math.e ** -27
        lost = lost + 1 / 2 * lmd * theta.dot(theta.T)
        if np.isnan(lost):
            return np.inf
        return lost

    def gradient_descent(self, theta, x_train, y_label):
        x_train = np.matrix(x_train)  # (n, 785)
        y_label = np.matrix(y_label)  # (n, 1)
        theta = np.matrix(theta)
        m = np.dot(x_train, theta.T)  # ()
        exp_m = self.sigmoid(m)  # (n, 1)
        dataloss = np.sum(np.multiply((exp_m - y_label), x_train), axis=0)  # (1, 785)
        return dataloss

    def fit(self, x_train, y_train):
        different_class = 10
        number_of_data = x_train.shape[0]
        number = x_train.shape[1]

        # build n classifiers, all_theta is a (10, 785) matrix
        the_theta = np.random.random((different_class, number + 1))

        # insert an all one column as the first column
        x_train = np.insert(x_train, 0, values=np.ones(number_of_data), axis=1)

        # classify by y, separately train 10 classifiers
        for i in range(0, different_class):
            theta = np.zeros(number + 1)
            y_i = np.array([1 if label == i else 0 for label in y_train])
            y_i = y_i.reshape(number_of_data, 1)

            # train 200 data each time to improve the efficiency
            size = 200
            for j in range(size, number_of_data + 1, size):
                y_j = y_i[j - size: j]
                x_j = x_train[j - size: j]
                # minimize a scalar function using a truncated Newton (TNC) algorithm
                theta_min = minimize(fun=self.calculate_lost, x0=theta, args=(x_j, y_j), method='TNC', jac=self.gradient_descent)
                # print(theta_min.success)
                theta = theta_min.x
            the_theta[i, :] = theta_min.x
        return the_theta

    def predict(self, x, the_theta):
        num_of_data = x.shape[0]
        # insert an all one column as the first column, same with training process
        x = np.insert(x, 0, values=np.ones(num_of_data), axis=1)
        x = np.matrix(x)
        the_theta = np.matrix(the_theta)
        # possibility of each class
        print('x shape is', x.shape, the_theta.T.shape)
        possibilities = self.sigmoid(x.dot(the_theta.T))
        # choose the highest one
        predict_label = np.argmax(possibilities, axis=1)
        return predict_label

    def calculate_accuracy(self,predict_result, correct_label):
        correct_rate = [1 if y_hat == y else 0 for (y_hat, y) in zip(predict_result, correct_label)]
        accuracy = (sum(correct_rate) / y_validation.shape[0])
        print('accuracy = ', accuracy * 100, '%')



import time

start_time = time.time()
# x_train,y_train,x_validation,y_validation = splitData(data_train,label_train)
if __name__ == "__main__":
    data_train, label_train, data_test, label_test = getData()
    x_train, y_train, x_validation, y_validation = splitData(data_train, label_train)
    x_train, y_train, x_validation, y_validation = getData()
    #normalize data
    x_train = normalize(x_train)
    x_validation = normalize(x_validation)
    data_test = normalize(data_test)


    n_components = 84
    PCA = PCA(x_train, n_components)
    PCA_training = PCA.reduced_dimension()
    PCA_validation = np.dot(x_validation, PCA.U_matrix())
    PCA_testing = np.dot(data_test, PCA.U_matrix())

    # number = 2000
    # PCA_training = PCA_training[:number]
    # y_train = y_train[:number]
    # PCA_validation = PCA_validation[:number]
    # y_validation = y_validation[:number]

    lgr = Logistic_Regression()
    # all_theta = lgr.fit(x_train,y_train)
    all_theta = lgr.fit(x_validation,y_validation)
    all_theta = lgr.fit(x_train, y_train)


    predict_label = lgr.predict(data_test,all_theta)
    lgr.calculate_accuracy(predict_label,y_validation)







    print("--- %s seconds ---" % (time.time() - start_time))
