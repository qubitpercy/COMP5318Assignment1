import h5py
import numpy as np

def getData():

    # Get Training data
    with h5py.File('./Input/train/images_training.h5', 'r') as H:
        data_train = np.copy(H['datatrain'])
    with h5py.File('/Users/chengpeng/PycharmProjects/COMP5318Assignment1/Assignment1data/Input/train/labels_training.h5', 'r') as H:
        label_train = np.copy(H['labeltrain'])



    # Get testing data
    with h5py.File('/Users/chengpeng/PycharmProjects/COMP5318Assignment1/Assignment1data/Input/test/images_testing.h5', 'r') as H:
        data_test = np.copy(H['datatest'])[:2000, ]
    with h5py.File('/Users/chengpeng/PycharmProjects/COMP5318Assignment1/Assignment1data/Input/test/labels_testing_2000.h5', 'r') as H:
        label_test = np.copy(H['labeltest'])
    # print('\nlabel test datas shape is:',label_test.shape'\nlabel test datas shape is:',label_test.shape)
    return data_train, label_train, data_test, label_test




def splitData(data_train,label_train,ratio=0.7):
    # split data into 2 groups Training data and Validation data to do cross validation
    a = int(30000*ratio)
    x_train = data_train[:a, :]
    y_train = label_train[:a, ]
    x_validation = data_train[a:, ]
    y_validation = label_train[a:, ]
    return x_train,y_train,x_validation,y_validation

def normalize(x_train):
    normalizer = (x_train - np.min(x_train)) / np.ptp(x_train)
    return normalizer

def standarlize(x_train):
    standarlizer = (x_train - np.mean(x_train)) / np.std(x_train)
    return standarlizer

# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range
#
#
# def standardization(data):
#     mu = np.mean(data, axis=0)
#     sigma = np.std(data, axis=0)
#     print(sigma)
#     # if np.isscalar(sigma):
#     #     if sigma == .0:
#     #         return (data - mu)
#     # else:
#     #     return (data - mu) / sigma


import time
start_time = time.time()
data_train,label_train,data_test,label_test=getData()

k_svd = 10 # number of singular values to save
# Singular Value Decomposition on training images

# u, s, vh = np.linalg.svd(data_train)
# data_svd = np.array([u[i][:,:k_svd] @ np.diag(s[i][:k_svd]) @ vh[i][:k_svd:,] for i in range(s.shape[0])])
# # Singular Value Decomposition on testing images
#
# u, s, vh = np.linalg.svd(data_test)
# data_test_svd = np.array([u[i][:,:k_svd] @ np.diag(s[i][:k_svd]) @ vh[i][:k_svd:,] for i in range(s.shape[0])])
# print(data_svd.shape)
# print(data_test_svd.shape)

x_train,y_train,x_validation,y_validation = splitData(data_train,label_train)

print("--- %s seconds ---" % (time.time() - start_time))