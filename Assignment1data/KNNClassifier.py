import numpy as np
from collections import Counter

from Assignment1data.getData import *
from Assignment1data.PCA import *


class kNNClassifier:
    def __init__(self,k):
        self.k = k
        self._y_train = None
        self._x_train = None

    def fit(self,x_train, y_train):
        print('y_train.shape: ', y_train.shape)
        assert x_train.shape[0] == y_train.shape[0]
        assert self.k <= x_train.shape[0]
        print("x_train.shape: ",x_train.shape)

        self._x_train = x_train
        self._y_train = y_train
        return self

    def predict(self,x_predict):
        assert self._x_train is not None
        assert self._y_train is not None
        assert x_predict.shape[1] == self._x_train.shape[1]
        list = []

        for i in range(x_predict.shape[0]):
            # print(i)
            distances = [np.sqrt(np.sum((x_train - x_predict[i,:]) ** 2)) for x_train in self._x_train]
            # for j in range(self._x_train.shape[0]):
            #     distances = sqrt(np.sum((self._x_train[j,:] - x_predict[i,:]) ** 2))
            #     print(i,'distances is:',distances)

            # distances = np.array[np.sum(np.sqrt((x_train - x_predict)**2)) for x_train in self._x_train]
            near = np.argsort(distances)

            """x = np.array([3, 1, 2])
                np.argsort(x)
                output: array([1, 2, 0])"""
            top_k = [self._y_train[i] for i in near[:self.k]]
            votes = Counter(top_k)
            # print(i,'top_k is',top_k)
            certainClass = votes.most_common(1)[0][0]
            certainClassparsint = int(certainClass)
            list.append(certainClassparsint)
            # print('class is ', certainClassparsint)
        y_predict = np.array(list)
        return y_predict


    def accuracy(self, x_test, y_test):
        y_predict = self.predict(x_predict=x_test)
        print('y_predict is: ',y_predict)
        print('y_test is:    ',y_test)
        return sum(y_predict == y_test) / len(y_test)

    def __repr__(self):
        return "KNN(k=%d)" % self.k



import time
start_time = time.time()

data_train,label_train,data_test,label_test=getData()

x_train,y_train,x_validation,y_validation = splitData(data_train,label_train)
# x_train = standardization(x_train)
# x_train = normalization(x_train)
# x_validation = standardization(x_validation)
# x_validation = normalization(x_validation)
#PCA
n_components = 200


PCA = PCA(x_train, n_components)
PCA_training = PCA.reduced_dimension()
PCA_validation = np.dot(x_validation,PCA.U_matrix())
PCA_testing = np.dot(data_test,PCA.U_matrix())

# number =500
# PCA_training = PCA_training[:number]
# y_train = y_train[:number]
# PCA_validation = PCA_validation[:number]
# y_validation =y_validation[:number]

knn = kNNClassifier(9)
knn.fit(PCA_training,y_train)
knn.fit(PCA_validation,y_validation)
print('x_train is',x_train)
# knn.fit(PCA_validation,y_validation)

predict_label = knn.predict(PCA_validation)
accuracy = knn.accuracy(PCA_testing,label_test)
print('accuracy is:',accuracy)


with h5py.File('KNN-predicted_labels.h5','w') as H:
    H.create_dataset('label',data=predict_label)





print("--- %s seconds ---" % (time.time() - start_time))