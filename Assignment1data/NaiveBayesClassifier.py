'''https://blog.csdn.net/leida_wt/article/details/84977524?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522160291483819725255544587%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=160291483819725255544587&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-3-84977524.pc_first_rank_v2_rank_v28&utm_term=%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF+numpy&spm=1018.2118.3001.4187'''
from numpy import *

from Assignment1data.getData import *
from Assignment1data.PCA import *


class NaiveBayesClassifier:
    def __init__(self):
        self.P_x_c_fun = None  # P(X|C)
        self.class_num = None
        self.P_c = None  # P(C) probability of C occurence

    def fit(self, x_train, y_train):
        # Calculate P(c)
        n = x_train.shape[0]  # n samples in x_train
        class_num = len(np.unique(y_train))  # number of total class
        print('number of classifier is: ', class_num)
        self.class_num = class_num
        P_c = [sum(y_train == i) / n for i in range(class_num)]
        print('P(X|C) is ', P_c)
        P_c = np.array(P_c).reshape(1, -1)  # 1*class_num
        self.P_c = P_c
        # calculate P(x|c)
        P_x_c_fun = []
        # Gaussian distribution parameter calculation，& provide statistic function
        list = []
        for i in range(class_num):
            data = x_train[(y_train == i)]
            mu = data.mean(axis=0)  # (nfeature,)
            sigma = np.cov(data.T)  # (nfeature,nfeature)
            sigma_determinants = (np.linalg.det(sigma))
            sigma_inv = (np.linalg.inv(sigma))
            first_part = (1 / (((2 * np.pi) ** (class_num / 2)) * (sigma_determinants ** 0.5))) * 10 ** 6

            def gaussian(mu, sigma_inv, first_part):
                def fun(x):
                    nonlocal mu, first_part, sigma_inv
                    exponent = float(-0.5 * ((x - mu).T).dot(sigma_inv).dot(x - mu))
                    rest = float(first_part * (np.exp(exponent)))
                    return rest

                return fun

            P_x_c_fun.append(gaussian(mu, sigma_inv, first_part))
        self.P_x_c_fun = P_x_c_fun

        P_x_c = np.empty((n, class_num))
        for i in range(n):
            for j in range(class_num):
                P_x_c[i, j] = P_x_c_fun[j](x_train[i])
        # calculate P(c|x)
        P_above = P_x_c * P_c
        PP = P_above / (1+P_above.sum(axis=1, keepdims=True))
        first_class = np.argmax(PP, axis=1)
        return first_class

    def predict(self, test_x):
        n = len(test_x)
        P_x_c_fun = self.P_x_c_fun
        P_x_c = np.empty((n, self.class_num))
        for i in range(n):
            for j in range(self.class_num):
                P_x_c[i, j] = P_x_c_fun[j](test_x[i])
        P_above = P_x_c * self.P_c
        PP = P_above +1/ (1+P_above.sum(axis=1, keepdims=True))
        first_class = np.argmax(PP, axis=1)
        return first_class

    def accuracy(self, x_test, y_test):
        y_predict = self.predict(test_x=x_test)
        print('y_predict is: ', y_predict)
        print('y_test is:    ', y_test)
        return sum(y_predict == y_test) / len(y_test)
# if __name__ == '__main__':
#     NB = NaiveBayesClassifier()
#     NB.test()
data_train,label_train,data_test,label_test=getData()
# data_train = normalize(data_train)
# data_train = standarlize(data_train)


x_train,y_train,x_validation,y_validation = splitData(data_train,label_train)


start_time = time.time()
data_train,label_train,data_test,label_test=getData()
data_train = normalize(data_train)
data_train = standarlize(data_train)


x_train,y_train,x_validation,y_validation = splitData(data_train,label_train)


start_time = time.time()
n_components = 84

#PCA
PCA = PCA(x_train, n_components)
PCA_training = PCA.reduced_dimension()
PCA_validation = np.dot(x_validation,PCA.U_matrix())
PCA_testing = np.dot(data_test,PCA.U_matrix())

#NBClassifier
NB = NaiveBayesClassifier()
res=NB.fit(PCA_training,y_train)
y_predict = NB.predict(PCA_validation)
accuracy = NB.accuracy(y_predict,y_validation)

predict = NB.predict(PCA_testing)
print('accuracy is: ',accuracy)
print("--- %s seconds ---" % (time.time() - start_time))
