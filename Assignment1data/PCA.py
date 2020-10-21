import numpy as np
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