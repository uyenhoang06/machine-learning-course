import os
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.decomposition import PCA
import gzip

eps = 0.1  # small epsilon value


class GaussianMixtureModel:
    def __init__(self, num_components=5, max_iter=500, tol=1e-5):
        # khởi tạo GMM với các tham số
        self.num_components = num_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, d = X.shape
        self.log_likelihood_loss = {}
        log_r_matrix = np.eye(self.num_components)
        log_r_matrix = log_r_matrix[np.random.choice(
            self.num_components, size=n)].T
        iter_num = 0

        for it in range(self.max_iter):
            # M-Step
            r_k = np.sum(log_r_matrix, axis=1) + \
                (10 * np.finfo(float).eps)
            pi = r_k / n
            means = log_r_matrix @ X / r_k[:, np.newaxis]  # (K, d)
            variances = log_r_matrix @ (X ** 2) / r_k[:, np.newaxis]
            variances -= means ** 2
            variances += eps  # Add epsilon to variances

            # E-Step
            sum_log_r, log_r_matrix = self._multivariate_gaussian(
                X, pi, variances, means)
            log_r_matrix = np.exp(log_r_matrix - sum_log_r)

            # compute loss
            self.log_likelihood_loss[it] = -np.sum(sum_log_r)
            loss_difference = 0
            if it > 1:
                loss_difference = np.abs(self.log_likelihood_loss[it] -
                                         self.log_likelihood_loss[it - 1]) / (np.abs(self.log_likelihood_loss[it]) + eps)
                if loss_difference <= self.tol:
                    iter_num = it
                    break

        print("EM for GMM converged after ", iter_num + 1,
              "iteration, with loss: ", self.log_likelihood_loss[iter_num])
        self.params = {'log_r_matrix': log_r_matrix,
                       'means': means, 'variances': variances, 'pi': pi}

    def predict_proba(self, X):
        class_probab_list = np.zeros((self.num_components, len(X)))
        for digit in range(self.num_components):
            sum_log_r, r = self._multivariate_gaussian(
                X, self.params['pi'], self.params['variances'], self.params['means'])
            class_probab_list[digit] = logsumexp(
                r, axis=0) * (X.shape[0] / len(X))
        return class_probab_list

    def _multivariate_gaussian(self, X, pi, variances, means):
        reversed_var = 1 / (variances + eps)  # Add epsilon to variances
        log_r_matrix = (X ** 2) @ reversed_var.T
        log_r_matrix -= 2 * X @ (means * reversed_var).T
        log_r_matrix[:] += np.sum(
            (means ** 2) * reversed_var, axis=1)
        log_r_matrix *= -0.5
        # Add epsilon to variances
        log_r_matrix[:] += np.log(pi) - 0.5 * \
            np.sum(np.log(variances + eps), axis=1)
        log_r_matrix = log_r_matrix.T
        sum_log_r = logsumexp(log_r_matrix, axis=0)
        return sum_log_r, log_r_matrix
