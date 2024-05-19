import os
import numpy as np
from scipy.special import logsumexp
from numpy import random
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist

eps = 0.1  # small epsilon value


def multivariate_gaussian(X, pi, variances, means):
    """
    X: examples with shape (n, d)
    pi: priors with shape (K, )
    variances and means shape: (K, d)
    log_r_matrix: log of r matrix with shape (K, n)
    """
    reversed_var = 1 / (variances + eps)  # Add epsilon to variances
    log_r_matrix = (X ** 2) @ reversed_var.T
    log_r_matrix -= 2 * X @ (means * reversed_var).T
    log_r_matrix[:] += np.sum((means ** 2) * reversed_var, axis=1)
    log_r_matrix *= -0.5
    # Add epsilon to variances
    log_r_matrix[:] += np.log(pi) - 0.5 * \
        np.sum(np.log(variances + eps), axis=1)
    log_r_matrix = log_r_matrix.T

    sum_log_r = logsumexp(log_r_matrix, axis=0)
    return sum_log_r, log_r_matrix


def em_gmm(X, gm_num):
    max_iter = 500
    tol = 1e-5

    # X dimensions
    n, d = X.shape
    # log-likelihood values of each iteration
    log_likelihood_loss = {}

    # initialization
    log_r_matrix = np.eye(gm_num)
    log_r_matrix = log_r_matrix[np.random.choice(gm_num, size=n)].T
    means = variances = pi = np.array([])
    iter_num = 0

    for it in range(max_iter):
        # M-Step
        # shape: (K,), sum of elements in log_r_matrix rows
        r_k = np.sum(log_r_matrix, axis=1) + (10 * np.finfo(float).eps)
        pi = r_k / n
        means = log_r_matrix @ X / r_k[:, np.newaxis]  # (K, d)
        variances = log_r_matrix @ (X ** 2) / r_k[:, np.newaxis]
        variances -= means ** 2
        variances += eps  # Add epsilon to variances

        # E-Step
        sum_log_r, log_r_matrix = multivariate_gaussian(
            X, pi, variances, means)
        log_r_matrix = np.exp(log_r_matrix - sum_log_r)

        # compute loss
        log_likelihood_loss[it] = -np.sum(sum_log_r)
        loss_difference = 0
        if it > 1:
            loss_difference = np.abs(
                log_likelihood_loss[it] - log_likelihood_loss[it - 1]) / (np.abs(log_likelihood_loss[it]) + eps)
            if loss_difference <= tol:
                iter_num = it
                break

    print("EM for GMM converged after ", iter_num + 1,
          "iteration, with loss: ", log_likelihood_loss[iter_num])
    GMM_Params = {'log_r_matrix': log_r_matrix,
                  'means': means, 'variances': variances, 'pi': pi}
    return GMM_Params, log_likelihood_loss


effective_features = 50   # number top effective features to be selected by PCA
gm_num = 5                # number of Gaussian Models in each GMM

# Loading and normalizing MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('Data loaded!')
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)

# flattening feature dataset
X_train = X_train.astype(float).reshape(-1, 28 * 28) / 255
X_test = X_test.astype(float).reshape(-1, 28 * 28) / 255

print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)

num_classes = 10
n = X_train.shape[0]

# Implementing PCA
pca = PCA(n_components=effective_features)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print("New dimension after PCA: ", X_train.shape[1])

GMM_Params = {}
log_likelihood_loss = {}
prior = np.zeros(num_classes)

for digit in range(num_classes):
    print('... GMM is fitting to digit  ', digit)
    X = X_train[y_train == digit]
    GMM_Params[digit], log_likelihood_loss[digit] = em_gmm(X, gm_num)
    print('GMM parameters computed for digit = ', digit)
    prior[digit] = (X.shape[0] / n)

print('....GMM parameteres for each digit were computed!')

print('predicting test labels ...')
num_test = X_test.shape[0]
class_probab_list = np.zeros((num_classes, num_test))

for digit in range(num_classes):
    sum_log_r, r = multivariate_gaussian(
        X_test, GMM_Params[digit]['pi'], GMM_Params[digit]['variances'], GMM_Params[digit]['means'])
    class_probab_list[digit] = logsumexp(r, axis=0) * prior[digit]

predictions = np.argmax(class_probab_list, axis=0)

correct_predictions = np.sum(predictions == y_test)
accuracy = correct_predictions / num_test

print("accuracy: ", accuracy)
