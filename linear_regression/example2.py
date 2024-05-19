from sklearn import linear_model
import math
import numpy as np


# Đọc dữ liệu từ tệp
with open('fuel.txt') as f:
    lines = f.readlines()

x_data = []
y_data = []
lines.pop(0)

for line in lines:
    splitted = line.replace('\n', '').split(',')
    splitted.pop(0)
    splitted = list(map(float, splitted))
    fuel = 1000 * splitted[1] / splitted[5]
    dlic = 1000 * splitted[0] / splitted[5]
    logMiles = math.log2(splitted[3])
    y_data.append([fuel])
    x_data.append([splitted[-1], dlic, splitted[2], logMiles])

x_data = np.asarray(x_data)
y_data = np.asarray(y_data)


###################################################################################################
# Sử dụng numpy
# Sử dụng thuật toán HoldHouse để triển khai QR
def qr_householder(A):
    # """ Compute QR decomposition of A using Householder reflection"""
    M = A.shape[0]
    N = A.shape[1]

    # set Q to the identity matrix
    Q = np.identity(M)

    # set R to zero matrix
    R = np.copy(A)

    for n in range(N):
        # vector to transform
        x = A[n:, n]
        k = x.shape[0]

        # compute ro=-sign(x0)||x||
        ro = -np.sign(x[0]) * np.linalg.norm(x)

        # compute the householder vector v
        e = np.zeros(k)
        e[0] = 1
        v = (1 / (x[0] - ro)) * (x - (ro * e))

        # apply v to each column of A to find R
        for i in range(N):
            R[n:, i] = R[n:, i] - (2 / (v@v)) * ((np.outer(v, v)) @ R[n:, i])

        # apply v to each column of Q
        for i in range(M):
            Q[n:, i] = Q[n:, i] - (2 / (v@v)) * ((np.outer(v, v)) @ Q[n:, i])

    return Q.transpose(), R


def linear_regression(x_data, y_data):
    # """
    # This function calculate linear regression base on x_data and y_data
    # :param x_data: vector
    # :param y_data: vector
    # :return: w (regression estimate)
    # """

    # add column 1
    x_bars = np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis=1)

    Q, R = qr_householder(x_bars)  # QR decomposition
    R_pinv = np.linalg.pinv(R)  # calculate inverse matrix of R
    A = np.dot(R_pinv, Q.T)  # apply formula

    return np.dot(A, y_data)


w = linear_regression(x_data, y_data)  # get result
print(w)
w = w.T.tolist()
# print(w)
line = ['Intercept', 'Tax', "Dlic", "Income", 'LogMiles']

res = list(zip(line, w[0]))
for o in res:
    print("{: >20}: {: >10}".format(*o))

###################################################################################################
# Sử dụng Scikit-Learn
X = x_data
y = y_data

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# Load training data here and assign to Xbar (obs. Data) and y (label)
# fit the model by Linear Regression
# Đã add thêm cột hệ số tự do
regr = linear_model.LinearRegression(fit_intercept=False)

# hoặc (vì chưa add vào hệ số tự do)
reg = linear_model.LinearRegression(fit_intercept=True)

# fit_intercept = False for calculating the bias
regr.fit(Xbar, y)
reg.fit(X, y)
print(reg.intercept_)
print(regr.intercept_)

coefs = reg.coef_
coefs = coefs.tolist()

r = list(zip(line, coefs[0]))
for o in r:
    print("{: >20}: {: >10}".format(*o))

print(coefs)
print(reg.coef_)
