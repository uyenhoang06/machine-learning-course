from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("E:\ML\linear_regression\SAT_GPA.csv")
print(data)
print(len(data))

# Show the description of data
print(data.describe())

# Set to training data (x, y)
y = data['GPA']
x = data['SAT']

# Remind that we need to put component x_0 = 1 to x
plt.scatter(x, y)
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()

X = x.to_numpy()
Y = y.to_numpy()

# Visualize data
plt.plot(X, Y, 'ro')
plt.axis()
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()

# 60 dòng đầu thuộc tập train
train_data = data.iloc[:60]
print(train_data)
print(len(train_data))

# các dòng còn lại thuộc tập validation
validation_data = data.iloc[60:, :]
print(validation_data)
print(len(validation_data))

# lập công thức hồi quy tuyến tính với dữ liệu thuộc tập train
y_train = train_data['GPA'].to_numpy().reshape(-1, 1)
x_train = train_data['SAT'].to_numpy().reshape(-1, 1)

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(x_train, y_train)
t_1 = reg.coef_
t_0 = reg.intercept_
print(t_1)
print(t_0)

one = np.ones((x_train.shape[0], 1))
Xbar = np.concatenate((one, x_train), axis=1)
line = ['Intercept', 'SAT']
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y_train)

coefs = regr.coef_
coefs = coefs.tolist()
print(coefs)

r = list(zip(line, coefs[0]))
for o in r:
    print("{: >20}: {: >10}".format(*o))

# Vẽ đường hồi quy
y_valid = validation_data['GPA'].to_numpy().reshape(-1, 1)
x_valid = validation_data['SAT'].to_numpy().reshape(-1, 1)

plt.scatter(x_valid, y_valid)
yhat = t_0 + t_1 * x_valid
fig = plt.plot(x_valid, yhat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()
