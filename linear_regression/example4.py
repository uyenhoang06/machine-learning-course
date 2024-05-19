from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_data = []
y_data = []
with open('vidu3_lin_reg.txt') as f:
    f.readline()
    for line in f:
        d = line.strip().split()
        d = list(map(float, d))
        y_data.append(d[6])
        x_data.append([d[1], d[2], d[3], d[4], d[5]])

print(x_data)
print(y_data)
        

