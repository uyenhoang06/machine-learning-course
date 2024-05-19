from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

from my_pca import MyPCA

iris = load_iris()
X = iris['data']
y = iris['target']

n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)

my_pca = MyPCA(n_components=2)
my_pca.fit(X)

print('Components:\n', my_pca.components_)
print('Explained variance ratio from scratch:\n',
      my_pca.explained_variance_ratio_)
print('Cumulative explained variance from scratch:\n',
      my_pca.cum_explained_variance_all)

X_proj = my_pca.transform(X)
print('Transformed data shape from scratch:', X_proj.shape)

print(X_proj[: 0])
print(X_proj[: 1])

X_std = StandardScaler().fit_transform(X)

pca = PCA(n_components=2).fit(X_std)

print('Components:\n', pca.components_)
print('Explained variance ratio:\n', pca.explained_variance_ratio_)

cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
print('Cumulative explained variance:\n', cum_explained_variance)

X_pca = pca.transform(X_std)  # Apply dimensionality reduction to X.
print(X_pca[: 0])
print(X_pca[: 1])
print('Transformed data shape:', X_pca.shape)
