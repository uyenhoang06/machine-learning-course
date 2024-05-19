max_abs_idx = np.argmax(np.abs(eigenvectors), axis=0)
signs = np.sign(eigenvectors[max_abs_idx, range(eigenvectors.shape[0])])
eigenvectors = eigenvectors*signs[np.newaxis, :]
eigenvectors = eigenvectors.T
