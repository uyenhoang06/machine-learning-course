import numpy as np

# Implement PCA
class MyPCA:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.mean_ = None
        self.std_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.explained_variance_ratio_ = None
        self.cum_explained_variance_ = None
        self.components_ = None
        self.scaled_data_ = None
        self.explained_variance_ratio_all = None

    def fit(self, X):
        X = X.copy()

        # Tính giá trị trung bình của mỗi cột
        self.mean_ = np.mean(X, axis=0)

        # # Tính độ lệch chuẩn của từng cột
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1e-16

        # Chuẩn hóa dữ liệu Z-score
        self.scaled_data_ = (X - self.mean_) /self.std_


        # Tính ma trận hiệp phương sai
        cov_matrix = np.cov(self.scaled_data_.T)

        # Tính trị riêng và vector riêng
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)


        # Sắp xếp các cặp trị riêng và vector riêng theo thứ tự giảm dần của trị riêng
        eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[i, :])
                     for i in range(len(eigenvalues))]
        
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])


        self.eigenvalues_ = eig_vals_sorted
        self.eigenvectors_ = eig_vecs_sorted

        # Chọn số lượng thành phần chính theo yêu cầu
        self.components_ = eig_vecs_sorted[:self.n_components, :]

        # Tính tỷ lệ lượng thông tin/ tỷ lệ phương sai được giải thích bởi mỗi thành phần chính đã chọn
        explained_variance_ratio = self.eigenvalues_[
            :self.n_components] / np.sum(self.eigenvalues_)

        self.explained_variance_ratio_ = explained_variance_ratio
        

        self.cum_explained_variance_ = np.cumsum(
            self.explained_variance_ratio_)

        self.explained_variance_ratio_all = self.eigenvalues_ / \
            np.sum(self.eigenvalues_)
        self.cum_explained_variance_all = np.cumsum(
            self.explained_variance_ratio_all)

    def transform(self, X):
        # Chuẩn hóa
        X_normalized = (X - self.mean_)/self.std_

        # Chiếu dữ liệu vào không gian mới
        return np.dot(X_normalized, self.components_.T)

    def fit_transform(self, X):
        X = X.copy()
        self.fit(X)
        return self.transform(X)
