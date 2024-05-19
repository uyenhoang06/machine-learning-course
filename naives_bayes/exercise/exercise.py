import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import pandas as pd

#   thực hiện đọc dữ liệu từ file có tên gọi là filename
#   dùng pandas đọc dữ liệu, tạo thành data frame có các cột là:
#               'Number', 'Class', 'ClumpThickness', 'UniformityofCellSize', 'UniformityofCellShape', 'MarginalAdhesion',
#               'SingleEpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses'
with open('week1/exercise/datacum.txt', encoding='utf-8') as my_file:
    # lưu những dòng hợp lệ vào 1 list
    # những dòng hợp lệ không bắt đầu bằng kí tự "#" và không là dòng trống
    # đối với những dòng hợp lệ, tách các giá trị trên dòng bởi dấu ',' và chuyển các phần tử này về kiểu nguyên
    lines = [[int(value) for value in line.strip().split(',')]
             for line in my_file if (not line.startswith('#')) and (line.strip() != '')]

    # khởi tạo dataframe từ list thu được
    data = pd.DataFrame(lines,  columns=(
        'Number', 'Class', 'ClumpThickness', 'UniformityofCellSize', 'UniformityofCellShape', 'MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses'))


# chọn cột Number là cột chỉ số của dataframe
data.set_index('Number', inplace=True)
print(len(data.index))


data.replace({'Class': {2: 0, 4: 1}}, inplace=True)

# lấy ngẫu nhien 80 mẫu thuộc nhãn 2
benign_test_indices = data[data['Class'] == 0].sample(n=80)
# print(len(benign_test_indices))

# lấy ngẫu nhiên 40 mẫu thuộc nhãn 4
malignant_test_indices = data[data['Class'] == 1].sample(n=40)
# print(len(malignant_test_indices))

# tạo tập dữ liệu test gồm 80 mẫu thuộc nhãn 2 và 40 mẫu thuộc nhãn 4 lấy ở trên
test_data = pd.concat([benign_test_indices, malignant_test_indices])

# tạo X chứa các đặc trưng của dữ liệu và nhãn y tương ứng
X_test = test_data.drop(columns=['Class'])
y_test_labels = test_data['Class']

# tạo tập dữ liệu training (gồm những dòng dữ liệu còn lại)
training_data = data.drop(test_data.index)

# Xác định đặc trưng X và nhãn y
X_train = training_data.drop(columns=['Class'])
y_train_labels = training_data['Class']
print(X_train)
print(y_train_labels)

# sử dụng scikit-learning tạo mô hình và huấn luyện mô hình dựa trên các đặc trưng, thuộc tính X và nhãn y tương ứng
model_sk = GaussianNB(priors=None)
model_sk.fit(X_train, y_train_labels)

# Dự đoán nhãn trên tập test
predictions = model_sk.predict(X_test)
print(f'predictions: {predictions}')

# confused matrix
confusion_matrix = metrics.confusion_matrix(y_test_labels, predictions)
print(f'Confusion Matrix: \n {confusion_matrix}')

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot()


# Đánh giá độ chính xác sử dụng hàm accuracy_score trong viện scikit
# accuracy_score nhận 2 input: kết quả dự đoán và kết quả thực tế
# hàm accuracy_score tính độ chính xác theo công thức:
# accuracy = (số lần dự đoán chính xác) / (tổng số dự đoán)
accuracy = accuracy_score(predictions, y_test_labels)
print(f'Accuracy on test data: {accuracy:.2f}')

# Tính accuracy không sử dụng hàm accuracy_score trong thư viện scikit-learn
correct_predictions = (predictions == y_test_labels).sum()
total_predictions = len(y_test_labels)
accuracy_manual = correct_predictions / total_predictions
print(f'Accuracy on test data (manual): {accuracy_manual:.2f}')

# Tính precision và recall sử dụng hàm precision_score và recall_score trong thư viện scikit-learn
# precision_score tính giá trị precision bằng tỉ lệ số điểm true positive trong số những điểm được phân loại là positive (TP + FP).
# recall_score tính giá trị recall bằng tỉ lệ số điểm true positive trong số những điểm thực sự là positive (TP + FN).
precision = precision_score(y_test_labels, predictions)
recall = recall_score(y_test_labels, predictions)

print(f'Precision: {precision:.5f}')
print(f'Recall: {recall:.5f}')

# Tính precision và recall không sử dụng hàm precision_score và recall_score trong thư viện scikit-learn
true_positives = ((predictions == 1) & (y_test_labels == 1)).sum()
false_positives = ((predictions == 1) & (y_test_labels == 0)).sum()
false_negatives = ((predictions == 0) & (y_test_labels == 1)).sum()

precision_manual = true_positives / (true_positives + false_positives)
recall_manual = true_positives / (true_positives + false_negatives)
print(f'Precision (manual): {precision_manual:.5f}')
print(f'Recall (manual): {recall_manual:.5f}')

plt.show()
