import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats

# Thiết lập kiểu hiển thị cho biểu đồ
sns.set(style='darkgrid', font_scale=1.4)

# Tải dữ liệu
data = None
try:
    data = pd.read_csv('C:/Users/vuhuu/OneDrive/Desktop/Ai/HocMayUngDung/Lab01/data.csv')  # Đọc dữ liệu từ file CSV
    if 'Unnamed: 32' in data.columns:  # Kiểm tra cột không cần thiết
        data.drop('Unnamed: 32', axis=1, inplace=True)  # Xóa cột không cần thiết
except FileNotFoundError:
    print("Không tìm thấy file. Vui lòng kiểm tra đường dẫn.")  # Thông báo nếu không tìm thấy file

# Xem trước dữ liệu
if data is not None:
    print('Kích thước Dataframe:', data.shape)  # In kích thước của dataframe
    print(data.head(3))  # In 3 dòng đầu tiên của dữ liệu
    
    # Tách đặc trưng và nhãn
    X = data.drop('diagnosis', axis=1)  # Đặc trưng
    y = data['diagnosis']  # Nhãn

    # Mã hóa nhãn thành nhị phân
    y = (y == 'M').astype('int')  # Chuyển đổi nhãn thành 0 và 1

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (tỷ lệ 80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    class kNN:
        '''Thuật toán k-Nearest Neighbours'''
        def __init__(self, k=3, metric='euclidean', p=None):
            self.k = k  # Số lượng hàng xóm gần nhất
            self.metric = metric  # Tham số khoảng cách
            self.p = p  # Tham số cho khoảng cách Minkowski
        
        def euclidean(self, v1, v2):
            return np.sqrt(np.sum((v1 - v2) ** 2))  # Khoảng cách Euclidean
        
        def manhattan(self, v1, v2):
            return np.sum(np.abs(v1 - v2))  # Khoảng cách Manhattan
        
        def minkowski(self, v1, v2, p=2):
            return np.sum(np.abs(v1 - v2) ** p) ** (1 / p)  # Khoảng cách Minkowski
        
        def fit(self, X_train, y_train):
            self.X_train = X_train  # Lưu tập huấn luyện
            self.y_train = y_train  # Lưu nhãn tương ứng
        
        def predict(self, X_test):
            preds = []  # Danh sách để lưu dự đoán
            for test_row in X_test:  # Duyệt qua từng dòng trong tập kiểm tra
                nearest_neighbours = self.get_neighbours(test_row)  # Lấy hàng xóm gần nhất
                majority = stats.mode(nearest_neighbours)[0]  # Tìm nhãn phổ biến nhất
                if isinstance(majority, np.ndarray):  # Kiểm tra nếu majority là mảng
                    majority = majority[0]  # Lấy giá trị đầu tiên
                preds.append(majority)  # Thêm nhãn vào danh sách dự đoán
            return np.array(preds)  # Trả về mảng dự đoán
        
        def get_neighbours(self, test_row):
            distances = []  # Danh sách để lưu khoảng cách
            for (train_row, train_class) in zip(self.X_train, self.y_train):  # Duyệt qua từng dòng trong tập huấn luyện
                # Tính khoảng cách tùy theo tham số đã chọn
                if self.metric == 'euclidean':
                    dist = self.euclidean(train_row, test_row)
                elif self.metric == 'manhattan':
                    dist = self.manhattan(train_row, test_row)
                elif self.metric == 'minkowski':
                    dist = self.minkowski(train_row, test_row, self.p)
                else:
                    raise ValueError('Các tham số hỗ trợ là euclidean, manhattan và minkowski')  # Thông báo lỗi nếu tham số không hợp lệ

                distances.append((dist, train_class))  # Lưu khoảng cách và nhãn
            
            distances.sort(key=lambda x: x[0])  # Sắp xếp theo khoảng cách
            return [neighbours[1] for neighbours in distances[:self.k]]  # Trả về k hàng xóm gần nhất
    
    def accuracy(preds, y_test):
        return 100 * (preds == y_test).mean()  # Tính độ chính xác

    # Áp dụng thuật toán kNN
    for metric in ['euclidean', 'manhattan']:
        clf = kNN(k=5, metric=metric)  # Khởi tạo mô hình kNN
        clf.fit(X_train.values, y_train.values)  # Huấn luyện mô hình
        preds = clf.predict(X_test.values)  # Dự đoán trên tập kiểm tra
        print(f'Metric: {metric}, độ chính xác: {accuracy(preds, y_test):.3f} %')  # In độ chính xác