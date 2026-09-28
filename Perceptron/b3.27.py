import numpy as np

# Định nghĩa vector w và x
w = np.array([1, 2, -10])
x = np.array([3, 4, 1])

# 1. Tính w^T x (tích vô hướng)
w_T_x = np.dot(w, x)
print(f"1. Tích vô hướng w^T x = {w_T_x}")

# 2. Xác định nhãn dự đoán
# Quy ước: nếu w^T x >= 0 thì nhãn là 1, ngược lại là -1
y_pred = 1 if w_T_x >= 0 else -1
print(f"2. Nhãn dự đoán của điểm dữ liệu (y_pred) = {y_pred}")

# 3. Kiểm tra với nhãn thực tế y = -1
y_true = -1
if y_pred != y_true:
    print("3. Nếu nhãn thực tế là y = -1, điểm dữ liệu bị phân lớp sai.")
else:
    print("3. Nếu nhãn thực tế là y = -1, điểm dữ liệu phân lớp đúng.")