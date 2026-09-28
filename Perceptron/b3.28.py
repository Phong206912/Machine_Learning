import numpy as np

# 1. Khai báo dữ liệu đầu vào
w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y = 1
eta = 1  # Tốc độ học

print(f"Trọng số ban đầu w: {w}")
print(f"Mẫu dữ liệu x: {x}")
print(f"Nhãn y: {y}\n")

# Câu 1: Kiểm tra mẫu có bị phân lớp sai hay không
w_dot_x = np.dot(w, x)
print(f"Câu 1: Giá trị w^T * x = {w_dot_x}")

# Điều kiện sai: y * (w^T * x) <= 0
is_misclassified = (y * w_dot_x) <= 0
if is_misclassified:
    print("-> Kết quả: Mẫu bị PHÂN LỚP SAI.")
else:
    print("-> Kết quả: Mẫu phân lớp đúng.")

# Câu 2: Nếu sai, thực hiện một bước cập nhật Perceptron
if is_misclassified:
    w_new = w + eta * y * x
    print(f"\nCâu 2: Trọng số sau khi cập nhật (w_new): {w_new}")
else:
    w_new = w

# Câu 3: Tính lại giá trị w^T * x sau cập nhật
w_new_dot_x = np.dot(w_new, x)
print(f"\nCâu 3: Giá trị w^T * x sau cập nhật là: {w_new_dot_x}")