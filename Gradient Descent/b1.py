import numpy as np

# Hàm tính giá trị của hàm số
def cost(x):
    return x**2 - 2

# Hàm tính đạo hàm
def grad(x):
    return 2 * x

# Thuật toán Gradient Descent
def myGD_bai1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        # Điều kiện dừng khi đạo hàm rất nhỏ (gần bằng 0)
        if abs(grad(x_new)) < 1e-3: 
            break
        x.append(x_new)
    return x, it

# Thử nghiệm với điểm khởi tạo x0 = 5 và learning rate eta = 0.1
x1, it1 = myGD_bai1(5, 0.1)
print(f"Solution x = {x1[-1]:.6f}, cost = {cost(x1[-1]):.6f}, after {it1} iterations")