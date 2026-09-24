import numpy as np

# Hàm tính giá trị của hàm số
def cost_g(x):
    return (1/3)*x**3 - x

# Hàm tính đạo hàm
def grad_g(x):
    return x**2 - 1

# Thuật toán Gradient Descent
def myGD_bai2(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad_g(x[-1])
        # Điều kiện dừng khi đạo hàm rất nhỏ (gần bằng 0)
        if abs(grad_g(x_new)) < 1e-3: 
            break
        x.append(x_new)
    return x, it

# Thử nghiệm với điểm khởi tạo x0 = 2 và learning rate eta = 0.1
x2, it2 = myGD_bai2(2, 0.1)
print(f"Solution x = {x2[-1]:.6f}, cost = {cost_g(x2[-1]):.6f}, after {it2} iterations")