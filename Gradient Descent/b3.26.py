def grad(x):
    return 2*x -4

def cost(x):
    return x**2 - 4*x + 5

def my_GD(eta, x0, iter):
    x = [x0]
    for it in range(iter):
        x_new = x[-1] - eta*grad(x[-1])
        print('Iteration = %d, f(x) = %.4f'%(it+1, cost(x_new)))

        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)

    return (x, it+1)

(x, it) = my_GD(0.2, 5, 4)
print('Solution x = %.4f, cost = %.4f, obtained after %d iterations'%(x[-1], cost(x[-1]), it))
print('Nhan xet: Sau moi buoc, x tien dan ve %.0f và f(x) giam dan ve %.0f, thuat toan hoi tu ve diem cuc tieu'% (x[-1], cost(x[-1])))