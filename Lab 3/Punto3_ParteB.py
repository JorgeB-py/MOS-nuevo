import numpy as np
from sympy import symbols, Matrix, diff, lambdify
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Formulacion matematica 
x, y, z, w = symbols('x y z w')
f_expr = (x - 1)**2 + 2*(y - 2)**2 + 3*(z - 3)**2 + 4*(w - 4)**2
vars_4d = [x, y, z, w]
grad_f = Matrix([diff(f_expr, var) for var in vars_4d])  
hess_f = grad_f.jacobian(vars_4d) 
f_func = lambdify(vars_4d, f_expr, 'numpy')
grad_func = lambdify(vars_4d, grad_f, 'numpy')
hess_func = lambdify(vars_4d, hess_f, 'numpy')

# 2. Implementacion  Newton-Raphson
def newton_raphson(x0, alpha=1.0, tol=1e-6, max_iter=100):
    xk = np.array(x0, dtype=float)
    history = [xk.copy()]
    for k in range(max_iter):
        grad = np.array(grad_func(*xk), dtype=float).flatten()
        hess = np.array(hess_func(*xk), dtype=float)
        if np.abs(np.linalg.det(hess)) < 1e-8:
            print(f"Iteración {k}: Hessiana singular")
            break
        dk = -np.linalg.solve(hess, grad)
        xk = xk + alpha * dk
        history.append(xk.copy())
        # 3. Criterio de parada
        if np.linalg.norm(grad) < tol:
            print(f"Convergió en {k+1} iteraciones")
            break
        if k == max_iter - 1:
            print(f"No convergió en {max_iter} iteraciones")
    return xk, np.array(history)

x0 = [0, 0, 0, 0]
x_star, history = newton_raphson(x0)
true_min = np.array([1, 2, 3, 4])
print(f"Punto inicial: {x0}")
print(f"Mínimo encontrado: {x_star}")
print(f"Valor en el mínimo: {f_func(*x_star):.6f}")

# 4. Visualizacon 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
h = np.array(history)
ax.plot(h[:, 0], h[:, 1], h[:, 2], 'b.-', label='Trayectoria (x, y, z)')
ax.scatter(x0[0], x0[1], x0[2], c='r', s=100, label='Punto inicial')
ax.scatter(x_star[0], x_star[1], x_star[2], c='g', s=100, label='Mínimo encontrado')
ax.scatter(true_min[0], true_min[1], true_min[2], c='blue', s=100, label='Mínimo teórico')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Convergenca en R^4 (proyección x-y-z)')
ax.legend()
plt.show()
