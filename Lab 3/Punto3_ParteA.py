from sympy import symbols, diff, Matrix, lambdify
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y = symbols('x y')
f_expr = (x- 1)**2 + 100*(y - x**2)**2 

vars_xy = [x, y]
grad_f_expr = Matrix([diff(f_expr, var) for var in vars_xy])
hess_f_expr = grad_f_expr.jacobian(vars_xy)

f = lambdify((x, y), f_expr, modules='numpy')
grad_f = lambdify((x, y), grad_f_expr, modules='numpy')
hess_f = lambdify((x, y), hess_f_expr, modules='numpy')

def newton_raphson(x0, alpha=1.0, tol=1e-6, max_iter=100):
    xk = np.array(x0, dtype=float)  
    history = [xk.copy()]
    for k in range(max_iter):
        grad = np.array(grad_f(*xk), dtype=float).flatten()  
        hess = np.array(hess_f(*xk), dtype=float)  
        if np.abs(np.linalg.det(hess)) < 1e-8:
            print(f"Iteración {k}: Hessiana singular, det={np.linalg.det(hess)}")
            break
        dk = -np.linalg.solve(hess, grad)
        xk = xk + alpha * dk
        history.append(xk.copy())
        if np.linalg.norm(grad) < tol:
            print(f"Convergió en {k+1} iteraciones")
            break
        if k == max_iter - 1:
            print(f"No convergió en {max_iter} iteraciones")
    return xk, np.array(history)

x0 = [0, 10]  
alpha = 0.1  
x_star, history = newton_raphson(x0, alpha)

# Resultados
print(f"Punto inicial: {x0}")
print(f"Mínimo encontrado: {x_star}")
print(f"Valor en el mínimo: {f(*x_star):.6f}")

# Visualización en R^3 como superficie
X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-1, 3, 100))
Z = f(X, Y)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, rstride=5, cstride=5)
ax.plot(history[:, 0], history[:, 1], f(history[:, 0], history[:, 1]), 
        'ro-', label='Trayectoria Newton-Raphson', markersize=5,color='green')
ax.scatter(x0[0], x0[1], f(*x0), color='black', s=100, label='Punto inicial')
ax.scatter(x_star[0], x_star[1], f(*x_star), color='red', s=100, label='Mínimo encontrado')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z = f(x, y)')
ax.set_title('Función de Rosenbrock como superficie en R^3')
ax.legend()
plt.show()

# Analisis de convergencia 
true_min = np.array([1, 1])
distances = [np.linalg.norm(h - true_min) for h in history]
print("\nAnálisis de convergencia hacia el mínimo (1, 1):")
print(f"Número de iteraciones: {len(history) - 1}")
print(f"Distancia inicial al mínimo: {distances[0]:.6f}")
print(f"Distancia final al mínimo: {distances[-1]:.6f}")
print(f"Error relativo final: {distances[-1] / distances[0]:.6f}")
print(f"Norma del gradiente inicial: {np.linalg.norm(grad_f(*x0)):.6f}")
print(f"Norma del gradiente final: {np.linalg.norm(grad_f(*x_star)):.6f}")