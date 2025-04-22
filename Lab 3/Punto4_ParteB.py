import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, Matrix, lambdify
import time

# 1. Definicion y calculo de gradiente y hessiana
x, y = symbols('x y')
f_expr = (x-2)**2*(y+2)**2 + (x+1)**2 + (y-1)**2
grad_f_expr = Matrix([diff(f_expr, var) for var in (x, y)])
hess_f_expr = grad_f_expr.jacobian([x, y])
f = lambdify((x, y), f_expr, 'numpy')
grad_f = lambdify((x, y), grad_f_expr, 'numpy')
hess_f = lambdify((x, y), hess_f_expr, 'numpy')

# 2. Gradiente Descendente
def gradient_descent(x0, alpha=0.01, max_iter=1000, tol=1e-6):
    xk = np.array(x0, dtype=float)
    history = [xk.copy()]
    start_time = time.time()
    
    for k in range(max_iter):
        grad = np.array(grad_f(*xk), dtype=float).flatten()
        if np.linalg.norm(grad) < tol:
            break
        xk -= alpha * grad
        history.append(xk.copy())
    
    exec_time = time.time() - start_time
    return xk, np.array(history), k, exec_time

# 3. Newton-Raphson
def newton_raphson(x0, alpha=1.0, max_iter=100, tol=1e-6):
    xk = np.array(x0, dtype=float)
    history = [xk.copy()]
    start_time = time.time()
    
    for k in range(max_iter):
        grad = np.array(grad_f(*xk), dtype=float).flatten()
        hess = np.array(hess_f(*xk), dtype=float)
        
        if np.linalg.norm(grad) < tol:
            break
            
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
            
        xk -= alpha * step
        history.append(xk.copy())
    
    exec_time = time.time() - start_time
    return xk, np.array(history), k, exec_time

# 4. Punto incial
x0 = np.array([-2, -3])

# Vario los alpha para encontrar el mejor
alphas_gd = [0.001, 0.005, 0.01, 0.05]
best_alpha_gd = None
best_iter_gd = float('inf')
for alpha in alphas_gd:
    _, _, iter_gd, _ = gradient_descent(x0, alpha=alpha)
    if iter_gd < best_iter_gd:
        best_iter_gd = iter_gd
        best_alpha_gd = alpha

alphas_nr = [0.1, 0.5, 1.0, 1.5]
best_alpha_nr = None
best_iter_nr = float('inf')
for alpha in alphas_nr:
    _, _, iter_nr, _ = newton_raphson(x0, alpha=alpha)
    if iter_nr < best_iter_nr:
        best_iter_nr = iter_nr
        best_alpha_nr = alpha

x_opt_gd, hist_gd, iter_gd, time_gd = gradient_descent(x0, alpha=best_alpha_gd)
x_opt_nr, hist_nr, iter_nr, time_nr = newton_raphson(x0, alpha=best_alpha_nr)

# 5. Visualización
plt.figure(figsize=(15, 6))

X, Y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-4, 2, 200))
Z = f(X, Y)
plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x,y)')
plt.plot(hist_gd[:, 0], hist_gd[:, 1], 'o-', color='blue', label=f'Grad. Desc. (α={best_alpha_gd:.3f}, {iter_gd} iter)')
plt.plot(hist_nr[:, 0], hist_nr[:, 1], 's-', color='red', label=f'Newton-Raphson (α={best_alpha_nr:.1f}, {iter_nr} iter)')
plt.scatter(x0[0], x0[1], color='black', s=100, label='Punto inicial (-2, -3)')
plt.scatter(x_opt_gd[0], x_opt_gd[1], color='blue', s=100, marker='*', label='Opt GD')
plt.scatter(x_opt_nr[0], x_opt_nr[1], color='red', s=100, marker='*', label='Opt NR')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trayectorias de optimización')
plt.legend()

true_min = np.array([-1.0, 1.0])  
error_gd = np.linalg.norm(hist_gd - true_min, axis=1)
error_nr = np.linalg.norm(hist_nr - true_min, axis=1)

plt.subplot(1, 2, 2)
plt.semilogy(error_gd, 'b-o', label='Gradiente Descendente')
plt.semilogy(error_nr, 'r-s', label='Newton-Raphson')
plt.xlabel('Iteración')
plt.ylabel('Error (log)')
plt.title('Convergencia del error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Análisis 
print("\n=== Comparación entre métodos ===")
print(f"Gradiente Descendente (α={best_alpha_gd:.3f}):")
print(f"- Iteraciones: {iter_gd}")
print(f"- Tiempo: {time_gd:.4f} s")
print(f"- Punto final: ({x_opt_gd[0]:.6f}, {x_opt_gd[1]:.6f})")
print(f"- Valor final: {f(*x_opt_gd):.6f}")

print(f"\nNewton-Raphson (α={best_alpha_nr:.1f}):")
print(f"- Iteraciones: {iter_nr}")
print(f"- Tiempo: {time_nr:.4f} s")
print(f"- Punto final: ({x_opt_nr[0]:.6f}, {x_opt_nr[1]:.6f})")
print(f"- Valor final: {f(*x_opt_nr):.6f}")

# 7. Tabla comparativa
print("\n=== Tabla comparativa ===")
print("| Criterio               | Gradiente Descendente | Newton-Raphson     |")
print("|------------------------|-----------------------|--------------------|")
print(f"| Iteraciones           | {iter_gd:>21} | {iter_nr:>18} |")
print(f"| Tiempo por iteración  | {time_gd/iter_gd:.6f} s    | {time_nr/iter_nr:.6f} s   |")
print(f"| Precisión final       | {error_gd[-1]:.2e}        | {error_nr[-1]:.2e}       |")
print("| Robustez a α          | Baja (muy sensible) | Alta (poco sensible)|")
print("| Costo computacional   | Bajo (solo gradiente)| Alto (Hessiano)    |")

