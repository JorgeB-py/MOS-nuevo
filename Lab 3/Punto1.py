from sympy import symbols, diff, lambdify
import numpy as np
import matplotlib.pyplot as plt

x = symbols('x')
funcion = 3*x**3 - 10*x**2 - 56*x + 50

grad_funcion = diff(funcion,x)
hessiana_funcion=diff(grad_funcion,x)

f=lambdify(x,funcion,modules='numpy')
grad_f = lambdify(x, grad_funcion, modules='numpy')  
hess_f = lambdify(x, hessiana_funcion, modules='numpy')


def newton_raphson(x0,alpha=1.0,tol=1e-6,max_iter=100):
    k=0
    xk = float(x0)
    history = [xk]
    while k <= max_iter:
        grad = grad_f(xk)
        hess = hess_f(xk)
        if abs(hess) < 1e-8:  
            print(f"x0={x0}, α={alpha}: Hessiana ≈ 0 en iteración {k}")
            break
        dk=-grad/hess
        xk_new = xk + alpha * dk
        history.append(xk_new)
        if abs(grad) < tol:
            break
        k += 1
        xk = xk_new
        if k > max_iter:
            print(f"x0={x0}, α={alpha}: No convergió en {max_iter} iteraciones")
            break
    return xk, history
    
x0_values = [-6, -2, 0, 2, 6] 
alpha_values = [1.0, 0.6, 0.3]  
results = {}

for x0 in x0_values:
    for alpha in alpha_values:
        x_star, history = newton_raphson(x0, alpha)
        results[(x0, alpha)] = {
            'x_star': x_star,
            'history': history,
            'iterations': len(history) - 1,
            'grad_norm': abs(grad_f(x_star)),
            'hess_sign': hess_f(x_star)  
        }
    
x_vals = np.linspace(-6, 6, 400)
f_vals = f(x_vals)
plt.figure(figsize=(12, 8))
plt.plot(x_vals, f_vals, label='f(x) = 3x³ - 10x² - 56x + 50', color='blue')

colors = {1.0: 'red', 0.6: 'green', 0.3: 'purple'}
markers = {1.0: 'o', 0.6: 's', 0.3: '^'}

for (x0, alpha), res in results.items():
    x_star = res['x_star']
    history = res['history']
    hess_sign = res['hess_sign']
    label = f'x0={x0}, α={alpha}'
    plt.plot(history, [f(x) for x in history], color=colors[alpha], marker=markers[alpha], 
             linestyle='--', alpha=0.5)
    if hess_sign > 0:
        plt.scatter(x_star, f(x_star), color=colors[alpha], marker='v', s=100, 
                    label=f'Mínimo {label}' if label not in plt.gca().get_legend_handles_labels()[1] else "")
    elif hess_sign < 0:
        plt.scatter(x_star, f(x_star), color=colors[alpha], marker='^', s=100, 
                    label=f'Máximo {label}' if label not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title('Newton-Raphson: Convergencia a extremos locales')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

print("\nAnálisis de convergencia:")
print("x0 | α | x* | Iteraciones | Gradiente | Tipo de extremo")
print("-" * 60)
for (x0, alpha), res in results.items():
    x_star = res['x_star']
    iters = res['iterations']
    grad_norm = res['grad_norm']
    hess_sign = res['hess_sign']
    extremum_type = 'Mínimo' if hess_sign > 0 else 'Máximo' if hess_sign < 0 else 'Indefinido'
    print(f"{x0:<3} | {alpha:<3} | {x_star:.6f} | {iters:<11} | {grad_norm:.6f} | {extremum_type}")