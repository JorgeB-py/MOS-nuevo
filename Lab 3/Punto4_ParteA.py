import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## 1. Cálculo analítico del gradiente
def gradiente(x, y):
    """Calcula el gradiente de L(x,y) = (x-2)² + (y+1)²"""
    return np.array([2*(x-2), 2*(y+1)])

## 2. Gradiente Descendente
def gradiente_descendente(x0, y0, alpha=0.1, max_iter=100, tol=1e-6):
    """
    Implementa el algoritmo de gradiente descendente
    
    Parámetros:
    x0, y0: Punto inicial
    alpha: Tasa de aprendizaje
    max_iter: Máximo de iteraciones
    tol: Tolerancia para la norma del gradiente
    
    Retorna:
    (x_opt, y_opt): Punto óptimo encontrado
    historia: Lista con todos los puntos visitados
    """
    x, y = x0, y0
    historia = [(x, y)]
    
    for i in range(max_iter):
        # Calcular gradiente
        g = gradiente(x, y)
        
        # Criterio de parada
        if np.linalg.norm(g) < tol:
            print(f"Convergencia alcanzada en {i} iteraciones")
            break
            
        x = x - alpha * g[0]
        y = y - alpha * g[1]
        
        historia.append((x, y))
        
    else:
        print("Máximo de iteraciones alcanzado")
    
    return (x, y), np.array(historia)

## 3. Uso de diferentes valores de alpha
def experimentar_alpha(x0, y0, alphas=[0.01, 0.1, 0.5, 0.9, 1.0, 1.1]):
    resultados = {}
    
    for alpha in alphas:
        print(f"\nProbando alpha = {alpha:.2f}")
        (x_opt, y_opt), historia = gradiente_descendente(x0, y0, alpha=alpha)
        resultados[alpha] = {
            'optimo': (x_opt, y_opt),
            'historia': historia,
            'iteraciones': len(historia)-1
        }
        
    return resultados

## 4. Resultados
def visualizar_resultados(resultados):
    plt.figure(figsize=(12, 8))
    
    x = np.linspace(-3, 5, 100)
    y = np.linspace(-4, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X-2)**2 + (Y+1)**2
    
    plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Valor de L(x,y)')
    
    for alpha, data in resultados.items():
        hist = data['historia']
        plt.plot(hist[:,0], hist[:,1], '.-', label=f'α={alpha:.2f} ({data["iteraciones"]} iter)')
    
    plt.plot(2, -1, 'r*', markersize=15, label='Óptimo teórico (2, -1)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trayectorias de Gradiente Descendente para diferentes α')
    plt.legend()
    plt.grid(True)
    plt.show()

## 5. Analisis de sensabilidad de alpha
def analizar_sensibilidad(resultados):
    """Analiza cómo afecta alpha al rendimiento"""
    print("\nAnálisis de sensibilidad al valor de alpha:")
    print("Alpha\tIteraciones\tError final")
    print("----------------------------------")
    
    for alpha, data in resultados.items():
        x_opt, y_opt = data['optimo']
        error = np.sqrt((x_opt-2)**2 + (y_opt+1)**2)
        print(f"{alpha:.2f}\t{data['iteraciones']}\t\t{error:.6f}")

if __name__ == "__main__":
    x0, y0 = np.random.uniform(-2, 2, 2)
    
    print("=== Implementación de Gradiente Descendente en 3D ===")
    print(f"Función a minimizar: L(x,y) = (x-2)² + (y+1)²")
    print(f"Punto inicial: ({x0:.4f}, {y0:.4f})")
    print(f"Solución teórica: (2, -1)")
    
    resultados = experimentar_alpha(x0, y0)
    
    visualizar_resultados(resultados)
    
    analizar_sensibilidad(resultados)
    
