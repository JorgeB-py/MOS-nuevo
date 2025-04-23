import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Definir el problema
# Maximizar Z = 3x1 + 2x2 + 5x3 
# sujeto a:
# x1 + x2 + x3 ≤100
# 2x1 + x2 + x3 ≤150 
# x1 + 4x2 + 2x3 ≤80 


# Convertir a forma estándar
# Matriz de coeficientes A (incluye variables de holgura)
A = np.array([
    [1, 1, 1, 1,0,0],  # x1 + x2 + x3+s1=100
    [2, 1,1, 0,1,0],   # 2x1 + x2 + x3 +s2 =150 
    [1, 4, 2, 0,0,1]   # x1 + 4x2 + 2x3 +s3 = 80 

], dtype=float)

# Vector del lado derecho (RHS)
b = np.array([100, 150,80], dtype=float)
c=np.array([-3,-2,-5,0,0,0], dtype=float)

# Tabla Simplex (restricciones + fila Z)
tabla = np.vstack([np.hstack([A, b.reshape(-1, 1)]), np.hstack([c, [0]])])

# Variables básicas iniciales
variablesBasicas = ['s1', 's2','s3']
columnas = ['x1', 'x2','x3', 's1', 's2','s3', 'RHS']
# tabla = np.hstack([A, b.reshape(-1, 1)])
# tabla = np.vstack([tabla, np.append(c, 0)])
# print(tabla)

iterations = [pd.DataFrame(tabla.copy(), index=variablesBasicas + ['Z'], columns=columnas)]

# print(iterations)
def simplex(tabla):
  while True:
    # print("ded")
    z_fila=tabla[-1,:-1]
    if np.all(z_fila>=0):
      break
    # for i in z_fila:
    #   print("Arguemnto ",i)
    pivot_col = np.argmin(z_fila)  # Devuelve el índice del valor más negativo
    # prueba=np.argmin(z_fila)
    # print("Pivot",pivot_col)
    # print("Prueba",prueba)
    ratios=[]
    # print(tabla)
    for i in range(len(tabla)-1):
      col_val=tabla[i,pivot_col]
      rhs_val=tabla[i,-1]
      ratios.append(rhs_val / col_val if col_val > 0 else np.inf) # Divide entre el pivote para ver que fila agarra central 
    # print("Los ratios fueron",ratios)
    pivot_fila=np.argmin(ratios)
    pivot_valor=tabla[pivot_fila,pivot_col]
    tabla[pivot_fila,:]/=pivot_valor
    for i in range(len(tabla)):
      if i!=pivot_fila:
        tabla[i,:]-= tabla[i,pivot_col] * tabla[pivot_fila,:]# Agarro toda la fila 
    variablesBasicas[pivot_fila]= columnas[pivot_col]
    iterations.append(pd.DataFrame(tabla.copy(), index=variablesBasicas + ['Z'], columns=columnas))
  return tabla     
    
  
tablaFinal = simplex(tabla.copy())
for i, df in enumerate(iterations):
    print(f"\n--- Iteración {i} ---")
    print(df)

# Solución óptima
solution = {var: 0 for var in columnas[:-1]}
for i, var in enumerate(variablesBasicas):
    solution[var] = tablaFinal[i, -1]
optimal_value = tablaFinal[-1, -1]
print("Solución óptima:", solution)
print("Valor óptimo de Z:", optimal_value)

# Interpretación geométrica (suponiendo x2 = 0)
plt.figure(figsize=(6, 6))
x = np.linspace(0, 100, 400)
plt.plot(x, 100 - x, label='x1 + x3 = 100 (x2=0)')
plt.plot(x, 150 - 2*x, label='2x1 + x3 = 150 (x2=0)')
plt.plot(x, (80 - x)/2, label='x1 + 2x3 = 80 (x2=0)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('x1')
plt.ylabel('x3')
plt.title('Interpretación geométrica (x2=0)')
plt.legend()
plt.grid(True)
plt.show()

# Análisis de sensibilidad básico
print("\n--- Análisis de sensibilidad básico ---")
c_perturbed = np.array([-2.9, -2.1, -5.1, 0, 0, 0], dtype=float)
table_perturbed = np.vstack([np.hstack([A, b.reshape(-1, 1)]), np.append(c_perturbed, 0)])
final_perturbed = simplex(table_perturbed.copy())
optimal_perturbed = final_perturbed[-1, -1]
print(f"Nuevo valor de Z con coeficientes perturbados: {optimal_perturbed:.2f}")
