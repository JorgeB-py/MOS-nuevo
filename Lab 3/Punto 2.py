import numpy as np
import matplotlib.pyplot as plt

# Definimos la función dada
# f(x) = x^5 - 8x^3 + 10x + 6
def f(x):
    return (x**5) - (8*x**3) + (10*x) + 6

# Derivada de la función f'(x) = 5x^4 - 24x^2 + 10
def df(x):
    return (5*(x**4)) - (24*(x**2)) + 10

# Segunda derivada de la función f''(x) = 20x^3 - 48x
def ddf(x):
    return (20*(x**3)) - (48*x)

# Implementación del método de Newton-Raphson para encontrar extremos locales
def newton_raphson(x:list):
    roots=[]  # Lista para almacenar las raíces encontradas
    rounds=[]  # Lista para evitar valores repetidos
    classified_extremes = {}  # Diccionario para clasificar los extremos
    a=1  # Factor de ajuste
    convergencia=0.0001  # Criterio de convergencia
    
    # Iteramos sobre cada valor inicial proporcionado
    for x0 in x:
        # Verificamos que la segunda derivada no sea cero (evita divisiones por cero)
        if abs(ddf(x0)) > 0:
            # Aplicamos el método de Newton-Raphson hasta alcanzar la convergencia
            while abs(df(x0)) > convergencia:
                var = x0 - a * (df(x0)) / (ddf(x0))
                x0 = var  # Actualizamos el valor de x0
        
        rounded = round(x0)  # Redondeamos para evitar duplicados
        
        if rounded not in rounds:
            roots.append(x0)  # Guardamos la raíz encontrada
            rounds.append(rounded)  # Evitamos duplicados
            
            # Clasificamos el extremo como mínimo o máximo local según la segunda derivada
            if ddf(x0) > 0:
                classified_extremes[x0] = ("Mínimo local", f(x0))
            elif ddf(x0) < 0:
                classified_extremes[x0] = ("Máximo local", f(x0))
            else:
                classified_extremes[x0] = ("Caso especial", f(x0))
    
    return roots, classified_extremes

# Definimos valores iniciales para Newton-Raphson
x_iniciales = [-3, -2, -1, 0, 1, 2, 3]
raices, clasificacion = newton_raphson(x_iniciales)

# Determinamos el mínimo y máximo global
valores_f = [clasificacion[x][1] for x in clasificacion]
min_global = min(valores_f)
max_global = max(valores_f)
min_punto = [x for x in clasificacion if clasificacion[x][1] == min_global][0]
max_punto = [x for x in clasificacion if clasificacion[x][1] == max_global][0]

# Graficar la función y los extremos
x_vals = np.linspace(-3.5, 3.5, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10,6))
plt.plot(x_vals, y_vals, label="f(x) = x^5-8x^3+10x+6", color="blue")  # Gráfica de la función
plt.axhline(0, color="gray", linestyle="--")  # Línea horizontal en y=0
plt.axvline(0, color="gray", linestyle="--")  # Línea vertical en x=0

# Graficamos los extremos locales en negro
for x in clasificacion:
    plt.scatter(x, clasificacion[x][1], color="black", label="Extremos locales")

# Destacamos el mínimo y máximo global en rojo
plt.scatter(min_punto, min_global, color="red", label="Mínimo global", s=100, edgecolors="black")
plt.scatter(max_punto, max_global, color="red", label="Máximo global", s=100, edgecolors="black")

# Configuración de la gráfica
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Extremos locales y globales de la función")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Imprimimos los resultados obtenidos
print("Raíces encontradas:", raices)
print("Clasificación de extremos:")
for x, (tipo, val) in clasificacion.items():
    print(f" x = {x}: {tipo}, f(x) = {val}")

print(f"\nMínimo global: x = {min_punto}, f(x) = {min_global}")
print(f"Máximo global: x = {max_punto}, f(x) = {max_global}")
