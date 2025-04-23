import numpy as np
import pandas as pd

# -- Configuración del Problema Original  
# Maximizar Z = 4x1 + 3x2
# sujeto a:
# x1 + 2x2 ≤ 8
# 3x1 + 2x2 ≤ 12
# x1, x2 ≥ 0

c_original_obj = np.array([4, 3], dtype=float)
A_original = np.array([
    [1, 2],
    [3, 2]
], dtype=float)
b_original = np.array([8, 12], dtype=float)

num_decision_vars = len(c_original_obj)
num_constraints = len(b_original)
num_slack_vars = num_constraints #

A_simplex = np.hstack([A_original, np.eye(num_slack_vars)])

c_simplex = np.concatenate([-c_original_obj, np.zeros(num_slack_vars)], axis=0)

columnas = [f'x{i+1}' for i in range(num_decision_vars)] + \
           [f's{i+1}' for i in range(num_slack_vars)] + \
           ['RHS']

variables_basicas_iniciales = [f's{i+1}' for i in range(num_slack_vars)]

tabla_inicial = np.vstack([
    np.hstack([A_simplex, b_original.reshape(-1, 1)]), 
    np.hstack([c_simplex, [0]])                       
])


def simplex(tabla_inicial, basicas_iniciales, cols):
    tabla = tabla_inicial.copy()
    variables_basicas = basicas_iniciales.copy()
    column_map = {name: i for i, name in enumerate(cols)}
    num_filas = tabla.shape[0] - 1 
    iteraciones_df = [pd.DataFrame(tabla.copy(), index=variables_basicas + ['Z'], columns=cols)]

    iter_count = 0
    max_iters = 100 

    while iter_count < max_iters:
        z_fila = tabla[-1, :-1]

        if np.all(z_fila >= -1e-9): 
            print(f"\nSolución óptima encontrada en {iter_count} iteraciones.")
            return tabla, variables_basicas, iteraciones_df

        pivot_col_idx = np.argmin(z_fila)
        columna_pivote = tabla[:-1, pivot_col_idx]

        if np.all(columna_pivote <= 1e-9):
             print("\nProblema no acotado.")
             return None, None, None 

        rhs = tabla[:-1, -1]
        ratios = np.full_like(rhs, np.inf, dtype=float) 
        mask_positivos = columna_pivote > 1e-9
        ratios[mask_positivos] = rhs[mask_positivos] / columna_pivote[mask_positivos]

        pivot_fila_idx = np.argmin(ratios)
        pivot_valor = tabla[pivot_fila_idx, pivot_col_idx]


        tabla[pivot_fila_idx, :] /= pivot_valor

        for i in range(tabla.shape[0]):
            if i != pivot_fila_idx:
                factor = tabla[i, pivot_col_idx]
                tabla[i, :] -= factor * tabla[pivot_fila_idx, :]

        variable_entrante = cols[pivot_col_idx]
        print(f"Iteración {iter_count+1}: Entra {variable_entrante}, Sale {variables_basicas[pivot_fila_idx]}")
        variables_basicas[pivot_fila_idx] = variable_entrante

        iter_count += 1
        iteraciones_df.append(pd.DataFrame(tabla.copy(), index=variables_basicas + ['Z'], columns=cols))

    # print("\nMáximo número de iteraciones alcanzado.")
    return tabla, variables_basicas, iteraciones_df 

def calcular_precios_sombra(tabla_final, num_decision, num_slack, cols):
    precios = {}
    z_fila_final = tabla_final[-1, :-1]
    col_map = {name: i for i, name in enumerate(cols)}

    for i in range(num_slack):
        slack_var_name = f's{i+1}'
        col_idx = col_map[slack_var_name]
        precios[slack_var_name] = z_fila_final[col_idx]
    return precios

def calcular_rangos_optimalidad_c(tabla_final, c_obj_originales, basicas_finales, cols):
    rangos_c = {}
    z_fila = tabla_final[-1, :-1]
    tableau_body = tabla_final[:-1, :-1]
    col_map = {name: i for i, name in enumerate(cols)}
    num_vars_total = len(z_fila)
    num_decision = len(c_obj_originales)

    indices_no_basicas = [col_map[name] for name in cols[:-1] if name not in basicas_finales]

    for i in range(num_decision):
        var_name = f'x{i+1}'
        col_idx = col_map[var_name]
        c_actual = c_obj_originales[i]

        if var_name not in basicas_finales:

            aumento_permitido = z_fila[col_idx]
            disminucion_permitida = np.inf
            rangos_c[var_name] = {
                'actual': c_actual,
                'aumento': aumento_permitido,
                'disminucion': disminucion_permitida,
                'rango': (c_actual - disminucion_permitida, c_actual + aumento_permitido)
            }
        else:
            fila_idx = basicas_finales.index(var_name)
            fila_tableau = tableau_body[fila_idx, :]

            aumento = np.inf
            disminucion = np.inf

            ratios_disminucion = []
            for j_nb in indices_no_basicas:
                if fila_tableau[j_nb] > 1e-9: 
                    ratios_disminucion.append(z_fila[j_nb] / fila_tableau[j_nb])
            if ratios_disminucion:
                disminucion = min(ratios_disminucion)

            ratios_aumento = []
            for j_nb in indices_no_basicas:
                if fila_tableau[j_nb] < -1e-9: 
                    ratios_aumento.append(-z_fila[j_nb] / fila_tableau[j_nb])
            if ratios_aumento:
                aumento = min(ratios_aumento)

            rangos_c[var_name] = {
                'actual': c_actual,
                'aumento': aumento,
                'disminucion': disminucion,
                'rango': (c_actual - disminucion, c_actual + aumento)
            }

    return rangos_c

def calcular_rangos_factibilidad_b(tabla_final, b_originales, basicas_finales, cols):
    rangos_b = {}
    rhs_final = tabla_final[:-1, -1]
    tableau_body = tabla_final[:-1, :-1]
    col_map = {name: i for i, name in enumerate(cols)}
    num_constraints = len(b_originales)

    for i in range(num_constraints): 
        constraint_label = f"Restricción {i+1} (b{i+1})"
        slack_var_name = f's{i+1}'
        col_idx_slack = col_map[slack_var_name]
        columna_slack = tableau_body[:, col_idx_slack]
        b_actual = b_originales[i]

        aumento = np.inf
        disminucion = np.inf

        ratios_disminucion = []
        for k in range(len(rhs_final)):
            if columna_slack[k] > 1e-9:
                ratios_disminucion.append(rhs_final[k] / columna_slack[k])
        if ratios_disminucion:
            disminucion = min(ratios_disminucion)

        ratios_aumento = []
        for k in range(len(rhs_final)):
            if columna_slack[k] < -1e-9:
                ratios_aumento.append(-rhs_final[k] / columna_slack[k])
        if ratios_aumento:
            aumento = min(ratios_aumento)

        rangos_b[constraint_label] = {
            'actual': b_actual,
            'aumento': aumento,
            'disminucion': disminucion,
            'rango': (b_actual - disminucion, b_actual + aumento)
        }

    return rangos_b

tabla_final, variables_basicas_finales, iteraciones = simplex(tabla_inicial, variables_basicas_iniciales, columnas)

if tabla_final is not None:
    print("\n  TABLA FINAL ÓPTIMA  ")
    df_final = pd.DataFrame(tabla_final, index=variables_basicas_finales + ['Z'], columns=columnas)
    print(df_final.round(4)) 

    print("\n  SOLUCIÓN ÓPTIMA  ")
    valor_optimo = tabla_final[-1, -1]
    print(f"Valor óptimo de Z: {valor_optimo:.4f}")

    print("Variables básicas:")
    solucion = {var: 0.0 for var in columnas[:-1]}
    rhs_final = tabla_final[:-1, -1]
    for i, var_basica in enumerate(variables_basicas_finales):
        solucion[var_basica] = rhs_final[i]
        print(f"  {var_basica} = {rhs_final[i]:.4f}")

    print("Variables no básicas (valor 0):")
    for var in columnas[:-1]:
        if var not in variables_basicas_finales:
            print(f"  {var} = 0.0000")

    print("\n PRECIOS SOMBRA ")
    precios = calcular_precios_sombra(tabla_final, num_decision_vars, num_slack_vars, columnas)
    for i in range(num_slack_vars):
         slack_var = f's{i+1}'
         print(f"  Restricción {i+1} ({slack_var}): {precios[slack_var]:.4f}")

    print("\n  ANÁLISIS DE SENSIBILIDAD: RANGOS DE OPTIMALIDAD (Coeficientes c)  ")
    rangos_c = calcular_rangos_optimalidad_c(tabla_final, c_original_obj, variables_basicas_finales, columnas)
    for var, data in rangos_c.items():
        print(f"  Variable {var}:")
        print(f"    Coeficiente actual: {data['actual']:.4f}")
        print(f"    Aumento permitido: {data['aumento'] if data['aumento'] != np.inf else 'Infinito':.4f}")
        print(f"    Disminución permitida: {data['disminucion'] if data['disminucion'] != np.inf else 'Infinito':.4f}")
        rango_inf = data['rango'][0] if data['rango'][0] != -np.inf else '-Infinito'
        rango_sup = data['rango'][1] if data['rango'][1] != np.inf else 'Infinito'
        print(f"    Rango de optimalidad: ({rango_inf}, {rango_sup})")


    print("\n  ANÁLISIS DE SENSIBILIDAD: RANGOS DE FACTIBILIDAD (Lado Derecho b)  ")
    rangos_b = calcular_rangos_factibilidad_b(tabla_final, b_original, variables_basicas_finales, columnas)
    for constraint, data in rangos_b.items():
        print(f"  {constraint}:")
        print(f"    Valor actual RHS: {data['actual']:.4f}")
        print(f"    Aumento permitido: {data['aumento'] if data['aumento'] != np.inf else 'Infinito':.4f}")
        print(f"    Disminución permitida: {data['disminucion'] if data['disminucion'] != np.inf else 'Infinito':.4f}")
        rango_inf = data['rango'][0] if data['rango'][0] != -np.inf else '-Infinito'
        rango_sup = data['rango'][1] if data['rango'][1] != np.inf else 'Infinito'
        print(f"    Rango de factibilidad: ({rango_inf}, {rango_sup})")

else:
    print("\n Sin solucion.")