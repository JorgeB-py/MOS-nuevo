from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, Constraint, SolverFactory, value, maximize
import time

C = [5, 8, 3, 7, 6, 9, 4, 10, 2, 11]
A = [
    [1, 2, 1, 1, 0, 0, 3, 1, 2, 1],
    [2, 1, 0, 2, 1, 1, 0, 3, 1, 2],
    [1, 1, 2, 0, 2, 1, 1, 0, 3, 1],
    [0, 2, 1, 1, 0, 2, 1, 1, 1, 1],
    [2, 0, 1, 1, 2, 1, 1, 0, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 2, 1, 0, 1, 2, 1, 1, 0],
    [1, 0, 1, 2, 1, 0, 1, 2, 1, 1]
]
b = [50, 60, 55, 40, 45, 70, 65, 50]

model = ConcreteModel()
model.I = range(1, 11)
model.J = range(1, 9)
model.x = Var(model.I, domain=NonNegativeReals)
model.obj = Objective(expr=sum(C[i-1] * model.x[i] for i in model.I), sense=maximize)

def restriccion_rule(m, j):
    return sum(A[j-1][i-1] * m.x[i] for i in m.I) <= b[j-1]

model.restricciones = Constraint(model.J, rule=restriccion_rule)

solver = SolverFactory('glpk')
start = time.time()
results = solver.solve(model, tee=True)
end = time.time()

print(f"\n>>> Tiempo de solución: {end - start:.4f} segundos")
print(">>> Estado del solver:", results.solver.status, results.solver.termination_condition)
print(">>> Valor óptimo Z =", value(model.obj))
print(">>> Variables óptimas:")
for i in model.I:
    print(f"  x[{i}] = {value(model.x[i])}")

#Esta es la implementación para el problema principal, las variaciones no fueron mostradas acá, pero sí en el documento.