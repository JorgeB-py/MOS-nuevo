import numpy as np

class SimplexDual:
    def __init__(self, A, b, c, senses):
        self.A = A.copy().astype(float)
        self.b = b.copy().astype(float)
        self.c = c.copy().astype(float)
        self.senses = senses.copy()
        self._standardize()
        self._build_tableau()
    
    def _standardize(self):
        # Asegurar b >= 0
        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.A[i, :] *= -1
                self.b[i] *= -1
                self.senses[i] = {'<=':'>=','>=':'<=','=':'='}[self.senses[i]]
        # Añadir variables de holgura, exceso y artificiales
        m, n_orig = self.A.shape
        self.artificials = []
        col_index = n_orig
        for i, sense in enumerate(self.senses):
            if sense == '<=':
                # holgura
                col = np.zeros(m); col[i] = 1
                self.A = np.hstack([self.A, col.reshape(-1,1)])
                col_index += 1
            elif sense == '>=':
                # exceso y artificial
                col_s = np.zeros(m); col_s[i] = -1
                col_a = np.zeros(m); col_a[i] = 1
                self.A = np.hstack([self.A, col_s.reshape(-1,1), col_a.reshape(-1,1)])
                self.artificials.append(col_index+1)
                col_index += 2
            else:
                # artificial
                col_a = np.zeros(m); col_a[i] = 1
                self.A = np.hstack([self.A, col_a.reshape(-1,1)])
                self.artificials.append(col_index)
                col_index += 1
        # Costos fase I
        self.c_phase1 = np.zeros(col_index)
        for j in self.artificials:
            self.c_phase1[j] = -1
        # Base inicial
        self.base = []
        slack_idx = n_orig
        art_idx = list(self.artificials)
        for sense in self.senses:
            if sense == '<=':
                self.base.append(slack_idx)
                slack_idx += 1
            elif sense == '>=':
                # base es variable artificial
                slack_idx += 1
                self.base.append(art_idx.pop(0))
                slack_idx += 0
            else:
                self.base.append(art_idx.pop(0))
        self.m, self.n = self.A.shape
    
    def _build_tableau(self):
        self.tableau = np.hstack([self.A, self.b.reshape(-1,1)])
    
    def _pivot(self, row, col):
        pv = self.tableau[row, col]
        self.tableau[row, :] /= pv
        for r in range(self.m):
            if r != row:
                self.tableau[r, :] -= self.tableau[r, col] * self.tableau[row, :]
    
    def _reduced_costs(self, c_vec):
        Cb = c_vec[self.base]
        Z = Cb @ self.tableau[:, :self.n]
        return c_vec - Z

    def phase_I(self):
        # Simplex primal para maximizar -W
        while True:
            rc = self._reduced_costs(self.c_phase1)
            if np.all(rc <= 1e-8):
                break
            j = int(np.argmax(rc))
            col = self.tableau[:, j]
            ratios = np.array([self.tableau[i,-1]/col[i] if col[i]>0 else np.inf for i in range(self.m)])
            i = int(np.argmin(ratios))
            self._pivot(i, j)
            self.base[i] = j

    def phase_II(self):
        # función objetivo real (max)
        c2 = np.concatenate([self.c, np.zeros(self.n - len(self.c))])
        # dual simplex
        while True:
            rhs = self.tableau[:, -1]
            if np.all(rhs >= -1e-8):
                break
            i = int(np.argmin(rhs))
            rc = self._reduced_costs(c2)
            candidates = [(rc[j]/self.tableau[i,j], j) for j in range(self.n) if self.tableau[i,j] < 0]
            j = min(candidates, key=lambda x: x[0])[1]
            self._pivot(i, j)
            self.base[i] = j
        self.c_phase2 = c2

    def solve(self):
        self.phase_I()
        self.phase_II()
        sol = np.zeros(self.n)
        for i, j in enumerate(self.base):
            sol[j] = self.tableau[i, -1]
        # extraer solo variables originales
        x = sol[:len(self.c)]
        Z_max = self.c @ x
        return x, Z_max

A = np.array([[2,1,-1],
              [1,-3,2],
              [1,1,1]], dtype=float)
b = np.array([10,5,15], dtype=float)
# Convertimos min Z=5x1-4x2+3x3 a max Z' = -Z
c = np.array([-5,4,-3], dtype=float)
senses = ['=', '>=', '<=']

sd = SimplexDual(A, b, c, senses)
x_opt, Zp_opt = sd.solve()
Z_min = -Zp_opt

print("Solución óptima (x1, x2, x3):", x_opt)
print("Valor mínimo de Z:", Z_min)