import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

# Configuración elegante para LaTeX (idéntica a la del prior RBF)
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# Mismo dominio que en la figura RBF, para que sean comparables
X = np.linspace(0, 10, 500).reshape(-1, 1)

# Kernel Matern 5/2 con los MISMOS hiperparametros que el prior RBF:
# sigma_f^2 = 1.0 (ConstantKernel) y ell = 1.5 (length_scale)
kernel = C(1.0) * Matern(length_scale=1.5, nu=2.5)
gp_prior = GaussianProcessRegressor(kernel=kernel, optimizer=None)

# Media (0) y desviacion estandar a priori
y_mean, y_std = gp_prior.predict(X, return_std=True)

# Mismas 3 muestras y misma semilla que el prior RBF, para comparar la regularidad
muestras = gp_prior.sample_y(X, n_samples=3, random_state=42)

# Dibujamos
plt.figure(figsize=(8, 5))
plt.plot(X, y_mean, 'k--', lw=2, label='Media predictiva $\\mu(\\mathbf{x}) = 0$')
plt.fill_between(X.ravel(), y_mean - 2*y_std, y_mean + 2*y_std,
                 color='dodgerblue', alpha=0.2, label='Incertidumbre (95%)')
plt.plot(X, muestras, lw=1.5, alpha=0.7)

plt.title("Proceso Gaussiano A Priori (kernel Mat\u00e9rn 5/2)")
plt.xlabel("$\\mathbf{x}$")
plt.ylabel("$f(\\mathbf{x})$")
plt.ylim(-4, 4)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("figura1b_gp_prior_matern.pdf", format='pdf', bbox_inches='tight')
print("Figura guardada: figura1b_gp_prior_matern.pdf")
