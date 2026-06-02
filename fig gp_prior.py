import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Configuración elegante para LaTeX
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# Dominio continuo
X = np.linspace(0, 10, 500).reshape(-1, 1)

# Kernel y modelo (sin datos)
kernel = 1.0 * RBF(length_scale=1.5)
gp_prior = GaussianProcessRegressor(kernel=kernel, optimizer=None)

# Extraemos media (0) y desviación estándar
y_mean, y_std = gp_prior.predict(X, return_std=True)

# Muestreamos funciones
muestras = gp_prior.sample_y(X, n_samples=3, random_state=42)

# Dibujamos
plt.figure(figsize=(8, 5))
plt.plot(X, y_mean, 'k--', lw=2, label='Media predictiva $\mu(\mathbf{x}) = 0$')
plt.fill_between(X.ravel(), y_mean - 2*y_std, y_mean + 2*y_std, color='dodgerblue', alpha=0.2, label='Incertidumbre (95%)')
plt.plot(X, muestras, lw=1.5, alpha=0.7)

plt.title("Proceso Gaussiano A Priori")
plt.xlabel("$\mathbf{x}$")
plt.ylabel("$f(\mathbf{x})$")
plt.ylim(-4, 4)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("figura1_gp_prior.pdf", format='pdf', bbox_inches='tight')
plt.show()