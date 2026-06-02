import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Configuración elegante para LaTeX
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

X = np.linspace(0, 10, 500).reshape(-1, 1)

# Observaciones ruidosas
X_train = np.array([1.0, 3.0, 4.5, 6.0, 7.5, 8.5]).reshape(-1, 1)
np.random.seed(42)
varianza_ruido = 0.5
y_train = (X_train.ravel() * np.sin(X_train.ravel())) + np.random.normal(0, np.sqrt(varianza_ruido), size=6)

# Modelo GP con ruido (alpha)
kernel = C(1.0) * RBF(length_scale=1.5)
gp_post = GaussianProcessRegressor(kernel=kernel, alpha=varianza_ruido, n_restarts_optimizer=10)
gp_post.fit(X_train, y_train)

# Predicción y muestreo
y_mean, y_std = gp_post.predict(X, return_std=True)
muestras = gp_post.sample_y(X, n_samples=3, random_state=42)

# Dibujamos
plt.figure(figsize=(8, 5))
plt.plot(X, X.ravel() * np.sin(X.ravel()), 'r:', lw=1.5, label='Función real latente $f(\mathbf{x})$')
plt.plot(X, y_mean, 'k--', lw=2, label='Media predictiva $\mu_*(\mathbf{x})$')
plt.fill_between(X.ravel(), y_mean - 2*y_std, y_mean + 2*y_std, color='dodgerblue', alpha=0.2, label='Incertidumbre (95%)')
plt.plot(X, muestras, lw=1.5, alpha=0.6)
plt.plot(X_train, y_train, 'ko', markersize=7, label='Observaciones $\mathbf{y}$')

plt.title("Proceso Gaussiano A Posteriori (con ruido)")
plt.xlabel("$\mathbf{x}$")
plt.ylabel("$f(\mathbf{x})$")
plt.ylim(-6, 8)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("figura2_gp_posterior.pdf", format='pdf', bbox_inches='tight')
plt.show()


