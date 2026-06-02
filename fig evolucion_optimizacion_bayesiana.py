import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# 1. Configuración de estilo académico para el TFG
SEED = 8
random.seed(SEED)
np.random.seed(SEED)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.titlesize": 12
})

# 2. Definición del problema
bounds = np.array([[-1.0, 2.0]])
noise = 0.2

def f(x, noise=0):
    return -np.sin(3*x) - x**2 + 0.7*x + noise*np.random.randn(*x.shape)

# Datos iniciales
X_init = np.array([[-0.7], [1.6]])
Y_init = f(X_init, noise=noise)

# Dominio para plotear (CORREGIDO PARA NUMPY 1.25+)
X_plot = np.arange(bounds[0, 0], bounds[0, 1], 0.01).reshape(-1, 1)
Y_plot = f(X_plot, noise=0)

# 3. Funciones del algoritmo (Expected Improvement y Optimizador)
def expected_improvement(X, X_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)
    
    # CORREGIDO: Aseguramos que mu y sigma tengan la misma dimensión (columna)
    sigma = sigma.reshape(-1, 1)
    mu = mu.reshape(-1, 1)
    
    mu_sample_opt = np.max(mu_sample)
    
    with np.errstate(divide='warn', invalid='ignore'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def propose_location(acquisition, X_sample, gpr, bounds, n_restarts=25):
    dim = X_sample.shape[1]
    min_val = float('inf')
    min_x = None

    def min_obj(X):
        return -acquisition(X.reshape(-1, dim), X_sample, gpr)

    for x0 in np.random.uniform(bounds[0, 0], bounds[0, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        
        current_val = res.fun[0] if isinstance(res.fun, np.ndarray) else res.fun
        
        if current_val < min_val:
            min_val = current_val
            min_x = res.x           
            
    return min_x.reshape(-1, 1)

# 4. Funciones de ploteo adaptadas
def plot_approximation(gpr, X_plot, Y_plot, X_sample, Y_sample, X_next=None, show_legend=False):
    mu, std = gpr.predict(X_plot, return_std=True)
    plt.fill_between(X_plot.ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     color='blue', alpha=0.15, label='Incertidumbre $\pm 1.96\sigma_*$') 
    plt.plot(X_plot, Y_plot, 'k--', lw=1.5, alpha=0.6, label='Función objetivo $f(x)$')
    plt.plot(X_plot, mu, 'b-', lw=1.5, label='Media predictiva $\mu_*(x)$')
    plt.plot(X_sample, Y_sample, 'kx', mew=2, markersize=7, label='Observaciones $\mathcal{D}_n$')
    
    if X_next is not None:
        plt.axvline(x=X_next, ls=':', c='r', lw=2, label='Siguiente $\mathbf{x}_{n+1}$')
    if show_legend:
        plt.legend(loc='lower left')

def plot_acquisition(X_plot, acq_value, X_next, show_legend=False):
    plt.plot(X_plot, acq_value, 'g-', lw=1.5, label='Mejora Esperada $\\alpha_{EI}(x)$')
    plt.fill_between(X_plot.ravel(), 0, acq_value.ravel(), color='green', alpha=0.1)
    plt.axvline(x=X_next, ls=':', c='r', lw=2, label='Máximo global')
    if show_legend:
        plt.legend(loc='upper right')

# 5. Ejecución y visualización de la evolución
m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

X_sample = X_init.copy()
Y_sample = Y_init.copy()

n_iter = 3

plt.figure(figsize=(10, n_iter * 3.5)) 
plt.subplots_adjust(hspace=0.4, wspace=0.2)

for i in range(n_iter):
    gpr.fit(X_sample, Y_sample)
    
    # Calcular siguiente punto y evaluar
    X_next = propose_location(expected_improvement, X_sample, gpr, bounds)
    Y_next = f(X_next, noise)
    
    # Izquierda: Modelo
    plt.subplot(n_iter, 2, 2 * i + 1)
    plot_approximation(gpr, X_plot, Y_plot, X_sample, Y_sample, X_next, show_legend=(i==0))
    plt.title(f'Iteración {i+1}: Proceso Gaussiano')
    plt.ylabel('Valor $y$')
    if i == n_iter - 1: plt.xlabel('Espacio de búsqueda $x$')
    
    # Derecha: Adquisición
    plt.subplot(n_iter, 2, 2 * i + 2)
    acq_val = expected_improvement(X_plot, X_sample, gpr)
    plot_acquisition(X_plot, acq_val, X_next, show_legend=(i==0))
    plt.title(f'Iteración {i+1}: Función de Adquisición')
    plt.ylabel('Utilidad $\\alpha_{EI}$')
    if i == n_iter - 1: plt.xlabel('Espacio de búsqueda $x$')
    
    # Actualizar dataset
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))

# Guardar en PDF vectorial para el TFG
plt.savefig('evolucion_optimizacion_bayesiana.pdf', format='pdf', bbox_inches='tight')
plt.show()