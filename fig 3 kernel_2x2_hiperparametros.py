
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Configuración elegante para tu TFG
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# 1. Definimos el dominio y unas observaciones de prueba similares a la imagen
X = np.linspace(-5, 5, 500).reshape(-1, 1)
X_train = np.array([-4, -3, -1, 0, 1.5]).reshape(-1, 1)
y_train = np.array([-0.5, -2.5, 0.5, -0.5, 1.5])

# 2. Preparamos la figura 2x2
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
axs = axs.ravel() # Aplanamos el array de ejes para iterar fácilmente

# Configuraciones de hiperparámetros: (longitud_escala l, varianza sigma_f^2)
configs = [
    (0.2, 1.0),   # Arriba-Izda: l pequeño (nervioso)
    (3.0, 1.0),   # Arriba-Dcha: l grande (muy suave)
    (1.0, 0.05),  # Abajo-Izda: sigma_f^2 pequeño (baja amplitud)
    (1.0, 5.0)    # Abajo-Dcha: sigma_f^2 grande (alta amplitud)
]

titulos = [
    r"$l = 0.2, \sigma_f^2 = 1.0$",
    r"$l = 3.0, \sigma_f^2 = 1.0$",
    r"$l = 1.0, \sigma_f^2 = 0.05$",
    r"$l = 1.0, \sigma_f^2 = 5.0$"
]

for i, (l, var) in enumerate(configs):
    # Definimos el kernel: C(varianza) * RBF(longitud)
    kernel = C(var) * RBF(length_scale=l)
    
    # alpha muy bajo (1e-4) para simular observaciones casi exactas sin inestabilidad numérica
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, optimizer=None)
    gp.fit(X_train, y_train)
    y_mean, y_std = gp.predict(X, return_std=True)
    
    # Dibujamos
    axs[i].plot(X, y_mean, color='steelblue', lw=1.8, label='Media $\mu_*$')
    axs[i].fill_between(X.ravel(), y_mean - 2*y_std, y_mean + 2*y_std, color='lightsteelblue', alpha=0.4, label='Incertidumbre (95%)')
    axs[i].plot(X_train, y_train, 'rx', markersize=8, markeredgewidth=1.5, label='Observaciones')
    
    axs[i].set_title(titulos[i])
    axs[i].set_xlim([-5, 5])
    
    # Solo ponemos etiquetas en los bordes exteriores para no ensuciar la gráfica
    if i > 1:
        axs[i].set_xlabel("$x$")
    if i % 2 == 0:
        axs[i].set_ylabel("$f(x)$")
        
    axs[i].legend(loc='best', framealpha=0.9)

plt.tight_layout()
plt.savefig("figura3_kernel_2x2.pdf", format='pdf', bbox_inches='tight')
plt.show()