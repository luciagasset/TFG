# -*- coding: utf-8 -*-
"""
====================================================================
 Optimizacion Bayesiana sobre la funcion de Forrester
 Capitulo de aplicaciones practicas - TFG Grado en Matematicas
====================================================================

Notas metodologicas:
  * Ajuste de hiperparametros por MAP: log-verosimilitud marginal
    regularizada con un prior LogNormal sobre la longitud de escala.
    Evita las soluciones degeneradas (l -> 0) de la maxima verosimilitud
    pura cuando hay pocos datos, y es coherente con el marco bayesiano.
  * Entrenamiento unificado con fit_gpytorch_mll (L-BFGS) en todo el cap.
  * El analisis de kernels e hiperparametros (Secs. 2-4) usa una
    inicializacion de 9 puntos: con suficientes datos la MLL identifica
    sin ambiguedad la escala optima.
  * El estudio de convergencia (Sec. 5) parte de una inicializacion
    deliberadamente reducida (4 puntos) para que el algoritmo deba
    explorar de forma efectiva y se aprecien las diferencias entre
    funciones de adquisicion.
  * El gap se calcula sobre f(x_hat) SIN ruido: x_hat se elige por la
    observacion ruidosa (lo unico que el algoritmo observa), pero el
    rendimiento se mide con el valor verdadero de la funcion.
  * La tabla de la Sec. 5 promedia sobre 25 semillas; las figuras
    ilustrativas usan una semilla fija, declarada en el texto.
====================================================================
"""
import torch, warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.double)

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.priors import LogNormalPrior
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import (ProbabilityOfImprovement,
                                 ExpectedImprovement, UpperConfidenceBound)
from botorch.optim import optimize_acqf

# --------------------------- ESTILO -------------------------------
mpl.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 12, "legend.fontsize": 10, "axes.grid": False,
    "axes.edgecolor": "black", "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "lines.linewidth": 1.5, "figure.dpi": 150, "savefig.bbox": "tight"})
C_REAL="#000000"; C_GP="#0055FF"; C_OBS="#FF1A1A"
C_ACQ1="#00D000"; C_ACQ2="#CC00FF"; C_ACQ3="#00CCCC"

# ------------------------ CONFIGURACION ---------------------------
SEED_FIG   = 42    # semilla de las figuras de las Secs. 1-4
SEED_BO    = 1     # semilla de la figura ilustrativa de la Sec. 5
N_SEEDS    = 25    # semillas del experimento estadistico (Sec. 5)
N_INIT_GP  = 9     # puntos iniciales para el analisis del GP (Secs. 2-4)
N_INIT_BO  = 4     # puntos iniciales para el estudio de convergencia
ITER       = 10    # iteraciones del bucle de BO
NIVEL_RUIDO = 0.2  # desviacion tipica del ruido aditivo
BETA_UCB   = 2.5   # parametro de exploracion de UCB en el bucle

# --------------------------- UTILIDADES ---------------------------
def forrester(x):
    """Opuesta de la funcion de Forrester. Maximizamos -f(x)."""
    return -((6*x - 2)**2) * torch.sin(12*x - 4)

def lengthscale_of(gp):
    k = gp.covar_module
    while hasattr(k, 'base_kernel'):
        k = k.base_kernel
    return k.lengthscale.item()

def nuevo_kernel(familia='matern'):
    """Kernel con prior LogNormal debilmente informativo sobre la
    longitud de escala (ajuste por MAP)."""
    prior = LogNormalPrior(loc=0.0, scale=1.0)
    if familia == 'matern':
        base = MaternKernel(nu=2.5, lengthscale_prior=prior)
    else:
        base = RBFKernel(lengthscale_prior=prior)
    return ScaleKernel(base)

def ajustar_gp(X, Y, familia='matern'):
    """Construye y ajusta un GP maximizando la MLL regularizada."""
    gp = SingleTaskGP(X, Y, covar_module=nuevo_kernel(familia),
                      outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval(); mll.eval()
    with torch.no_grad():
        valor_mll = mll(gp(X), gp.train_targets).sum().item()
    return gp, valor_mll

# ============================================================
# 1. FUNCION OBJETIVO, OPTIMO ANALITICO Y DATOS
# ============================================================
X_plot = torch.linspace(0, 1, 200, dtype=torch.double).unsqueeze(-1)
Y_plot_real = forrester(X_plot)

# Optimo analitico por malla muy fina
_xx = torch.linspace(0, 1, 200001, dtype=torch.double)
_ff = forrester(_xx)
X_OPTIMO = _xx[_ff.argmax()].item()
F_OPTIMO = _ff.max().item()
print(f"Optimo analitico:  x* = {X_OPTIMO:.4f}   f* = {F_OPTIMO:.4f}")

# Datos para el analisis del GP (Secs. 2-4)
torch.manual_seed(SEED_FIG)
X_gp = torch.rand(N_INIT_GP, 1, dtype=torch.double)
Y_gp = forrester(X_gp) + NIVEL_RUIDO * torch.randn_like(X_gp)

# ---- FIGURA 1: problema inicial ----
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(X_plot.numpy(), Y_plot_real.numpy(), color=C_REAL, ls='--',
        label=r'Opuesta de Forrester $-f(x)$')
ax.scatter(X_gp.numpy(), Y_gp.numpy(), color=C_OBS, s=40, zorder=5,
           edgecolors='black', linewidths=0.5,
           label=r'Observaciones $y=-f(x)+\epsilon$')
for i in range(N_INIT_GP):
    ax.plot([X_gp[i].item()]*2,
            [forrester(X_gp[i]).item(), Y_gp[i].item()],
            color='gray', ls=':', alpha=0.8)
ax.set_xlabel('$x$'); ax.set_ylabel('$-f(x)$'); ax.legend()
plt.savefig("1_forrester_inicial.pdf"); plt.close()

# ============================================================
# 2. COMPARATIVA DE FAMILIAS DE KERNELS
#    Ambos kernels ajustan su MLL: comparativa justa.
# ============================================================
print("\n--- Seccion 2: comparativa de kernels ---")
torch.manual_seed(SEED_FIG)
gp_rbf, mll_rbf = ajustar_gp(X_gp, Y_gp, 'rbf')
gp_mat, mll_mat = ajustar_gp(X_gp, Y_gp, 'matern')
with torch.no_grad():
    p_rbf = gp_rbf(X_plot); inf_r, sup_r = p_rbf.confidence_region()
    p_mat = gp_mat(X_plot); inf_m, sup_m = p_mat.confidence_region()
med_rbf, med_mat = p_rbf.mean.numpy(), p_mat.mean.numpy()
print(f"  RBF       -> MLL = {mll_rbf:7.3f}   l = {lengthscale_of(gp_rbf):.4f}")
print(f"  Matern52  -> MLL = {mll_mat:7.3f}   l = {lengthscale_of(gp_mat):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, med, inf, sup, tit in [
        (axes[0], med_rbf, inf_r, sup_r, r"Kernel RBF ($C^\infty$)"),
        (axes[1], med_mat, inf_m, sup_m, "Kernel Mat\u00e9rn 5/2 ($C^2$)")]:
    ax.plot(X_plot.numpy(), Y_plot_real.numpy(), color=C_REAL, ls='--', alpha=0.6)
    ax.fill_between(X_plot.numpy().flatten(), inf.numpy(), sup.numpy(),
                    color=C_GP, alpha=0.15)
    ax.plot(X_plot.numpy(), med, color=C_GP, lw=2)
    ax.scatter(X_gp.numpy(), Y_gp.numpy(), color=C_OBS, s=40, zorder=5,
               edgecolors='black', linewidths=0.5)
    ax.set_title(tit); ax.set_xlabel('$x$')
axes[0].set_ylabel('Predicci\u00f3n GP')
plt.tight_layout(); plt.savefig("2_comparativa_kernels.pdf"); plt.close()

# ============================================================
# 3. EFECTO DE LA LONGITUD DE ESCALA
# ============================================================
print("\n--- Seccion 3: efecto de la longitud de escala ---")
torch.manual_seed(SEED_FIG)
gp_opt, mll_opt = ajustar_gp(X_gp, Y_gp, 'matern')
L_OPT     = lengthscale_of(gp_opt)
OUT_OPT   = gp_opt.covar_module.outputscale.item()
NOISE_OPT = gp_opt.likelihood.noise.item()
print(f"  l_opt = {L_OPT:.4f}   outputscale = {OUT_OPT:.4f}"
      f"   noise = {NOISE_OPT:.4f}   MLL = {mll_opt:.4f}")

configuraciones = [("Sobreadaptado", 0.02),
                   ("Optimizado", L_OPT),
                   ("Subadaptado", 2.0)]
print("  Tabla MLL por configuracion:")
tabla_mll = []
for nombre, l_val in configuraciones:
    gp = SingleTaskGP(X_gp, Y_gp,
                      covar_module=ScaleKernel(MaternKernel(nu=2.5)),
                      outcome_transform=Standardize(m=1))
    gp.covar_module.base_kernel.lengthscale = l_val
    gp.covar_module.outputscale = OUT_OPT
    gp.likelihood.noise = NOISE_OPT
    gp.eval()
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp); mll.eval()
    with torch.no_grad():
        v = mll(gp(X_gp), gp.train_targets).sum().item()
    tabla_mll.append((nombre, l_val, v))
    print(f"    {nombre:<14} l = {l_val:.4f}   MLL = {v:7.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
torch.manual_seed(10)
col_m = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e377c2"]
for i, (nombre, l_val) in enumerate(configuraciones):
    gp = SingleTaskGP(X_gp, Y_gp,
                      covar_module=ScaleKernel(MaternKernel(nu=2.5)),
                      outcome_transform=Standardize(m=1))
    gp.covar_module.base_kernel.lengthscale = l_val
    gp.covar_module.outputscale = OUT_OPT
    gp.likelihood.noise = NOISE_OPT
    gp.eval()
    with torch.no_grad():
        pr = gp(X_plot)
        media = pr.mean.numpy()
        inf, sup = pr.confidence_region()
        muestras = pr.sample(torch.Size([5])).numpy()
    ax = axes[i]
    ax.plot(X_plot.numpy(), Y_plot_real.numpy(), color=C_REAL, ls='--', alpha=0.5)
    ax.fill_between(X_plot.numpy().flatten(), inf.numpy(), sup.numpy(),
                    color=C_GP, alpha=0.1)
    for j in range(5):
        ax.plot(X_plot.numpy(), muestras[j], color=col_m[j], alpha=0.6, lw=1.2)
    ax.plot(X_plot.numpy(), media, color=C_GP, lw=2.5)
    ax.scatter(X_gp.numpy(), Y_gp.numpy(), color=C_OBS, s=50, zorder=10,
               edgecolors='black', linewidths=1)
    tt = (f"Optimizado ($\\ell={l_val:.4f}$)" if nombre == "Optimizado"
          else f"{nombre} ($\\ell={l_val:.2f}$)")
    ax.set_title(tt); ax.set_xlabel('$x$')
axes[0].set_ylabel('Funciones muestreadas $f(x)$')
plt.tight_layout(); plt.savefig("3_efecto_hyperparametros.pdf"); plt.close()

# ============================================================
# 4. FUNCIONES DE ADQUISICION (analisis estatico)
# ============================================================
print("\n--- Seccion 4: adquisicion estatica ---")
mejor_val = Y_gp.max().item()
X_acq = X_plot.unsqueeze(1)
fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
axes[0].plot(X_plot.numpy(), med_mat, color=C_GP, label='Media GP')
axes[0].fill_between(X_plot.numpy().flatten(), inf_m.numpy(), sup_m.numpy(),
                     color=C_GP, alpha=0.15)
axes[0].scatter(X_gp.numpy(), Y_gp.numpy(), color=C_OBS, zorder=5,
                edgecolors='black', linewidths=0.3)
axes[0].set_ylabel("GP a posteriori")
xis = [0.0, 0.1, 0.5]; betas = [0.1, 2.0, 5.0]
cols = [C_ACQ1, C_ACQ2, C_ACQ3]
with torch.no_grad():
    for xi, c in zip(xis, cols):
        axes[1].plot(X_plot.numpy(),
            ProbabilityOfImprovement(gp_mat, mejor_val+xi)(X_acq).numpy(),
            color=c, label=fr"$\xi={xi}$")
        axes[2].plot(X_plot.numpy(),
            ExpectedImprovement(gp_mat, mejor_val+xi)(X_acq).numpy(),
            color=c, label=fr"$\xi={xi}$")
    for b, c in zip(betas, cols):
        axes[3].plot(X_plot.numpy(),
            UpperConfidenceBound(gp_mat, beta=b)(X_acq).numpy(),
            color=c, label=fr"$\beta={b}$")
axes[1].set_ylabel("PI"); axes[1].legend(loc="upper right")
axes[2].set_ylabel("EI"); axes[2].legend(loc="upper right")
axes[3].set_ylabel("UCB"); axes[3].set_xlabel("$x$")
axes[3].legend(loc="upper right")
plt.tight_layout(); plt.savefig("4_adquisicion_estatica.pdf"); plt.close()

# ============================================================
# 5. ESTUDIO DE CONVERGENCIA: BUCLE DE OPTIMIZACION BAYESIANA
# ============================================================
FUNCS = ['PI', 'EI', 'UCB']
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

def correr_bo(acq_name, X0, Y0, guardar_historial=False):
    """Ejecuta ITER iteraciones de BO. Devuelve x_hat (mejor punto
    segun la observacion ruidosa) y el gap respecto al optimo real."""
    Xa, Ya = X0.clone(), Y0.clone()
    historial = []
    for t in range(ITER):
        gp = SingleTaskGP(Xa, Ya, covar_module=nuevo_kernel('matern'),
                          outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        mejor_y = Ya.max()
        if acq_name == 'PI':
            acq = ProbabilityOfImprovement(gp, mejor_y)
        elif acq_name == 'EI':
            acq = ExpectedImprovement(gp, mejor_y)
        else:
            acq = UpperConfidenceBound(gp, beta=BETA_UCB)
        nuevo_x, _ = optimize_acqf(acq, bounds=bounds, q=1,
                                   num_restarts=10, raw_samples=64)
        nuevo_y = (forrester(nuevo_x)
                   + NIVEL_RUIDO * torch.randn(1, 1, dtype=torch.double))
        if guardar_historial:
            gp.eval()
            with torch.no_grad():
                pr = gp(X_plot); inf, sup = pr.confidence_region()
                av = acq(X_plot.unsqueeze(1)).numpy()
            historial.append({'media': pr.mean.numpy(), 'inf': inf.numpy(),
                               'sup': sup.numpy(), 'acq': av,
                               'Xo': Xa.clone(), 'Yo': Ya.clone(),
                               'nx': nuevo_x.item()})
        Xa = torch.cat([Xa, nuevo_x]); Ya = torch.cat([Ya, nuevo_y])
    x_hat = Xa[Ya.argmax()].item()
    f_real = forrester(torch.tensor(x_hat)).item()
    return {'x_hat': x_hat, 'f_real': f_real, 'gap': abs(f_real - F_OPTIMO),
            'n_evals': Xa.numel(), 'historial': historial}

# ---- 5a. Figura ilustrativa (semilla fija SEED_BO) ----
print("\n--- Seccion 5: figura ilustrativa de las trayectorias ---")
torch.manual_seed(SEED_BO)
X_bo = torch.rand(N_INIT_BO, 1, dtype=torch.double)
Y_bo = forrester(X_bo) + NIVEL_RUIDO * torch.randn_like(X_bo)

resultados = {}
for acq_name in FUNCS:
    torch.manual_seed(SEED_BO)
    resultados[acq_name] = correr_bo(acq_name, X_bo, Y_bo,
                                     guardar_historial=True)
    r = resultados[acq_name]
    print(f"  {acq_name:<4} -> x_hat = {r['x_hat']:.4f}"
          f"   f(x_hat) = {r['f_real']:.4f}   gap = {r['gap']:.4f}")

muestra = [1, 3, 5, 8]
fig, axes = plt.subplots(6, 4, figsize=(14, 11), sharex=True)
plt.subplots_adjust(hspace=0.15, wspace=0.1)
for i, acq_name in enumerate(FUNCS):
    fg, fa = i*2, i*2 + 1
    for col, t in enumerate(muestra):
        d = resultados[acq_name]['historial'][t]
        ag, aa = axes[fg, col], axes[fa, col]
        ag.plot(X_plot.numpy(), Y_plot_real.numpy(), color=C_REAL,
                ls='--', alpha=0.4, lw=1.2)
        ag.fill_between(X_plot.numpy().flatten(), d['inf'], d['sup'],
                        color=C_GP, alpha=0.15)
        ag.plot(X_plot.numpy(), d['media'], color=C_GP, lw=1.5)
        ag.scatter(d['Xo'].numpy(), d['Yo'].numpy(), c=C_OBS,
                   edgecolors='black', linewidths=0.5, zorder=5)
        ag.axvline(d['nx'], color='black', ls=':', alpha=0.8, lw=1.2)
        ag.set_yticks([])
        if col == 0:
            ag.set_ylabel(acq_name, fontweight='bold', fontsize=12)
        if i == 0:
            ag.set_title(f"Iteraci\u00f3n {t}", fontsize=10)
        a = d['acq']
        an = (a - a.min()) / (a.max() - a.min() + 1e-9)
        aa.fill_between(X_plot.numpy().flatten(), 0, an,
                        color=C_ACQ1, alpha=0.15)
        aa.plot(X_plot.numpy(), an, color=C_ACQ1, lw=1.5)
        aa.axvline(d['nx'], color='black', ls=':', alpha=0.8, lw=1.2)
        aa.set_yticks([]); aa.set_ylim(0, 1.1)
        if i == len(FUNCS) - 1:
            aa.set_xlabel("$x$")
plt.savefig("5_trayectorias_grid.pdf"); plt.close()

# ---- 5b. Experimento estadistico (N_SEEDS semillas) ----
print(f"\n--- Seccion 5: experimento estadistico ({N_SEEDS} semillas) ---")
gaps = {f: [] for f in FUNCS}
for s in range(N_SEEDS):
    torch.manual_seed(2000 + s)
    X0 = torch.rand(N_INIT_BO, 1, dtype=torch.double)
    Y0 = forrester(X0) + NIVEL_RUIDO * torch.randn_like(X0)
    for acq_name in FUNCS:
        torch.manual_seed(2000 + s)
        gaps[acq_name].append(correr_bo(acq_name, X0, Y0)['gap'])

print("\n" + "=" * 66)
print(f"{'Funcion':<14}{'Gap medio':>12}{'Desv. tip.':>12}"
      f"{'Mediana':>11}{'Gap max':>11}")
print("=" * 66)
estadisticas = {}
for f in FUNCS:
    g = np.array(gaps[f])
    estadisticas[f] = {'media': g.mean(), 'std': g.std(),
                       'mediana': np.median(g), 'max': g.max()}
    et = f if f != 'UCB' else f'UCB (b={BETA_UCB})'
    print(f"{et:<14}{g.mean():>12.4f}{g.std():>12.4f}"
          f"{np.median(g):>11.4f}{g.max():>11.4f}")
print("=" * 66)
print(f"Config: {N_INIT_BO} pts iniciales + {ITER} iter, "
      f"{N_SEEDS} semillas, ruido sigma={NIVEL_RUIDO}")
print("\nTodas las figuras PDF se han generado correctamente.")

# ============================================================
# 6. ESTUDIOS DE SENSIBILIDAD ADICIONALES
#    (sigma_f, sigma_y, numero de datos y numero de iteraciones)
# ============================================================

# ---- 6a. Efecto de sigma_f^2 (outputscale) y sigma_y^2 (noise) ----
# Se fijan los demas hiperparametros
# en su valor optimo y se varia uno cada vez.
print("\n--- Seccion 6a: sensibilidad a sigma_f y sigma_y ---")

def _posterior_fijando(lengthscale, outputscale, noise):
    gp = SingleTaskGP(X_gp, Y_gp,
                      covar_module=ScaleKernel(MaternKernel(nu=2.5)),
                      outcome_transform=Standardize(m=1))
    gp.covar_module.base_kernel.lengthscale = lengthscale
    gp.covar_module.outputscale = outputscale
    gp.likelihood.noise = noise
    gp.eval()
    with torch.no_grad():
        pr = gp(X_plot)
        return pr.mean.numpy(), pr.confidence_region()

fila_sf = [(r"$\sigma_f^2$ bajo (%.2f)" % (OUT_OPT*0.15), OUT_OPT*0.15, NOISE_OPT),
           (r"$\sigma_f^2$ optimizado (%.2f)" % OUT_OPT, OUT_OPT,       NOISE_OPT),
           (r"$\sigma_f^2$ alto (%.2f)" % (OUT_OPT*6.0), OUT_OPT*6.0,   NOISE_OPT)]
fila_sy = [(r"$\sigma_y^2$ bajo (0,002)",      OUT_OPT, 0.002),
           (r"$\sigma_y^2$ intermedio (0,04)", OUT_OPT, 0.04),
           (r"$\sigma_y^2$ alto (0,40)",       OUT_OPT, 0.40)]

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for fila, conjunto in enumerate([fila_sf, fila_sy]):
    for col, (nombre, out, noi) in enumerate(conjunto):
        media, (inf, sup) = _posterior_fijando(L_OPT, out, noi)
        ax = axes[fila, col]
        ax.plot(X_plot.numpy(), Y_plot_real.numpy(), color=C_REAL, ls='--', alpha=0.5)
        ax.fill_between(X_plot.numpy().flatten(), inf.numpy(), sup.numpy(),
                        color=C_GP, alpha=0.12)
        ax.plot(X_plot.numpy(), media, color=C_GP, lw=2.2)
        ax.scatter(X_gp.numpy(), Y_gp.numpy(), color=C_OBS, s=42, zorder=10,
                   edgecolors='black', linewidths=0.8)
        ax.set_title(nombre)
        if fila == 1:
            ax.set_xlabel('$x$')
    axes[fila, 0].set_ylabel('Predicci\u00f3n GP')
plt.tight_layout(); plt.savefig("6_sensibilidad_sf_sy.pdf"); plt.close()

# ---- 6b. Efecto del numero de datos iniciales y de iteraciones ----
# Convergencia del gap con la Mejora Esperada, para tres tamanñs del diseno inicial, promediada sobre
# varias semillas. El eje x = numero de iteraciones; cada curva = un
# tamano de diseno inicial.
print("--- Seccion 6b: convergencia segun datos e iteraciones ---")

N_SEEDS_SENS = 15          # subir a N_SEEDS (25) para igualar la Sec. 5
INITS        = [2, 4, 8]   # tamaños del diseno inicial a comparar
SEED_BASE_S  = 3000        # semilla base, declarada para reproducibilidad
COL_INIT     = {2: "#d62728", 4: C_GP, 8: "#2ca02c"}

def convergencia_ei(X0, Y0, n_iter):
    """Gap del incumbente |f(x_hat)-f*| tras cada iteracion, con EI.
    Coherente con correr_bo: x_hat se elige por la observacion ruidosa
    y el gap se mide con el valor exacto de la funcion."""
    Xa, Ya = X0.clone(), Y0.clone()
    gaps = []
    for t in range(n_iter):
        gp = SingleTaskGP(Xa, Ya, covar_module=nuevo_kernel('matern'),
                          outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acq = ExpectedImprovement(gp, Ya.max())
        nuevo_x, _ = optimize_acqf(acq, bounds=bounds, q=1,
                                   num_restarts=10, raw_samples=64)
        nuevo_y = (forrester(nuevo_x)
                   + NIVEL_RUIDO * torch.randn(1, 1, dtype=torch.double))
        Xa = torch.cat([Xa, nuevo_x]); Ya = torch.cat([Ya, nuevo_y])
        x_hat = Xa[Ya.argmax()].item()
        gaps.append(abs(forrester(torch.tensor(x_hat)).item() - F_OPTIMO))
    return np.array(gaps)

curvas = {n: [] for n in INITS}
for s in range(N_SEEDS_SENS):
    for n in INITS:
        torch.manual_seed(SEED_BASE_S + s)
        X0 = torch.rand(n, 1, dtype=torch.double)
        Y0 = forrester(X0) + NIVEL_RUIDO * torch.randn_like(X0)
        curvas[n].append(convergencia_ei(X0, Y0, ITER))

fig, ax = plt.subplots(figsize=(8, 5))
it = np.arange(1, ITER + 1)
for n in INITS:
    arr = np.vstack(curvas[n])
    media = arr.mean(0)
    err = arr.std(0) / np.sqrt(arr.shape[0])
    ax.plot(it, media, color=COL_INIT[n], label=f"{n} puntos iniciales")
    ax.fill_between(it, media - err, media + err, color=COL_INIT[n], alpha=0.15)
ax.set_yscale('log')
ax.set_xlabel("Número de iteraciones de la optimización bayesiana")
ax.set_ylabel(r"Gap medio respecto al $|f(\hat{x})-f^*|$")
ax.legend()
plt.tight_layout(); plt.savefig("7_convergencia_presupuesto.pdf"); plt.close()
print("Figuras 6 y 7 generadas correctamente.")
