"""
====================================================================
 Optimizacion Bayesiana de hiperparametros de XGBoost
 Caso de estudio: deteccion de fraude en tarjetas de credito
 Capitulo de aplicaciones practicas - TFG Grado en Matematicas
====================================================================

Objetivo
--------
Maximizar el AUC-PR (area bajo la curva precision-recall) de un modelo
XGBoost utilizando la optimizacion bayesiana para optimizar los hiperparámetros del modelo.

Metodologia
-----------
  * Funcion objetivo: AUC-PR medio en validacion cruzada estratificada
    de k pliegues. Promediar sobre los pliegues estabiliza la senal y
    evita el sobreajuste a una unica particion de validacion.
  * Modelo sustituto: un unico Proceso Gaussiano (kernel
    Matern 5/2 con ARD y ajuste MAP mediante un prior LogNormal sobre
    la longitud de escala). El MISMO GP que dirige la optimizacion es
    el que se emplea para las visualizaciones: las graficas muestran
    el modelo real, no una reconstruccion a posteriori.
  * Funcion de adquisicion: Expected Improvement (EI). El bucle de BO
    se implementa de forma explicita (GP -> EI -> optimize_acqf), en
    coherencia directa con el marco teorico del trabajo.
  * El espacio de busqueda se normaliza al hipercubo [0,1]^d; cada
    hiperparametro se transforma a su escala natural al evaluar.
  * El desbalanceo se gestiona dejando que el propio BO optimice
    scale_pos_weight como un hiperparametro mas.
  * El conjunto de test se aparta ANTES de cualquier otro paso y solo
    se evalua una vez, al final: no interviene en la optimizacion.

Hiperparametros optimizados (3, a modo ilustrativo)
---------------------------------------------------
  - learning_rate    (escala logaritmica)
  - max_depth        (entero)
  - scale_pos_weight (escala logaritmica; gestiona el desbalanceo)
El resto de hiperparametros de XGBoost se fijan en valores estandar.

Dataset
-------
Credit Card Fraud Detection (Kaggle). El archivo 'creditcard.csv' debe
encontrarse en la ruta indicada por RUTA_CSV. Contiene transacciones
con tarjeta de credito descritas por las variables V1..V28 (componentes
principales anonimizadas), el importe (Amount) y la etiqueta de fraude
(Class), con un fuerte desbalanceo entre clases.
====================================================================
"""
import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.double)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             confusion_matrix)
from xgboost import XGBClassifier

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import LogNormalPrior
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

# --------------------------- ESTILO -------------------------------
mpl.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 12, "legend.fontsize": 10, "axes.grid": False,
    "axes.edgecolor": "black", "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.6, "figure.dpi": 150, "savefig.bbox": "tight"})
C_GP = "#0055FF"; C_OBS = "#FF1A1A"; C_ACQ = "#00A000"; C_INIT = "#888888"

# ------------------------ CONFIGURACION ---------------------------
SEED      = 42
RUTA_CSV  = "creditcard.csv"   # ruta al CSV de Kaggle (Credit Card Fraud Detection)
N_INIT    = 8                  # evaluaciones iniciales (sondeo aleatorio)
N_ITER    = 25                 # iteraciones del bucle de BO
CV_FOLDS  = 4                  # pliegues de la validacion cruzada
N_ESTIM   = 300                # n_estimators fijo de XGBoost

np.random.seed(SEED)
torch.manual_seed(SEED)

# ==================================================================
# 1. CARGA DE DATOS
# ==================================================================
if not os.path.exists(RUTA_CSV):
    raise FileNotFoundError(
        f"No se encuentra el dataset en '{RUTA_CSV}'. Descarga el archivo "
        "'creditcard.csv' (Credit Card Fraud Detection, Kaggle) y colocalo "
        "en esa ruta antes de ejecutar el script.")

print(f"Cargando dataset desde '{RUTA_CSV}'...")
df = pd.read_csv(RUTA_CSV)
if "Time" in df.columns:
    df = df.drop(columns=["Time"])

X = df.drop(columns=["Class"]).values
y = df["Class"].values.astype(int)
print(f"Observaciones : {len(y)}")
print(f"Fraude        : {y.sum()}  ({100 * y.mean():.3f} %)")

# El conjunto de test se aparta AQUI, antes de cualquier otro paso.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)
print(f"Entrenamiento   : {len(y_tr)}   Test: {len(y_te)}")

# ==================================================================
# 2. ESPACIO DE HIPERPARAMETROS
# ==================================================================
# (nombre, minimo, maximo, escala): 'log', 'lin' o 'int'.
ESPACIO = [
    ("learning_rate",    0.01,   0.30,   "log"),
    ("max_depth",        3,      10,     "int"),
    ("scale_pos_weight", 1.0,    600.0,  "log"),
]
DIM = len(ESPACIO)
NOMBRES = [h[0] for h in ESPACIO]

# Hiperparametros de XGBoost que se mantienen fijos.
FIJOS = dict(n_estimators=N_ESTIM, subsample=0.9, colsample_bytree=0.9,
             min_child_weight=1.0, tree_method="hist",
             eval_metric="aucpr", n_jobs=-1, random_state=SEED)

def desnormalizar(u):
    """Transforma un punto u del hipercubo [0,1]^d a hiperparametros
    en su escala natural."""
    hp = {}
    for valor, (nombre, lo, hi, escala) in zip(u, ESPACIO):
        v = float(np.clip(valor, 0.0, 1.0))
        if escala == "log":
            x = np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo)))
        else:
            x = lo + v * (hi - lo)
        if escala == "int":
            x = int(round(x))
        hp[nombre] = x
    return hp

def a_natural(u_col, idx):
    """Transforma una columna normalizada a la escala natural del
    hiperparametro idx-esimo (para los ejes de las figuras)."""
    lo, hi, esc = ESPACIO[idx][1:]
    if esc == "log":
        return np.exp(np.log(lo) + u_col * (np.log(hi) - np.log(lo)))
    return lo + u_col * (hi - lo)

# ==================================================================
# 3. FUNCION OBJETIVO: AUC-PR EN VALIDACION CRUZADA ESTRATIFICADA
# ==================================================================
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

def objetivo(u):
    """AUC-PR medio en validacion cruzada estratificada del punto u
    del hipercubo normalizado. Es la funcion que el BO maximiza."""
    hp = desnormalizar(u)
    scores = []
    for idx_t, idx_v in skf.split(X_tr, y_tr):
        modelo = XGBClassifier(**FIJOS, **hp)
        modelo.fit(X_tr[idx_t], y_tr[idx_t])
        p = modelo.predict_proba(X_tr[idx_v])[:, 1]
        scores.append(average_precision_score(y_tr[idx_v], p))
    return float(np.mean(scores))

# ==================================================================
# 4. BUCLE DE OPTIMIZACION BAYESIANA
# ==================================================================
def ajustar_gp(train_X, train_Y):
    """GP con kernel Matern 5/2 (ARD) y prior LogNormal sobre la
    longitud de escala. Ajuste por MAP con fit_gpytorch_mll."""
    kernel = ScaleKernel(MaternKernel(
        nu=2.5, ard_num_dims=DIM,
        lengthscale_prior=LogNormalPrior(loc=0.0, scale=1.0)))
    gp = SingleTaskGP(train_X, train_Y, covar_module=kernel,
                      outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    return gp

print("\n" + "=" * 64)
print("OPTIMIZACION BAYESIANA")
print("=" * 64)

# --- 4a. Diseno inicial: muestreo aleatorio en el hipercubo ---
torch.manual_seed(SEED)
U = torch.rand(N_INIT, DIM, dtype=torch.double)
Y = torch.tensor([[objetivo(u.numpy())] for u in U], dtype=torch.double)
for i in range(N_INIT):
    print(f"  init {i+1:2d}/{N_INIT}   AUC-PR = {Y[i].item():.4f}")

bounds = torch.stack([torch.zeros(DIM, dtype=torch.double),
                      torch.ones(DIM, dtype=torch.double)])

# Historial: (n_eval, auc_pr, mejor_acumulado, fase, valor_EI)
historial = []
for i in range(N_INIT):
    historial.append((i + 1, Y[i].item(), Y[:i+1].max().item(),
                      "inicial", np.nan))

# --- 4b. Bucle iterativo guiado por Expected Improvement ---
for t in range(N_ITER):
    gp = ajustar_gp(U, Y)
    ei = ExpectedImprovement(gp, best_f=Y.max())
    cand, ei_val = optimize_acqf(ei, bounds=bounds, q=1,
                                 num_restarts=10, raw_samples=128)
    y_nuevo = objetivo(cand.squeeze(0).numpy())
    U = torch.cat([U, cand])
    Y = torch.cat([Y, torch.tensor([[y_nuevo]], dtype=torch.double)])
    mejor = Y.max().item()
    historial.append((N_INIT + t + 1, y_nuevo, mejor, "BO",
                      float(ei_val)))
    print(f"  iter {t+1:2d}/{N_ITER}   AUC-PR = {y_nuevo:.4f}"
          f"   |  mejor = {mejor:.4f}   |  EI = {float(ei_val):.2e}")

# --- 4c. Mejor configuracion encontrada ---
idx_mejor = Y.argmax().item()
u_mejor = U[idx_mejor].numpy()
hp_mejor = desnormalizar(u_mejor)
aucpr_cv = Y.max().item()
aucpr_init = max(h[1] for h in historial if h[3] == "inicial")

print("\n" + "=" * 64)
print("MEJOR CONFIGURACION ENCONTRADA")
print("=" * 64)
for k, v in hp_mejor.items():
    print(f"  {k:<18} = {v:.5f}" if isinstance(v, float)
          else f"  {k:<18} = {v}")
print(f"  AUC-PR (CV {CV_FOLDS}-fold) = {aucpr_cv:.4f}")

# ==================================================================
# 5. EVALUACION FINAL SOBRE EL CONJUNTO DE TEST
# ==================================================================
modelo_final = XGBClassifier(**FIJOS, **hp_mejor)
modelo_final.fit(X_tr, y_tr)
proba_te = modelo_final.predict_proba(X_te)[:, 1]
aucpr_te = average_precision_score(y_te, proba_te)
prec, rec, umbral = precision_recall_curve(y_te, proba_te)

# Matriz de confusion con el umbral que maximiza el F1
f1 = 2 * prec * rec / (prec + rec + 1e-12)
u_opt = umbral[max(0, np.argmax(f1[:-1]))]
pred_te = (proba_te >= u_opt).astype(int)
cm = confusion_matrix(y_te, pred_te)

print("\n" + "=" * 64)
print("RESULTADO SOBRE TEST")
print("=" * 64)
print(f"  AUC-PR en test          = {aucpr_te:.4f}")
print(f"  AUC-PR base (prevalencia) = {y_te.mean():.4f}")
print(f"  Umbral optimo (max F1)  = {u_opt:.4f}")
print(f"  Matriz de confusion [[TN,FP],[FN,TP]] = {cm.tolist()}")

# ==================================================================
# 6. VISUALIZACIONES
#    Cinco figuras, cada una con una funcion especifica:
#      1) convergencia        -> el BO mejora y se estabiliza
#      2) media posterior     -> el modelo sustituto aprendido
#      3) incertidumbre       -> donde el GP "no sabe" (pareja de 2)
#      4) decaimiento de EI   -> la senal interna que guia la busqueda
#      5) curva precision-recall del modelo final -> la metrica objetivo
# ==================================================================
print("\nGenerando figuras...")
hist = np.array([(h[0], h[1], h[2]) for h in historial])
fases = [h[3] for h in historial]
ei_hist = np.array([h[4] for h in historial])
n_ev, y_ev, y_best = hist[:, 0], hist[:, 1], hist[:, 2]

# --- FIG 1: curva de convergencia ---
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(n_ev, y_best, color=C_GP, lw=2, label="Mejor AUC-PR acumulado")
ax.scatter(n_ev, y_ev,
           c=[C_INIT if f == "inicial" else C_ACQ for f in fases],
           s=32, zorder=5, edgecolors='black', linewidths=0.4)
ax.axvline(N_INIT + 0.5, color='black', ls=':', alpha=0.6)
ax.scatter([], [], c=C_INIT, label="Sondeo inicial (aleatorio)")
ax.scatter([], [], c=C_ACQ, label="Evaluacion guiada por EI")
ax.set_xlabel("Numero de evaluacion")
ax.set_ylabel("AUC-PR (validacion cruzada)")
ax.set_title("Convergencia de la optimizacion bayesiana")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig("bo_1_convergencia.pdf"); plt.close()

# --- Posterior del GP sobre la rebanada learning_rate x scale_pos_weight ---
# (max_depth se fija en su valor optimo). Mismo GP que dirige el BO.
gp_final = ajustar_gp(U, Y)
RES = 70
g_lr  = torch.linspace(0, 1, RES, dtype=torch.double)   # learning_rate
g_spw = torch.linspace(0, 1, RES, dtype=torch.double)   # scale_pos_weight
GLR, GSPW = torch.meshgrid(g_lr, g_spw, indexing='xy')
malla = torch.stack([
    GLR.reshape(-1),
    torch.full((RES * RES,), u_mejor[1], dtype=torch.double),  # max_depth fijo
    GSPW.reshape(-1)], dim=-1)
with torch.no_grad():
    post = gp_final.posterior(malla)
    media = post.mean.reshape(RES, RES).numpy()
    sigma = post.variance.sqrt().reshape(RES, RES).numpy()

ext_lr  = a_natural(np.linspace(0, 1, RES), 0)   # eje x natural
ext_spw = a_natural(np.linspace(0, 1, RES), 2)   # eje y natural
px  = a_natural(U[:, 0].numpy(), 0)
py  = a_natural(U[:, 2].numpy(), 2)

# --- FIG 2: media posterior del GP (el modelo sustituto) ---
fig, ax = plt.subplots(figsize=(7, 5))
cf = ax.contourf(ext_lr, ext_spw, media, levels=30, cmap="viridis")
ax.scatter(px, py, c="white", s=34, edgecolors='black', linewidths=0.7,
           label="Puntos evaluados")
ax.scatter([px[idx_mejor]], [py[idx_mejor]], c=C_OBS, s=140, marker="*",
           edgecolors='black', linewidths=0.8, label="Optimo", zorder=6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("learning_rate"); ax.set_ylabel("scale_pos_weight")
ax.set_title("Media posterior del GP (AUC-PR estimado)")
plt.colorbar(cf, ax=ax, label="AUC-PR estimado")
ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig("bo_2_media_gp.pdf"); plt.close()

# --- FIG 3: incertidumbre posterior del GP (pareja de la FIG 2) ---
fig, ax = plt.subplots(figsize=(7, 5))
cf = ax.contourf(ext_lr, ext_spw, sigma, levels=30, cmap="magma")
ax.scatter(px, py, c="cyan", s=34, edgecolors='black', linewidths=0.7,
           label="Puntos evaluados")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("learning_rate"); ax.set_ylabel("scale_pos_weight")
ax.set_title("Desviacion tipica posterior del GP (incertidumbre)")
plt.colorbar(cf, ax=ax, label="Desviacion tipica")
ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig("bo_3_incertidumbre_gp.pdf"); plt.close()

# --- FIG 4: decaimiento del valor de Expected Improvement ---
mask_bo = np.array([f == "BO" for f in fases])
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(n_ev[mask_bo], ei_hist[mask_bo], color=C_ACQ, lw=1.8,
        marker='o', markersize=4, markeredgecolor='black',
        markeredgewidth=0.4)
ax.set_yscale("log")
ax.set_xlabel("Numero de evaluacion")
ax.set_ylabel("Valor de EI en el punto elegido (escala log)")
ax.set_title("Decaimiento de la mejora esperada (EI)")
plt.tight_layout(); plt.savefig("bo_4_decaimiento_ei.pdf"); plt.close()

# --- FIG 5: curva precision-recall del modelo final (en test) ---
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(rec, prec, color=C_GP, lw=2,
        label=f"Modelo optimizado (AUC-PR = {aucpr_te:.4f})")
ax.fill_between(rec, prec, alpha=0.18, color=C_GP)
ax.axhline(y_te.mean(), color=C_INIT, ls='--',
           label=f"Clasificador base ({y_te.mean():.4f})")
ax.set_xlabel("Recall (sensibilidad)"); ax.set_ylabel("Precision")
ax.set_title("Curva Precision-Recall en el conjunto de test")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig("bo_5_precision_recall.pdf"); plt.close()

print("Figuras generadas:")
for f in ["bo_1_convergencia", "bo_2_media_gp", "bo_3_incertidumbre_gp",
          "bo_4_decaimiento_ei", "bo_5_precision_recall"]:
    print(f"  {f}.pdf")

# ==================================================================
# 7. RESUMEN FINAL DE RESULTADOS
# ==================================================================
print("\n" + "#" * 64)
print("# RESUMEN DE RESULTADOS")
print("#" * 64)
print(f"Observaciones / fraude   : {len(y)} / {y.sum()} "
      f"({100*y.mean():.3f} %)")
print(f"Evaluaciones BO          : {N_INIT} iniciales + {N_ITER} iteraciones")
print(f"Validacion cruzada       : {CV_FOLDS}-fold estratificada")
print("Mejores hiperparametros  :")
for k, v in hp_mejor.items():
    print(f"   - {k:<18}: {v:.5f}" if isinstance(v, float)
          else f"   - {k:<18}: {v}")
print(f"AUC-PR (CV)              : {aucpr_cv:.4f}")
print(f"AUC-PR (test)            : {aucpr_te:.4f}")
print(f"AUC-PR base (test)       : {y_te.mean():.4f}")
print(f"Mejora sobre sondeo ini. : {aucpr_init:.4f} -> {aucpr_cv:.4f}")
print(f"Matriz de confusion      : {cm.tolist()}")
print("#" * 64)
