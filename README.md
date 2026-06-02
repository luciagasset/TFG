# Optimización Bayesiana

> Trabajo de Fin de Grado — Grado en Matemáticas

Implementación y estudio de la **optimización bayesiana**, una técnica para
optimizar funciones costosas de evaluar y sin forma analítica, basada en
**Procesos Gaussianos** como modelo sustituto. El proyecto recorre el método
de principio a fin: desde sus fundamentos matemáticos hasta dos aplicaciones
prácticas, una controlada y otra sobre un problema real.

---

## De qué trata

La idea central de la optimización bayesiana es sustituir una función
desconocida y cara de evaluar por una **distribución de probabilidad sobre
funciones**. Un Proceso Gaussiano modela esa distribución y, en cada punto,
no solo predice un valor sino también su **incertidumbre**. Una *función de
adquisición* explota esa incertidumbre para decidir, de forma inteligente,
dónde evaluar a continuación, equilibrando la exploración de zonas
desconocidas con la explotación de las prometedoras.

Este repositorio contiene el código que acompaña a la memoria del trabajo,
organizado en tres bloques: las figuras que ilustran la teoría y dos casos
de estudio.

## Casos de estudio

**Función de Forrester (validación controlada).**
Al conocerse su forma exacta, permite visualizar el comportamiento del
modelo, analizar la influencia de los hiperparámetros del Proceso Gaussiano
y comparar funciones de adquisición sobre un problema donde se conoce la
respuesta.

**Detección de fraude con tarjetas de crédito (problema real).**
Ajuste de hiperparámetros de un modelo **XGBoost** sobre un dataset real,
multidimensional y con un desbalanceo extremo entre clases (~0,2 % de
fraude). La función objetivo es el **AUC-PR** medio en validación cruzada
estratificada, y el conjunto de test se aparta antes de todo y solo se
evalúa una vez.

Resultado: el algoritmo localizó de forma autónoma una configuración con
**AUC-PR ≈ 0,86** tanto en validación cruzada como en test, dos valores casi
idénticos que confirman que el modelo no sobreajusta. Más allá de la cifra,
el trabajo discute *cuándo* la optimización bayesiana aporta valor real: su
ventaja crece cuanto más costosa o más sensible a su configuración es la
función objetivo.

## Stack técnico

`Python` · `PyTorch` · `GPyTorch` · `BoTorch` · `XGBoost` · `scikit-learn` ·
`NumPy` · `Matplotlib`

El bucle de optimización (Proceso Gaussiano → Expected Improvement →
optimización de la adquisición) está implementado de forma explícita, en
correspondencia directa con el desarrollo teórico de la memoria.

---

## Estructura del repositorio

```
optimizacion-bayesiana-tfg/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # el dataset NO se incluye (ver más abajo)
├── teoria/                # figuras de los capítulos teóricos
│   ├── fig_gp_prior.py                     -> Proceso Gaussiano a priori (RBF)
│   ├── fig_gp_prior_matern.py              -> Proceso Gaussiano a priori (Matérn 5/2)
│   ├── fig_gp_posterior.py                 -> Proceso Gaussiano a posteriori
│   ├── fig_kernel_2x2_hiperparametros.py   -> efecto de los hiperparámetros del kernel
│   └── fig_evolucion_optimizacion_bayesiana.py  -> evolución del algoritmo
├── forrester/             # caso de estudio sintético
│   └── forrester_bo.py
└── fraude/                # caso de estudio real (XGBoost)
    └── fraude_bo.py
```

Cada script genera sus figuras en formato PDF en la carpeta desde la que se
ejecuta.

## El dataset

El experimento de fraude utiliza el dataset **Credit Card Fraud Detection**
de Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

No se incluye en el repositorio por su tamaño y sus condiciones de uso. Para
reproducir el experimento:

1. Descarga `creditcard.csv` desde el enlace anterior.
2. Colócalo en la carpeta `data/` (o junto a `fraude_bo.py`).
3. Si lo dejas en `data/`, ajusta en `fraude_bo.py` la variable
   `RUTA_CSV = "../data/creditcard.csv"`.

Si el archivo no se encuentra, el script se detiene con un mensaje
indicándolo (no genera datos artificiales).

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt

# Figuras teóricas (por ejemplo fig_gp_prior.py)
python teoria/fig_gp_prior.py

# Caso sintético (función de Forrester)
python forrester/forrester_bo.py

# Caso real (detección de fraude); requiere creditcard.csv
python fraude/fraude_bo.py
```

Todos los experimentos fijan una semilla aleatoria al inicio de cada script,
de modo que los resultados son reproducibles.

---

*El desarrollo matemático completo y la discusión de los
resultados, se encuentran en la memoria del Trabajo de Fin de Grado.*
