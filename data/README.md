# Datos

Esta carpeta está reservada para el dataset utilizado en el experimento de
detección de fraude. El archivo **no se incluye** en el repositorio por su
tamaño y por las condiciones de uso de Kaggle.

## Cómo obtenerlo

1. Descarga el dataset **Credit Card Fraud Detection** desde Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Coloca el archivo `creditcard.csv` en esta carpeta (`data/`).
3. En `fraude/fraude_bo.py`, asegúrate de que la ruta apunta aquí:
   `RUTA_CSV = "../data/creditcard.csv"`

El dataset contiene transacciones con tarjeta de crédito descritas por las
variables V1..V28 (componentes principales anonimizadas), el importe
(Amount) y la etiqueta de fraude (Class), con un fuerte desbalanceo entre
clases (~0,2 % de fraude).
