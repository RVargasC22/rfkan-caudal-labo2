# RFKAN para pronóstico de caudal — Laboratorio 2

Implementación propia de **RFKAN** (Recurrent Fourier-Kolmogorov Arnold Network,
Rong, Lin & Xie, *Scientific Reports* 15:4684, 2025, DOI
[10.1038/s41598-025-88959-5](https://doi.org/10.1038/s41598-025-88959-5))
adaptada al dataset Rainfall-Runoff: 336 h de historia (12 variables) → caudal
de las próximas 48 h (mm/h) en 508 cuencas.

El paper pronostica potencia fotovoltaica day-ahead. Propone una KAN de dos capas
(n → 2n+1 → salida) con dos cambios: **nodos kernel recurrentes**
(`h(t) = W_hh h(t−1) + W_hz x(t)`, Ec. 3) y **series de Fourier** como función
aprendible de cada arista en lugar de B-splines (Ec. 8), entrenada con
**refinamiento de grilla** 3 → 5 → 10 → 20 → 50 → 100. Todo el código de las capas
KAN (Fourier y B-spline) está escrito desde cero, sin librerías de KAN.

## Archivos

| Archivo | Contenido |
|---|---|
| `config.py` | Hiperparámetros. `[paper]` = Tabla 4/5, `[adapt]` = adaptación al dataset. |
| `dataset.py` | Carga de `train.h5`/`test.h5`, particiones (split 0/1), min-max [0,1] con estadísticos de train, carga en GPU. |
| `kan_layers.py` | Capas KAN desde cero: `FourierKANLayer` (Ec. 8) y `SplineKANLayer` (B-splines con Cox–de Boor), ambas con `extend_grid`. |
| `model.py` | `RFKAN` y las variantes de ablación KAN / RKAN / FKAN (Tabla 3). Nodos kernel recurrentes (Ec. 3). |
| `metrics.py` | RMSE, MAE, CORR (= R², Ec. 9), KGE, NSE por cuenca, métricas por horizonte. |
| `train.py` | Entrenamiento con refinamiento de grilla `[3,5,10,20,50,100]`, historial y checkpoints por época. |
| `baselines.py` | Persistencia, ARIMA(2,1,1), SVR RBF, ELM, LSTM, GRU, TCN (Tabla 5). |
| `informer.py` | Informer desde cero (atención ProbSparse, destilación, decoder generativo): Tabla 7. |
| `wvs.py` | WVS = WOA-VMD-SCINet desde cero (Pearson, VMD en GPU, WOA, SCINet): Tabla 7. |
| `tacdpg.py` | TACDPG desde cero (random forest, CatBoost, DDPG con críticos gemelos y recompensa adaptativa): Tabla 7. |
| `experiments.py` | Suites `seqlen` (Fig. 5 + Tabla 3), `grid` (Fig. 4), `baselines`, `seeds`, `advanced` (Tabla 7), `seqlen_base` (LSTM/GRU/TCN × L). |
| `predict.py` | CSV de test `Id,q_01..q_48` (mm/h) para todas las muestras de `test.h5`. |
| `pkl_store.py` | Vuelca el detalle de cada brazo a `outputs/experiments/pkl/<arm>.pkl` + `ALL.pkl`. |
| `analyze_results.py` | Tablas, Fig. 4/5, error por horizonte, NSE por cuenca, error por régimen y en picos → `outputs/analysis/`. Solo CPU. |
| `run_all.sh` | Corre todo el pipeline: las seis suites, `predict.py` y `analyze_results.py`. |

## Instalación

Python 3.12 y una GPU con CUDA (todo el dataset se carga en GPU, ~4.5 GB; los
experimentos se corrieron en una RTX PRO 4000 de 16 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## Datos

Dataset Rainfall-Runoff del curso (carpeta de Google Drive
`1crKbJBhLHQEVJGG-agOxMnKLDnBP2x4w`): `train.h5` (5.1 GB), `test.h5`,
`test_targets.csv`, `metadata.json`.

- `train.h5`: `X [272142, 336, 12]`, `y [N, 48]`, `basin_id`, `split`
  (254 000 train / 18 142 val). `test.h5`: `X [27983, 336, 12]`.
- Las mismas 508 cuencas en train, val y test. Canal 11 = caudal específico (mm/h).
- Sin faltantes; sin normalizar.

Por defecto se buscan en `../data/` (carpeta hermana del repo). Para otra ruta:
`export RFKAN_DATA_DIR=/ruta/a/data`.

## Cómo ejecutar

```bash
# un modelo
python train.py --model RFKAN --seq_len 72        # variantes: --model KAN | RKAN | FKAN

# pipeline completo (todas las suites + predicciones de test + análisis)
./run_all.sh                                       # usa `python`; otro intérprete: PY=../.venv/bin/python ./run_all.sh

# solo el análisis, a partir de resultados ya calculados
python analyze_results.py
```

Salidas en `outputs/`:

- `outputs/experiments/experiments_results.csv`: una fila por brazo (56 brazos), con métricas de val y test.
- `outputs/experiments/<arm>_history.csv`: historial por época.
- `outputs/experiments/predictions_test_RFKAN_L72.csv`: predicciones de test (formato `Id,q_01..q_48`).
- `outputs/analysis/`: figuras y tablas en Markdown.
- Checkpoints (`*.pt`), predicciones crudas (`*.npz`) y `pkl/` se regeneran y no se versionan.

### Detalle completo en `.pkl`

```python
import pickle
d = pickle.load(open("outputs/experiments/pkl/ALL.pkl", "rb"))
d["results"]                           # DataFrame: una fila por brazo
a = d["arms"]["RFKAN_L72"]
a["info"]                              # config del brazo, params, tiempos, métricas
a["history"]                           # DataFrame por época (grid, loss, métricas val, tiempos)
a["val"]["pred"], a["val"]["target"]   # [N, 48] mm/h
a["val"]["metrics"], a["val"]["per_horizon"], a["val"]["nse_per_basin"]
a["test"][...]                         # idem para test (con test_targets.csv)
a["config"], a["scaler"]               # snapshot de config.py y min/max
```

Regenerar desde los archivos de `outputs/`: `python pkl_store.py`.

## Diferencias con el paper

### Adaptación del problema

- **Entrada**: 12 nodos normales por paso de tiempo = 12 canales del dataset
  ("Input nodes 12" de la Tabla 4) y 12 nodos kernel recurrentes. El largo de
  ventana (`SEQ_LEN`) se elige por ablación, análogo a la Fig. 5; el modelo principal usa L = 72.
- **Salida**: la capa 2 tiene 48 nodos (una por hora de horizonte) en lugar de un
  único nodo; se lee en el último paso de la ventana.
- **Sin meteorología futura**: el paper usa la meteorología del día pronosticado.
  Aquí solo se usa la historia `X` (L ≤ 336 h) y los nodos recurrentes modelan los
  desfases dentro de ella, como indicó el profesor. `y_aux` no se usa en ningún
  brazo (en test no existe), así que no hay fuga de futuro. La alternativa,
  pronosticar primero la meteorología de las 48 h con `y_aux` como objetivo y
  alimentar con ella el modelo de caudal, queda como trabajo futuro.
- **Variantes sin recurrencia** (KAN/FKAN): ventana aplanada como entrada y mismo
  ancho oculto 2·12+1 = 25 que RFKAN.
- **Preprocesamiento**: el dataset no tiene faltantes (no aplica la interpolación
  del paper); min-max [0, 1] por canal con estadísticos solo de train; métricas en mm/h.
- **Evaluación**: el paper evalúa un día por estación (288 puntos); aquí, 27 983
  ventanas × 48 h. El análisis por estación se aproxima con cuartiles de temperatura,
  porque el dataset no trae fechas.

### Cambios necesarios para entrenar

- **Salida residual** (`RESIDUAL_OUTPUT`): ŷ = último caudal + σ_Δ·núcleo(x), con
  σ_Δ = 0.0046 (cambio típico del caudal en 48 h, escala [0, 1]). La versión literal
  diverge al refinar la grilla (val RMSE 0.27 → 3.6 mm/h). Se aplica igual a RFKAN,
  ablaciones y baselines neuronales.
- **lr por etapa de grilla** = lr·(3/g) (`LR_GRID_DECAY`): los armónicos nuevos
  arrancan en 0 y Adam los movería ~lr por paso, ruido frente a σ_Δ.
- **Pérdida** = MSE / σ_Δ² (mismo óptimo que la MSE). Sin escalar, la MSE normalizada
  ronda 2.5·10⁻⁵ y el 99 % de los gradientes de LSTM/GRU/TCN queda bajo el ε de Adam
  (10⁻⁸): el LSTM terminaba exactamente en persistencia.
- **Grid schedule**: "50 steps por grilla" se traduce a 3 épocas por grilla, 18 en
  total, el mismo presupuesto para todos los modelos.

### Modelos base y avanzados

- **LSTM/GRU/TCN** con la misma ventana L = 72 que RFKAN: con 336 pasos el LSTM no sale
  de persistencia en 18 épocas. La Tabla 5 del paper describe un LSTM y la Tabla 6
  reporta un GRU: se corrieron ambos.
- **ELM**: ridge con λ = 0.001. **SVR** (C = 0.0043 del paper): 10 000 ventanas de
  train, 24 h de historia. **ARIMA(2,1,1)**: por ventana, sobre un submuestreo de 2000
  ventanas de test (NaN/∞ → persistencia).
- **Informer, WVS y TACDPG** (Tabla 7): implementaciones propias. WVS y TACDPG se
  implementaron desde sus resúmenes; los valores no publicados son supuestos:
  WOA con K ∈ [2, 8], α ∈ [100, 4000], 10 ballenas × 15 iteraciones; VMD por ventana
  (sin fuga de futuro); recompensa de TACDPG con z0 = 1, z1 = 0.1, λ = 0.01; episodios
  de 64 ventanas, γ = 0.9, τ = 0.005; la acción de TACDPG es la corrección residual de las 48 h.
- **RKAN con L = 336**: la grilla 100 no entra en 16 GB; se reporta su mejor checkpoint (época 12).

### Inconsistencias del paper

1. La Tabla 4 dice `Input seq_len 12`, pero el texto concluye que el mejor largo es 2.
2. La Tabla 5 da hiperparámetros de LSTM y la Tabla 6 reporta GRU.
3. El texto de la Tabla 6 dice "MSE reducido 70.74 %", pero la tabla reporta MAE.
4. "CORR" se define como R², no como correlación.

## Resultados (test: 27 983 ventanas × 48 h, mm/h)

### Ablación (Tabla 3), L = 72

| Modelo | RMSE | MAE | CORR (R²) | NSE mediano | Parámetros | Tiempo (s) |
|---|---|---|---|---|---|---|
| Persistencia | 0.1188 | 0.0246 | 0.496 | 0.492 | — | 0 |
| KAN | 0.1100 | 0.0350 | 0.567 | 0.318 | 319 k | 88 |
| FKAN | 0.1172 | 0.0288 | 0.510 | 0.415 | 4 560 k | 49 |
| RKAN | **0.1033** | 0.0270 | 0.619 | 0.520 | 44 k | 663 |
| **RFKAN** (propuesta) | 0.1052 | 0.0285 | 0.604 | 0.478 | 122 k | 392 |

### Comparación con modelos base y avanzados (Tablas 6 y 7)

| Modelo | RMSE | MAE | CORR (R²) | NSE mediano | Tiempo (min) |
|---|---|---|---|---|---|
| ARIMA(2,1,1) * | 6.7283 | 0.2037 | −1377.8 | 0.733 | 1 |
| SVR | 0.1187 | 0.0256 | 0.497 | 0.432 | 1 |
| ELM | 0.1423 | 0.0717 | 0.277 | −0.409 | 0 |
| LSTM | **0.1012** | **0.0241** | 0.634 | 0.569 | 5 |
| GRU | 0.1015 | 0.0251 | 0.632 | 0.535 | 5 |
| TCN | 0.1032 | 0.0285 | 0.620 | 0.518 | 5 |
| Informer | 0.1032 | 0.0284 | 0.620 | 0.533 | 36 |
| WVS | 0.1061 | 0.0236 | 0.598 | 0.566 | 70 |
| TACDPG | 0.1205 | 0.0349 | 0.481 | 0.301 | 1 |
| RFKAN | 0.1052 | 0.0285 | 0.604 | 0.478 | 7 |

\* ARIMA: 2000 ventanas; en la cuenca típica es el mejor, pero explota en algunas ventanas.

### Refinamiento de grilla (Fig. 4), RFKAN L = 72

| Variante | RMSE val | RMSE test | Mejor grilla |
|---|---|---|---|
| 3 → 100 (estándar) | 0.1076 | 0.1052 | 20 |
| 3 → 200 | 0.1076 | 0.1052 | 20 |
| grilla 3 fija | 0.1096 | 0.1070 | 3 |
| grilla 100 fija | 0.1405 | 0.1220 | 100 |
| sin reducir el lr | 0.1136 | 0.1071 | 5 |

### Qué se reproduce y qué no

| Afirmación del paper | Resultado | ¿Se reproduce? |
|---|---|---|
| La recurrencia mejora a KAN | FKAN → RFKAN −10.2 %, KAN → RKAN −6.1 % de RMSE | sí |
| Fourier mejora la precisión | RKAN → RFKAN +1.9 % (peor) | no |
| Fourier reduce el tiempo | RFKAN 0.59× el tiempo de RKAN | sí |
| Grilla óptima = 100 | mejor época en grilla 10–20 | no |
| El refinamiento de grilla ayuda | no mejora la precisión, pero sin él (grilla 100 fija) o sin bajar el lr el modelo empeora | parcial |
| Largo de entrada óptimo = 2 | las variantes recurrentes son insensibles a L (2–336 h); solo las aplanadas prefieren L corto | parcial |
| RFKAN supera a GRU/TCN (−58 % RMSE) | LSTM, GRU y TCN < RFKAN | no |
| RFKAN supera a ARIMA/SVM/ELM | sí | sí |
| RFKAN supera a los modelos avanzados | gana en validación; en test Informer lo supera y WVS empata | parcial |

Otros resultados:

- **Semillas**: RFKAN es estable (test 0.1058 ± 0.0006 en 4 semillas); el LSTM queda en
  persistencia en 2 de 4 (0.1100 ± 0.0102).
- **Picos**: todos los modelos entrenados subestiman el 1 % de caudales más altos en un 32–39 %.
- **Por cuenca**: RFKAN mejora a persistencia en el 53 % de las cuencas (LSTM: 68 %).
- **Horizonte**: el error crece con la hora pronosticada; a las 48 h RFKAN reduce el RMSE
  de persistencia en un 12 %.

Figuras y tablas en `outputs/analysis/`; predicciones de test en
`outputs/experiments/predictions_test_RFKAN_L72.csv` (RMSE 0.1051 contra `test_targets.csv`).
