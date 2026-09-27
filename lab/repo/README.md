# RFKAN para pronóstico de caudal — Laboratorio 2

Implementación propia de **RFKAN** (Recurrent Fourier-Kolmogorov Arnold Network,
Rong, Lin & Xie, *Scientific Reports* 15:4684, 2025) adaptada al dataset
Rainfall-Runoff: 336 h de historia (12 variables) → caudal de las próximas 48 h (mm/h).

## Archivos

| Archivo | Contenido |
|---|---|
| `config.py` | Hiperparámetros. `[paper]` = Tabla 4/5, `[adapt]` = adaptación al dataset. |
| `dataset.py` | Carga de `train.h5`/`test.h5`, particiones (split 0/1), min-max [0,1] con estadísticos de train, carga en GPU. |
| `kan_layers.py` | Capas KAN desde cero: `FourierKANLayer` (Ec. 8) y `SplineKANLayer` (B-splines, KAN original), ambas con `extend_grid`. |
| `model.py` | `RFKAN` y las variantes de ablación KAN / RKAN / FKAN (Tabla 3). Nodos kernel recurrentes (Ec. 3). |
| `metrics.py` | RMSE, MAE, CORR (= R², Ec. 9), KGE, NSE por cuenca, métricas por horizonte. |
| `train.py` | Entrenamiento con refinamiento de grilla `[3,5,10,20,50,100]`, historial y checkpoints por época. |
| `baselines.py` | Persistencia, ARIMA(2,1,1), SVR RBF, ELM, LSTM, GRU, TCN (Tabla 5). |
| `informer.py` | Informer desde cero (ProbSparse attention, destilación, decoder generativo): modelo avanzado de la Tabla 7. |
| `wvs.py` | WVS = WOA-VMD-SCINet desde cero (Pearson, VMD en GPU, WOA, SCINet): Tabla 7. |
| `tacdpg.py` | TACDPG desde cero (random forest, CatBoost, DDPG con críticos gemelos y recompensa adaptativa): Tabla 7. |
| `experiments.py` | Suites `seqlen` (Fig. 5 + Tabla 3), `grid` (Fig. 4), `baselines`, `seeds`, `advanced` (Informer, Tabla 7), `seqlen_base` (LSTM/GRU/TCN × L, Fig. 5). |
| `predict.py` | CSV de test `Id,q_01..q_48` (mm/h) para todas las muestras de `test.h5`. |
| `pkl_store.py` | Vuelca todo el detalle de cada brazo a `outputs/experiments/pkl/<arm>.pkl` + `ALL.pkl`. |
| `analyze_results.py` | Tablas (Tabla 3/6, semillas, grilla), Fig. 4/5, error por horizonte, NSE por cuenca, por régimen (caudal, lluvia, temperatura) y en picos → `outputs/analysis/`. Solo CPU. |

## Detalle completo en `.pkl`

Cada brazo deja `outputs/experiments/pkl/<arm>.pkl` (y `ALL.pkl` los junta):

```python
import pickle
d = pickle.load(open("outputs/experiments/pkl/ALL.pkl", "rb"))
d["results"]                      # DataFrame: una fila por brazo
a = d["arms"]["RFKAN_L72"]
a["info"]                         # config del brazo, params, tiempos, métricas
a["history"]                      # DataFrame por época (grid, loss, métricas val, tiempos)
a["val"]["pred"], a["val"]["target"]   # [N, 48] mm/h
a["val"]["metrics"], a["val"]["per_horizon"], a["val"]["nse_per_basin"]
a["test"][...]                    # idem para test (con test_targets.csv)
a["config"], a["scaler"]          # snapshot de config.py y min/max
```

Regenerar desde disco: `../.venv/bin/python pkl_store.py`.

## Cómo ejecutar

```bash
cd lab/repo
../.venv/bin/python train.py --model RFKAN --seq_len 72
# variantes: --model KAN | RKAN | FKAN
# pipeline completo (todas las suites + predict + análisis):
../run_all.sh
```

Los datos se esperan en `lab/data/` (`train.h5`, `test.h5`, `test_targets.csv`,
`metadata.json`). Salidas en `lab/repo/outputs/<out>/`:
`<run>_history.csv`, `<run>_best.pt`, `<run>_last.pt`.

## Adaptaciones respecto al paper

- **Entrada**: 12 nodos normales por paso de tiempo = 12 canales del dataset
  (coincide con "Input nodes 12" de la Tabla 4); 12 nodos kernel recurrentes.
  El largo de ventana usado (`SEQ_LEN`) se elige por ablación, análogo a la Fig. 5.
- **Salida**: la capa 2 tiene 48 nodos (una por hora de horizonte) en lugar de
  un único nodo; se lee en el último paso de la ventana.
- **Variantes sin recurrencia** (KAN/FKAN): ventana aplanada como entrada y mismo
  ancho oculto 2·12+1 = 25 que RFKAN.
- **Grid schedule**: "50 steps por grilla" se traduce a `STAGE_EPOCHS` épocas por
  grilla (254 000 muestras de entrenamiento).
- **Salida residual** (`RESIDUAL_OUTPUT`): ŷ = último caudal + σ_Δ·núcleo(x), igual para
  RFKAN, ablaciones y baselines neuronales. Sin esto RFKAN diverge al refinar la grilla.
- **lr por etapa de grilla** = lr·(3/g) (`LR_GRID_DECAY`): los armónicos nuevos arrancan en 0
  y Adam los movería ~lr por paso → ruido de alta frecuencia.
- **Preprocesamiento**: el dataset no tiene faltantes (no aplica interpolación);
  min-max [0,1] por canal con estadísticos solo de train; métricas en mm/h.
- **Pérdida** = MSE / σ_Δ² (mismo óptimo que la MSE). Sin escalar, la MSE normalizada
  ronda 2.5e-5 y los gradientes de LSTM/GRU/TCN quedan bajo el eps de Adam (1e-8).
  La primera corrida (con ese problema) quedó archivada en `outputs/experiments_v1/`.
- **Baselines neuronales** (LSTM/GRU/TCN) con la misma ventana `SEQ_LEN` = 72 que RFKAN:
  con 336 pasos el LSTM no sale de persistencia en 18 épocas.
- **ELM**: ridge con λ = 0.001 sin escalar. **SVR**: 10 000 ventanas de train, 24 h de
  historia. **ARIMA**: por ventana, sobre un submuestreo de 2000 (NaN/∞ → persistencia).
- **RKAN con L=336**: grilla 100 no entra en 16 GB; se reporta su mejor checkpoint (época 12).
- **Modelos avanzados (Tabla 7)**: Informer, WVS y TACDPG, implementaciones propias. WVS y
  TACDPG se implementaron desde sus resúmenes (texto completo no accesible): supuestos en
  `docs/GAP_ANALISIS.md`. El análisis por estación se aproxima con cuartiles de temperatura,
  porque el dataset no trae fechas.

## Resultados (test, 27 983 ventanas × 48 h, mm/h)

Detalle completo en `../../docs/RESULTADOS.md`; figuras y tablas en `outputs/analysis/`.

| Modelo | test RMSE | test CORR (R²) |
|---|---|---|
| Persistencia | 0.1188 | 0.496 |
| KAN L=72 | 0.1100 | 0.567 |
| FKAN L=72 | 0.1172 | 0.510 |
| RKAN L=72 | 0.1033 | 0.619 |
| **RFKAN L=72** (propuesta) | **0.1052** | **0.604** |
| LSTM / GRU / TCN (L=72) | 0.1012 / 0.1015 / 0.1032 | 0.634 / 0.632 / 0.620 |

- La recurrencia mejora a KAN; Fourier no mejora a splines (sí es más rápido).
- Refinar la grilla hasta 100 no aporta precisión: el mejor checkpoint queda en grilla 10–20.
- LSTM y GRU superan a RFKAN, pero el LSTM queda en persistencia en 2 de 4 semillas;
  RFKAN es estable (0.1058 ± 0.0006).
- Todos subestiman los picos de crecida (p99) en 32–39 %.

Predicciones de test: `outputs/experiments/predictions_test_RFKAN_L72.csv`.
