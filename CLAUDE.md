# CLAUDE.md — Laboratorio 2 (Maestría, Deep Learning)

Contexto del proyecto para sesiones futuras. Mismo esquema de trabajo que
`../Labo1/` (paper → código propio → ablaciones → docs → informe → slides).

## Qué hay que entregar

- **Código fuente** (notebook o scripts .py) con: carga de datos y particiones,
  preprocesamiento, implementación del método del paper, entrenamiento y
  validación, **modelo base**, métricas, **ablación**, predicciones de test.
  Si son scripts: markdown con resultados + instrucciones de ejecución +
  diferencias importantes respecto al paper.
- **Video explicativo, máx. 18 min**: Metodología (6) / Implementación (6) /
  Resultados (6). Evaluación **grupal**.
- **Entrega:** jueves **1-oct-2026 23:59** (formulario: https://forms.gle/GDUh9wn96ny91gbN7).
- **Sesión de preguntas individual:** sábado 3-oct-2026 (horario de clase).
- Consigna completa: `Laboratorio 2.md`.
- La implementación de la propuesta central debe ser **propia** (no se permite
  usar una implementación existente de RFKAN/KAN/FourierKAN). Por eso
  `kan_layers.py` implementa Fourier-KAN y B-spline-KAN desde cero.

## Estructura del directorio

- `Laboratorio 2.md` — consigna.
- `s41598-025-88959-5 (1).pdf` — paper original (copia en `docs/`).
- `Rainfall-Runoff/` — descarga parcial original (solo test + metadata).
- `docs/` — `Rong2025_RFKAN_PV.pdf` + `.txt` (texto extraído con pdftotext).
  Pendiente: `NOTAS_PAPER.md`, `RESULTADOS.md`, `GAP_ANALISIS.md`,
  `ENTREGABLES.md`, informe `.docx`.
- `presentacion/` — pendiente (html + pptx, como Labo 1).
- `lab/` — entorno de trabajo:
  - `.venv/` Python 3.12 (uv), torch 2.14+cu130, h5py, pandas, sklearn,
    statsmodels, python-pptx, python-docx.
  - `data/` — dataset completo bajado con gdown desde el Drive del curso
    (`train.h5` 5.1 GB, `test.h5`, `test_targets.csv`, `metadata.json`,
    `leer_datos.py`). Drive: `https://drive.google.com/drive/folders/1crKbJBhLHQEVJGG-agOxMnKLDnBP2x4w`
  - `repo/` — código (ver `lab/repo/README.md`).

## El paper

Rong, Lin & Xie — *Recurrent Fourier-Kolmogorov Arnold Networks for
photovoltaic power forecasting*, **Scientific Reports 15:4684 (2025)**,
DOI `10.1038/s41598-025-88959-5`. Liaoning Technical University.

- Problema original: pronóstico day-ahead de potencia PV (planta de 45 kW en
  Tieling, 2023, muestreo 5 min, 105 120 puntos).
- **RFKAN** = KAN de 2 capas (n → 2n+1 → salida) con dos cambios:
  1. **Nodos kernel recurrentes** (Ec. 3): `h(t) = W_hh h(t-1) + W_hz x(t)`,
     que entran a cada capa en paralelo con los nodos normales (Ec. 4).
  2. **Serie de Fourier** en lugar de B-splines como función de activación
     aprendible (Ec. 8): `φ(x) = Σ_k a_k cos(kx) + b_k sin(kx)`.
  - **Refinamiento de grilla**: grid `[3,5,10,20,50,100]`, 50 steps por grilla
    (Tabla 4); el modelo hereda la curva aprendida al pasar a grilla más fina.
- Ablación (Tabla 3): KAN / RKAN / FKAN / RFKAN. Otras: tamaño de grilla
  (Fig. 4), largo de secuencia de entrada (Fig. 5).
- Baselines (Tabla 5): ARIMA(2,1,1), SVM RBF C=0.0043, ELM (3 capas, ReLU,
  reg 0.001), LSTM (3 capas, lr 1e-3, bs 32, Adam), TCN (3 conv, stride 2).
  Inconsistencia del paper: Tabla 5 dice LSTM, Tabla 6 reporta GRU.
- Métricas: RMSE, MAE, "CORR" = R² (1 − SSE/SST).

## El dataset del laboratorio

Rainfall-Runoff horario. `train.h5`: `X [272142, 336, 12]`, `y [N, 48]`,
`y_aux [N, 48, 11]`, `basin_id`, `split` (254 000 train / 18 142 val).
`test.h5`: `X [27983, 336, 12]`, `basin_id`. **508 cuencas, las mismas en
train/val/test.** Canal 11 = caudal específico (mm/h), objetivo. Sin NaN,
sin normalizar. `test_targets.csv` trae los caudales reales de test (formato
`Id,q_01..q_48`) → se pueden calcular métricas de test.

Referencia: **persistencia** (repetir último caudal) RMSE ≈ 0.128 mm/h.

## Estado

- ✅ Datos descargados, venv creado.
- ✅ `kan_layers.py`, `model.py`, `dataset.py`, `metrics.py`, `train.py`
  escritos; extend_grid verificado (Fourier exacto, spline por lstsq ~1e-5).
- ✅ Divergencia resuelta: salida residual sobre persistencia (σ_Δ = 0.00463 en
  escala normalizada) + lr por etapa ∝ 3/g. Barrido corto: RFKAN 0.112 vs
  persistencia 0.138 val RMSE; recurrencia ayuda (RFKAN/RKAN < FKAN/KAN).
- ✅ `baselines.py`, `experiments.py`, `predict.py`, `pkl_store.py`, `lab/run_all.sh`.
- ✅ `lab/run_all.sh` v2 (25-sep), log `lab/run_all.log` + `lab/logs_run_all_v2_oom_L336.log`.
  v1 archivada en `outputs/experiments_v1/`: la MSE normalizada (~2.5e-5) dejaba
  los gradientes bajo el eps de Adam (LSTM = persistencia). v2: pérdida / σ_Δ²,
  baselines neuronales con L = SEQ_LEN = 72, ELM con λ sin escalar por N.
  Persistencia/SVR/ARIMA copiados de v1 (no dependen de la pérdida).
  Resultados en `lab/repo/outputs/experiments/experiments_results.csv`.
- **Todo el detalle en `.pkl`**: `outputs/experiments/pkl/<arm>.pkl` + `ALL.pkl`
  (historial por época, predicciones y targets val/test en mm/h, métricas,
  por horizonte, NSE por cuenca, config, scaler). La suite 1 (seqlen) arrancó
  antes de agregar el pkl → sus pkl se generan con el `dump_all()` al final
  de la suite 2 (o a mano: `python pkl_store.py`).
- ✅ analyze_results.py (probado con los brazos de seqlen; necesita `tabulate`).
- Incidentes: 24-sep 20:00 OOM en KAN_L336 (extend_grid spline con 4032 entradas) → lstsq por bloques;
  25-sep grid 500 no entra en 16 GB (desborda a RAM en WSL, ~40 W) → grid_ext = [..100, 200].
  L=336 movido al final de run_all.sh (RKAN_L168 tardó 86 min).
- ✅ docs (NOTAS, ECUACIONES, REFERENCIAS, IMPLEMENTACION, GAP, RESULTADOS, ENTREGABLES).
- ✅ Informe: `docs/Informe_Laboratorio2_RFKAN.docx` generado por `docs/build_informe.py` (lee el CSV).
- ✅ Presentación: fuente única `presentacion/content.py` → `build_pptx.py` (pptx + SPEECH.md) y
  `build_deck.py` (deck web, artifact https://claude.ai/artifact/7Eh6oV3GD1UnMwT5ofg6Ss; figuras como
  assets, URLs en `deck_assets.json`).
- Resultado clave v2: recurrencia ayuda; Fourier no (RKAN > RFKAN); LSTM/GRU > familia KAN en test,
  pero LSTM queda en persistencia en 2/4 semillas; picos subestimados 32–39 %.
- ✅ Pipeline v2 COMPLETO (26-sep 06:52), 56 brazos: incluye Tabla 7 completa (Informer, WVS, TACDPG,
  implementaciones propias; PDFs de WVS/TACDPG en docs/) y LSTM/GRU/TCN × L (Fig. 5). RKAN_L336: OOM en grilla 100 (época 16) →
  se registró su mejor checkpoint (época 12). Predicciones de test:
  `outputs/experiments/predictions_test_RFKAN_L72.csv` (test RMSE 0.1051).
- ⚠️ El deck web (artifact 7Eh6oV3GD1UnMwT5ofg6Ss) es de la cuenta anterior; desde la cuenta actual
  no se puede editar. Le faltan L=336, Informer y los baselines por L. El pptx local (20 slides) está al día.

## Notas operativas

- Ejecutar desde `lab/repo/` con `../.venv/bin/python`.
- gdown/uv necesitan `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`
  (proxy con certificado propio) y correr fuera del sandbox.
- Todo el dataset se carga en GPU (~4.5 GB). GPU: RTX PRO 4000 Blackwell 16 GB, RAM 31 GB.
- Tiempos medidos (bs 256, grid 100): RFKAN L=72 ≈ 20 ms/step (~16 s/época),
  L=336 ≈ 100 ms/step; RKAN L=336 ≈ 240 ms/step.
