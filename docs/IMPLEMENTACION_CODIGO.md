# Implementación en código — RFKAN aplicado al pronóstico de caudal

Este documento explica **cómo se llevó el paper a código** y qué decisiones se
tomaron que **no están en el paper**, porque el dataset del laboratorio es
distinto. Complementa a `ECUACIONES_PAPER.md` (teoría) y a `NOTAS_PAPER.md`.
Las diferencias una por una están en `GAP_ANALISIS.md`; aquí se cuenta la misma
historia de forma narrativa, pensada para el bloque "Implementación" del video.

---

## El cambio de dominio: de paneles solares a ríos

| | Paper | Laboratorio |
|---|---|---|
| Qué se predice | potencia de **una** planta solar de 45 kW, un año, cada 5 min | caudal específico (mm/h) de **508 cuencas**, cada 1 h |
| Entrada | clima + potencia pasada | 336 h × 12 variables (11 meteorológicas + caudal) |
| Salida | 288 pasos (2 días) | 48 pasos (2 días) |
| Tamaño | 105 120 puntos, test = 1 día por estación | 254 000 ventanas de train, 18 142 de validación y 27 983 de test |

Estructuralmente es el mismo problema: regresión multivariada, multi-paso y
day-ahead. Pero hay dos diferencias que pesan mucho:

1. **La potencia PV es muy periódica** (ciclo día/noche). El caudal no lo es: es
   una señal suave con **picos raros y abruptos** después de la lluvia. El
   argumento central del paper a favor de Fourier ("captura periodicidad") no
   aplica directamente.
2. **El caudal a 48 h está muy determinado por el caudal actual.** Repetir el
   último valor (persistencia) ya da RMSE ≈ 0.138 mm/h en validación. Cualquier
   modelo tiene que aprender la **corrección** sobre eso.

---

## Estructura del repositorio (`lab/repo/`)

| Archivo | Qué hace | Paper |
|---|---|---|
| `config.py` | Hiperparámetros: `[paper]` = Tablas 4/5, `[adapt]` = decisión nuestra | Tablas 4, 5 |
| `dataset.py` | Lee `train.h5`/`test.h5`, separa train/val por `split`, agrega `test_targets.csv`, min-max [0, 1] con estadísticos de train, carga todo en GPU | Preprocesamiento (pág. 4) |
| `kan_layers.py` | `FourierKANLayer` y `SplineKANLayer` desde cero, con `extend_grid` | Ec. 2, 8; KAN (ref. 39) |
| `model.py` | `RecurrentKernelNodes`, `RFKAN` (4 variantes), `Forecaster` (salida residual) | Ec. 1, 3–7; Fig. 1; Tabla 3 |
| `train.py` | Bucle Adam + MSE, refinamiento de grilla por etapas, historial y checkpoints | Fig. 4; Tabla 4 |
| `metrics.py` | RMSE, MAE, CORR (= R²), NSE por cuenca, KGE, por horizonte | Ec. 9 |
| `baselines.py` | Persistencia, ARIMA(2,1,1), SVR RBF, ELM, LSTM, GRU, TCN | Tabla 5 |
| `informer.py` | Informer desde cero: ProbSparse attention, destilación, decoder generativo | Tabla 7 |
| `experiments.py` | Suites `seqlen`, `grid`, `baselines`, `seeds`, `advanced`, `seqlen_base` (una fila por "brazo" en `experiments_results.csv`) | Tabla 3, 6, 7; Fig. 4, 5 |
| `pkl_store.py` | Todo el detalle de cada brazo en `outputs/experiments/pkl/<arm>.pkl` | — |
| `predict.py` | CSV de test `Id, q_01..q_48` (mm/h) | — |
| `analyze_results.py` | Tablas, figuras y análisis de errores (horizonte, cuenca, régimen, picos) | Fig. 4, 5; Tablas 3, 6 |
| `../run_all.sh` | Corre todo en orden | — |

---

## Datos y preprocesamiento (`dataset.py`)

- **Particiones:** las del curso. `split == 0` es train (254 000) y
  `split == 1` es validación (18 142). Test es `test.h5` (27 983) y sus
  objetivos vienen de `test_targets.csv`, ordenados por `Id`. Las 508 cuencas
  aparecen en los tres conjuntos.
- **Faltantes y outliers:** `metadata.json` declara `imputation_applied: false`
  y verificamos que no hay NaN. Los pasos 1 y 2 del preprocesamiento del paper
  (interpolación lineal) no aplican.
- **Normalización:** min-max a [0, 1] por canal (paso 3 del paper), con
  mínimos y máximos **solo de train** para no filtrar información de
  validación/test. Las métricas se calculan siempre en mm/h tras invertir la
  escala.
- **Rendimiento:** todo el dataset (~4.5 GB en float32) se carga en la GPU una
  vez. Los batches se arman por indexación, sin `DataLoader`.

## El modelo (`model.py`, `kan_layers.py`)

**RFKAN** con 12 nodos de entrada (los 12 canales, igual que "Input nodes 12"
de la Tabla 4) y 12 nodos kernel recurrentes (igual que la Tabla 4):

```
x (B, L, 12) ─┬──────────────────────────┐
              └─ rec0: h(t)=W_hh h(t−1)+W_hz x(t) ─┤ concat (B, L, 24)
                                                   ▼
                        layer1: FourierKAN 24 → 25 (en cada t)
                                    x1 (B, L, 25)
              ┌─────────────────────┴───────────────┐
              │           rec1 (misma recurrencia) ─┤
              ▼ último paso t = L                   ▼
                    concat [x1(L), h1(L)]  (B, 50)
                                    ▼
                        layer2: FourierKAN 50 → 48
                                    ▼
                  ŷ = q(t) + σ_Δ · salida  (Forecaster)
```

Decisiones:

- **Las variantes de ablación** se construyen todas con la misma clase `RFKAN`
  (`build_model`):
  - `basis="spline"` elige splines (KAN/RKAN); `"fourier"` elige Fourier
    (FKAN/RFKAN).
  - `recurrent=False` quita los nodos recurrentes (KAN/FKAN). En ese caso, como
    dice el paper ("n = a × b"), la ventana se **aplana** a `12·L` entradas y se
    mantiene el ancho oculto de 25.
- **Salida de 48 nodos:** el paper tiene un único nodo de salida. Nosotros
  leemos las 48 horas desde la capa 2 en el último paso.
- **φ(x, h):** se interpreta como nodos en paralelo, es decir, entradas
  concatenadas (ver Ec. 4 en `ECUACIONES_PAPER.md`).

## El problema que apareció: divergencia al refinar la grilla

La primera versión, fiel al paper (salida directa, lr constante), **divergía**:
val RMSE 0.27 → 3.6 mm/h al pasar a grilla 50–100. El diagnóstico tuvo dos
partes:

1. En escala [0, 1], el cambio del caudal en 48 h tiene desvío
   **σ_Δ = 0.0046**. El modelo tiene que producir correcciones minúsculas sobre
   un valor que ya conoce (el caudal actual).
2. Al refinar, los armónicos nuevos arrancan en 0 (la función no cambia), pero
   **Adam mueve cada parámetro nuevo ~lr por paso**, sin importar la escala del
   gradiente. Con 100 armónicos por arista, eso es ruido de alta frecuencia
   enorme frente a 0.0046.

Dos cambios lo resolvieron (barrido corto: RFKAN 0.112 contra persistencia
0.138):

- **Salida residual** (`Forecaster`): `ŷ = q(t) + σ_Δ · núcleo(x)`. El núcleo
  aprende la corrección en una escala ~N(0, 1) y arranca cerca de persistencia.
  Se aplica **igual** a RFKAN, a sus ablaciones y a los baselines neuronales, y
  los clásicos (ELM, SVR) también predicen el delta. Así la comparación es
  justa.
- **lr por etapa de grilla ∝ 3/g** (`LR_GRID_DECAY = 1`): cada vez que se
  refina se crea un Adam nuevo con `lr · 3/g`. La suite `grid` incluye el brazo
  `RFKAN_L72_nodecay` para medir este efecto.

## Entrenamiento (`train.py`)

- **Pérdida:** MSE sobre las 48 horas, **dividida por σ_Δ²**; es decir, la MSE
  del error medido en desvíos típicos del cambio de caudal. El paper no dice qué
  pérdida usa; MSE es consistente con reportar RMSE. La división no cambia el
  óptimo pero sí la escala del gradiente. Sin ella, la MSE normalizada ronda
  2.5e-5 y el 99 % de los gradientes de LSTM, GRU y TCN (y el 74 % de los de
  RKAN) quedan por debajo del `eps` de Adam (1e-8), así que esos parámetros casi
  no se mueven. La primera corrida completa (archivada en
  `outputs/experiments_v1/`) tenía este problema: el LSTM terminó exactamente
  en persistencia.
- **Optimizador:** Adam, lr 1e-3, batch 256 para la familia KAN (el paper no los
  reporta) y clip de gradiente 1.0.
- **Grilla:** `[3, 5, 10, 20, 50, 100]`, con **3 épocas por grilla** (18 en
  total). El paper usa "50 steps" por grilla sobre ~100 k puntos; con 254 000
  ventanas, 3 épocas son ~3000 pasos por grilla. Es la traducción más cercana
  que entra en el tiempo disponible.
- **Selección de modelo:** el checkpoint con menor val RMSE (`_best.pt`), que
  es el usado para test.
- **Baselines neuronales:** mismas 18 épocas (el paper da el mismo presupuesto
  de iteraciones a todos), batch 32 (Tabla 5) y **la misma ventana L = 72 que
  RFKAN**. Con L = 336, el LSTM de 3 capas no sale de persistencia en 18 épocas
  (gradiente desvanecido en 336 pasos); con L = 72 llega a 0.109 en 3 épocas.

## Baselines (`baselines.py`)

- **Persistencia:** `ŷ(t+h) = q(t)`. No está en el paper, pero es el modelo base
  natural en hidrología.
- **ARIMA(2,1,1):** se ajusta sobre las 336 h de caudal de **cada** ventana (no
  hay "entrenamiento"). Por costo se evalúa en un submuestreo fijo de 2000
  ventanas de val y 2000 de test. En ~1 % de las ventanas el ajuste da NaN o
  infinito; en esos casos se usa persistencia. Algunos pronósticos finitos
  explotan (hasta 299 mm/h): es una limitación real de ARIMA con d = 1 y se
  reporta tal cual.
- **SVR** (RBF, C = 0.0043): entrada = últimas 24 h × 12 canales aplanadas, un
  SVR por hora de horizonte, 10 000 ventanas de train (SVR es O(n²)).
- **ELM:** 3 capas ocultas aleatorias ReLU de 512 unidades + ridge
  `(HᵀH + λI)β = HᵀΔ` con λ = 0.001, en forma cerrada. En la primera versión,
  λ estaba escalado por N; eso regularizaba tanto que el ELM quedaba en
  persistencia.
- **LSTM** (Tabla 5) y **GRU** (Tabla 6, por la inconsistencia del paper):
  3 capas, 64 unidades, cabeza MLP con ReLU.
- **TCN:** 3 convoluciones 1D con stride 2, padding "same" y ReLU.

## Modelo avanzado: Informer (`informer.py`)

El paper compara RFKAN contra Informer, WVS y TACDPG "con los parámetros de la
literatura" (Tabla 7). Implementamos **Informer** (Zhou et al., AAAI 2021) desde
cero, con los hiperparámetros de su paper: d_model 512, 8 cabezas, 2 capas de
encoder y 1 de decoder, d_ff 2048, factor 5, dropout 0.05, lr 1e-4 y batch 32.

- **ProbSparse self-attention:** para cada consulta se mide M = max − media de
  sus scores sobre una muestra de u = 5·ln(L) claves. Solo las u consultas con M
  más alto atienden de verdad; el resto toma la media de V en el encoder, o la
  suma acumulada en el decoder con máscara causal.
- **Destilación** entre capas del encoder: conv + BatchNorm + ELU + maxpool, que
  reduce el largo a la mitad.
- **Decoder generativo:** la entrada son las últimas 48 h conocidas más ceros
  para las 48 a predecir, y se produce todo el horizonte en una sola pasada. El
  modo es MS: 12 canales de entrada y 1 de salida.
- **Ventana:** encoder de 96 h (la del paper de Informer para horizonte 48).
- **Embedding:** conv1d + posicional, sin embedding temporal porque el dataset
  no trae fechas.
- **Salida:** la misma salida residual y la misma pérdida que el resto.

## Modelos avanzados: WVS (`wvs.py`) y TACDPG (`tacdpg.py`)

Implementados a partir de los papers completos (`docs/Zhao2024_WOA-VMD-SCINet.pdf`,
`docs/Zhang2024_TACDPG.pdf`). Lo que el paper no especifica está marcado como
supuesto en `config.py` y en `GAP_ANALISIS.md`.

**WVS** (Zhao et al., Energy Reports 12, 2024):

1. **Pearson:** las 3 variables meteorológicas más correlacionadas con el
   caudal, porque el paper usa entrada de dimensión 4 = 1 subsecuencia + 3
   variables.
2. **VMD propia, en lotes en GPU:** descompone en K modos el caudal de las
   últimas 96 h de cada ventana, usando solo su historia. Como no tiene
   parámetros entrenables, se precalcula una vez.
3. **WOA propio:** optimiza (K, α) con la entropía de envolvente como fitness,
   igual que el paper.
4. **Un SCINet propio por modo:** cada uno recibe [modo_k, 3 variables] y da
   salida 1, y la predicción es la suma de los modos. Hiperparámetros de su
   Tabla 5: levels 3, stacks 1, hidden 1, kernel 4, dropout 0.5. Los K SCINet se
   implementan como una red con convoluciones agrupadas, que da el mismo
   resultado y es 2.6 veces más rápida.

**TACDPG** (Zhang et al., Sustain. Energy Technol. Assess. 67, 2024):

1. **Random forest:** selecciona las variables de entrada.
2. **Outliers según su Ec. 1:** |y − μ| > 1.5σ, con μ y σ calculados por cuenca.
3. **CatBoost:** predice si la ventana es outlier y, con eso, decide qué
   agente predice (paso 7 del marco).
4. **Dos agentes TD3, uno por clase:**
   - actor convolucional y dos críticos MLP gemelos, cada uno con su red objetivo;
   - buffer de experiencia y actualización suave de las redes objetivo;
   - recompensa adaptativa de la Ec. 2, −[z0 − j(z0 − z1)/j_max]·|y − a|·e^(−λi).
5. **Episodios:** 64 ventanas consecutivas del archivo, con 256 episodios en
   paralelo.

## Experimentos (`experiments.py`)

Todos con semilla 42, el mismo split y el mismo presupuesto:

| Suite | Brazos | Análogo |
|---|---|---|
| `seqlen` | KAN/RKAN/FKAN/RFKAN × L ∈ {2, 12, 24, 72, 168, 336} | Fig. 5 + Tabla 3 |
| `grid` | RFKAN L=72: grilla extendida (…, 200), fija 3, fija 100, sin decaimiento de lr | Fig. 4 |
| `baselines` | Persistencia, ELM, SVR, ARIMA, LSTM, GRU, TCN | Tablas 5, 6 |
| `seeds` | RFKAN y LSTM con semillas 1, 2, 3 | — (el paper no lo hace) |
| `advanced` | Informer (implementación propia) | Tabla 7 |
| `seqlen_base` | LSTM / GRU / TCN × L ∈ {2, 12, 24, 168} | Fig. 5 (curvas de los otros modelos) |

Cada brazo guarda:

- `<arm>_history.csv`: métricas por época.
- `<arm>_best.pt` y `<arm>_last.pt`: checkpoints.
- `<arm>_preds.npz`: predicciones de val y test en mm/h.
- Una fila en `experiments_results.csv`.
- `pkl/<arm>.pkl`: todo lo anterior junto, más métricas por horizonte, NSE por
  cuenca, snapshot de la config y scaler.

## Problemas de ingeniería encontrados

- **Memoria de GPU (16 GB):**
  - RKAN con L=168 tardó 86 min porque en WSL, cuando la VRAM se llena, CUDA
    desborda a RAM en lugar de dar error y todo se vuelve lento.
  - La grilla 500 no entra con L=72, así que el schedule extendido se cortó en
    200.
  - La extensión de grilla spline con 4032 entradas (KAN aplanado, L=336) pedía
    6.9 GB de una vez; se resolvió por bloques de 256 entradas.
- **Reproducibilidad:** semilla fija para PyTorch/NumPy y generador propio para
  el orden de los batches. TF32 está activado por velocidad, así que las cifras
  pueden variar en el cuarto decimal entre GPUs.

## Cómo ejecutar

```bash
cd lab/repo
../.venv/bin/python train.py --model RFKAN --seq_len 72     # un modelo
../run_all.sh                                               # todo el pipeline
../.venv/bin/python predict.py --arm RFKAN_L72              # CSV de test
../.venv/bin/python analyze_results.py                      # tablas y figuras
```
