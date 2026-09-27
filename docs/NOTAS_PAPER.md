# Notas del paper — Rong, Lin & Xie 2025, RFKAN

Datos extraídos del PDF (`Rong2025_RFKAN_PV.txt`, `Rong2025_RFKAN_PV.pdf`).
Referencia rápida para la presentación y para la implementación.

**Cita:** D. Rong, Z. Lin, G. Xie. *Recurrent Fourier-Kolmogorov Arnold Networks
for photovoltaic power forecasting*. Scientific Reports 15:4684 (2025).
DOI `10.1038/s41598-025-88959-5`. Liaoning Technical University (Huludao, China).
Recibido 29-sep-2024, aceptado 3-feb-2025.

## Problema

Pronóstico **day-ahead** de potencia fotovoltaica (PV): dadas variables
meteorológicas y potencia pasada, predecir la potencia del día siguiente.
Motivación: despacho de la red eléctrica. Los autores sostienen que:

- los modelos tipo MLP escalan mal en parámetros;
- **KAN** (Liu et al. 2024) reduce parámetros pero (a) no extrae bien
  dependencias temporales entre pasos y (b) ajusta mal series periódicas, y los
  splines son costosos de entrenar.

## Propuesta: RFKAN

KAN de **2 capas** (teorema de Kolmogorov–Arnold: `n → 2n+1 → salida`) con dos
modificaciones:

1. **Nodos kernel recurrentes** (inspirados en RNN, Ec. 3):
   `h_{l,i}(t) = W_hh h_{l,i}(t−1) + W_hz x_{l,i}(t)`.
   Estos nodos entran **en paralelo** con los nodos normales a cada capa; cada
   función de arista pasa a depender del tiempo y de la memoria:
   `x_{l+1,j}(t) = Σ_i φ_{l,j,i,t}(x_{l,i}(t), h_{l,i}(t))` (Ec. 4).
   Hay tantos nodos recurrentes como nodos normales.
2. **Serie de Fourier** en lugar de B-splines como función de activación
   aprendible (Ec. 8):
   `φ_F(x) = Σ_{i=1..d} Σ_{k=1..g} a_ik cos(k x_i) + b_ik sin(k x_i)`,
   con `g` = tamaño de grilla (número de armónicos) y `d` = dimensión de entrada.
   Argumento: captura periodicidad global con pocos parámetros y entrena más
   rápido que splines.

Flujo (Fig. 1): nodos recurrentes de la entrada → Layer 1 (Fourier, grilla
gruesa, salida `2n+1` nodos) → de nuevo nodos recurrentes sobre esa salida →
Layer 2 (misma grilla) → predicción. **Refinamiento de grilla:** tras entrenar
en la grilla gruesa, el modelo "hereda la curva aprendida" y sigue entrenando
en una grilla más fina.

Número de nodos normales de entrada: `n = a × b` (a = largo de la serie,
b = variables por paso). Tabla 4 dice "Input nodes 12".

## Datos del paper

- Planta PV de 45 kW en **Tieling** (Liaoning), todo 2023, muestreo **5 min**
  → 105 120 puntos.
- Variables: temperatura, humedad relativa, nubosidad, irradiancia, viento
  (velocidad y dirección) + potencia PV.
- Preprocesamiento: dirección de viento en código binario de 3 bits (Tabla 2);
  faltantes por interpolación lineal; outliers (irradiancia fuera de 0–1000
  W/m², potencia ≤ 0 o > capacidad) por interpolación; **min-max a [0, 1]**.
- Split **cronológico**. Ablación: test = 10 de mayo. Comparación por estación:
  test = 10-mar, 11-jun, 2-sep, 20-dic (un día por estación).
- Datos **no públicos** (Liaoning Tieling Power Supply Company).

## Métricas (Ec. 9)

MAE, RMSE y **"CORR" = R²** = `1 − Σ(Y−Y')² / Σ(Y−Ȳ)²` (a pesar del nombre, no
es la correlación de Pearson). En el texto de la Tabla 6 dice "MSE" cuando
quiere decir MAE.

## Experimentos y resultados reportados

### Ablación de arquitectura (Tabla 3, test 10-may)

| Modelo | RMSE | MAE | CORR | Tiempo (s) |
|---|---|---|---|---|
| KAN | 5.4265 | 3.0141 | 0.8679 | 4125.84 |
| RKAN | 4.3688 | 2.6427 | 0.9246 | 4793.62 |
| FKAN | 4.1829 | 2.7677 | 0.9192 | 3125.84 |
| **RFKAN** | **3.4168** | **1.6146** | **0.9461** | 3519.38 |

- RKAN = RFKAN sin Fourier (splines + recurrencia).
- FKAN = RFKAN sin recurrencia (Fourier sin nodos recurrentes).
- Reportan: Fourier reduce el tiempo de entrenamiento un 24.24 %. RFKAN reduce
  RMSE 37 % y MAE 46 %, y sube CORR 8.3 %, contra KAN.

### Grilla (Fig. 4)

Grilla `[3, 5, 10, 20, 50, 100]`, **50 steps por grilla**. El RMSE cae en cada
refinamiento. Por encima de 100 el tiempo crece mucho y el sobre-refinamiento
empeora el pronóstico, así que eligen tope 100.

### Largo de la secuencia de entrada (Fig. 5)

Con grilla 100 y horizonte de **288 pasos** (2 días a 5 min), barren el largo de
entrada de 1 a 288. El tiempo de los otros modelos crece "exponencialmente"; el
mejor RFKAN se obtiene con **largo 2**. (Tabla 4 dice `Input seq_len 12`;
inconsistente con el texto.)

### Hiperparámetros (Tabla 4 y Tabla 5)

- **RFKAN:** Step 50, Grid [3,5,10,20,50,100], K 3, Input nodes 12,
  Input seq_len 12, Recurrent kernel nodes 12.
- **ARIMA:** (2, 1, 1).
- **SVM:** kernel RBF, C = 0.0043.
- **ELM:** 3 capas ocultas, ReLU, regularización 0.001.
- **LSTM:** 3 capas, lr 0.001, batch 32, ReLU, Adam.
- **TCN:** 3 capas convolucionales, stride 2, padding "same", ReLU.
- Presupuesto: 50 iteraciones × 6 grillas = 300 iteraciones para todos.

### Comparación con baselines (Tabla 6, por estación)

RFKAN gana en las 4 estaciones: RMSE 2.57–3.42 contra 6.2–10.3 de los demás
(ARIMA, ELM, SVM, **GRU**, TCN). **La Tabla 5 lista LSTM, pero la Tabla 6
reporta GRU.** Declaran al menos −58.6 % de RMSE y +10.3 % de CORR. RFKAN es el
más lento (~1140 s contra ~300 s de ARIMA y ~1000 s de GRU/TCN).

### Comparación con modelos avanzados (Tabla 7)

Contra Informer, WVS (WOA-VMD-SCINet) y TACDPG: RFKAN RMSE 2.57–3.42 contra
3.35–4.20. Declaran al menos −5 % de RMSE/MAE, +2 % de CORR y −24 % de tiempo.

## Conclusión del paper

1. La recurrencia mejora la extracción de dependencias; Fourier captura
   ciclos rápido.
2. Óptimo: grilla 100 y largo de entrada 2.
3. RFKAN supera a baselines y modelos avanzados en todas las estaciones.

Trabajo futuro: precisión de los datos y data drift.

## Puntos débiles / ambigüedades (para el gap analysis)

- **Evaluación:** un solo día de test por estación (288 puntos), sin varias
  semillas ni intervalos de confianza.
- **Arquitectura:** no se especifica cómo se combinan `x` y `h` dentro de φ
  (Ec. 4), ni cómo se lee la salida multi-paso (288 valores) de una capa de
  salida de 1 nodo.
- **Grilla:** "grilla más fina" con Fourier = más armónicos. No explican cómo se
  hereda la curva (para Fourier es trivial: los nuevos coeficientes empiezan en 0).
- **Hiperparámetros:** optimizador, lr y batch de RFKAN no se reportan.
- **Inconsistencias:** `seq_len` 12 (Tabla 4) contra 2 (texto); LSTM (Tabla 5)
  contra GRU (Tabla 6); "MSE" contra MAE.
- **Tiempos:** KAN y RKAN tardan más que FKAN, lo que es coherente, pero los
  tiempos absolutos (miles de segundos para 105 k puntos) sugieren una
  implementación poco eficiente de splines (pykan).
