# Resultados — RFKAN para pronóstico de caudal

Todos los números salen de `lab/repo/outputs/experiments/experiments_results.csv`
(corrida v2, con la pérdida escalada por σ_Δ²). El detalle completo de cada brazo
está en `outputs/experiments/pkl/<arm>.pkl`, y las tablas y figuras de análisis
en `outputs/analysis/` (`analyze_results.py`).

**Condiciones comunes a todos los brazos:**
- Split del curso: 254 000 ventanas de train y 18 142 de val. Test = `test.h5`
  (27 983 ventanas × 48 h), con los objetivos de `test_targets.csv`.
- Semilla 42 y 18 épocas (3 por grilla × 6 grillas). Se reporta el checkpoint
  con menor val RMSE.
- Métricas en mm/h. CORR = R² (Ec. 9 del paper). NSE = mediana del NSE por
  cuenca (508 cuencas).
- Persistencia (repetir el último caudal): val RMSE **0.1379**, test **0.1188**.

> Corrida v1 (archivada en `outputs/experiments_v1/`): la MSE normalizada dejaba
> los gradientes por debajo del ε de Adam, y el LSTM quedaba en persistencia. Los
> números de v1 **no** se usan en ningún entregable.

---

## 1. Ablación de arquitectura (análogo a la Tabla 3), L = 72

| Modelo | Variante | val RMSE | test RMSE | test MAE | test CORR | NSE med. | Tiempo (s) |
|---|---|---|---|---|---|---|---|
| KAN | splines, sin recurrencia | 0.1206 | 0.1100 | 0.0350 | 0.567 | 0.318 | 88 |
| **RKAN** | splines + recurrencia | **0.1048** | **0.1033** | **0.0270** | **0.619** | **0.520** | 663 |
| FKAN | Fourier, sin recurrencia | 0.1350 | 0.1172 | 0.0288 | 0.510 | 0.415 | 49 |
| RFKAN | Fourier + recurrencia (paper) | 0.1076 | 0.1052 | 0.0285 | 0.604 | 0.478 | 392 |
| Persistencia | — | 0.1379 | 0.1188 | 0.0246 | 0.496 | 0.492 | 0 |

- **La recurrencia decide.** FKAN → RFKAN: −10.2 % de RMSE de test.
  KAN → RKAN: −6.1 %. Coincide con el paper.
- **Fourier no mejora a splines.** RKAN → RFKAN: +1.9 % de RMSE de test (peor).
  El paper reporta −22 % (RKAN 4.37 → RFKAN 3.42). **No se reproduce.**
- **Fourier sí es más rápido.** RFKAN tarda 0.59× lo que RKAN (L=72) y
  0.17× con L=168. El paper reporta −24 %. Coincide, y la diferencia es mayor.
- RFKAN mejora a persistencia en **−11.4 %** de RMSE de test.

## 2. Largo de la ventana de entrada (análogo a la Fig. 5)

val RMSE (mm/h). Figura: `outputs/analysis/fig5_seqlen.png`.

| Modelo | L=2 | L=12 | L=24 | L=72 | L=168 | L=336 |
|---|---|---|---|---|---|---|
| KAN | 0.1109 | 0.1104 | 0.1134 | 0.1206 | 0.1263 | 0.1282 |
| RKAN | 0.1060 | 0.1065 | 0.1059 | **0.1048** | 0.1052 | 0.1053² |
| FKAN | 0.1115 | 0.1207 | 0.1271 | 0.1350 | 0.1380 | 0.1380 |
| RFKAN | 0.1091 | 0.1106 | 0.1093 | 0.1076 | 0.1076 | 0.1083 |

test RMSE: RKAN 0.1047 / 0.1037 / **0.1027** / 0.1033 / 0.1033 / 0.1034; RFKAN
0.1059 / 0.1065 / 0.1056 / 0.1052 / 0.1049 / 0.1059 (L = 2 / 12 / 24 / 72 / 168 / 336).

² RKAN con L = 336 no entra en 16 GB con grilla 100: se quedó sin memoria de GPU
al evaluar la época 16 (dos veces: primero en `extend_grid`, que se arregló
evaluando por bloques, y después en la evaluación de validación). Se reporta su
mejor checkpoint (época 12, grilla 20), el mismo criterio de selección que el
resto de los brazos. Con L = 168 y grilla 100, cada época tardó ~25 min porque la
VRAM desbordaba a RAM.

- **Con recurrencia, el largo casi no importa:** RKAN y RFKAN varían menos de
  0.003 entre L=2 y L=336.
- **Sin recurrencia, peor cuanto más larga es la ventana.** Con la ventana
  aplanada crecen las entradas (12·L) sin estructura temporal. FKAN con L=168 y
  L=336 queda en persistencia (0.1380).
**Modelos base para distintos largos** (val RMSE / test RMSE; la Fig. 5 del paper
también incluye a los otros modelos):

| Modelo | L=2 | L=12 | L=24 | L=72 | L=168 |
|---|---|---|---|---|---|
| LSTM | 0.1062 / 0.1051 | 0.1034 / 0.1031 | **0.1016** / 0.1020 | 0.1021 / 0.1012 | 0.1379 / 0.1188 ³ |
| GRU | 0.1055 / 0.1059 | 0.1044 / 0.1027 | 0.1020 / 0.1022 | 0.1028 / 0.1015 | 0.1017 / **0.1011** |
| TCN | 0.1051 / 0.1060 | 0.1064 / 0.1050 | 0.1049 / 0.1045 | 0.1051 / 0.1032 | 0.1059 / 0.1058 |

³ LSTM con L = 168 no sale de persistencia (igual que en 2 de 4 semillas con L = 72).

- **Con L = 2 todos los modelos recurrentes empatan** (0.105–0.106 val), KAN
  incluida.
- **Con más historia, LSTM y GRU mejoran hasta L = 24–72**, mientras que RKAN y
  RFKAN quedan planos. La memoria lineal de los nodos kernel recurrentes
  (Ec. 3) no aprovecha la historia larga; las compuertas de LSTM y GRU sí.
- **Tiempo:** no crece "exponencialmente" con L para los baselines (a diferencia
  de lo que dice el paper). Con cuDNN, LSTM y GRU tardan 4–6 min con cualquier
  L. La familia KAN con splines sí se encarece mucho con L (RKAN: 85 s con L=2,
  103 min con L=168).
- **El óptimo de L = 2 del paper solo vale para las variantes sin
  recurrencia**, que con L=2 quedan cerca de las recurrentes. Para RFKAN, el
  mejor largo es L = 72–168.

## 3. Refinamiento de grilla (análogo a la Fig. 4), RFKAN L = 72

Figura: `outputs/analysis/fig4_grid.png`.

| Variante | val RMSE | test RMSE | Mejor grilla |
|---|---|---|---|
| refinamiento 3 → 100, lr ∝ 3/g (estándar) | **0.1076** | **0.1052** | 20 |
| refinamiento 3 → 200 | 0.1076 | 0.1052 | 20 |
| grilla 3 fija | 0.1096 | 0.1070 | 3 |
| refinamiento 3 → 100 con lr constante | 0.1136 | 0.1071 | 5 |
| grilla 100 fija desde el inicio | 0.1405 | 0.1220 | 100 |

- **Mejor época por grilla (20 brazos de la suite seqlen):** grilla 5: 2,
  grilla 10: 10, grilla 20: 5, grilla 100: 3. Los 3 casos en grilla 100 son
  FKAN, cuyo mejor valor sigue siendo malo (0.1115–0.1350). Ningún modelo
  competitivo mejora al llegar a 100. El paper reporta el óptimo en 100.
- **Refinar no aporta precisión:** grilla 3 fija queda a +1.9 % de val RMSE
  (+1.7 % en test) del refinamiento completo.
- **Pero la forma de llegar a una grilla fina importa.** Empezar con 100
  armónicos queda peor que persistencia (0.1405). Refinar sin bajar el lr se
  degrada al pasar a grilla 20 (val 0.197) y termina en 0.181, peor que
  persistencia. El refinamiento progresivo con lr
  decreciente es lo que permite tener una grilla fina sin romper el modelo.
- 500 armónicos no entran en 16 GB con L=72, así que el schedule extendido
  llega a 200.

**Predicciones de test** (requisito de la consigna):
`outputs/experiments/predictions_test_RFKAN_L72.csv`, 27 983 filas con formato
`Id, q_01..q_48` en mm/h. Las predicciones negativas se recortan a 0, porque el
caudal específico no puede ser negativo. Test RMSE 0.1051, MAE 0.0273,
CORR 0.605.

## 4. Comparación con los modelos base (análogo a la Tabla 6), test

| Modelo | RMSE | MAE | CORR | KGE | NSE med. | Tiempo (s) |
|---|---|---|---|---|---|---|
| Persistencia | 0.1188 | 0.0246 | 0.496 | 0.736 | 0.492 | 0 |
| ARIMA(2,1,1)¹ | 6.7283 | 0.2037 | −1378 | −35.2 | **0.733** | 68 |
| SVR (RBF, C = 0.0043) | 0.1187 | 0.0256 | 0.497 | 0.731 | 0.432 | 61 |
| ELM (3 × 512 ReLU, λ = 0.001) | 0.1423 | 0.0717 | 0.277 | 0.608 | −0.409 | 0.4 |
| **LSTM** (3 capas, L=72) | **0.1012** | **0.0241** | **0.634** | 0.681 | 0.569 | 327 |
| GRU (3 capas, L=72) | 0.1015 | 0.0251 | 0.632 | 0.699 | 0.535 | 318 |
| TCN (3 conv, stride 2, L=72) | 0.1032 | 0.0285 | 0.620 | 0.658 | 0.518 | 273 |
| RKAN L=72 | 0.1033 | 0.0270 | 0.619 | 0.692 | 0.520 | 663 |
| RFKAN L=72 | 0.1052 | 0.0285 | 0.604 | 0.682 | 0.478 | 392 |

¹ ARIMA: submuestreo fijo de 2000 ventanas; en ~1 % de ellas el ajuste da
NaN/∞ y se usa persistencia. Persistencia en ese mismo submuestreo: val RMSE
0.1678.

- **LSTM y GRU superan a toda la familia KAN; TCN empata con RKAN.** El paper
  reporta RFKAN −58 % de RMSE frente a GRU/TCN. **No se reproduce.** Aquí las
  condiciones son iguales para todos: mismas particiones, presupuesto, salida
  residual y pérdida. Además, la evaluación es mucho más amplia (27 983
  ventanas de 508 cuencas, contra 4 días de una planta).
- **ARIMA:** en la cuenca típica le gana a todos (NSE mediano 0.733), pero en
  ~0.5 % de las ventanas la diferenciación extrapola una tendencia y explota
  (hasta 299 mm/h), lo que arruina el RMSE global. Recortar al rango de train no
  alcanza (test RMSE 0.86).
- **SVR con el C del paper** regulariza tanto que predice un cambio nulo:
  persistencia.
- **ELM:** con λ = 0.001 sobreajusta (peor que persistencia). Con λ·N (v1)
  quedaba en persistencia (val 0.1371).

## 4b. Modelos avanzados (análogo a la Tabla 7), test

Los tres son implementaciones propias: `informer.py`, `wvs.py` y `tacdpg.py`.
Los detalles y los supuestos están en `IMPLEMENTACION_CODIGO.md` y en
`GAP_ANALISIS.md`.

| Modelo | val RMSE | test RMSE | test MAE | test CORR | NSE med. | Parámetros | Tiempo (s) |
|---|---|---|---|---|---|---|---|
| Informer | 0.1099 | 0.1032 | 0.0284 | 0.620 | 0.533 | 11.3 M | 2137 |
| WVS (WOA-VMD-SCINet) | 0.1087 | 0.1061 | **0.0236** | 0.598 | 0.566 | 40 k | 4202 |
| TACDPG | 0.1393 | 0.1205 | 0.0349 | 0.481 | 0.301 | 394 k | 69 |
| **RFKAN L=72** | **0.1076** | 0.1052 | 0.0285 | 0.604 | 0.478 | 122 k | 392 |
| RKAN L=72 | 0.1048 | 0.1033 | 0.0270 | 0.619 | 0.520 | 44 k | 663 |
| LSTM | 0.1021 | 0.1012 | 0.0241 | 0.634 | 0.569 | 94 k | 327 |

- **Validación:** RFKAN supera a los tres modelos avanzados, como afirma el
  paper.
- **Test:** Informer es mejor que RFKAN (−1.9 %) y WVS empata con él (+0.9 %).
  WVS tiene además menor MAE que RFKAN y un NSE mediano más alto. La ventaja de RFKAN
  sobre los modelos avanzados **no es consistente** entre validación y test.
- **Tiempo:** RFKAN tarda 0.18× lo que Informer y 0.09× lo que WVS. El paper
  reporta −24 %: **se reproduce**, con una diferencia mayor.
- **WVS:** WOA elige K = 5 modos y α = 100; Pearson, humedad, temperatura y
  precipitación. Tiene el menor MAE entre los modelos principales (ventana de 72 h y
  avanzados; con otras ventanas, LSTM_L2 0.0226 y GRU_L24 0.0231 lo bajan), pero
  su RMSE queda al nivel de RFKAN.
- **TACDPG queda peor que persistencia.**
  - El random forest elige evaporación potencial, radiación de onda corta,
    precipitación, viento u/v y caudal.
  - El 8.6 % de las ventanas de train son outliers según la Ec. 1.
  - CatBoost acierta el 93 % en val.
  - Los dos agentes TD3 tienen su mejor validación en la primera evaluación y
    después se degradan. El aprendizaje por refuerzo, con una recompensa
    escalar por ventana, no logra aprender una corrección útil de 48 h. Los
    parámetros de la recompensa y del entrenamiento vienen del material
    suplementario, que no estuvo disponible, así que son supuestos.

## 5. Robustez a la semilla (semillas 42, 1, 2, 3), L = 72

| Modelo | test RMSE por semilla | media ± desvío |
|---|---|---|
| RFKAN | 0.1052 · 0.1061 · 0.1065 · 0.1054 | **0.1058 ± 0.0006** |
| LSTM | 0.1012 · 0.1011 · **0.1188** · **0.1188** | 0.1100 ± 0.0102 |

- **El LSTM no sale de persistencia en 2 de 4 semillas** (seed2, seed3): val
  RMSE 0.1379 durante las 18 épocas.
- **RFKAN es estable:** rango de 0.0013 entre semillas.
- **Lectura:** el LSTM es el mejor modelo *cuando entrena*, pero RFKAN es más
  robusto y en promedio sobre semillas lo supera. Con la salida residual, los
  modelos KAN arrancan en persistencia y se alejan de ella de forma fiable.

## 6. Análisis de errores (test, L = 72)

Figuras: `fig_horizonte.png`, `fig_nse_cuencas.png`, `fig_hidrogramas.png`.
Tablas: `errores_por_regimen.md`, `picos.md`.

**RMSE por régimen** (mm/h):

| Régimen | Persist. | KAN | RKAN | RFKAN | LSTM | TCN |
|---|---|---|---|---|---|---|
| caudal actual Q1 (bajo) | **0.0457** | 0.0525 | 0.0463 | 0.0475 | 0.0460 | 0.0480 |
| caudal actual Q4 (alto) | 0.2124 | 0.1878 | 0.1782 | 0.1817 | **0.1734** | 0.1765 |
| sin lluvia en 24 h | 0.0689 | 0.0711 | 0.0685 | 0.0693 | **0.0676** | 0.0693 |
| lluvia 24 h alta (top 10 %) | 0.3211 | 0.2589 | 0.2342 | 0.2395 | **0.2236** | 0.2274 |
| temperatura Q1 (frío) | 0.1401 | 0.1200 | 0.1139 | 0.1168 | **0.1106** | 0.1145 |
| temperatura Q4 (cálido) | 0.0913 | 0.0907 | 0.0798 | 0.0816 | **0.0776** | 0.0794 |

- **La ganancia sobre persistencia está casi toda en caudal alto y con
  lluvia.** Con lluvia alta, RFKAN −25 % y LSTM −30 %. En caudal bajo nadie le
  gana a persistencia.
- **El error es mayor en frío** (cuartil 1 de temperatura, proxy de invierno).
  Es el análogo del análisis por estación del paper, que también encuentra
  peor desempeño en invierno.

**Picos** (y ≥ p99 = 0.753 mm/h):

| Modelo | RMSE picos | Sesgo relativo | RMSE resto |
|---|---|---|---|
| Persistencia | 0.958 | −24 % | 0.071 |
| KAN | 0.900 | −32 % | 0.064 |
| RKAN | 0.905 | −36 % | 0.050 |
| RFKAN | 0.918 | −38 % | 0.052 |
| LSTM | 0.910 | −38 % | 0.045 |
| TCN | 0.913 | −39 % | 0.048 |

- **Ningún modelo anticipa las crecidas.** Todos subestiman los picos en
  32–39 %, más que persistencia, porque la MSE sobre 48 h con mayoría de horas
  de recesión los empuja a curvas suaves. Es la limitación principal.
- **En caudal bajo, la familia KAN oscila hora a hora** (RFKAN es el que más)
  y LSTM/TCN dan curvas suaves: las 48 salidas de la capa 2 no tienen acople
  temporal entre sí.

## 7. Conclusiones

1. RFKAN implementado desde cero funciona en caudal: −11 % de RMSE de test
   frente a persistencia y mejor que ARIMA, SVR y ELM.
2. **La recurrencia es la contribución que se sostiene** (coincide con el paper).
3. **Fourier no se sostiene:** más rápido, pero menos preciso que splines, y el
   que más oscila. El argumento del paper (periodicidad) no aplica al caudal.
4. **Refinar la grilla hasta 100 no ayuda a la precisión.** El esquema
   progresivo con lr decreciente sí es necesario para usar grillas finas.
5. **LSTM y GRU superan a RFKAN** en igualdad de condiciones, aunque el LSTM
   falla en 2 de 4 semillas (y con L=168), y RFKAN es estable. Frente a los
   modelos avanzados (Tabla 7), RFKAN gana en validación y es mucho más rápido,
   pero en test Informer lo supera y WVS empata; TACDPG no llega a persistencia.
6. **Limitación principal:** los picos de crecida (−32 a −39 %).
