# Gap analysis — paper contra nuestra implementación

Qué del paper se implementó tal cual, qué se adaptó y por qué, y qué no se hizo.
Leyenda: ✅ igual al paper · 🔧 adaptado (con justificación) · ❌ no implementado.

## 1. Arquitectura

| Componente | Paper | Nuestra implementación | Estado |
|---|---|---|---|
| KAN de 2 capas `n → 2n+1 → out` | Ec. 1, Fig. 1 | `RFKAN`: 12 (+12 recurrentes) → 25 → 48 | ✅ |
| Serie de Fourier como activación | Ec. 8 | `FourierKANLayer`, `g` armónicos, + bias | ✅ |
| Nodos kernel recurrentes | Ec. 3, lineales | `RecurrentKernelNodes`, W_hh y W_hz matrices sin bias | 🔧 el paper no dice si W son escalares o matrices |
| φ(x, h) | Ec. 4, sin detalle | nodos en paralelo: `φ_x(x) + φ_h(h)` (concatenación) | 🔧 interpretación de la Fig. 1 |
| Recurrencia también antes de la capa 2 | Fig. 1 | `rec1` sobre la salida de la capa 1 | ✅ |
| Salida | 1 nodo | 48 nodos (una por hora), leídos en el último paso | 🔧 la consigna pide 48 h |
| Splines (KAN/RKAN) | pykan (ref. 39) | `SplineKANLayer` propio, K = 3, base silu | ✅ (propio, lo exige la consigna) |
| Refinamiento de grilla | hereda la curva | Fourier: copia exacta; spline: mínimos cuadrados | ✅ |
| Variantes sin recurrencia | "removing the RNN structure" | ventana aplanada `n = 12·L`, oculto 25 | ✅ (`n = a × b` del paper) |
| **Salida residual** | no existe | `ŷ = q(t) + σ_Δ · núcleo(x)` | 🔧 sin esto diverge (ver §4) |

## 2. Entrenamiento

| Aspecto | Paper | Nosotros | Estado |
|---|---|---|---|
| Grilla | [3, 5, 10, 20, 50, 100] | igual | ✅ |
| Pasos por grilla | 50 "steps" | 3 épocas (~3000 pasos de batch 256) | 🔧 254 k ventanas contra 105 k puntos |
| Presupuesto de los baselines | 300 iteraciones, igual para todos | 18 épocas, igual para todos | ✅ misma idea |
| Optimizador, lr, batch de RFKAN | no reportados | Adam 1e-3, batch 256, clip 1.0 | 🔧 |
| lr por etapa | no reportado | `lr · 3/g` | 🔧 necesario (§4); ablación `nodecay` |
| Pérdida | no reportada | MSE / σ_Δ² (evita gradientes bajo el eps de Adam) | 🔧 |
| Largo de entrada | "2" (texto) / "12" (Tabla 4) | barrido {2, 12, 24, 72, 168, 336}; principal L = 72 | 🔧 |
| Grilla > 100 | "degrada" | probado hasta 200; 500 no entra en 16 GB | 🔧 |
| RKAN con L = 336 | — | grilla 100 no entra en 16 GB: se reporta el mejor checkpoint (época 12, grilla 20) | 🔧 memoria |

## 3. Datos, evaluación y experimentos

| Aspecto | Paper | Nosotros | Estado |
|---|---|---|---|
| Dataset | PV Tieling, 1 planta, 5 min | Rainfall-Runoff, 508 cuencas, 1 h | 🔧 consigna |
| Faltantes/outliers | interpolación lineal | no hay NaN (verificado) | ✅ n/a |
| Codificación binaria de la dirección del viento | Tabla 2 | el viento viene en componentes u/v | ✅ n/a |
| Normalización | min-max [0, 1] | min-max [0, 1] con estadísticos de train | ✅ |
| Split | cronológico, 1 día de test por estación | split del curso (train/val) + test.h5 | 🔧 consigna |
| Métricas | RMSE, MAE, CORR (= R²) | las tres + NSE por cuenca + KGE | ✅ + extras |
| Ablación Tabla 3 | KAN / RKAN / FKAN / RFKAN | igual, para 6 largos de ventana | ✅ ampliada |
| Fig. 4 (grilla) | curvas por grilla | historial por época + grilla fija 3, fija 100 y extendida | ✅ ampliada |
| Fig. 5 (largo de entrada) | 1–288, todos los modelos | familia KAN 2–336 y LSTM/GRU/TCN 2–168 | ✅ |
| Baselines Tabla 5/6 | ARIMA, SVM, ELM, LSTM/GRU, TCN | todos, más LSTM **y** GRU, más persistencia; los neuronales con L = 72 (igual que RFKAN) | ✅ |
| ARIMA | sobre la serie | por ventana, submuestreo de 2000 | 🔧 costo |
| SVM | todo el train | SVR sobre 10 000 ventanas, 24 h de historia | 🔧 O(n²) |
| Análisis por estación | 4 días de test | error por cuartil de temperatura (proxy de estación) | 🔧 las fechas no están en el dataset |
| Semillas | 1 corrida | 3 semillas extra para RFKAN y LSTM | ✅ + extra |
| Modelos avanzados (Tabla 7) | Informer, WVS, TACDPG | Informer (Zhou et al. 2021), WVS = Pearson + WOA + VMD + SCINet, TACDPG = RF + CatBoost + DDPG con críticos gemelos | ✅ los tres, implementaciones propias; WVS y TACDPG desde sus resúmenes (texto completo no accesible) |
| `y_aux` (meteorología futura) | n/a | no usado | — opcional según la consigna |

### WVS y TACDPG: qué sale del paper y qué es supuesto

Papers completos en `docs/Zhao2024_WOA-VMD-SCINet.pdf` y `docs/Zhang2024_TACDPG.pdf`.

| Componente | Del paper | Nuestra implementación |
|---|---|---|
| WVS: selección de variables | Pearson, se descartan las de baja correlación | ✅ las 3 meteorológicas con mayor \|r\| |
| WVS: fitness de WOA | entropía de envolvente sobre [K, α] | ✅ igual; límites K ∈ [2, 8], α ∈ [100, 4000], 10 ballenas × 15 iteraciones (supuesto) |
| WVS: entrada de SCINet | "cada subsecuencia combinada con las variables", entrada de dimensión 4, salida 1 | ✅ un SCINet por modo (convoluciones agrupadas), la predicción es la suma de modos |
| WVS: SCINet | levels 3, stacks 1, hidden 1, dropout 0.5, kernel 4 (la Tabla 5 dice 1), 50 épocas | ✅ kernel 4 · 🔧 18 épocas (mismo presupuesto que el resto) · lr 3e-3 (supuesto) |
| WVS: VMD | sobre la serie completa | 🔧 por ventana (solo historia, sin fuga de futuro): no hay objetivo por modo, se entrena con la pérdida del total |
| TACDPG: features | random forest | ✅ |
| TACDPG: outliers | Ec. 1, \|y − μ\| > κσ con κ = 1.5 | ✅ μ y σ por cuenca; una ventana es outlier si alguna de sus 48 h lo es |
| TACDPG: CatBoost | predice si hay outlier; enruta a TACDPG de outliers o de normales | ✅ dos agentes, uno por clase |
| TACDPG: red | actor conv (online/target), 2 críticos MLP gemelos (online/target), soft update τ, buffer | ✅ TD3 completo |
| TACDPG: recompensa | Ec. 2: −[z0 − j(z0 − z1)/j_max]·\|y − a\|·e^(−λi) | ✅ igual; z0 = 1, z1 = 0.1, λ = 0.01 (supuestos: material suplementario no disponible) |
| TACDPG: episodios | serie temporal de la planta, paso = instante | 🔧 episodio = 64 ventanas consecutivas del archivo, 256 episodios en paralelo; γ = 0.9, τ = 0.005 (supuestos) |
| TACDPG: acción | potencia en t | 🔧 corrección de las 48 h en unidades de σ_Δ (salida residual) |

## 4. Por qué la salida residual y el lr decreciente no son "trampa"

- Con la formulación literal (salida directa en [0, 1], lr constante), RFKAN
  diverge al refinar: val RMSE 0.27 → 3.6 mm/h, 26 veces peor que persistencia.
- **Causa del lr:** Adam normaliza el gradiente, así que cada armónico nuevo se
  mueve ~lr por paso aunque su gradiente sea minúsculo. Con 100 armónicos y un
  objetivo cuyo cambio típico es σ_Δ = 0.0046, eso es ruido.
- **Causa del objetivo:** predecir el caudal "desde cero" obliga a la red a
  reconstruir un valor que ya está en la entrada. La forma residual es la
  práctica estándar en hidrología y en pronóstico multi-paso.
- **Se aplica a todos por igual:** familia KAN, LSTM, GRU y TCN. ELM y SVR
  también predicen el delta. Ningún modelo recibe una ventaja que los otros no
  tengan.
- La propuesta central (**nodos recurrentes + Fourier + refinamiento de
  grilla**) no cambia, y cada parte se evalúa por ablación.

## 5. Inconsistencias del paper encontradas

1. **Largo de entrada:** la Tabla 4 dice `Input seq_len 12`, pero el texto y la
   conclusión dicen que el mejor es **2**.
2. **LSTM contra GRU:** la Tabla 5 da hiperparámetros de LSTM y la Tabla 6
   reporta **GRU**. Corrimos ambos.
3. **"MSE" contra MAE:** el texto de la Tabla 6 dice "MSE reducido 70.74 %",
   pero la tabla reporta MAE.
4. **"CORR":** se define como R², no como correlación.
5. **Fig. 5:** el horizonte es 288 pasos y la salida de la red es 1 nodo; no se
   explica cómo se produce la salida multi-paso.
6. **Evaluación estrecha:** un día de test por estación (288 puntos), sin
   semillas ni intervalos. Con 27 983 ventanas × 48 h y 508 cuencas, nuestra
   evaluación es mucho más amplia. Por eso las diferencias entre variantes son
   más chicas y más confiables.

## 6. Qué se reproduce y qué no (resultados finales, ver `RESULTADOS.md`)

| Afirmación del paper | Nuestro resultado (test, L = 72) | ¿Se reproduce? |
|---|---|---|
| La recurrencia mejora a KAN | FKAN → RFKAN −10.2 %, KAN → RKAN −6.1 % de RMSE | ✅ sí |
| Fourier mejora la precisión | RKAN → RFKAN +1.9 % (peor) | ❌ no |
| Fourier reduce el tiempo (−24 %) | RFKAN 0.59× RKAN (L=72), 0.17× (L=168) | ✅ sí, más aún |
| Grilla óptima = 100 | mejor época en grilla 10–20; ningún modelo competitivo mejora en 100 | ❌ no |
| El refinamiento de grilla ayuda | no mejora la precisión, pero sin él (grilla 100 fija) o sin bajar el lr, el modelo queda peor que persistencia | ⚠️ parcial |
| Largo de entrada óptimo = 2 | recurrentes insensibles a L; solo las variantes sin recurrencia prefieren L corto | ⚠️ parcial |
| RFKAN supera a GRU/TCN (−58 % RMSE) | LSTM 0.1012, GRU 0.1015, TCN 0.1032 < RFKAN 0.1052 | ❌ no |
| RFKAN supera a ARIMA/SVM/ELM | sí (ARIMA explota, SVR ≈ persistencia, ELM sobreajusta) | ✅ sí |
| RFKAN supera a Informer (Tabla 7) | val: RFKAN 0.1076 < Informer 0.1099; test: Informer 0.1032 < RFKAN 0.1052 | ⚠️ no consistente |
| RFKAN supera a WVS (Tabla 7) | val: 0.1076 < 0.1087; test: 0.1052 ≈ 0.1061; WVS tiene menor MAE | ⚠️ empate |
| RFKAN supera a TACDPG (Tabla 7) | TACDPG 0.1205 de test, peor que persistencia | ✅ sí |
| RFKAN más rápido que los avanzados (−24 %) | RFKAN 0.18× Informer, 0.09× WVS (TACDPG tarda menos, pero no aprende) | ✅ sí |
| El tiempo de los otros modelos crece exponencialmente con L (Fig. 5) | LSTM/GRU/TCN tardan 4–6 min con cualquier L; la que se encarece es la familia KAN con splines | ❌ no |
| (no lo mide) Robustez a la semilla | RFKAN estable en 4/4 semillas; LSTM queda en persistencia en 2/4 | — a favor de RFKAN |

- **Explicación de lo que no se reproduce:** el argumento del paper para usar
  Fourier es la periodicidad de la potencia solar, y el caudal a 48 h no es
  periódico. En caudal bajo, las salidas de la familia KAN oscilan hora a hora
  (RFKAN es el que más).
- **Limitación compartida por todos los modelos:** los picos de crecida (p99)
  se subestiman en 32–39 %.
