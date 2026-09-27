# Guía de ecuaciones del paper — Rong, Lin & Xie 2025, "Recurrent Fourier-Kolmogorov Arnold Networks for photovoltaic power forecasting" (*Scientific Reports* 15:4684)

Cada ecuación con su número, página, fórmula, significado de cada variable,
explicación simple y **dónde está en nuestro código**. Sigue el orden del paper.

---

## ¿De qué trata el paper?

Pronosticar la **potencia de una planta solar para el día siguiente** a partir de
la potencia pasada y del clima (temperatura, humedad, nubes, irradiancia,
viento). Es un problema de **regresión de series de tiempo multivariadas**.

La propuesta parte de las **KAN** (Kolmogorov–Arnold Networks, Liu et al. 2024).
Una KAN se diferencia de un MLP en dónde están las no linealidades:

- **MLP:** los pesos de las aristas son números y la no linealidad (ReLU) está
  fija en los nodos.
- **KAN:** en cada arista hay una **función univariada aprendible** (un spline) y
  los nodos solo suman.

Los autores le hacen dos cambios:

1. **Nodos kernel recurrentes:** una memoria lineal tipo RNN que se agrega como
   entrada extra a cada capa, para que la red "vea" la historia de la secuencia.
2. **Serie de Fourier** en lugar de spline como función de cada arista: más
   rápida de entrenar y, según ellos, mejor para datos periódicos como el sol.

Al modelo resultante lo llaman **RFKAN**. Lo validan con:

- una **ablación** KAN / RKAN / FKAN / RFKAN (Tabla 3);
- el efecto del **tamaño de grilla** (Fig. 4) y del **largo de la entrada**
  (Fig. 5);
- una comparación contra **baselines** (ARIMA, SVM, ELM, LSTM/GRU, TCN) y
  contra **modelos avanzados** (Informer, WVS, TACDPG), por estación.

**Qué tratan de demostrar:**

- que cada cambio mejora a KAN por separado y que juntos mejoran más (RMSE
  −37 %);
- que Fourier reduce el tiempo de entrenamiento (−24 %);
- que RFKAN le gana a todos los demás en las cuatro estaciones.

---

## Ec. (1) — Teorema de Kolmogorov–Arnold (pág. 2)

```
f(x_1, …, x_n) = Σ_{q=1}^{2n+1} Φ_q ( Σ_{p=1}^{n} φ_{q,p}(x_p) )
```

- `x_p`: la p-ésima variable de entrada (n en total).
- `φ_{q,p}: [0,1] → ℝ`: función **interna** de una variable, una por cada par
  (entrada p, nodo oculto q).
- `Φ_q: ℝ → ℝ`: función **externa** de una variable.
- `2n+1`: cantidad de términos (nodos ocultos) que exige el teorema.

**En simple:** cualquier función continua de muchas variables se puede escribir
solo con sumas y funciones de **una** variable. Esto se lee como una red de dos
capas: la primera va de `n` a `2n+1` nodos (las `φ`) y la segunda de `2n+1` a 1
(las `Φ`).

**En el código:** `model.py`, `RFKAN.__init__` → `hidden = 2 * n_in + 1`
(= 25 con 12 canales). `layer1` es la capa `n → 2n+1` y `layer2` la capa
`2n+1 → salida`.

## Ec. (2) — Capa KAN (pág. 3)

```
Φ = { φ_{q,p} },   p = 1..n_in,   q = 1..n_out
```

**En simple:** una capa KAN es una **matriz de funciones**: una función
aprendible por cada arista (entrada p → salida q). La salida q es la suma sobre
p de `φ_{q,p}(x_p)`. Si la función fuese `φ(x) = w·x`, sería una capa lineal.

**En el código:** `kan_layers.py`. `FourierKANLayer` y `SplineKANLayer`
guardan los coeficientes de las `in_dim × out_dim` funciones en un tensor
(`coef` con forma `(2, out, in, g)` y `spline_w` con forma `(out, in, g+k)`,
respectivamente) y suman sobre las entradas con `einsum`.

## Ec. (3) — Nodos kernel recurrentes (pág. 4)

```
h_{l,i}(t) = W_hh · h_{l,i}(t−1) + W_hz · x_{l,i}(t)
```

- `x_{l,i}(t)`: valor del nodo i de la capa l en el instante t.
- `h_{l,i}(t)`: "memoria histórica" de ese nodo en t.
- `W_hh`: peso de la memoria anterior. `W_hz`: peso de la entrada actual.

**En simple:** es una RNN **lineal** (sin tanh): la memoria de hoy es una
mezcla de la de ayer y de la entrada de hoy. Resume todo lo que pasó antes del
instante t.

**En el código:** `model.py`, `RecurrentKernelNodes`. `W_hh` y `W_hz` son
matrices `dim × dim` sin bias, así que mezclan nodos (el paper escribe los
índices por nodo pero no dice si los pesos son escalares o matrices).
`W_hh` se inicializa ortogonal con ganancia 0.9 para que la recurrencia lineal
no explote. `h(0) = 0` y se recorre la ventana paso a paso.

## Ec. (4) — Salida de la capa con memoria (pág. 4)

```
x_{l+1,j}(t) = Σ_{i=1}^{n_l} φ_{l,j,i,t}( x_{l,i}(t), h_{l,i}(t) )
```

**En simple:** cada arista ahora recibe **dos** cosas: el valor actual del nodo
y su memoria. El nodo de salida sigue siendo una suma.

**En el código:** el paper no dice cómo una función de una variable recibe dos
entradas. Lo implementamos como **nodos en paralelo** (así lo dibuja la Fig. 1):
se concatenan `[x(t), h(t)]` y la capa KAN tiene `2n` entradas, de modo que
`φ(x, h) = φ_x(x) + φ_h(h)`.

- Capa 1: `model.py:_layer1_input` →
  `torch.cat([x, self.rec0(x)], -1)` (forma `(B, T, 24)`). `layer1` es
  `24 → 25`, aplicada en cada paso t.
- Capa 2: `_layer2_input` → nodos recurrentes `rec1` sobre la salida de la
  capa 1, y `[x1(T), h1(T)]` del **último paso** de la ventana entra a
  `layer2` (`50 → 48`).

## Ec. (5)–(6) — Forma matricial (pág. 4)

```
Φ_{l,t} = [ φ_{l,j,i,t}(·) ]  (matriz n_{l+1} × n_l),      x_{l+1,t} = Φ_{l,t} x_{l,t}
```

**En simple:** la misma capa escrita como "matriz de funciones aplicada a un
vector". El subíndice `t` indica que la capa se aplica en cada instante.

**En el código:** es el `einsum` de `FourierKANLayer.forward`
(`"big,oig->bo"`) aplicado a todos los pasos t a la vez (se aplana `B·T`).

## Ec. (7) — Composición de capas (pág. 4)

```
KAN(x, t) = (Φ_{L−1,t} ∘ … ∘ Φ_{1,t} ∘ Φ_{0,t})(x, t)
```

**En simple:** una KAN profunda es la composición de sus capas. Acá `L = 2`.

**En el código:** `RFKAN.forward` → `layer2(_layer2_input(layer1(_layer1_input(x))))`.

## Ec. (8) — Serie de Fourier como función de activación (pág. 4)

```
φ_F(x) = Σ_{i=1}^{d} Σ_{k=1}^{g} ( a_ik cos(k x_i) + b_ik sin(k x_i) )
```

- `d`: dimensión de entrada (cantidad de aristas que llegan al nodo).
- `g`: **tamaño de grilla** = cantidad de armónicos.
- `a_ik, b_ik`: coeficientes aprendibles (uno por par arista–armónico, y por
  nodo de salida).

**En simple:** en vez de dibujar la función de cada arista con un spline
(pedazos de polinomio sobre una grilla de puntos), se escribe como suma de
senos y cosenos de frecuencias 1, 2, …, g. "Grilla más fina" significa "más
armónicos", es decir, funciones con más detalle.

**En el código:** `kan_layers.py:FourierKANLayer`.

- `coef[0]` son los `a` y `coef[1]` los `b`, con forma `(out, in, g)`.
- Se agregó un `bias` por salida.
- Inicialización "suave": la amplitud cae como `1/k`, para no arrancar con
  ruido de alta frecuencia.
- Como las entradas están en `[0, 1]` (min-max) y `k ≥ 1`, las funciones no son
  periódicas dentro del rango de datos cuando g es chico; se comportan como
  una base general.

### Spline (el KAN original, para KAN/RKAN)

```
φ(x) = w_b · silu(x) + w_s · Σ_c c_c B_c(x)
```

B-splines de orden K = 3 (Tabla 4: `K 3`) sobre `g` intervalos, más una base
residual `silu`, como en Liu et al. 2024. **En el código:**
`kan_layers.py:SplineKANLayer`, con la recursión de Cox–de Boor escrita a mano.

### Refinamiento de grilla (texto de pág. 3 y Fig. 4)

"Tras entrenar en la grilla gruesa, RFKAN hereda la curva aprendida y se
entrena en una grilla más fina." Grilla `[3, 5, 10, 20, 50, 100]`.

- **Fourier:** pasar de `g` a `g'` armónicos es **exacto**. Se copian los
  `a_k, b_k` y los nuevos arrancan en 0, así que la función no cambia
  (`FourierKANLayer.extend_grid`).
- **Spline:** la grilla nueva no contiene a la vieja. Se recalcula el rango de
  los nodos con los datos y se reajustan los coeficientes por **mínimos
  cuadrados** para reproducir la curva vieja (`SplineKANLayer.extend_grid`,
  resuelto por bloques de 256 entradas).
- **Entrenamiento:** `train.py:fit` llama a `extend_grid` cada `STAGE_EPOCHS`
  épocas.

## Ec. (9) — Métricas (pág. 5)

```
MAE  = Σ |Y_n − Y'_n| / N
RMSE = sqrt( Σ |Y_n − Y'_n|² / N )
CORR = 1 − Σ (Y_n − Y'_n)² / Σ (Y_n − Ȳ)²
```

- `Y_n`: valor real. `Y'_n`: pronóstico. `Ȳ`: media de los reales. `N`: puntos
  pronosticados.

**En simple:** MAE es el error promedio en unidades de la variable. RMSE es lo
mismo pero castiga más los errores grandes. "CORR" **no es una correlación**:
es el **R²** (coeficiente de determinación). Vale 1 si el pronóstico es
perfecto, 0 si es tan bueno como predecir siempre la media, y es negativo si es
peor.

**En el código:** `metrics.py` (`mae`, `rmse`, `corr_r2`), calculadas en mm/h
sobre todas las muestras × 48 horas. Agregamos dos métricas estándar en
hidrología que el paper no usa:

- **NSE por cuenca:** es el mismo R² calculado dentro de cada cuenca; se
  reporta la mediana sobre las 508 cuencas.
- **KGE** (Kling–Gupta).

---

## Resumen: ecuación → código

| Ec. | Qué es | Dónde |
|---|---|---|
| 1 | Kolmogorov–Arnold, `n → 2n+1 → out` | `model.py:RFKAN.__init__` |
| 2, 5, 6 | Capa KAN (matriz de funciones) | `kan_layers.py:FourierKANLayer`, `SplineKANLayer` |
| 3 | Nodos kernel recurrentes | `model.py:RecurrentKernelNodes` |
| 4 | φ(x, h) con nodos en paralelo | `model.py:_layer1_input`, `_layer2_input` |
| 7 | Composición de capas | `model.py:RFKAN.forward` |
| 8 | Serie de Fourier | `kan_layers.py:FourierKANLayer.forward` |
| — | Refinamiento de grilla | `kan_layers.py:*.extend_grid`, `train.py:fit` |
| 9 | MAE, RMSE, CORR (= R²) | `metrics.py` |
