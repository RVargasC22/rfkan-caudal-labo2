# 2. La idea del paper: RFKAN

**Paper:** Rong, Lin & Xie, *Scientific Reports* (2025). Lo escribieron para
predecir cuánta energía va a producir **una planta solar** al día siguiente.

## Punto de partida: las redes KAN

Una red neuronal común (MLP) es como una cadena de **multiplicaciones por
números fijos**, con una "curva" fija en el medio (por ejemplo, "si es negativo,
vale 0"). Lo que aprende son los números.

Una **KAN** (Kolmogorov–Arnold Network) da vuelta la idea: en cada conexión, en
lugar de un número, hay **una curva que se aprende**. Es como si cada cable de la
red pudiera doblarse con la forma que haga falta. Detrás hay un teorema de
matemática: cualquier función complicada de muchas variables se puede armar
sumando funciones simples de **una sola** variable.

En la KAN original, esas curvas se dibujan con **splines**: trozos de curva
pegados, como las piezas de una pista de autos armable.

## Cambio 1: darle memoria (la "R" de RFKAN)

Una KAN normal mira cada momento por separado. El paper le agrega **nodos
recurrentes**: una memoria que se va actualizando hora a hora.

> Es como leer un libro recordando lo que pasó en los capítulos anteriores, en
> lugar de leer cada página como si fuera la primera.

La fórmula es sencilla: *memoria de hoy = un poco de la memoria de ayer + un
poco de lo que veo hoy*.

## Cambio 2: usar ondas en lugar de splines (la "F")

En vez de dibujar cada curva con trozos de pista, el paper la arma **sumando
ondas**: senos y cosenos de distintas frecuencias. Es una **serie de Fourier**.

> Es como hacer un sonido complejo mezclando notas musicales puras.

Su argumento: la energía solar es **muy repetitiva** (sube de día, baja de
noche, todos los días), y las ondas son ideales para cosas que se repiten.
Además, dicen que se entrena más rápido.

## Refinar la grilla

El entrenamiento empieza con pocas ondas (3) y va agregando más: 5, 10, 20, 50
y 100. Es como un pintor que primero hace un boceto grueso y después va
agregando detalle, **sin borrar** lo que ya pintó.

## Qué afirma el paper

- Cada cambio por separado mejora a la KAN, y juntos mejoran más (el error baja
  un 37 %).
- Lo mejor es usar 100 ondas y mirar solo 2 pasos hacia atrás.
- RFKAN le gana a todos los demás modelos: ARIMA, SVM, ELM, GRU, TCN y
  otros más avanzados.

Encontramos contradicciones en el propio paper. Una tabla dice "mirar 12
pasos" y el texto dice "2". En una tabla usan LSTM y en otra GRU. Y la métrica
que llaman "CORR" no es una correlación, sino el R².
