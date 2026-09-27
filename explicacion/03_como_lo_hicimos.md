# 3. Cómo lo hicimos

## Todo desde cero

La consigna prohíbe usar una implementación ya hecha del método. Por eso
programamos nosotros:

- las capas KAN, tanto la de ondas (Fourier) como la de splines;
- la memoria recurrente;
- el refinamiento de la grilla.

También programamos desde cero los tres "modelos avanzados" del paper:
Informer, WVS y TACDPG.

## Del sol al río: qué tuvimos que cambiar

| En el paper (sol) | En el laboratorio (río) | Qué hicimos |
|---|---|---|
| Predice 1 valor | Hay que predecir 48 horas | La última capa da 48 salidas, una por hora |
| Una sola planta | 508 cuencas distintas | Un solo modelo para todas |
| Mira 2 pasos atrás | Tenemos 336 horas | Probamos 2, 12, 24, 72, 168 y 336 horas |
| Datos con huecos | Datos completos | No hizo falta rellenar nada |

Los datos se escalan entre 0 y 1, como en el paper, usando **solo** los números
de entrenamiento, para no "espiar" el test.

## Problema 1: la red se volvía loca al agregar ondas

La primera versión, hecha tal cual el paper, **explotaba**: al pasar de 20 a
100 ondas, el error se multiplicaba por 13.

**Por qué:** en escala 0–1, el caudal cambia muy poquito en 48 horas (en
promedio, 0.005). Cada onda nueva, al empezar a aprender, se movía mucho más que
eso y metía ruido.

**Qué hicimos:**

- **Predecir el cambio, no el valor.** El modelo parte de "mañana igual que
  ahora" y aprende solo **la corrección**. Es como un navegador que parte de "seguí
  derecho" y solo te avisa cuándo doblar.
- **Aprender más despacio a medida que hay más ondas.** Así las ondas nuevas
  no desarman lo que ya estaba.

Esto lo aplicamos **igual a todos los modelos**, para que la comparación sea
justa.

## Problema 2: algunos modelos no aprendían nada

Después vimos que el LSTM daba **exactamente** el mismo error que "mañana igual
que ahora". Estaba quieto.

**Por qué:** los errores eran tan chiquitos que las señales de aprendizaje
quedaban por debajo del mínimo que el optimizador (Adam) puede notar. Es como
susurrarle a alguien que usa auriculares con ruido.

**Qué hicimos:** multiplicamos el error por una escala fija. Eso no cambia qué
es "lo mejor", pero hace que la señal se escuche. **Repetimos todos los
experimentos** con esta corrección.

## Problema 3: memoria de la tarjeta gráfica

Algunas combinaciones, como ventanas muy largas con 100 o 500 ondas, no entraban
en los 16 GB de la GPU. Las resolvimos procesando por partes. En un caso (RKAN
con 336 horas) usamos el mejor punto guardado antes de quedarnos sin memoria, y
lo dejamos anotado.

## Qué experimentos corrimos (56 en total)

1. **Ablación:** las 4 versiones (KAN, con memoria, con ondas, con las dos) para
   6 largos de ventana.
2. **Grilla:** refinar hasta 100 o hasta 200, grilla fija chica, grilla fija
   grande, refinar sin bajar la velocidad de aprendizaje.
3. **Modelos base:** "mañana igual que ahora", ARIMA, SVR, ELM, LSTM, GRU y TCN.
4. **Modelos avanzados:** Informer, WVS y TACDPG.
5. **Semillas:** repetimos RFKAN y LSTM 4 veces cada uno, para ver si el
   resultado depende de la suerte.

Todos con los mismos datos, la misma semilla y el mismo presupuesto de
entrenamiento (18 pasadas por los datos).
