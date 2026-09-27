# 5. Preguntas probables y cómo responderlas

La sesión de preguntas evalúa: entender el método, explicar el código,
justificar las decisiones, interpretar los resultados y reconocer los errores y
las limitaciones.

## Sobre el método

**¿Qué es una KAN y en qué se diferencia de un MLP?**
En un MLP las conexiones son números y la no linealidad es fija. En una KAN
cada conexión es una curva que se aprende, y los nodos solo suman. Viene del
teorema de Kolmogorov–Arnold: toda función de muchas variables se arma sumando
funciones de una variable.

**¿Qué agregan los nodos recurrentes?**
Memoria: h(t) = W·h(t−1) + W·x(t). Es una RNN lineal, sin activación. Entra
junto con los datos normales a cada capa, así cada conexión ve el valor actual y
un resumen del pasado.

**¿Por qué el paper usa Fourier?**
Porque la potencia solar es periódica (día/noche) y las ondas representan bien
lo periódico con pocos parámetros; además entrenan más rápido.

**¿Qué es refinar la grilla?**
Aumentar el número de ondas (3 → 100) durante el entrenamiento. Con Fourier es
exacto: se copian las ondas viejas y las nuevas empiezan en cero, así que la
curva no cambia al refinar.

## Sobre el código

**¿Dónde está cada cosa?**
- `kan_layers.py`: las capas de ondas y de splines.
- `model.py`: RFKAN, sus 4 variantes y la salida residual.
- `train.py`: el entrenamiento con refinamiento de grilla.
- `experiments.py`: todos los experimentos.
- `predict.py`: el CSV de test.

**¿Cómo salen las 4 variantes de la ablación?**
De la misma clase, cambiando dos opciones: el tipo de curva (splines u ondas) y
si tiene memoria o no.

**¿Cómo se implementan los splines sin librerías?**
Con la fórmula recursiva de Cox–de Boor. Al refinar la grilla, los coeficientes
nuevos se calculan por mínimos cuadrados para que la curva quede igual.

## Sobre las decisiones

**¿Por qué predecir el cambio y no el valor?**
Porque la versión literal explotaba: el cambio típico es de 0.005 en escala
0–1, y las ondas nuevas metían ruido mucho mayor. Partir de "mañana igual que
ahora" y aprender la corrección lo estabilizó. Se aplicó a **todos** los
modelos, así que no le da ventaja a ninguno.

**¿Por qué escalaron la pérdida?**
Porque el error era tan chico que los gradientes quedaban por debajo del
épsilon de Adam (10⁻⁸) y el LSTM no aprendía nada. Escalar no cambia el óptimo.
Lo descubrimos, lo corregimos y repetimos todo.

**¿Por qué 72 horas de ventana?**
Porque en el barrido de 2 a 336 horas RFKAN da su mejor error con 72 y 168 horas (empatan), y 72 es mucho más barato. Los
modelos base usan la misma ventana para comparar en igualdad.

**¿Por qué 18 épocas?**
El paper da 50 pasos por grilla × 6 grillas, con el mismo presupuesto para
todos. Nosotros usamos 3 épocas por grilla × 6 = 18, también iguales para todos.

## Sobre los resultados

**¿Se reproduce el paper?**
En parte:
- ✅ la memoria mejora;
- ✅ Fourier es más rápido;
- ✅ RFKAN le gana a ARIMA, SVR, ELM y TACDPG;
- ❌ Fourier no mejora la precisión;
- ❌ la grilla óptima no es 100;
- ❌ LSTM y GRU le ganan a RFKAN.

**¿Por qué no se reproduce?**
- **Otro problema:** el río no es periódico como el sol.
- **Evaluación mucho más amplia:** 28 000 casos de 508 ríos, contra 4 días de
  una sola planta.
- **Igualdad de condiciones:** todos los modelos usan los mismos datos, el
  mismo presupuesto y la misma salida residual.

**¿Cuál es la limitación más importante?**
Las crecidas: todos subestiman los picos en un 32–39 %. La pérdida (error
cuadrático) premia acertar las horas tranquilas, que son la mayoría.

**¿Qué harían distinto?**
- Usar una pérdida que le dé más peso a los picos.
- Usar el clima futuro (`y_aux`) como ayuda durante el entrenamiento.
- Probar curvas locales (wavelets) en vez de ondas globales.

## Sobre errores y honestidad

**¿Hubo errores en el camino?**
Sí, y los documentamos:
- la divergencia al refinar la grilla;
- los gradientes bajo el épsilon de Adam, que nos hicieron repetir todo;
- ARIMA dando infinitos en ~1 % de las ventanas;
- problemas de memoria de la GPU.

Todo está en `docs/GAP_ANALISIS.md` y `docs/IMPLEMENTACION_CODIGO.md`.

**¿Qué partes del paper no pudieron reproducir exactamente?**
- **El análisis por estación:** el dataset no trae fechas, así que usamos la
  temperatura como aproximación.
- **Algunos hiperparámetros de TACDPG:** están en un material suplementario que
  no conseguimos, así que son supuestos declarados.
