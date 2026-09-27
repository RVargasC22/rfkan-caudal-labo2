# Guion del video (≤ 18 min: Metodología 6 / Implementación 6 / Resultados 6)

Generado por build_pptx.py desde las notas de content.py.

## Metodología

**1. RFKAN para pronóstico de caudal**

Hola. En este video presentamos nuestra implementación del paper Recurrent Fourier-Kolmogorov Arnold Networks, de Rong, Lin y Xie, publicado en Scientific Reports en 2025. El paper propone un modelo para pronosticar la potencia de una planta solar. Nosotros lo implementamos desde cero y lo adaptamos al problema del laboratorio: pronosticar el caudal de ríos. Vamos a ver la metodología, la implementación y los resultados.

**2. El problema: 48 horas de caudal**

Empecemos por el problema. Cada muestra del dataset es una ventana de 336 horas, es decir, 14 días, de una cuenca. En cada hora hay 12 variables: 11 meteorológicas, como la precipitación, la temperatura, la radiación o el viento, y el caudal. A partir de eso hay que predecir el caudal de las siguientes 48 horas, en milímetros por hora, que es el caudal dividido por el área de la cuenca. Hay 508 cuencas anónimas, y las mismas cuencas aparecen en train, validación y test. Una propiedad importante: el caudal cambia poco en 48 horas la mayor parte del tiempo. Repetir el último valor, lo que llamamos persistencia, ya es un buen modelo base. Lo difícil son los eventos de lluvia.

**3. El paper: RFKAN para potencia solar**

El paper resuelve un problema parecido en otro dominio: pronosticar para el día siguiente la potencia de una planta fotovoltaica de 45 kilowatts en China, con datos cada 5 minutos durante un año. Su propuesta, RFKAN, parte de las redes de Kolmogorov-Arnold, o KAN, y les hace dos cambios. El primero son los nodos kernel recurrentes, una memoria tipo RNN para capturar la dependencia temporal. El segundo es reemplazar los splines de la KAN por series de Fourier, con el argumento de que la potencia solar es muy periódica y Fourier la representa con pocos parámetros, y además entrena más rápido. El entrenamiento usa refinamiento de grilla: empieza con una grilla gruesa y la va refinando en seis etapas, de 3 a 100.

**4. Kolmogorov–Arnold: funciones en las aristas**

¿Qué es una KAN? Se basa en el teorema de Kolmogorov-Arnold, que dice que cualquier función continua de muchas variables se puede escribir usando solo sumas y funciones de una variable. La fórmula tiene dos niveles: funciones internas phi, una por cada par de entrada y término, y funciones externas Phi. Una KAN convierte eso en una red de dos capas. La diferencia con un perceptrón multicapa es dónde está la no linealidad. En un MLP las aristas tienen pesos y la activación es fija en los nodos. En una KAN, cada arista tiene su propia función aprendible de una variable, y los nodos solo suman. En nuestro caso, con 12 canales de entrada, la red queda de 12 a 25 nodos y de 25 a 48 salidas, una por hora.

**5. Nodos kernel recurrentes**

El primer aporte del paper son los nodos kernel recurrentes, ecuación 3. La memoria h en el tiempo t es una combinación lineal de la memoria en t menos 1 y de la entrada en t. Es una RNN sin función de activación. Esa memoria resume toda la historia anterior. La ecuación 4 dice que cada función de la capa recibe ahora dos cosas: el valor actual del nodo y su memoria. El paper no explica cómo una función de una variable recibe dos entradas. Nosotros lo implementamos como muestra la figura 1: los nodos recurrentes entran en paralelo con los nodos normales, así que la capa suma una función del valor y otra de la memoria.

**6. Fourier en lugar de splines**

El segundo aporte es la ecuación 8. En lugar de un spline, cada función de arista es una serie de Fourier: una suma de cosenos y senos de frecuencias 1, 2, hasta g, con coeficientes aprendibles a y b. Aquí g es el tamaño de la grilla, que pasa a ser el número de armónicos. Refinar la grilla significa agregar armónicos: los coeficientes existentes se copian y los nuevos arrancan en cero, así que la función aprendida se hereda exactamente. Retengan el argumento del paper para usar Fourier: la periodicidad de la potencia solar. Vamos a volver a él en los resultados, porque el caudal no tiene ese ciclo.

**7. Lo que reporta el paper**

Esta es la ablación del paper, la tabla 3. Comparan cuatro modelos: KAN, la base; RKAN, con recurrencia y splines; FKAN, con Fourier y sin recurrencia; y RFKAN, con las dos cosas. Cada cambio mejora a KAN y juntos mejoran más: el RMSE baja un 37 por ciento. Reportan además grilla óptima 100, largo óptimo 2, y que RFKAN supera a todos los baselines. Encontramos algunas inconsistencias. La tabla de hiperparámetros dice largo de entrada 12, pero el texto dice que el óptimo es 2. La tabla 5 da los parámetros de un LSTM, pero la tabla 6 reporta un GRU, así que corrimos ambos. Y la métrica que llaman CORR es en realidad el coeficiente de determinación, R cuadrado.

## Implementación

**8. Del sol al río: qué se adaptó**

Pasamos a la implementación. Esta tabla resume el cambio de dominio. El objetivo pasa de potencia a caudal, y de una planta a 508 cuencas. La salida del paper es un único nodo; nosotros necesitamos 48 horas, así que la capa 2 tiene 48 nodos de salida. El largo de entrada del paper es 2; como la tabla y el texto no coinciden, lo tratamos como hiperparámetro y barrimos de 2 a 336 horas, igual que la figura 5 del paper. El modelo principal usa 72 horas. La evaluación también cambia mucho: el paper evalúa un día por estación, unos 288 puntos; nosotros evaluamos casi 28 mil ventanas de 48 horas. La normalización es la del paper, min-max con estadísticos de entrenamiento.

**9. La arquitectura implementada**

Esta es la arquitectura que implementamos. La entrada son L pasos con 12 canales. Se le agregan los nodos recurrentes, y juntos entran a la capa 1, una capa KAN de Fourier de 24 a 25 nodos que se aplica en cada paso de tiempo. Sobre esa salida se calculan otra vez los nodos recurrentes, y en el último paso los dos entran a la capa 2, de 50 a 48. La salida se suma al último caudal observado, algo que explicamos en el próximo slide. Todo está escrito desde cero, como pide la consigna: la capa Fourier, la capa con B-splines usando la recursión de Cox-de Boor y la extensión de grilla por mínimos cuadrados.

**10. Problema 1: divergencia al refinar la grilla**

El primer problema serio fue que la versión literal divergía. Al refinar la grilla a 50 o 100 armónicos, el error de validación subía de 0.27 a 3.6 milímetros por hora. Lo diagnosticamos así: en escala normalizada, el cambio típico del caudal en 48 horas tiene un desvío de apenas 0.0046. Y Adam mueve cada parámetro nuevo del orden de la tasa de aprendizaje por paso, sin importar el tamaño del gradiente. Con cien armónicos por arista, eso es ruido enorme frente a 0.0046. Lo resolvimos con dos cambios. Primero, una salida residual: el modelo predice la corrección sobre el último caudal, escalada por ese desvío. Segundo, bajamos la tasa de aprendizaje en cada etapa, proporcional a 3 sobre g. La salida residual se aplica igual a todos los modelos neuronales, así que la comparación sigue siendo justa.

**11. Problema 2: gradientes bajo el ε de Adam**

El segundo problema lo descubrimos al ver que el LSTM daba exactamente el mismo error que persistencia, hasta el cuarto decimal. Medimos los gradientes: la pérdida en escala normalizada vale del orden de 2.5 por 10 a la menos 5, y el 99 por ciento de los gradientes del LSTM, el GRU y la TCN quedaban por debajo de 10 a la menos 8, que es el épsilon de Adam. Con eso, Adam prácticamente no mueve los pesos. La solución fue dividir la pérdida por sigma al cuadrado: el óptimo es el mismo, pero los gradientes quedan en una escala útil. Repetimos todos los experimentos con esta corrección. También vimos que el LSTM con 336 pasos no aprende en el presupuesto de 18 épocas, así que los modelos base neuronales usan la misma ventana de 72 horas que RFKAN.

**12. Experimentos: mismo split, semilla y presupuesto**

Estos son los experimentos. La suite seqlen es la ablación de la tabla 3 repetida para seis largos de ventana, lo que además reproduce la figura 5. La suite grid es la figura 4: comparamos el refinamiento estándar con una grilla que llega a 200, grillas fijas en 3 y en 100, y un refinamiento sin reducir la tasa de aprendizaje. La suite baselines tiene los modelos de la tabla 5, más persistencia, que es el modelo base natural en hidrología. Y la suite seeds repite RFKAN y el LSTM con tres semillas más, algo que el paper no hace. Todos con la misma partición, semilla y presupuesto de 18 épocas. Y la suite advanced entrena los tres modelos avanzados de la tabla 7, implementados por nosotros.

## Resultados

**13. Ablación: la recurrencia decide (L = 72)**

Pasamos a los resultados. Esta es nuestra versión de la tabla 3, con ventana de 72 horas, evaluada sobre las casi 28 mil ventanas de test. La fila de abajo es persistencia: 0.1188 de RMSE. RFKAN obtiene 0.1052, una mejora de 11 por ciento. Ahora, ¿qué componente aporta? Agregar recurrencia a la versión Fourier mejora el RMSE un 10 por ciento; agregarla a la versión con splines, un 6. Eso coincide con el paper. Pero cambiar splines por Fourier, de RKAN a RFKAN, empeora un 1.9 por ciento. El paper reporta una mejora del 22 por ciento. Lo que sí se reproduce es la velocidad: RFKAN entrena en 0.59 veces el tiempo de RKAN. El mejor modelo de la familia es RKAN, con splines y recurrencia.

**14. Largo de ventana: la recurrencia lo hace irrelevante**

Esta es nuestra versión de la figura 5: el error de validación para ventanas de 2 a 336 horas. Las variantes sin recurrencia, que reciben la ventana aplanada, empeoran a medida que la ventana crece: con 168 horas, FKAN ya queda igual que persistencia. RKAN y RFKAN, en cambio, son casi planos: varían menos de tres milésimas entre 2 y 336 horas. Los modelos base, en gris, se comportan distinto: con 2 horas todos empatan, pero el LSTM y el GRU mejoran hasta 24 o 72 horas. La memoria lineal de los nodos recurrentes no aprovecha la historia larga, y las compuertas sí. El paper encuentra que el mejor largo es 2; en nuestros datos eso solo vale para las variantes sin recurrencia.

**15. Grilla: refinar no mejora, pero protege**

Esta es la ablación de la grilla, análoga a la figura 4. El paper dice que la grilla óptima es 100. En nuestros datos no: el mejor punto de RFKAN está en grilla 20, extender el refinamiento hasta 200 da exactamente el mismo resultado, y una grilla fija en 3 queda apenas un 2 por ciento peor. En los 20 brazos del barrido de ventana, ningún modelo competitivo mejora al llegar a 100. Sin embargo, la forma de llegar a una grilla fina sí importa. Si se arranca directamente con 100 armónicos, o si se refina sin bajar la tasa de aprendizaje, el modelo termina peor que persistencia. El refinamiento progresivo con tasa decreciente funciona como un esquema de regularización: no mejora la precisión, pero permite usar grillas finas sin romper el modelo.

**16. Contra los modelos base: ganan LSTM y GRU**

Esta es la comparación con los modelos base de la tabla 5. Aquí el resultado es el contrario al del paper. Con el mismo presupuesto, la misma ventana y la misma salida residual, el LSTM obtiene 0.1012 y el GRU 0.1015, mejor que toda la familia KAN, y la TCN empata con RKAN. El paper reportaba que RFKAN mejoraba el RMSE en más del 58 por ciento frente a GRU y TCN. Los modelos clásicos sí quedan por debajo de RFKAN. ARIMA es interesante: en la cuenca típica es el mejor, con el NSE mediano más alto, pero en algunas ventanas extrapola una tendencia y explota, así que su RMSE global es enorme. El SVR con el C del paper está tan regularizado que queda en persistencia, y el ELM sobreajusta.

**17. Modelos avanzados (análogo a la Tabla 7)**

El paper también compara contra tres modelos avanzados, y los implementamos los tres desde cero. Informer, un Transformer con atención ProbSparse. WVS, que descompone el caudal con VMD, optimiza esa descomposición con el algoritmo de ballenas y predice cada modo con un SCINet. Y TACDPG, un agente de aprendizaje por refuerzo con críticos gemelos y un clasificador CatBoost para outliers. En validación RFKAN supera a los tres, como dice el paper. En test, Informer lo supera y WVS empata, con menor error absoluto medio que RFKAN. TACDPG no llega ni a persistencia: el aprendizaje por refuerzo no logra aprender una corrección útil de 48 horas. Lo que sí se reproduce siempre es la velocidad: RFKAN es cinco veces más rápido que Informer y once veces más rápido que WVS.

**18. Robustez: el LSTM falla en 2 de 4 semillas**

Repetimos RFKAN y el LSTM con tres semillas más. RFKAN es muy estable: las cuatro semillas quedan dentro de una milésima y media. El LSTM, en cambio, no sale de persistencia en 2 de sus 4 semillas: su error queda clavado durante las 18 épocas. Cuando entrena, es el mejor modelo; en promedio sobre semillas, RFKAN lo supera. Este es un punto a favor de RFKAN que el paper no mide. A la derecha, el error por hora de horizonte: crece con el horizonte, porque las primeras horas son casi persistencia y las últimas dependen de lluvia futura que el modelo no ve.

**19. La limitación: los picos de crecida**

El análisis de errores muestra dónde está la ganancia y dónde está la limitación. Casi toda la mejora sobre persistencia ocurre con caudal alto y después de lluvias fuertes, donde RFKAN reduce el error un 25 por ciento y el LSTM un 30. Con caudal bajo, nadie le gana a persistencia, y el error es mayor en los meses fríos. La limitación principal son los picos: en el 1 por ciento de caudales más altos, todos los modelos entrenados subestiman el pico entre un 32 y un 39 por ciento. En estos tres ejemplos se ve: ningún modelo anticipa la crecida. Y en el panel de caudal bajo se ve otra cosa: la familia KAN oscila hora a hora, RFKAN más que ninguno, mientras que el LSTM y la TCN dan curvas suaves.

**20. Conclusiones**

Para cerrar. Implementamos RFKAN desde cero y lo adaptamos al caudal: mejora a persistencia un 11 por ciento y a los modelos clásicos. De sus dos contribuciones, la recurrencia se sostiene, igual que en el paper. Fourier no: entrena más rápido, pero es menos preciso que los splines, porque su justificación era la periodicidad de la señal solar y el caudal no la tiene. Refinar la grilla hasta 100 no ayuda, aunque el esquema progresivo es necesario. Y en igualdad de condiciones, el LSTM y el GRU superan a RFKAN, aunque el LSTM falla en la mitad de las semillas. Como trabajo futuro: pérdidas que den más peso a los picos, usar la meteorología futura como supervisión auxiliar y bases de funciones locales en lugar de Fourier. Gracias.
