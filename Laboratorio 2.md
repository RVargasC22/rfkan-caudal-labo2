Laboratorio 2 – Pronóstico de caudal

**Fecha de entrega del video y código:** Jueves 1 de octubre \- 11:59 PM  
**Sesión de preguntas y revisión:** Sábado 3 de octubre (horario de clases)

1. **Descripción del Problema**

En este laboratorio deberán implementar un modelo para pronosticar el caudal de ríos utilizando observaciones históricas de caudal y variables meteorológicas.  
Cada muestra contiene 336 horas de mediciones (14 días) de una cuenca. A partir de esta información, el modelo deberá predecir el caudal en las siguientes 48 horas.  
El problema corresponde a una tarea de pronóstico supervisada con entradas multivariadas y múltiples pasos futuros. Se utilizan varias variables como entrada, pero la variable principal que se debe pronosticar es el caudal.  
Cada grupo deberá implementar y adaptar al dataset el método propuesto por el paper que seleccionó. Cada grupo trabajará con el paper seleccionado: 

* [Google Sheet para la sección 1](https://docs.google.com/spreadsheets/d/1KS29XFxphp22PDxkq9d_34ixDOaa-39bz-sN5DRiyBg/edit?usp=sharing)  
* [Google Sheet para la sección 2](https://docs.google.com/spreadsheets/d/1Q4_gEC_DOcCt3iHdZGILs6zsjnkXd1680yeubBAyh6E/edit?usp=sharing)

2. **Dataset**

Fuente: [Rainfall-Runoff](https://drive.google.com/drive/folders/1crKbJBhLHQEVJGG-agOxMnKLDnBP2x4w?usp=drive_link)  
El dataset contiene series temporales horarias de distintas cuencas. Cada muestra corresponde a una ventana de una sola cuenca e incluye 12 variables históricas:

* Caudal  
* Precipitación total  
* Temperatura  
* Presión  
* Humedad específica  
* Radiación de onda corta  
* Radiación de onda larga  
* Componentes del viento en dos direcciones  
* Evaporación potencial  
* Fracción convectiva  
* Energía potencial

La variable objetivo corresponde al caudal expresado en mm/h (caudal normalizado por el área de la cuenca). Las predicciones entregadas deberán utilizar esta misma unidad.  
Los datos están organizados de la siguiente manera:

* **train.h5**: datos de entrenamiento y validación.  
  * **x**: información histórica, con dimensiones \[N, 336, 12\].  
  * **y**: caudal de las siguientes 48 horas, con dimensiones \[N, 48\].  
  * **y\_aux**: valores futuros de las 11 variables meteorológicas, con dimensiones \[N, 48, 11\]. Su uso como objetivos auxiliares de entrenamiento es opcional.  
  * **basin\_id**: identificador anónimo de la cuenca correspondiente a cada muestra.  
  * **split**: partición asignada a cada muestra. El valor 0 corresponde a entrenamiento y el valor 1 a validación.  
* **test.h5**: datos para generar las predicciones finales.  
  * Contiene **X** y **basin\_id**.  
  * No contiene los valores futuros de caudal ni de las variables meteorológicas.  
* **metadata.json**: descripción de las variables, su orden y las dimensiones de los datos.

3. **Tareas**

Cada grupo deberá realizar las siguiente tareas:

* Estudiar el paper elegido e identificar su propuesta central y sus componentes principales.  
* Implementar el método propuesto en el paper.  
* Adaptar la arquitectura, las entradas, la salida y la función de pérdida.  
* Entrenar el modelo utilizando los datos de entrenamiento.  
* Realizar estudios de ablación sobre un componente importante de la propuesta.  
* Generar predicciones para todas las muestras del conjunto de test.  
* Analizar los resultados, los principales errores y las limitaciones del modelo.

La adaptación dependerá del enfoque del paper. Por ejemplo:

* Si el método predice todas las variables de una serie, podrán adaptar su salida para predecir únicamente el caudal o utilizar las variables meteorológicas como objetivos auxiliares.  
* Si el método fue diseñado para otro horizonte de predicción, deberán adaptarlo para producir los 48 valores horarios requeridos.  
* Si utiliza descomposiciones, representaciones temporales o mecanismos de atención, deberán explicar cómo se aplican a las variables del dataset.  
* Si algún componente requiere información que no está disponible, podrán reemplazarlo o simplificarlo, siempre que justifiquen la decisión y conserven la propuesta central del método.

Se pueden utilizar PyTorch y bibliotecas comunes para el procesamiento de audio. Sin embargo, **la implementación de la propuesta central del paper deberá ser realizada por el grupo**. No está permitido utilizar una implementación existente del mismo método.

4. **Entrega (Jueves 3\)**

Deberán subir sus soluciones al [formulario de entrega](https://forms.gle/GDUh9wn96ny91gbN7). El envío debe incluir:

1. **Código fuente:** Puede ser en un Jupyter Notebook o scripts python. Debe incluir:  
   * Carga de los datos y selección de las particiones.  
   * Preprocesamiento y transformaciones utilizadas.  
   * Implementación del método del paper.  
   * Entrenamiento y validación.  
   * Implementación del modelo base.  
   * Cálculo de las métricas.  
   * Experimento de ablación.  
   * Generación de predicciones para el conjunto de test.

Los gráficos y resultados se pueden presentar en las celdas del notebook, o pueden enviar un markdown (en caso de enviar su solución como archivos .py). También deberán incluir instrucciones breves para ejecutar el código, en caso de presentar scripts, y señalar cualquier diferencia importante respecto a la propuesta original

2. **Video explicativo (máximo 18 minutos)**  
   * **Metodología** (6 min): explicar el paper y de los componentes implementados.  
   * **Implementación** (6 min): adaptación al dataset, decisiones tomadas y detalles del entrenamiento.  
   * **Resultados** (6 min): comparación con el modelo base, ablación, análisis de errores y limitaciones.

Esta evaluación será **grupal**.

5. **Sesión de Preguntas (Sábado 5\)**

El sábado, durante el horario de clases, se realizará una sesión de preguntas. Las preguntas podrán abordar cualquier parte del paper, el código, los experimentos o los resultados presentados.  
Se evaluará:

* La comprensión del método implementado.  
* La capacidad para explicar el funcionamiento del código.  
* La justificación de las decisiones tomadas.  
* La interpretación de los resultados.  
* La identificación de errores y limitaciones.

Las preguntas están enfocadas en lo que ustedes hicieron y en verificar que realmente comprendan su solución.  
Esta evaluación será **individual**.

