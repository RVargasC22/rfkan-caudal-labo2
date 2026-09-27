# 1. El problema: adivinar el caudal de un río

## Qué es el caudal

El **caudal** es cuánta agua pasa por un punto del río en un momento dado.
Aquí se mide en **milímetros por hora (mm/h)**: el agua que pasa, dividida por
el área de la cuenca. Así se pueden comparar un río grande y uno chico. Una
**cuenca** es toda la zona de terreno cuya lluvia termina en ese río.

## Qué nos dan

Para cada ejemplo tenemos **14 días (336 horas) de historia** de una cuenca, con
12 datos por hora:

- El **caudal**, que es lo que queremos predecir.
- Once datos del clima: lluvia, temperatura, humedad, presión, viento (dos
  direcciones), radiación del sol (dos tipos), evaporación y dos variables más
  de la atmósfera.

Con eso hay que predecir el caudal de **las 48 horas siguientes**, hora por hora.

Tenemos **508 cuencas** y unos 300 000 ejemplos:

| Conjunto | Ejemplos | Para qué sirve |
|---|---|---|
| Entrenamiento | 254 000 | El modelo aprende de ellos |
| Validación | 18 000 | Elegimos cuál versión del modelo es mejor |
| Test | 28 000 | Medimos el resultado final, sin haberlos usado antes |

## Por qué es difícil

- **Casi siempre el río cambia poco.** Si hoy lleva 0.2 mm/h, mañana
  probablemente lleve algo parecido. Por eso el modelo más tonto posible,
  **"mañana igual que ahora"** (le decimos *persistencia*), ya acierta bastante.
  Cualquier modelo inteligente tiene que ganarle a eso.
- **Pero a veces hay crecidas.** Después de una lluvia fuerte, el caudal puede
  multiplicarse por 10 o por 50 en pocas horas. Son raras, pero son las que
  importan: inundaciones, represas, alertas.
- **La lluvia futura no se ve.** El modelo solo ve el pasado. Si mañana llueve
  fuerte y hoy no hay señales, nadie puede adivinarlo.

## Cómo medimos si un modelo es bueno

- **RMSE**: el error típico en mm/h. Castiga mucho los errores grandes.
  **Más bajo es mejor.**
- **MAE**: el error promedio en mm/h. **Más bajo es mejor.**
- **CORR** (en realidad R²): qué tan bien sigue el modelo las subidas y bajadas.
  1 es perfecto; 0 es tan malo como predecir siempre el promedio. **Más alto es
  mejor.**
- **NSE por cuenca**: el mismo R², pero calculado en cada río por separado.
  Reportamos el valor del río "del medio" (la mediana).

Como referencia, "mañana igual que ahora" tiene **RMSE 0.119 mm/h** en test.
