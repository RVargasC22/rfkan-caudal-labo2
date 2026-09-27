# 4. Qué encontramos

Errores en test, en mm/h (más bajo es mejor). "Mañana igual que ahora" da
**0.119**.

## 1. La memoria sí sirve

| Versión | Error | Qué tiene |
|---|---|---|
| KAN | 0.110 | nada |
| KAN + ondas (FKAN) | 0.117 | solo ondas |
| KAN + memoria (RKAN) | **0.103** | solo memoria |
| KAN + memoria + ondas (RFKAN, el del paper) | 0.105 | las dos cosas |

Agregar memoria baja el error un 6–10 %. **Coincide con el paper.**

## 2. Las ondas no sirven para los ríos

Cambiar splines por ondas **empeora un poco** (+2 %). El paper decía que
mejoraba un 22 %.

**Por qué:** las ondas son buenas para cosas que se repiten, y el sol se repite
todos los días. El río no: pasa días tranquilo y de golpe crece. En los
gráficos se ve que, con caudal bajo, las versiones con ondas hacen un
**zigzag** que no existe en la realidad.

Lo que sí se confirma: con ondas se entrena **más rápido**, entre 1.7 y 6 veces.

## 3. Refinar hasta 100 ondas no ayuda

El mejor momento de casi todos los modelos está en 10 o 20 ondas. Ir hasta 100
no mejora nada. Pero **empezar directo con 100**, o refinar sin bajar la
velocidad de aprendizaje, **arruina** el modelo: queda peor que "mañana igual
que ahora". El refinamiento gradual sirve como **protección**, no como mejora.

## 4. Cuánto pasado mirar

- **Los modelos con memoria casi no cambian** si miran 2 horas o 14 días.
- **Los que no tienen memoria empeoran** cuanto más pasado les damos: se
  marean con tantos datos.
- **LSTM y GRU sí aprovechan más historia**, hasta 1–3 días.

## 5. Contra otros modelos: RFKAN no gana

| Modelo | Error |
|---|---|
| **LSTM** | **0.101** |
| GRU | 0.102 |
| Informer | 0.103 |
| TCN | 0.103 |
| RKAN | 0.103 |
| RFKAN (el del paper) | 0.105 |
| WVS | 0.106 |
| ELM | 0.142 |
| TACDPG | 0.121 (peor que "mañana igual") |

El paper decía que RFKAN le ganaba a todos por mucho. **En nuestros datos no:**
un LSTM común le gana. Pero hay matices:

- **RFKAN es muy estable.** Con 4 semillas distintas siempre da ~0.106. El LSTM,
  en 2 de 4 intentos, **no aprendió nada** y se quedó en "mañana igual que ahora".
- **RFKAN es rápido:** 5 veces más que Informer y 11 veces más que WVS.
- **TACDPG** (aprendizaje por refuerzo) no logró aprender. Premiar o castigar una
  predicción de 48 horas es una señal mucho más débil que decirle directamente
  el error.

## 6. Dónde se equivocan todos

- **Las crecidas.** En el 1 % de momentos con más agua, **todos** los modelos se
  quedan cortos por un 32–39 %. Ninguno anticipa bien un pico.
- **Después de lluvias fuertes.** Ahí está casi toda la mejora sobre "mañana
  igual que ahora", pero también los errores más grandes.
- **En días fríos** (invierno), el error es mayor.
- **Cuanto más lejos, peor.** La hora 1 es fácil; la hora 48 depende de lluvia
  que todavía no cayó.

## En una frase

> La idea de darle memoria a una KAN funciona también en ríos; la idea de usar
> ondas no, porque el río no es periódico como el sol. Y el problema de fondo,
> las crecidas, sigue sin resolver para todos los modelos.
