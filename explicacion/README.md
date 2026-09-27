# El laboratorio explicado en palabras simples

Esta carpeta cuenta todo el trabajo sin fórmulas ni jerga, para repasar antes
del video y de la sesión de preguntas. La versión técnica completa está en
`docs/`; los números exactos, en `docs/RESULTADOS.md`.

| Archivo | Qué explica |
|---|---|
| [01_el_problema.md](01_el_problema.md) | Qué hay que predecir y por qué es difícil |
| [02_la_idea_del_paper.md](02_la_idea_del_paper.md) | Qué propone el paper (RFKAN), con analogías |
| [03_como_lo_hicimos.md](03_como_lo_hicimos.md) | Cómo lo adaptamos y qué problemas tuvimos |
| [04_que_encontramos.md](04_que_encontramos.md) | Resultados y qué significan |
| [05_preguntas_para_la_defensa.md](05_preguntas_para_la_defensa.md) | Preguntas probables y cómo responderlas |

## El resumen en cinco frases

1. Hay que adivinar cuánta agua va a llevar un río en las próximas 48 horas,
   mirando las últimas dos semanas de lluvia, temperatura y caudal.
2. El paper propone una red neuronal nueva, RFKAN, que fue pensada para paneles
   solares. Sus dos ideas son darle memoria y usar ondas (senos y cosenos) como
   piezas de construcción.
3. La construimos desde cero, la adaptamos a los ríos y la comparamos contra
   más de diez modelos distintos en 508 cuencas.
4. **La memoria funciona; las ondas no.** Los ríos no se repiten como el sol,
   que sale todos los días a la misma hora.
5. Ningún modelo logra anticipar bien las crecidas grandes: es el problema
   abierto más importante.
