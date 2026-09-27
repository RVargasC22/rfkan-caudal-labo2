# Entregables — Laboratorio 2

Checklist de lo que pide la consigna (`Laboratorio 2.md`) contra lo que ya está
listo. Ver `../CLAUDE.md` para el contexto completo del proyecto.

## Estado

| Entregable | Estado | Dónde está |
|---|---|---|
| Código: carga de datos y particiones | ✅ | `lab/repo/dataset.py` |
| Código: preprocesamiento | ✅ | `lab/repo/dataset.py` (min-max con estadísticos de train) |
| Código: método del paper (propio) | ✅ | `lab/repo/kan_layers.py`, `lab/repo/model.py` |
| Código: entrenamiento y validación | ✅ | `lab/repo/train.py` |
| Código: modelo base | ✅ | `lab/repo/baselines.py` (persistencia, ARIMA, SVR, ELM, LSTM, GRU, TCN) |
| Código: métricas | ✅ | `lab/repo/metrics.py` |
| Código: ablación | ✅ | `lab/repo/experiments.py` (suites `seqlen`, `grid`) |
| Código: predicciones de test | ✅ | `lab/repo/predict.py` → `outputs/experiments/predictions_test_RFKAN_L72.csv` (etapa 5 de `run_all.sh`) |
| Markdown con resultados, instrucciones y diferencias | ✅ | `lab/repo/README.md`, `docs/RESULTADOS.md`, `docs/GAP_ANALISIS.md` |
| Informe escrito (.docx) | ✅ | `docs/Informe_Laboratorio2_RFKAN.docx` (se regenera con `docs/build_informe.py`) |
| Slides para el video | ✅ | deck web https://claude.ai/artifact/7Eh6oV3GD1UnMwT5ofg6Ss + `presentacion/RFKAN_Presentacion.pptx` + `SPEECH.md` (fuente única: `presentacion/content.py`) |
| Video explicativo (≤ 18 min) | ⬜ | grabar: Metodología 6 / Implementación 6 / Resultados 6 |
| Entrega | ⬜ | jueves 1-oct-2026 23:59, en https://forms.gle/GDUh9wn96ny91gbN7 |
| Sesión de preguntas individual | ⬜ | sábado 3-oct-2026 (horario de clase) |

## Documentos de referencia

- `NOTAS_PAPER.md`: resumen del paper, hiperparámetros (Tablas 4/5), resultados
  originales (Tablas 3, 6, 7) e inconsistencias.
- `ECUACIONES_PAPER.md`: cada ecuación explicada y dónde está en el código.
- `IMPLEMENTACION_CODIGO.md`: narrativa de la implementación (para el bloque
  "Implementación" del video).
- `GAP_ANALISIS.md`: tabla de fidelidad al paper (✅ / 🔧 / ❌) y justificación de
  las adaptaciones.
- `RESULTADOS.md`: todas las corridas con números reales (fuente de las cifras
  del informe y las slides).
- `REFERENCIAS.md`: paper, dataset, software.

## Guion sugerido del video

- **Metodología (6 min):** problema, KAN (Ec. 1–2), nodos recurrentes (Ec. 3–4),
  Fourier (Ec. 8), refinamiento de grilla, experimentos del paper.
- **Implementación (6 min):** adaptación al caudal (48 salidas, ventana, 508
  cuencas), capas propias, divergencia y salida residual, baselines, pipeline.
- **Resultados (6 min):** ablación (Tabla 3 análoga), largo de entrada (Fig. 5),
  grilla (Fig. 4), baselines, semillas, análisis de errores (picos, lluvia,
  estación) y conclusiones.
