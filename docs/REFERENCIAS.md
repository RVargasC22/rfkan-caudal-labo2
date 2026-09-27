# Referencias — Laboratorio 2

## 1. Paper asignado

**Recurrent Fourier-Kolmogorov Arnold Networks for photovoltaic power forecasting**

- Autores: Desheng Rong, Zhongbao Lin, Guomin Xie
- Afiliación: Faculty of Electrical and Control Engineering, Liaoning Technical
  University, Huludao, China
- Revista: *Scientific Reports* 15 (2025) 4684 — acceso abierto
- DOI: `10.1038/s41598-025-88959-5` — https://doi.org/10.1038/s41598-025-88959-5
- PDF: `docs/Rong2025_RFKAN_PV.pdf` (copia de `s41598-025-88959-5 (1).pdf`)
- Texto extraído (para búsqueda): `docs/Rong2025_RFKAN_PV.txt`
- Notas: `docs/NOTAS_PAPER.md`. Ecuaciones: `docs/ECUACIONES_PAPER.md`.
- Código de los autores: no publicado. Datos: no públicos (Liaoning Tieling
  Power Supply Company).

## 2. Trabajos citados por el paper que usamos

- **KAN:** Z. Liu et al., *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756
  (2024). Es la base de la capa spline (KAN/RKAN) y del refinamiento de grilla
  por mínimos cuadrados. Ref. 39 del paper.
- **Informer:** H. Zhou et al., *Informer: Beyond Efficient Transformer for Long
  Sequence Time-Series Forecasting*, AAAI 2021. Reimplementado en
  `lab/repo/informer.py` (modelo avanzado de la Tabla 7). El paper de RFKAN cita
  una variante: la Ref. 43 es Ma et al., *Forecasting air quality index in Yan'an
  using temporal encoded informer*, Expert Syst. Appl. 255 (2024). Como la
  codificación temporal requiere fechas que el dataset no trae, implementamos el
  Informer original.
- **WVS:** Y. Zhao, X. Peng, T. Tu, Z. Li, P. Yan, C. Li, *WOA-VMD-SCINet:
  Hybrid model for accurate prediction of ultra-short-term photovoltaic
  generation power considering seasonal variations*, Energy Reports 12:3470–3487
  (2024). https://www.sciencedirect.com/science/article/pii/S2352484724006048 —
  Ref. 44. PDF: `docs/Zhao2024_WOA-VMD-SCINet.pdf`. Reimplementado en `lab/repo/wvs.py`.
- **TACDPG:** R. Zhang, S. Bu, M. Zhou, G. Li, B. Zhan, Z. Zhang, *Deep
  reinforcement learning based interpretable photovoltaic power prediction
  framework*, Sustainable Energy Technologies and Assessments (2024).
  https://www.sciencedirect.com/science/article/abs/pii/S2213138824002261 —
  Ref. 45. PDF: `docs/Zhang2024_TACDPG.pdf` (67:103830). Reimplementado en `lab/repo/tacdpg.py`.
- Componentes: VMD (Dragomiretskiy & Zosso, IEEE TSP 2014), WOA (Mirjalili &
  Lewis, Adv. Eng. Softw. 2016), SCINet (Liu et al., NeurIPS 2022), TD3 /
  críticos gemelos (Fujimoto et al., ICML 2018), CatBoost (Prokhorenkova et al.,
  NeurIPS 2018).
- **LSTM, sesgo de olvido = 1:** R. Jozefowicz, W. Zaremba, I. Sutskever, *An
  Empirical Exploration of Recurrent Network Architectures*, ICML 2015.

## 3. Dataset

- **Rainfall-Runoff** (Drive del curso):
  https://drive.google.com/drive/folders/1crKbJBhLHQEVJGG-agOxMnKLDnBP2x4w
- Copia local: `lab/data/` (`train.h5`, `test.h5`, `test_targets.csv`,
  `metadata.json`, `leer_datos.py`).
- Según `metadata.json`: series horarias, sin normalizar ni imputar, cuencas
  anónimas (`source_identifiers_removed: true`). `y_aux` solo sirve como
  supervisión futura, no como entrada en inferencia.

## 4. Consigna

`Laboratorio 2.md` (raíz del proyecto).

- Video ≤ 18 min (Metodología 6 / Implementación 6 / Resultados 6).
- Entrega: jueves 1-oct-2026 23:59, en https://forms.gle/GDUh9wn96ny91gbN7.
- Sesión de preguntas individual: sábado 3-oct-2026.

## 5. Software

- PyTorch 2.14 (CUDA 13.0), NumPy, pandas, h5py, scikit-learn (SVR),
  statsmodels (ARIMA), matplotlib, tabulate, python-docx, python-pptx.
- Las capas KAN (spline y Fourier) están implementadas desde cero en
  `lab/repo/kan_layers.py`. **No** se usan pykan, efficient-kan ni FourierKAN.
