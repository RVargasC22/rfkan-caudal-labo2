"""
Genera docs/Informe_Laboratorio2_RFKAN.docx a partir de los resultados en disco:
  lab/repo/outputs/experiments/experiments_results.csv   (una fila por brazo)
  lab/repo/outputs/analysis/*.png, *.md                  (analyze_results.py)
Correr de nuevo cada vez que cambian los resultados:
  lab/.venv/bin/python docs/build_informe.py
"""
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "lab/repo/outputs/experiments"
ANA = ROOT / "lab/repo/outputs/analysis"
OUT = ROOT / "docs/Informe_Laboratorio2_RFKAN.docx"
L = 72
FAM = ["KAN", "RKAN", "FKAN", "RFKAN"]
DESC = {"KAN": "splines, sin recurrencia", "RKAN": "splines + recurrencia",
        "FKAN": "Fourier, sin recurrencia", "RFKAN": "Fourier + recurrencia (propuesta)"}

res = pd.read_csv(EXP / "experiments_results.csv").drop_duplicates("arm", keep="last").set_index("arm")


def r(arm, col, default=float("nan")):
    return res.loc[arm, col] if arm in res.index and col in res.columns else default


def f4(x):
    return "—" if pd.isna(x) else f"{x:.4f}"


def pct(a, b):
    """Cambio relativo de a respecto de b, en %."""
    return 100 * (a - b) / b


doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2)
    s.top_margin = s.bottom_margin = Cm(2.0)


def H(text, level=1):
    doc.add_heading(text, level)


def P(text="", bold_prefix=None, italic=False, size=None):
    p = doc.add_paragraph()
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    run = p.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return p


def B(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)


def EQ(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name, run.font.size = "Cambria Math", Pt(11)


def T(header, rows, widths=None, caption=None, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = str(h)
    for k, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            if k in bold_rows:
                for p in cells[i].paragraphs:
                    for run in p.runs:
                        run.bold = True
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
    if widths:
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Cm(w)
    if caption:
        c = doc.add_paragraph()
        run = c.add_run(caption)
        run.italic, run.font.size = True, Pt(9)
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return t


def FIG(name, caption, width=16):
    path = ANA / name
    if not path.exists():
        P(f"[Figura pendiente: {name}]", italic=True)
        return
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = c.add_run(caption)
    run.italic, run.font.size = True, Pt(9)


def md_table(name):
    """Lee una tabla markdown de outputs/analysis -> (header, rows)."""
    path = ANA / name
    if not path.exists():
        return None
    lines = [l for l in path.read_text().splitlines() if l.startswith("|")]
    split = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    return split[0], [row for row in split[2:]]


# ------------------------------------------------------------------ portada
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = t.add_run("Laboratorio 2 — Pronóstico de caudal con RFKAN")
run.bold, run.font.size = True, Pt(20)
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("Recurrent Fourier-Kolmogorov Arnold Networks (Rong, Lin & Xie, Scientific Reports 15:4684, 2025) "
          "adaptadas al dataset Rainfall-Runoff").italic = True
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.add_run("Maestría — Deep Learning · Septiembre 2026")

rf, rk, kn, fk = (f"{m}_L{L}" for m in ("RFKAN", "RKAN", "KAN", "FKAN"))
pers = "Persistence"
H("Resumen", 1)
P(f"Implementamos desde cero RFKAN, una red de Kolmogorov–Arnold de dos capas con nodos kernel "
  f"recurrentes y activaciones en serie de Fourier, y la adaptamos a pronosticar 48 h de caudal "
  f"específico (mm/h) en 508 cuencas a partir de 336 h de historia de 12 variables. Replicamos la "
  f"ablación del paper (KAN / RKAN / FKAN / RFKAN), el barrido de largo de entrada (Fig. 5) y de "
  f"grilla (Fig. 4), y los baselines de la Tabla 5 (ARIMA, SVR, ELM, LSTM, GRU, TCN), más "
  f"persistencia. Con ventana L = {L} h, RFKAN obtiene RMSE de test {f4(r(rf, 'test_RMSE'))} mm/h "
  f"contra {f4(r(pers, 'test_RMSE'))} de persistencia ({pct(r(rf, 'test_RMSE'), r(pers, 'test_RMSE')):+.1f} %). "
  f"La recurrencia es el componente que más aporta: RKAN y RFKAN superan a KAN y FKAN en todos los "
  f"largos de ventana. En cambio, reemplazar splines por Fourier no mejora la precisión en este "
  f"problema (RKAN {f4(r(rk, 'test_RMSE'))} contra RFKAN {f4(r(rf, 'test_RMSE'))} en test), aunque sí "
  f"reduce el tiempo de entrenamiento. Frente a los modelos base de la Tabla 5, el resultado del paper "
  f"no se reproduce: con el mismo presupuesto y la misma salida residual, LSTM ({f4(r('LSTM', 'test_RMSE'))}) "
  f"y GRU ({f4(r('GRU', 'test_RMSE'))}) superan a toda la familia KAN, y TCN ({f4(r('TCN', 'test_RMSE'))}) "
  f"empata con RKAN. De los modelos avanzados de la Tabla 7 (implementaciones propias), Informer supera a RFKAN en test, "
  f"WVS empata y TACDPG no llega a persistencia; RFKAN es entre 5 y 11 veces más rápido que Informer y WVS. "
  f"Todos los modelos subestiman los picos de crecida, que son la principal limitación.")

# ------------------------------------------------------------------ 1
H("1. Introducción", 1)
P("La consigna pide implementar el método de un paper asignado, adaptarlo al pronóstico de caudal, "
  "entrenarlo, compararlo contra un modelo base, hacer una ablación de un componente importante y "
  "generar predicciones para todo el conjunto de test. El paper asignado propone RFKAN para "
  "pronóstico day-ahead de potencia fotovoltaica. El problema es estructuralmente análogo "
  "(regresión multivariada y multi-paso a dos días), pero el caudal no tiene el ciclo diario "
  "marcado de la potencia solar y está dominado por eventos de lluvia raros y abruptos. Esa "
  "diferencia es la clave para leer los resultados.")
P("El código está en lab/repo/ (Python + PyTorch). Las capas KAN, tanto la de splines como la de "
  "Fourier, son implementación propia; no se usa pykan, efficient-kan ni FourierKAN. Todos los "
  "experimentos se corren con lab/run_all.sh y dejan su detalle completo (historial por época, "
  "predicciones de val y test, métricas por horizonte y por cuenca) en archivos .pkl.")

# ------------------------------------------------------------------ 2
H("2. El método del paper (Rong et al., 2025)", 1)
H("2.1. Kolmogorov–Arnold Networks", 2)
P("El teorema de Kolmogorov–Arnold establece que toda función continua de n variables puede "
  "escribirse con sumas y funciones de una variable (Ec. 1):")
EQ("f(x₁,…,xₙ) = Σ_{q=1}^{2n+1} Φ_q( Σ_{p=1}^{n} φ_{q,p}(x_p) )")
P("Una KAN lleva esto a una red: cada arista tiene una función univariada aprendible φ, "
  "originalmente un B-spline, y los nodos solo suman. El paper usa dos capas: n → 2n+1 → salida.")
H("2.2. Nodos kernel recurrentes", 2)
P("Para capturar dependencias temporales, cada capa recibe, además de los nodos normales x(t), "
  "nodos de memoria (Ec. 3 y 4):")
EQ("h(t) = W_hh · h(t−1) + W_hz · x(t)")
EQ("x_{l+1,j}(t) = Σ_i φ_{l,j,i,t}( x_{l,i}(t), h_{l,i}(t) )")
H("2.3. Serie de Fourier como activación", 2)
P("En lugar de splines, cada función de arista es una serie de Fourier truncada a g armónicos "
  "(Ec. 8), según los autores más rápida de entrenar y más apta para datos periódicos:")
EQ("φ_F(x) = Σ_i Σ_{k=1}^{g} a_ik cos(k·x_i) + b_ik sin(k·x_i)")
P("El entrenamiento usa refinamiento de grilla: se entrena con g = 3 y, cada 50 pasos, se pasa a "
  "una grilla más fina [3, 5, 10, 20, 50, 100], heredando la función aprendida.")
H("2.4. Experimentos y resultados originales", 2)
T(["Modelo", "RMSE", "MAE", "CORR (R²)", "Tiempo (s)"],
  [["KAN", "5.4265", "3.0141", "0.8679", "4125.84"],
   ["RKAN", "4.3688", "2.6427", "0.9246", "4793.62"],
   ["FKAN", "4.1829", "2.7677", "0.9192", "3125.84"],
   ["RFKAN", "3.4168", "1.6146", "0.9461", "3519.38"]],
  caption="Tabla 1. Ablación del paper (Tabla 3; planta PV de 45 kW, test = 10 de mayo).", bold_rows=(3,))
P("El paper reporta además que el mejor largo de entrada es 2 y la mejor grilla 100, y que RFKAN "
  "supera a ARIMA, SVM, ELM, GRU y TCN en las cuatro estaciones (RMSE −58 %) y a Informer, WVS y "
  "TACDPG (RMSE −5 %). Detectamos inconsistencias: la Tabla 4 fija seq_len = 12 pero el texto dice "
  "que el óptimo es 2; la Tabla 5 da hiperparámetros de LSTM y la Tabla 6 reporta GRU; y la métrica "
  "CORR es en realidad el R².")

# ------------------------------------------------------------------ 3
H("3. Dataset y adaptación", 1)
T(["", "Paper", "Laboratorio"],
  [["Variable objetivo", "potencia PV (kW)", "caudal específico (mm/h)"],
   ["Series", "1 planta, 1 año, 5 min", "508 cuencas, horario"],
   ["Entrada", "clima + potencia", "336 h × 12 variables (11 meteorológicas + caudal)"],
   ["Salida", "288 pasos (2 días)", "48 pasos (2 días)"],
   ["Muestras", "105 120 puntos", "254 000 train / 18 142 val / 27 983 test"],
   ["Test", "1 día por estación", "test.h5 + test_targets.csv"]],
  widths=[3.5, 5, 7.5], caption="Tabla 2. Del problema del paper al del laboratorio.")
P("Particiones y preprocesamiento. Usamos las particiones del curso (split 0 = train, 1 = "
  "validación) y el test.h5 completo. El dataset no tiene faltantes, así que la interpolación del "
  "paper no aplica. Normalizamos con min-max a [0, 1] por canal, como el paper, con estadísticos "
  "solo de train. Las métricas se calculan en mm/h.")
P("Adaptaciones de arquitectura (todas justificadas en docs/GAP_ANALISIS.md):")
B(" la capa 2 tiene 48 nodos, una por hora de horizonte, leída en el último paso de la ventana.",
  "Salida de 48 h:")
B(" 12 nodos normales por paso (los 12 canales, como \"Input nodes 12\" de la Tabla 4) y 12 nodos "
  "recurrentes. φ(x, h) se implementa como dos conjuntos de aristas en paralelo, φ_x(x) + φ_h(h).",
  "Entrada:")
B(" KAN y FKAN reciben la ventana aplanada (n = 12·L entradas, como indica el paper) con el mismo "
  "ancho oculto 25.", "Variantes sin recurrencia:")
B(" ŷ = q(t) + σ_Δ · núcleo(x), con q(t) el último caudal observado y σ_Δ el desvío del cambio de "
  "caudal en train. Se aplica igual a todos los modelos neuronales.", "Salida residual:")
B(" lr · 3/g en cada etapa de grilla.", "Tasa de aprendizaje por etapa:")

# ------------------------------------------------------------------ 4
H("4. Implementación", 1)
T(["Archivo", "Contenido"],
  [["dataset.py", "carga de h5, particiones, min-max, datos en GPU"],
   ["kan_layers.py", "FourierKANLayer y SplineKANLayer desde cero, con extend_grid"],
   ["model.py", "RecurrentKernelNodes, RFKAN (4 variantes), Forecaster residual"],
   ["train.py", "Adam + MSE/σ_Δ², refinamiento de grilla por etapas, checkpoints"],
   ["baselines.py", "persistencia, ARIMA, SVR, ELM, LSTM, GRU, TCN"],
   ["informer.py", "Informer desde cero (ProbSparse, destilación, decoder generativo)"],
   ["wvs.py", "WVS: Pearson, VMD en GPU, WOA, un SCINet por modo"],
   ["tacdpg.py", "TACDPG: random forest, CatBoost (outliers), dos agentes TD3"],
   ["metrics.py", "RMSE, MAE, CORR (= R²), NSE por cuenca, KGE"],
   ["experiments.py", "suites seqlen, grid, baselines, seeds"],
   ["predict.py", "CSV de test Id, q_01..q_48"],
   ["analyze_results.py", "tablas, figuras y análisis de errores"]],
  widths=[4, 12], caption="Tabla 3. Estructura del código (lab/repo/).")
H("4.1. Problemas encontrados y cómo se resolvieron", 2)
B(" con la formulación literal (salida directa y lr constante), RFKAN diverge al refinar la grilla "
  "(val RMSE 0.27 → 3.6 mm/h). El cambio típico del caudal en 48 h es σ_Δ = 0.0046 en escala "
  "[0, 1], y Adam mueve cada armónico nuevo del orden de lr por paso. Se resolvió con la salida "
  "residual y el lr ∝ 3/g.", "Divergencia al refinar:")
B(" con la MSE en escala normalizada (~2.5·10⁻⁵), el 99 % de los gradientes de LSTM, GRU y TCN "
  "quedaba por debajo del ε de Adam (10⁻⁸) y el LSTM no salía de persistencia. Dividimos la pérdida "
  "por σ_Δ²: el óptimo es el mismo, pero los gradientes quedan en una escala útil. Todos los "
  "resultados de este informe son de la corrida con esta corrección.", "Gradientes bajo el ε de Adam:")
B(" con 336 pasos, el LSTM de 3 capas no aprende en 18 épocas. Los baselines neuronales usan la "
  f"misma ventana L = {L} que RFKAN.", "LSTM con ventana larga:")
B(" la extensión de grilla spline con 4032 entradas pedía 6.9 GB de una vez (se resolvió por "
  "bloques), la grilla 500 no entra en 16 GB (el schedule extendido llega a 200) y RKAN con L = 336 "
  "se quedó sin memoria en la época 16 (grilla 100): se reporta su mejor checkpoint (época 12, "
  "grilla 20).", "Memoria:")
B(" en ~1 % de las ventanas el ajuste da NaN o infinito; ahí se usa persistencia.", "ARIMA:")

# ------------------------------------------------------------------ 5
H("5. Metodología experimental", 1)
P(f"Todos los brazos usan la misma partición, la semilla 42 y el mismo presupuesto: 18 épocas "
  f"(3 por grilla × 6 grillas), el análogo de las 300 iteraciones del paper. Se usa Adam con lr 10⁻³, "
  f"batch 256 para la familia KAN y 32 para los baselines neuronales (Tabla 5), y clip de gradiente "
  f"1.0. Se selecciona la época con menor RMSE de validación y se reporta test sobre las 27 983 "
  f"ventanas × 48 h. Métricas: las del paper (RMSE, MAE, CORR = R²) y, por ser estándar en "
  f"hidrología, el NSE por cuenca (mediana sobre 508 cuencas) y el KGE.")
T(["Suite", "Brazos", "Análogo del paper"],
  [["seqlen", "KAN/RKAN/FKAN/RFKAN × L ∈ {2, 12, 24, 72, 168, 336}", "Tabla 3 + Fig. 5"],
   ["grid", f"RFKAN L={L}: grilla hasta 200, fija 3, fija 100, sin reducir el lr", "Fig. 4"],
   ["baselines", "persistencia, ARIMA(2,1,1), SVR, ELM, LSTM, GRU, TCN", "Tablas 5 y 6"],
   ["seeds", "RFKAN y LSTM con semillas 1, 2, 3", "— (no está en el paper)"],
   ["advanced", "Informer, WVS, TACDPG (implementaciones propias)", "Tabla 7"],
   ["seqlen_base", "LSTM / GRU / TCN × L ∈ {2, 12, 24, 168}", "Fig. 5 (otros modelos)"]],
  widths=[2.5, 9.5, 4], caption="Tabla 4. Experimentos.")

# ------------------------------------------------------------------ 6
H("6. Resultados", 1)
H(f"6.1. Ablación de arquitectura (análogo a la Tabla 3), L = {L}", 2)
rows = [[m, DESC[m], f4(r(f"{m}_L{L}", "val_RMSE")), f4(r(f"{m}_L{L}", "test_RMSE")),
         f4(r(f"{m}_L{L}", "test_MAE")), f4(r(f"{m}_L{L}", "test_CORR")),
         f4(r(f"{m}_L{L}", "test_NSE_median")), f"{r(f'{m}_L{L}', 'train_time_s'):.0f}"] for m in FAM]
rows.append(["Persistencia", "último caudal", f4(r(pers, "val_RMSE")), f4(r(pers, "test_RMSE")),
             f4(r(pers, "test_MAE")), f4(r(pers, "test_CORR")), f4(r(pers, "test_NSE_median")), "0"])
T(["Modelo", "Variante", "RMSE val", "RMSE test", "MAE test", "CORR test", "NSE med.", "Tiempo (s)"],
  rows, caption="Tabla 5. Ablación KAN / RKAN / FKAN / RFKAN (métricas en mm/h; NSE = mediana por cuenca).")
d_rec = pct(r(rf, "test_RMSE"), r(fk, "test_RMSE"))
d_four = pct(r(rf, "test_RMSE"), r(rk, "test_RMSE"))
P(f"La recurrencia es el componente decisivo: agregarla a la versión Fourier (FKAN → RFKAN) cambia el "
  f"RMSE de test en {d_rec:+.1f} %. Reemplazar splines por Fourier en la versión recurrente "
  f"(RKAN → RFKAN) lo cambia en {d_four:+.1f} %; es decir, Fourier no aporta precisión aquí. "
  f"Fourier sí es más rápido: RFKAN entrena en {r(rf, 'train_time_s') / r(rk, 'train_time_s'):.2f}× "
  f"el tiempo de RKAN (el paper reporta −24 %). La explicación más probable es que el argumento del "
  f"paper a favor de Fourier es la periodicidad de la potencia solar, y el caudal a 48 h no es "
  f"periódico: los armónicos altos agregan oscilaciones en caudales bajos (Sección 6.7).")

H("6.2. Largo de la ventana de entrada (análogo a la Fig. 5)", 2)
lens = sorted({int(x) for x in res.loc[res["suite"] == "seqlen", "seq_len"].dropna()})
T(["Modelo"] + [f"L={x}" for x in lens],
  [[m] + [f4(r(f"{m}_L{x}", "val_RMSE")) for x in lens] for m in FAM],
  caption="Tabla 6. RMSE de validación (mm/h) según el largo de ventana L (horas). RKAN con L=336: "
          "mejor checkpoint antes del corte por memoria (época 12 de 18).")
T(["Modelo"] + [f"L={x}" for x in lens if x != 336],
  [[m] + [f4(r(m if x == L else f"{m}_L{x}", "val_RMSE")) for x in lens if x != 336] for m in ("LSTM", "GRU", "TCN")],
  caption="Tabla 6b. RMSE de validación de los modelos base según L (la Fig. 5 del paper también los incluye).")
FIG("fig5_seqlen.png", "Figura 1. RMSE y CORR de validación contra el largo de ventana (escala logarítmica); "
                       "modelos base en gris punteado.")
best_L = min(lens, key=lambda x: r(f"RFKAN_L{x}", "val_RMSE"))
P(f"Los modelos recurrentes son casi insensibles al largo de la ventana: RKAN y RFKAN se mantienen en "
  f"un rango estrecho para todo L. Los modelos sin recurrencia, que reciben la ventana aplanada, "
  f"empeoran a medida que L crece, porque el número de entradas (12·L) aumenta sin que la red tenga "
  f"estructura temporal. El mejor largo para RFKAN es L = {best_L} h. El paper encuentra el óptimo en "
  f"L = 2, lo que en nuestro caso solo vale para las variantes sin recurrencia. Con L = 2 todos los modelos "
  f"recurrentes empatan (≈0.105 de val RMSE). Con más historia, LSTM y GRU mejoran hasta 24–72 h, mientras "
  f"que RKAN y RFKAN quedan planos: la memoria lineal de los nodos kernel recurrentes (Ec. 3) no aprovecha la "
  f"historia larga, y las compuertas de LSTM y GRU sí. El LSTM con L = 168 no sale de persistencia. El tiempo de "
  f"los baselines no crece de forma exponencial con L, como afirma el paper: con cuDNN tardan 4–6 min con "
  f"cualquier ventana. La que se encarece es la familia KAN con splines: RKAN tarda "
  f"{r('RKAN_L2', 'train_time_s'):.0f} s con L = 2 y {r('RKAN_L168', 'train_time_s') / 60:.0f} min con L = 168.")

H("6.3. Refinamiento de grilla (análogo a la Fig. 4)", 2)
g = [(f"RFKAN_L{L}", "refinamiento 3→100, lr ∝ 3/g (estándar)"),
     (f"RFKAN_L{L}_grid_ext", "refinamiento 3→200"),
     (f"RFKAN_L{L}_fixed_g3", "grilla 3 fija"),
     (f"RFKAN_L{L}_fixed_g100", "grilla 100 fija desde el inicio"),
     (f"RFKAN_L{L}_nodecay", "refinamiento 3→100 con lr constante")]
T(["Variante", "RMSE val", "RMSE test", "Mejor grilla"],
  [[d, f4(r(a, "val_RMSE")), f4(r(a, "test_RMSE")),
    "—" if pd.isna(r(a, "best_grid")) else f"{r(a, 'best_grid'):.0f}"] for a, d in g],
  widths=[8, 2.5, 2.5, 2.5], caption="Tabla 7. Ablación del refinamiento de grilla.")
FIG("fig4_grid.png", "Figura 2. RMSE de validación por época; las líneas punteadas marcan cada cambio de grilla.")
bg = res.loc[res["suite"] == "seqlen", "best_grid"].dropna()
P(f"En los {len(bg)} brazos de la suite seqlen, la mejor época cae en grilla "
  f"{', '.join(f'{int(k)} ({v})' for k, v in bg.value_counts().sort_index().items())} "
  f"(entre paréntesis, cuántos brazos). Refinar hasta 100 no mejora la precisión, a diferencia del "
  f"paper. Sí importa cómo se llega a una grilla fina: empezar directamente con 100 armónicos o "
  f"refinar sin reducir el lr degrada el modelo hasta quedar peor que persistencia. El refinamiento "
  f"progresivo con lr decreciente funciona como un esquema de regularización.")

H("6.4. Comparación con los modelos base (análogo a la Tabla 6)", 2)
base = ["Persistence", "ARIMA", "SVR", "ELM", "LSTM", "GRU", "TCN", rk, rf]
rows = [[a, f4(r(a, "test_RMSE")), f4(r(a, "test_MAE")), f4(r(a, "test_CORR")),
         f4(r(a, "test_NSE_median")), f"{r(a, 'train_time_s', 0):.0f}"] for a in base if a in res.index]
T(["Modelo", "RMSE", "MAE", "CORR", "NSE med.", "Tiempo (s)"], rows,
  caption="Tabla 8. Test (27 983 ventanas × 48 h). ARIMA: submuestreo fijo de 2000 ventanas.")
best_nn = min(["LSTM", "GRU", "TCN"], key=lambda a: r(a, "test_RMSE"))
P(f"Con la pérdida corregida y la misma ventana, los baselines neuronales son competitivos: {best_nn} "
  f"obtiene el mejor RMSE de test ({f4(r(best_nn, 'test_RMSE'))}), "
  f"{pct(r(best_nn, 'test_RMSE'), r(rf, 'test_RMSE')):+.1f} % respecto de RFKAN. El paper reporta lo "
  f"contrario (RFKAN −58 % de RMSE frente a GRU y TCN). Nuestra evaluación iguala condiciones "
  f"(mismas particiones, presupuesto de 18 épocas, salida residual y pérdida para todos) y es mucho más "
  f"amplia (27 983 ventanas de 508 cuencas contra 4 días de una planta).")
P("ARIMA(2,1,1) se ajusta por ventana. En la cuenca típica le gana a persistencia (NSE mediano más "
  "alto), pero en algunas ventanas la diferenciación extrapola una tendencia y el pronóstico explota, "
  "lo que dispara su RMSE global. SVR con C = 0.0043 (Tabla 5) está tan regularizado que predice "
  "casi siempre un cambio nulo, es decir, persistencia. El ELM con λ = 0.001 sobreajusta y queda peor "
  "que persistencia; con λ escalado por el número de muestras (primera corrida) quedaba igual a "
  "persistencia (val RMSE 0.1371). Ninguna de las dos regularizaciones le sirve en este problema.")

P("Predicciones de test: lab/repo/outputs/experiments/predictions_test_RFKAN_L72.csv, con 27 983 "
  "filas en el formato Id, q_01..q_48 (mm/h). Las predicciones negativas se recortan a 0.")
H("6.5. Modelos avanzados: Informer, WVS y TACDPG (análogo a la Tabla 7)", 2)
arms7 = [a for a in ("Informer", "WVS", "TACDPG", rf, rk, "LSTM") if a in res.index]
if "Informer" in arms7:
    T(["Modelo", "RMSE val", "RMSE test", "CORR test", "Parámetros", "Tiempo (s)"],
      [[a, f4(r(a, "val_RMSE")), f4(r(a, "test_RMSE")), f4(r(a, "test_CORR")), f"{r(a, 'params'):.0f}",
        f"{r(a, 'train_time_s'):.0f}"] for a in arms7],
      caption="Tabla 9. Modelos avanzados, implementaciones propias: Informer (Zhou et al. 2021), WVS = "
              "WOA-VMD-SCINet (Zhao et al. 2024) y TACDPG (Zhang et al. 2024), contra RFKAN, RKAN y LSTM.")
    P(f"Informer tiene {r('Informer', 'params') / r(rf, 'params'):.0f} veces más parámetros que RFKAN y tarda "
      f"{r('Informer', 'train_time_s') / r(rf, 'train_time_s'):.1f} veces más en entrenar. En validación queda "
      f"{'por detrás' if r('Informer', 'val_RMSE') > r(rf, 'val_RMSE') else 'por delante'} de RFKAN "
      f"({f4(r('Informer', 'val_RMSE'))} contra {f4(r(rf, 'val_RMSE'))}) y en test "
      f"{'por delante' if r('Informer', 'test_RMSE') < r(rf, 'test_RMSE') else 'por detrás'} "
      f"({f4(r('Informer', 'test_RMSE'))} contra {f4(r(rf, 'test_RMSE'))}). El paper reporta RFKAN por delante de "
      f"Informer y un 24 % más rápido: la ventaja de tiempo se reproduce; la de precisión, no de forma consistente. ")
    P(f"WVS (Pearson, VMD por ventana, WOA sobre (K, α) con entropía de envolvente y un SCINet por modo) obtiene "
      f"{f4(r('WVS', 'test_RMSE'))} de RMSE de test, empatado con RFKAN, y el menor MAE entre los modelos con ventana de 72 h y los avanzados "
      f"({f4(r('WVS', 'test_MAE'))}), pero tarda {r('WVS', 'train_time_s') / r(rf, 'train_time_s'):.0f} veces más. "
      f"TACDPG (random forest, CatBoost para outliers según la Ec. 1 de su paper, dos agentes TD3 con recompensa "
      f"adaptativa) queda en {f4(r('TACDPG', 'test_RMSE'))}, peor que persistencia: sus agentes tienen la mejor "
      f"validación en la primera evaluación y luego se degradan. El aprendizaje por refuerzo con una recompensa "
      f"escalar no aprende una corrección útil de 48 h. Varios hiperparámetros de WVS y TACDPG están en material "
      f"suplementario no disponible y son supuestos declarados (docs/GAP_ANALISIS.md).")
H("6.6. Robustez a la semilla", 2)
rows = []
for name, arms in (("RFKAN", [rf] + [f"{rf}_seed{s}" for s in (1, 2, 3)]),
                   ("LSTM", ["LSTM"] + [f"LSTM_seed{s}" for s in (1, 2, 3)])):
    sub = res.loc[[a for a in arms if a in res.index]]
    if len(sub) > 1:
        rows.append([name, len(sub)] + [f"{sub[c].mean():.4f} ± {sub[c].std():.4f}"
                                        for c in ("val_RMSE", "test_RMSE", "test_CORR")])
if rows:
    T(["Modelo", "n", "RMSE val", "RMSE test", "CORR test"], rows,
      caption="Tabla 10. Media ± desvío sobre semillas (42, 1, 2, 3).")
    seeds = [a for a in res.index if a == "LSTM" or a.startswith("LSTM_seed")]
    stuck = [a for a in seeds if r(a, "val_RMSE") >= 0.99 * r(pers, "val_RMSE")]
    ok = [a for a in seeds if a not in stuck]
    rfs = [a for a in res.index if a == rf or a.startswith(f"{rf}_seed")]
    P(f"RFKAN es estable: sus {len(rfs)} semillas quedan entre {min(r(a, 'test_RMSE') for a in rfs):.4f} y "
      f"{max(r(a, 'test_RMSE') for a in rfs):.4f} de RMSE de test. El LSTM, en cambio, no sale de "
      f"persistencia en {len(stuck)} de sus {len(seeds)} semillas ({', '.join(stuck)}): su RMSE de "
      f"validación queda en {r(pers, 'val_RMSE'):.4f} durante las 18 épocas. Cuando entrena, es el mejor "
      f"modelo (test {', '.join(f4(r(a, 'test_RMSE')) for a in ok)}); en promedio sobre semillas, RFKAN "
      f"lo supera. Es un resultado a favor de RFKAN que el paper no mide: con la salida residual, los "
      f"modelos KAN arrancan en persistencia y se alejan de ella de forma fiable, mientras que el LSTM "
      f"de 3 capas puede quedar atrapado en ese punto.")
else:
    P("[Pendiente: la suite seeds todavía no terminó.]", italic=True)

H("6.7. Análisis de errores", 2)
FIG("fig_horizonte.png", "Figura 3. RMSE de test por hora de horizonte.", 13)
P("El error crece con el horizonte: las primeras horas están dominadas por la persistencia del "
  "caudal y las últimas por la lluvia futura, que el modelo no observa.")
FIG("fig_nse_cuencas.png", "Figura 4. Distribución del NSE por cuenca en test.", 12)
tab = md_table("errores_por_regimen.md")
if tab:
    T(tab[0], tab[1], caption="Tabla 11. RMSE de test (mm/h) por régimen: caudal actual, lluvia de "
                              "las últimas 24 h y temperatura media (proxy de la estación).")
    P("El error se concentra en caudales altos y después de lluvias fuertes, donde la recurrencia "
      "aporta más. Con temperaturas bajas (invierno), el error es mayor.")
tab = md_table("picos.md")
if tab:
    T(tab[0], tab[1], caption="Tabla 12. Error en picos (1 % de caudales más altos del test).")
    P("Todos los modelos entrenados subestiman los picos de crecida en un 32–39 % (persistencia, 24 %). Es la limitación principal: "
      "una MSE sobre 48 horas, con una mayoría de horas de recesión suave, empuja al modelo a "
      "predecir cerca de la persistencia.")
FIG("fig_hidrogramas.png", "Figura 5. Ejemplos de test: pico extremo, pico moderado y caudal bajo.", 17)
P("En caudales bajos (panel derecho), los modelos de la familia KAN producen trayectorias que "
  "oscilan hora a hora, y RFKAN es el que más oscila, mientras que LSTM y TCN dan curvas suaves. Las "
  "48 salidas de la capa 2 no tienen ningún acople temporal entre sí, y las funciones de activación "
  "de grilla fina (sobre todo los armónicos altos de Fourier) amplifican pequeñas diferencias de "
  "entrada. Un LSTM o una TCN comparten parámetros a lo largo del tiempo y producen salidas más "
  "regulares.")

# ------------------------------------------------------------------ 7
H("7. Conclusiones y limitaciones", 1)
B(f" RFKAN mejora a persistencia en {-pct(r(rf, 'test_RMSE'), r(pers, 'test_RMSE')):.1f} % de RMSE "
  "de test y a los baselines clásicos (ARIMA, SVR, ELM).", "El método funciona adaptado al caudal:")
B(f" LSTM ({f4(r('LSTM', 'test_RMSE'))}) y GRU ({f4(r('GRU', 'test_RMSE'))}) superan a RFKAN "
  f"({f4(r(rf, 'test_RMSE'))}) en test en igualdad de condiciones. La ventaja que reporta el paper no "
  "se reproduce en este dataset. Sin embargo, el LSTM no entrena en 2 de 4 semillas, y RFKAN sí en "
  "todas: RFKAN es menos preciso pero más robusto.", "Pero no supera a los baselines recurrentes:")
B(" en todos los largos de ventana, las variantes con nodos kernel recurrentes superan a las que "
  "no los tienen. Coincide con el paper.", "La recurrencia es la contribución que se sostiene:")
B(" Fourier es más rápido, pero no más preciso que splines, y es la variante que más oscila en "
  "caudales bajos. No reproduce el resultado del paper, cuyo argumento (periodicidad) no aplica al caudal.",
  "Fourier no se sostiene:")
B(" el mejor modelo queda en grilla 10–20; refinar hasta 100 no ayuda, pero el esquema progresivo "
  "con lr decreciente es necesario para que una grilla fina no degrade el modelo.",
  "Refinamiento de grilla:")
B(" los picos de crecida se subestiman en 32–39 %, no se usa la meteorología futura (y_aux) y varios "
  "hiperparámetros de WVS y TACDPG son supuestos (material suplementario no disponible).", "Limitaciones:")
B(" pérdidas que pesen los picos (por ejemplo, NSE o cuantiles), usar y_aux como supervisión "
  "auxiliar y una base de funciones local (wavelets) en lugar de Fourier global.", "Trabajo futuro:")

# ------------------------------------------------------------------ 8
H("8. Referencias", 1)
for ref in [
    "D. Rong, Z. Lin, G. Xie. Recurrent Fourier-Kolmogorov Arnold Networks for photovoltaic power "
    "forecasting. Scientific Reports 15:4684 (2025). doi:10.1038/s41598-025-88959-5",
    "Z. Liu et al. KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756 (2024).",
    "R. Jozefowicz, W. Zaremba, I. Sutskever. An empirical exploration of recurrent network "
    "architectures. ICML (2015).",
    "Dataset Rainfall-Runoff del curso. Código: lab/repo/ · Resultados: "
    "lab/repo/outputs/experiments/ · Documentación: docs/."]:
    doc.add_paragraph(ref, style="List Number")

doc.save(OUT)
print("->", OUT)
