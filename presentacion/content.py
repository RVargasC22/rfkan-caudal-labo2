"""
Contenido de la presentacion (una sola fuente para build_pptx.py y build_deck.py).
Los numeros se leen de lab/repo/outputs/experiments/experiments_results.csv.

Cada slide es un dict:
  id, section ("Metodologia" | "Implementacion" | "Resultados"), title,
  body: lista de bloques
     ("text", str) | ("bullets", [str]) | ("table", header, rows) |
     ("image", "fig.png", caption) | ("eq", [str]) | ("big", [(valor, etiqueta)]) |
     ("chart_line", categories, {serie: valores}, titulo_eje_y) |
     ("flow", [str])   -- cajas conectadas por flechas
  notes: guion para el video
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "lab/repo/outputs/experiments"
ANA = ROOT / "lab/repo/outputs/analysis"
L = 72
FAM = ["KAN", "RKAN", "FKAN", "RFKAN"]

res = pd.read_csv(EXP / "experiments_results.csv").drop_duplicates("arm", keep="last").set_index("arm")


def r(arm, col):
    return res.loc[arm, col] if arm in res.index and col in res.columns else float("nan")


def f4(x):
    return "—" if pd.isna(x) else f"{x:.4f}"


def f3(x):
    return "—" if pd.isna(x) else f"{x:.3f}"


def pct(a, b):
    return 100 * (a - b) / b


def md_table(name):
    path = ANA / name
    if not path.exists():
        return None
    lines = [l for l in path.read_text().splitlines() if l.startswith("|")]
    split = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    return split[0], split[2:]


rf, rk, kn, fk, pers = f"RFKAN_L{L}", f"RKAN_L{L}", f"KAN_L{L}", f"FKAN_L{L}", "Persistence"
lens = sorted({int(x) for x in res.loc[res["suite"] == "seqlen", "seq_len"].dropna()})
bg = res.loc[res["suite"] == "seqlen", "best_grid"].dropna().astype(int).value_counts().sort_index()


def seeds_rows():
    rows = []
    for name, arms in (("RFKAN", [rf] + [f"{rf}_seed{s}" for s in (1, 2, 3)]),
                       ("LSTM", ["LSTM"] + [f"LSTM_seed{s}" for s in (1, 2, 3)])):
        sub = res.loc[[a for a in arms if a in res.index]]
        if len(sub) > 1:
            rows.append([name, str(len(sub)), f"{sub['test_RMSE'].mean():.4f} ± {sub['test_RMSE'].std():.4f}",
                         f"{sub['test_CORR'].mean():.3f} ± {sub['test_CORR'].std():.3f}"])
    return rows or [["(pendiente)", "", "", ""]]


SLIDES = [
    # ------------------------------------------------------------ Metodologia
    dict(id="cover", section="Metodologia", title="RFKAN para pronóstico de caudal",
         body=[("text", "Recurrent Fourier-Kolmogorov Arnold Networks (Rong, Lin & Xie, Scientific Reports "
                        "15:4684, 2025) adaptadas a 508 cuencas · Laboratorio 2, Deep Learning")],
         notes="Presentamos la implementación de RFKAN, un paper de pronóstico de potencia solar, adaptado "
               "al pronóstico de caudal. El video tiene tres partes: metodología, implementación y resultados."),
    dict(id="problema", section="Metodologia", title="El problema: 48 horas de caudal",
         body=[("big", [("336 h", "de historia, 12 variables"), ("48 h", "de caudal a pronosticar"),
                        ("508", "cuencas, las mismas en train, val y test")]),
               ("text", "Entrada: precipitación, temperatura, radiación, viento, humedad, presión, evaporación "
                        "y caudal. Salida: caudal específico en mm/h. 254 000 ventanas de train, 18 142 de "
                        "validación y 27 983 de test.")],
         notes="Cada muestra es una ventana de 14 días de una cuenca, con 11 variables meteorológicas y el "
               "caudal. Hay que predecir las siguientes 48 horas de caudal. Son 508 cuencas anónimas, que "
               "aparecen en las tres particiones."),
    dict(id="paper", section="Metodologia", title="El paper: RFKAN para potencia solar",
         body=[("bullets", ["Pronóstico day-ahead de una planta fotovoltaica de 45 kW (Tieling, China, 2023).",
                            "Parte de una KAN de dos capas y le agrega dos cosas:",
                            "1. nodos kernel recurrentes, para capturar la dependencia temporal;",
                            "2. series de Fourier en lugar de splines, para extraer periodicidad y entrenar más rápido.",
                            "Entrena con refinamiento de grilla: 3 → 5 → 10 → 20 → 50 → 100."])],
         notes="El paper asignado es de Scientific Reports 2025. Pronostica la potencia de una planta solar "
               "para el día siguiente. Su propuesta, RFKAN, parte de las redes de Kolmogorov-Arnold y les "
               "agrega memoria recurrente y activaciones de Fourier."),
    dict(id="kan", section="Metodologia", title="Kolmogorov–Arnold: funciones en las aristas",
         body=[("eq", ["f(x₁,…,xₙ) = Σ_{q=1}^{2n+1} Φ_q( Σ_{p=1}^{n} φ_{q,p}(x_p) )"]),
               ("bullets", ["Un MLP tiene pesos en las aristas y una no linealidad fija en los nodos.",
                            "Una KAN tiene una función univariada aprendible φ en cada arista; los nodos solo suman.",
                            "Dos capas: n → 2n+1 → salida. Con 12 canales, 12 → 25 → 48."])],
         notes="El teorema de Kolmogorov-Arnold dice que cualquier función continua de muchas variables se "
               "escribe con sumas y funciones de una variable. Una KAN lleva eso a una red: en cada arista "
               "hay una función aprendible. En la KAN original es un B-spline."),
    dict(id="recurrencia", section="Metodologia", title="Nodos kernel recurrentes",
         body=[("eq", ["h(t) = W_hh · h(t−1) + W_hz · x(t)",
                       "x_{l+1,j}(t) = Σ_i φ_{l,j,i,t}( x_{l,i}(t), h_{l,i}(t) )"]),
               ("text", "Una memoria lineal tipo RNN que entra en paralelo con los nodos normales a cada capa. "
                        "Cada arista ve el valor actual y un resumen de todo lo anterior.")],
         notes="El primer cambio son los nodos kernel recurrentes, la ecuación 3. Es una RNN lineal: la "
               "memoria de hoy combina la de ayer con la entrada de hoy. Esos nodos se suman como entradas "
               "extra a cada capa KAN."),
    dict(id="fourier", section="Metodologia", title="Fourier en lugar de splines",
         body=[("eq", ["φ_F(x) = Σ_i Σ_{k=1}^{g} a_ik cos(k·x_i) + b_ik sin(k·x_i)"]),
               ("bullets", ["g = tamaño de grilla = número de armónicos.",
                            "Refinar la grilla = agregar armónicos; los nuevos arrancan en 0 y la función heredada no cambia.",
                            "Argumento del paper: la potencia solar es periódica, y Fourier la captura con pocos parámetros."])],
         notes="El segundo cambio reemplaza el spline de cada arista por una serie de Fourier con g armónicos. "
               "Refinar la grilla significa agregar armónicos. Guarden este argumento: la periodicidad de la "
               "potencia solar. Vamos a volver a él en los resultados."),
    dict(id="paper_res", section="Metodologia", title="Lo que reporta el paper",
         body=[("table", ["Modelo", "RMSE", "MAE", "CORR", "Tiempo (s)"],
                [["KAN", "5.43", "3.01", "0.868", "4126"], ["RKAN", "4.37", "2.64", "0.925", "4794"],
                 ["FKAN", "4.18", "2.77", "0.919", "3126"], ["RFKAN", "3.42", "1.61", "0.946", "3519"]]),
               ("bullets", ["Óptimo: grilla 100 y largo de entrada 2. Supera a ARIMA, SVM, ELM, GRU y TCN.",
                            "Inconsistencias: seq_len 12 contra 2; LSTM (Tabla 5) contra GRU (Tabla 6); CORR es R²."])],
         notes="La ablación del paper muestra que cada cambio mejora a KAN y que juntos mejoran más. Hay "
               "inconsistencias: la tabla de hiperparámetros dice largo 12 pero el texto dice 2, y los "
               "baselines cambian de LSTM a GRU entre tablas. La métrica CORR es en realidad el R cuadrado."),
    # ------------------------------------------------------------ Implementacion
    dict(id="adaptacion", section="Implementacion", title="Del sol al río: qué se adaptó",
         body=[("table", ["", "Paper", "Laboratorio"],
                [["Objetivo", "potencia PV", "caudal (mm/h)"], ["Series", "1 planta, 5 min", "508 cuencas, 1 h"],
                 ["Salida", "1 nodo, 288 pasos", "48 nodos, 48 h"], ["Entrada", "largo 2", f"barrido 2–336; principal L = {L}"],
                 ["Test", "1 día por estación", "27 983 ventanas"]]),
               ("text", "Min-max [0, 1] con estadísticos de train, como el paper. No hay faltantes: la "
                        "interpolación del paper no aplica.")],
         notes="Estos son los cambios de dominio. Lo más importante: la salida pasa a tener 48 nodos, uno "
               "por hora, y barremos el largo de entrada. La normalización es la del paper, con "
               "estadísticos solo de train."),
    dict(id="arquitectura", section="Implementacion", title="La arquitectura implementada",
         body=[("flow", ["x (L × 12)", "+ h(t) recurrente", "Capa 1 Fourier-KAN 24 → 25", "+ h(t) recurrente",
                         "Capa 2 Fourier-KAN 50 → 48", "ŷ = q(t) + σ_Δ · salida"]),
               ("bullets", ["Capas KAN propias (kan_layers.py): Fourier y B-spline con Cox–de Boor, sin librerías.",
                            "φ(x, h) = φ_x(x) + φ_h(h): nodos normales y recurrentes en paralelo, como la Fig. 1.",
                            "KAN y FKAN (sin recurrencia): ventana aplanada, 12·L entradas y ancho oculto 25."])],
         notes="Todo está implementado desde cero: la capa Fourier, la capa spline con la recursión de "
               "Cox-de Boor y la extensión de grilla por mínimos cuadrados. Las cuatro variantes de la "
               "ablación salen de la misma clase cambiando dos banderas."),
    dict(id="divergencia", section="Implementacion", title="Problema 1: divergencia al refinar la grilla",
         body=[("big", [("0.27 → 3.6", "val RMSE al pasar a grilla 50–100"), ("σ_Δ = 0.0046", "cambio típico en 48 h (escala [0, 1])")]),
               ("bullets", ["Adam mueve cada armónico nuevo del orden de lr por paso: ruido frente a 0.0046.",
                            "Salida residual: ŷ = q(t) + σ_Δ · núcleo(x), igual para todos los modelos neuronales.",
                            "lr por etapa ∝ 3/g. Con eso, RFKAN baja de persistencia."])],
         notes="La versión literal divergía. El caudal cambia muy poco en escala normalizada, y cada armónico "
               "nuevo se mueve con el tamaño de paso de Adam. La solución fue predecir la corrección sobre "
               "el último caudal y bajar el learning rate en cada etapa. Se aplica igual a todos los modelos."),
    dict(id="eps", section="Implementacion", title="Problema 2: gradientes bajo el ε de Adam",
         body=[("big", [("99 %", "de los gradientes de LSTM, GRU y TCN < 10⁻⁸"), ("2.5·10⁻⁵", "valor de la MSE normalizada")]),
               ("bullets", ["El LSTM terminaba exactamente en persistencia.",
                            "Pérdida = MSE / σ_Δ²: mismo óptimo, gradientes en escala útil.",
                            f"Además, LSTM, GRU y TCN usan la misma ventana L = {L} que RFKAN: con 336 pasos, el LSTM no aprende en 18 épocas."])],
         notes="El segundo problema lo descubrimos cuando el LSTM daba exactamente persistencia. Medimos los "
               "gradientes: el 99 por ciento estaba por debajo del epsilon de Adam. Escalamos la pérdida por "
               "sigma al cuadrado y repetimos todos los experimentos."),
    dict(id="experimentos", section="Implementacion", title="Experimentos: mismo split, semilla y presupuesto",
         body=[("table", ["Suite", "Qué varía", "Análogo"],
                [["seqlen", "KAN / RKAN / FKAN / RFKAN × L ∈ {2, 12, 24, 72, 168, 336}", "Tabla 3 + Fig. 5"],
                 ["seqlen_base", "LSTM / GRU / TCN × L ∈ {2, 12, 24, 168}", "Fig. 5"],
                 ["grid", "grilla hasta 200, fija 3, fija 100, sin reducir el lr", "Fig. 4"],
                 ["baselines", "persistencia, ARIMA, SVR, ELM, LSTM, GRU, TCN", "Tablas 5–6"],
                 ["advanced", "Informer, WVS, TACDPG (propios)", "Tabla 7"],
                 ["seeds", "RFKAN y LSTM × 3 semillas", "—"]]),
               ("text", "18 épocas por brazo (3 por grilla). Todo el detalle de cada brazo queda en un .pkl: "
                        "historial, predicciones de val y test, métricas por horizonte y por cuenca.")],
         notes="Cuatro suites de experimentos, todas con la misma partición, semilla y presupuesto de 18 "
               "épocas. Todo queda guardado en archivos pkl para el análisis."),
    # ------------------------------------------------------------ Resultados
    dict(id="ablacion", section="Resultados", title=f"Ablación: la recurrencia decide (L = {L})",
         body=[("table", ["Modelo", "RMSE val", "RMSE test", "CORR test", "NSE med.", "Tiempo (s)"],
                [[m, f4(r(f"{m}_L{L}", "val_RMSE")), f4(r(f"{m}_L{L}", "test_RMSE")), f3(r(f"{m}_L{L}", "test_CORR")),
                  f3(r(f"{m}_L{L}", "test_NSE_median")), f"{r(f'{m}_L{L}', 'train_time_s'):.0f}"] for m in FAM]
                + [["Persistencia", f4(r(pers, "val_RMSE")), f4(r(pers, "test_RMSE")), f3(r(pers, "test_CORR")),
                    f3(r(pers, "test_NSE_median")), "0"]]),
               ("text", f"Agregar recurrencia a Fourier (FKAN → RFKAN): {pct(r(rf, 'test_RMSE'), r(fk, 'test_RMSE')):+.1f} % de RMSE de test. "
                        f"Cambiar splines por Fourier (RKAN → RFKAN): {pct(r(rf, 'test_RMSE'), r(rk, 'test_RMSE')):+.1f} %.")],
         notes="Este es el resultado central. La recurrencia mejora claramente. Fourier, en cambio, no mejora "
               "a los splines: RKAN, que tiene splines y recurrencia, es el mejor. Fourier sí entrena más rápido."),
    dict(id="seqlen", section="Resultados", title="Largo de ventana: la recurrencia lo hace irrelevante",
         body=[("chart_line", [f"L={x}" for x in lens],
                {**{m: [r(f"{m}_L{x}", "val_RMSE") for x in lens] for m in FAM},
                 **{m: [r(m if x == L else f"{m}_L{x}", "val_RMSE") for x in lens] for m in ("LSTM", "GRU", "TCN")}},
                "RMSE val (mm/h)"),
               ("text", "RKAN y RFKAN son casi planos en L; KAN y FKAN (aplanados) empeoran. LSTM y GRU mejoran hasta "
                        "24–72 h; el LSTM con 168 h no entrena.")],
         notes="Barrimos el largo de ventana de 2 a 336 horas. Las variantes recurrentes casi no cambian. Las "
               "que reciben la ventana aplanada empeoran con más historia. El óptimo de largo 2 del paper solo "
               "vale para ellas."),
    dict(id="grilla", section="Resultados", title="Grilla: refinar no mejora, pero protege",
         body=[("table", ["Variante RFKAN", "RMSE val", "RMSE test"],
                [["refinamiento 3 → 100 (estándar)", f4(r(rf, "val_RMSE")), f4(r(rf, "test_RMSE"))],
                 ["refinamiento 3 → 200", f4(r(f"{rf}_grid_ext", "val_RMSE")), f4(r(f"{rf}_grid_ext", "test_RMSE"))],
                 ["grilla 3 fija", f4(r(f"{rf}_fixed_g3", "val_RMSE")), f4(r(f"{rf}_fixed_g3", "test_RMSE"))],
                 ["grilla 100 fija", f4(r(f"{rf}_fixed_g100", "val_RMSE")), f4(r(f"{rf}_fixed_g100", "test_RMSE"))],
                 ["refinamiento sin reducir el lr", f4(r(f"{rf}_nodecay", "val_RMSE")), f4(r(f"{rf}_nodecay", "test_RMSE"))]]),
               ("text", "Mejor época por grilla en la suite seqlen: " + ", ".join(f"g={k}: {v}" for k, v in bg.items())
                        + ". Ninguna mejora al llegar a 100.")],
         notes="El paper dice que la grilla óptima es 100. En nuestros datos, el mejor modelo queda en grilla "
               "10 o 20. Pero el refinamiento progresivo importa: empezar con 100 armónicos o no bajar el "
               "learning rate deja al modelo peor que persistencia."),
    dict(id="baselines", section="Resultados", title="Contra los modelos base: ganan LSTM y GRU",
         body=[("table", ["Modelo", "RMSE test", "MAE test", "CORR test", "NSE med."],
                [[a, f4(r(a, "test_RMSE")), f4(r(a, "test_MAE")), f3(r(a, "test_CORR")), f3(r(a, "test_NSE_median"))]
                 for a in ["Persistence", "ARIMA", "SVR", "ELM", "LSTM", "GRU", "TCN", rk, rf] if a in res.index]),
               ("text", f"LSTM y GRU superan a toda la familia KAN; el paper reportaba lo contrario. ARIMA: 2000 "
                        f"ventanas, algunas explotan. SVR (C = 0.0043) queda en persistencia; ELM sobreajusta.")],
         notes="Comparamos con los modelos base de la Tabla 5 más persistencia, que es el modelo base "
               "natural en hidrología. Los números están todos sobre el test completo, salvo ARIMA."),
    dict(id="avanzados", section="Resultados", title="Modelos avanzados (análogo a la Tabla 7)",
         body=[("table", ["Modelo", "RMSE val", "RMSE test", "MAE test", "Parámetros", "Tiempo (min)"],
                [[a, f4(r(a, "val_RMSE")), f4(r(a, "test_RMSE")), f4(r(a, "test_MAE")), f"{r(a, 'params') / 1e3:.0f} k",
                  f"{r(a, 'train_time_s') / 60:.0f}"] for a in ("Informer", "WVS", "TACDPG", rf, rk) if a in res.index]),
               ("text", "Implementaciones propias. RFKAN gana en validación y es el más rápido de los competitivos; "
                        "en test Informer lo supera y WVS empata. TACDPG no llega a persistencia.")],
         notes="El paper también compara contra tres modelos avanzados, y los implementamos los tres desde cero. "
               "Informer, un Transformer con atención ProbSparse. WVS, que descompone el caudal con VMD, optimiza "
               "esa descomposición con el algoritmo de ballenas y predice cada modo con un SCINet. Y TACDPG, un "
               "agente de aprendizaje por refuerzo con críticos gemelos y un clasificador CatBoost para outliers. "
               "En validación RFKAN supera a los tres, como dice el paper. En test, Informer lo supera y WVS empata, "
               "con menor error absoluto medio que RFKAN. TACDPG no llega ni a persistencia: el aprendizaje por refuerzo no logra "
               "aprender una corrección útil de 48 horas. Lo que sí se reproduce siempre es la velocidad: RFKAN es "
               "cinco veces más rápido que Informer y once veces más rápido que WVS."),
    dict(id="semillas", section="Resultados", title="Robustez: el LSTM falla en 2 de 4 semillas",
         body=[("table", ["Modelo", "n", "RMSE test", "CORR test"], seeds_rows()),
               ("image", "fig_horizonte.png", "RMSE de test por hora de horizonte")],
         notes="Con cuatro semillas, el desvío es chico frente a las diferencias entre variantes. El error "
               "crece con el horizonte: las primeras horas son casi persistencia y las últimas dependen de "
               "lluvia que el modelo no ve."),
    dict(id="picos", section="Resultados", title="La limitación: los picos de crecida",
         body=[("image", "fig_hidrogramas.png", "Ejemplos de test: pico extremo, pico moderado y caudal bajo"),
               ("text", "Todos los modelos subestiman el 1 % de caudales más altos en un 24–39 %. En caudales "
                        "bajos, la familia KAN oscila hora a hora (RFKAN más); LSTM y TCN dan curvas suaves.")],
         notes="La limitación principal son los picos. Ningún modelo los anticipa. Y en caudal bajo se ve el "
               "zigzag de los armónicos de Fourier, que es la evidencia más directa de que esta base no es "
               "la adecuada para el caudal."),
    dict(id="conclusiones", section="Resultados", title="Conclusiones",
         body=[("bullets", [f"RFKAN mejora a persistencia en {-pct(r(rf, 'test_RMSE'), r(pers, 'test_RMSE')):.0f} % de RMSE de test, pero LSTM y GRU lo superan.",
                            "La recurrencia es la contribución que se sostiene, igual que en el paper.",
                            "RFKAN es estable en las 4 semillas; el LSTM queda en persistencia en 2 de 4.",
                            "Fourier es más rápido, pero no más preciso que splines: el caudal no es periódico.",
                            "Refinar hasta grilla 100 no ayuda; el refinamiento progresivo con lr decreciente sí es necesario.",
                            "Trabajo futuro: pérdidas que pesen los picos, usar y_aux y bases locales (wavelets)."])],
         notes="En resumen: la idea de agregar memoria recurrente a una KAN funciona y se transfiere al caudal. "
               "La idea de Fourier no, porque su justificación era la periodicidad de la señal solar. Y el "
               "problema abierto son los picos de crecida."),
]

LABEL = {"Metodologia": "Metodología", "Implementacion": "Implementación", "Resultados": "Resultados"}

# Guion completo (Metodologia e Implementacion). Reemplaza las notas cortas de arriba.
NOTES = {
    "cover": "Hola. En este video presentamos nuestra implementación del paper Recurrent Fourier-Kolmogorov "
             "Arnold Networks, de Rong, Lin y Xie, publicado en Scientific Reports en 2025. El paper propone "
             "un modelo para pronosticar la potencia de una planta solar. Nosotros lo implementamos desde cero "
             "y lo adaptamos al problema del laboratorio: pronosticar el caudal de ríos. Vamos a ver la "
             "metodología, la implementación y los resultados.",
    "problema": "Empecemos por el problema. Cada muestra del dataset es una ventana de 336 horas, es decir, 14 "
                "días, de una cuenca. En cada hora hay 12 variables: 11 meteorológicas, como la precipitación, "
                "la temperatura, la radiación o el viento, y el caudal. A partir de eso hay que predecir el "
                "caudal de las siguientes 48 horas, en milímetros por hora, que es el caudal dividido por el "
                "área de la cuenca. Hay 508 cuencas anónimas, y las mismas cuencas aparecen en train, "
                "validación y test. Una propiedad importante: el caudal cambia poco en 48 horas la mayor parte "
                "del tiempo. Repetir el último valor, lo que llamamos persistencia, ya es un buen modelo base. "
                "Lo difícil son los eventos de lluvia.",
    "paper": "El paper resuelve un problema parecido en otro dominio: pronosticar para el día siguiente la "
             "potencia de una planta fotovoltaica de 45 kilowatts en China, con datos cada 5 minutos durante "
             "un año. Su propuesta, RFKAN, parte de las redes de Kolmogorov-Arnold, o KAN, y les hace dos "
             "cambios. El primero son los nodos kernel recurrentes, una memoria tipo RNN para capturar la "
             "dependencia temporal. El segundo es reemplazar los splines de la KAN por series de Fourier, con "
             "el argumento de que la potencia solar es muy periódica y Fourier la representa con pocos "
             "parámetros, y además entrena más rápido. El entrenamiento usa refinamiento de grilla: empieza "
             "con una grilla gruesa y la va refinando en seis etapas, de 3 a 100.",
    "kan": "¿Qué es una KAN? Se basa en el teorema de Kolmogorov-Arnold, que dice que cualquier función "
           "continua de muchas variables se puede escribir usando solo sumas y funciones de una variable. La "
           "fórmula tiene dos niveles: funciones internas phi, una por cada par de entrada y término, y "
           "funciones externas Phi. Una KAN convierte eso en una red de dos capas. La diferencia con un "
           "perceptrón multicapa es dónde está la no linealidad. En un MLP las aristas tienen pesos y la "
           "activación es fija en los nodos. En una KAN, cada arista tiene su propia función aprendible de "
           "una variable, y los nodos solo suman. En nuestro caso, con "
           "12 canales de entrada, la red queda de 12 a 25 nodos y de 25 a 48 salidas, una por hora.",
    "recurrencia": "El primer aporte del paper son los nodos kernel recurrentes, ecuación 3. La memoria h en "
                   "el tiempo t es una combinación lineal de la memoria en t menos 1 y de la entrada en t. Es "
                   "una RNN sin función de activación. Esa memoria resume toda la historia anterior. La "
                   "ecuación 4 dice que cada función de la capa recibe ahora dos cosas: el valor actual del "
                   "nodo y su memoria. El paper no explica cómo una función de una variable recibe dos "
                   "entradas. Nosotros lo implementamos como muestra la figura 1: los nodos recurrentes entran "
                   "en paralelo con los nodos normales, así que la capa suma una función del valor y otra de "
                   "la memoria.",
    "fourier": "El segundo aporte es la ecuación 8. En lugar de un spline, cada función de arista es una serie "
               "de Fourier: una suma de cosenos y senos de frecuencias 1, 2, hasta g, con coeficientes "
               "aprendibles a y b. Aquí g es el tamaño de la grilla, que pasa a ser el número de armónicos. "
               "Refinar la grilla significa agregar armónicos: los coeficientes existentes se copian y los "
               "nuevos arrancan en cero, así que la función aprendida se hereda exactamente. Retengan el "
               "argumento del paper para usar Fourier: la periodicidad de la potencia solar. Vamos a volver "
               "a él en los resultados, porque el caudal no tiene ese ciclo.",
    "paper_res": "Esta es la ablación del paper, la tabla 3. Comparan cuatro modelos: KAN, la base; RKAN, con "
                 "recurrencia y splines; FKAN, con Fourier y sin recurrencia; y RFKAN, con las dos cosas. Cada "
                 "cambio mejora a KAN y juntos mejoran más: el RMSE baja un 37 por ciento. Reportan además grilla óptima 100, largo "
                 "óptimo 2, y que RFKAN supera a todos los baselines. Encontramos algunas inconsistencias. La tabla de "
                 "hiperparámetros dice largo de entrada 12, pero el texto dice que el óptimo es 2. La tabla 5 "
                 "da los parámetros de un LSTM, pero la tabla 6 reporta un GRU, así que corrimos ambos. Y la "
                 "métrica que llaman CORR es en realidad el coeficiente de determinación, R cuadrado.",
    "adaptacion": "Pasamos a la implementación. Esta tabla resume el cambio de dominio. El objetivo pasa de "
                  "potencia a caudal, y de una planta a 508 cuencas. La salida del paper es un único nodo; "
                  "nosotros necesitamos 48 horas, así que la capa 2 tiene 48 nodos de salida. El largo de "
                  "entrada del paper es 2; como la tabla y el texto no coinciden, lo tratamos como "
                  "hiperparámetro y barrimos de 2 a 336 horas, igual que la figura 5 del paper. El modelo "
                  "principal usa 72 horas. La evaluación también cambia mucho: el paper evalúa un día por "
                  "estación, unos 288 puntos; nosotros evaluamos casi 28 mil ventanas de 48 horas. La "
                  "normalización es la del paper, min-max con estadísticos de entrenamiento.",
    "arquitectura": "Esta es la arquitectura que implementamos. La entrada son L pasos con 12 canales. Se le "
                    "agregan los nodos recurrentes, y juntos entran a la capa 1, una capa KAN de Fourier de 24 "
                    "a 25 nodos que se aplica en cada paso de tiempo. Sobre esa salida se calculan otra vez "
                    "los nodos recurrentes, y en el último paso los dos entran a la capa 2, de 50 a 48. La "
                    "salida se suma al último caudal observado, algo que explicamos en el próximo slide. Todo "
                    "está escrito desde cero, como pide la consigna: la capa Fourier, la capa con B-splines "
                    "usando la recursión de Cox-de Boor y la extensión de grilla por mínimos cuadrados.",
    "divergencia": "El primer problema serio fue que la versión literal divergía. Al refinar la grilla a 50 o "
                   "100 armónicos, el error de validación subía de 0.27 a 3.6 milímetros por hora. Lo "
                   "diagnosticamos así: en escala normalizada, el cambio típico del caudal en 48 horas tiene "
                   "un desvío de apenas 0.0046. Y Adam mueve cada parámetro nuevo del orden de la tasa de "
                   "aprendizaje por paso, sin importar el tamaño del gradiente. Con cien armónicos por arista, "
                   "eso es ruido enorme frente a 0.0046. Lo resolvimos con dos cambios. Primero, una salida "
                   "residual: el modelo predice la corrección sobre el último caudal, escalada por ese desvío. "
                   "Segundo, bajamos la tasa de aprendizaje en cada etapa, proporcional a 3 sobre g. La "
                   "salida residual se aplica igual a todos los modelos neuronales, así que la comparación "
                   "sigue siendo justa.",
    "eps": "El segundo problema lo descubrimos al ver que el LSTM daba exactamente el mismo error que "
           "persistencia, hasta el cuarto decimal. Medimos los gradientes: la pérdida en escala normalizada "
           "vale del orden de 2.5 por 10 a la menos 5, y el 99 por ciento de los gradientes del LSTM, el GRU y "
           "la TCN quedaban por debajo de 10 a la menos 8, que es el épsilon de Adam. Con eso, Adam "
           "prácticamente no mueve los pesos. La solución fue dividir la pérdida por sigma al cuadrado: el "
           "óptimo es el mismo, pero los gradientes quedan en una escala útil. Repetimos todos los "
           "experimentos con esta corrección. También vimos que el LSTM con 336 pasos no aprende en el "
           "presupuesto de 18 épocas, así que los modelos base neuronales usan la misma ventana de 72 horas "
           "que RFKAN.",
    "experimentos": "Estos son los experimentos. La suite seqlen es la ablación de la tabla 3 repetida para "
                    "seis largos de ventana, lo que además reproduce la figura 5. La suite grid es la figura "
                    "4: comparamos el refinamiento estándar con una grilla que llega a 200, grillas fijas en 3 "
                    "y en 100, y un refinamiento sin reducir la tasa de aprendizaje. La suite baselines tiene "
                    "los modelos de la tabla 5, más persistencia, que es el modelo base natural en hidrología. "
                    "Y la suite seeds repite RFKAN y el LSTM con tres semillas más, algo que el paper no hace. "
                    "Todos con la misma partición, semilla y presupuesto de 18 épocas. Y la suite advanced "
                    "entrena los tres modelos avanzados de la tabla 7, implementados por nosotros.",
}

_lstm_seeds = [a for a in res.index if a == "LSTM" or a.startswith("LSTM_seed")]
_stuck = [a for a in _lstm_seeds if r(a, "val_RMSE") >= 0.99 * r(pers, "val_RMSE")]
NOTES.update({
    "ablacion": f"Pasamos a los resultados. Esta es nuestra versión de la tabla 3, con ventana de 72 horas, "
                f"evaluada sobre las casi 28 mil ventanas de test. La fila de abajo es persistencia: "
                f"{f4(r(pers, 'test_RMSE'))} de RMSE. RFKAN obtiene {f4(r(rf, 'test_RMSE'))}, una mejora de "
                f"{-pct(r(rf, 'test_RMSE'), r(pers, 'test_RMSE')):.0f} por ciento. Ahora, ¿qué componente aporta? "
                f"Agregar recurrencia a la versión Fourier mejora el RMSE un "
                f"{-pct(r(rf, 'test_RMSE'), r(fk, 'test_RMSE')):.0f} por ciento; agregarla a la versión con splines, "
                f"un {-pct(r(rk, 'test_RMSE'), r(kn, 'test_RMSE')):.0f}. Eso coincide con el paper. Pero cambiar "
                f"splines por Fourier, de RKAN a RFKAN, empeora un {pct(r(rf, 'test_RMSE'), r(rk, 'test_RMSE')):.1f} "
                f"por ciento. El paper reporta una mejora del 22 por ciento. Lo que sí se reproduce es la velocidad: "
                f"RFKAN entrena en {r(rf, 'train_time_s') / r(rk, 'train_time_s'):.2f} veces el tiempo de RKAN. "
                f"El mejor modelo de la familia es RKAN, con splines y recurrencia.",
    "seqlen": "Esta es nuestra versión de la figura 5: el error de validación para ventanas de 2 a 336 horas. "
              "Las variantes sin recurrencia, que reciben la ventana aplanada, empeoran a medida que la ventana "
              "crece: con 168 horas, FKAN ya queda igual que persistencia. RKAN y RFKAN, en cambio, son casi "
              "planos: varían menos de tres milésimas entre 2 y 336 horas. Los modelos base, en gris, se "
              "comportan distinto: con 2 horas todos empatan, pero el LSTM y el GRU mejoran hasta 24 o 72 horas. "
              "La memoria lineal de los nodos recurrentes no aprovecha la historia larga, y las compuertas sí. El "
              "paper encuentra que el mejor largo es 2; en nuestros datos eso solo vale para las variantes sin "
              "recurrencia.",
    "grilla": "Esta es la ablación de la grilla, análoga a la figura 4. El paper dice que la grilla óptima es "
              "100. En nuestros datos no: el mejor punto de RFKAN está en grilla 20, extender el refinamiento "
              "hasta 200 da exactamente el mismo resultado, y una grilla fija en 3 queda apenas un 2 por ciento "
              "peor. En los 20 brazos del barrido de ventana, ningún modelo competitivo mejora al llegar a 100. "
              "Sin embargo, la forma de llegar a una grilla fina sí importa. Si se arranca directamente con 100 "
              "armónicos, o si se refina sin bajar la tasa de aprendizaje, el modelo termina peor que "
              "persistencia. El refinamiento progresivo con tasa decreciente funciona como un esquema de "
              "regularización: no mejora la precisión, pero permite usar grillas finas sin romper el modelo.",
    "baselines": f"Esta es la comparación con los modelos base de la tabla 5. Aquí el resultado es el contrario "
                 f"al del paper. Con el mismo presupuesto, la misma ventana y la misma salida residual, el LSTM "
                 f"obtiene {f4(r('LSTM', 'test_RMSE'))} y el GRU {f4(r('GRU', 'test_RMSE'))}, mejor que toda la "
                 f"familia KAN, y la TCN empata con RKAN. El paper reportaba que RFKAN mejoraba el RMSE en más del "
                 f"58 por ciento frente a GRU y TCN. Los modelos clásicos sí quedan por debajo de RFKAN. ARIMA es "
                 f"interesante: en la cuenca típica es el mejor, con el NSE mediano más alto, pero en algunas "
                 f"ventanas extrapola una tendencia y explota, así que su RMSE global es enorme. El SVR con el C del "
                 f"paper está tan regularizado que queda en persistencia, y el ELM sobreajusta.",
    "semillas": f"Repetimos RFKAN y el LSTM con tres semillas más. RFKAN es muy estable: las cuatro semillas "
                f"quedan dentro de una milésima y media. El LSTM, en cambio, no sale de persistencia en "
                f"{len(_stuck)} de sus {len(_lstm_seeds)} semillas: su error queda clavado durante las 18 épocas. "
                f"Cuando entrena, es el mejor modelo; en promedio sobre semillas, RFKAN lo supera. Este es un punto "
                f"a favor de RFKAN que el paper no mide. A la derecha, el error por hora de horizonte: crece con "
                f"el horizonte, porque las primeras horas son casi persistencia y las últimas dependen de lluvia "
                f"futura que el modelo no ve.",
    "picos": "El análisis de errores muestra dónde está la ganancia y dónde está la limitación. Casi toda la "
             "mejora sobre persistencia ocurre con caudal alto y después de lluvias fuertes, donde RFKAN reduce "
             "el error un 25 por ciento y el LSTM un 30. Con caudal bajo, nadie le gana a persistencia, y el "
             "error es mayor en los meses fríos. La limitación principal son los picos: en el 1 por ciento de "
             "caudales más altos, todos los modelos entrenados subestiman el pico entre un 32 y un 39 por ciento. "
             "En estos tres ejemplos se ve: ningún modelo anticipa la crecida. Y en el panel de caudal bajo se ve "
             "otra cosa: la familia KAN oscila hora a hora, RFKAN más que ninguno, mientras que el LSTM y la TCN "
             "dan curvas suaves.",
    "conclusiones": "Para cerrar. Implementamos RFKAN desde cero y lo adaptamos al caudal: mejora a persistencia "
                    "un 11 por ciento y a los modelos clásicos. De sus dos contribuciones, la recurrencia se "
                    "sostiene, igual que en el paper. Fourier no: entrena más rápido, pero es menos preciso que los "
                    "splines, porque su justificación era la periodicidad de la señal solar y el caudal no la "
                    "tiene. Refinar la grilla hasta 100 no ayuda, aunque el esquema progresivo es necesario. Y en "
                    "igualdad de condiciones, el LSTM y el GRU superan a RFKAN, aunque el LSTM falla en la mitad de "
                    "las semillas. Como trabajo futuro: pérdidas que den más peso a los picos, usar la meteorología "
                    "futura como supervisión auxiliar y bases de funciones locales en lugar de Fourier. Gracias.",
})
for _s in SLIDES:
    _s["notes"] = NOTES.get(_s["id"], _s["notes"])
