"""
Configuracion central del Laboratorio 2 -- RFKAN para pronostico de caudal.

Paper: Rong, Lin & Xie (2025). "Recurrent Fourier-Kolmogorov Arnold Networks
for photovoltaic power forecasting". Scientific Reports 15:4684.
DOI 10.1038/s41598-025-88959-5

Los valores marcados [paper] vienen literalmente de la Tabla 4 / Tabla 5.
Los marcados [adapt] son adaptaciones al dataset Rainfall-Runoff (justificadas
en docs/GAP_ANALISIS.md).
"""
from pathlib import Path

# --- Rutas -------------------------------------------------------------------
LAB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR / "data"
TRAIN_H5 = DATA_DIR / "train.h5"
TEST_H5 = DATA_DIR / "test.h5"
TEST_TARGETS = DATA_DIR / "test_targets.csv"
METADATA = DATA_DIR / "metadata.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# --- Datos -------------------------------------------------------------------
HISTORY_HOURS = 336            # ventana de entrada del dataset (14 dias)
FORECAST_HOURS = 48            # horizonte a predecir
N_CHANNELS = 12                # 11 meteo + caudal
TARGET_CHANNEL = 11            # specific_discharge (mm/h)

# --- RFKAN (Tabla 4) ---------------------------------------------------------
GRID_SCHEDULE = [3, 5, 10, 20, 50, 100]   # [paper] refinamiento de grilla, tope 100
SPLINE_ORDER = 3                          # [paper] K=3 (solo aplica a la variante spline / KAN)
INPUT_NODES = 12                          # [paper] = 12 canales del dataset (coincide)
RECURRENT_NODES = 12                      # [paper] nodos kernel recurrentes = nodos de entrada
SEQ_LEN = 72                              # [adapt] paper usa 12 pasos de 5 min (Tabla 4);
                                          #   aca se elige por ablacion (Fig. 5 analoga)
STAGE_EPOCHS = 3                          # [adapt] "50 steps por grilla" -> 3 epocas por grilla
                                          #   (254k muestras; ver GAP_ANALISIS.md)
LR_GRID_DECAY = 1.0                       # [adapt] lr de cada etapa = lr * (g0 / g)^LR_GRID_DECAY.
                                          #   Sin esto la grilla fina diverge (Adam mueve cada
                                          #   armonico nuevo ~lr por paso -> ruido de alta frecuencia)
RESIDUAL_OUTPUT = True                    # [adapt] salida = ultimo caudal observado + delta*sigma_delta
                                          #   (persistencia como punto de partida; ver GAP_ANALISIS.md)

# --- Entrenamiento -----------------------------------------------------------
BATCH_SIZE = 256               # [adapt] familia KAN (paper no lo reporta para RFKAN)
LEARNING_RATE = 1e-3           # [paper] Tabla 5 (Adam, lr 1e-3)
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
SEED = 42
NUM_WORKERS = 0                # los tensores viven en GPU, no hace falta DataLoader

# --- Baselines (Tabla 5) -----------------------------------------------------
BASELINE_BATCH_SIZE = 32       # [paper] LSTM: batch 32, Adam, lr 1e-3, 3 capas, ReLU
BASELINE_HIDDEN = 64
ARIMA_ORDER = (2, 1, 1)        # [paper] AR=2, d=1, MA=1
SVR_C = 0.0043                 # [paper] RBF, C=0.0043
ELM_HIDDEN_LAYERS = 3          # [paper] 3 capas ocultas, ReLU, reg 0.001
ELM_REG = 1e-3
TCN_LAYERS = 3                 # [paper] 3 capas conv, stride 2, padding same, ReLU

# --- Modelo avanzado (Tabla 7): Informer, hiperparametros de Zhou et al. 2021 --
INFORMER_SEQ_LEN = 96          # [lit] ventana del encoder (ETTh, horizonte 48)
INFORMER_LABEL_LEN = 48        # [lit] horas conocidas al inicio del decoder
INFORMER_D_MODEL = 512
INFORMER_HEADS = 8
INFORMER_E_LAYERS = 2
INFORMER_D_LAYERS = 1
INFORMER_D_FF = 2048
INFORMER_FACTOR = 5            # [lit] u = factor * ln(L) consultas activas
INFORMER_DROPOUT = 0.05
INFORMER_LR = 1e-4             # [lit] Adam lr 1e-4, batch 32

# --- Modelo avanzado (Tabla 7): WVS = WOA-VMD-SCINet (Zhao et al. 2024) --------
WVS_SEQ_LEN = 96               # [lit] ventana de SCINet (ETTh, horizonte 48); potencia de 2 x 3 niveles
WVS_N_FEATURES = 3             # [paper] entrada de dimension 4 = 1 subsecuencia + 3 variables (Pearson)
WVS_VMD_ITERS = 100            # [adapt] iteraciones fijas de VMD (en lotes, sin criterio de parada)
WVS_WOA_POP = 10               # [adapt] ballenas
WVS_WOA_ITERS = 15             # [adapt] iteraciones de WOA
WVS_LEVELS = 3                 # [paper] Tabla 5 / Sec. 3.2: levels 3, stacks 1, hidden 1, dropout 0.5
WVS_HIDDEN = 1
WVS_KERNEL = 4                 # [paper] texto de la Sec. 3.2 (la Tabla 5 dice 1)
WVS_DROPOUT = 0.5
WVS_LR = 3e-3                  # [lit] no reportado en el paper; repo de SCINet (ETTh1)

# --- Modelo avanzado (Tabla 7): TACDPG (Zhang et al. 2024) ------------------
TACDPG_N_FEATURES = 6          # [adapt] canales elegidos por random forest (+ caudal)
TACDPG_KAPPA = 1.5             # [paper] Ec. 1: outlier si |y - mu| > 1.5 sigma
TACDPG_SEQ_LEN = 72            # [adapt] misma ventana que RFKAN
TACDPG_WIDTH = 64
TACDPG_ACTION_MAX = 5.0        # accion acotada: |correccion| <= 5 sigma_delta
TACDPG_ENVS = 256              # episodios en paralelo (= batch de interaccion)
TACDPG_EPISODE_LEN = 64        # pasos (ventanas consecutivas) por episodio
TACDPG_BATCH = 256             # minibatch del buffer
TACDPG_BUFFER = 200_000
# Los siguientes estan en el material suplementario del paper (no disponible):
TACDPG_Z0, TACDPG_Z1 = 1.0, 0.1   # [supuesto] pesos de la recompensa al inicio / al final (Ec. 2)
TACDPG_LAMBDA = 0.01           # [supuesto] decaimiento exponencial por paso (Ec. 2)
TACDPG_GAMMA = 0.9             # [supuesto] descuento
TACDPG_TAU = 0.005             # [supuesto; TD3] actualizacion suave
TACDPG_LR_ACTOR = 1e-4         # [supuesto; TD3/DDPG] actor 1e-4, critico 1e-3
TACDPG_LR_CRITIC = 1e-3
TACDPG_NOISE = 0.5             # [supuesto] ruido de exploracion inicial (decae a 0.05)
