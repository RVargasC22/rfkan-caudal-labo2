"""
Predicciones para TODAS las muestras de test.h5 (requisito de la consigna).

Salida: outputs/experiments/predictions_test_<arm>.csv con el mismo formato que
test_targets.csv: Id, q_01 .. q_48 (caudal en mm/h). Id = indice de fila en test.h5.
Si test_targets.csv existe, imprime las metricas de test.
"""
import argparse

import numpy as np
import pandas as pd

import config
from dataset import get_data
from metrics import all_metrics
from model import build_forecaster
from train import load_best, predict

OUT = config.OUTPUT_DIR / "experiments"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=f"RFKAN_L{config.SEQ_LEN}")
    ap.add_argument("--model", default=None, help="por defecto se infiere del nombre del brazo")
    ap.add_argument("--seq_len", type=int, default=None)
    a = ap.parse_args()
    res = pd.read_csv(OUT / "experiments_results.csv").set_index("arm")
    name = a.model or res.loc[a.arm, "model"]
    L = a.seq_len or int(res.loc[a.arm, "seq_len"])
    _, _, test, scaler, _ = get_data()
    model = load_best(build_forecaster(name, scaler.delta_std, seq_len=L), a.arm, OUT)
    p = np.clip(predict(model, test, scaler), 0.0, None)   # caudal especifico >= 0
    cols = [f"q_{h:02d}" for h in range(1, config.FORECAST_HOURS + 1)]
    df = pd.DataFrame(p, columns=cols)
    df.insert(0, "Id", np.arange(len(p)))
    path = OUT / f"predictions_test_{a.arm}.csv"
    df.to_csv(path, index=False, float_format="%.9g")
    print(f"{len(df)} filas -> {path}")
    if test.y is not None:
        m = all_metrics(test.y_raw, p, test.basin_id)
        print("test:", {k: round(v, 4) for k, v in m.items()})


if __name__ == "__main__":
    main()
