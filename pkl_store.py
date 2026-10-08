"""
Guarda en .pkl TODO el detalle de cada entrenamiento y sus pronosticos.

  outputs/experiments/pkl/<arm>.pkl   -- un brazo:
      arm, info (fila de experiments_results.csv: config, params, tiempos, metricas),
      history (DataFrame por epoca: grid, loss, metricas val, tiempos),
      val / test: {pred, target, basin_id, metrics, per_horizon, nse_per_basin}
      (pred y target en mm/h, [N, 48]), config (snapshot de config.py), scaler.
  outputs/experiments/pkl/ALL.pkl     -- dict {arm: <contenido de arriba>} + results.

Se llama desde experiments.record() al terminar cada brazo, y tambien a mano
para reconstruir todo desde los archivos en disco:
    ../.venv/bin/python pkl_store.py
"""
import pickle

import numpy as np
import pandas as pd

import config
from metrics import all_metrics, nse_per_basin, per_horizon

OUT = config.OUTPUT_DIR / "experiments"
PKL_DIR = OUT / "pkl"


def _config_snapshot():
    return {k: (str(v) if hasattr(v, "resolve") else v) for k, v in vars(config).items()
            if k.isupper()}


def _split_detail(pred, y, basin, subset=None):
    if pred is None or len(pred) == 0 or y is None:
        return None
    if subset is not None:
        y, basin = y[subset], basin[subset]
    return {"pred": pred.astype(np.float32), "target": y.astype(np.float32), "basin_id": basin,
            "subset_idx": subset, "metrics": all_metrics(y, pred, basin),
            "per_horizon": {k: v.tolist() for k, v in per_horizon(y, pred).items()},
            "nse_per_basin": nse_per_basin(y, pred, basin)}


def dump_arm(arm):
    from dataset import get_data
    _, val, test, scaler, _ = get_data()
    res = pd.read_csv(OUT / "experiments_results.csv")
    info = res[res["arm"] == arm].iloc[-1].to_dict() if arm in set(res["arm"]) else {}
    hist_path = OUT / f"{arm}_history.csv"
    preds = np.load(OUT / f"{arm}_preds.npz")
    sub_v = sub_t = None
    if arm == "ARIMA":                                  # evaluado en submuestreo
        idx = np.load(OUT / "ARIMA_subset_idx.npz")
        sub_v, sub_t = idx["val"], idx["test"]
    d = {"arm": arm, "info": info,
         "history": pd.read_csv(hist_path) if hist_path.exists() else None,
         "val": _split_detail(preds["val"], val.y_raw, val.basin_id, sub_v),
         "test": _split_detail(preds["test"], test.y_raw if test.y is not None else None,
                               test.basin_id, sub_t),
         "config": _config_snapshot(),
         "scaler": {"lo": scaler.lo, "hi": scaler.hi, "delta_std": scaler.delta_std}}
    PKL_DIR.mkdir(parents=True, exist_ok=True)
    with open(PKL_DIR / f"{arm}.pkl", "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def dump_all():
    """Reconstruye todos los .pkl por brazo y el maestro ALL.pkl."""
    res = pd.read_csv(OUT / "experiments_results.csv")
    alld = {"results": res, "arms": {}}
    for arm in res["arm"]:
        if (OUT / f"{arm}_preds.npz").exists():
            alld["arms"][arm] = dump_arm(arm)
            print("pkl:", arm)
    with open(PKL_DIR / "ALL.pkl", "wb") as f:
        pickle.dump(alld, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{len(alld['arms'])} brazos -> {PKL_DIR / 'ALL.pkl'}")


if __name__ == "__main__":
    dump_all()
