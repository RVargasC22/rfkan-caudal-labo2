"""
Entrenamiento comun para la familia KAN (con refinamiento de grilla) y para
los baselines neuronales (LSTM / GRU / TCN).

Protocolo de grilla (paper, Fig. 4 / Tabla 4): se arranca con grilla 3 y cada
STAGE_EPOCHS se refina -> 3, 5, 10, 20, 50, 100. En cada refinamiento el modelo
hereda la funcion aprendida (extend_grid) y se re-crea el optimizador (los
tensores de coeficientes cambian de tamano).

Guardado incremental por epoca: <run>_history.csv, <run>_last.pt, <run>_best.pt
(mejor RMSE de validacion en mm/h).
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from dataset import get_data
from metrics import all_metrics


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def predict(model, data, scaler, batch_size=512):
    """Predicciones en mm/h (numpy, [N, 48])."""
    model.eval()
    out = [model(xb) for xb, _ in data.batches(batch_size, shuffle=False)]
    return scaler.inverse_y(torch.cat(out).float().cpu().numpy())


def fit(model, run_name, out_dir, epochs, batch_size, lr=config.LEARNING_RATE,
        grid_schedule=None, stage_epochs=None, max_train=None, verbose=True, data=None):
    """Entrena `model`. Si grid_schedule no es None, aplica refinamiento de grilla
    (familia KAN): len(grid_schedule) * stage_epochs epocas en total."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, _, scaler, _ = get_data()
    if data is not None:                    # datos transformados (p. ej. WVS con modos VMD)
        train, val = data
    device = train.X.device
    model = model.to(device)
    if grid_schedule is not None:
        epochs = len(grid_schedule) * stage_epochs
    gen = torch.Generator().manual_seed(config.SEED)
    # MSE en unidades de sigma_delta (desvio del cambio de caudal en 48 h, ~0.0046
    # en escala [0, 1]). Mismo optimo que la MSE normalizada, pero sin esto los
    # gradientes quedan por debajo del eps de Adam (1e-8) y LSTM/TCN no aprenden.
    inv_s2 = 1.0 / scaler.delta_std ** 2
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    history, best, t0 = [], float("inf"), time.time()
    n_train = train.n if max_train is None else min(max_train, train.n)

    for ep in range(epochs):
        if grid_schedule is not None and ep > 0 and ep % stage_epochs == 0:
            g = grid_schedule[ep // stage_epochs]
            sample = train.X[torch.randperm(train.n, generator=gen)[:4096].to(device)]
            model.extend_grid(g, sample)
            stage_lr = lr * (grid_schedule[0] / g) ** config.LR_GRID_DECAY
            opt = torch.optim.Adam(model.parameters(), lr=stage_lr, weight_decay=config.WEIGHT_DECAY)
        model.train()
        te, tot, cnt = time.time(), 0.0, 0
        for i, (xb, yb) in enumerate(train.batches(batch_size, generator=gen)):
            if i * batch_size >= n_train:
                break
            loss = loss_fn(model(xb), yb) * inv_s2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
            tot += loss.item() / inv_s2 * len(xb)
            cnt += len(xb)
        train_time = time.time() - te
        pv = predict(model, val, scaler)
        m = all_metrics(val.y_raw, pv, val.basin_id)
        row = {"epoch": ep + 1, "grid": getattr(model, "grid", None), "train_mse_norm": tot / cnt,
               "train_rmse_mmh": float(np.sqrt(tot / cnt)) * float(scaler.span[config.TARGET_CHANNEL]),
               **{f"val_{k}": v for k, v in m.items()}, "epoch_time_s": train_time,
               "elapsed_s": time.time() - t0}
        if not np.isfinite(row["train_mse_norm"]):
            raise RuntimeError(f"{run_name}: loss no finita en epoca {ep + 1}")
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / f"{run_name}_history.csv", index=False)
        ckpt = {"model": model.state_dict(), "epoch": ep + 1, "grid": row["grid"]}
        torch.save(ckpt, out_dir / f"{run_name}_last.pt")
        if m["RMSE"] < best:
            best = m["RMSE"]
            torch.save(ckpt, out_dir / f"{run_name}_best.pt")
        if verbose:
            print(f"[{run_name}] ep {ep + 1:3d}/{epochs} grid={row['grid']} "
                  f"train_rmse={row['train_rmse_mmh']:.4f} val_RMSE={m['RMSE']:.4f} "
                  f"val_MAE={m['MAE']:.4f} val_CORR={m['CORR']:.4f} "
                  f"NSEmed={m['NSE_median']:.3f} ({train_time:.0f}s)", flush=True)
    return history


def load_best(model, run_name, out_dir, which="best"):
    ck = torch.load(out_dir / f"{run_name}_{which}.pt", map_location="cuda")
    if ck.get("grid"):
        _resize_for_grid(model, ck["grid"])
    model.load_state_dict(ck["model"])
    return model.cuda()


def _resize_for_grid(model, grid):
    """Pone las capas KAN en la grilla `grid` para poder cargar el state_dict."""
    from kan_layers import FourierKANLayer, SplineKANLayer
    for mod in model.modules():
        if isinstance(mod, FourierKANLayer) and mod.grid != grid:
            mod.extend_grid(grid)
        elif isinstance(mod, SplineKANLayer) and mod.grid != grid:
            mod.grid = grid
            mod.knots = torch.zeros(mod.in_dim, grid + 2 * mod.k + 1, device=mod.knots.device)
            mod.spline_w = nn.Parameter(torch.zeros(mod.out_dim, mod.in_dim, grid + mod.k,
                                                    device=mod.base_w.device))


def main():
    from model import build_forecaster
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RFKAN", choices=["KAN", "RKAN", "FKAN", "RFKAN"])
    ap.add_argument("--seq_len", type=int, default=config.SEQ_LEN)
    ap.add_argument("--stage_epochs", type=int, default=config.STAGE_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--out", default="train_run")
    a = ap.parse_args()
    set_seed()
    scaler = get_data()[3]
    model = build_forecaster(a.model, scaler.delta_std, seq_len=a.seq_len)
    name = a.run_name or f"{a.model}_L{a.seq_len}"
    fit(model, name, config.OUTPUT_DIR / a.out, None, a.batch_size,
        grid_schedule=config.GRID_SCHEDULE, stage_epochs=a.stage_epochs, max_train=a.max_train)


if __name__ == "__main__":
    main()
