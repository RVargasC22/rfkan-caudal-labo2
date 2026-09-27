"""
Todos los experimentos del laboratorio, mismo split / semilla / presupuesto.

  --suite seqlen   : Fig. 5 + Tabla 3 del paper. KAN / RKAN / FKAN / RFKAN para
                     varios largos de ventana de entrada L (horas).
  --suite grid     : Fig. 4 del paper + ablacion del refinamiento de grilla.
                       RFKAN_grid_ext   : schedule extendido [3..100, 200] (500 no entra en 16 GB con L=72)
                       RFKAN_fixed_g3   : grilla 3 fija, sin refinamiento
                       RFKAN_fixed_g100 : grilla 100 fija desde el inicio
                       RFKAN_nodecay    : refinamiento sin bajar el lr por etapa
  --suite baselines: Tabla 5/6 del paper: persistencia, ARIMA, SVR, ELM, LSTM, GRU, TCN.
  --suite seeds    : RFKAN (L elegido) y el mejor baseline con 3 semillas.
  --suite advanced : Tabla 7: Informer, WVS (WOA-VMD-SCINet) y TACDPG (implementaciones propias).
  --suite seqlen_base: Fig. 5 completa: LSTM / GRU / TCN para L en {2, 12, 24, 168}.

Cada brazo guarda <arm>_history.csv, <arm>_best.pt y <arm>_preds.npz
(predicciones val + test en mm/h) en outputs/experiments/, agrega una fila a
experiments_results.csv (val y test: RMSE, MAE, CORR, KGE, NSE mediana, tiempo)
y vuelca todo el detalle en pkl/<arm>.pkl (ver pkl_store.py); al final de cada
suite se regenera pkl/ALL.pkl.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

import config
import baselines as B
from dataset import get_data
from metrics import all_metrics
from model import build_forecaster
from pkl_store import dump_all, dump_arm
from train import fit, load_best, predict, set_seed

OUT = config.OUTPUT_DIR / "experiments"
RESULTS = OUT / "experiments_results.csv"
KAN_FAMILY = ("KAN", "RKAN", "FKAN", "RFKAN")


def record(arm, suite, pv, pt, train_time, extra=None):
    _, val, test, _, _ = get_data()
    row = {"arm": arm, "suite": suite, "train_time_s": train_time, **(extra or {})}
    row.update({f"val_{k}": v for k, v in all_metrics(val.y_raw, pv, val.basin_id).items()})
    if pt is not None and test.y is not None:
        row.update({f"test_{k}": v for k, v in all_metrics(test.y_raw, pt, test.basin_id).items()})
    np.savez_compressed(OUT / f"{arm}_preds.npz", val=pv, test=pt if pt is not None else np.empty(0))
    df = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
    df = df[df.get("arm", pd.Series(dtype=str)) != arm] if len(df) else df
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS, index=False)
    dump_arm(arm)
    print(f"==> {arm}: val RMSE {row['val_RMSE']:.4f} CORR {row['val_CORR']:.4f} | "
          f"test RMSE {row.get('test_RMSE', float('nan')):.4f} ({train_time:.0f}s)", flush=True)
    return row


def done(arm):
    return RESULTS.exists() and arm in set(pd.read_csv(RESULTS)["arm"])


def run_neural(arm, suite, name, seq_len, grid_schedule=None, stage_epochs=config.STAGE_EPOCHS,
               epochs=None, batch_size=config.BATCH_SIZE, seed=config.SEED, lr_decay=None, fixed_grid=None,
               lr=config.LEARNING_RATE):
    if done(arm):
        print(f"skip {arm} (ya en resultados)")
        return
    train, val, test, scaler, _ = get_data()
    set_seed(seed)
    kw = {"grid": fixed_grid} if fixed_grid else {}
    model = build_forecaster(name, scaler.delta_std, seq_len=seq_len, **kw)
    old = config.LR_GRID_DECAY
    if lr_decay is not None:
        config.LR_GRID_DECAY = lr_decay
    t = time.time()
    try:
        hist = fit(model, arm, OUT, epochs, batch_size, lr, grid_schedule=grid_schedule, stage_epochs=stage_epochs)
    except torch.OutOfMemoryError:
        # solo RKAN con L=336 puede no entrar en 16 GB: se reintenta con la mitad de batch
        del model
        torch.cuda.empty_cache()
        set_seed(seed)
        batch_size //= 2
        model = build_forecaster(name, scaler.delta_std, seq_len=seq_len, **kw)
        t = time.time()
        hist = fit(model, arm, OUT, epochs, batch_size, lr, grid_schedule=grid_schedule, stage_epochs=stage_epochs)
    tt = time.time() - t
    config.LR_GRID_DECAY = old
    model = load_best(build_forecaster(name, scaler.delta_std, seq_len=seq_len, **kw), arm, OUT)
    n_params = sum(p.numel() for p in model.parameters())
    best_ep = int(pd.DataFrame(hist)["val_RMSE"].idxmin()) + 1
    record(arm, suite, predict(model, val, scaler), predict(model, test, scaler), tt,
           {"model": name, "seq_len": seq_len, "seed": seed, "params": n_params,
            "epochs": len(hist), "best_epoch": best_ep, "best_grid": hist[best_ep - 1]["grid"]})
    del model
    torch.cuda.empty_cache()


def kan(arm, suite, name, seq_len, **kw):
    run_neural(arm, suite, name, seq_len, grid_schedule=kw.pop("grid_schedule", config.GRID_SCHEDULE), **kw)


def suite_seqlen(lengths):
    for L in lengths:
        for name in KAN_FAMILY:
            kan(f"{name}_L{L}", "seqlen", name, L)


def suite_grid(L):
    kan(f"RFKAN_L{L}_grid_ext", "grid", "RFKAN", L, grid_schedule=config.GRID_SCHEDULE + [200])
    kan(f"RFKAN_L{L}_fixed_g3", "grid", "RFKAN", L, grid_schedule=[3] * 6)
    kan(f"RFKAN_L{L}_fixed_g100", "grid", "RFKAN", L, grid_schedule=[100] * 6, fixed_grid=100)
    kan(f"RFKAN_L{L}_nodecay", "grid", "RFKAN", L, lr_decay=0.0)


def suite_baselines(n_arima=2000, n_svr=10000):
    train, val, test, scaler, (tr, va, te) = get_data()
    epochs = len(config.GRID_SCHEDULE) * config.STAGE_EPOCHS      # mismo presupuesto que RFKAN
    Xtr, Xva, Xte = (d.X.cpu().numpy() for d in (train, val, test))
    ytr = train.y.cpu().numpy()
    inv = scaler.inverse_y

    if not done("Persistence"):
        record("Persistence", "baselines", inv(B.persistence(Xva)), inv(B.persistence(Xte)), 0.0,
               {"model": "Persistence", "seq_len": 1})

    if not done("ELM"):
        m, tt = B.timed(B.ELMBaseline(scaler.delta_std).fit, Xtr, ytr)
        record("ELM", "baselines", inv(m.predict(Xva)), inv(m.predict(Xte)), tt, {"model": "ELM", "seq_len": 24})

    if not done("SVR"):
        rng = np.random.default_rng(config.SEED)
        idx = rng.choice(len(Xtr), n_svr, replace=False)
        m, tt = B.timed(B.SVRBaseline(scaler.delta_std).fit, Xtr[idx], ytr[idx])
        record("SVR", "baselines", inv(m.predict(Xva)), inv(m.predict(Xte)), tt,
               {"model": "SVR", "seq_len": 24, "n_train": n_svr})

    if not done("ARIMA"):
        # ARIMA no se entrena: se ajusta por muestra. Se evalua en un submuestreo fijo
        # de val y test; las filas no evaluadas quedan con persistencia y se reportan
        # las metricas SOLO sobre el submuestreo (columna n_eval).
        rng = np.random.default_rng(config.SEED)
        iv = np.sort(rng.choice(len(Xva), n_arima, replace=False))
        it = np.sort(rng.choice(len(Xte), n_arima, replace=False))
        t = time.time()
        pv_sub = inv(B.arima_forecast(Xva[iv]))
        pt_sub = inv(B.arima_forecast(Xte[it]))
        tt = time.time() - t
        np.savez_compressed(OUT / "ARIMA_subset_idx.npz", val=iv, test=it)
        row = {"arm": "ARIMA", "suite": "baselines", "train_time_s": tt, "model": "ARIMA",
               "seq_len": 336, "n_eval": n_arima}
        row.update({f"val_{k}": v for k, v in all_metrics(val.y_raw[iv], pv_sub, val.basin_id[iv]).items()})
        row.update({f"test_{k}": v for k, v in all_metrics(test.y_raw[it], pt_sub, test.basin_id[it]).items()})
        # referencia: persistencia sobre el MISMO submuestreo
        row["val_RMSE_persistence_same_subset"] = all_metrics(
            val.y_raw[iv], inv(B.persistence(Xva[iv])))["RMSE"]
        np.savez_compressed(OUT / "ARIMA_preds.npz", val=pv_sub, test=pt_sub)
        df = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
        pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(RESULTS, index=False)
        dump_arm("ARIMA")
        print(f"==> ARIMA (n={n_arima}): val RMSE {row['val_RMSE']:.4f}", flush=True)

    # misma ventana que RFKAN (SEQ_LEN): con L=336 el LSTM no sale de persistencia en 18 epocas
    for name in ("LSTM", "GRU", "TCN"):
        run_neural(name, "baselines", name, config.SEQ_LEN, epochs=epochs,
                   batch_size=config.BASELINE_BATCH_SIZE)


def suite_advanced():
    """Tabla 7: modelos avanzados, implementaciones propias: Informer, WVS (WOA-VMD-SCINet), TACDPG."""
    epochs = len(config.GRID_SCHEDULE) * config.STAGE_EPOCHS
    run_neural("Informer", "advanced", "Informer", config.INFORMER_SEQ_LEN, epochs=epochs,
               batch_size=config.BASELINE_BATCH_SIZE, lr=config.INFORMER_LR)
    run_wvs(epochs)
    run_tacdpg(epochs)


def _train_sample(train, n, seed=config.SEED):
    idx = np.sort(np.random.default_rng(seed).choice(train.n, n, replace=False))
    t = torch.as_tensor(idx, device=train.X.device)
    return idx, train.X[t].cpu().numpy(), train.y[t].cpu().numpy()


def run_wvs(epochs):
    import wvs
    if done("WVS"):
        print("skip WVS (ya en resultados)")
        return
    train, val, test, scaler, _ = get_data()
    t = time.time()
    _, Xs, _ = _train_sample(train, 20000)
    p = wvs.fit_params(Xs, scaler.delta_std)                         # Pearson + WOA(K, alpha)
    print(f"WVS: K={p['K']} alpha={p['alpha']:.0f} variables={p['features']} "
          f"entropia={p['woa_fitness']:.3f} ({time.time() - t:.0f}s)", flush=True)
    tr, va, te = (wvs.with_modes(d, scaler.delta_std) for d in (train, val, test))
    set_seed(config.SEED)
    t = time.time()
    hist = fit(build_forecaster("WVS", scaler.delta_std), "WVS", OUT, epochs, config.BASELINE_BATCH_SIZE,
               config.WVS_LR, data=(tr, va))
    tt = time.time() - t
    model = load_best(build_forecaster("WVS", scaler.delta_std), "WVS", OUT)
    best = int(pd.DataFrame(hist)["val_RMSE"].idxmin()) + 1
    record("WVS", "advanced", predict(model, va, scaler), predict(model, te, scaler), tt,
           {"model": "WVS", "seq_len": config.WVS_SEQ_LEN, "seed": config.SEED, "epochs": len(hist),
            "params": sum(q.numel() for q in model.parameters()), "best_epoch": best})
    del tr, va, te, model
    torch.cuda.empty_cache()


def run_tacdpg(epochs):
    import tacdpg
    if done("TACDPG"):
        print("skip TACDPG (ya en resultados)")
        return
    train, val, test, scaler, (tr, va, te) = get_data()
    t = time.time()
    idx, Xs, Ys = _train_sample(train, 50000)
    dY = (Ys - Xs[:, -1:, config.TARGET_CHANNEL]) / scaler.delta_std
    sel, imp = tacdpg.select_features(Xs, dY)                         # random forest
    mu, sd = tacdpg.basin_stats(tr["y"], tr["basin_id"])              # Ec. 1 por cuenca, en mm/h
    lab_tr = tacdpg.outlier_labels(tr["y"], tr["basin_id"], mu, sd)
    W = [d.X[:, -24:, sel].cpu().numpy() for d in (train, val, test)]
    c_tr, c_va, c_te = tacdpg.catboost_classes(W[0][idx], lab_tr[idx], W)
    del W
    lab_va = tacdpg.outlier_labels(va["y"], va["basin_id"], mu, sd)
    acc = float((c_va == lab_va).mean())
    print(f"TACDPG: variables={sel} outliers_train={lab_tr.mean():.3f} CatBoost_acc_val={acc:.3f} "
          f"({time.time() - t:.0f}s)", flush=True)
    model = tacdpg.TACDPG(sel).to(train.X.device)
    torch.manual_seed(config.SEED)
    t, hist = time.time(), []
    for name, agent, cls in (("normal", model.normal, 0), ("outlier", model.outlier, 1)):
        rows = np.flatnonzero(lab_tr == cls)                          # cada agente se entrena con su clase
        vrows = torch.as_tensor(np.flatnonzero(c_va == cls), device=train.X.device)

        def val_fn(agent=agent, vrows=vrows):
            with torch.no_grad():
                agent.eval()
                x = val.X[vrows]
                p = x[:, -1:, config.TARGET_CHANNEL] + scaler.delta_std * agent.actor(model.state(x))
                agent.train()
            idx_np = vrows.cpu().numpy()
            return all_metrics(val.y_raw[idx_np], scaler.inverse_y(p.cpu().numpy()), val.basin_id[idx_np])

        hist += tacdpg.train_agent(agent, model, train, rows, scaler, epochs, name, val_fn,
                                   log=lambda m: print(m, flush=True))
    tt = time.time() - t
    pd.DataFrame(hist).to_csv(OUT / "TACDPG_history.csv", index=False)
    torch.save(model.state_dict(), OUT / "TACDPG_best.pt")
    record("TACDPG", "advanced", tacdpg.predict_tacdpg(model, val, c_va, scaler),
           tacdpg.predict_tacdpg(model, test, c_te, scaler), tt,
           {"model": "TACDPG", "seq_len": config.TACDPG_SEQ_LEN, "seed": config.SEED,
            "params": sum(q.numel() for q in model.normal.actor.parameters()) * 2,
            "catboost_acc_val": acc, "outlier_frac_train": float(lab_tr.mean())})
    del model
    torch.cuda.empty_cache()


def suite_seqlen_base(lengths=(2, 12, 24, 168)):
    """Fig. 5 completa: los baselines neuronales tambien para varios largos de ventana."""
    epochs = len(config.GRID_SCHEDULE) * config.STAGE_EPOCHS
    for L in lengths:
        for name in ("LSTM", "GRU", "TCN"):
            run_neural(f"{name}_L{L}", "seqlen_base", name, L, epochs=epochs, batch_size=config.BASELINE_BATCH_SIZE)


def suite_seeds(L, seeds=(1, 2, 3), baseline="LSTM"):
    epochs = len(config.GRID_SCHEDULE) * config.STAGE_EPOCHS
    for s in seeds:
        kan(f"RFKAN_L{L}_seed{s}", "seeds", "RFKAN", L, seed=s)
        run_neural(f"{baseline}_seed{s}", "seeds", baseline, config.SEQ_LEN, epochs=epochs,
                   batch_size=config.BASELINE_BATCH_SIZE, seed=s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["seqlen", "grid", "baselines", "seeds", "advanced", "seqlen_base"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[2, 12, 24, 72, 168, 336])
    ap.add_argument("--seq_len", type=int, default=config.SEQ_LEN)
    ap.add_argument("--baseline", default="LSTM")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    get_data()[3].save(OUT / "scaler.json")
    if a.suite == "seqlen":
        suite_seqlen(a.lengths)
    elif a.suite == "grid":
        suite_grid(a.seq_len)
    elif a.suite == "baselines":
        suite_baselines()
    elif a.suite == "advanced":
        suite_advanced()
    elif a.suite == "seqlen_base":
        suite_seqlen_base()
    else:
        suite_seeds(a.seq_len, baseline=a.baseline)
    dump_all()


if __name__ == "__main__":
    main()
