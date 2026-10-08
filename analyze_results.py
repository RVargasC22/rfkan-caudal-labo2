"""
Graficos y analisis de errores a partir de los resultados guardados (solo CPU,
no toca la GPU ni reentrena). Lee outputs/experiments/experiments_results.csv,
los *_history.csv y pkl/<arm>.pkl.

Salidas en outputs/analysis/:
  tabla3_ablacion.md      KAN / RKAN / FKAN / RFKAN (analogo a la Tabla 3)
  tabla6_comparacion.md   RFKAN vs baselines en test (analogo a la Tabla 6)
  tabla7_avanzados.md     RFKAN vs Informer (analogo a la Tabla 7)
  tabla_seeds.md          media +- desvio sobre semillas
  tabla_grid.md           ablacion del refinamiento de grilla
  fig5_seqlen.png         RMSE val vs largo de ventana L (analogo a la Fig. 5)
  fig4_grid.png           curvas de val RMSE por epoca con la grilla (analogo a la Fig. 4)
  fig_horizonte.png       RMSE por hora de horizonte
  fig_nse_cuencas.png     CDF del NSE por cuenca
  fig_hidrogramas.png     ejemplos de pronostico en test
  errores_por_regimen.md  RMSE por cuantil de caudal, lluvia reciente y temperatura (estacion)
  picos.md                error en los picos (1% de caudales mas altos)
"""
import pickle

import h5py
import matplotlib
import numpy as np
import pandas as pd

import config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = config.OUTPUT_DIR / "experiments"
OUT = config.OUTPUT_DIR / "analysis"
L = config.SEQ_LEN
MAIN = f"RFKAN_L{L}"
KAN_FAMILY = ("KAN", "RKAN", "FKAN", "RFKAN")
TEMP, PRECIP = 7, 8                               # canales de metadata.json
METRICS = ["RMSE", "MAE", "CORR", "KGE", "NSE_median"]


def load_arm(arm):
    p = EXP / "pkl" / f"{arm}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


def md_table(df, path, title, floatfmt=".4f"):
    with open(path, "w") as f:
        f.write(f"# {title}\n\n{df.to_markdown(floatfmt=floatfmt)}\n")
    print(f"-> {path.name}")


def ablation_table(res):
    rows = res[res["arm"].isin([f"{m}_L{L}" for m in KAN_FAMILY])].set_index("arm")
    cols = [f"{s}_{m}" for s in ("val", "test") for m in METRICS] + ["params", "train_time_s"]
    md_table(rows[cols], OUT / "tabla3_ablacion.md", f"Ablacion (Tabla 3), L={L}")


def comparison_table(res):
    arms = ["Persistence", "ARIMA", "SVR", "ELM", "LSTM", "GRU", "TCN", "Informer"] + [f"{m}_L{L}" for m in KAN_FAMILY]
    rows = res[res["arm"].isin(arms)].set_index("arm").reindex([a for a in arms if a in set(res["arm"])])
    cols = [f"test_{m}" for m in METRICS] + ["train_time_s"]
    md_table(rows[cols], OUT / "tabla6_comparacion.md",
             "Comparacion en test (Tabla 6). ARIMA: submuestreo de 2000 ventanas")


def advanced_table(res):
    arms = [a for a in ("Informer", MAIN, f"RKAN_L{L}", "LSTM") if a in set(res["arm"])]
    if "Informer" in arms:
        cols = [f"test_{m}" for m in METRICS] + ["params", "train_time_s"]
        md_table(res.set_index("arm").loc[arms, cols], OUT / "tabla7_avanzados.md",
                 "Modelo avanzado (Tabla 7): Informer contra RFKAN")


def seeds_table(res):
    groups = {"RFKAN": [MAIN] + [f"{MAIN}_seed{s}" for s in (1, 2, 3)],
              "LSTM": ["LSTM"] + [f"LSTM_seed{s}" for s in (1, 2, 3)]}
    out = []
    for name, arms in groups.items():
        r = res[res["arm"].isin(arms)]
        if len(r) < 2:
            continue
        row = {"modelo": name, "n": len(r)}
        for m in ("val_RMSE", "val_CORR", "test_RMSE", "test_CORR", "test_NSE_median"):
            row[m] = f"{r[m].mean():.4f} ± {r[m].std():.4f}"
        out.append(row)
    if out:
        md_table(pd.DataFrame(out).set_index("modelo"), OUT / "tabla_seeds.md", "Robustez a la semilla")


def grid_table(res):
    rows = res[(res["suite"] == "grid") | (res["arm"] == MAIN)].set_index("arm")
    if len(rows) > 1:
        md_table(rows[["val_RMSE", "val_CORR", "test_RMSE", "test_CORR", "best_grid", "train_time_s"]],
                 OUT / "tabla_grid.md", f"Ablacion del refinamiento de grilla, L={L}")


def fig_seqlen(res):
    r = res[res["suite"] == "seqlen"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in KAN_FAMILY:
        s = r[r["model"] == m].sort_values("seq_len")
        for a, k in zip(ax, ("val_RMSE", "val_CORR")):
            a.plot(s["seq_len"], s[k], "o-", label=m)
    for a, k in zip(ax, ("RMSE val (mm/h)", "CORR val")):
        a.set_xscale("log"); a.set_xlabel("largo de ventana L (h)"); a.set_ylabel(k); a.grid(alpha=.3)
    for m, c in (("LSTM", "0.4"), ("GRU", "0.55"), ("TCN", "0.7")):    # baselines: lineas punteadas grises
        s = res[(res["model"] == m) & res["suite"].isin(["seqlen_base", "baselines"])].sort_values("seq_len")
        for a, k in zip(ax, ("val_RMSE", "val_CORR")):
            a.plot(s["seq_len"], s[k], "s--", color=c, ms=4, lw=1, label=m)
    p = res.loc[res["arm"] == "Persistence", "val_RMSE"]
    if len(p):
        ax[0].axhline(p.iloc[0], color="k", ls="--", lw=1, label="persistencia")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig5_seqlen.png", dpi=150); plt.close(fig)
    print("-> fig5_seqlen.png")


def fig_grid(res):
    arms = [MAIN] + list(res.loc[res["suite"] == "grid", "arm"]) + [f"RKAN_L{L}"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for arm in arms:
        h = EXP / f"{arm}_history.csv"
        if h.exists():
            d = pd.read_csv(h)
            ax.plot(np.arange(1, len(d) + 1), d["val_RMSE"], "o-", ms=3, label=arm)
    d = pd.read_csv(EXP / f"{MAIN}_history.csv")
    for e in np.flatnonzero(np.diff(d["grid"].values)) + 1.5:     # cambios de grilla
        ax.axvline(e, color="gray", lw=.5, ls=":")
    ax.set_xlabel("epoca (lineas punteadas: cambio de grilla 3→5→10→20→50→100)")
    ax.set_ylabel("RMSE val (mm/h)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig4_grid.png", dpi=150); plt.close(fig)
    print("-> fig4_grid.png")


def fig_horizon(arms):
    fig, ax = plt.subplots(figsize=(8, 4))
    for arm, d in arms.items():
        ax.plot(np.arange(1, 49), d["test"]["per_horizon"]["RMSE"], label=arm)
    ax.set_xlabel("horizonte (h)"); ax.set_ylabel("RMSE test (mm/h)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_horizonte.png", dpi=150); plt.close(fig)
    print("-> fig_horizonte.png")


def fig_nse(arms):
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, d in arms.items():
        v = np.sort(np.clip(list(d["test"]["nse_per_basin"].values()), -1, 1))
        ax.plot(v, np.linspace(0, 1, len(v)), label=f"{arm} (med {np.median(v):.2f})")
    ax.set_xlabel("NSE por cuenca (test, recortado a [-1, 1])"); ax.set_ylabel("fraccion de cuencas")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_nse_cuencas.png", dpi=150); plt.close(fig)
    print("-> fig_nse_cuencas.png")


def fig_hydrographs(arms, Xq):
    y = arms[MAIN]["test"]["target"]
    picks = [int(np.argmax(y.max(1))), int(np.argsort(y.max(1))[-200]), int(np.argsort(y.max(1))[len(y) // 2])]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    for a, i in zip(ax, picks):
        a.plot(np.arange(-72, 0), Xq[i, -72:], "k", lw=1, label="observado (entrada)")
        a.plot(np.arange(0, 48), y[i], "k--", lw=1.5, label="observado (objetivo)")
        for arm, d in arms.items():
            a.plot(np.arange(0, 48), d["test"]["pred"][i], lw=1, label=arm)
        a.set_title(f"test #{i}"); a.set_xlabel("h"); a.grid(alpha=.3)
    ax[0].set_ylabel("caudal (mm/h)"); ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "fig_hidrogramas.png", dpi=150); plt.close(fig)
    print("-> fig_hidrogramas.png")


def by_regime(arms, X):
    """RMSE test por cuartil de: caudal actual, lluvia de las ultimas 24 h y
    temperatura media de la ventana (proxy de estacion)."""
    keys = {"caudal_actual": X[:, -1, config.TARGET_CHANNEL],
            "lluvia_24h": X[:, -24:, PRECIP].sum(1),
            "temperatura_media": X[:, :, TEMP].mean(1)}
    parts = []
    for k, v in keys.items():
        q = pd.qcut(v, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop") if k != "lluvia_24h" \
            else pd.cut(v, [-np.inf, 1e-6, *np.quantile(v[v > 1e-6], [.5, .9]), np.inf],
                        labels=["sin lluvia", "baja", "media", "alta (top 10%)"])
        rows = {}
        for arm, d in arms.items():
            e2 = ((d["test"]["pred"] - d["test"]["target"]) ** 2).mean(1)
            rows[arm] = pd.Series(e2).groupby(np.asarray(q), observed=True).mean() ** .5
        t = pd.DataFrame(rows)
        t.index = [f"{k}: {i}" for i in t.index]
        t["n"] = pd.Series(np.asarray(q)).value_counts().reindex([str(i).split(": ")[1] for i in t.index]).values
        parts.append(t)
    md_table(pd.concat(parts), OUT / "errores_por_regimen.md", "RMSE test (mm/h) por regimen")


def peaks(arms):
    y = arms[MAIN]["test"]["target"]
    thr = np.quantile(y, 0.99)
    m = y >= thr
    rows = {}
    for arm, d in arms.items():
        p = d["test"]["pred"]
        rows[arm] = {"RMSE_picos": float(np.sqrt(((p - y)[m] ** 2).mean())),
                     "sesgo_rel_picos_%": float(100 * ((p - y)[m] / y[m]).mean()),
                     "RMSE_resto": float(np.sqrt(((p - y)[~m] ** 2).mean()))}
    md_table(pd.DataFrame(rows).T, OUT / "picos.md", f"Error en picos (y >= p99 = {thr:.3f} mm/h, test)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(EXP / "experiments_results.csv")
    ablation_table(res); comparison_table(res); advanced_table(res); seeds_table(res); grid_table(res)
    fig_seqlen(res); fig_grid(res)
    names = ["Persistence", f"KAN_L{L}", f"RKAN_L{L}", MAIN, "LSTM", "TCN"]
    arms = {a: d for a in names if (d := load_arm(a)) is not None and d["test"] is not None}
    with h5py.File(config.TEST_H5, "r") as f:
        X = f["X"][:]
    fig_horizon(arms); fig_nse(arms); fig_hydrographs(arms, X[..., config.TARGET_CHANNEL])
    by_regime(arms, X); peaks(arms)


if __name__ == "__main__":
    main()
