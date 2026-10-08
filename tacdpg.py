"""
TACDPG (Zhang, Bu, Zhou, Li, Zhan, Zhang, Sustainable Energy Technologies and
Assessments 67:103830, 2024): marco de prediccion con aprendizaje por refuerzo
profundo que considera outliers. Tercer "modelo avanzado" de la Tabla 7 del
paper de RFKAN. Implementacion propia, siguiendo el texto del paper:

  1. Seleccion de variables con random forest (importancia por canal).
  2. Outliers (Ec. 1): g = 1 si y > mu + k sigma, -1 si y < mu - k sigma, 0 si no,
     con k = 1.5. mu y sigma son la media y el desvio del caudal de cada cuenca
     (train). Una ventana es outlier si alguna de sus 48 h lo es.
  3. CatBoost predice si la ventana es outlier (paso 5 del marco). En test, si
     CatBoost predice outlier se usa el agente entrenado con outliers; si no, el
     agente de valores normales (paso 7).
  4. Cada agente es un TACDPG: actor convolucional (online + target), dos
     criticos MLP gemelos (online + target), buffer de experiencia, objetivo
     y = r + gamma * min(Q1', Q2')(s', a'), actualizacion diferida del actor y
     actualizacion suave de los targets (Ec. 3-11, TD3).
     Recompensa adaptativa (Ec. 2):
        r = -[z0 - j (z0 - z1) / j_max] * |y - a| * exp(-lambda i)
     con j = episodio, i = paso dentro del episodio.
  Adaptacion al caudal: un episodio es una secuencia de ventanas consecutivas del
  archivo (como los pasos de tiempo de la serie PV del paper); el estado son las
  ultimas TACDPG_SEQ_LEN h de las variables elegidas y la accion es la correccion
  de las 48 h sobre el ultimo caudal, en unidades de sigma_delta (salida residual,
  igual que los demas modelos). Se simulan TACDPG_ENVS episodios en paralelo.
  z0, z1, lambda, gamma, tau y los tamaños estan en el material suplementario del
  paper (no disponible): son supuestos declarados en config.py y GAP_ANALISIS.md.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn

import config


# ------------------------------------------------------- RF, outliers, CatBoost
def select_features(X, dY, n=20000, seed=config.SEED):
    """Importancia de random forest por canal (features: media de 24 h y ultimo valor)."""
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    F = np.concatenate([X[idx, -24:].mean(1), X[idx, -1]], 1)
    rf = RandomForestRegressor(100, max_depth=10, n_jobs=-1, random_state=seed).fit(F, dY[idx].mean(1))
    imp = rf.feature_importances_.reshape(2, -1).sum(0)
    sel = sorted(set(np.argsort(imp)[::-1][:config.TACDPG_N_FEATURES].tolist()) | {config.TARGET_CHANNEL})
    return [int(c) for c in sel], imp


def basin_stats(y, basin):
    """Media y desvio del caudal de cada cuenca (train)."""
    ids = np.unique(basin)
    mu = {b: float(y[basin == b].mean()) for b in ids}
    sd = {b: float(y[basin == b].std()) for b in ids}
    return mu, sd


def outlier_labels(y, basin, mu, sd, kappa=config.TACDPG_KAPPA):
    """Ec. 1 por hora; la ventana es outlier (1) si alguna de sus 48 h lo es."""
    m = np.array([mu[b] for b in basin])[:, None]
    s = np.array([sd[b] for b in basin])[:, None]
    return (np.abs(y - m) > kappa * s).any(1).astype(int)


def catboost_classes(W_sample, labels, splits, seed=config.SEED):
    """CatBoost sobre las ultimas 24 h de las variables elegidas (W: (N, 24, k))."""
    from catboost import CatBoostClassifier
    flat = lambda W: W.reshape(len(W), -1)
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, random_seed=seed, verbose=0,
                             thread_count=14).fit(flat(W_sample), labels)
    return [clf.predict(flat(W)).astype(int).ravel() for W in splits]


# ------------------------------------------------------------------- redes
class Actor(nn.Module):
    """Capa de entrada + convoluciones apiladas + capa totalmente conectada + salida."""

    def __init__(self, c, L, width=config.TACDPG_WIDTH):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c, width, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(width, width, 5, padding=2, stride=2), nn.ReLU(),
                                  nn.Conv1d(width, width, 5, padding=2, stride=2), nn.ReLU(), nn.Flatten())
        n = width * ((L + 3) // 4)
        self.fc = nn.Sequential(nn.Linear(n, 2 * width), nn.ReLU(), nn.Linear(2 * width, config.FORECAST_HOURS))

    def forward(self, s):                                    # s: (B, L, c)
        return config.TACDPG_ACTION_MAX * torch.tanh(self.fc(self.conv(s.transpose(1, 2))))


class Critic(nn.Module):
    """MLP sobre (estado aplanado, accion)."""

    def __init__(self, c, L, width=4 * config.TACDPG_WIDTH):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c * L + config.FORECAST_HOURS, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s.flatten(1), a], 1)).squeeze(1)


class Agent(nn.Module):
    def __init__(self, c, L):
        super().__init__()
        self.actor, self.q1, self.q2 = Actor(c, L), Critic(c, L), Critic(c, L)
        self.actor_t, self.q1_t, self.q2_t = (copy.deepcopy(m) for m in (self.actor, self.q1, self.q2))
        for m in (self.actor_t, self.q1_t, self.q2_t):
            m.requires_grad_(False)

    @torch.no_grad()
    def soft_update(self, tau=config.TACDPG_TAU):             # Ec. 9-11
        for src, dst in ((self.actor, self.actor_t), (self.q1, self.q1_t), (self.q2, self.q2_t)):
            for p, pt in zip(src.parameters(), dst.parameters()):
                pt.mul_(1 - tau).add_(tau * p)


class TACDPG(nn.Module):
    """Dos agentes (valores normales / outliers), elegidos por la clase de CatBoost."""

    def __init__(self, features, seq_len=config.TACDPG_SEQ_LEN):
        super().__init__()
        self.features, self.seq_len = features, seq_len
        self.normal, self.outlier = Agent(len(features), seq_len), Agent(len(features), seq_len)

    def state(self, X):
        return X[:, -self.seq_len:, self.features]


# ---------------------------------------------------------------- entrenamiento
def train_agent(agent, model, data, rows, scaler, epochs, name, val_fn, seed=config.SEED, log=print):
    """TD3 con episodios de TACDPG_EPISODE_LEN ventanas consecutivas (dentro de `rows`),
    TACDPG_ENVS episodios en paralelo. Presupuesto: `epochs` pasadas sobre `rows`."""
    C = config
    dev = data.X.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.as_tensor(rows, device=dev)
    n, E, B = len(rows), C.TACDPG_EPISODE_LEN, C.TACDPG_ENVS
    j_max = max(1, int(np.ceil(epochs * n / (B * E))))
    eval_every = max(1, j_max // epochs)
    opt_a = torch.optim.Adam(agent.actor.parameters(), lr=C.TACDPG_LR_ACTOR)
    opt_c = torch.optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=C.TACDPG_LR_CRITIC)
    cap = C.TACDPG_BUFFER
    buf_s = torch.zeros(cap, dtype=torch.long, device=dev)          # indices de fila (estado)
    buf_s2 = torch.zeros(cap, dtype=torch.long, device=dev)         # siguiente estado
    buf_a = torch.zeros(cap, C.FORECAST_HOURS, device=dev)
    buf_r = torch.zeros(cap, device=dev)
    buf_d = torch.zeros(cap, device=dev)                            # fin de episodio
    ptr, size, updates = 0, 0, 0
    ds = scaler.delta_std
    hist, best, best_state, t0 = [], float("inf"), None, time.time()
    for j in range(j_max):
        w = C.TACDPG_Z0 - j * (C.TACDPG_Z0 - C.TACDPG_Z1) / j_max   # peso lineal por episodio (Ec. 2)
        sigma = C.TACDPG_NOISE * (1 - j / j_max) + 0.05
        start = torch.randint(0, max(1, n - E), (B,), generator=g).to(dev)
        for i in range(E):
            s_idx = rows[(start + i).clamp_max(n - 1)]
            s2_idx = rows[(start + i + 1).clamp_max(n - 1)]
            x = data.X[s_idx]
            s = model.state(x)
            with torch.no_grad():
                a = (agent.actor(s) + sigma * torch.randn(B, C.FORECAST_HOURS, device=dev)).clamp(
                    -C.TACDPG_ACTION_MAX, C.TACDPG_ACTION_MAX)
                target = (data.y[s_idx] - x[:, -1:, C.TARGET_CHANNEL]) / ds
                r = -w * (a - target).abs().mean(1) * np.exp(-C.TACDPG_LAMBDA * i)     # Ec. 2
            k = torch.arange(ptr, ptr + B, device=dev) % cap
            buf_s[k], buf_s2[k], buf_a[k], buf_r[k] = s_idx, s2_idx, a, r
            buf_d[k] = float(i == E - 1)
            ptr, size = (ptr + B) % cap, min(size + B, cap)
            # una actualizacion TD3 por paso
            b = torch.randint(0, size, (C.TACDPG_BATCH,), device=dev)
            sb, s2b = model.state(data.X[buf_s[b]]), model.state(data.X[buf_s2[b]])
            with torch.no_grad():
                a2 = agent.actor_t(s2b)
                a2 = (a2 + (0.2 * torch.randn_like(a2)).clamp(-0.5, 0.5)).clamp(-C.TACDPG_ACTION_MAX, C.TACDPG_ACTION_MAX)
                yq = buf_r[b] + C.TACDPG_GAMMA * (1 - buf_d[b]) * torch.min(agent.q1_t(s2b, a2), agent.q2_t(s2b, a2))
            loss_c = ((agent.q1(sb, buf_a[b]) - yq) ** 2).mean() + ((agent.q2(sb, buf_a[b]) - yq) ** 2).mean()
            opt_c.zero_grad(set_to_none=True)
            loss_c.backward()
            opt_c.step()
            if updates % 2 == 0:                                     # actor diferido + soft update
                loss_a = -agent.q1(sb, agent.actor(sb)).mean()
                opt_a.zero_grad(set_to_none=True)
                loss_a.backward()
                opt_a.step()
                agent.soft_update()
            updates += 1
        if (j + 1) % eval_every == 0 or j == j_max - 1:
            m = val_fn()
            if m["RMSE"] < best:
                best = m["RMSE"]
                best_state = copy.deepcopy(agent.state_dict())
            hist.append({"agent": name, "episode": j + 1, "val_RMSE": m["RMSE"], "val_CORR": m["CORR"],
                         "val_NSE_median": m["NSE_median"], "elapsed_s": time.time() - t0})
            log(f"[TACDPG-{name}] episodio {j + 1}/{j_max} val_RMSE={m['RMSE']:.4f} "
                f"val_CORR={m['CORR']:.4f} ({time.time() - t0:.0f}s)")
    agent.load_state_dict(best_state)
    return hist


@torch.no_grad()
def predict_tacdpg(model, data, cls, scaler, batch_size=2048):
    """Clase de CatBoost = 1 -> agente de outliers; 0 -> agente normal."""
    model.eval()
    cls = torch.as_tensor(cls, device=data.X.device).bool()
    out = []
    for i in range(0, data.n, batch_size):
        x = data.X[i:i + batch_size]
        s = model.state(x)
        a = torch.where(cls[i:i + batch_size, None], model.outlier.actor(s), model.normal.actor(s))
        out.append(x[:, -1:, config.TARGET_CHANNEL] + scaler.delta_std * a)
    return scaler.inverse_y(torch.cat(out).float().cpu().numpy())
