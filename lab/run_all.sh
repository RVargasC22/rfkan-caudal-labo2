#!/usr/bin/env bash
# Orquestador: corre todo el pipeline de experimentos secuencialmente.
#   1. experiments.py --suite seqlen    -- Fig. 5 + Tabla 3 (KAN/RKAN/FKAN/RFKAN x L)
#   2. experiments.py --suite grid      -- Fig. 4 + ablacion del refinamiento de grilla
#   3. experiments.py --suite baselines -- Tabla 5/6 (persistencia, ARIMA, SVR, ELM, LSTM, GRU, TCN)
#   4. experiments.py --suite seeds     -- robustez: 3 semillas RFKAN + mejor baseline
#   4b. seqlen L=336                    -- al final: RKAN con L=336 tarda horas
#   4c. experiments.py --suite advanced   -- Tabla 7: Informer
#   4d. experiments.py --suite seqlen_base -- Fig. 5: LSTM/GRU/TCN para varios L
#   5. predict.py                       -- CSV de test (formato Id,q_01..q_48)
#   6. analyze_results.py               -- graficos + analisis de errores
set -e
cd "$(dirname "$0")/repo"
PY=../.venv/bin/python
L=${SEQ_LEN:-72}

echo "[$(date +%H:%M:%S)] === 1/6 seqlen ==="
$PY experiments.py --suite seqlen --lengths 72 2 12 24 168
echo "[$(date +%H:%M:%S)] === 2/6 grid (L=$L) ==="
$PY experiments.py --suite grid --seq_len $L
echo "[$(date +%H:%M:%S)] === 3/6 baselines ==="
$PY experiments.py --suite baselines
echo "[$(date +%H:%M:%S)] === 4/6 seeds ==="
$PY experiments.py --suite seeds --seq_len $L
echo "[$(date +%H:%M:%S)] === 4b/6 seqlen L=336 ==="
$PY experiments.py --suite seqlen --lengths 336
echo "[$(date +%H:%M:%S)] === 4c/6 advanced (Informer) ==="
$PY experiments.py --suite advanced
echo "[$(date +%H:%M:%S)] === 4d/6 seqlen_base ==="
$PY experiments.py --suite seqlen_base
echo "[$(date +%H:%M:%S)] === 5/6 predict ==="
$PY predict.py --arm RFKAN_L$L
echo "[$(date +%H:%M:%S)] === 6/6 analyze ==="
$PY analyze_results.py
echo "[$(date +%H:%M:%S)] === PIPELINE COMPLETO ==="
