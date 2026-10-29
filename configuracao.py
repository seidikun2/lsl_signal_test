"""Edite somente este arquivo e execute: python main.py"""

CONFIG = {
    "duration": 10,          # duração do teste, em segundos
    "chunk": 16,             # amostras acumuladas por envio
    "inlet_buffer": 60,      # capacidade do buffer, em segundos
    "receiver_pause": 0,     # pausa artificial do receptor (0 desliga)
    "pause_every": 2,        # intervalo entre pausas, em segundos
    "sensors": [             # frequência (Hz) e canais de cada sensor
        {"name": "eeg", "kind": "EEG", "rate": 500,  "channels": 8},
        {"name": "emg", "kind": "EMG", "rate": 2000, "channels": 4},
        {"name": "imu", "kind": "IMU", "rate": 100,  "channels": 9},
    ],
    # Opcional, dentro de cada sensor: "drift_ppm": 200,
    # "drift_slope_ppm_s": 1000 para um drift que aumenta durante o teste.
    "analysis_limits": {     # limites ilustrativos para sinalizar problemas
        "latency_ms": 100, "jitter_ms": 10,
        "drift_ppm": 100, "accumulated_ms": 1, "latency_growth_ms": 20,
    },
}

# Para variar parâmetros automaticamente, mude ativar para True.
# Quando ativa, esta varredura testa um sensor por vez com os parâmetros abaixo.
# Duração, pausas e buffer continuam sendo os definidos em CONFIG.
VARREDURA = {
    "ativar": False,
    "sensor": "EEG",
    "frequencias": [100, 500, 1000, 2000],
    "canais": [1, 8, 32],
    "chunks": [1, 16, 128],
    "repeticoes": 1,
}
