"""Edit this file, then run: python main.py"""

TESTS = {
    "run": ["eeg", "streams"],        # run either experiment, or both
    "length_mode": "seconds",        # "seconds" or "samples" per stream
    "duration_s": 5,                  # use 10 for a longer recording
    "samples_per_stream": 1024,       # only used in "samples" mode
    "eeg": {
        "rates": [128, 256, 512, 1024, 2048],  # alternatively [100, 200, 400, 800]
        "channels": [16, 32, 64, 128],
        "chunks": [16],               # e.g. [1, 16, 128] to compare batching
        "buffers_s": [60],
        "pauses_s": [0],
        "repeats": 1,
    },
    "streams": {"repeats": 1},        # add streams below, one at a time
}

CONFIG = {
    "chunk": 16,                      # samples per send
    "inlet_buffer": 60,               # seconds of retention
    "receiver_pause": 0,              # seconds; 0 disables artificial pauses
    "pause_every": 2,
    "sensors": [                      # fixed parameters for the streams experiment
        {"name": "eeg", "kind": "EEG", "rate": 512, "channels": 32},
        {"name": "emg", "kind": "EMG", "rate": 2000, "channels": 8},
        {"name": "imu", "kind": "IMU", "rate": 100, "channels": 9},
    ],
    # Each marker class repeats periodically. All cause a visible response 200 ms later.
    "markers": [
        {"label": "A", "interval": 2., "first": .3, "delay": .2, "width": .08, "amplitude": 1.},
        {"label": "B", "interval": 2., "first": .9, "delay": .2, "width": .08, "amplitude": -1.},
        {"label": "C", "interval": 2., "first": 1.5, "delay": .2, "width": .08, "amplitude": .6},
    ],
    "event_tolerance_ms": 10,         # RX interval error relative to observed TX interval
    "analysis_limits": {
        "latency_ms": 100, "jitter_ms": 10,
        "drift_ppm": 100, "accumulated_ms": 1, "latency_growth_ms": 20,
    },
    # Optional sensor fields: drift_ppm, drift_slope_ppm_s. Default: zero injected drift.
}
