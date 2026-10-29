# LSL simulation

**Edit `config.json`. Run `main.py`. Open `results/summary.html` in a browser.**

```bash
python -m pip install -r requirements.txt
python main.py
```

At startup, the complete test table and estimated runtime are printed and saved
as `results/plan.csv` and `results/plan.png`. To inspect the plan without recording:

```bash
python main.py --plan-only
```

## Files

```text
main.py             runs the tests
transmission.py     generates and sends signals and markers
reception.py        receives and records values and timestamps
analysis.py         computes metrics, plots and the complete report
helpers.py          shared configuration and recording utilities
config.json         all user parameters
results/
  plan.csv          every planned case, status, estimates and actual runtime
  plan.png          the test table
  summary.html      ALL cases, metrics, marker tables and signal snapshots
  test_1/           EEG matrix: CSV tables, images and recording files
  test_2/           incremental streams: CSV tables, images and recording files
```

No per-case directories. Each case has a short ID such as `T1_01`.
Snapshots show one or two channels with TX/RX markers and the expected response time.
The report includes every case; expand a snapshot or event table when needed.

## Configuration

In `config.json`, `tests.run` selects `eeg`, `streams`, or both:

- **Test 1:** 16/32/64/128 EEG channels × 128/256/512/1024/2048 Hz.
- **Test 2:** add EEG, EMG and IMU progressively, keeping existing parameters fixed.

Choose equal seconds or equal samples with `length_mode`. Set `duration_s`,
`samples_per_stream`, rates, channels, chunks, buffers and pauses in the same JSON.
The default has 23 cases, 5 seconds per stream; startup and analysis add overhead.
Markers A/B/C cause a visible response 200 ms later; their timings are editable.

Tables separate amplitude integrity, latency, acquisition drift and marker-response
errors. Reconstruction uses each signal's time vector, without removing delays.
Acquisition time is simulated; read time is measured. One extra transport index
channel detects loss and is excluded from stored signal channels.

Previous results are preserved **outside this project** before a new default run.
To reanalyze one saved recording: `python analysis.py results/test_1/T1_01_recording.npz`.
`checks.py` contains optional regression checks (`python -B -m pytest -p no:cacheprovider checks.py -q`).

[Completed test plan](results/plan.png) · [All Test 1 metrics](results/test_1/metrics.csv) · [All Test 2 metrics](results/test_2/metrics.csv).
Git includes the tables and figures. Numeric recording archives remain local.
