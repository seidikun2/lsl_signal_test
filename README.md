# LSL: when does the receiver fall behind?

Edit **`config.json`**, run **`python main.py`**, and open **`results/summary.html`**.
In VS Code, keep using the Python interpreter that already works for this project.
The dependencies are unchanged. Reports, plots, CSV diagnostics, and console output
are in English.

```bash
python -B main.py --plan-only
python -B main.py
```

## Get the code through GitHub Desktop

Publishing a cloud environment snapshots that environment; it does not commit or
push code to GitHub, and it does not update a local checkout.

For this update, use **Fetch origin**, then choose **`codex/english-reports`** under
**Current branch**. Open that checkout in VS Code. Keep any uncommitted local work
safe before switching branches. The new code becomes available on `main` after
the branch is reviewed and merged.

## Target-machine protocol: selected in config.json

`"active_suite": "local"` selects the extended protocol without extra arguments.
It contains **93 runs** and **47.5 minutes of scheduled acquisition**, plus startup
and queue-draining time:

| Experiment | Settings | Repetitions |
| --- | --- | --- |
| Capacity boundary | Loads 0.8, 0.9, 0.95, 1.0, 1.05; 30 s each | 6 |
| Incremental streams | EEG → EEG+EMG → EEG+EMG+IMU; 30 s per stage | 6 |
| Recovery under continuous input | 10 s at load 0.6, 10 s at 1.5, 20 s back at 0.6 | 6 |
| EEG rate sweep | 64, 128, 256, 512, 1024, 2048, 4096, 8192, **16384 Hz**; 30 s each | 3 |
| Simultaneous-rate sweep | EEG at 512, 2048, 8192, **16384 Hz**, with EMG at 2000 Hz and IMU at 100 Hz | 3 |

Incremental stages preserve existing sensors' rates and channel counts from
`acquisition.sensors`. One serial consumer shares its processing time across the
streams, with a controlled cost of 400 microseconds per sample. The two rate
sweeps impose no artificial processing wait: reading, conversion, generation and
transport still consume actual machine time.

Recovery requires an observed backlog during overload. Queue and delay window
medians must both return to their baseline references plus 50 ms for two
consecutive 1 s windows, while transmission continues. The report separates
return time, confirmation time, losses, and the drain after transmission stops.

The prespecified primary comparison is loads **>=1.0 versus <=0.9**, paired by
repetition. Each stream in an incremental stage has separate statistics; multiple
streams from one run are not counted as independent repetitions.

The report records the acquisition machine's operating system and Python version.
Short cloud runs validate test functionality. Run the extended protocol on the
machine intended for acquisition and processing. Controlled waiting models a
known processing budget; benchmarking an actual EEG algorithm requires executing
that algorithm instead.

To repeat the earlier short exploration, select `"active_suite": "quick"`.
`--local` explicitly selects the extended protocol but is unnecessary with the
current configuration.

## What the measurements distinguish

- **Constant delay:** chunks take time to form, but the queue does not grow.
- **Accumulation:** input arrives faster than the consumer processes it.
- **Saturation with discard:** the queue stops growing because samples are lost.
- **Recovery:** the queue returns to baseline while input continues.
- **Drain:** remaining samples are processed after acquisition ends.

**Total delay = transmitter chunk formation/scheduling + transport/queue until
read + batch processing.** It runs from each sample's acquisition timestamp to
processing completion, not the sum of all samples' delays.

Added delay is the difference between the last and first valid production windows.
**Queue ms/s** measures the additional signal time accumulated per wall-clock second.
Queue counts come directly from `StreamInlet.samples_available()` before reading
and after processing; they are not inferred from acquisition-clock drift.
A 1 s buffer stores one second of signal, not a guarantee of one-second latency:
a consumer at load 3 needs about 3 s to process that content.

## Acquisition, batch reading, and reconstruction have different rates

Nyquist applies to the **analog signal bandwidth**, not how often `pull_chunk` is
called. For a band-limited signal with highest frequency `f_max`, use acquisition
`f_s > 2 * f_max`, an antialias filter before analog-to-digital conversion, and a
practical margin. Later decimation also requires filtering before reducing rate.
A sensor may declare `bandwidth_hz`; the program rejects
`rate <= 2 * bandwidth_hz`. This checks the declared configuration; it neither
implements a filter nor estimates bandwidth automatically.

`acquisition.reconstruction_rate_hz` selects a **common 16384 Hz display grid**.
It does not change native acquisition rates or create new measurements. A 100 Hz
IMU still supplies 100 measurements/s. This grid is calculated during analysis;
the receiver does not need a timer waking 16384 times/s.

Samples within one batch can share a read time while retaining distinct acquisition
timestamps. Stacking all their values at that read time is a delivery diagnostic,
not causal waveform reconstruction. The plots now separate:

- **TX and RX by acquisition time:** offline reconstruction at original timestamps,
  without interpolation across lost sample indices.
- **Latest delivered sample:** a held-value curve updated only after real reads.
  Intermediate batch samples are omitted from this display only, which can make
  a fast waveform appear undersampled despite intact reception.
- **All-sample playback:** `playback_delay_s`, default 0.25 s, is an explicit output
  delay. Every sample is displayed at acquisition time plus this delay, only if it
  has arrived. Late batches or missing samples produce gaps. This is native-rate
  digital sample holding, not ideal band-limited reconstruction or delay removal.
- **Within-age-limit samples:** green values satisfy `delivery_max_age_s`; gray
  retains the last value even if stale. Age starts at **acquisition**, not arrival.

Offline interpolation can use future samples. It cannot prove real-time
availability or recover information lost to aliasing. On a 16384 Hz grid, each
100 Hz IMU sample occupies about 164 grid positions without gaining actual time
resolution. A fixed playback delay cannot absorb an indefinitely growing backlog.
Ideal rectangular marker pulses are not band-limited analog signals; marker timing
is separate from the base oscillation bandwidth.

A shared consumer can delay other streams while processing one batch. An
application can separate inlet reading from heavy processing, retain LSL-corrected
timestamps, use bounded queues, and define explicit stale-data/discard policies.
Independent readers only move the queue if processing capacity remains insufficient.
The current experiment deliberately retains a shared consumer to measure competition.
LSL clock correction alone does not establish perfect physical synchronization.

## Short protocol (`active_suite: quick`): 66 runs

Each acquisition lasts 4 s. Axes are varied separately and run order is randomized
with a fixed seed.

| Axis | Values | Other settings held fixed | Repetitions |
| --- | --- | --- | --- |
| Processing load | 0.25, 0.6, 0.9, 1.05, 1.5, 3 | 512 Hz, 32 channels, 8 s buffer | 6 |
| Receiver buffer | 1 and 2 s | Load 3; compare with the load-3, 8 s-buffer runs | 3 |
| Input rate | 64, 256, 1024, 4096, 16384 Hz | Cost 976.5625 us/sample, 1 s buffer | 3 |
| Send chunk | 1, 128, 512 samples | Load 0.25; compare with the default 16-sample chunk | 3 |

Load is `rate_hz * processing_seconds_per_sample`. Above 1, imposed processing
already requires more time than available; reading, conversion and scheduling can
bring the transition below 1. The receiver reads at most 32 samples per batch.
The sender buffer stays large while receiver capacity is varied.

The short protocol uses 0.5 s window medians, excludes startup and draining from
growth regression, and computes 95% bootstrap intervals **between runs**. Its
prespecified exact two-sided sign test compares loads >=1.5 versus <=0.6, paired
by repetition. Secondary sweeps and boundary estimates are descriptive.
Thousands of correlated samples are never treated as independent repetitions.

Short acquisitions bound a transition region, not universal machine capacity.
Increase duration and repetitions to confirm a boundary. A drain that reaches
`max_drain_s` with pending samples is **incomplete**: pending data do not become
fictitious losses. Its p99 and maximum describe processed samples only.

## Four output files, without figure subdirectories

| File | Contents |
| --- | --- |
| `results/summary.html` | Standalone English report with embedded plots |
| `results/metrics.csv` | One row per run/stream, including parameters and failures |
| `results/timeline.csv` | Queue depths and read/processing timestamps per batch |
| `results/recordings.npz` | Timing traces, measured previews, embedded config/plan/metrics |

Waveforms are checked by RMSE during acquisition. Only a short preview of up to
two measured RX channels is retained, rather than every waveform from every run.
Older timing-only recordings cannot supply missing RX previews; values are never
invented. Reanalysis translates older diagnostic strings into English while
preserving numeric measurements and the original recording archive.
`recordings.npz` stays local and is not committed.

Measurements are checkpointed after each completed run. Technical failures make
the command fail; overload losses remain visible experimental outcomes.
Repeated default runs archive previous results **outside the project** under
`../lsl_signal_test-history/`. Use a fresh output directory to keep separate runs:

```bash
python -B main.py --out /path/to/fresh-output
python -B main.py --reanalyze results/recordings.npz
python -B -m pytest -p no:cacheprovider checks.py -q
```

`--plan-only` neither transmits nor moves existing results. `main.py` starts the
workflow; `buffering.py` runs and analyzes queues; `transmission.py`, `reception.py`,
`analysis.py`, and `helpers.py` support signal/marker experiments; `checks.py` tests
both workflows.

## Signal and marker experiments

The original workflow remains available with nine rates from 64 through 16384 Hz:
36 EEG cases (16/32/64/128 channels) and three incremental-stream stages.

```bash
python -B main.py --signals --plan-only --out /path/to/fresh-plan
python -B main.py --signals --out /path/to/fresh-signal-results
python -B analysis.py /path/to/fresh-signal-results/test_1/T1_01_recording.npz
```

For a new installation, use `requirements.txt`; `pytest` is required only for the
optional checks. The existing VS Code interpreter can continue to be used.
