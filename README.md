# LSL — EEG results first

In VS Code, keep the Python interpreter that already works and run **`main.py`**.
Open **`results/summary.html`** when recording finishes. Results remain in English.

```bash
python main.py
```

`config.json` selects **`"active_suite": "normal"`**. These tests add no artificial
processing wait. View the recorded results first; analysis is a separate step.

## Test 1 — EEG only

Nine sampling rates: **64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 Hz**.
Each rate is tested with **16, 32, 64, 128 channels**, three repetitions per
combination: **108 EEG recordings**.

The report starts with color maps: sampling rate across columns, channel count
across rows. Maps show:

- Delay after the send call.
- Time from the sample timestamp until it was read, including batch formation.
- Missing samples.
- Difference between sent and received values.
- EEG/marker timing difference in saved sample timestamps.
- EEG/marker timing difference at actual read times.

Each cell shows the highest recorded value across repetitions; every run remains
in the result table. Gray means no measurement, not zero. A red outline marks a
failed recording. Read the printed color scale rather than interpreting color as
a pass/fail decision. **p99** means 99% of samples had a delay at or below the value.

Choose a case to see its sent and received signals and marker timings. Plot labels
are deliberately simple. No automatic explanation is presented ahead of the data.

## Test 2 — adding signals

A separate report section shows **EEG → EEG + EMG → EEG + EMG + IMU**. Existing
signals keep the same rates and channels. The default rates are EEG 512 Hz,
EMG 2000 Hz, and IMU 100 Hz. Three repetitions per stage give **nine recordings**.

This section has its own maps, signal selection, marker measurements, and table.
It is not pooled with the EEG matrix.

## Recorded time vectors

Each signal packet carries its sample index and the sender's time counter.
The receiver saves that counter unchanged and also records its own read clock.
One fixed initial LSL clock offset maps a shared start; the first arrival of each
stream is never reset to zero and curves are never shifted to make them match.

Each case archive contains these arrays:

| Array | Meaning |
| --- | --- |
| `tx_<signal>__sender_time_s` | Sender sample counter |
| `tx_<signal>__sender_send_time_s` | Time of the actual Python send call |
| `rx_<signal>__sender_time_s` | Sender counter carried in the received packet |
| `rx_<signal>__receiver_time_s` | Receiver's own read-clock counter |

The first signal plot compares **all sample values**, using the sample counter
carried with the data. The second shows **when values became available**, using
the receiver's own clock. It shows the latest value at each read: samples in one
batch share one measured read time. Individual arrival times are not invented.
The delay plot uses the two recorded counters directly and separates sample-to-read
delay from send-call-to-read delay.

## Markers and EEG changes

Markers **A, B, C** cause configured changes in the sent EEG, normally **200 ms**
later. The response is detected from the actual sent and received EEG values,
not assumed from the schedule. EMG and IMU are not given artificial EEG responses.

Plots show the sent marker/change pair and received marker/change pair. The table
lists their times and separates **stored timing** from **read-time timing**.
Batching can affect the latter even when sample timestamps are preserved. Missing
markers, lost response onsets, overlaps, and responses falling beyond recording
end are listed explicitly. These tests measure this simulated EEG/marker path.

## Running and output files

The default run has **117 cases**, five seconds of signal each. Startup, clock
measurement, saving, and plotting add time; the estimated total is printed before
recording. The included example report is a short cloud functionality check,
not the complete default run or a performance measurement of your PC.

```bash
python main.py --plan-only
python main.py --out /path/to/fresh-results
python main.py --reanalyze results/recordings.zip
python -B -m pytest -p no:cacheprovider checks.py -q
```

There are four result files, without figure folders:

| File | Contents |
| --- | --- |
| `summary.html` | Standalone report, maps and selectable signal plots |
| `metrics.csv` | Every case and signal measurement |
| `markers.csv` | Every EEG marker/response measurement |
| `recordings.zip` | Original arrays, time counters, plan, settings, and machine information |

The ZIP contains one compressed NumPy recording per case; full signal arrays can
make it large. It stays local and is ignored by Git. Reanalysis uses the saved
configuration and recordings, not newly edited settings. It does not transmit.
Previous default results are preserved outside the project in
`../lsl_signal_test-history/`. Custom output directories must be fresh.
`--plan-only` does not replace existing results. `--signals` selects the same normal
EEG/marker workflow explicitly. Single older recordings can still be reanalyzed
with `python analysis.py path/to/T1_01_recording.npz`.

## Extra tests — only after looking at normal results

```bash
python main.py --local --out /path/to/fresh-extra-results
```

This separate protocol deliberately slows processing. It is not the default LSL
test. A processing-time ratio of **1** uses all time available between samples;
**2** takes twice as long. For example, 1000 samples/s allow 1 ms per sample:
adding 2 ms of processing gives ratio 2. Extra plots use milliseconds per sample.

**Recovery** means the queue returns to its initial level after that added wait
is removed, while transmission continues. It is shown last. The separate `quick`
and `local` settings remain available for those experiments; their older
`recordings.npz` archives can still be reanalyzed.

## Get the update through GitHub Desktop

Use **Fetch origin**, select **`codex/english-reports`** in **Current branch**, and
open the project in VS Code. Publishing a cloud environment does not send code to
GitHub. This branch must be merged before its changes appear on `main`.
