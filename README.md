# LSL signal check

Start with **`main.py`**. The project has four Python files:

| File | Responsibility |
| --- | --- |
| `main.py` | Read the settings and run the chosen experiment |
| `transmission.py` | Generate simulated EEG, EMG, IMU and markers |
| `reception.py` | Receive streams, saving sample times, read times and buffer snapshots |
| `analysis.py` | Save recordings, compare values and create the interactive report |

Use the Python interpreter that already works in VS Code. Settings are in
**`config.json`**. Install or update the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Simulated signals

```bash
python main.py --out results-new
```

EEG is tested first at different rates and channel counts. A separate experiment
adds EEG, then EMG, then IMU, keeping their native rates at **512/2000/70 Hz**.
Rates need not be integer multiples; decimal rates are supported.

The sender publishes each A/B/C marker, then enables its waveform change. The
first available sample of each sensor carries that change, lasting 80 ms.
At 512 Hz, one sample interval is 1.953 ms; at 70 Hz, it is 14.286 ms. Those
native intervals can produce different onset times without a synchronization
failure. The report detects the actual changes in the sent and received arrays.
Older recordings retain their original 200 ms effect timing.

One Python loop reads all streams with a common **target** of 2048 cycles/s.
This setting does not change a sensor's sampling rate or resample its data.
One read can return several samples. Their sample timestamps stay distinct even
when their Python read time is the same. A buffer holds samples; timestamps and
LSL clock correction place samples from different outlets on a shared clock.

## Real eego mylab and LabRecorder

Start the device LSL outlet. In LabRecorder, select the same EEG and optional
PsychoPy marker outlets, and start recording an XDF. Then run Python:

```bash
python main.py --real-eeg --out results-real-01
```

The stream name is configured in `real_eeg.stream_name`. Capture duration is
`real_eeg.duration_s`; Ctrl+C saves a shorter capture. To find outlet names,
run `python main.py --list-streams`. Set `real_eeg.marker_name` if more than one
marker outlet is present.

Stop LabRecorder after Python finishes. Compare the two recordings:

```bash
python -m pip install pyxdf
python main.py --compare-eeg results-real-01/recordings.npz --reference /path/to/session.xdf
```

The XDF stays on your machine; its size does not require uploading it. Comparison
preserves original timestamps and checks channels, values and markers in the
shared recording range. Repeated or backward timestamps are reported. Equal
repeated-time blocks can be compared in recorded order; unequal, partial or
conflicting blocks remain ambiguous. Zero value difference applies only to the
matched subset. A backward clock leaves unmatched counts unknown.

Real EEG receives no artificial response. Source timestamps allow measuring
sample age at Python reads; actual amplifier push-call times are unavailable.
Comparing two recordings cannot establish physical stimulus onset or amplifier
latency. That needs a recorded trigger or sensor.

## Interactive results

Open **`summary.html`** in the output folder, using your browser. It works offline.
All plots and tables are in English, with separate sections for EEG and additional
streams. Heatmaps give an overview; choose a recording and signal below them.

- **Zoom:** drag over a signal or scroll. Use the toolbar to pan or save an image.
  All five time axes zoom together. Double-click or **Full recording** resets it.
- **Markers:** select a source or read event to zoom around it. Each label keeps
  the same color; source markers are filled, received markers are hollow.
  Thick vertical lines are solid at source and dashed at reception. Triangles
  mark detected waveform changes; the event rows distinguish stored and read times.
- **Signal:** the first panel uses sample timestamps, holding each value until
  the next sample so line interpolation cannot make a change appear before its
  actual first sample. The second holds the latest
  value available at actual send/read calls. All channel-1 points are included
  across the full recording; value comparisons still check every channel.
- **Delay:** inspect the timeline and the first/last median, maximum and p99
  (99% of measured delays are at or below this value). End minus start compares
  the first and last second, shortened to one third of the interval in brief captures.
- **Waiting samples:** snapshots show samples queued in this inlet before each
  read. Rising counts together with increasing delay support accumulation here.
  Snapshots can miss peaks and do not measure every upstream buffer.

**Sample → read** includes the time since the sample's corrected timestamp.
**Send call → read**, available in simulations, starts at the actual push call.
The gray sender-wait curve shows time spent before that call. These distinguish
a late sender from a slow delivery/read path. Clock-correction uncertainty remains;
repeated or backward source timestamps limit the sample-age interpretation.
No acceptable delay threshold is assumed.

The marker table separates sent marker-to-change spacing, received spacing and
added timing error. First changed sample preservation is checked. Read-time gaps
show when Python actually had the marker and effect; a negative gap means the
EEG chunk was read first. Additional-stream results also compare each sensor's
effect with simultaneous EEG, subtracting the native source offset.

## Saved files and existing recordings

Each run creates **one folder with four files**: `recordings.npz`, `summary.html`,
`metrics.csv`, and `markers.csv`. The NPZ contains arrays and JSON without pickle.
Report generation and XDF comparison preserve that raw file.

Rebuild an existing recording's report without acquiring again:

```bash
python main.py --report results-real-01/recordings.npz
```

Use `--compare-eeg` with the XDF again to include reference comparisons; these are
not stored in the raw capture. Old files show unrecorded buffer or marker-push
fields as **not measured**, and keep the waveform originally transmitted. A new
capture is needed for those measurements and immediate simulated marker effects.
Recordings from before the four-script project require their earlier code version.
Use a fresh output directory for each new capture.
