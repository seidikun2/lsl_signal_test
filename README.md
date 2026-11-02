# LSL — EEG results first

In VS Code, keep the Python interpreter that already works and run **`main.py`**.
Open **`results/summary.html`** when recording finishes. Results remain in English.

```bash
python main.py
```

`config.json` selects **`"active_suite": "normal"`**. These tests add no artificial
processing wait. View the recorded results first; analysis is a separate step.

## Test 1 — EEG only

Six sampling rates: **500, 512, 731.5, 1000, 1500, 2048 Hz**.
Each rate is tested with **16, 32, 64, 128 channels**, three repetitions per
combination: **72 EEG recordings**. Rates are not restricted to integer multiples;
731.5 Hz also exercises a genuinely decimal sampling rate.

The sender transmits one sample at a time. Earlier tests grouped 16 samples:
at 500 Hz this added up to 30 ms before sending, versus about 7 ms at 2048 Hz.
That made the live view improve with rate even when every recorded value matched.

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
signals keep the same rates and channels. The fixed rates are EEG **512 Hz**,
EMG **2000 Hz**, and IMU **70 Hz**. Three repetitions per stage give **nine recordings**.

This section has its own maps, signal selection, marker measurements, and table.
It is not pooled with the EEG matrix.

In this test only, the same known event changes all simulated sensors. A separate
plot shows its sent and received sample times together, then its actual read times.
The table compares each sensor with EEG. It separates the difference already
present at sending, due to native sampling times, from any added difference after
reception. These generated streams share one sender clock; this does not measure
clock offsets in independent physical devices.

Unrelated rates do not inherently cause accumulating delay. Samples simply fall
at different instants: 512 Hz gives about 1.95 ms per sample, while 70 Hz gives
about 14.29 ms. A common event can therefore appear in different first samples.
The table shows this difference at sending, separately from any added difference
after reception. The buffer holds pending samples; timestamps and clock mapping
preserve their positions in time. Decimal sensor rates are accepted as well.

## Acquisition, reading, and display

- **Acquisition rate:** new samples per second produced by each sensor.
- **Receiver reading target:** **2048 cycles/s**, unchanged across all normal
  tests. Each cycle pulls buffered samples from every stream, including
  markers. A cycle can return zero, one, or several samples. No samples are
  discarded to force every sensor to have the same rate.
- **Display rate:** **2048 steps/s**, unchanged across normal plots. The live view
  keeps the latest value until another value becomes available.

The computer may miss a reading deadline. The report lists the measured reading
rate and reading gaps rather than assuming the target was achieved. Reading
faster does not create samples or improve the original sensor bandwidth. Nyquist
applies to **acquisition**: sample above twice the highest signal frequency, with
appropriate filtering. Reading frequency is a separate buffer and delay choice.

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
| `rx_<signal>__read_attempt_time_s` | Every buffer reading attempt, including empty reads |

The first signal plot compares **all sample values**, using the sample counter
carried with the data. The second is explicitly a **live view**, not the full
reconstruction: both signals are shown on the same fixed display clock, keeping
the latest available value at each step. Reception availability uses the receiver's
own measured read clock. A read may return multiple samples; they all remain in
the recording even if the live view shows only the latest. Individual arrival
times are not invented.
The delay plot uses the two recorded counters directly and separates sample-to-read
delay from send-call-to-read delay.

## Markers and EEG changes

Markers **A, B, C** cause configured changes in the sent EEG, normally **200 ms**
later. The response is detected from the actual sent and received EEG values,
not assumed from the schedule. In Test 2, EMG and IMU also receive the known
shared event, scaled to their own signal values, for the synchronization check.

Plots use **one color per marker**, thick lines, and explicit sent/received rows.
**Solid lines and filled symbols mean sent; dashed lines and hollow symbols mean
received.** A circle marks the marker; a triangle marks the EEG change. The table
lists their times and separates **stored timing** from **read-time timing**.
Batching can affect the latter even when sample timestamps are preserved. Missing
markers, lost response onsets, overlaps, and responses falling beyond recording
end are listed explicitly. These tests measure this simulated EEG/marker path.

## Running and output files

The default run has **81 cases**, five seconds of signal each. Startup, clock
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
| `markers.csv` | Marker responses and event timing between simultaneous sensors |
| `recordings.zip` | Original arrays, time counters, plan, settings, and machine information |

The ZIP contains one compressed NumPy recording per case; full signal arrays can
make it large. It stays local and is ignored by Git. Reanalysis uses the saved
configuration and recordings, not newly edited settings. It does not transmit.
Previous default results are preserved outside the project in
`../lsl_signal_test-history/`. Custom output directories must be fresh.
`--plan-only` does not replace existing results. `--signals` selects the same normal
EEG/marker workflow explicitly. Single older recordings can still be reanalyzed
with `python analysis.py path/to/T1_01_recording.npz`.

## Test 3 — real ANT Neuro eego mylab, separately

This mode receives the existing device outlet. It does not generate EEG, add
sample-counter channels, resample, filter, or change the amplifier rate. It uses
the stream's actual channel count and keeps every channel, including auxiliaries.
The configured name is **`EE225-020034-000905_on_DESKTOP-U197OKE`**, from the
working receiver example. Keep using the same VS Code Python interpreter.

1. Start the eego LSL outlet. Start the PsychoPy marker outlet if needed.
2. In **LabRecorder**, select the **same EEG and marker outlets** and start an XDF
   recording before starting Python. Keep both receivers running together.
3. Run the Python capture in the VS Code terminal:

   ```bash
   python main.py --list-streams
   python main.py --real-eeg
   ```

   The default capture lasts **60 seconds**. Ctrl+C saves a shorter recording.
   To select a marker outlet explicitly, use its exact name:

   ```bash
   python main.py --real-eeg --marker-name PsychoPyMarkers --duration 60
   ```

4. Stop LabRecorder after Python finishes and let it close the XDF file.
5. Install the optional XDF reader in that same interpreter, then compare:

   ```bash
   python -m pip install pyxdf
   python main.py --compare-eeg results-real/recordings.npz --reference /path/to/session.xdf
   ```

Open **`results-real/summary.html`**. The report first shows the actual EEG,
received markers, read delays and the age of the last EEG sample at capture stop.
The reference section adds an overlay and a
channel map of value differences, counts unmatched samples in the shared time
range, and checks marker values and original timestamps. Zero value difference
means the matched values agree. Missing timestamps, nonfinite values, channel
order changes and missing marker streams remain visible; they are not replaced
with successful zero measurements.

The XDF reader explicitly disables clock adjustment and timestamp smoothing for
this comparison. Both receivers must consume the same outlet at the same time:
matching uses the original source timestamps within **1 microsecond**, with no
curve alignment, interpolation or value scaling. Samples outside the common
recording range are excluded from the loss comparison. Available channel labels
and units are checked before comparing; otherwise original channel order is used
and the missing label check is stated. A changed marker value is reported even
when its timestamp matches. Repeated markers are preserved individually.

Raw timestamps, corrected timestamps, read times, clock estimates and outlet
metadata are saved in **`recordings.npz`**. The other three files are
**`summary.html`**, **`metrics.csv`**, and **`markers.csv`**. Comparison does not
modify the raw NPZ. This folder stays local and separate from synthetic results.
For another capture, use a fresh directory such as `--out results-real-02`.
Names, duration and buffer settings are in `config.json` under `real_eeg`.

With `marker_name` empty, Python looks for one outlet of type `Markers`; if none
is found it records EEG alone and says so. Multiple candidates require an exact
`--marker-name`. Marker recording must also be enabled in LabRecorder. A minimal
PsychoPy pattern is:

```python
from pylsl import StreamInfo, StreamOutlet
markers = StreamOutlet(StreamInfo('PsychoPyMarkers', 'Markers', 1, 0,
                                  'string', 'psychopy-markers'))
# In the experiment, where win is the PsychoPy window:
win.callOnFlip(markers.push_sample, ['A'])
win.flip()
```

The software marker is emitted at the flip callback; it is not a measured physical
screen-onset timestamp. No artificial marker-triggered response is injected into
real EEG. Recording the same LSL stream twice checks its transport and storage;
physical EEG/stimulus synchronization needs a recorded trigger or sensor. The
hardware path must be checked on the acquisition PC; cloud checks use test outlets.

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
