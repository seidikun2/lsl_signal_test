# LSL transmission experiments

**Edit `settings.py`, run `main.py`, open the summary.**

```bash
python -m pip install -r requirements.txt
python main.py
```

In VS Code, select your Python interpreter and use **Run Python File** on `main.py`.
Use `git pull` to receive updates.

## Two experiments

| Experiment | What changes | What stays fixed |
| --- | --- | --- |
| `eeg` | EEG channels × sampling rate | One EEG stream; configured chunk and buffer |
| `streams` | Add one stream at each stage | Each existing stream keeps its rate and channels |

Defaults: **20 EEG cases** (16/32/64/128 channels × 128/256/512/1024/2048 Hz),
then **3 stages**: EEG → EEG+EMG → EEG+EMG+IMU.
The default is 5 seconds per stream: 115 seconds of acquisition, plus startup and analysis.
The complete run takes several minutes. Choose one experiment in `TESTS["run"]`
or reduce its parameter lists for a quick check.

Set `length_mode` to `"seconds"` for equal duration (`duration_s`, e.g. 10), or
`"samples"` for exactly `samples_per_stream` points per stream. In samples mode,
faster streams finish earlier; responses to recorded markers beyond a stream's end are marked
`outside_recording`. Rates such as 100/200/400/800 Hz are also supported.

## Markers and visible responses

Classes **A, B and C** each repeat every 2 seconds, with different starting times
and pulse amplitudes. Each marker produces a visible signal change **200 ms later**.
Edit `interval`, `first`, `delay`, `width` and `amplitude` in `CONFIG["markers"]`.

The analysis detects the response in the actual recorded waveform against the
known synthetic baseline. Tables show the expected interval, original TX interval,
RX acquisition interval and RX delivery interval. Plots show the original and
reconstructed signals around each marker class. Sampling quantization, lost
onsets, missing markers, overlapping responses and recording boundaries are explicit.
Response plots/detection use channel 0; amplitude integrity is checked on all channels.

## Results

```text
main.py                     run this
settings.py                 edit this
lsl_lab/                    generator, receiver, analysis and tests
results/run_timestamp/
  summary.md                experiment overview
  plan.csv                  every planned case and its execution status
  eeg/                      case table, channel × rate heatmaps, individual cases
  streams/                  stage table, per-stream comparisons, individual cases
    case_name/
      summary.md            timing, marker and response tables
      figures/              signal reconstruction and event-response plots
      data/                 values, timestamps and detailed reports
```

Open Markdown previews with **Ctrl+Shift+V** in VS Code.
[Measured examples](results/example/summary.md).

No delivery-delay compensation or online alignment is applied. Clock mapping,
acquisition drift and delivery delay are reported separately. These are software
simulations, not physical sensor validation. Recording uses RAM until saving;
split very long runs into shorter cases. Tests: `python -m pytest lsl_lab/tests -q`.
Acquisition timestamps are simulated; Python read times are measured. One extra
sample-index channel is transported for loss detection and excluded from saved signal channels.
