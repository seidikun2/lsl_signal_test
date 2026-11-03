# LSL signal check

Start with **`main.py`**. The project has four Python files:

| File | Responsibility |
| --- | --- |
| `main.py` | Read the settings and run the chosen experiment |
| `transmission.py` | Generate simulated EEG, EMG, IMU and markers |
| `reception.py` | Receive simulated or real LSL outlets |
| `analysis.py` | Save data, compare samples and produce plots |

Use the Python interpreter that already works in VS Code. Settings are in
**`config.json`**. Install the dependencies only if needed:

```bash
python -m pip install -r requirements.txt
```

## Simulated signals

```bash
python main.py
```

EEG is tested first at different rates and channel counts. A separate experiment
adds EEG, then EMG, then IMU, keeping their native rates at **512/2000/70 Hz**.
Decimal rates are supported. A/B/C markers cause known changes 200 ms later.
The sender transmits individual samples; one receiver reads every stream with a
common target of 2048 cycles/s. Actual read times are saved.

## Real eego mylab and LabRecorder

Start the device LSL outlet. In LabRecorder, select the same EEG and optional
PsychoPy marker outlets, and start recording an XDF. Then run Python:

```bash
python main.py --real-eeg
```

The stream name is already configured. The capture lasts 60 seconds; Ctrl+C saves
a shorter capture. To find outlet names, run `python main.py --list-streams`.
Set `real_eeg.marker_name` if more than one marker outlet is present.

Stop LabRecorder after Python finishes. Compare the two recordings:

```bash
python -m pip install pyxdf
python main.py --compare-eeg results-real/recordings.npz --reference /path/to/session.xdf
```

The comparison preserves original timestamps and checks channels, values and
markers in the shared recording range. It does not shift curves, resample or
change units. Real EEG receives no artificial response. This checks two LSL
recordings; a physical stimulus-onset check needs a recorded trigger or sensor.

Repeated or backward timestamps no longer stop the report. The terminal and
report identify which recording contains them. Equal repeated-time blocks can
be compared in their recorded sample order. Unequal blocks, partial EEG blocks
at capture edges and conflicting clock matches are marked **ambiguous**, not
counted as proven loss. No timestamps or source arrays are changed. The report
shows how many samples were actually compared; zero value difference applies
only to those matched samples. A clock going backwards also affects the delay
curve, so it is not proof that the receiving buffer is filling. Missing counts
remain unknown in that case. The matched-value plot uses recorded sample order,
not reconstructed time; a source live view is omitted when its times repeat or
go backwards. No new capture is needed to inspect these checks.

## Results

Open **`results/summary.html`**, or **`results-real/summary.html`** for real EEG.
Plots and tables are in English. EEG and additional streams have separate
sections. The maps appear first; detailed signal plots expand below them.

The first signal plot keeps every recorded sample at its sample time. The second
is a live view on a fixed 2048 Hz display clock, holding the latest available
value. The third shows the measured delay. Thick marker plots use the same color
per marker: solid/filled means source, dashed/hollow means received. Real marker
source times are mapped with LSL clock correction.

Each run creates **one folder with four files**: `recordings.npz`, `summary.html`,
`metrics.csv`, and `markers.csv`. The NPZ holds arrays and JSON metadata without
pickle. XDF comparison updates the report and CSV files, preserving the raw NPZ.
NPZ recordings from the earlier project require that earlier version of the code.
Use a fresh output directory for another run:

```bash
python main.py --out results-new
python main.py --real-eeg --out results-real-02
```

Rates need not be integer multiples. Each sensor has its own sampling instants;
the buffer holds samples until reading, while timestamps and clock correction
keep their positions in time. Missing data are distinguished from measured zeros.
