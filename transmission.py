"""Generate signals, add known marker effects, and send samples through LSL."""

import json
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

EFFECT_DELAY = 0.0
EFFECT_WIDTH = 0.08
AMPLITUDE = {"EEG": 60.0, "EMG": 140.0, "IMU": 3.0}


def marker_events(duration):
    """A, B and C repeat every two seconds."""
    events = []
    for label, first, strength in [("A", 0.3, 1), ("B", 0.9, -1), ("C", 1.5, 0.6)]:
        for stamp in np.arange(first, duration, 2.0):
            events.append({"label": label, "time": float(stamp), "strength": strength})
    return sorted(events, key=lambda event: event["time"])


def signal(sensor, indices, events=()):
    """The same formula is used for sending and checking simulated changes."""
    t = np.asarray(indices)[:, None] / sensor["rate"]
    channel = np.arange(sensor["channels"])[None, :]
    kind = sensor["kind"]
    if kind == "EEG":
        values = 20 * np.sin(2 * np.pi * (8 + channel % 5) * t)
        values += 8 * np.sin(2 * np.pi * 22 * t + 0.1 * channel)
    elif kind == "EMG":
        values = 70 * np.sin(2 * np.pi * (85 - channel % 5) * t)
        values *= 0.2 + 0.8 * np.sin(2 * np.pi * 0.7 * t) ** 2
    elif kind == "IMU":
        values = np.sin(2 * np.pi * (0.7 + 0.1 * (channel % 3)) * t)
    else:
        raise ValueError(f"Unknown sensor kind: {kind}")
    for event in events:
        onset = event["time"] + EFFECT_DELAY
        active = (t >= onset) & (t < onset + EFFECT_WIDTH)
        values += AMPLITUDE[kind] * event["strength"] * active
    return values.astype(np.float32)


def send(sensors, duration, identity, ready, finished, destination):
    """Run in a separate process. Wait until the receiver is ready to read."""
    outlets = {}
    saved = {}
    for sensor in sensors:
        name = sensor["name"]
        info = StreamInfo(
            f"{identity}_{name}",
            sensor["kind"],
            sensor["channels"] + 2,
            sensor["rate"],
            "double64",
            f"{identity}:{name}",
        )
        outlets[name] = StreamOutlet(info)
        saved[name] = {
            key: [] for key in ("values", "timestamps", "push_times", "index", "counter")
        }
    markers = StreamOutlet(
        StreamInfo(f"{identity}_markers", "Markers", 1, 0, "string", f"{identity}:markers")
    )
    if not ready.wait(30):
        raise TimeoutError("Receiver did not finish connecting")
    start = local_clock() + 0.2
    events = marker_events(duration)
    schedule = [{"label": "start", "time": 0}] + events + [{"label": "end", "time": duration}]
    next_sample = {sensor["name"]: 0 for sensor in sensors}
    marker_index = 0
    marker_times, marker_values, marker_push_times = [], [], []
    active_events = []
    try:
        while not finished.is_set():
            due = [
                (next_sample[s["name"]] / s["rate"], s)
                for s in sensors
                if next_sample[s["name"]] < int(np.ceil(duration * s["rate"]))
            ]
            if marker_index < len(schedule):
                due.append((schedule[marker_index]["time"], None))
            if not due:
                break
            # Publish the marker before any sample at the same scheduled time.
            counter, sensor = min(due, key=lambda item: (item[0], item[1] is not None))
            stamp = start + counter
            time.sleep(max(0, stamp - local_clock()))
            if sensor is None:
                event = schedule[marker_index]
                label = event["label"]
                marker_push_times.append(local_clock())
                markers.push_sample([label], timestamp=stamp)
                marker_times.append(stamp)
                marker_values.append([label])
                if "strength" in event:
                    active_events.append(event)
                marker_index += 1
                continue
            name = sensor["name"]
            index = next_sample[name]
            values = signal(sensor, [index], active_events)[0]
            # Only simulated packets add these two verification channels.
            packet = [*values.tolist(), index, counter]
            pushed = local_clock()
            outlets[name].push_sample(packet, timestamp=stamp)
            for key, value in zip(saved[name], (values, stamp, pushed, index, counter)):
                saved[name][key].append(value)
            next_sample[name] += 1
    except KeyboardInterrupt:
        pass
    arrays = {
        f"{name}__{key}": np.asarray(values)
        for name, fields in saved.items()
        for key, values in fields.items()
    }
    for sensor in sensors:
        key = sensor["name"] + "__values"
        arrays[key] = arrays[key].reshape(-1, sensor["channels"])
    arrays.update(
        start=np.array(start),
        events=np.array(json.dumps(events)),
        effect_delay_s=np.array(EFFECT_DELAY),
        effect_width_s=np.array(EFFECT_WIDTH),
        markers__timestamps=np.asarray(marker_times),
        markers__push_times=np.asarray(marker_push_times),
        markers__values=np.asarray(marker_values, dtype=str).reshape(-1, 1),
    )
    np.savez_compressed(destination, **arrays)
    finished.wait(30)  # Keep outlets alive until the receiver has drained them.
