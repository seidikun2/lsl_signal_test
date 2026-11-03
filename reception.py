"""Receive simulated or real LSL streams without changing their sample times."""

import time
from xml.etree import ElementTree

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop, resolve_streams


def list_streams():
    for info in resolve_streams(wait_time=2):
        print(
            f"{info.name()} | {info.type()} | {info.channel_count()} channels | {info.nominal_srate():g} Hz"
        )


def record_real(settings, receiver):
    """The amplifier rate and channel count come from the actual outlet."""
    streams = [{"name": "eeg", "stream_name": settings["stream_name"]}]
    marker_name = settings["marker_name"]
    if not marker_name:
        found = [
            info
            for info in resolve_streams(wait_time=2)
            if info.type().lower() in ("marker", "markers")
        ]
        if len(found) > 1:
            raise ValueError(
                "Multiple marker outlets found; set real_eeg.marker_name in config.json"
            )
        marker_name = found[0].name() if found else ""
    if marker_name == settings["stream_name"]:
        raise ValueError("EEG and markers must be different outlets")
    if marker_name:
        streams.append({"name": "markers", "stream_name": marker_name})
    return record(streams, settings["duration_s"], receiver)


def record(specifications, duration, receiver, ready=None):
    """One reading loop serves every stream, including streams with different rates."""
    simulated = ready is not None
    inlets, streams, buffers, offsets = {}, {}, {}, {}
    case = {
        "simulated": simulated,
        "status": "completed",
        "error": "",
        "streams": streams,
        "receiver": dict(receiver),
    }
    try:
        for spec in specifications:
            found = resolve_byprop("name", spec["stream_name"], timeout=10)
            if len(found) != 1:
                raise RuntimeError(
                    f"Expected one LSL outlet named {spec['stream_name']}; found {len(found)}"
                )
            name = spec["name"]
            inlet = StreamInlet(
                found[0], max_buflen=receiver["buffer_s"], recover=False, processing_flags=0
            )
            inlets[name] = inlet
            inlet.open_stream(timeout=10)
            offsets[name] = inlet.time_correction(timeout=10)
            info = inlet.info()
            channels = ElementTree.fromstring(info.as_xml()).findall("./desc/channels/channel")
            count = info.channel_count() - (2 if simulated and name != "markers" else 0)
            streams[name] = {
                "stream_name": info.name(),
                "kind": info.type(),
                "rate": info.nominal_srate(),
                "channels": count,
                "source_id": info.source_id(),
                "uid": info.uid(),
                "labels": [c.findtext("label", "") for c in channels][:count],
                "units": [c.findtext("unit", "") for c in channels][:count],
                "initial_offset": offsets[name],
            }
            buffers[name] = {
                key: []
                for key in (
                    "values",
                    "timestamps",
                    "corrected",
                    "received_at",
                    "correction",
                    "poll_times",
                    "queue_samples",
                )
            }
            if simulated and name != "markers":
                buffers[name].update(index=[], counter=[])
        for inlet in inlets.values():
            inlet.flush()  # Exclude data queued while clock setup was running.
        case["start"] = local_clock()
        if ready is not None:
            ready.set()
        period = 1 / receiver["read_hz"]
        next_read = case["start"]
        refresh = next_read + 2
        deadline = next_read + duration + (5 if simulated else 0)
        end_seen = False
        last_data = next_read
        print(
            "Receiving: " + ", ".join(f"{name} {s['rate']:g} Hz" for name, s in streams.items()),
            flush=True,
        )
        try:
            while local_clock() < deadline:
                now = local_clock()
                if now >= refresh:
                    for name, inlet in inlets.items():
                        try:
                            offsets[name] = inlet.time_correction(timeout=0.001)
                        except TimeoutError:
                            pass
                    refresh = local_clock() + 2
                for name, inlet in inlets.items():
                    buffers[name]["poll_times"].append(local_clock())
                    buffers[name]["queue_samples"].append(inlet.samples_available())
                    values, timestamps = inlet.pull_chunk(timeout=0, max_samples=4096)
                    received = local_clock()
                    if not timestamps:
                        continue
                    data = np.asarray(values, dtype=str if name == "markers" else float)
                    if simulated and name != "markers":
                        buffers[name]["index"].append(data[:, -2].astype(np.int64))
                        buffers[name]["counter"].append(data[:, -1])
                        data = data[:, :-2].astype(np.float32)
                    stamps = np.asarray(timestamps)
                    for key, value in (
                        ("values", data),
                        ("timestamps", stamps),
                        ("corrected", stamps + offsets[name]),
                        ("received_at", np.full(len(stamps), received)),
                        ("correction", np.full(len(stamps), offsets[name])),
                    ):
                        buffers[name][key].append(value)
                    if simulated and name == "markers" and "end" in data[:, 0]:
                        end_seen = True
                    last_data = received
                if end_seen and local_clock() - last_data > 0.2:
                    break
                now = local_clock()
                next_read += period * max(1, int((now - next_read) // period) + 1)
                time.sleep(max(0, next_read - now))
            if simulated and not end_seen:
                case["status"] = "incomplete"
        except KeyboardInterrupt:
            case["status"] = "stopped by user"
        except Exception as error:
            case["status"], case["error"] = "failed", str(error)
        case["end"] = local_clock()
    finally:
        for inlet in inlets.values():
            try:
                inlet.close_stream()
            except Exception:
                pass  # Closing a connection must not prevent saving its data.
    for name, fields in buffers.items():
        for key, pieces in fields.items():
            if key in ("poll_times", "queue_samples"):
                streams[name][key] = np.asarray(pieces)
            elif pieces:
                streams[name][key] = np.concatenate(pieces)
            else:
                shape = (0, streams[name]["channels"]) if key == "values" else (0,)
                streams[name][key] = np.empty(
                    shape, dtype=str if name == "markers" and key == "values" else float
                )
    if not len(streams["eeg"]["values"]):
        case["error"] = "No EEG samples received"
        if case["status"] == "completed":
            case["status"] = "incomplete"
    return case
