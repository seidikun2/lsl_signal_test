"""Save the recording, compare samples, and show results before explanations."""

import base64
import csv
import io
import json
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

COLORS = {"A": "#c45a00", "B": "#7951b0", "C": "#008578"}


def save_recordings(output, cases):
    """One NPZ contains arrays plus readable JSON metadata; no pickle or folders."""
    output.mkdir(parents=True, exist_ok=True)
    metadata, arrays = [], {}
    for case in cases:
        entry = {key: value for key, value in case.items() if key != "streams"}
        entry["streams"] = {}
        for name, stream in case["streams"].items():
            entry["streams"][name] = {}
            for key, value in stream.items():
                if isinstance(value, np.ndarray):
                    arrays[f"{case['id']}__{name}__{key}"] = value
                else:
                    entry["streams"][name][key] = value
        metadata.append(entry)
    np.savez_compressed(
        output / "recordings.npz", metadata=np.array(json.dumps(metadata)), **arrays
    )


def load_recordings(path):
    with np.load(path, allow_pickle=False) as saved:
        cases = json.loads(str(saved["metadata"]))
        by_id = {case["id"]: case for case in cases}
        for key in saved.files:
            if key != "metadata":
                case_id, name, field = key.split("__")
                by_id[case_id]["streams"][name][field] = saved[key]
    return cases


def timestamp_checks(stamps):
    """Describe the stored times without repairing or dropping any samples."""
    stamps = np.asarray(stamps, dtype=float)
    valid = stamps[np.isfinite(stamps)]
    steps = np.diff(stamps)
    backward = np.flatnonzero(steps < 0)
    return dict(
        samples=len(stamps),
        repeated_times=int(len(valid) - len(np.unique(valid))),
        backward_steps=int(len(backward)),
        invalid_times=int(len(stamps) - len(valid)),
        first_backward_sample=int(backward[0] + 2) if len(backward) else None,
        largest_backward_ms=float(-np.min(steps[backward]) * 1000) if len(backward) else None,
    )


def nearest_times(times, targets):
    """Find nearest time groups; a tied choice is ambiguous, not a match."""
    after = np.clip(np.searchsorted(targets, times), 0, len(targets) - 1)
    before = np.maximum(np.searchsorted(targets, times) - 1, 0)
    first, second = abs(times - targets[before]), abs(times - targets[after])
    nearest = np.where(first <= second, before, after)
    close = np.minimum(first, second) <= 1e-6
    unique = (before == after) | (first != second)
    return nearest, close, unique


def match_times(captured, reference, markers=False):
    """Use lookup indices only; keep the original arrays and sample order intact."""
    captured, reference = np.asarray(captured), np.asarray(reference)
    result = dict(
        pairs=np.empty((0, 2), dtype=int),
        comparison_status="not compared",
        matched_samples=0,
        compared_by_order=0,
        ambiguous_python=0,
        ambiguous_xdf=0,
        unmatched_python=None,
        unmatched_xdf=None,
        python_in_scope=0,
        xdf_in_scope=0,
        comparison_note="No shared valid timestamps. Confirm the recordings overlap and inspect the timestamp checks.",
    )
    left = np.flatnonzero(np.isfinite(captured))
    right = np.flatnonzero(np.isfinite(reference))
    if not len(left) or not len(right):
        return result
    start = max(np.min(captured[left]), np.min(reference[right]))
    end = min(np.max(captured[left]), np.max(reference[right]))
    left = left[(captured[left] >= start - 1e-6) & (captured[left] <= end + 1e-6)]
    right = right[(reference[right] >= start - 1e-6) & (reference[right] <= end + 1e-6)]
    if start > end or not len(left) or not len(right):
        return result
    # Sort lookup indices, never the recordings themselves. This also finds
    # unique timestamps correctly when an outlet's clock moves backwards.
    left = left[np.argsort(captured[left], kind="stable")]
    right = right[np.argsort(reference[right], kind="stable")]
    right_times = reference[right]
    capture_times, capture_starts, capture_counts = np.unique(
        captured[left], return_index=True, return_counts=True
    )
    reference_times, reference_starts, reference_counts = np.unique(
        reference[right], return_index=True, return_counts=True
    )
    to_right, left_close, left_unique = nearest_times(capture_times, reference_times)
    to_left, right_close, right_unique = nearest_times(reference_times, capture_times)
    ambiguous_left = np.zeros(len(captured), dtype=bool)
    ambiguous_right = np.zeros(len(reference), dtype=bool)
    groups, ordered = [], []
    for i, j in enumerate(to_right):
        if not left_close[i]:
            continue
        capture_group, reference_group = (
            left[capture_starts[i] : capture_starts[i] + capture_counts[i]],
            right[reference_starts[j] : reference_starts[j] + reference_counts[j]],
        )
        repeated = len(capture_group) > 1 or len(reference_group) > 1
        # A repeated-time block at an acquisition edge may be only part of a
        # batch. Unequal/noncontiguous blocks cannot identify individual samples.
        boundary = (
            not markers
            and repeated
            and (
                capture_group[0] == 0
                or capture_group[-1] == len(captured) - 1
                or reference_group[0] == 0
                or reference_group[-1] == len(reference) - 1
            )
        )
        safe = (
            left_unique[i]
            and right_unique[j]
            and to_left[j] == i
            and len(capture_group) == len(reference_group)
            and np.all(np.diff(capture_group) == 1)
            and np.all(np.diff(reference_group) == 1)
            and not boundary
        )
        if safe:
            groups.append(np.column_stack((capture_group, reference_group)))
            ordered.extend([repeated] * len(capture_group))
        else:
            ambiguous_left[capture_group] = True
            first = np.searchsorted(right_times, capture_times[i] - 1e-6, side="left")
            last = np.searchsorted(right_times, capture_times[i] + 1e-6, side="right")
            ambiguous_right[right[first:last]] = True
    for j, i in enumerate(to_left):
        if right_close[j] and (not right_unique[j] or not left_unique[i] or to_right[i] != j):
            ambiguous_right[
                right[reference_starts[j] : reference_starts[j] + reference_counts[j]]
            ] = True
    pairs = np.concatenate(groups) if groups else result["pairs"]
    order_conflict = False
    backward = np.any(np.diff(captured) < 0) or np.any(np.diff(reference) < 0)
    if len(pairs):
        pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
        if np.any(np.diff(pairs[:, 1]) <= 0):
            order_conflict = True
            ambiguous_left[pairs[:, 0]] = True
            ambiguous_right[pairs[:, 1]] = True
            pairs = result["pairs"]
            result["comparison_note"] = (
                "Time matches conflict with recorded sample order; no automatic alignment was applied."
            )
        else:
            ambiguous_left[pairs[:, 0]] = False
            ambiguous_right[pairs[:, 1]] = False
            result.update(
                unmatched_python=(
                    None if backward else int(len(left) - len(pairs) - np.sum(ambiguous_left[left]))
                ),
                unmatched_xdf=(
                    None
                    if backward
                    else int(len(right) - len(pairs) - np.sum(ambiguous_right[right]))
                ),
                compared_by_order=int(sum(ordered)),
                comparison_status="partial" if backward else "compared",
                comparison_note="Repeated-time blocks with equal sizes are compared in recorded order; ambiguous samples remain unverified.",
            )
            if backward:
                result[
                    "comparison_note"
                ] += " Unmatched counts are unknown when times move backwards."
    result.update(
        pairs=pairs,
        matched_samples=len(pairs),
        python_in_scope=len(left),
        xdf_in_scope=len(right),
        ambiguous_python=int(np.sum(ambiguous_left)),
        ambiguous_xdf=int(np.sum(ambiguous_right)),
    )
    if len(pairs) and (
        result["ambiguous_python"]
        or result["ambiguous_xdf"]
        or result["unmatched_python"]
        or result["unmatched_xdf"]
    ):
        result["comparison_status"] = "partial"
    if len(pairs) and (not np.isfinite(captured).all() or not np.isfinite(reference).all()):
        result["comparison_status"] = "partial"
    if (
        not len(pairs)
        and not order_conflict
        and (result["ambiguous_python"] or result["ambiguous_xdf"])
    ):
        result["comparison_note"] = (
            "No unambiguous sample pairs. The recorded times do not safely identify the same samples."
        )
    return result


def compare_xdf(case, path):
    """XDF import must preserve source timestamps and channel order."""
    import pyxdf  # Optional: python -m pip install pyxdf

    print(f"Reading XDF: {path}", flush=True)
    references, _ = pyxdf.load_xdf(
        path,
        select_streams=[{"name": stream["stream_name"]} for stream in case["streams"].values()],
        synchronize_clocks=False,
        dejitter_timestamps=False,
        handle_clock_resets=False,
    )
    case["timestamp_checks"] = []
    for name, stream in case["streams"].items():
        matches = [
            item
            for item in references
            if item["info"].get("name", [""])[0] == stream["stream_name"]
        ]
        for field in ("source_id", "uid"):
            matches = [
                item
                for item in matches
                if not stream[field]
                or not item["info"].get(field, [""])[0]
                or item["info"][field][0] == stream[field]
            ]
        if len(matches) != 1:
            if name == "markers":
                case["reference_note"] = "No unique matching marker stream in XDF."
                continue
            raise ValueError("No unique matching EEG outlet/session in XDF")
        item = matches[0]
        values = np.asarray(item["time_series"], dtype=str if name == "markers" else None)
        if name == "markers" and not values.size:
            case["reference_note"] = "No marker events in XDF to compare."
            continue
        if values.ndim != 2 or values.shape[1] != stream["channels"]:
            raise ValueError(f"Channel counts differ for {name}")
        description = (item["info"].get("desc") or [{}])[0] or {}
        channel_info = (description.get("channels") or [{}])[0] or {}
        channels = channel_info.get("channel", [])
        for field, tag in (("labels", "label"), ("units", "unit")):
            labels = [channel.get(tag, [""])[0] for channel in channels]
            if any(a and b and a != b for a, b in zip(stream[field], labels)):
                raise ValueError(
                    f"Channel {field}/order differ; no automatic conversion was applied"
                )
        for origin, stamps in (("Python", stream["timestamps"]), ("XDF", item["time_stamps"])):
            checks = timestamp_checks(stamps)
            case["timestamp_checks"].append(dict(stream=name, recording=origin, **checks))
            print(
                f"[{name}] {origin}: {checks['samples']} samples; "
                f"repeated times={checks['repeated_times']}; backward steps={checks['backward_steps']}; "
                f"invalid times={checks['invalid_times']}",
                flush=True,
            )
            if checks["backward_steps"]:
                print(
                    f"[{name}] {origin}: first backward sample={checks['first_backward_sample']}; "
                    f"largest backward step={checks['largest_backward_ms']:.6g} ms",
                    flush=True,
                )
        matching = match_times(stream["timestamps"], item["time_stamps"], name == "markers")
        print(
            f"[{name}] {matching['comparison_status']}: {matching['matched_samples']} matched; "
            f"ambiguous Python={matching['ambiguous_python']}, XDF={matching['ambiguous_xdf']}",
            flush=True,
        )
        stream.update(
            reference_values=values,
            reference_timestamps=np.asarray(item["time_stamps"]),
            pairs=matching.pop("pairs"),
            missing_reference=matching["unmatched_python"],
            missing_received=matching["unmatched_xdf"],
            comparison_info=matching,
        )
    marker_stream = case["streams"].get("markers", {})
    if "pairs" in marker_stream and len(marker_stream["pairs"]):
        different = sum(
            not np.array_equal(marker_stream["values"][a], marker_stream["reference_values"][b])
            for a, b in marker_stream["pairs"]
        )
        without_python = marker_stream["missing_received"]
        without_reference = marker_stream["missing_reference"]
        case["reference_note"] = (
            f"Markers: {len(marker_stream['pairs'])} matched; "
            f"{without_python if without_python is not None else 'unknown'} reference events without a Python match; "
            f"{without_reference if without_reference is not None else 'unknown'} Python events without a reference match; "
            f"{different} changed values. Counts cover the common time range."
        )
    elif "comparison_info" in marker_stream:
        case["reference_note"] = "Markers: " + marker_stream["comparison_info"]["comparison_note"]
    case["reference_note"] = case.get(
        "reference_note", "EEG comparison coverage is shown below. No marker stream recorded."
    )


def marker_results(case):
    """Keep marker sample times and Python read times side by side."""
    markers = case["streams"].get("markers")
    if markers is None:
        return []
    rows = []
    if case["simulated"]:
        stamps = markers.get("sent_timestamps", np.empty(0))
        values = markers.get("sent_values", np.empty((0, 1), dtype=str))
    else:
        stamps, values = markers["timestamps"], markers["values"]
    for index, (stamp, value) in enumerate(zip(stamps, values)):
        label = " / ".join(value)
        if case["simulated"] and label in ("start", "end"):
            continue
        row = dict(
            case=case["id"],
            signal="markers",
            label=label,
            marker_time_s=None,
            read_time_s=None,
            reference_matches=None,
            reference_timestamp=None,
            received_marker_time_s=None,
            marker_sent_at_s=None,
            marker_delay_ms=None,
            marker_after_send_ms=None,
            marker_value_preserved=None,
        )
        if case["simulated"]:
            row["marker_time_s"] = float(stamp - case["source_start"])
            found = np.flatnonzero(abs(markers["timestamps"] - stamp) < 1e-6)
            received = int(found[0]) if len(found) else None
        else:
            received = index
            row["marker_time_s"] = float(markers["corrected"][index] - case["start"])
        if received is not None:
            row["read_time_s"] = float(markers["received_at"][received] - case["start"])
            row["received_marker_time_s"] = float(markers["corrected"][received] - case["start"])
            row["marker_delay_ms"] = 1000 * (row["read_time_s"] - row["received_marker_time_s"])
            if case["simulated"]:
                row["marker_value_preserved"] = bool(
                    np.array_equal(value, markers["values"][received])
                )
        if case["simulated"] and "sent_push_times" in markers:
            row["marker_sent_at_s"] = float(
                markers["sent_push_times"][index] - case["source_start"]
            )
            if received is not None:
                pushed = markers["sent_push_times"][index] + markers["correction"][received]
                row["marker_after_send_ms"] = float(
                    (markers["received_at"][received] - pushed) * 1000
                )
        rows.append(row)
    for received, reference in markers.get("pairs", np.empty((0, 2), dtype=int)):
        rows[received]["reference_matches"] = bool(
            np.array_equal(markers["values"][received], markers["reference_values"][reference])
        )
        rows[received]["reference_timestamp"] = float(markers["reference_timestamps"][reference])
    return rows


def effect_results(case, name, stream, markers):
    """Find waveform changes in the actual arrays, rather than assuming they arrived."""
    if not case["simulated"] or "sent_values" not in stream:
        return []
    from transmission import AMPLITUDE, signal

    sensor = dict(kind=stream["kind"], rate=stream["rate"], channels=1)
    source_residual = stream["sent_values"][:, 0] - signal(sensor, stream["sent_index"])[:, 0]
    received_residual = stream["values"][:, 0] - signal(sensor, stream["index"])[:, 0]
    rows = []
    for marker in markers:
        event = next(
            (
                e
                for e in case.get("events", [])
                if e["label"] == marker["label"] and abs(e["time"] - marker["marker_time_s"]) < 1e-6
            ),
            None,
        )
        if event is None:
            continue
        row = dict(
            marker,
            signal=name,
            source_effect_time_s=None,
            received_effect_time_s=None,
            effect_read_time_s=None,
            source_marker_to_effect_ms=None,
            received_marker_to_effect_ms=None,
            added_sample_time_error_ms=None,
            read_marker_to_effect_ms=None,
            source_push_gap_ms=None,
            added_arrival_gap_ms=None,
            onset_sample_preserved=None,
        )
        detected = []
        for times, residual in (
            (stream["sent_counter"], source_residual),
            (stream["counter"], received_residual),
        ):
            # Covers old recordings with a 200 ms effect, without moving that effect.
            changed = (
                (times >= event["time"] - 1e-9)
                & (times < event["time"] + 0.5)
                & (
                    residual * np.sign(event["strength"])
                    > abs(AMPLITUDE[stream["kind"]] * event["strength"]) / 2
                )
            )
            found = np.flatnonzero(changed)
            detected.append(int(found[0]) if len(found) else None)
        sent, received = detected
        if sent is not None:
            row["onset_sample_preserved"] = False
            row["source_effect_time_s"] = float(stream["sent_counter"][sent])
            row["source_marker_to_effect_ms"] = 1000 * (
                row["source_effect_time_s"] - row["marker_time_s"]
            )
            if marker["marker_sent_at_s"] is not None:
                row["source_push_gap_ms"] = 1000 * (
                    stream["sent_push_times"][sent]
                    - case["source_start"]
                    - marker["marker_sent_at_s"]
                )
        if received is not None:
            row["received_effect_time_s"] = float(stream["corrected"][received] - case["start"])
            row["effect_read_time_s"] = float(stream["received_at"][received] - case["start"])
            if marker["received_marker_time_s"] is not None:
                row["received_marker_to_effect_ms"] = 1000 * (
                    row["received_effect_time_s"] - marker["received_marker_time_s"]
                )
                row["read_marker_to_effect_ms"] = 1000 * (
                    row["effect_read_time_s"] - marker["read_time_s"]
                )
            if sent is not None:
                row["onset_sample_preserved"] = bool(
                    stream["sent_index"][sent] == stream["index"][received]
                )
        if (
            row["onset_sample_preserved"]
            and row["marker_value_preserved"]
            and row["received_marker_to_effect_ms"] is not None
            and row["source_marker_to_effect_ms"] is not None
        ):
            row["added_sample_time_error_ms"] = (
                row["received_marker_to_effect_ms"] - row["source_marker_to_effect_ms"]
            )
        if (
            row["onset_sample_preserved"]
            and row["marker_value_preserved"]
            and row["read_marker_to_effect_ms"] is not None
            and row["source_push_gap_ms"] is not None
        ):
            row["added_arrival_gap_ms"] = (
                row["read_marker_to_effect_ms"] - row["source_push_gap_ms"]
            )
        rows.append(row)
    return rows


def stream_offsets(events):
    """Compare simultaneous sensor effects with EEG, removing native sample spacing."""
    eeg = {(e["case"], e["marker_time_s"]): e for e in events if e["signal"] == "eeg"}
    rows = []
    for event in events:
        reference = eeg.get((event["case"], event["marker_time_s"]))
        if event["signal"] in ("eeg", "markers") or reference is None:
            continue
        row = dict(
            case=event["case"],
            signal=event["signal"],
            label=event["label"],
            marker_time_s=event["marker_time_s"],
            source_vs_eeg_ms=None,
            stored_vs_eeg_ms=None,
            added_vs_eeg_ms=None,
            added_arrival_vs_eeg_ms=None,
        )
        if (
            event["source_effect_time_s"] is not None
            and reference["source_effect_time_s"] is not None
        ):
            row["source_vs_eeg_ms"] = 1000 * (
                event["source_effect_time_s"] - reference["source_effect_time_s"]
            )
        if (
            event["received_effect_time_s"] is not None
            and reference["received_effect_time_s"] is not None
        ):
            row["stored_vs_eeg_ms"] = 1000 * (
                event["received_effect_time_s"] - reference["received_effect_time_s"]
            )
        preserved = event["onset_sample_preserved"] and reference["onset_sample_preserved"]
        if (
            preserved
            and row["stored_vs_eeg_ms"] is not None
            and row["source_vs_eeg_ms"] is not None
        ):
            row["added_vs_eeg_ms"] = row["stored_vs_eeg_ms"] - row["source_vs_eeg_ms"]
        if (
            preserved
            and event["source_push_gap_ms"] is not None
            and reference["source_push_gap_ms"] is not None
        ):
            row["added_arrival_vs_eeg_ms"] = 1000 * (
                event["effect_read_time_s"] - reference["effect_read_time_s"]
            ) - (event["source_push_gap_ms"] - reference["source_push_gap_ms"])
        rows.append(row)
    return rows


def latency_series(case, stream):
    """Clock-corrected sample age, and transport time where push times are known."""
    reads = stream["received_at"] - case["start"]
    sample_age = (stream["received_at"] - stream["corrected"]) * 1000
    after_send = np.full(len(reads), np.nan)
    if "sent_index" in stream:
        _, sent, received = np.intersect1d(
            stream["sent_index"], stream["index"], return_indices=True
        )
        pushed = stream["sent_push_times"][sent] + stream["correction"][received]
        after_send[received] = (stream["received_at"][received] - pushed) * 1000
    return reads, sample_age, after_send


def latency_summary(times, values, prefix, unit="ms"):
    valid = np.isfinite(times) & np.isfinite(values)
    times, values = times[valid], values[valid]
    result = {prefix + key + "_" + unit: None for key in ("p99", "max", "first", "last", "change")}
    if len(values):
        width = min(1.0, (times[-1] - times[0]) / 3)
        first = float(np.median(values[times <= times[0] + width]))
        last = float(np.median(values[times >= times[-1] - width]))
        result.update(
            {
                prefix + "p99_" + unit: float(np.percentile(values, 99)),
                prefix + "max_" + unit: float(np.max(values)),
                prefix + "first_" + unit: first,
                prefix + "last_" + unit: last,
                prefix + "change_" + unit: last - first,
            }
        )
    return result


def delay_windows(times, values):
    """One-second windows on the receiver clock; no fitted trend or clock repair."""
    valid = np.isfinite(times) & np.isfinite(values)
    times, values = times[valid], values[valid]
    bins, first = np.unique(np.floor(times), return_index=True)
    pieces = np.split(values, first[1:]) if len(values) else []
    return (
        bins + 0.5,
        [float(np.median(p)) for p in pieces],
        [float(np.percentile(p, 99)) for p in pieces],
    )


def measurements(case, name, stream):
    """Direct measurements only: values, missing samples and measured delay."""
    reads, delay, after_send = latency_series(case, stream)
    row = dict(
        case=case["id"],
        signal=name,
        status=case["status"],
        rate_hz=stream["rate"],
        channels=stream["channels"],
        received=len(reads),
        missing=None,
        value_difference=None,
        different_values=None,
        delay_p99_ms=None,
        delay_change_ms=None,
        after_send_p99_ms=None,
        counter_difference_ms=None,
        invalid_values=int(np.sum(~np.isfinite(stream["values"]))),
    )
    polls = stream["poll_times"]
    row["reading_hz"] = float((len(polls) - 1) / (polls[-1] - polls[0])) if len(polls) > 1 else None
    row.update(latency_summary(reads, delay, "delay_"))
    row.update(latency_summary(reads, after_send, "after_send_"))
    checks = timestamp_checks(stream["timestamps"])
    row["sample_clock_regular"] = not any(
        checks[k] for k in ("repeated_times", "backward_steps", "invalid_times")
    )
    queue = stream.get("queue_samples", np.empty(0))
    row["queue_max_samples"] = int(np.max(queue)) if len(queue) else None
    row["queue_change_samples"] = None
    if len(queue):
        summary = latency_summary(polls, queue, "queue_", unit="samples")
        row["queue_change_samples"] = summary["queue_change_samples"]
    if "sent_index" in stream:
        _, sent, received = np.intersect1d(
            stream["sent_index"], stream["index"], return_indices=True
        )
        row["missing"] = int(len(np.setdiff1d(stream["sent_index"], stream["index"])))
        expected, actual = stream["sent_values"][sent], stream["values"][received]
        if len(received):
            counter_error = stream["counter"][received] - stream["sent_counter"][sent]
            row["counter_difference_ms"] = float(np.max(abs(counter_error)) * 1000)
        row.update(
            latency_summary(
                stream["sent_counter"],
                (stream["sent_push_times"] - stream["sent_timestamps"]) * 1000,
                "before_send_",
            )
        )
    elif "pairs" in stream:
        received, reference = stream["pairs"].T
        row["missing"] = stream["missing_received"]
        row["reference_missing"] = stream["missing_reference"]
        expected, actual = stream["reference_values"][reference], stream["values"][received]
        row.update(stream["comparison_info"])
    else:
        expected = actual = np.empty(0)
    if len(actual):
        valid = np.isfinite(actual) & np.isfinite(expected)
        equal = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
        row["different_values"] = int(np.sum(~equal))
        if valid.any():
            row["value_difference"] = float(np.max(abs(actual[valid] - expected[valid])))
    return row


def image(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def signal_plot(case, name, stream, events):
    """All recorded points, first channel. One selected plot is rendered per section."""
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.25, 0.22, 0.23, 0.13, 0.17],
        subplot_titles=(
            "Sample values at their timestamps — channel 1 (held until next sample)",
            "Latest value available at actual send / read time — channel 1",
            "Latency over the recording",
            "Samples waiting before each read",
            "Markers and detected waveform changes",
        ),
    )
    # Keep these axes visible even in older files without queue snapshots or markers.
    for row in (4, 5):
        fig.add_trace(
            go.Scatter(x=[0], y=[None], showlegend=False, hoverinfo="skip"), row=row, col=1
        )

    def trace(x, y, label, row, color, dash="solid", shape="linear", mode="lines", symbol="circle"):
        fig.add_trace(
            go.Scattergl(
                x=np.asarray(x),
                y=np.asarray(y),
                name=label,
                mode=mode,
                line=dict(color=color, width=1.5, dash=dash, shape=shape),
                marker=dict(color=color, size=9, symbol=symbol, line=dict(width=2)),
                hovertemplate="%{x:.6f} s"
                + ("<br>%{y:.5g}" if row != 5 else "")
                + "<extra>%{fullData.name}</extra>",
            ),
            row=row,
            col=1,
        )

    reads, age, after_send = latency_series(case, stream)
    values = stream["values"][:, 0].astype(float).copy()
    sample_times = stream["corrected"] - case["start"]
    gaps = (
        np.diff(stream["index"]) != 1
        if case["simulated"]
        else (
            np.diff(stream["timestamps"]) > 1.5 / stream["rate"]
            if stream["rate"]
            else np.zeros(max(0, len(values) - 1), dtype=bool)
        )
    )
    breaks = np.flatnonzero(gaps) + 1
    value_times = np.insert(sample_times, breaks, np.nan)
    values = np.insert(values, breaks, np.nan)
    if "sent_values" in stream:
        trace(
            stream["sent_counter"],
            stream["sent_values"][:, 0],
            "Sent signal",
            1,
            "#2166ac",
            shape="hv",
        )
        trace(
            stream["sent_push_times"] - case["source_start"],
            stream["sent_values"][:, 0],
            "Sent — available",
            2,
            "#2166ac",
            shape="hv",
        )
    trace(value_times, values, "Received signal", 1, "#d66000", "dash", "hv")
    # A chunk shares one read time. Its last sample becomes the latest available value.
    last = np.r_[np.diff(reads) != 0, True] if len(reads) else np.empty(0, dtype=bool)
    trace(
        reads[last], stream["values"][last, 0], "Received — available", 2, "#d66000", "dash", "hv"
    )
    trace(reads, age, "Sample → read", 3, "#93b7d5")
    for series, label, color, dash in (
        (age, "Sample → read", "#2166ac", "solid"),
        (after_send, "Send call → read", "#b2182b", "dash"),
    ):
        if np.isfinite(series).any():
            x, median, p99 = delay_windows(reads, series)
            trace(x, median, label + " — median / second", 3, color, dash)
            trace(x, p99, label + " — p99 / second", 3, color, "dot")
    if "sent_push_times" in stream:
        trace(
            stream["sent_counter"],
            (stream["sent_push_times"] - stream["sent_timestamps"]) * 1000,
            "Sample → send call (sender wait)",
            3,
            "#888888",
            "dot",
        )
    queue = stream.get("queue_samples", [])
    if len(queue):
        trace(stream["poll_times"] - case["start"], queue, "Waiting in this inlet", 4, "#008578")
    else:
        fig.add_annotation(
            text="Not recorded in this file",
            x=0.5,
            y=0.5,
            xref="x4 domain",
            yref="y4 domain",
            showarrow=False,
        )

    methods = [
        ("marker_time_s", "Marker — source time", "circle"),
        ("received_marker_time_s", "Marker — stored time", "circle-open"),
        ("read_time_s", "Marker — read time", "diamond-open"),
        ("source_effect_time_s", "Effect — sent sample", "triangle-up"),
        ("received_effect_time_s", "Effect — stored sample", "triangle-up-open"),
        ("effect_read_time_s", "Effect — read time", "square-open"),
    ]
    labels = list(dict.fromkeys(event["label"] for event in events))
    for label in labels:
        color = COLORS.get(label, "#555555")
        selected = [e for e in events if e["label"] == label]
        for field, caption, symbol in methods:
            points = [e[field] for e in selected if e.get(field) is not None]
            if points:
                trace(
                    points,
                    [caption] * len(points),
                    label + ": " + caption,
                    5,
                    color,
                    mode="markers",
                    symbol=symbol,
                )
        # Thick marker lines: same color, solid at source, dashed at reception.
        finite_values = stream["values"][:, 0][np.isfinite(stream["values"][:, 0])]
        if len(finite_values):
            low, high = float(finite_values.min()), float(finite_values.max())
            for field, row, dash in (("marker_time_s", 1, "solid"), ("read_time_s", 2, "dash")):
                points = [e[field] for e in selected if e.get(field) is not None]
                fig.add_trace(
                    go.Scattergl(
                        x=[v for t in points for v in (t, t, None)],
                        y=[v for _ in points for v in (low, high, None)],
                        name=label + " marker",
                        showlegend=False,
                        line=dict(color=color, width=3, dash=dash),
                        hovertemplate=label + " marker<br>%{x:.6f} s<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Delay (ms)", row=3, col=1)
    fig.update_yaxes(title_text="Samples", row=4, col=1, rangemode="tozero")
    fig.update_yaxes(
        type="category", categoryorder="array", categoryarray=[m[1] for m in methods], row=5, col=1
    )
    duration = max(0.1, case["end"] - case["start"])
    fig.update_xaxes(range=[0, duration], title_text="", showgrid=True)
    fig.update_xaxes(title_text="Time from shared start (s)", row=5, col=1)
    fig.update_layout(
        template="plotly_white",
        height=1300,
        dragmode="zoom",
        hovermode="closest",
        title=f"{case['id']} / {name.upper()} — {stream['rate']:g} Hz, {stream['channels']} channels",
        margin=dict(l=160, r=35, t=90, b=210),
        legend=dict(orientation="h", y=-0.08),
        font=dict(size=12),
    )
    return fig


def interactive_view(identity, plots):
    """Embed Plotly data without a server, CDN, or separate HTML assets."""
    options = "".join(
        f'<option value="{i}">{escape(p["label"])}</option>' for i, p in enumerate(plots)
    )
    payload = (
        "["
        + ",".join(
            '{"figure":'
            + plot["figure"].to_json()
            + ',"events":'
            + json.dumps(plot["events"])
            + ',"summary":'
            + json.dumps(plot["summary"])
            + "}"
            for plot in plots
        )
        + "]"
    )
    payload = payload.replace("</", "<\\/")
    return f"""<div class="viewer" id="{identity}">
<label>Recording / signal <select class="case-select">{options}</select></label>
<label>Marker <select class="event-select"></select></label>
<button class="reset" type="button">Full recording</button>
<p>Drag to zoom; scroll to zoom; use the toolbar to pan or save an image. Double-click to reset.
Click a legend item to hide or show it. All five time axes zoom together. All recorded points of channel 1 are included.</p>
<div class="cards"></div><div class="chart"></div>
<script type="application/json" class="plot-data">{payload}</script></div>"""


INTERACTIVE_SCRIPT = r"""
<script>
for (const viewer of document.querySelectorAll('.viewer')) {
  const plots = JSON.parse(viewer.querySelector('.plot-data').textContent);
  const chart = viewer.querySelector('.chart');
  const cases = viewer.querySelector('.case-select');
  const markers = viewer.querySelector('.event-select');
  const cards = viewer.querySelector('.cards');
  const config = {responsive:true, scrollZoom:true, displaylogo:false};
  function fullRange() {
    const update = {};
    for (let i = 1; i <= 5; i++) {
      update['xaxis' + (i === 1 ? '' : i) + '.range'] = plots[cases.value].figure.layout.xaxis.range;
      update['yaxis' + (i === 1 ? '' : i) + '.autorange'] = true;
    }
    Plotly.relayout(chart, update);
  }
  function show() {
    const plot = plots[cases.value];
    markers.replaceChildren(new Option('Choose a marker…', ''));
    for (const event of plot.events) markers.add(new Option(event.label, event.time));
    cards.innerHTML = plot.summary;
    // Plotly mutates its input during zoom; keep the stored full extent intact.
    Plotly.react(chart, plot.figure.data, structuredClone(plot.figure.layout), config);
  }
  cases.onchange = show;
  markers.onchange = () => {
    if (markers.value === '') return;
    const t = Number(markers.value), update = {};
    for (let i = 1; i <= 5; i++) update['xaxis' + (i === 1 ? '' : i) + '.range'] = [t - 0.08, t + 0.4];
    Plotly.relayout(chart, update);
  };
  viewer.querySelector('.reset').onclick = fullRange;
  show();
}
</script>
"""


def maps(rows, eeg):
    x_key, y_key = ("rate_hz", "channels") if eeg else ("stream_count", "signal")
    xs, ys = sorted({row[x_key] for row in rows}), sorted({row[y_key] for row in rows})
    fields = [
        ("delay_p99_ms", "Sample-to-read p99 (ms)"),
        ("delay_change_ms", "Delay change (ms)"),
        ("missing", "Missing samples"),
        ("value_difference", "Largest value difference"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, fields):
        values = np.full((len(ys), len(xs)), np.nan)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                cell = [
                    row[field]
                    for row in rows
                    if row[x_key] == x and row[y_key] == y and row[field] is not None
                ]
                if cell:
                    values[i, j] = max(cell)
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}" if np.isfinite(values[i, j]) else "—",
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
        finite = values[np.isfinite(values)]
        limit = max(abs(finite)) if len(finite) else 1
        limit = limit or 1
        plot = ax.imshow(
            np.ma.masked_invalid(values),
            aspect="auto",
            cmap=plt.get_cmap(
                "coolwarm" if field == "delay_change_ms" else "viridis"
            ).with_extremes(bad="#ddd"),
            vmin=-limit if field == "delay_change_ms" else 0,
            vmax=limit,
        )
        ax.set(
            title=title,
            xticks=range(len(xs)),
            xticklabels=xs,
            yticks=range(len(ys)),
            yticklabels=ys,
            xlabel="Sampling rate (Hz)" if eeg else "Active streams",
            ylabel="EEG channels" if eeg else "Signal",
        )
        fig.colorbar(plot, ax=ax, shrink=0.75)
    fig.suptitle(
        "Highest measured value across repetitions. Gray = no measurement. Read each color scale."
    )
    return image(fig)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table(rows):
    if not rows:
        return "<p>No measurements available.</p>"
    fields = list(dict.fromkeys(key for row in rows for key in row))

    def cell(value):
        return (
            "—"
            if value is None
            else escape(f"{value:.5g}" if isinstance(value, float) else str(value))
        )

    header = "".join(f"<th>{escape(key.replace('_', ' '))}</th>" for key in fields)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(row.get(key))}</td>" for key in fields) + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><tr>{header}</tr>{body}</table></div>'


def selected_table(rows, columns):
    return table([{label: row.get(field) for field, label in columns} for row in rows])


def summary_cards(row):
    cards = []
    for title, field, unit in (
        ("Sample → read: 99% ≤", "delay_p99_ms", "ms"),
        ("Sample → read: end − start", "delay_change_ms", "ms"),
        ("Send call → read: 99% ≤", "after_send_p99_ms", "ms"),
        ("Send call → read: end − start", "after_send_change_ms", "ms"),
        ("Most samples waiting", "queue_max_samples", "samples"),
    ):
        value = row.get(field)
        shown = "Not measured" if value is None else f"{value:.3f} {unit}"
        cards.append(f'<div class="card"><small>{title}</small><strong>{shown}</strong></div>')
    note = (
        ""
        if row["sample_clock_regular"]
        else "<p class=warning>Sample timestamps repeat or go backwards. Sample-to-read delay depends on those timestamps; it cannot establish buffer accumulation.</p>"
    )
    return "".join(cards) + note


def write_report(output, cases):
    metrics, markers = [], []
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>LSL results</title>',
        "<meta name=viewport content='width=device-width, initial-scale=1'><style>body{font:16px system-ui;max-width:1400px;margin:20px auto;padding:20px;color:#163047}img{max-width:100%}section{margin:40px 0;border-top:2px solid #ddd}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}summary{cursor:pointer;padding:8px}p{line-height:1.5}select,button{font:inherit;padding:8px;margin:6px;max-width:100%}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{background:#eef4f8;border-radius:8px;padding:14px;flex:1;min-width:180px}.card strong{display:block;margin-top:8px;font-size:21px}.warning{background:#fff0cb;padding:12px}</style>",
        "<script>" + get_plotlyjs() + "</script>",
        "<h1>LSL — recorded results</h1><p><b>Values preserved?</b> Check sample counts and value differences. <b>Arriving on time?</b> Check delay over time and samples waiting in the inlet. <b>Markers synchronized?</b> Zoom into a marker and compare the measured waveform changes. These are separate measurements.</p>",
        "<p><b>Sample → read</b> is the age of a sample when Python reads it, using LSL clock correction. <b>Send call → read</b> starts at the sender's actual push call (simulation only). <b>End − start</b> compares median delays in the first and last second, shortened to one third of the recorded interval for brief captures. Positive means a larger delay at the end. The per-second curves show whether that increase persists or disappears.</p>",
    ]
    for group in dict.fromkeys(case["group"] for case in cases):
        parts.append(f"<section><h2>{escape(group)}</h2>")
        plots, group_rows, group_events = [], [], []
        for case in (item for item in cases if item["group"] == group):
            events = marker_results(case)
            markers.extend(events)
            group_events.extend(events)
            count = len(case["streams"]) - ("markers" in case["streams"])
            for name, stream in case["streams"].items():
                if name == "markers":
                    continue
                row = measurements(case, name, stream)
                row["stream_count"] = count
                group_rows.append(row)
                effects = effect_results(case, name, stream, events)
                markers.extend(effects)
                group_events.extend(effects)
                if len(stream["values"]):
                    plotted_events = effects if case["simulated"] else events
                    jumps = []
                    for event in plotted_events:
                        for field, caption in (
                            ("marker_time_s", "source"),
                            ("read_time_s", "read"),
                        ):
                            if event.get(field) is not None:
                                t = event[field]
                                jumps.append(
                                    dict(label=f"{event['label']} — {caption} {t:.6f} s", time=t)
                                )
                    plots.append(
                        dict(
                            label=f"{case['id']} / {name.upper()} — {stream['rate']:g} Hz",
                            figure=signal_plot(case, name, stream, plotted_events),
                            events=jumps,
                            summary=summary_cards(row),
                        )
                    )
            if case.get("reference_note"):
                parts.append(f'<p>{escape(case["reference_note"])}</p>')
            if case.get("timestamp_checks"):
                parts.append(
                    "<details><summary>XDF coverage and original timestamp checks</summary><h3>Original timestamp checks</h3>"
                    + table(case["timestamp_checks"])
                )
                coverage = [
                    dict(
                        signal=name,
                        **{
                            key: value
                            for key, value in stream["comparison_info"].items()
                            if key != "comparison_note"
                        },
                    )
                    for name, stream in case["streams"].items()
                    if "comparison_info" in stream
                ]
                parts.append("<h3>Comparison coverage</h3>" + table(coverage))
                notes = dict.fromkeys(
                    stream["comparison_info"]["comparison_note"]
                    for stream in case["streams"].values()
                    if "comparison_info" in stream
                )
                parts.extend(f"<p>{escape(note)}</p>" for note in notes)
                parts.append("</details>")
            if case["error"]:
                parts.append(f'<p>{escape(case["id"] + ": " + case["error"])}</p>')
        metrics.extend(group_rows)
        if group != "Real EEG" and group_rows:
            parts.append(f'<img src="{maps(group_rows, group == "EEG only")}">')
        if group == "Real EEG":
            stream = cases[0]["streams"]["eeg"]
            if "pairs" in stream and len(stream["pairs"]):
                received, reference = stream["pairs"].T
                differences = np.ma.masked_invalid(
                    abs(stream["values"][received] - stream["reference_values"][reference])
                )
                fig, (wave, ax) = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
                shown = min(1024, len(received))
                wave.plot(stream["values"][received[:shown], 0], label="Python", lw=1.3)
                wave.plot(stream["reference_values"][reference[:shown], 0], "--", label="XDF", lw=1)
                wave.set(
                    title="Matched values — recorded sample order, not reconstructed time",
                    xlabel="Matched sample pair",
                    ylabel="Original value",
                )
                wave.legend()
                channel_errors = differences.max(axis=0)
                valid_errors = channel_errors.compressed()
                scale = float(np.max(valid_errors)) if len(valid_errors) else 1
                plot = ax.imshow(
                    channel_errors[None, :],
                    aspect="auto",
                    cmap=plt.get_cmap("viridis").with_extremes(bad="#ddd"),
                    vmin=0,
                    vmax=scale or 1,
                )
                ax.set(
                    title="Largest value difference per channel — Python versus XDF",
                    xlabel="Channel",
                    yticks=[],
                )
                fig.colorbar(plot, ax=ax)
                parts.append(f'<img src="{image(fig)}">')
        parts.extend(
            [
                "<h3>Recorded signals</h3>",
                (
                    interactive_view(f"viewer-{len(metrics)}", plots)
                    if plots
                    else "<p>No signal recorded.</p>"
                ),
                "<h3>1. Sample values — every channel</h3>",
                selected_table(
                    group_rows,
                    [
                        ("case", "Recording"),
                        ("signal", "Signal"),
                        ("rate_hz", "Rate (Hz)"),
                        ("channels", "Channels"),
                        ("received", "Received"),
                        ("missing", "Without a match"),
                        ("different_values", "Changed values"),
                        ("value_difference", "Largest difference"),
                    ],
                ),
                "<h3>2. Arrival delay</h3>",
                selected_table(
                    group_rows,
                    [
                        ("case", "Recording"),
                        ("signal", "Signal"),
                        ("delay_first_ms", "Start median (ms)"),
                        ("delay_last_ms", "End median (ms)"),
                        ("delay_change_ms", "End − start (ms)"),
                        ("delay_max_ms", "Largest delay (ms)"),
                        ("after_send_p99_ms", "Send → read p99 (ms)"),
                        ("after_send_change_ms", "Send → read change (ms)"),
                        ("queue_max_samples", "Most waiting samples"),
                    ],
                ),
                "<p>Each buffer point is a snapshot before a read, for this inlet only. Rising delay together with rising waiting counts supports accumulation here. Delay alone does not identify its cause. Unmeasured fields are shown as —. No acceptable latency limit has been assumed.</p>",
                "<h3>3. Marker and waveform timing</h3>",
                (
                    "<p>New simulations publish each marker before activating its waveform change. Each sensor changes at its first available sample; different sampling rates have different time steps. The report detects changes in the saved sent and received values of channel 1. Older recordings keep their original effect timing.</p>"
                    if group != "Real EEG"
                    else "<p>Real markers are shown at their corrected timestamp and actual Python read time. No artificial effect is added to real EEG. Amplifier send-call times and physical stimulus onset are not available from these recordings.</p>"
                ),
                (
                    selected_table(
                        [e for e in group_events if e["signal"] != "markers"],
                        [
                            ("case", "Recording"),
                            ("signal", "Signal"),
                            ("label", "Marker"),
                            ("marker_time_s", "Marker time (s)"),
                            ("source_marker_to_effect_ms", "Sent marker → change (ms)"),
                            ("received_marker_to_effect_ms", "Stored marker → change (ms)"),
                            ("added_sample_time_error_ms", "Added stored-time error (ms)"),
                            ("read_marker_to_effect_ms", "Read marker → change (ms)"),
                            ("added_arrival_gap_ms", "Added arrival gap (ms)"),
                            ("onset_sample_preserved", "First changed sample preserved"),
                        ],
                    )
                    if group != "Real EEG"
                    else selected_table(
                        group_events,
                        [
                            ("label", "Marker"),
                            ("marker_time_s", "Marker timestamp (s)"),
                            ("read_time_s", "Python read time (s)"),
                            ("marker_delay_ms", "Marker delay (ms)"),
                        ],
                    )
                ),
                (
                    "<h4>Other sensors relative to simultaneous EEG</h4><p>The sent offset includes each sensor's native sample spacing. Added stored-time error subtracts that offset. Added arrival gap subtracts the source push-call gap. Positive means this sensor was later than EEG.</p>"
                    + selected_table(
                        stream_offsets(group_events),
                        [
                            ("case", "Recording"),
                            ("signal", "Signal"),
                            ("label", "Marker"),
                            ("marker_time_s", "Marker time (s)"),
                            ("source_vs_eeg_ms", "Sent offset (ms)"),
                            ("stored_vs_eeg_ms", "Stored offset (ms)"),
                            ("added_vs_eeg_ms", "Added stored-time error (ms)"),
                            ("added_arrival_vs_eeg_ms", "Added arrival gap (ms)"),
                        ],
                    )
                    if group == "Additional streams"
                    else ""
                ),
                "<details><summary>All measurements — also saved in metrics.csv</summary>",
                table(group_rows),
                "</details>",
                "<details><summary>Marker times — every event</summary>",
                table(group_events),
                "</details></section>",
            ]
        )
    parts.append(
        "<p>p99 means 99% of measured delays are at or below that value. Send and read times bracket software calls, not a physical transmission measurement; clock-correction uncertainty remains. A read marker → change gap can be negative when Python reads the EEG chunk first. Added arrival gap subtracts the measured source push-call gap. A preserved first changed sample is required to interpret an effect-onset difference as timing alone.</p><details><summary>XDF comparison and timing limits</summary><p>Zero value difference applies only to matched samples; ambiguous samples remain unverified. XDF comparison uses original source timestamps within 1 microsecond. Equal repeated-time blocks are compared in recorded order; partial, unequal or conflicting blocks are ambiguous. Original signals are never reordered or repaired. Backward clocks leave unmatched counts unknown. Unmatched samples are not proof of loss. Repeated or backward source times make sample-age curves unreliable as evidence of delay accumulation. Real EEG contains no artificial response: two LSL recordings cannot establish physical stimulus or amplifier timing.</p></details>"
    )
    parts.append(INTERACTIVE_SCRIPT + "</html>")
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "markers.csv", markers)
    (output / "summary.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"Results: {output / 'summary.html'}", flush=True)
