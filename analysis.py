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
        rows.append(row)
    for received, reference in markers.get("pairs", np.empty((0, 2), dtype=int)):
        rows[received]["reference_matches"] = bool(
            np.array_equal(markers["values"][received], markers["reference_values"][reference])
        )
        rows[received]["reference_timestamp"] = float(markers["reference_timestamps"][reference])
    return rows


def measurements(case, name, stream):
    """Direct measurements only: values, missing samples and measured delay."""
    times = stream["counter"] if case["simulated"] else stream["corrected"] - case["start"]
    reads = stream["received_at"] - case["start"]
    delay = (reads - times) * 1000
    row = dict(
        case=case["id"],
        signal=name,
        status=case["status"],
        rate_hz=stream["rate"],
        channels=stream["channels"],
        received=len(times),
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
    if len(delay):
        quarter = max(1, len(delay) // 4)
        row["delay_p99_ms"] = float(np.percentile(delay, 99))
        row["delay_change_ms"] = float(np.median(delay[-quarter:]) - np.median(delay[:quarter]))
    if "sent_index" in stream:
        _, sent, received = np.intersect1d(
            stream["sent_index"], stream["index"], return_indices=True
        )
        row["missing"] = int(len(np.setdiff1d(stream["sent_index"], stream["index"])))
        expected, actual = stream["sent_values"][sent], stream["values"][received]
        if len(received):
            push_times = stream["sent_push_times"][sent] - case["source_start"]
            row["after_send_p99_ms"] = float(
                np.percentile((reads[received] - push_times) * 1000, 99)
            )
            counter_error = stream["counter"][received] - stream["sent_counter"][sent]
            row["counter_difference_ms"] = float(np.max(abs(counter_error)) * 1000)
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
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), constrained_layout=True)
    times = stream["counter"] if case["simulated"] else stream["corrected"] - case["start"]
    reads = stream["received_at"] - case["start"]
    window = min(2.0, case["end"] - case["start"])
    if "sent_values" in stream:
        source_times, source_values = stream["sent_counter"], stream["sent_values"][:, 0]
        axes[0].plot(source_times, source_values, label="Sent signal", lw=1.3)
    elif "reference_values" in stream:
        source_times = stream["reference_timestamps"] + stream["initial_offset"] - case["start"]
        source_values = stream["reference_values"][:, 0]
        visible = np.isfinite(source_times) & (source_times >= 0) & (source_times <= window)
        axes[0].plot(source_times[visible], source_values[visible], label="Reference", lw=1.3)
    else:
        source_times, source_values = np.empty(0), np.empty(0)
    value_times = (
        stream["timestamps"] + stream["initial_offset"] - case["start"]
        if "reference_values" in stream
        else times
    )
    values = stream["values"][:, 0].astype(float).copy()
    if case["simulated"]:
        gaps = np.diff(stream["index"]) != 1
    elif stream["rate"] > 0:
        gaps = np.diff(stream["timestamps"]) > 1.5 / stream["rate"]
    else:
        gaps = np.zeros(max(0, len(values) - 1), dtype=bool)
    values[np.r_[False, gaps]] = np.nan
    axes[0].plot(value_times, values, "--", label="Received signal", lw=1)
    axes[0].set(title="All sample values — original sample times", ylabel="Value", xlim=(0, window))
    grid = np.arange(int(window * 2048) + 1) / 2048
    for t, values, label, style, repeated_ok in (
        (source_times, source_values, "Sent / reference", "-", False),
        (reads, stream["values"][:, 0], "Received", "--", True),
    ):
        held = np.full(len(grid), np.nan)
        if len(t):
            # One pull can legitimately give several samples the same read time.
            bad_steps = np.diff(t) < 0 if repeated_ok else np.diff(t) <= 0
            if not np.isfinite(t).all() or np.any(bad_steps):
                axes[1].text(
                    0.02,
                    0.95,
                    f"{label}: live view unavailable (irregular sample times).",
                    transform=axes[1].transAxes,
                    va="top",
                    fontsize=9,
                )
                continue
            indices = np.searchsorted(t, grid, side="right") - 1
            present = indices >= 0
            held[present] = values[indices[present]]
            axes[1].step(grid, held, where="post", label=label, ls=style)
    axes[1].set(
        title="Live view at 2048 Hz — latest available value", ylabel="Value", xlim=(0, window)
    )
    axes[2].plot(times, (reads - times) * 1000, lw=0.8)
    axes[2].set(
        title="Python read time minus the recorded sample time",
        ylabel="Delay (ms)",
        xlabel="Time from shared start (s)",
    )
    selected = [event for event in events if event["signal"] in (name, "markers")]
    labels = list(dict.fromkeys(event["label"] for event in selected))
    for event in selected:
        y, color = labels.index(event["label"]), COLORS.get(event["label"], "#245cb0")
        for stamp, shift, style, fill in (
            (event["marker_time_s"], 0.18, "-", color),
            (event["read_time_s"], -0.18, "--", "white"),
        ):
            if stamp is not None:
                axes[3].vlines(
                    stamp, y + shift - 0.14, y + shift + 0.14, color=color, linestyles=style, lw=2.8
                )
                axes[3].plot(stamp, y + shift, "o", color=color, mfc=fill, ms=8, mew=2)
                if 0 <= stamp <= window:
                    axes[0 if style == "-" else 1].axvline(stamp, color=color, ls=style, lw=2)
    axes[3].set_yticks(
        [i + shift for i in range(len(labels)) for shift in (0.18, -0.18)],
        [f"{label} — {kind}" for label in labels for kind in ("source", "received")],
    )
    axes[3].set(
        title="Markers — solid / filled: marker time; dashed / hollow: Python read time.",
        xlabel="Time from shared start (s)",
        xlim=(0, window),
        ylim=(-0.5, max(0.5, len(labels) - 0.5)),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{case['id']} / {name.upper()} — {stream['rate']:g} Hz, {stream['channels']} channels"
    )
    return image(fig)


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


def write_report(output, cases):
    metrics, markers = [], []
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>LSL results</title>',
        "<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px}img{max-width:100%}section{margin:30px 0}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}summary{cursor:pointer;padding:8px}p{line-height:1.5}</style>",
        "<h1>LSL — recorded results</h1><p>Original sample times and actual read times are kept separately. No curve alignment is applied. Simulated markers cause a waveform change 200 ms later; inspect the recorded waveforms directly.</p>",
    ]
    for group in dict.fromkeys(case["group"] for case in cases):
        parts.append(f"<section><h2>{escape(group)}</h2>")
        figures, group_rows, group_events = [], [], []
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
                if len(stream["values"]):
                    figures.append(
                        f'<details><summary>{escape(case["id"] + " / " + name)} — {stream["rate"]:g} Hz</summary><img loading="lazy" src="{signal_plot(case, name, stream, events)}"></details>'
                    )
            if case.get("reference_note"):
                parts.append(f'<p>{escape(case["reference_note"])}</p>')
            if case.get("timestamp_checks"):
                parts.append("<h3>Original timestamp checks</h3>" + table(case["timestamp_checks"]))
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
                *figures,
                "<h3>Measurements</h3>",
                table(group_rows),
                "<details><summary>Marker times — every event</summary>",
                table(group_events),
                "</details></section>",
            ]
        )
    parts.append(
        "<p>p99: 99% of samples had a delay at or below this value. Delay change compares the last and first quarter of the recording. Zero value difference applies only to matched samples; ambiguous samples remain unverified. Gray means no measurement. Different sampling rates need not be integer multiples. Buffers hold data; timestamps and clock correction place samples in time.</p><p>XDF comparison uses original source timestamps within 1 microsecond. Equal repeated-time blocks are compared in recorded order; partial, unequal or conflicting blocks are marked ambiguous. Lookup indices do not reorder the original signals. When clocks go backwards, unmatched counts remain unknown rather than treating a clock reversal as loss. Unmatched samples are not proof of loss. Repeated or backward sample times also affect the delay curve; they do not prove that a buffer is filling. Real EEG contains no artificial response. This checks LSL transport and storage, not physical stimulus onset.</p></html>"
    )
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "markers.csv", markers)
    (output / "summary.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"Results: {output / 'summary.html'}", flush=True)
