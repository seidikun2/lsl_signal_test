"""Start here: choose simulation, real EEG capture, or XDF comparison."""

import argparse
import json
import multiprocessing as mp
import sys
import tempfile
import uuid
from itertools import product
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

import analysis
import reception
import transmission


def simulation_plan(settings):
    """EEG first; adding sensors is a separate experiment with fixed native rates."""
    if not settings["sensors"] or settings["sensors"][0]["name"] != "eeg":
        raise ValueError("Keep the EEG sensor first in simulation.sensors")
    names = [sensor["name"] for sensor in settings["sensors"]]
    if len(names) != len(set(names)) or "markers" in names:
        raise ValueError("Sensor names must be unique; markers is reserved")
    plan = []
    for rate, channels in product(settings["eeg_rates_hz"], settings["eeg_channels"]):
        sensor = {"name": "eeg", "kind": "EEG", "rate": rate, "channels": channels}
        for _ in range(settings["repeats"]):
            plan.append(("EEG only", [sensor]))
    for count in range(1, len(settings["sensors"]) + 1):
        for _ in range(settings["repeats"]):
            plan.append(("Additional streams", settings["sensors"][:count]))
    return plan


def simulate(sensors, duration, receiver):
    """The sender is a separate process; this process receives and saves data."""
    context = mp.get_context("spawn")
    ready, finished = context.Event(), context.Event()
    identity = uuid.uuid4().hex[:10]
    specs = [{**sensor, "stream_name": f"{identity}_{sensor['name']}"} for sensor in sensors]
    specs.append({"name": "markers", "stream_name": f"{identity}_markers"})
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "sent.npz"
        sender = context.Process(
            target=transmission.send, args=(sensors, duration, identity, ready, finished, source)
        )
        sender.start()
        try:
            case = reception.record(specs, duration, receiver, ready)
        finally:
            finished.set()
            sender.join(timeout=10)
            if sender.is_alive():
                sender.terminate()
                sender.join()
        if source.exists():
            with np.load(source, allow_pickle=False) as saved:
                case["source_start"] = float(saved["start"])
                case["events"] = json.loads(str(saved["events"]))
                case["start"] = case["source_start"] + case["streams"]["eeg"]["initial_offset"]
                for key in saved.files:
                    if "__" in key:
                        name, field = key.split("__")
                        case["streams"][name]["sent_" + field] = saved[key]
        if sender.exitcode != 0:
            case["status"], case["error"] = "failed", f"Sender exit code: {sender.exitcode}"
        return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--out", help="Fresh output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real-eeg", action="store_true", help="Receive the actual amplifier")
    mode.add_argument(
        "--compare-eeg", metavar="RECORDING", help="Compare a recordings.npz with XDF"
    )
    mode.add_argument("--list-streams", action="store_true", help="Show available LSL outlets")
    parser.add_argument("--reference", help="Simultaneous LabRecorder XDF file")
    args = parser.parse_args()
    if bool(args.compare_eeg) != bool(args.reference):
        parser.error("Use --compare-eeg and --reference together")
    if args.list_streams:
        reception.list_streams()
        return
    if args.compare_eeg:
        cases = analysis.load_recordings(args.compare_eeg)
        if len(cases) != 1 or cases[0]["simulated"]:
            parser.error("Choose one recording made with --real-eeg")
        analysis.compare_xdf(cases[0], args.reference)
        analysis.write_report(Path(args.compare_eeg).parent, cases)
        return
    settings = json.loads(Path(args.config).read_text())
    output = (
        Path(args.out)
        if args.out
        else Path(__file__).with_name("results-real" if args.real_eeg else "results")
    )
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be empty; use --out with a new directory")
    receiver = settings["receiver"]
    if not np.isfinite(receiver["read_hz"]) or receiver["read_hz"] <= 0 or receiver["buffer_s"] < 1:
        parser.error("Reading rate and buffer size must be positive")
    if args.real_eeg and (
        not np.isfinite(settings["real_eeg"]["duration_s"])
        or settings["real_eeg"]["duration_s"] <= 0
    ):
        parser.error("Real EEG duration must be positive and finite")
    cases = []
    try:
        if args.real_eeg:
            case = reception.record_real(settings["real_eeg"], receiver)
            case.update(id="real", group="Real EEG")
            cases.append(case)
        else:
            simulation = settings["simulation"]
            if (
                not np.isfinite(simulation["duration_s"])
                or simulation["duration_s"] <= 0
                or not isinstance(simulation["repeats"], int)
                or simulation["repeats"] < 1
            ):
                parser.error("Duration and repetitions must be positive")
            plan = simulation_plan(simulation)
            for _, sensors in plan:
                for sensor in sensors:
                    if (
                        not np.isfinite(sensor["rate"])
                        or sensor["rate"] <= 0
                        or not isinstance(sensor["channels"], int)
                        or sensor["channels"] < 1
                    ):
                        parser.error(
                            "Sensor rates must be positive; channel counts must be positive integers"
                        )
            for number, (group, sensors) in enumerate(plan, 1):
                print(f"[{number}/{len(plan)}] {group}", flush=True)
                case = simulate(sensors, simulation["duration_s"], receiver)
                case.update(id=f"case_{number:03}", group=group)
                cases.append(case)
                if case["status"] != "completed":
                    break
        if cases and cases[-1]["status"] in ("failed", "incomplete"):
            raise RuntimeError(cases[-1]["error"] or "Recording did not finish")
    finally:
        if cases:
            analysis.save_recordings(output, cases)
            analysis.write_report(output, cases)


if __name__ == "__main__":
    main()
