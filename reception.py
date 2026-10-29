"""Receive LSL streams and record values and timestamps."""
import json
import time
from pathlib import Path

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop
from helpers import save_json, wall_duration


def receive(cfg, run_id, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'config.json', cfg)
    inlets, states, corrections = {}, {}, []
    specs = [(s['name'], s['channels']) for s in cfg['sensors']] + [('markers', 0)]
    for name, channels in specs:
        found = resolve_byprop('source_id', f'{run_id}:{name}', minimum=1, timeout=30)
        if not found:
            raise TimeoutError(f'LSL stream not found: {run_id}:{name}')
        if found[0].channel_count() != (channels+1 if name != 'markers' else 1):
            raise ValueError(f'Channel count mismatch for {name}; use the same config on TX and RX')
        inlet = StreamInlet(found[0], max_buflen=cfg['inlet_buffer'], max_chunklen=0,
                            recover=False, processing_flags=0)
        inlet.open_stream(timeout=10)
        correction = inlet.time_correction(timeout=10)
        inlets[name] = inlet
        states[name] = dict(ids=[], values=[], raw=[], corrected=[], arrival=[], correction=[], current=correction)
        corrections.append(dict(stream=name, measured=local_clock(), offset=correction))
    marker_log = []
    begin = local_clock()
    marker_duration = cfg['duration']
    deadline = begin + cfg['start_delay'] + max(wall_duration(cfg), marker_duration) + 30
    refresh = begin + cfg['correction_interval']
    pause_at = begin + cfg['start_delay'] + cfg['pause_every']
    end_seen = None
    complete = False
    try:
        while local_clock() < deadline:
            now = local_clock()
            if now >= pause_at and cfg['receiver_pause']:
                time.sleep(cfg['receiver_pause'])
                pause_at = local_clock() + cfg['pause_every']
            if now >= refresh:
                for name, inlet in inlets.items():
                    try:
                        offset = inlet.time_correction(timeout=.001)
                        states[name]['current'] = offset
                        corrections.append(dict(stream=name, measured=local_clock(), offset=offset))
                    except TimeoutError:
                        pass  # retain the last valid real-time estimate
                refresh = local_clock() + cfg['correction_interval']
            for name, inlet in inlets.items():
                samples, ts = inlet.pull_chunk(timeout=0., max_samples=4096)
                arrival = local_clock()  # read completion, not a hardware/socket timestamp
                if not ts:
                    continue
                st = states[name]
                c = st['current']
                if name == 'markers':
                    for sample, raw in zip(samples, ts):
                        event = json.loads(sample[0])
                        marker_log.append(dict(**event, raw=raw, corrected=raw+c,
                                               arrival=arrival, correction=c))
                        if event['kind'] == 'end':
                            end_seen = arrival
                    continue
                data = np.asarray(samples)
                st['ids'].append(data[:, -1].astype(np.int64))
                st['values'].append(data[:, :-1].astype(np.float32))
                st['raw'].append(np.asarray(ts))
                st['corrected'].append(np.asarray(ts)+c)
                st['arrival'].append(np.full(len(ts), arrival))
                st['correction'].append(np.full(len(ts), c))
            if end_seen is not None and local_clock()-end_seen > 1.:
                complete = True
                break
            time.sleep(.001)
    finally:
        for s in cfg['sensors']:
            st = states[s['name']]
            arrays = {}
            for k in ('ids', 'values', 'raw', 'corrected', 'arrival', 'correction'):
                arrays[k] = np.concatenate(st[k]) if st[k] else np.empty(
                    (0, s['channels']) if k == 'values' else (0,))
            np.savez_compressed(out / f'rx_{s["name"]}.npz', **arrays)
        save_json(out / 'rx_markers.json', marker_log)
        save_json(out / 'corrections.json', corrections)
        save_json(out / 'rx_meta.json', dict(run_id=run_id, complete=complete))
        for inlet in inlets.values():
            inlet.close_stream()
    if not complete:
        raise TimeoutError('End marker not received; partial receiver logs were saved')
