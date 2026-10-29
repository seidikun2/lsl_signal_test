"""Real LSL outlets/inlets, explicit timestamps and unmodified receiver clocks."""
import json
import time
from pathlib import Path

import numpy as np
from pylsl import (StreamInfo, StreamInlet, StreamOutlet, local_clock,
                   resolve_byprop, cf_double64, cf_string)

from .signals import signal, sample_elapsed
from .online import OnlineAligner


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def wall_duration(cfg):
    return float(max(sample_elapsed(cfg['duration'], s) for s in cfg['sensors']))


def transmit(cfg, run_id, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'config.json', cfg)
    outlets, states = {}, {}
    for s in cfg['sensors']:
        info = StreamInfo(f'{run_id}_{s["name"]}', s['kind'], s['channels']+1,
                          s['rate'], cf_double64, f'{run_id}:{s["name"]}')
        info.desc().append_child_value('validation_index_channel', str(s['channels']))
        info.desc().append_child_value('signal_channels', str(s['channels']))
        info.desc().append_child_value('synthetic', 'true')
        outlets[s['name']] = StreamOutlet(info, chunk_size=0, max_buffered=cfg['inlet_buffer'])
        states[s['name']] = dict(next=0, ids=[], values=[], sample=[], generated=[], sent=[])
    markers = StreamOutlet(StreamInfo(f'{run_id}_markers', 'Markers', 1, 0,
                                      cf_string, f'{run_id}:markers'))
    # No samples are sent until every inlet has connected.
    deadline = time.monotonic() + 45
    while not (markers.have_consumers() and all(o.have_consumers() for o in outlets.values())):
        if time.monotonic() > deadline:
            raise TimeoutError('Not all receivers connected within 45 s')
        time.sleep(.02)
    start = local_clock() + cfg['start_delay']
    scale = 1 + cfg['drift_ppm'] * 1e-6
    events = [('start', 0., -1)] + [('sync', float(t), i) for i, t in
              enumerate(np.arange(.5, cfg['duration'], 1.))] + [('end', max(cfg['duration'], wall_duration(cfg)*scale), -2)]
    event_index = 0
    marker_log = []
    save_json(out / 'tx_meta.json', dict(run_id=run_id, start=start, complete=False))
    while True:
        due = []
        for s in cfg['sensors']:
            st = states[s['name']]
            count = int(np.ceil(cfg['duration'] * s['rate']))
            if st['next'] < count:
                end = min(st['next']+cfg['chunk'], count)
                due.append((float(start+sample_elapsed((end-1)/s['rate'], s)), s, end))
        if event_index < len(events):
            due.append((start + events[event_index][1]/scale, None, None))
        if not due:
            break
        when, s, end = min(due, key=lambda x: x[0])
        remaining = when-local_clock()
        if remaining > 0:
            time.sleep(remaining)
        if s is None:
            kind, nominal, event_id = events[event_index]
            sent = local_clock()
            payload = dict(kind=kind, id=event_id, nominal=nominal, sample_time=when, sent=sent)
            markers.push_sample([json.dumps(payload)], timestamp=when, pushthrough=True)
            marker_log.append(payload)
            event_index += 1
            continue
        st = states[s['name']]
        ids = np.arange(st['next'], end, dtype=np.int64)
        generated = local_clock()
        values = signal(s['kind'], ids, s['rate'], s['channels'])
        ts = start + sample_elapsed(ids/s['rate'], s)
        packet = np.column_stack((values, ids))
        sent = local_clock()
        outlets[s['name']].push_chunk(packet.tolist(), ts.tolist(), pushthrough=True)
        for key, value in [('ids', ids), ('values', values), ('sample', ts),
                           ('generated', np.full(len(ids), generated)), ('sent', np.full(len(ids), sent))]:
            st[key].append(value)
        st['next'] = end
    for name, st in states.items():
        np.savez_compressed(out / f'tx_{name}.npz', **{
            key: np.concatenate(st[key]) for key in ('ids', 'values', 'sample', 'generated', 'sent')})
    save_json(out / 'tx_markers.json', marker_log)
    save_json(out / 'tx_meta.json', dict(run_id=run_id, start=start, complete=True))
    # Keep outlets alive while receivers drain, including under deliberate pauses.
    time.sleep(max(2., cfg['receiver_pause'] + 1.))


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
    aligner = OnlineAligner(cfg['sensors'], cfg['online_rate'], cfg['online_lag'])
    online_times, online_values = [], {s['name']: [] for s in cfg['sensors']}
    marker_duration = cfg['duration']/(1+cfg['drift_ppm']*1e-6)
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
                aligner.add(name, np.asarray(ts)+c, data[:, :-1])
            aligned = aligner.pull(local_clock())
            if aligned is not None:
                grid, values = aligned
                online_times.append(grid)
                for name, value in values.items():
                    online_values[name].append(value)
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
        if online_times:
            np.savez_compressed(out / 'online.npz', time=np.concatenate(online_times), **{
                name: np.concatenate(values) for name, values in online_values.items()})
        save_json(out / 'rx_meta.json', dict(run_id=run_id, complete=complete))
        for inlet in inlets.values():
            inlet.close_stream()
    if not complete:
        raise TimeoutError('End marker not received; partial receiver logs were saved')
