"""Gera sinais e marcadores, simula a aquisição e transmite por LSL."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock, cf_double64, cf_string
from config import load, fresh, save_json, sample_elapsed, wall_duration


def pilot(t):
    """Common coded 40-ms pulse every second, beginning at t=0.5 s."""
    event = np.floor(t - 0.5).astype(np.int64)
    phase = t - 0.5 - event
    polarity = np.where(((event * 1103515245 + 12345) >> 4) & 1, 1., -1.)
    return np.where((t >= .5) & (phase < .04), polarity, 0.)


def signal(kind, indices, rate, channels):
    t = np.asarray(indices, dtype=float)[:, None] / rate
    c = np.arange(channels)[None, :]
    # Frequencies remain below Nyquist, even in the 100-Hz sweep.
    if kind.upper() == 'EEG':
        y = 20*np.sin(2*np.pi*(8 + c % 5)*t + .17*c)
        y += 8*np.sin(2*np.pi*min(22., .35*rate)*t + .11*c)
        amplitude = 35.
    elif kind.upper() == 'EMG':
        carrier = min(85., .37*rate)
        y = 70*np.sin(2*np.pi*(carrier-c % 5)*t + .23*c)
        y *= .2 + .8*np.sin(2*np.pi*.7*t)**2
        amplitude = 100.
    elif kind.upper() == 'IMU':
        y = np.sin(2*np.pi*(.7 + .1*(c % 3))*t + .4*c)
        amplitude = 2.
    else:
        raise ValueError(f'Unknown sensor: {kind}')
    return (y + amplitude*pilot(t)).astype(np.float32)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Gerar e transmitir sinais LSL')
    parser.add_argument('--config', default='configs/single.json')
    parser.add_argument('--id', default='teste', help='Mesmo ID no receptor')
    parser.add_argument('--out', default='runs/tx01')
    args = parser.parse_args(argv)
    transmit(load(args.config), args.id, fresh(args.out))


if __name__ == '__main__':
    main()
