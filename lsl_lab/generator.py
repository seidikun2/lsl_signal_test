"""Generate signals and markers, simulate acquisition and transmit over LSL."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock, cf_double64, cf_string
from .config import load, fresh, save_json, sample_elapsed, wall_duration, sample_count, marker_events


def signal(kind, indices, rate, channels, events=(), elapsed=None):
    t = np.asarray(indices, dtype=float)[:, None] / rate
    c = np.arange(channels)[None, :]
    # Frequencies remain below Nyquist, even in the 100-Hz sweep.
    if kind.upper() == 'EEG':
        y = 20*np.sin(2*np.pi*(8 + c % 5)*t + .17*c)
        y += 8*np.sin(2*np.pi*min(22., .35*rate)*t + .11*c)
        amplitude = 60.
    elif kind.upper() == 'EMG':
        carrier = min(85., .37*rate)
        y = 70*np.sin(2*np.pi*(carrier-c % 5)*t + .23*c)
        y *= .2 + .8*np.sin(2*np.pi*.7*t)**2
        amplitude = 140.
    elif kind.upper() == 'IMU':
        y = np.sin(2*np.pi*(.7 + .1*(c % 3))*t + .4*c)
        amplitude = 3.
    else:
        raise ValueError(f'Unknown sensor: {kind}')
    clock = np.asarray(elapsed)[:, None] if elapsed is not None else t
    for event in events:
        onset = event['nominal']+event['delay']
        pulse = (clock >= onset) & (clock < onset+event['width'])
        y += amplitude*event['amplitude']*pulse
    return y.astype(np.float32)


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
    events = [dict(kind='start', nominal=0., id=-1)]+marker_events(cfg)
    events += [dict(kind='end', nominal=max(cfg['duration'], wall_duration(cfg)), id=-2)]
    stimuli = [e for e in events if e['kind'] == 'stimulus']
    event_index = 0
    marker_log = []
    save_json(out / 'tx_meta.json', dict(run_id=run_id, start=start, complete=False))
    while True:
        due = []
        for s in cfg['sensors']:
            st = states[s['name']]
            count = sample_count(cfg, s)
            if st['next'] < count:
                end = min(st['next']+cfg['chunk'], count)
                due.append((float(start+sample_elapsed((end-1)/s['rate'], s)), s, end))
        if event_index < len(events):
            due.append((start + events[event_index]['nominal'], None, None))
        if not due:
            break
        when, s, end = min(due, key=lambda x: x[0])
        remaining = when-local_clock()
        if remaining > 0:
            time.sleep(remaining)
        if s is None:
            sent = local_clock()
            payload = dict(events[event_index], sample_time=when, sent=sent)
            markers.push_sample([json.dumps(payload)], timestamp=when, pushthrough=True)
            marker_log.append(payload)
            event_index += 1
            continue
        st = states[s['name']]
        ids = np.arange(st['next'], end, dtype=np.int64)
        generated = local_clock()
        elapsed = sample_elapsed(ids/s['rate'], s)
        values = signal(s['kind'], ids, s['rate'], s['channels'], stimuli, elapsed)
        ts = start + elapsed
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
    parser = argparse.ArgumentParser(description='Generate and transmit LSL signals')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1]/'settings.py'))
    parser.add_argument('--id', default='test', help='Same ID as the receiver')
    parser.add_argument('--out', default='results/manual/tx')
    args = parser.parse_args(argv)
    transmit(load(args.config), args.id, fresh(args.out)/'data')


if __name__ == '__main__':
    main()
