"""Receive LSL streams and record values and timestamps."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop
from .config import load, fresh, save_json, wall_duration


class OnlineAligner:
    def __init__(self, sensors, rate, lag=.25):
        self.rate, self.lag, self.next_time = rate, lag, None
        self.rates = {s['name']: s['rate'] for s in sensors}
        self.data = {s['name']: (np.empty(0), np.empty((0, s['channels']))) for s in sensors}

    def add(self, name, timestamps, values):
        old_t, old_y = self.data[name]
        if not len(timestamps):
            return
        # A backward correction supersedes buffered data at the new timestamps.
        cut = np.searchsorted(old_t, timestamps[0], side='left')
        old_t, old_y = old_t[:cut], old_y[:cut]
        t = np.concatenate((old_t, timestamps))
        y = np.concatenate((old_y, values))
        # A clock correction step can produce nonmonotonic timestamps. Keep the
        # newest strictly increasing sequence; do not interpolate across a reversal.
        keep = np.r_[True, t[1:] > np.maximum.accumulate(t[:-1])]
        self.data[name] = t[keep], y[keep]

    def pull(self, now):
        if any(len(t) < 2 for t, _ in self.data.values()):
            return None
        if self.next_time is None:
            self.next_time = np.ceil(max(t[0] for t, _ in self.data.values())*self.rate)/self.rate
        end = min(now-self.lag, *(t[-1] for t, _ in self.data.values()))
        n = int(np.floor((end-self.next_time)*self.rate))+1
        if n <= 0:
            return None
        grid = self.next_time + np.arange(n)/self.rate
        result = {}
        for name, (t, y) in self.data.items():
            values = np.column_stack([np.interp(grid, t, y[:, c], left=np.nan, right=np.nan)
                                      for c in range(y.shape[1])])
            right = np.searchsorted(t, grid, side='right').clip(1, len(t)-1)
            gap = (t[right]-t[right-1]) > 1.5/self.rates[name]
            exact = np.isclose(grid, t[right-1], rtol=0, atol=1e-9) | np.isclose(grid, t[right], rtol=0, atol=1e-9)
            gap &= ~exact
            values[gap] = np.nan
            result[name] = values
            cut = max(0, np.searchsorted(t, grid[-1])-1)
            self.data[name] = t[cut:], y[cut:]
        self.next_time += n/self.rate
        return grid, result


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
    aligner = OnlineAligner(cfg['sensors'], cfg['online_rate'], cfg['online_lag']) if cfg.get('online_align', False) else None
    online_times, online_values = [], {s['name']: [] for s in cfg['sensors']}
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
                if aligner is not None:
                    aligner.add(name, np.asarray(ts)+c, data[:, :-1])
            aligned = aligner.pull(local_clock()) if aligner is not None else None
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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Receive and record LSL signals')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1]/'settings.py'))
    parser.add_argument('--id', default='test', help='Same ID as the generator')
    parser.add_argument('--out', default='results/manual/rx')
    args = parser.parse_args(argv)
    receive(load(args.config), args.id, fresh(args.out)/'data')


if __name__ == '__main__':
    main()
