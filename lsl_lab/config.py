"""Experiment validation and shared timing helpers."""
import json
import math
import runpy
from pathlib import Path

import numpy as np


def validate(config):
    cfg = json.loads(json.dumps(config))
    defaults = dict(duration=10., chunk=16, inlet_buffer=60, correction_interval=2.,
                    receiver_pause=0., pause_every=2., drift_ppm=0., start_delay=2., event_tolerance_ms=10.)
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    if not cfg.get('sensors'):
        raise ValueError('At least one sensor is required')
    names = []
    for s in cfg['sensors']:
        if s['kind'].upper() not in ('EEG', 'EMG', 'IMU'):
            raise ValueError('kind must be EEG, EMG or IMU')
        name = s['name']
        if not name or name == 'markers' or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Sensor names must be short alphanumeric identifiers')
        names.append(name)
        for k in ('rate', 'channels'):
            if not math.isfinite(s[k]) or s[k] <= 0:
                raise ValueError(f'{k} must be positive and finite')
        if int(s['channels']) != s['channels']:
            raise ValueError('channels must be an integer')
        s['channels'] = int(s['channels'])
        if 'samples' in s and (not isinstance(s['samples'], int) or s['samples'] < 1):
            raise ValueError('samples must be a positive integer')
        s.setdefault('drift_ppm', cfg['drift_ppm'])
        if not math.isfinite(s['drift_ppm']) or abs(s['drift_ppm']) >= 100000:
            raise ValueError('Sensor abs(drift_ppm) must be < 100000')
        ramp = s.setdefault('drift_slope_ppm_s', 0.)
        if not math.isfinite(ramp):
            raise ValueError('drift_slope_ppm_s must be finite')
        # The phase inversion must stay on its positive-rate branch.
        if (1+s['drift_ppm']*1e-6)**2+2*ramp*1e-6*(sample_count(cfg, s)/s['rate']+1/s['rate']) <= 0:
            raise ValueError('Acquisition rate would reach zero during the experiment')
    if len(set(names)) != len(names):
        raise ValueError('Sensor names must be unique')
    cfg.setdefault('online_rate', max(s['rate'] for s in cfg['sensors']))
    cfg.setdefault('online_lag', .25)
    if cfg['online_rate'] < max(s['rate'] for s in cfg['sensors']):
        raise ValueError('online_rate must be >= the highest sensor rate (no anti-alias downsampling is applied)')
    for k in ('duration', 'inlet_buffer', 'correction_interval', 'pause_every', 'start_delay', 'online_rate', 'online_lag'):
        if not math.isfinite(cfg[k]) or cfg[k] <= 0:
            raise ValueError(f'{k} must be positive and finite')
    if int(cfg['inlet_buffer']) != cfg['inlet_buffer']:
        raise ValueError('inlet_buffer must be an integer number of seconds')
    if int(cfg['chunk']) != cfg['chunk'] or cfg['chunk'] < 1:
        raise ValueError('chunk must be a positive integer')
    if not math.isfinite(cfg['drift_ppm']) or abs(cfg['drift_ppm']) >= 100000:
        raise ValueError('abs(drift_ppm) must be < 100000')
    if not math.isfinite(cfg['receiver_pause']) or cfg['receiver_pause'] < 0:
        raise ValueError('receiver_pause must be finite and nonnegative')
    if not math.isfinite(cfg['event_tolerance_ms']) or cfg['event_tolerance_ms'] <= 0:
        raise ValueError('event_tolerance_ms must be positive')
    cfg.setdefault('markers', [dict(label='A', interval=2., first=.3, delay=.2, width=.08, amplitude=1.),
                               dict(label='B', interval=2., first=.9, delay=.2, width=.08, amplitude=-1.),
                               dict(label='C', interval=2., first=1.5, delay=.2, width=.08, amplitude=.6)])
    labels = []
    for marker in cfg['markers']:
        labels.append(marker['label'])
        if not marker['label'] or not marker['label'].isascii() or not marker['label'].isalnum():
            raise ValueError('Marker labels must be ASCII alphanumeric strings')
        for key in ('interval', 'width'):
            if not math.isfinite(marker[key]) or marker[key] <= 0:
                raise ValueError(f'Marker {key} must be positive')
        for key in ('first', 'delay'):
            if not math.isfinite(marker[key]) or marker[key] < 0:
                raise ValueError(f'Marker {key} must be nonnegative')
        if not math.isfinite(marker['amplitude']) or marker['amplitude'] == 0:
            raise ValueError('Marker amplitude must be finite and nonzero')
    if len(labels) != len(set(labels)):
        raise ValueError('Marker labels must be unique')
    return cfg


def load(path):
    path = Path(path)
    if path.suffix == '.py':
        settings = runpy.run_path(str(path))
        content = json.loads(json.dumps(settings['CONFIG']))
        tests = settings.get('TESTS', {})
        content['duration'] = tests.get('duration_s', content.get('duration', 10))
        if tests.get('length_mode') == 'samples':
            for sensor in content['sensors']:
                sensor['samples'] = tests['samples_per_stream']
            content['duration'] = max(s['samples']/s['rate'] for s in content['sensors'])
    else:
        content = json.loads(path.read_text())
    return validate(content)


def sample_elapsed(nominal, sensor):
    """Invert acquisition phase for a linearly changing frequency offset.

    nominal = scale * elapsed + 0.5 * slope * elapsed**2.
    Rationalized quadratic root also works for zero slope.
    """
    scale = 1+sensor['drift_ppm']*1e-6
    ramp = sensor.get('drift_slope_ppm_s', 0.)*1e-6
    discriminant = scale**2+2*ramp*np.asarray(nominal)
    if np.any(discriminant <= 0):
        raise ValueError('Acquisition rate would reach zero during the experiment')
    return 2*np.asarray(nominal)/(scale+np.sqrt(discriminant))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def wall_duration(cfg):
    return float(max(sample_elapsed(sample_count(cfg, s)/s['rate'], s) for s in cfg['sensors']))


def sample_count(cfg, sensor):
    return sensor.get('samples', int(np.ceil(cfg['duration']*sensor['rate'])))


def marker_events(cfg):
    """Periodic classes share the transmitter clock; every event has a unique ID."""
    events = []
    for marker in cfg['markers']:
        for sequence, nominal in enumerate(np.arange(marker['first'], cfg['duration'], marker['interval'])):
            events.append(dict(kind='stimulus', nominal=float(nominal), label=marker['label'],
                               sequence=sequence, delay=marker['delay'], width=marker['width'],
                               amplitude=marker['amplitude'], interval=marker['interval']))
    events.sort(key=lambda e: (e['nominal'], e['label']))
    for identity, event in enumerate(events):
        event['id'] = identity
    return events


def fresh(path):
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    return out
