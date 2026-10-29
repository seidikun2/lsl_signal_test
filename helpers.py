"""Experiment validation and shared timing helpers."""
import json
import math
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
    for k in ('duration', 'inlet_buffer', 'correction_interval', 'pause_every', 'start_delay'):
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
    if not math.isfinite(cfg.get('snapshot_seconds',2)) or cfg.get('snapshot_seconds',2) <= 0:
        raise ValueError('snapshot_seconds must be positive')
    channels = cfg.get('plot_channels',[0,1])
    if not isinstance(channels,list) or not 1<=len(channels)<=2 or any(not isinstance(c,int) or c<0 for c in channels):
        raise ValueError('plot_channels must contain one or two nonnegative channel indices')
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


def read_settings(path):
    """Read the single user-facing JSON configuration."""
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    return {'TESTS': document['tests'], 'CONFIG': document['acquisition']}


def prepare_results(path):
    """Archive obsolete folders and previous outputs outside the checkout."""
    import shutil
    from datetime import datetime
    project=Path(__file__).resolve().parent
    path=Path(path).resolve()
    default=project/'results'
    previous=path.exists() and any(p.is_file() for p in path.rglob('*'))
    if previous and path!=default:
        raise ValueError(f'Output directory is not empty: {path}')
    names=['codigo','lsl_lab','lsltest','resultados','runs','reuns','tests','__pycache__','.pytest_cache']
    legacy=[project/name for name in names if (project/name).is_dir()]
    legacy+=list(project.glob('*.egg-info'))
    if previous or legacy:
        archive=project.parent/(project.name+'-history')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        archive.mkdir(parents=True)
        for folder in legacy:
            shutil.move(folder,archive/folder.name)
        if previous:
            shutil.move(path,archive/'results')
        print(f'Previous results and obsolete folders preserved at {archive}',flush=True)
    path.mkdir(parents=True,exist_ok=True)
    return path


def pack_recording(source, destination):
    """Save numeric traces and JSON metadata together, without pickle."""
    arrays = {}
    metadata = {}
    for path in Path(source).glob('*.npz'):
        if not path.stem.startswith(('tx_', 'rx_')):
            continue
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                arrays[path.stem+'__'+key] = data[key]
    for path in Path(source).glob('*.json'):
        metadata[path.stem] = json.loads(path.read_text())
    arrays['metadata_json'] = np.array(json.dumps(metadata, allow_nan=False))
    np.savez_compressed(destination, **arrays)


def unpack_recording(source, destination):
    """Restore a saved recording to a temporary analysis directory."""
    import re
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
        groups = {}
        for key in data.files:
            if key == 'metadata_json':
                continue
            stem, field = key.rsplit('__', 1)
            if not re.fullmatch(r'[A-Za-z0-9_-]+',stem) or not re.fullmatch(r'[A-Za-z0-9_]+',field):
                raise ValueError('Invalid recording key')
            groups.setdefault(stem, {})[field] = data[key]
        for stem, values in groups.items():
            np.savez(destination/(stem+'.npz'), **values)
        for name, value in metadata.items():
            if not name.replace('_','').isalnum():
                raise ValueError('Invalid metadata key')
            save_json(destination/(name+'.json'), value)
