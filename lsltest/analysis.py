import csv
import json
from pathlib import Path

import numpy as np


def distribution(x):
    x = np.asarray(x)
    if not len(x):
        return dict(mean=None, median=None, p95=None, p99=None, min=None, max=None, std=None)
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)),
                min=float(np.min(x)), max=float(np.max(x)), std=float(np.std(x)))


def slope(x, y):
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    return float(np.polyfit(x-x[0], y-y[0], 1)[0])


def compare(tx, rx, rate):
    ids = rx['ids'].astype(np.int64)
    expected = tx['ids'].astype(np.int64)
    unique, counts = np.unique(ids, return_counts=True)
    common, it, ir = np.intersect1d(expected, ids, return_indices=True)
    metrics = dict(expected=len(expected), received=len(ids), matched=len(common),
                   missing=int(len(np.setdiff1d(expected, unique))),
                   unexpected=int(len(np.setdiff1d(unique, expected))),
                   duplicates=int(np.sum(counts-1)), reordered=int(np.sum(np.diff(ids) < 0)))
    if not len(common):
        metrics['error'] = 'No matching samples'
        return metrics, None
    correction = rx['correction'][ir]
    sample = tx['sample'][it] + correction
    arrival = rx['arrival'][ir]
    elapsed = common/rate
    # Source timestamps are mapped to the receiver domain using the estimate
    # available when each sample was read, never using future observations.
    latency = arrival-sample
    queue = arrival-(tx['sent'][it]+correction)
    timestamp_error = rx['corrected'][ir]-sample
    timing_tx = tx['sample'][it]-tx['sample'][it[0]]-(elapsed-elapsed[0])
    timing_rx = rx['corrected'][ir]-rx['corrected'][ir[0]]-(elapsed-elapsed[0])
    diff = rx['values'][ir].astype(float)-tx['values'][it]
    metrics.update(
        latency_ms=distribution(latency*1000),
        send_to_read_ms=distribution(queue*1000),
        batching_ms=distribution((tx['sent'][it]-tx['sample'][it])*1000),
        generation_lateness_ms=distribution((tx['generated'][it]-tx['sample'][it])*1000),
        timestamp_error_ms=distribution(timestamp_error*1000),
        rmse=float(np.sqrt(np.mean(diff**2))), max_abs_error=float(np.max(np.abs(diff))),
        rmse_per_channel=np.sqrt(np.mean(diff**2, axis=0)).tolist(),
        tx_timing_drift_ppm=slope(elapsed, timing_tx)*1e6 if len(common)>1 else None,
        rx_timing_drift_ppm=slope(elapsed, timing_rx)*1e6 if len(common)>1 else None,
        latency_trend_ppm=slope(elapsed, latency)*1e6 if len(common)>1 else None,
        tx_accumulated_drift_ms=float(timing_tx[-1]*1000),
        rx_accumulated_drift_ms=float(timing_rx[-1]*1000),
        latency_change_ms=float((latency[-1]-latency[0])*1000))
    return metrics, dict(common=common, it=it, ir=ir, elapsed=elapsed,
                         latency=latency, queue=queue, timing_tx=timing_tx,
                         timing_rx=timing_rx, timestamp_error=timestamp_error)


def analyze(directory, tx_directory=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(directory)
    source = Path(tx_directory) if tx_directory else out
    cfg = json.loads((source/'config.json').read_text())
    txmeta = json.loads((source/'tx_meta.json').read_text())
    rxmeta = json.loads((out/'rx_meta.json').read_text())
    if txmeta['run_id'] != rxmeta['run_id']:
        raise ValueError('TX and RX run IDs differ')
    report = dict(run_id=txmeta['run_id'], tx_complete=txmeta['complete'],
                  rx_complete=rxmeta['complete'], sensors={})
    rows = []
    rxmarkers = json.loads((out/'rx_markers.json').read_text())
    txmarkers = json.loads((source/'tx_markers.json').read_text())
    expected_markers = {(e['kind'], e['id']) for e in txmarkers}
    got_markers = [(e['kind'], e['id']) for e in rxmarkers]
    report['markers'] = dict(expected=len(txmarkers), received=len(rxmarkers),
        missing=len(expected_markers-set(got_markers)), duplicates=len(got_markers)-len(set(got_markers)),
        latency_ms=distribution([(e['arrival']-e['corrected'])*1000 for e in rxmarkers]))
    marker_alignment = {}
    for s in cfg['sensors']:
        name = s['name']
        with np.load(source/f'tx_{name}.npz') as f:
            tx = dict(f)
        with np.load(out/f'rx_{name}.npz') as f:
            rx = dict(f)
        metrics, data = compare(tx, rx, s['rate'])
        report['sensors'][name] = metrics
        rows.append(dict(sensor=name, rate=s['rate'], channels=s['channels'], chunk=cfg['chunk'],
                         expected=metrics['expected'], received=metrics['received'], missing=metrics['missing'],
                         p99_ms=metrics.get('latency_ms', {}).get('p99'), rmse=metrics.get('rmse'),
                         drift_ppm=metrics.get('rx_timing_drift_ppm')))
        if data is None:
            continue
        ir, it = data['ir'], data['it']
        fig, axes = plt.subplots(5, 1, figsize=(12, 15), constrained_layout=True)
        origin = txmeta['start']
        # Map receiver timestamps back for display using the same online estimate.
        corrected_source = rx['corrected'][ir]-rx['correction'][ir]
        read_source = rx['arrival'][ir]-rx['correction'][ir]
        view = data['elapsed'] <= min(3., cfg['duration'])
        txview = tx['ids']/s['rate'] <= min(3., cfg['duration'])
        axes[0].plot(tx['sample'][txview]-origin, tx['values'][txview, 0], label='TX: sample clock')
        received_values = rx['values'][ir, 0].astype(float).copy()
        received_values[np.r_[False, np.diff(data['common']) > 1]] = np.nan
        axes[0].plot(corrected_source[view]-origin, received_values[view], '--', label='RX: LSL corrected')
        axes[0].plot(read_source[view]-origin, received_values[view], alpha=.5, label='RX: read completion')
        axes[0].set_ylabel('Channel 0 (synthetic units)')
        axes[1].plot(data['elapsed'], data['latency']*1000, label='Sample to read')
        axes[1].plot(data['elapsed'], data['queue']*1000, label='Send to read', alpha=.7)
        axes[1].set_ylabel('Delay (ms)')
        axes[2].plot(data['elapsed'], data['timing_tx']*1000, label='TX vs nominal rate')
        axes[2].plot(data['elapsed'], data['timing_rx']*1000, '--', label='RX corrected vs nominal rate')
        axes[2].set_ylabel('Accumulated timing error (ms)')
        axes[3].hist(data['latency']*1000, bins=min(80, max(5, len(ir)//10)), label='Delay distribution')
        axes[3].set_xlabel('Delay (ms)')
        axes[3].set_ylabel('Samples')
        axes[4].plot(data['elapsed'], (rx['correction'][ir]-rx['correction'][ir[0]])*1000,
                     label='LSL offset change')
        axes[4].set_ylabel('Clock correction change (ms)')
        for i, ax in enumerate(axes):
            if i != 3:
                ax.set_xlabel('Time (s)')
            ax.legend()
            ax.grid(alpha=.25)
        fig.suptitle(f'{name}: {s["rate"]} Hz / {s["channels"]} ch / chunk {cfg["chunk"]}')
        fig.savefig(out/f'{name}.png', dpi=130)
        plt.close(fig)
        aligned_events = []
        for event in rxmarkers:
            if event['kind'] != 'sync':
                continue
            target_id = int(np.ceil(event['nominal']*s['rate']))
            match = np.flatnonzero(rx['ids'] == target_id)
            if not len(match):
                continue
            j = match[0]
            # Includes sample-grid quantization; report it explicitly.
            aligned_events.append(dict(id=event['id'], sample_id=target_id,
                nominal_quantization_ms=(target_id/s['rate']-event['nominal'])*1000,
                sample_minus_marker_ms=float((rx['corrected'][j]-event['corrected'])*1000),
                read_minus_marker_read_ms=float((rx['arrival'][j]-event['arrival'])*1000)))
        marker_alignment[name] = aligned_events
    report['marker_alignment'] = marker_alignment
    if any(marker_alignment.values()):
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for name, events in marker_alignment.items():
            ax.plot([e['id'] for e in events], [e['sample_minus_marker_ms'] for e in events],
                    '.-', label=name)
        ax.set(xlabel='Sync marker ID', ylabel='Sample minus marker (ms)',
               title='Online LSL-corrected alignment across sensors')
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(out/'sync.png', dpi=130)
        plt.close(fig)
    report['notes'] = [
        'Latency ends at Python pull_chunk completion; it includes polling, batching and scheduling.',
        'Corrected latency depends on LSL clock estimation and is not a hardware latency measurement.',
        'TX and RX timestamp equality validates transport, not independent physical synchronization.',
        'Timing drift is relative to nominal acquisition rate; latency trend is not clock drift.',
        'online.npz is causal linear interpolation without extrapolation; missing intervals are NaN.',
        'Signals are synthetic, with a common coded pilot; units are illustrative.'
    ]
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report
