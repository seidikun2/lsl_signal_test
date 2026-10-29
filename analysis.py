"""Reconstruct by timestamp, measure errors and generate figures and summaries."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from helpers import sample_count


DEFAULT_LIMITS = dict(latency_ms=100., jitter_ms=10., drift_ppm=100.,
                      accumulated_ms=1., latency_growth_ms=20.)


def interpolate_segments(times, values, ids, grid, rate):
    """Linear interpolation only within contiguous, increasing sample segments."""
    result = np.full((len(grid), values.shape[1]), np.nan)
    if not len(times):
        return result
    cuts = np.flatnonzero((np.diff(ids) != 1) | (np.diff(times) <= 0) |
                         (np.diff(times) > 1.5/rate))+1
    for positions in np.split(np.arange(len(times)), cuts):
        if not len(positions):
            continue
        t, y = times[positions], values[positions]
        if len(t) == 1:
            exact = np.isclose(grid, t[0], rtol=0, atol=1e-9)
            result[exact] = y[0]
            continue
        left, right = np.searchsorted(grid, [t[0]-1e-9, t[-1]+1e-9])
        g = grid[left:right]
        j = np.searchsorted(t, g, side='right').clip(1, len(t)-1)
        weight = np.clip((g-t[j-1])/(t[j]-t[j-1]), 0, 1)
        result[left:right] = y[j-1] + weight[:, None]*(y[j]-y[j-1])
    return result


def delivery_hold(times, values, grid, ids):
    """Latest available sample per Python read; do not invent within-chunk times."""
    result = np.full((len(grid), values.shape[1]), np.nan)
    if not len(times):
        return result
    # Reads have nondecreasing monotonic times. Equal times belong to one batch.
    if np.any(np.diff(times) < 0):
        raise ValueError('Read-completion timestamps must be nondecreasing')
    last = np.r_[np.diff(times) > 0, True]
    if np.any(np.diff(ids) < 0):
        ends = np.flatnonzero(last)+1
        starts = np.r_[0, ends[:-1]]
        selected = np.array([a+np.argmax(ids[a:b]) for a, b in zip(starts, ends)])
    else:
        selected = np.flatnonzero(last)
    t, y, last_ids = times[selected], values[selected], ids[selected]
    # If packets were reordered, the latest acquisition ID already available wins.
    newest = last_ids == np.maximum.accumulate(last_ids)
    t, y = t[newest], y[newest]
    j = np.searchsorted(t, grid+1e-9, side='right')-1
    valid = j >= 0
    result[valid] = y[j[valid]]
    return result


def temporal_reconstruction(tx, rx, rate, origin, anchor_offset):
    # A single clock-domain translation preserves all time-varying corrections.
    # Never subtract each series' first timestamp or align by correlation here.
    grid_start = float(tx['sample'][0]-origin)
    rx_origin = origin+anchor_offset
    end = max(float(tx['sample'][-1]-origin),
              float(rx['arrival'][-1]-rx_origin) if len(rx['arrival']) else grid_start)
    grid = grid_start+np.arange(int(np.ceil((end-grid_start)*rate))+1)/rate
    reference = interpolate_segments(tx['sample']-origin, tx['values'], tx['ids'], grid, rate)
    acquisition = interpolate_segments(rx['corrected']-rx_origin, rx['values'], rx['ids'], grid, rate)
    delivery = delivery_hold(rx['arrival']-rx_origin, rx['values'], grid, rx['ids'])
    generated = delivery_hold(tx['generated']-origin, tx['values'], grid, tx['ids'])
    metrics = {}
    for name, values in [('acquisition', acquisition), ('delivery', delivery)]:
        mask = np.isfinite(reference).all(axis=1) & np.isfinite(values).all(axis=1)
        errors = values[mask]-reference[mask]
        metrics[name] = dict(points=int(mask.sum()),
            coverage=float(mask.sum()/max(1, np.isfinite(reference).all(axis=1).sum())),
            rmse=float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            rmse_per_channel=np.sqrt(np.mean(errors**2, axis=0)).tolist() if errors.size else [])
    return dict(time=grid, tx=reference, generated=generated,
                rx_acquisition=acquisition, rx_delivery=delivery), metrics


def windows(elapsed, latency, timing, width=2.):
    result = []
    bins = np.floor((elapsed-elapsed[0])/width).astype(int)
    for b in np.unique(bins):
        selected = bins == b
        t, delay, error = elapsed[selected], latency[selected], timing[selected]
        drift = float(np.polyfit(t-t[0], error-error[0], 1)[0]) if len(t)>1 and np.ptp(t)>0 else None
        result.append(dict(time=float(np.mean(t)), samples=int(selected.sum()),
                           latency_median_ms=float(np.median(delay)*1000),
                           latency_p99_ms=float(np.percentile(delay, 99)*1000),
                           drift_ppm=drift*1e6 if drift is not None else None))
    return result


def diagnose(metrics, trends, limits=None):
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    if any(not np.isfinite(x) or x <= 0 for x in limits.values()):
        raise ValueError('Analysis limits must be positive finite numbers')
    flags = []
    def flag(code, text):
        flags.append(dict(code=code, message=text))
    if metrics['missing']:
        flag('loss', f"{metrics['missing']} samples lost")
    if metrics['duplicates'] or metrics['reordered'] or metrics['unexpected']:
        flag('sequence', 'Duplicate, reordered or unexpected sample indices')
    if 'latency_ms' not in metrics:
        flag('no_samples', 'No matching samples available for timing analysis')
        return flags, limits
    if metrics['rmse'] > 0:
        flag('amplitude', 'Received values differ from transmitted values')
    if metrics['latency_ms']['p99'] > limits['latency_ms']:
        flag('latency', f"Latency p99 > {limits['latency_ms']:g} ms")
    if metrics['latency_ms']['std'] > limits['jitter_ms']:
        flag('jitter', f"Jitter > {limits['jitter_ms']:g} ms")
    if abs(metrics.get('rx_timing_drift_ppm') or 0) > limits['drift_ppm']:
        flag('drift', f"Acquisition drift > {limits['drift_ppm']:g} ppm in absolute value")
    if abs(metrics['rx_accumulated_drift_ms']) > limits['accumulated_ms']:
        flag('accumulated_drift', f"Accumulated acquisition drift > {limits['accumulated_ms']:g} ms in absolute value")
    if metrics.get('corrected_timestamp_reversals', 0):
        flag('timestamps', 'Corrected timestamps are not strictly increasing')
    timestamp_error = metrics['timestamp_error_ms']
    if max(abs(timestamp_error['min']), abs(timestamp_error['max'])) > limits['accumulated_ms']:
        flag('timestamp_mismatch', 'Received timestamps differ from source timestamps beyond the time limit')
    if len(trends) >= 4:
        n = max(1, len(trends)//3)
        growth = (np.median([w['latency_median_ms'] for w in trends[-n:]]) -
                  np.median([w['latency_median_ms'] for w in trends[:n]]))
        if growth > limits['latency_growth_ms']:
            flag('latency_growth', f'Median latency increased by {growth:.2f} ms between early and late windows')
        first = [abs(w['drift_ppm']) for w in trends[:n] if w['drift_ppm'] is not None]
        last = [abs(w['drift_ppm']) for w in trends[-n:] if w['drift_ppm'] is not None]
        if first and last and np.median(last)-np.median(first) > limits['drift_ppm']:
            flag('growing_drift', 'Absolute local drift increased between early and late windows')
    return flags, limits


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
    read_error = (arrival-arrival[0])-(elapsed-elapsed[0])
    diff = rx['values'][ir].astype(float)-tx['values'][it]
    metrics.update(
        latency_ms=distribution(latency*1000),
        send_to_read_ms=distribution(queue*1000),
        batching_ms=distribution((tx['sent'][it]-tx['sample'][it])*1000),
        generation_lateness_ms=distribution((tx['generated'][it]-tx['sample'][it])*1000),
        timestamp_error_ms=distribution(timestamp_error*1000),
        rmse=float(np.sqrt(np.mean(diff**2))), max_abs_error=float(np.max(np.abs(diff))),
        rmse_per_channel=np.sqrt(np.mean(diff**2, axis=0)).tolist(),
        corrected_timestamp_reversals=int(np.sum(np.diff(rx['corrected']) <= 0)),
        tx_timing_drift_ppm=slope(elapsed, timing_tx)*1e6 if len(common)>1 else None,
        rx_timing_drift_ppm=slope(elapsed, timing_rx)*1e6 if len(common)>1 else None,
        latency_trend_ppm=slope(elapsed, latency)*1e6 if len(common)>1 else None,
        tx_accumulated_drift_ms=float(timing_tx[-1]*1000),
        rx_accumulated_drift_ms=float(timing_rx[-1]*1000),
        arrival_accumulated_error_ms=float(read_error[-1]*1000),
        delivery_drift_ppm=slope(elapsed, read_error)*1e6 if len(common)>1 else None,
        latency_change_ms=float((latency[-1]-latency[0])*1000))
    return metrics, dict(common=common, it=it, ir=ir, elapsed=elapsed,
                         latency=latency, queue=queue, timing_tx=timing_tx,
                         timing_rx=timing_rx, timestamp_error=timestamp_error,
                         read_error=read_error)


def case_result(name, cfg, sensor, metrics=None, error=''):
    m = metrics or {}
    codes = [issue['code'] for issue in m.get('issues', [])]
    limit = m.get('limits', {**DEFAULT_LIMITS, **cfg.get('analysis_limits', {})})
    delivery = m.get('arrival_accumulated_error_ms')
    if delivery is not None and abs(delivery)>limit['accumulated_ms']:
        codes.append('delivery_variation')
    events = m.get('responses', {})
    if events.get('missing', 0):
        codes.append('missing_response')
    if events.get('delivery_mismatch', 0):
        codes.append('event_timing')
    if events.get('acquisition_mismatch', 0):
        codes.append('event_timestamp_error')
    if events.get('overlap', 0):
        codes.append('overlapping_responses')
    return dict(case=name, experiment=cfg.get('experiment','single'), stream=sensor['name'],
        status='failed' if error else 'completed', stream_count=len(cfg['sensors']),
        rate_hz=sensor['rate'], channels=sensor['channels'], chunk=cfg['chunk'], buffer_s=cfg['inlet_buffer'],
        pause_s=cfg['receiver_pause'], duration_s=sample_count(cfg,sensor)/sensor['rate'],
        injected_drift_ppm=sensor.get('drift_ppm',cfg['drift_ppm']),
        injected_slope_ppm_s=sensor.get('drift_slope_ppm_s',0),
        received=m.get('received'), expected=m.get('expected'), missing=m.get('missing'),
        latency_p99_ms=m.get('latency_ms',{}).get('p99'), jitter_ms=m.get('latency_ms',{}).get('std'),
        source_drift_ppm=m.get('tx_timing_drift_ppm'), acquisition_drift_ppm=m.get('rx_timing_drift_ppm'),
        accumulated_drift_ms=m.get('rx_accumulated_drift_ms'), delivery_error_ms=delivery,
        delivery_trend_ppm=m.get('delivery_drift_ppm'), latency_change_ms=m.get('latency_change_ms'),
        sample_rmse=m.get('rmse'), delivery_rmse=m.get('temporal_reconstruction',{}).get('delivery',{}).get('rmse'),
        responses_expected=events.get('expected'), responses_detected=events.get('detected'),
        responses_outside_recording=events.get('outside_recording'),
        response_error_ms=events.get('delivery_error_ms',{}).get('max'),
        drift_flag=abs(m['rx_timing_drift_ppm'])>limit['drift_ppm'] if m.get('rx_timing_drift_ppm') is not None else None,
        issues=';'.join(codes), error=error, latency_limit_ms=limit['latency_ms'],
        drift_limit_ppm=limit['drift_ppm'], accumulated_limit_ms=limit['accumulated_ms'])


def plot_snapshot(out, case, sensor, cfg, tx, rx, markers_tx, markers_rx, origin, offset):
    """Show at most two channels in the original common time domain."""
    import matplotlib.pyplot as plt
    channels=[c for c in cfg.get('plot_channels',[0,1]) if c<sensor['channels']][:2] or [0]
    window=cfg.get('snapshot_seconds',2.)
    fig,axes=plt.subplots(len(channels)+1,1,figsize=(13,3*len(channels)+2),constrained_layout=True)
    tx_time=tx['sample']-origin
    rx_time=rx['corrected']-origin-offset
    read_time=rx['arrival']-origin-offset
    colors={m['label']:plt.get_cmap('tab10')(i) for i,m in enumerate(cfg['markers'])}
    for ax,channel in zip(axes[:-1],channels):
        source=(tx_time>=0)&(tx_time<=window)
        acquisition=(rx_time>=0)&(rx_time<=window)
        delivery=(read_time>=0)&(read_time<=window)
        values=rx['values'][:,channel].astype(float).copy()
        values[np.r_[False,np.diff(rx['ids'])!=1]]=np.nan
        ax.plot(tx_time[source],tx['values'][source,channel],label='Transmitted',lw=1.2)
        ax.plot(rx_time[acquisition],values[acquisition],'--',label='Received: acquisition timestamps',lw=1)
        positions=np.flatnonzero(delivery)
        positions=positions[::max(1,len(positions)//1500)]
        ax.scatter(read_time[positions],rx['values'][positions,channel],s=5,alpha=.45,label='Received: read timestamps')
        for event in markers_tx:
            if event['kind']!='stimulus': continue
            t=event['sample_time']-origin
            if 0<=t<=window:
                color=colors[event['label']]
                ax.axvline(t,color=color,lw=.9)
                ax.text(t,.97,event['label'],transform=ax.get_xaxis_transform(),color=color,va='top')
                ax.axvline(t+event['delay'],color=color,ls=':',lw=.8)
        for event in markers_rx:
            if event['kind']!='stimulus': continue
            t=event['arrival']-origin-offset
            if 0<=t<=window: ax.axvline(t,color=colors[event['label']],ls='--',lw=.6,alpha=.7)
        ax.set(xlim=(0,window),ylabel=f'Channel {channel+1}')
        ax.grid(alpha=.15)
    axes[0].legend(loc='upper right',fontsize=8)
    axes[-1].plot(rx['raw']-origin,(rx['arrival']-rx['corrected'])*1000,lw=.7)
    axes[-1].set(xlabel='Time from common origin (s)',ylabel='Read latency (ms)',title='Full recording: acquisition-to-read delay')
    axes[-1].grid(alpha=.2)
    fig.suptitle(f'{case} / {sensor["name"]}: {sensor["rate"]:g} Hz, {sensor["channels"]} channels\n'
                 'Solid marker = TX; dashed = RX read; dotted = expected response. No time realignment.')
    fig.savefig(Path(out)/f'{case}_{sensor["name"]}_snapshot.png',dpi=120)
    plt.close(fig)


def write_metrics(out, rows):
    with (Path(out)/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def write_flat_comparison(out, rows):
    """One compact overview per test; no per-case folders."""
    import matplotlib.pyplot as plt
    out=Path(out)
    for start in range(0,len(rows),20):
        page=rows[start:start+20]
        fig,axes=plt.subplots(1,3,sharey=True,figsize=(15,max(4,.4*len(page)+2)),constrained_layout=True)
        for ax,(key,title,limit) in zip(axes,[('latency_p99_ms','Latency p99 (ms)','latency_limit_ms'),
                ('acquisition_drift_ppm','Acquisition drift (ppm)','drift_limit_ppm'),
                ('accumulated_drift_ms','Accumulated drift (ms)','accumulated_limit_ms')]):
            for i,r in enumerate(page):
                if r[key] is None: ax.text(0,i,'not measured',va='center')
                else: ax.barh(i,r[key],color='firebrick' if abs(r[key])>r[limit] else 'steelblue')
            ax.set_yticks(range(len(page)),[r['case']+'/'+r['stream'] for r in page],fontsize=8)
            ax.set_xlabel(title); ax.grid(axis='x',alpha=.2)
        axes[0].set_ylim(len(page)-.5,-.5)
        fig.suptitle('All measured cases — red: configured limit exceeded')
        fig.savefig(out/f'comparison_{start//20+1:02}.png',dpi=120); plt.close(fig)


def write_report(root, plan, rows, event_rows, marker_rows):
    """One self-contained HTML report with every case, not a selected subset."""
    from html import escape
    root=Path(root)
    def cell(value):
        if value is None: return '—'
        if isinstance(value,float): return f'{value:.3f}'
        return escape(str(value))
    def table(records, columns):
        if not records: return '<p>Not measured yet.</p>'
        text='<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(label)+'</th>' for _,label in columns)+'</tr></thead><tbody>'
        for row in records:
            issue=row.get('issues','') or row.get('status','')=='failed'
            text+='<tr'+(' class="issue"' if issue else '')+'>'+''.join('<td>'+cell(row.get(key))+'</td>' for key,_ in columns)+'</tr>'
        return text+'</tbody></table></div>'
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><title>LSL simulation results</title>',
          '<style>body{font:15px system-ui;margin:24px;color:#20242a}table{border-collapse:collapse;width:100%}th,td{padding:7px;border:1px solid #ddd;text-align:left;white-space:nowrap}th{background:#e8eef4;position:sticky;top:0}.scroll{overflow:auto;margin:15px 0}.issue{background:#fff0e8}img{max-width:100%;height:auto}details{margin:15px 0}small{color:#555}</style>',
          '<h1>LSL simulation — complete results</h1>',
          '<p>All planned cases are listed below. Blank measurements are not zero. Reload this file while the run progresses.</p>',
          '<p><a href="plan.csv">Plan CSV</a> · <a href="plan.png">Plan and estimated runtime</a></p>']
    html.append(table(plan,[('case','Case'),('test','Test'),('status','Status'),('sensors','Streams'),('rates_hz','Rates (Hz)'),
          ('channels','Channels'),('samples','Planned samples'),('chunk','Chunk'),('buffer_s','Buffer (s)'),('pause_s','Pause (s)'),
          ('acquisition_s','Acquire (s)'),('estimate_low_s','Estimate low (s)'),('estimate_high_s','Estimate high (s)'),('actual_s','Actual (s)'),('error','Error')]))
    for number,experiment in [(1,'eeg'),(2,'streams')]:
        selected=[r for r in rows if r['experiment']==experiment]
        if not any(p['test']==number for p in plan): continue
        folder=f'test_{number}'
        html += [f'<h2>Test {number}: '+('EEG channels × rate' if number==1 else 'Incremental streams')+'</h2>']
        downloads=[f'<a href="{folder}/{file}">{label}</a>' for file,label in
                   [('metrics.csv','All case metrics'),('events.csv','Every response event'),('markers.csv','Marker spacing by class')]
                   if (root/folder/file).exists()]
        html.append('<p>'+(' · '.join(downloads) if downloads else 'Measurements will appear after recording.')+'</p>')
        html.append(table(selected,[('case','Case'),('stream','Stream'),('stream_count','Streams'),('rate_hz','Hz'),('channels','Channels'),
              ('received','Received'),('expected','Transmitted'),('missing','Missing'),('latency_p99_ms','Latency p99 (ms)'),('jitter_ms','Jitter (ms)'),
              ('source_drift_ppm','TX drift (ppm)'),('acquisition_drift_ppm','RX drift (ppm)'),('accumulated_drift_ms','Accumulated drift (ms)'),
              ('delivery_error_ms','Delivery change (ms)'),('response_error_ms','Max response error (ms)'),('issues','Flags'),('error','Error')]))
        for image_path in sorted((root/folder).glob('*.png')):
            if '_snapshot' not in image_path.stem:
                html.append(f'<img src="{folder}/{escape(image_path.name)}" alt="All case comparisons">')
        events=[e for e in event_rows if e['test']==number]
        markers=[e for e in marker_rows if e['test']==number]
        html += ['<h3>Marker spacing — every case and class</h3>',table(markers,[('case','Case'),('class','Class'),('expected','TX'),('received','RX'),
                 ('missing','Missing'),('duplicates','Duplicates'),('expected_ms','Expected (ms)'),('tx_ms','TX median (ms)'),
                 ('rx_acquisition_ms','RX timestamp median (ms)'),('rx_delivery_ms','RX read median (ms)')]),
                 '<details><summary>Every marker-to-response measurement</summary>',
                 table(events,[('case','Case'),('stream','Stream'),('label','Class'),('sequence','Event'),('expected_ms','Expected (ms)'),('tx_ms','TX (ms)'),
                      ('rx_acquisition_ms','RX timestamps (ms)'),('rx_delivery_ms','RX read (ms)'),('delivery_error_ms','Delivery error (ms)'),('status','Status')]),'</details>',
                 '<h3>Signal snapshots — all recorded cases and streams</h3>']
        for r in selected:
            filename=f'{r["case"]}_{r["stream"]}_snapshot.png'
            if (root/folder/filename).exists():
                html.append(f'<details><summary>{escape(r["case"]+" / "+r["stream"])} — {r["rate_hz"]:g} Hz, {r["channels"]} channels</summary><img loading="lazy" src="{folder}/{filename}" alt="Transmitted and received signals with markers"></details>')
    html += ['<p>Delivery change is a last-minus-first timing difference, including chunk phase and queues; it is not pure clock drift. Accumulated drift is a time difference, not a sum of delays.</p>',
             '<p>Sample timestamps are simulated; read times are measured. Clock mapping does not remove delivery delay. Short-run drift estimates include LSL clock-estimation noise. Marker response detection uses the known synthetic baseline.</p>',
             '<p>Flag thresholds are configurable, not physical synchronization guarantees. A recording NPZ file is saved per case; earlier result folders are preserved outside the project.</p></html>']
    (root/'summary.html').write_text('\n'.join(html),encoding='utf-8')


def write_eeg_matrix(out, rows):
    import matplotlib.pyplot as plt
    out=Path(out)
    groups=sorted({(r['chunk'],r['buffer_s'],r['pause_s']) for r in rows})
    for index,group in enumerate(groups,1):
        selected=[r for r in rows if (r['chunk'],r['buffer_s'],r['pause_s'])==group]
        rates=sorted({r['rate_hz'] for r in selected}); channels=sorted({r['channels'] for r in selected})
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for ax,(key,label) in zip(axes.flat,[('latency_p99_ms','Latency p99 (ms)'),('acquisition_drift_ppm','Acquisition drift (ppm)'),
                                            ('missing','Missing samples'),('response_error_ms','Max absolute response delivery error (ms)')]):
            matrix=np.full((len(channels),len(rates)),np.nan)
            for i,c in enumerate(channels):
                for j,rate in enumerate(rates):
                    values=[r[key] for r in selected if r['channels']==c and r['rate_hz']==rate and r[key] is not None]
                    if values: matrix[i,j]=np.median(values)
            finite=matrix[np.isfinite(matrix)]
            bound=float(np.max(np.abs(finite))) if len(finite) else 1.
            bound=bound or 1.
            signed=key=='acquisition_drift_ppm'
            cmap=plt.get_cmap('coolwarm' if signed else 'viridis').with_extremes(bad='lightgray')
            image=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap=cmap,
                            vmin=-bound if signed else 0,vmax=bound)
            for i in range(len(channels)):
                for j in range(len(rates)):
                    value='not measured' if not np.isfinite(matrix[i,j]) else (f'{matrix[i,j]:g}' if key=='missing' else f'{matrix[i,j]:.2f}')
                    ax.text(j,i,value,ha='center',va='center',fontsize=8,
                            bbox=dict(facecolor='white',alpha=.75,edgecolor='none'))
            ax.set_xticks(range(len(rates)),[f'{r:g}' for r in rates]); ax.set_yticks(range(len(channels)),channels)
            ax.set(xlabel='Sampling rate (Hz)',ylabel='EEG channels',title=label)
            fig.colorbar(image,ax=ax,shrink=.8)
        fig.suptitle(f'EEG matrix — chunk {group[0]}, buffer {group[1]:g}s, pause {group[2]:g}s\nMedian of measured repetitions; see plan.csv for execution status')
        filename=f'eeg_matrix_{index:02}.png'
        fig.savefig(out/filename,dpi=120); plt.close(fig)


def analyze_responses(cfg, sensor, tx, rx, txmarkers, rxmarkers, figures):
    """Detect actual pulse samples against the known synthetic baseline."""
    import matplotlib.pyplot as plt
    from transmission import signal
    stimuli=[e for e in txmarkers if e['kind']=='stimulus']
    received={e['id']:e for e in rxmarkers if e['kind']=='stimulus'}
    scale={'EEG':60.,'EMG':140.,'IMU':3.}[sensor['kind'].upper()]
    baseline_tx=signal(sensor['kind'],tx['ids'],sensor['rate'],1)[:,0]
    baseline_rx=signal(sensor['kind'],rx['ids'],sensor['rate'],1)[:,0]
    residual_tx=tx['values'][:,0]-baseline_tx
    residual_rx=rx['values'][:,0]-baseline_rx
    records=[]
    for event in stimuli:
        onset=event['sample_time']+event['delay']
        stop=onset+event['width']
        record=dict(marker_id=event['id'],label=event['label'],sequence=event['sequence'],
                    expected_ms=event['delay']*1000,status='outside_recording',
                    tx_ms=None,rx_acquisition_ms=None,rx_delivery_ms=None,
                    acquisition_error_ms=None,delivery_error_ms=None,tx_sample_id=None,rx_sample_id=None)
        if not len(tx['sample']) or onset>tx['sample'][-1]+1e-9:
            records.append(record); continue
        overlap=any(other['id']!=event['id'] and
                    max(onset,other['sample_time']+other['delay']) <
                    min(stop,other['sample_time']+other['delay']+other['width'])-1e-9 for other in stimuli)
        if overlap:
            record['status']='overlap'; records.append(record); continue
        threshold=abs(scale*event['amplitude'])*.5
        sign=np.sign(event['amplitude'])
        ti=np.flatnonzero((tx['sample']>=onset-1e-9)&(tx['sample']<stop)&(sign*residual_tx>threshold))
        ri=np.flatnonzero((rx['raw']>=onset-1e-9)&(rx['raw']<stop)&(sign*residual_rx>threshold))
        record['status']='missing_response'
        if len(ti):
            t=ti[0]
            record['tx_ms']=float((tx['sample'][t]-event['sample_time'])*1000)
            record['tx_sample_id']=int(tx['ids'][t])
        if len(ri): record['rx_sample_id']=int(rx['ids'][ri[0]])
        marker=received.get(event['id'])
        if not len(ti): record['status']='missing_source_response'
        elif not marker: record['status']='missing_marker'
        elif len(ri) and rx['ids'][ri[0]]!=tx['ids'][ti[0]]: record['status']='onset_lost'
        elif len(ri):
            r=ri[0]
            record['status']='detected'
            record['rx_acquisition_ms']=float((rx['corrected'][r]-marker['corrected'])*1000)
            record['rx_delivery_ms']=float((rx['arrival'][r]-marker['arrival'])*1000)
            record['acquisition_error_ms']=record['rx_acquisition_ms']-record['tx_ms']
            record['delivery_error_ms']=record['rx_delivery_ms']-record['tx_ms']
        records.append(record)
    errors=[abs(r['delivery_error_ms']) for r in records if r['delivery_error_ms'] is not None]
    acq=[abs(r['acquisition_error_ms']) for r in records if r['acquisition_error_ms'] is not None]
    return dict(expected=sum(r['status']!='outside_recording' for r in records),
                detected=sum(r['status']=='detected' for r in records),
                missing=sum(r['status'] in ('missing_response','missing_source_response','onset_lost','missing_marker') for r in records),
                overlap=sum(r['status']=='overlap' for r in records),
                outside_recording=sum(r['status']=='outside_recording' for r in records),
                delivery_mismatch=sum(e>cfg['event_tolerance_ms'] for e in errors),
                acquisition_mismatch=sum(e>cfg['event_tolerance_ms'] for e in acq),
                acquisition_error_ms=distribution(acq),delivery_error_ms=distribution(errors),events=records)


def analyze_marker_classes(cfg, txmarkers, rxmarkers, figures):
    import matplotlib.pyplot as plt
    summaries={}
    classes=cfg.get('markers',[])
    if not classes: return summaries
    for marker in classes:
        label=marker['label']
        tx=[e for e in txmarkers if e.get('label')==label]
        raw_rx=[e for e in rxmarkers if e.get('label')==label]
        rx=sorted({e['id']:e for e in reversed(raw_rx)}.values(),key=lambda e:e['sequence'])
        expected_ids={e['id'] for e in tx}
        result=dict(expected=len(tx),received=len(rx),missing=len(expected_ids-{e['id'] for e in rx}),
                    duplicates=len(raw_rx)-len(rx),interval_s=marker['interval'])
        for name,events,key in [('tx',tx,'sample_time'),('rx_acquisition',rx,'corrected'),('rx_delivery',rx,'arrival')]:
            intervals=[]; errors=[]; positions=[]
            for a,b in zip(events,events[1:]):
                gap=b['sequence']-a['sequence']
                intervals.append((b[key]-a[key])*1000)
                errors.append(((b[key]-a[key])-gap*marker['interval'])*1000)
                positions.append(b['sequence'])
            result[name+'_interval_ms']=distribution(intervals)
            result[name+'_interval_error_ms']=distribution(errors)
        summaries[label]=result
    return summaries


def analyze(directory, output_dir=None, case='case'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source = recorded = Path(directory)
    figures = Path(output_dir) if output_dir else source
    figures.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((source/'config.json').read_text())
    txmeta = json.loads((source/'tx_meta.json').read_text())
    rxmeta = json.loads((recorded/'rx_meta.json').read_text())
    if txmeta['run_id'] != rxmeta['run_id']:
        raise ValueError('TX and RX run IDs differ')
    report = dict(run_id=txmeta['run_id'], tx_complete=txmeta['complete'],
                  rx_complete=rxmeta['complete'], sensors={})
    correction_path = recorded/'corrections.json'
    observations = json.loads(correction_path.read_text()) if correction_path.exists() else []
    # One fixed domain translation for every stream, not a fit to the signals.
    anchor_offset = next((o['offset'] for o in observations if o['stream'] != 'markers'), None)
    report['clock_anchor'] = dict(tx_origin=txmeta['start'], rx_minus_tx_initial=anchor_offset,
        method='Fixed initial LSL clock offset; no per-signal start alignment or delay removal')
    rxmarkers = json.loads((recorded/'rx_markers.json').read_text())
    txmarkers = json.loads((source/'tx_markers.json').read_text())
    expected_markers = {(e['kind'], e['id']) for e in txmarkers}
    got_markers = [(e['kind'], e['id']) for e in rxmarkers]
    report['markers'] = dict(expected=len(txmarkers), received=len(rxmarkers),
        missing=len(expected_markers-set(got_markers)), duplicates=len(got_markers)-len(set(got_markers)),
        latency_ms=distribution([(e['arrival']-e['corrected'])*1000 for e in rxmarkers]))
    for s in cfg['sensors']:
        name = s['name']
        with np.load(source/f'tx_{name}.npz') as f:
            tx = dict(f)
        with np.load(recorded/f'rx_{name}.npz') as f:
            rx = dict(f)
        metrics, data = compare(tx, rx, s['rate'])
        report['sensors'][name] = metrics
        if data is not None:
            if anchor_offset is None:
                anchor_offset = float(rx['correction'][0])
                report['clock_anchor']['rx_minus_tx_initial'] = anchor_offset
            reconstructed, temporal = temporal_reconstruction(tx, rx, s['rate'], txmeta['start'], anchor_offset)
            metrics['temporal_reconstruction'] = temporal
            trends = windows(data['elapsed'], data['latency'], data['timing_rx'])
            metrics['windows'] = trends
            metrics['issues'], metrics['limits'] = diagnose(metrics, trends, cfg.get('analysis_limits'))
            plot_snapshot(figures, case, s, cfg, tx, rx, txmarkers, rxmarkers, txmeta['start'], anchor_offset)
        else:
            metrics['issues'], metrics['limits'] = diagnose(metrics, [], cfg.get('analysis_limits'))
        if data is None:
            continue
        metrics['responses']=analyze_responses(cfg,s,tx,rx,txmarkers,rxmarkers,figures)
    report['marker_classes']=analyze_marker_classes(cfg,txmarkers,rxmarkers,figures)
    for metrics in report['sensors'].values():
        for key in ('missing','duplicates'):
            if report['markers'][key]:
                metrics['issues'].append(dict(code='marker_'+key,message=f'{report["markers"][key]} marker {key}'))
    report['notes'] = [
        'Latency ends at Python pull_chunk completion; it includes polling, batching and scheduling.',
        'Corrected latency depends on LSL clock estimation and is not a hardware latency measurement.',
        'TX and RX timestamp equality validates transport, not independent physical synchronization.',
        'Timing drift is relative to nominal acquisition rate; latency trend is not clock drift.',
        'Naive mode records delivery times; online interpolation is disabled unless explicitly enabled.',
        'Signals are synthetic; marker-triggered responses are detected from waveform changes against a known baseline.',
        'Temporal reconstruction uses each series timestamps on a shared grid, without delay compensation.',
        'Delivery reconstruction holds the latest available sample per read; within-chunk arrival times are not observable.',
        'Issue thresholds are illustrative configurable limits, not sensor requirements or physical certification.'
    ]
    (recorded/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    import tempfile
    from helpers import unpack_recording
    parser=argparse.ArgumentParser(description='Reanalyze one saved recording')
    parser.add_argument('recording',type=Path)
    args=parser.parse_args()
    case=args.recording.stem.removesuffix('_recording')
    with tempfile.TemporaryDirectory(prefix='lsl_analysis_') as temporary:
        unpack_recording(args.recording,temporary)
        analyze(temporary,args.recording.parent,case)
    print('Snapshots refreshed beside the recording file.')


if __name__=='__main__':
    main()
