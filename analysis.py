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


def delivery_hold(times, values, grid, ids, acquisition_times=None, max_age_s=None):
    """Latest available sample per Python read; do not invent within-chunk times."""
    if max_age_s is not None and (not np.isfinite(max_age_s) or max_age_s<=0 or acquisition_times is None):
        raise ValueError('A positive max_age_s requires acquisition timestamps')
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
    selected = selected[newest]
    t, y = t[newest], y[newest]
    j = np.searchsorted(t, grid+1e-9, side='right')-1
    valid = j >= 0
    if max_age_s is not None:
        ages = grid-np.asarray(acquisition_times)[selected][j.clip(0)]
        valid &= (ages>=-1e-9) & (ages<=max_age_s)
    result[valid] = y[j[valid]]
    return result


def buffered_playback(acquisition, values, arrival, ids, grid, rate, delay_s):
    """Replay every sample on an explicit delayed clock, only after real arrival.

    This is causal zero-order playback of the sampled signal, not an ideal
    band-limited reconstruction. Late samples and missing indices make gaps.
    """
    if not np.isfinite(delay_s) or delay_s<0:
        raise ValueError('Playback delay must be finite and nonnegative')
    result=np.full((len(grid),values.shape[1]),np.nan)
    latest=np.full(len(grid),-1,dtype=np.int64)
    target=grid-delay_s
    cuts=np.flatnonzero((np.diff(ids)!=1)|(np.diff(acquisition)<=0)|(np.diff(acquisition)>1.5/rate))+1
    for positions in np.split(np.arange(len(ids)),cuts):
        if not len(positions):
            continue
        stamps=acquisition[positions]
        j=np.searchsorted(stamps,target+1e-9,side='right')-1
        p=positions[j.clip(0)]
        valid=(j>=0)&(target<stamps[-1]+1/rate-1e-9)&(arrival[p]<=grid+1e-9)&(ids[p]>=latest)
        result[valid]=values[p[valid]]
        latest[valid]=ids[p[valid]]
    return result


def temporal_reconstruction(tx, rx, rate, origin, anchor_offset, grid_rate=None, until=None, max_age_s=None, playback_delay_s=None):
    # A single clock-domain translation preserves all time-varying corrections.
    # Never subtract each series' first timestamp or align by correlation here.
    grid_start = float(tx['sample'][0]-origin)
    rx_origin = origin+anchor_offset
    end = max(float(tx['sample'][-1]-origin),
              float(rx['arrival'][-1]-rx_origin) if len(rx['arrival']) else grid_start)
    grid_rate = rate if grid_rate is None else grid_rate
    if not np.isfinite(grid_rate) or grid_rate<=0:
        raise ValueError('Reconstruction grid rate must be positive and finite')
    if until is not None:
        if not np.isfinite(until) or until<grid_start:
            raise ValueError('Reconstruction end must be finite and at or after its start')
        end = until
        steps = int(np.floor((end-grid_start)*grid_rate+1e-9))
    else:
        steps = int(np.ceil((end-grid_start)*grid_rate))
    grid = grid_start+np.arange(steps+1)/grid_rate
    reference = interpolate_segments(tx['sample']-origin, tx['values'], tx['ids'], grid, rate)
    acquisition = interpolate_segments(rx['corrected']-rx_origin, rx['values'], rx['ids'], grid, rate)
    delivery = delivery_hold(rx['arrival']-rx_origin, rx['values'], grid, rx['ids'])
    fresh = delivery_hold(rx['arrival']-rx_origin, rx['values'], grid, rx['ids'],
                          rx['corrected']-rx_origin, max_age_s) if max_age_s is not None else delivery
    generated = (delivery_hold(tx['generated']-origin, tx['values'], grid, tx['ids'])
                 if 'generated' in tx else np.full_like(reference,np.nan))
    playback=(buffered_playback(rx['corrected']-rx_origin,rx['values'],rx['arrival']-rx_origin,
                               rx['ids'],grid,rate,playback_delay_s) if playback_delay_s is not None else None)
    metrics = {}
    for name, values in [('acquisition', acquisition), ('delivery', delivery)]:
        mask = np.isfinite(reference).all(axis=1) & np.isfinite(values).all(axis=1)
        errors = values[mask]-reference[mask]
        metrics[name] = dict(points=int(mask.sum()),
            coverage=float(mask.sum()/max(1, np.isfinite(reference).all(axis=1).sum())),
            rmse=float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            rmse_per_channel=np.sqrt(np.mean(errors**2, axis=0)).tolist() if errors.size else [])
    return dict(time=grid, tx=reference, generated=generated,
                rx_acquisition=acquisition, rx_delivery=delivery, rx_fresh=fresh, rx_playback=playback), metrics


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
    if 'sender_time_s' in tx and 'sender_time_s' in rx and 'receiver_time_s' in rx:
        metrics['counter_error_ms']=float(np.max(np.abs(tx['sender_time_s'][it]-rx['sender_time_s'][ir]))*1000)
        metrics['counter_delay_ms']=distribution((rx['receiver_time_s'][ir]-rx['sender_time_s'][ir])*1000)
        if 'sender_send_time_s' in tx:
            metrics['counter_after_send_ms']=distribution((rx['receiver_time_s'][ir]-tx['sender_send_time_s'][it])*1000)
    attempts=rx.get('read_attempt_time_s',np.empty(0))
    if len(attempts)>1 and np.ptp(attempts)>0:
        metrics['receiver_measured_hz']=float((len(attempts)-1)/np.ptp(attempts))
        metrics['read_gap_ms']=distribution(np.diff(attempts)*1000)
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
        receiver_target_hz=cfg.get('receiver_poll_hz'), receiver_measured_hz=m.get('receiver_measured_hz'),
        read_gap_p99_ms=m.get('read_gap_ms',{}).get('p99'),
        read_gap_max_ms=m.get('read_gap_ms',{}).get('max'),
        injected_drift_ppm=sensor.get('drift_ppm',cfg['drift_ppm']),
        injected_slope_ppm_s=sensor.get('drift_slope_ppm_s',0),
        received=m.get('received'), expected=m.get('expected'), missing=m.get('missing'),
        latency_p99_ms=m.get('counter_delay_ms',m.get('latency_ms',{})).get('p99'), jitter_ms=m.get('latency_ms',{}).get('std'),
        counter_error_ms=m.get('counter_error_ms'),
        after_send_p99_ms=m.get('counter_after_send_ms',m.get('send_to_read_ms',{})).get('p99'),
        source_drift_ppm=m.get('tx_timing_drift_ppm'), acquisition_drift_ppm=m.get('rx_timing_drift_ppm'),
        accumulated_drift_ms=m.get('rx_accumulated_drift_ms'), delivery_error_ms=delivery,
        delivery_trend_ppm=m.get('delivery_drift_ppm'), latency_change_ms=m.get('latency_change_ms'),
        sample_rmse=m.get('rmse'), delivery_rmse=m.get('temporal_reconstruction',{}).get('delivery',{}).get('rmse'),
        responses_expected=events.get('expected'), responses_detected=events.get('detected'),
        responses_outside_recording=events.get('outside_recording'),
        response_error_ms=events.get('delivery_error_ms',{}).get('max'),
        marker_sample_error_ms=events.get('acquisition_error_ms',{}).get('max'),
        drift_flag=abs(m['rx_timing_drift_ppm'])>limit['drift_ppm'] if m.get('rx_timing_drift_ppm') is not None else None,
        issues=';'.join(codes), error=error, latency_limit_ms=limit['latency_ms'],
        drift_limit_ppm=limit['drift_ppm'], accumulated_limit_ms=limit['accumulated_ms'])


def recorded_time_vectors(tx,rx,origin,offset):
    """Use transported sender counters and the receiver's recorded clock."""
    sent=tx['sender_time_s'] if 'sender_time_s' in tx else tx['sample']-origin
    carried=rx['sender_time_s'] if 'sender_time_s' in rx else rx['raw']-origin
    received=rx['receiver_time_s'] if 'receiver_time_s' in rx else rx['arrival']-origin-offset
    return sent,carried,received


def live_view(tx,rx,origin,offset,rate,seconds,channel):
    """Compare both latest-value views on one fixed clock; preserve real delay."""
    sent,_,received=recorded_time_vectors(tx,rx,origin,offset)
    grid=np.arange(int(np.floor(seconds*rate))+1)/rate
    source=delivery_hold(sent,tx['values'][:,[channel]],grid,tx['ids'])[:,0]
    available=delivery_hold(received,rx['values'][:,[channel]],grid,rx['ids'])[:,0]
    return grid,source,available


MARKER_COLORS=('#c45a00','#7951b0','#008578','#b82f65','#2474b5')


def plot_snapshot(out,case,sensor,cfg,tx,rx,markers_tx,markers_rx,origin,offset,response_events=()):
    """Show sample values, actual read availability, and the measured delay."""
    import matplotlib.pyplot as plt
    channel=next((c for c in cfg.get('plot_channels',[0]) if c<sensor['channels']),0)
    window=cfg.get('snapshot_seconds',2.)
    eeg=sensor['kind'].upper()=='EEG'
    fig,axes=plt.subplots(4 if eeg else 3,1,figsize=(14,11 if eeg else 8),constrained_layout=True)
    sent,carried,received=recorded_time_vectors(tx,rx,origin,offset)
    source=(sent>=0)&(sent<=window)
    recorded=(carried>=0)&(carried<=window)
    values=rx['values'][:,channel].astype(float).copy()
    values[np.r_[False,np.diff(rx['ids'])!=1]]=np.nan
    axes[0].plot(sent[source],tx['values'][source,channel],label='Sent signal',lw=1.2)
    axes[0].plot(carried[recorded],values[recorded],'--',label='Received signal',lw=1)
    axes[0].set(title='Signal values — sample time carried by the sender',ylabel='Value',xlim=(0,window))
    display_rate=cfg.get('reconstruction_rate_hz',2048)
    grid,source_live,received_live=live_view(tx,rx,origin,offset,display_rate,window,channel)
    axes[1].step(grid,source_live,where='post',label='Sent signal',lw=1.2)
    axes[1].step(grid,received_live,where='post',label='Received signal',lw=1.2,ls='--',color='tab:green')
    axes[1].set(title=f'Live view at {display_rate:g} Hz — latest available value, not the full reconstruction',ylabel='Value',xlim=(0,window))
    axes[2].plot(carried,(received-carried)*1000,lw=.8,label='Since sample time')
    common,it,ir=np.intersect1d(tx['ids'],rx['ids'],return_indices=True)
    if len(common):
        send_times=tx['sender_send_time_s'][it] if 'sender_send_time_s' in tx else tx['sent'][it]-origin
        axes[2].plot(carried[ir],(received[ir]-send_times)*1000,lw=.8,label='Since send call')
    axes[2].set(title='Delay until the receiver read the sample',ylabel='Delay (ms)',xlabel='Time from shared start (s)')
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:
        ax.legend(loc='upper right',fontsize=8)
    if eeg:
        from matplotlib.lines import Line2D
        labels=list(dict.fromkeys(e['label'] for e in response_events))
        colors={label:MARKER_COLORS[i%len(MARKER_COLORS)] for i,label in enumerate(labels)}
        for event in response_events:
            y=labels.index(event['label'])
            color=colors[event['label']]
            for label,first,last,dy,style,fill in [
                ('Sent',event.get('marker_sent_s'),event.get('signal_changed_sent_s'),.20,'-',color),
                ('Received',event.get('marker_received_s'),event.get('signal_changed_received_s'),-.20,'--','white')]:
                if first is not None and first<=window:
                    axes[3].plot(first,y+dy,marker='o',ms=9,color=color,mfc=fill,mew=2)
                if last is not None and last<=window:
                    axes[3].plot(last,y+dy,marker='^',ms=9,color=color,mfc=fill,mew=2)
                if first is not None and last is not None and first<=window:
                    axes[3].plot([first,last],[y+dy,y+dy],color=color,lw=2.8,ls=style)
                if first is not None and 0<=first<=window:
                    axes[0 if label=='Sent' else 1].axvline(first,color=color,ls=style,lw=2.2,alpha=.8)
        axes[3].set_yticks([i+d for i in range(len(labels)) for d in (.20,-.20)],
                           [f'{label} — {kind}' for label in labels for kind in ('sent','received')])
        axes[3].set_ylim(-.55,max(.55,len(labels)-.45))
        axes[3].set(title='Markers and the EEG change they caused',xlabel='Time from shared start (s)',xlim=(0,window),ylabel='Marker')
        axes[3].legend(handles=[Line2D([],[],color='.25',lw=2.8,ls=style,marker=marker,ms=7,mfc=fill,label=label)
            for label,style,marker,fill in [('Sent marker','-','o','.25'),('Received marker','--','o','white'),
                ('Sent EEG change','-','^','.25'),('Received EEG change','--','^','white')]],fontsize=8,ncol=2,loc='upper left')
    for ax in axes:ax.grid(alpha=.18)
    caption='Top: all sample values. Second: fixed display clock. '
    caption+=('Marker color identifies A, B or C; solid = sent, dashed = received.' if eeg else 'All samples remain saved; no curve shifting.')
    fig.suptitle(f'{case} / {sensor["name"].upper()} — {sensor["rate"]:g} Hz, {sensor["channels"]} channels\n'+caption,fontsize=10)
    fig.savefig(Path(out)/f'{case}_{sensor["name"]}_snapshot.png',dpi=115)
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


def result_matrix(rows,key,x_values,y_values,x_key,y_key):
    """Keep missing/failed cells distinct from measured zeros."""
    result=np.full((len(y_values),len(x_values)),np.nan)
    failed=np.zeros(result.shape,dtype=bool)
    for i,y in enumerate(y_values):
        for j,x in enumerate(x_values):
            cell=[r for r in rows if r.get(x_key)==x and r.get(y_key)==y]
            failed[i,j]=any(r.get('status')=='failed' for r in cell)
            values=[]
            for r in cell:
                value=(100*r['missing']/r['expected'] if key=='missing_pct' and r.get('expected') and r.get('missing') is not None else r.get(key))
                if r.get('status')!='failed' and value is not None and np.isfinite(value):values.append(value)
            if values:result[i,j]=max(values)
    return result,failed


def _result_maps(out,rows,x_values,y_values,x_key,y_key,x_labels,y_labels,title,filename):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,3,figsize=(18,9),constrained_layout=True)
    fields=[('after_send_p99_ms','Delay after send p99 (ms)'),('latency_p99_ms','Sample-to-read delay p99 (ms)'),('missing_pct','Missing samples (%)'),
            ('sample_rmse','Difference in signal values'),('marker_sample_error_ms','Signal / marker\nstored-time difference (ms)'),
            ('response_error_ms','Signal / marker\narrival-time difference (ms)')]
    for ax,(key,label) in zip(axes.flat,fields):
        matrix,failed=result_matrix(rows,key,x_values,y_values,x_key,y_key)
        finite=matrix[np.isfinite(matrix)]
        upper=float(np.max(finite)) if len(finite) else 1.
        image=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap=plt.get_cmap('viridis').with_extremes(bad='#dddddd'),vmin=0,vmax=upper or 1.)
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                value=matrix[i,j]
                text='—' if not np.isfinite(value) else f'{value:.2f}'
                ax.text(j,i,text,ha='center',va='center',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
                if failed[i,j]:ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor='red',lw=2))
        ax.set_xticks(range(len(x_values)),x_labels)
        ax.set_yticks(range(len(y_values)),y_labels)
        ax.set(title=label,xlabel='Sampling rate (Hz)' if x_key=='rate_hz' else 'Active signals',ylabel='EEG channels' if y_key=='channels' else 'Received signal')
        fig.colorbar(image,ax=ax,shrink=.8)
    fig.suptitle(title+'\nHighest recorded value across repetitions. Gray = not measured / not active. Red outline = a failed run.',fontsize=11)
    Path(out).mkdir(parents=True,exist_ok=True)
    fig.savefig(Path(out)/filename,dpi=120)
    plt.close(fig)


def write_eeg_matrix(out,rows,plan=None):
    if not rows:return
    groups=sorted({(r['chunk'],r['buffer_s'],r['pause_s']) for r in rows})
    for index,group in enumerate(groups,1):
        selected=[r for r in rows if (r['chunk'],r['buffer_s'],r['pause_s'])==group]
        rates=sorted({r['rate_hz'] for r in selected})
        channels=sorted({r['channels'] for r in selected})
        if plan:
            eeg=[p for p in plan if p['test']==1]
            rates=sorted(set(rates)|{float(p['rates_hz']) for p in eeg})
            channels=sorted(set(channels)|{int(p['channels']) for p in eeg})
        _result_maps(out,selected,rates,channels,'rate_hz','channels',[f'{r:g}' for r in rates],channels,
                     f'Test 1 — EEG only; {group[0]} samples per send',f'eeg_matrix_{index:02}.png')


def write_stream_matrix(out,rows):
    if not rows:return
    stages=sorted({r['stream_count'] for r in rows})
    streams=list(dict.fromkeys(r['stream'] for r in rows))
    labels=[' + '.join(streams[:count]).upper() for count in stages]
    _result_maps(out,rows,stages,streams,'stream_count','stream',labels,[s.upper() for s in streams],
                 'Test 2 — adding signals; existing rates and channels stay fixed','streams_matrix.png')


def write_results_report(root,plan,rows,events,assets,maps,machine):
    """Results first; two separate experiments; short explanations, no verdicts."""
    import html
    import json
    from collections import Counter
    root=Path(root)
    esc=lambda value:html.escape(str(value))
    def cell(value):
        if value is None:return '—'
        return f'{value:.3g}' if isinstance(value,float) else esc(value)
    def table(items,columns):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(label)+'</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(r.get(key))+'</td>' for key,_ in columns)+'</tr>' for r in items)+'</tbody></table></div>'
    counts=Counter(p['status'] for p in plan)
    read_note=('No extra processing wait is imposed in these two tests.' if all(p.get('pause_s',0)==0 for p in plan)
               else 'An extra receiver pause is configured; check the parameters before treating this as normal reading.')
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>LSL — EEG and additional signals</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#17233a;background:#f6f8fc}section{background:white;padding:24px;margin:18px 0;border-radius:10px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef2f8}.scroll{overflow:auto}p{line-height:1.5}select{padding:8px;max-width:100%;margin:10px 0}summary{cursor:pointer;padding:10px 0}code{background:#eef2f8;padding:2px 5px}</style>',
        '<h1>LSL — recorded results</h1>',
        f'<p>{counts["completed"]} completed · {counts["failed"]} failed · {counts["planned"]+counts["running"]} not finished. {read_note}</p>']
    if machine.get('label'):parts.append('<p><b>'+esc(machine['label'])+'</b></p>')
    if machine:parts.append('<p>Recorded on '+esc(machine.get('system',''))+' / Python '+esc(machine.get('python',''))+'.</p>')
    targets=sorted({p['receiver_poll_hz'] for p in plan if p.get('receiver_poll_hz')})
    if targets:
        parts.append('<p>Receiver reading target: '+', '.join(f'{v:g} Hz' for v in targets)+'. Every cycle reads buffered samples from every stream. The operating system can delay a cycle; actual reading rates and gaps are listed below. Reading faster does not change sensor sampling rates.</p>')
    for experiment,number,title in [('eeg',1,'Test 1 — EEG only'),('streams',2,'Test 2 — adding signals')]:
        selected=[r for r in rows if r['experiment']==experiment]
        scheduled=[p for p in plan if p['test']==number]
        if not scheduled:continue
        parts.append('<section><h2>'+title+'</h2>')
        if number==1:parts.append('<p>Rows are EEG channel counts. Columns are sampling rates. Each cell shows the highest measured value across repetitions. The table below keeps every run.</p>')
        else:parts.append('<p>First EEG, then EEG + EMG, then EEG + EMG + IMU. Existing signals keep their rates and channel counts. This is a separate experiment from the EEG matrix.</p>')
        for key,image in maps.items():
            if key.startswith(experiment):parts.append('<img src="'+image+'" alt="'+esc(title)+' result maps">')
        if not any(key.startswith(experiment) for key in maps):parts.append('<p>Maps will be available when the recorded cases are processed.</p>')
        parts.append('<p><b>After send:</b> time from the send call to the read. <b>Sample to read:</b> also includes waiting for the sender to fill a batch. <b>p99:</b> 99% of samples had a delay at or below this value. <b>Signal difference:</b> zero means the values match. The two marker maps compare the stored sample timing and the actual read timing separately. Colors follow each map\'s printed scale; they are not pass/fail categories.</p>')
        choices=[r for r in selected if r['case']+'_'+r['stream'] in assets]
        parts.append('<h3>Sent signal and received signal</h3>')
        if choices:
            options=''.join('<option value="'+esc(r['case']+'_'+r['stream'])+'">'+esc(f'{r["case"]} / {r["stream"].upper()} — {r["rate_hz"]:g} Hz, {r["channels"]} channels')+'</option>' for r in choices)
            parts.append(f'<label>Recorded case <select data-gallery="signal-{number}">{options}</select></label><img id="signal-{number}" alt="Sent and received signal, time counters, delay and EEG markers">')
        else:parts.append('<p>No signal recording is available yet.</p>')
        parts.append('<p>The first plot uses every recorded sample and its carried sender time. The second is a live view on one fixed display clock: both curves keep the latest value available at each display step. It is not the full reconstruction. Empty reads retain the previous value; all samples remain saved. The delay plot uses actual read times. Curves are never shifted to fit. A faster reading loop cannot replace a higher acquisition rate.</p>')
        measured_events=[e for e in events if e['test']==number and e['stream'].lower()=='eeg']
        parts.append('<h3>EEG markers and the change in the signal</h3>')
        parts.append('<p>A, B and C cause the configured change in the sent EEG, normally 200 ms later. One color identifies each marker. Solid lines and filled symbols mean sent; dashed lines and hollow symbols mean received. Circles are markers; triangles are EEG changes. Sent times use the source sample counter; received times use the receiver read counter.</p>')
        if measured_events:
            display=[]
            for e in measured_events:
                r=dict(e)
                r['result']={'detected':'Change found','outside_recording':'Change falls after recording ends','missing_response':'EEG change missing','missing_source_response':'Sent EEG change missing','onset_lost':'First changed sample missing','missing_marker':'Marker missing','overlap':'Overlapping changes'}.get(e['status'],e['status'])
                display.append(r)
            parts.append('<details><summary>Show the recorded marker and EEG-change times</summary>'+table(display,[('case','Case'),('label','Marker'),('marker_sent_s','Sent marker (s)'),('signal_changed_sent_s','Sent EEG change (s)'),('marker_received_s','Received marker (s)'),('signal_changed_received_s','Received EEG change (s)'),('acquisition_error_ms','Stored timing difference (ms)'),('delivery_error_ms','Arrival timing difference (ms)'),('result','Recorded result')])+'</details>')
        else:parts.append('<p>No marker-response measurements are available for this section.</p>')
        if number==2:
            parts.append('<h3>Synchronization between sensors</h3>')
            sync_cases=[p['case'] for p in scheduled if p['case']+'_sync' in assets]
            if sync_cases:
                parts.append('<p>The same known event changes every simulated sensor. The first chart compares sample times for analysis; the second compares actual read times. Different acquisition rates can place the first changed sample slightly apart. The table separates that original difference from any added timing difference.</p>')
                options=''.join('<option value="'+esc(case+'_sync')+'">'+esc(case)+'</option>' for case in sync_cases)
                parts.append('<label>Simultaneous recording <select data-gallery="stream-sync">'+options+'</select></label><img id="stream-sync" alt="Same observed event in all sensors, before and after transmission">')
                timing=[e for e in events if e['test']==2 and e['stream']!=e.get('reference_stream')]
                parts.append('<details><summary>Time differences relative to EEG — every common event</summary>'+table(timing,[('case','Case'),('stream','Signal'),('label','Event'),('source_offset_ms','Sent difference (ms)'),('stored_offset_ms','Received sample difference (ms)'),('stored_offset_error_ms','Added sample difference (ms)'),('read_offset_ms','Read-time difference (ms)'),('status','Event result')])+'</details>')
            else:parts.append('<p>No simultaneous common-event recording is available yet.</p>')
        parts.append('<details><summary>Every recorded result</summary>'+table(selected,[('case','Case'),('stream','Signal'),('rate_hz','Acquisition (Hz)'),('channels','Channels'),('receiver_target_hz','Reading target (Hz)'),('receiver_measured_hz','Measured reading (Hz)'),('read_gap_p99_ms','Reading gap p99 (ms)'),('read_gap_max_ms','Largest reading gap (ms)'),('received','Samples received'),('missing','Samples missing'),('after_send_p99_ms','After send p99 (ms)'),('latency_p99_ms','Sample to read p99 (ms)'),('sample_rmse','Signal difference'),('counter_error_ms','Carried time difference (ms)'),('status','Run result'),('error','Error')])+'</details></section>')
    parts+=['<section><h2>After viewing the results</h2><p>These are recorded measurements, not an automatic diagnosis. Sample loss, delivery delay and marker timing are separate results. The simulated sensors share one sender clock; independent physical devices need their own clock synchronization measurements.</p>',
        '<details><summary>Extra tests — slow processing and recovery</summary><p>Run <code>python main.py --local --out results-extra</code> only as a separate experiment. These tests deliberately add processing time. Recovery means the waiting data return to the initial level after that extra wait is removed, while transmission continues. A processing-time ratio of 1 uses all available time; 2 takes twice as long. These extra tests are not part of the normal LSL check above.</p></details>',
        '<p>Files: <code>metrics.csv</code>, <code>markers.csv</code>, and <code>recordings.zip</code>. The archive contains the original signal arrays and the sender/receiver time vectors for reanalysis.</p></section>']
    failed=[p for p in plan if p['status']=='failed']
    if failed:parts.append('<section><h2>Runs that did not finish</h2>'+table(failed,[('case','Case'),('error','Error')])+'</section>')
    payload=json.dumps(assets,separators=(',',':')).replace('<','\\u003c')
    parts.append('<script type="application/json" id="signal-images">'+payload+'</script><script>const images=JSON.parse(document.getElementById("signal-images").textContent);document.querySelectorAll("select[data-gallery]").forEach(s=>{const update=()=>{document.getElementById(s.dataset.gallery).src=images[s.value]};s.addEventListener("change",update);update()});</script></html>')
    (root/'summary.html').write_text('\n'.join(parts),encoding='utf-8')


def analyze_responses(cfg, sensor, tx, rx, txmarkers, rxmarkers, figures, origin=None, offset=0.):
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
    origin=float(tx['sample'][0]) if origin is None else origin
    for event in stimuli:
        onset=event['sample_time']+event['delay']
        stop=onset+event['width']
        record=dict(marker_id=event['id'],label=event['label'],sequence=event['sequence'],
                    expected_ms=event['delay']*1000,status='outside_recording',
                    tx_ms=None,rx_acquisition_ms=None,rx_delivery_ms=None,
                    acquisition_error_ms=None,delivery_error_ms=None,tx_sample_id=None,rx_sample_id=None,
                    marker_sent_s=float(event['sample_time']-origin),marker_received_s=None,
                    signal_changed_sent_s=None,signal_changed_received_s=None,signal_sample_received_s=None)
        marker=received.get(event['id'])
        if marker:record['marker_received_s']=float(marker['arrival']-origin-offset)
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
            record['signal_changed_sent_s']=float(tx['sample'][t]-origin)
        if len(ri):
            record['rx_sample_id']=int(rx['ids'][ri[0]])
            record['signal_changed_received_s']=float(rx['arrival'][ri[0]]-origin-offset)
            record['signal_sample_received_s']=float(rx['corrected'][ri[0]]-origin-offset)
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
            record['signal_changed_received_s']=float(rx['arrival'][r]-origin-offset)
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


def stream_timing(events,reference='eeg'):
    """Compare the same observed event across streams without fitting clocks."""
    reference_events={e['marker_id']:e for e in events if e['stream']==reference}
    result=[]
    for event in events:
        row=dict(event,reference_stream=reference,source_offset_ms=None,stored_offset_ms=None,
                 stored_offset_error_ms=None,read_offset_ms=None)
        other=reference_events.get(event['marker_id'])
        if other and event['status']==other['status']=='detected':
            row['source_offset_ms']=(event['signal_changed_sent_s']-other['signal_changed_sent_s'])*1000
            row['stored_offset_ms']=(event['signal_sample_received_s']-other['signal_sample_received_s'])*1000
            row['stored_offset_error_ms']=row['stored_offset_ms']-row['source_offset_ms']
            row['read_offset_ms']=(event['signal_changed_received_s']-other['signal_changed_received_s'])*1000
        result.append(row)
    return result


def plot_stream_timing(out,case,events,seconds):
    import matplotlib.pyplot as plt
    streams=list(dict.fromkeys(e['stream'] for e in events))
    labels=list(dict.fromkeys(e['label'] for e in events))
    colors={label:MARKER_COLORS[i%len(MARKER_COLORS)] for i,label in enumerate(labels)}
    fig,axes=plt.subplots(2,1,figsize=(14,8),constrained_layout=True)
    for ax,received_key,title in zip(axes,['signal_sample_received_s','signal_changed_received_s'],
            ['Synchronization for analysis — sample times saved in each stream',
             'When each change became available — actual receiver read times']):
        for event in events:
            if event['status']!='detected':continue
            i=streams.index(event['stream']);color=colors[event['label']]
            for key,y,style,fill in [('signal_changed_sent_s',2*i,'-',color),(received_key,2*i+1,'--','white')]:
                stamp=event.get(key)
                if stamp is None or not 0<=stamp<=seconds:continue
                ax.vlines(stamp,y-.28,y+.28,color=color,linestyles=style,lw=3)
                ax.plot(stamp,y,'o',ms=8,color=color,mfc=fill,mew=2)
                if i==0 and key=='signal_changed_sent_s':
                    ax.text(stamp,-.65,'Marker '+event['label'],color=color,ha='center',fontsize=10,weight='bold')
        ax.set_yticks(range(2*len(streams)),[f'{s.upper()} — {kind}' for s in streams for kind in ('sent','received')])
        ax.set(xlim=(0,seconds),ylim=(2*len(streams)-.4,-1.2),title=title,xlabel='Time from shared start (s)')
        ax.grid(axis='x',alpha=.25)
    fig.suptitle(f'{case} — the same known event in every sensor\n'
                 'Same color = same event. Solid / filled = sent. Dashed / hollow = received. No curve shifting.',fontsize=11)
    fig.savefig(Path(out)/f'{case}_sync_snapshot.png',dpi=115)
    plt.close(fig)


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
    report['markers']['stimulus_sent']=sum(e['kind']=='stimulus' for e in txmarkers)
    report['markers']['stimulus_received']=sum(e['kind']=='stimulus' for e in rxmarkers)
    for s in cfg['sensors']:
        name = s['name']
        with np.load(source/f'tx_{name}.npz') as f:
            tx = dict(f)
        with np.load(recorded/f'rx_{name}.npz') as f:
            rx = dict(f)
        metrics, data = compare(tx, rx, s['rate'])
        report['sensors'][name] = metrics
        if s['kind'].upper()=='EEG' or cfg.get('shared_sensor_events'):
            metrics['responses']=analyze_responses(cfg,s,tx,rx,txmarkers,rxmarkers,figures,
                                                  origin=txmeta['start'],offset=anchor_offset or 0.)
        if data is not None:
            if anchor_offset is None:
                anchor_offset = float(rx['correction'][0])
                report['clock_anchor']['rx_minus_tx_initial'] = anchor_offset
            reconstructed, temporal = temporal_reconstruction(tx, rx, s['rate'], txmeta['start'], anchor_offset)
            metrics['temporal_reconstruction'] = temporal
            trends = windows(data['elapsed'], data['latency'], data['timing_rx'])
            metrics['windows'] = trends
            metrics['issues'], metrics['limits'] = diagnose(metrics, trends, cfg.get('analysis_limits'))
            plot_snapshot(figures, case, s, cfg, tx, rx, txmarkers, rxmarkers, txmeta['start'], anchor_offset,
                          response_events=metrics.get('responses',{}).get('events',[]))
        else:
            metrics['issues'], metrics['limits'] = diagnose(metrics, [], cfg.get('analysis_limits'))
        if data is None:
            continue
    reference=next((s['name'] for s in cfg['sensors'] if s['kind'].upper()=='EEG'),cfg['sensors'][0]['name'])
    report['stream_timing']=stream_timing([dict(stream=name,**event)
        for name,metrics in report['sensors'].items() for event in metrics.get('responses',{}).get('events',[])],reference)
    if cfg.get('shared_sensor_events') and len(cfg['sensors'])>1:
        plot_stream_timing(figures,case,report['stream_timing'],cfg.get('snapshot_seconds',2.))
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
