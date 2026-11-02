"""Controlled LSL consumer overload, direct queue telemetry, and one compact report."""
import base64
import csv
import html
import io
import json
import math
import multiprocessing as mp
import platform
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np


# Compatibility for recordings created before report output was standardized.
LEGACY_STATES = {
    'limite de observação': 'observation limit',
    'descarte observado': 'data loss observed',
    'fila crescente': 'growing queue',
    'janela insuficiente': 'insufficient windows',
    'sem crescimento sustentado': 'no sustained growth',
    'sobrecarga não demonstrada': 'overload not demonstrated',
    'fila recuperada; drenagem incompleta': 'queue recovered; drain incomplete',
    'recuperou com descarte': 'recovered with data loss',
    'recuperou durante transmissão': 'recovered during transmission',
    'recuperação não demonstrada': 'recovery not demonstrated',
    'falha técnica': 'technical failure',
}


def english_rows(rows):
    """Translate legacy diagnostics without changing saved samples or metrics."""
    return [dict(row, state=LEGACY_STATES.get(row.get('state'),row.get('state'))) for row in rows]


def build_plan(cfg):
    positive = ('duration_s', 'rate_hz', 'channels', 'tx_chunk', 'read_chunk',
                'buffer_s', 'repeats', 'extra_repeats', 'max_drain_s', 'window_s')
    for key in positive:
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('channels', 'tx_chunk', 'read_chunk', 'buffer_s', 'repeats', 'extra_repeats'):
        if int(cfg[key]) != cfg[key]:
            raise ValueError(f'{key} must be an integer')
    if cfg['duration_s'] < 4 * cfg['window_s']:
        raise ValueError('duration_s must include at least four analysis windows')
    if not math.isfinite(cfg['rate_sweep_processing_s_per_sample']) or cfg['rate_sweep_processing_s_per_sample'] < 0:
        raise ValueError('Processing cost must be finite and nonnegative')
    for key in ('loads', 'buffers_s', 'rates_hz', 'tx_chunks'):
        if len(set(cfg[key])) != len(cfg[key]) or any(not math.isfinite(v) or v <= 0 for v in cfg[key]):
            raise ValueError(f'{key} must contain unique positive finite values')
    if not cfg['loads']:
        raise ValueError('At least one load is required')
    for key in ('buffers_s', 'tx_chunks'):
        if any(int(v) != v for v in cfg[key]):
            raise ValueError(f'{key} must contain integers')
    rows = []

    def add(axis, value, repeat, **overrides):
        row = dict(axis=axis, value=value, repeat=repeat, rate_hz=cfg['rate_hz'],
                   channels=cfg['channels'], tx_chunk=cfg['tx_chunk'], read_chunk=cfg['read_chunk'],
                   buffer_s=cfg['buffer_s'], duration_s=cfg['duration_s'],
                   processing_s_per_sample=.25/cfg['rate_hz'])
        row.update(overrides)
        row['load'] = row['rate_hz'] * row['processing_s_per_sample']
        row['expected'] = math.ceil(row['rate_hz'] * row['duration_s'])
        rows.append(row)

    for repeat in range(1, cfg['repeats']+1):
        for load in cfg['loads']:
            add('load', load, repeat, processing_s_per_sample=load/cfg['rate_hz'])
    for repeat in range(1, cfg['extra_repeats']+1):
        for size in cfg['buffers_s']:
            add('buffer', size, repeat, buffer_s=size, processing_s_per_sample=3/cfg['rate_hz'])
        for rate in cfg['rates_hz']:
            add('rate', rate, repeat, rate_hz=rate, buffer_s=1,
                processing_s_per_sample=cfg['rate_sweep_processing_s_per_sample'])
        for chunk in cfg['tx_chunks']:
            add('chunk', chunk, repeat, tx_chunk=chunk)
    # Randomize order, retaining repeat IDs for the prespecified paired comparison.
    np.random.default_rng(cfg['seed']).shuffle(rows)
    for i, row in enumerate(rows, 1):
        row['case'] = f'B{i:03}'
    return rows


def local_plan(document):
    """Target-machine protocol. Existing signal parameters stay fixed across stages."""
    cfg = dict(document['buffer_tests'], **document['local_tests'])
    # Reuse validation, without running the exploratory axes from the short protocol.
    cfg.update(extra_repeats=1, buffers_s=[], rates_hz=[], tx_chunks=[])
    plan = build_plan(cfg)
    sensors = document['acquisition']['sensors']
    if not sensors or len({s['name'] for s in sensors}) != len(sensors):
        raise ValueError('Local streams require unique sensor names')
    from helpers import validate
    validate(dict(sensors=sensors, duration=cfg['duration_s']))
    cfg['stream_sensors'] = [dict(s) for s in sensors]
    for key,default in [('reconstruction_rate_hz',16384),('delivery_max_age_s',.25),('snapshot_seconds',2.)]:
        cfg[key] = document['acquisition'].get(key,default)
        if not math.isfinite(cfg[key]) or cfg[key]<=0:
            raise ValueError(f'{key} must be positive and finite')
    cfg['playback_delay_s']=document['acquisition'].get('playback_delay_s',.25)
    if not math.isfinite(cfg['playback_delay_s']) or cfg['playback_delay_s']<0:
        raise ValueError('playback_delay_s must be finite and nonnegative')
    cost = cfg['stream_processing_s_per_sample']
    if not math.isfinite(cost) or cost < 0:
        raise ValueError('Stream processing cost must be finite and nonnegative')
    recovery = cfg['recovery']
    for key in ('baseline_s', 'overload_s', 'recovery_s', 'baseline_load', 'overload_load', 'tolerance_ms', 'sustain_s'):
        if not math.isfinite(recovery[key]) or recovery[key] <= 0:
            raise ValueError(f'Recovery {key} must be positive and finite')
    if not recovery['baseline_load'] < 1 < recovery['overload_load']:
        raise ValueError('Recovery needs baseline_load < 1 < overload_load')
    if min(recovery['baseline_s'], recovery['overload_s']) < 4*cfg['window_s']:
        raise ValueError('Recovery baseline and overload need at least four windows each')
    if recovery['recovery_s'] < recovery['sustain_s']+cfg['window_s']:
        raise ValueError('Recovery phase must allow the sustained recovery check')
    if not 0 < cfg['primary_low_max'] < cfg['primary_high_min']:
        raise ValueError('Primary comparison must have distinct low and high ranges')
    if not any(v<=cfg['primary_low_max'] for v in cfg['loads']) or not any(v>=cfg['primary_high_min'] for v in cfg['loads']):
        raise ValueError('Local loads must include both prespecified comparison ranges')
    template = plan[0]
    for repeat in range(1, cfg['repeats']+1):
        for count in range(1, len(sensors)+1):
            selected = [dict(s) for s in sensors[:count]]
            plan.append(dict(template, axis='streams', value=count, repeat=repeat, sensors=selected,
                             processing_s_per_sample=cost, load=sum(s['rate'] for s in selected)*cost))
        duration = sum(recovery[k] for k in ('baseline_s','overload_s','recovery_s'))
        plan.append(dict(template, axis='recovery', value=recovery['overload_load'], repeat=repeat,
                         duration_s=duration, expected=math.ceil(duration*cfg['rate_hz']),
                         processing_s_per_sample=recovery['baseline_load']/cfg['rate_hz'],
                         load=recovery['baseline_load'], recovery=dict(recovery)))
    rate_repeats = cfg.get('rate_repeats',3)
    if not isinstance(rate_repeats,int) or rate_repeats<1:
        raise ValueError('rate_repeats must be a positive integer')
    # This sweep has no imposed processing cost by default: it tests the actual
    # sender/receiver pipeline separately from the controlled-overload protocol.
    native_rates = document['local_tests'].get('rates_hz',[])
    mixed_rates = cfg.get('stream_eeg_rates_hz',[])
    rate_cost = cfg.get('rate_processing_s_per_sample',0.)
    if not math.isfinite(rate_cost) or rate_cost<0:
        raise ValueError('rate_processing_s_per_sample must be nonnegative and finite')
    for rates in (native_rates,mixed_rates):
        if len(set(rates))!=len(rates) or any(not math.isfinite(r) or r<=0 for r in rates):
            raise ValueError('Sweep rates must be unique, positive and finite')
    eeg = [s for s in sensors if s['kind'].upper()=='EEG']
    if mixed_rates and len(eeg)!=1:
        raise ValueError('Mixed-rate sweep requires exactly one EEG sensor')
    for repeat in range(1,rate_repeats+1):
        for rate in native_rates:
            plan.append(dict(template,axis='rate_native',value=rate,repeat=repeat,rate_hz=rate,
                             expected=math.ceil(rate*cfg['duration_s']),processing_s_per_sample=rate_cost,load=rate*rate_cost))
        for rate in mixed_rates:
            selected = [dict(s,rate=rate) if s['kind'].upper()=='EEG' else dict(s) for s in sensors]
            validate(dict(sensors=selected,duration=cfg['duration_s']))
            plan.append(dict(template,axis='streams_rate',value=rate,repeat=repeat,sensors=selected,
                             processing_s_per_sample=rate_cost,load=sum(s['rate'] for s in selected)*rate_cost))
    cfg['rates_hz'] = native_rates
    np.random.default_rng(cfg['seed']).shuffle(plan)
    for i, case in enumerate(plan, 1):
        case['case'] = f'L{i:03}'
    cfg['profile'] = 'local'
    return cfg, plan


def processing_cost(case, elapsed):
    """Wall-time schedule: remove overload while samples continue arriving."""
    recovery = case.get('recovery')
    if recovery:
        first = recovery['baseline_s']
        last = first+recovery['overload_s']
        load = recovery['overload_load'] if first <= elapsed < last else recovery['baseline_load']
        return load/case['rate_hz']
    return case['processing_s_per_sample']


def send(case, source_id, ready, done, release, directory, max_drain, common_start=None):
    from pylsl import StreamInfo, StreamOutlet, cf_double64, local_clock
    from transmission import signal
    info = StreamInfo(source_id, case.get('kind','EEG'), case['channels']+1, case['rate_hz'], cf_double64, source_id)
    # Vary only the inlet capacity; do not shrink the sender's buffer along with it.
    outlet = StreamOutlet(info, chunk_size=0, max_buffered=max(60, math.ceil(case['duration_s']+max_drain+2)))
    try:
        if not ready.wait(30):
            raise TimeoutError('Receiver did not become ready')
        start = common_start.value if common_start is not None else local_clock()+.15
        sent = np.empty(case['expected'])
        samples = start+np.arange(case['expected'])/case['rate_hz']
        for first in range(0, case['expected'], case['tx_chunk']):
            end = min(first+case['tx_chunk'], case['expected'])
            # Timestamp is acquisition time; a packet waits for its last sample.
            time.sleep(max(0., samples[end-1]-local_clock()))
            ids = np.arange(first, end)
            values = signal(case.get('kind','EEG'), ids, case['rate_hz'], case['channels'])
            stamp = local_clock()
            outlet.push_chunk(np.column_stack((values, ids)).tolist(), samples[first:end].tolist(), True)
            sent[first:end] = stamp
        time.sleep(max(0., start+case['duration_s']-local_clock()))
        np.savez(Path(directory)/'tx.npz', start=start, sample=samples, sent=sent)
        done.set()
        if not release.wait(max_drain+5):
            raise TimeoutError('Receiver did not release the transmitter')
    finally:
        del outlet


def record(case, max_drain):
    """One real LSL stream. Sleep models a known serial processing budget, not CPU work."""
    from pylsl import StreamInlet, resolve_byprop, local_clock
    context = mp.get_context('spawn')
    ready, done, release = (context.Event() for _ in range(3))
    common_start = context.Value('d', 0.)
    source_id = 'buffer_'+uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='lsl_buffer_') as temporary:
        tx = context.Process(target=send, args=(case, source_id, ready, done, release, temporary, max_drain, common_start))
        tx.start()
        inlet = None
        try:
            found = resolve_byprop('source_id', source_id, timeout=25)
            if not found:
                raise TimeoutError('LSL stream not found')
            inlet = StreamInlet(found[0], max_buflen=case['buffer_s'], recover=False, processing_flags=0)
            inlet.open_stream(timeout=10)
            correction = inlet.time_correction(timeout=10)
            common_start.value = local_clock()+.15
            ready.set()
            deadline = local_clock()+case['duration_s']+max_drain+2
            batches, ids, values, stamps, reads, finishes = [], [], [], [], [], []
            quiet_since = None
            drained = False
            end_known = False
            while local_clock() < deadline:
                if tx.exitcode is not None and tx.exitcode != 0:
                    raise RuntimeError(f'Transmitter failed: {tx.exitcode}')
                if done.is_set() and not end_known:
                    with np.load(Path(temporary)/'tx.npz') as metadata:
                        deadline = float(metadata['start'])+correction+case['duration_s']+max_drain
                    end_known = True
                before = inlet.samples_available()
                data, timestamps = inlet.pull_chunk(timeout=.01, max_samples=case['read_chunk'])
                read = local_clock()
                if not timestamps:
                    if done.is_set():
                        if quiet_since is None:
                            quiet_since = read
                        if read-quiet_since >= .3:
                            drained = True
                            break
                    continue
                quiet_since = None
                data = np.asarray(data)
                # The batch is not processed until this budget has elapsed.
                elapsed = read-common_start.value-correction
                work = len(timestamps)*processing_cost(case, elapsed)
                if work>0:
                    time.sleep(work)
                finish = local_clock()
                after = inlet.samples_available()
                count = len(timestamps)
                batches.append([read, finish, before, after, count])
                ids.append(data[:, -1].astype(np.int64))
                values.append(data[:, :-1].astype(np.float32))
                stamps.extend(timestamps)
                reads.extend([read]*count)
                finishes.extend([finish]*count)
            remaining = inlet.samples_available()
            # Completing the final batch at the deadline is a complete drain even
            # if there was no time left for the quiet-period observation.
            if ids and len(np.unique(np.concatenate(ids))) == case['expected']:
                drained = True
            if not done.is_set():
                raise TimeoutError('Transmitter did not finish within the observation limit')
            with np.load(Path(temporary)/'tx.npz') as archive:
                trace = {key: archive[key].copy() for key in archive.files}
            trace.update(ids=np.concatenate(ids) if ids else np.array([], dtype=np.int64),
                         values=np.concatenate(values) if values else np.empty((0, case['channels'])),
                         raw=np.asarray(stamps), read=np.asarray(reads), processed=np.asarray(finishes),
                         batches=np.asarray(batches).reshape(-1, 5), correction=np.array(correction),
                         drained=np.array(drained), remaining=np.array(remaining))
            return trace
        finally:
            if inlet is not None:
                inlet.close_stream()
            release.set()
            tx.join(timeout=5)
            if tx.is_alive():
                tx.terminate()
                tx.join()
            if tx.exitcode not in (None, 0):
                # Preserve the exception from setup/recording if there already is one.
                import sys
                if sys.exc_info()[0] is None:
                    raise RuntimeError(f'Transmitter exit status {tx.exitcode}')


def record_multiple(case, max_drain):
    """Independent senders share one serial, round-robin consumer."""
    from pylsl import StreamInlet, resolve_byprop, local_clock
    context = mp.get_context('spawn')
    ready, release = context.Event(), context.Event()
    common_start = context.Value('d',0.)
    specs, processes, inlets, states = {}, {}, {}, {}
    with tempfile.TemporaryDirectory(prefix='lsl_streams_') as temporary:
        try:
            for sensor in case['sensors']:
                name = sensor['name']
                spec = dict(case, rate_hz=sensor['rate'], channels=sensor['channels'], kind=sensor['kind'],
                            expected=math.ceil(sensor['rate']*case['duration_s']),
                            run_case=case['case'], case=case['case']+'_'+name, stream=name)
                # Every stream keeps its original signal and acquisition parameters.
                specs[name] = spec
                directory = Path(temporary)/name
                directory.mkdir()
                done = context.Event()
                source_id = 'incremental_'+uuid.uuid4().hex
                process = context.Process(target=send,args=(spec,source_id,ready,done,release,str(directory),max_drain,common_start))
                process.start()
                processes[name] = (process, done, source_id)
            for name, (_,_,source_id) in processes.items():
                found = resolve_byprop('source_id',source_id,timeout=25)
                if not found:
                    raise TimeoutError(f'Stream not found: {name}')
                inlet = StreamInlet(found[0],max_buflen=case['buffer_s'],recover=False,processing_flags=0)
                inlets[name] = inlet
                inlet.open_stream(timeout=10)
                offset = inlet.time_correction(timeout=10)
                states[name] = dict(ids=[],values=[],raw=[],read=[],processed=[],batches=[],
                                    correction=offset,quiet=None,drained=False)
            common_start.value = local_clock()+.15
            ready.set()
            deadline = common_start.value+case['duration_s']+max_drain
            while local_clock()<deadline and not all(st['drained'] for st in states.values()):
                received = False
                for name, inlet in inlets.items():
                    process, done, _ = processes[name]
                    if process.exitcode is not None and process.exitcode != 0:
                        raise RuntimeError(f'{name} transmitter failed: {process.exitcode}')
                    st = states[name]
                    if st['drained']:
                        continue
                    before = inlet.samples_available()
                    data,timestamps = inlet.pull_chunk(timeout=0.,max_samples=case['read_chunk'])
                    read = local_clock()
                    if not timestamps:
                        if done.is_set():
                            if st['quiet'] is None:
                                st['quiet'] = read
                            st['drained'] = read-st['quiet']>=.3
                        continue
                    received = True
                    st['quiet'] = None
                    data = np.asarray(data)
                    # One consumer: time spent on EMG also delays reading EEG and IMU.
                    work = len(timestamps)*case['processing_s_per_sample']
                    if work>0:
                        time.sleep(work)
                    finish = local_clock()
                    count = len(timestamps)
                    st['batches'].append([read,finish,before,inlet.samples_available(),count])
                    st['ids'].append(data[:,-1].astype(np.int64))
                    st['values'].append(data[:,:-1].astype(np.float32))
                    st['raw'].extend(timestamps)
                    st['read'].extend([read]*count)
                    st['processed'].extend([finish]*count)
                if not received:
                    time.sleep(.001)
            result = []
            for name,st in states.items():
                if not processes[name][1].is_set():
                    raise TimeoutError(f'Transmitter did not finish: {name}')
                with np.load(Path(temporary)/name/'tx.npz') as data:
                    trace = {key:data[key].copy() for key in data.files}
                ids = np.concatenate(st['ids']) if st['ids'] else np.array([],dtype=np.int64)
                drained = st['drained'] or len(np.unique(ids))==specs[name]['expected']
                trace.update(ids=ids, values=np.concatenate(st['values']) if st['values'] else np.empty((0,specs[name]['channels'])),
                             raw=np.asarray(st['raw']),read=np.asarray(st['read']),processed=np.asarray(st['processed']),
                             batches=np.asarray(st['batches']).reshape(-1,5),correction=np.array(st['correction']),
                             drained=np.array(drained),remaining=np.array(inlets[name].samples_available()))
                result.append((specs[name],trace))
            return result
        finally:
            for inlet in inlets.values():
                inlet.close_stream()
            release.set()
            for process,_,_ in processes.values():
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join()
            import sys
            if sys.exc_info()[0] is None and any(p.exitcode for p,_,_ in processes.values()):
                raise RuntimeError('An incremental-stream transmitter failed')


def fit_windows(times, values, duration, width):
    """One median per wall-time window, restricted to active production (no drain)."""
    points = []
    for start in np.arange(width, duration-width/2, width):
        selected = (times >= start) & (times < start+width) & (times < duration)
        if selected.any():
            points.append([start+width/2, float(np.median(values[selected]))])
    points = np.asarray(points).reshape(-1, 2)
    if len(points) < 4:
        return None, None, points
    slope = float(np.polyfit(points[:, 0], points[:, 1], 1)[0])
    return slope, float(points[-1, 1]-points[0, 1]), points


def summarize(case, trace, width):
    from transmission import signal
    start = float(trace['start'])
    ids = trace['ids']
    if not len(ids):
        raise RuntimeError('No samples received')
    if np.any((ids < 0) | (ids >= case['expected'])):
        raise ValueError('Invalid received sample index')
    unique = np.unique(ids)
    corrected = trace['raw']+float(trace['correction'])
    read_delay = trace['read']-corrected
    total_delay = trace['processed']-corrected
    batch_wait = trace['sent'][ids]-trace['sample'][ids]
    after_send = trace['read']-(trace['sent'][ids]+float(trace['correction']))
    batches = trace['batches']
    elapsed = batches[:, 1]-start-float(trace['correction'])
    queue_s = batches[:, 3]/case['rate_hz']
    q_slope, q_growth, queue_windows = fit_windows(elapsed, queue_s*1000, case['duration_s'], width)
    lag_slope, lag_growth, lag_windows = fit_windows(trace['processed']-start-float(trace['correction']),
                                                   total_delay*1000, case['duration_s'], width)
    # Unconsumed tail at a time limit is not labeled as dropped data.
    missing = case['expected']-len(unique)
    loss = missing if bool(trace['drained']) else None
    active = (elapsed >= width) & (elapsed <= case['duration_s'])
    max_queue = int(np.max(batches[:, 2:4]))
    growing = q_slope is not None and lag_slope is not None and q_slope > 10 and lag_growth > 50
    if not bool(trace['drained']):
        state = 'observation limit'
    elif loss:
        state = 'data loss observed'
    elif growing:
        state = 'growing queue'
    elif q_slope is None:
        state = 'insufficient windows'
    else:
        state = 'no sustained growth'
    expected_values = signal(case.get('kind','EEG'), ids, case['rate_hz'], case['channels'])
    row = dict(case, state=state, processed=len(ids), missing=missing, lost=loss,
               drained=bool(trace['drained']), remaining=int(trace['remaining']),
               duplicates=len(ids)-len(unique), reordered=int(np.sum(np.diff(ids)<0)),
               rmse=float(np.sqrt(np.mean((expected_values-trace['values'])**2))),
               read_p50_ms=float(np.median(read_delay)*1000), read_p99_ms=float(np.percentile(read_delay, 99)*1000),
               total_p50_ms=float(np.median(total_delay)*1000), total_p99_ms=float(np.percentile(total_delay, 99)*1000),
               total_max_ms=float(np.max(total_delay)*1000),
               batch_p50_ms=float(np.median(batch_wait)*1000),
               after_send_p50_ms=float(np.median(after_send)*1000),
               processing_p50_ms=float(np.median(trace['processed']-trace['read'])*1000),
               queue_peak_samples=max_queue, queue_peak_ms=1000*max_queue/case['rate_hz'],
               configured_capacity_samples=case['buffer_s']*case['rate_hz'],
               queue_slope_ms_s=q_slope, added_delay_ms=lag_growth, delay_slope_ms_s=lag_slope,
               queue_growth_ms=q_growth, drain_s=max(0., float(trace['processed'][-1])-start-float(trace['correction'])-case['duration_s']),
               active_processed=int(batches[active, 4].sum()),
               active_window_count=len(queue_windows), error='')
    trace['queue_windows'] = queue_windows
    trace['delay_windows'] = lag_windows
    return row


def interval(values):
    """Run-level bootstrap, never treating individual samples as independent runs."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return None, None, None
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None, None
    means = np.random.default_rng(194).choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def recovery_metrics(case, trace, width):
    recovery = case['recovery']
    first = recovery['baseline_s']
    last = first+recovery['overload_s']
    queue = trace['queue_windows']
    delay = trace['delay_windows']
    baseline_q = queue[queue[:,0]<first-width/2,1]
    baseline_d = delay[delay[:,0]<first-width/2,1]
    result = dict(recovered=False,overload_observed=False,recovery_s=None,recovery_confirmed_s=None,
                  baseline_queue_ms=None,baseline_delay_ms=None)
    qs,qg,_ = fit_windows(queue[:,0]-first,queue[:,1],recovery['overload_s'],width)
    ds,dg,_ = fit_windows(delay[:,0]-first,delay[:,1],recovery['overload_s'],width)
    result.update(queue_slope_ms_s=qs,queue_growth_ms=qg,delay_slope_ms_s=ds,added_delay_ms=dg)
    if not len(baseline_q) or not len(baseline_d):
        return result
    result['baseline_queue_ms'] = float(np.percentile(baseline_q,95))
    result['baseline_delay_ms'] = float(np.percentile(baseline_d,95))
    delay_by_time = dict(delay)
    result['overload_observed'] = any(
        stamp-width/2>=first and stamp+width/2<=last and
        depth>result['baseline_queue_ms']+recovery['tolerance_ms'] and
        stamp in delay_by_time and delay_by_time[stamp]>result['baseline_delay_ms']+recovery['tolerance_ms']
        for stamp,depth in queue)
    if not result['overload_observed']:
        return result
    required = math.ceil(recovery['sustain_s']/width)
    streak = []
    for stamp, depth in queue:
        if stamp-width/2 < last or stamp+width/2 > case['duration_s']:
            continue
        good = (depth<=result['baseline_queue_ms']+recovery['tolerance_ms'] and
                stamp in delay_by_time and
                delay_by_time[stamp]<=result['baseline_delay_ms']+recovery['tolerance_ms'])
        if not good or (streak and stamp-streak[-1]>width*1.01):
            streak=[]
        if good:
            streak.append(float(stamp))
        if len(streak)>=required:
            result.update(recovered=True,recovery_s=max(0.,streak[0]-width/2-last),
                          recovery_confirmed_s=stamp+width/2-last)
            break
    return result


def sign_test(differences):
    differences = np.asarray(differences)
    differences = differences[np.abs(differences)>1e-9]
    n = len(differences)
    if not n:
        return 1., 0
    k = min(int(np.sum(differences>0)), int(np.sum(differences<0)))
    return min(1., 2*sum(math.comb(n, i) for i in range(k+1))/2**n), n


def statistics(rows, cfg=None):
    rows = english_rows(rows)
    cfg = cfg or {}
    low_max, high_min = cfg.get('primary_low_max',.6), cfg.get('primary_high_min',1.5)
    groups = []
    for axis, value, stream in sorted({(r['axis'], r['value'], r.get('stream','')) for r in rows}):
        selected = [r for r in rows if (r['axis'], r['value'],r.get('stream','')) == (axis, value,stream)]
        valid = [r for r in selected if not r.get('error')]
        g = dict(axis=axis, value=value, stream=stream, n=len(valid), failures=len(selected)-len(valid),
                 growing=sum(r['state']=='growing queue' for r in valid),
                 drops=sum(r['lost'] is not None and r['lost']>0 for r in valid),
                 censored=sum(not r['drained'] for r in valid))
        for field in ('queue_slope_ms_s', 'delay_slope_ms_s', 'total_p99_ms', 'total_max_ms',
                      'queue_peak_ms', 'drain_s', 'added_delay_ms', 'read_p50_ms', 'batch_p50_ms',
                      'after_send_p50_ms', 'processing_p50_ms'):
            g[field] = interval([r[field] for r in valid])
        groups.append(g)
    # One prespecified primary test: within-repeat high load (>=1.5) minus low (<=0.6).
    # The independent unit is a complete run pair; sample-level p-values would be pseudoreplication.
    pairs = []
    for repeat in sorted({r['repeat'] for r in rows if r['axis']=='load'}):
        subset = [r for r in rows if r['axis']=='load' and r['repeat']==repeat]
        low = [r for r in subset if r['value']<=low_max]
        high = [r for r in subset if r['value']>=high_min]
        if low and high and all(not r.get('error') and r['queue_slope_ms_s'] is not None for r in low+high):
            pairs.append(float(np.mean([r['queue_slope_ms_s'] for r in high])-np.mean([r['queue_slope_ms_s'] for r in low])))
    p, n = sign_test(pairs)
    return groups, dict(effect=interval(pairs), p=p if n else None, pairs=len(pairs), non_ties=n)


def figure_data(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=130, bbox_inches='tight')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def plots(rows, traces, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    load_groups = [g for g in groups if g['axis']=='load']
    for ax, field, label in zip(axes.flat, ['queue_slope_ms_s', 'total_p99_ms', 'queue_peak_ms', 'drain_s'],
                               ['Queue growth (ms per second)', 'Total delay p99 (ms)',
                                'Peak queue, expressed as signal time (ms)', 'Time to process the remaining data (s)']):
        for g in load_groups:
            mean, lo, hi = g[field]
            if mean is None:
                continue
            ax.plot(g['value'], mean, 'o', color='#245cb0')
            if lo is not None:
                ax.vlines(g['value'], lo, hi, color='#245cb0', linewidth=2)
        valid = [g for g in load_groups if g[field][0] is not None]
        ax.plot([g['value'] for g in valid], [g[field][0] for g in valid], color='#245cb0', alpha=.5)
        ax.axvline(1, color='#a33', ls='--', label='nominal cost = available time')
        ax.set(xlabel='Nominal load: processing cost / interarrival time', ylabel=label)
        ax.grid(alpha=.2)
    fig.suptitle('When does the receiver fall behind? Mean and 95% run-level bootstrap CI')
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    overview = figure_data(fig)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    available = [r for r in rows if not r.get('error') and r['axis']=='load']
    for load in sorted({r['value'] for r in available}):
        selected = sorted([r for r in available if r['value']==load], key=lambda r:r['total_p99_ms'])
        r = selected[len(selected)//2]
        t = traces[r['case']]
        x = t['batches'][:, 1]-float(t['start'])-float(t['correction'])
        label = f'load {load:g} ({r["case"]})'
        axes[0].plot(x, t['batches'][:, 3]/r['rate_hz'], label=label)
        axes[1].plot(t['delay_windows'][:, 0], t['delay_windows'][:, 1], label=label)
        times = t['processed']-float(t['start'])-float(t['correction'])
        axes[2].plot(times, np.arange(1,len(times)+1)/r['expected'], label=label)
    if available:
        duration = available[0]['duration_s']
        xx = np.linspace(0, max(duration, axes[2].get_xlim()[1]), 100)
        axes[2].plot(xx, np.minimum(xx/duration, 1), '--', color='black', label='sent fraction, ideal')
        for ax in axes:
            ax.axvline(duration, color='#a33', ls='--', label='transmission ends' if ax is axes[0] else None)
    for ax, label in zip(axes, ['Queue waiting for processing (seconds of signal)',
                               'Total delay per window (ms)', 'Fraction of samples processed']):
        ax.set(ylabel=label)
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3, fontsize=8)
    if available:
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles[-1:], labels[-1:], fontsize=9)
    axes[2].set(xlabel='Wall time since acquisition start (s)')
    fig.suptitle('Measured timeline: one representative run per load (selected by median p99)')
    fig.tight_layout()
    timeline = figure_data(fig)
    plt.close(fig)

    if not any(g['axis'] in ('buffer','rate','chunk') for g in groups):
        return overview, timeline, None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, axis, title in zip(axes, ['buffer', 'rate', 'chunk'], ['Buffer size (s)', 'Input rate (Hz)', 'Samples per send']):
        selected = [g for g in groups if g['axis']==axis and g['total_p99_ms'][0] is not None]
        for g in selected:
            mean, lo, hi = g['total_p99_ms']
            ax.plot(g['value'], mean, 'o', color='#245cb0')
            if lo is not None:
                ax.vlines(g['value'], lo, hi, color='#245cb0')
            ax.annotate(f'{g["drops"]} loss / {g["censored"]} limited', (g['value'],mean),
                        xytext=(0,8), textcoords='offset points', ha='center', fontsize=7)
        ax.set(xlabel=title, ylabel='Total delay p99 (ms)')
        if axis in ('rate','chunk'):
            ax.set_xscale('log', base=2)
        ax.grid(alpha=.2)
    fig.suptitle('Extremes and intermediate values: other parameters are held fixed within each panel')
    fig.tight_layout()
    extremes = figure_data(fig)
    plt.close(fig)
    return overview, timeline, extremes


def saturation_plot(rows, traces):
    import matplotlib.pyplot as plt
    candidates = [r for r in rows if not r.get('error') and
                  (r['axis']=='buffer' or (r['axis']=='load' and r['value']==3))]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for i, size in enumerate(sorted({r['buffer_s'] for r in candidates})):
        color = f'C{i}'
        selected = sorted([r for r in candidates if r['buffer_s']==size], key=lambda r:r['total_p99_ms'])
        row = selected[len(selected)//2]
        t = traces[row['case']]
        origin = float(t['start'])+float(t['correction'])
        elapsed = t['batches'][:, 1]-origin
        capacity = size*row['rate_hz']
        axes[0].plot(elapsed, 100*t['batches'][:, 3]/capacity, color=color, label=f'buffer {size:g}s ({row["case"]})')
        # Index gaps identify missing earlier samples, not time waiting in a queue.
        if not row['duplicates'] and not row['reordered']:
            gaps = t['ids']+1-np.arange(1,len(t['ids'])+1)
            axes[1].step(t['processed']-origin, gaps, where='post', color=color, label=f'buffer {size:g}s')
            if row['lost'] is not None:
                axes[1].plot(t['processed'][-1]-origin,row['lost'],'o',color=color)
    axes[0].axhline(100, color='#a33', ls='--', label='configured capacity')
    axes[0].set(ylabel='Queue / configured capacity (%)')
    axes[1].set(ylabel='Missing samples identified by sequence index',xlabel='Wall time since acquisition start (s)')
    if candidates:
        for ax in axes:
            ax.axvline(candidates[0]['duration_s'],color='#a33',ls=':',label='end of transmission' if ax is axes[0] else None)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('Same overloaded consumer (load 3): larger buffers retain more delay; smaller buffers discard data')
    fig.tight_layout()
    result = figure_data(fig)
    plt.close(fig)
    return result


def retain_preview(case, trace, seconds):
    """Keep measured RX values for a short preview, then release full waveforms."""
    positions = np.flatnonzero(trace['raw']-float(trace['start'])<=seconds+1/case['rate_hz'])
    trace['preview_positions'] = positions
    trace['preview_values'] = trace.pop('values')[positions,:2].copy()


def waveform_preview(row, trace, cfg):
    from transmission import signal
    from analysis import temporal_reconstruction
    seconds = cfg.get('snapshot_seconds',2.)
    positions = trace['preview_positions']
    channels = trace['preview_values'].shape[1]
    ids = np.flatnonzero(trace['sample']-float(trace['start'])<=seconds+1/row['rate_hz'])
    tx = dict(ids=ids,sample=trace['sample'][ids],values=signal(row.get('kind','EEG'),ids,row['rate_hz'],channels))
    rx = dict(ids=trace['ids'][positions],values=trace['preview_values'],
              raw=trace['raw'][positions],corrected=trace['raw'][positions]+float(trace['correction']),
              arrival=trace['read'][positions])
    return temporal_reconstruction(tx,rx,row['rate_hz'],float(trace['start']),float(trace['correction']),
        grid_rate=cfg.get('reconstruction_rate_hz',16384),until=seconds,
        max_age_s=cfg.get('delivery_max_age_s',.25),playback_delay_s=cfg.get('playback_delay_s',.25))[0]


def sampling_sections(rows, traces, groups, cfg):
    import matplotlib.pyplot as plt
    sections = []
    for axis,title in [('rate_native','EEG alone: measured rate sweep up to 16384 Hz'),
                       ('streams_rate','Three simultaneous streams: vary EEG and observe the others')]:
        selected = [g for g in groups if g['axis']==axis and g['n']]
        if not selected:
            continue
        fig,axes = plt.subplots(3,1,figsize=(13,9),sharex=True)
        for i,name in enumerate(sorted({g['stream'] for g in selected})):
            color = f'C{i}'
            stream_groups = [g for g in selected if g['stream']==name]
            for ax,field in zip(axes[:2],['total_p99_ms','queue_slope_ms_s']):
                usable = [g for g in stream_groups if g[field][0] is not None]
                ax.plot([g['value'] for g in usable],[g[field][0] for g in usable],'o-',color=color,label=name.upper() or 'EEG')
                for g in usable:
                    _,lo,hi = g[field]
                    if lo is not None:
                        ax.vlines(g['value'],lo,hi,color=color)
            for g in stream_groups:
                runs = [r for r in rows if r['axis']==axis and r['value']==g['value'] and r.get('stream','')==name and not r.get('error')]
                loss = [100*r['lost']/r['expected'] for r in runs if r['lost'] is not None]
                if loss:
                    axes[2].plot(g['value'],np.mean(loss),'o',color=color,label=name.upper() or 'EEG')
                if g['censored']:
                    axes[0].annotate(f'{g["censored"]} incomplete run(s)',(g['value'],g['total_p99_ms'][0]),fontsize=7)
        axes[0].set_ylabel('Total delay p99 (ms), mean and 95% CI')
        axes[1].set_ylabel('Queue growth (ms/s), mean and 95% CI')
        axes[2].set(ylabel='Mean loss (%) in fully drained runs',xlabel='EEG rate (Hz); other streams retain their rates')
        for ax in axes:
            ax.set_xscale('log',base=2)
            ax.set_xticks(sorted({g['value'] for g in selected}),[f'{v:g}' for v in sorted({g['value'] for g in selected})])
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=9)
        fig.suptitle(title)
        fig.tight_layout()
        image = figure_data(fig)
        plt.close(fig)
        sections.append(f'<section><h2>{title}</h2><p>Imposed processing cost per sample: '
                        f'{cfg.get("rate_processing_s_per_sample",0)*1e6:g} µs. Zero removes the artificial wait, but reading, '
                        'conversion, generation and transport still consume wall time. In simultaneous-stream curves, '
                        'only the EEG rate changes; EMG and IMU retain their rates and channels. Streams are not pooled as independent repetitions. '
                        'For incomplete runs, p99 covers processed samples only; unknown total loss is never replaced with zero.</p>'
                        f'<img src="{image}" alt="Rate sweep and effect on each stream"></section>')
    candidates = [r for r in rows if not r.get('error') and 'preview_values' in traces[r['case']]]
    if not candidates:
        return ''.join(sections)+'<section><p>This older recording contains timing data but no RX values for the new waveform preview. A new acquisition is required; received values are never synthesized.</p></section>'
    mixed = [r for r in candidates if r['axis'] in ('streams_rate','streams')]
    if mixed:
        axis = 'streams_rate' if any(r['axis']=='streams_rate' for r in mixed) else 'streams'
        maximum = max(r['value'] for r in mixed if r['axis']==axis)
        runs = sorted({r['run_case'] for r in mixed if r['axis']==axis and r['value']==maximum},
                      key=lambda run:np.mean([r['total_p99_ms'] for r in mixed if r['run_case']==run]))
        run = runs[len(runs)//2]
        preview_rows = [r for r in mixed if r['run_case']==run]
    else:
        preview_rows = [min(candidates,key=lambda r:r['total_p99_ms'])]
    fig,axes = plt.subplots(len(preview_rows),1,figsize=(13,3*len(preview_rows)),sharex=True,squeeze=False)
    for ax,row in zip(axes[:,0],preview_rows):
        reconstructed = waveform_preview(row,traces[row['case']],cfg)
        grid = reconstructed['time']
        ax.plot(grid,reconstructed['tx'][:,0],label='TX: acquisition',lw=1)
        ax.plot(grid,reconstructed['rx_acquisition'][:,0],'--',label='RX: acquisition (offline interpolation)',lw=.9)
        ax.step(grid,reconstructed['rx_delivery'][:,0],where='post',color='.65',lw=.8,label='Last delivered value, including stale data')
        ax.step(grid,reconstructed['rx_fresh'][:,0],where='post',color='tab:green',lw=.9,label='Available after reading, within age limit')
        ax.step(grid,reconstructed['rx_playback'][:,0],where='post',color='tab:purple',lw=.9,
                label=f'All-sample playback, delay {cfg.get("playback_delay_s",.25):g}s')
        ax.set_ylabel(f'{row.get("stream","eeg").upper()} / {row["rate_hz"]:g} Hz\nChannel 1')
        ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,.96))
    axes[-1,0].set_xlabel('Shared time axis and grid; live values update only after an actual read')
    fig.suptitle(f'Reconstruction on a common grid of {cfg.get("reconstruction_rate_hz",16384):g} Hz — measured RX values')
    fig.tight_layout(rect=(0,0,1,.86 if len(preview_rows)>1 else .7))
    image = figure_data(fig)
    plt.close(fig)
    sections.append(f'<section><h2>Reconstruction: held values, availability and sample age</h2><p>All streams use '
                    f'the same grid of {cfg.get("reconstruction_rate_hz",16384):g} Hz. Native acquisition rates are preserved. '
                    'Samples within a batch share a read time: the causal curve shows the newest available sample, '
                    'held until the next read, without inventing arrival times within the batch. '
                    f'The green curve accepts only samples whose age since acquisition is ≤{1000*cfg.get("delivery_max_age_s",.25):g} ms; '
                    'gray shows the last value even when stale. The offline curve may interpolate using later samples '
                    'and does not represent real-time availability. Missing acquisition samples are not interpolated. '
                    f'Purple playback preserves all samples at their acquisition times shifted by the explicitly configured delay of '
                    f'{cfg.get("playback_delay_s",.25):g} s: a batch that arrives after its playback deadline produces a gap. '
                    'It simulates a playback buffer with native-rate sample holding; it is neither ideal band-limited reconstruction '
                    'nor transport-delay removal. The instantaneous latest-value view may omit intermediate batch samples from the display only. '
                    'This is a short channel-1 preview, not every waveform from the run.</p>'
                    f'<img src="{image}" alt="Causal held-value reconstruction without stacking every batch sample at one read time"></section>')
    return ''.join(sections)


def local_sections(rows, traces, groups, cfg):
    """Show each stream separately; never pool correlated streams as repetitions."""
    import matplotlib.pyplot as plt
    sections = []
    valid = [r for r in rows if not r.get('error')]
    incremental = [r for r in valid if r['axis']=='streams']
    if incremental:
        fig,axes = plt.subplots(2,2,figsize=(13,8))
        names = [s['name'] for s in cfg['stream_sensors']]
        last_stage = max(r['value'] for r in incremental)
        runs = sorted({r['run_case'] for r in incremental if r['value']==last_stage},
                      key=lambda run:np.mean([r['total_p99_ms'] for r in incremental if r['run_case']==run]))
        central = runs[len(runs)//2]
        for i,name in enumerate(names):
            color = f'C{i}'
            selected = [g for g in groups if g['axis']=='streams' and g['stream']==name]
            for ax,field in [(axes[0,0],'total_p99_ms'),(axes[1,0],'queue_slope_ms_s')]:
                usable = [g for g in selected if g[field][0] is not None]
                ax.plot([g['value'] for g in usable],[g[field][0] for g in usable],'o-',color=color,label=name.upper())
                for g in usable:
                    mean,lo,hi = g[field]
                    if lo is not None:
                        ax.vlines(g['value'],lo,hi,color=color)
            row = next((r for r in incremental if r['run_case']==central and r['stream']==name),None)
            if row:
                t = traces[row['case']]
                origin = float(t['start'])+float(t['correction'])
                axes[0,1].plot(t['batches'][:,1]-origin,1000*t['batches'][:,3]/row['rate_hz'],color=color,label=name.upper())
                axes[1,1].plot(t['delay_windows'][:,0],t['delay_windows'][:,1],color=color,label=name.upper())
        for ax in axes[:,0]:
            ax.set_xticks(range(1,len(names)+1),['+'.join(n.upper() for n in names[:count]) for count in range(1,len(names)+1)])
            ax.set_xlabel('Active streams sharing one consumer')
        axes[0,0].set_ylabel('Total delay p99 (ms), mean and 95% CI')
        axes[1,0].set_ylabel('Queue growth (ms/s), mean and 95% CI')
        axes[0,1].set_ylabel('Waiting queue (milliseconds of signal)')
        axes[1,1].set_ylabel('Total delay per window (ms)')
        for ax in axes[:,1]:
            ax.set_xlabel(f'Wall time (s), run {central}')
            ax.axvline(cfg['duration_s'],ls=':',color='#a33')
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('Adding streams: existing streams retain their rates, channels and per-sample processing cost')
        fig.tight_layout()
        image = figure_data(fig)
        plt.close(fig)
        parameters = '; '.join(f'{s["name"].upper()}: {s["rate"]:g} Hz / {s["channels"]} channels' for s in cfg['stream_sensors'])
        sections.append(f'<section><h2>Does adding streams affect those already active?</h2><p>{html.escape(parameters)}. '
                        f'One consumer shares its time across streams; controlled cost of '
                        f'{cfg["stream_processing_s_per_sample"]*1e6:g} µs per sample, with existing parameters held fixed. '
                        'Each stream has its own curve and confidence interval. Streams from one run are not independent repetitions. '
                        'Time plots use streams from the same representative run, without combining different runs.</p>'
                        f'<img src="{image}" alt="Effect of adding EEG, EMG and IMU on latency and queues"></section>')
    recovery = [r for r in valid if r['axis']=='recovery']
    if recovery:
        selected = sorted(recovery,key=lambda r: r['recovery_s'] if r['recovery_s'] is not None else float('inf'))
        row = selected[len(selected)//2]
        t = traces[row['case']]
        schedule = row['recovery']
        first = schedule['baseline_s']
        last = first+schedule['overload_s']
        origin = float(t['start'])+float(t['correction'])
        fig,axes = plt.subplots(3,1,figsize=(13,9),sharex=True)
        axes[0].plot(t['batches'][:,1]-origin,1000*t['batches'][:,3]/row['rate_hz'],label='measured queue')
        axes[1].plot(t['delay_windows'][:,0],t['delay_windows'][:,1],label='total delay per window')
        if row['baseline_queue_ms'] is not None:
            axes[0].axhline(row['baseline_queue_ms']+schedule['tolerance_ms'],ls='--',color='green',label='baseline reference + tolerance')
            axes[1].axhline(row['baseline_delay_ms']+schedule['tolerance_ms'],ls='--',color='green',label='baseline reference + tolerance')
        axes[2].plot(t['processed']-origin,np.arange(1,len(t['ids'])+1)/row['expected'],label='processed fraction')
        xx = np.linspace(0,max(row['duration_s'],axes[2].get_xlim()[1]),100)
        axes[2].plot(xx,np.minimum(xx/row['duration_s'],1),'k--',label='sent fraction, ideal')
        for ax in axes:
            ax.axvspan(first,last,color='orange',alpha=.18,label='temporary overload')
            ax.axvline(last,color='#a33',ls=':',label='return to normal processing cost')
            ax.axvline(row['duration_s'],color='black',ls=':',label='transmission ends')
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Queue (milliseconds of signal)')
        axes[1].set_ylabel('Total delay (ms)')
        axes[2].set(ylabel='Processed fraction',xlabel='Wall time since start (s)')
        axes[0].legend(ncol=2,fontsize=8)
        axes[2].legend(fontsize=8,ncol=2)
        fig.suptitle(f'Recovery during continuous transmission — representative run {row["case"]}')
        fig.tight_layout()
        image = figure_data(fig)
        plt.close(fig)
        cells = []
        for r in recovery:
            cells.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in
                         [r['case'],r['state'],fmt(r['recovery_s'],2),fmt(r['recovery_confirmed_s'],2),str(r['lost'])])+'</tr>')
        recovered = sum(r['recovered'] and r['lost']==0 and r['drained'] for r in recovery)
        sections.append(f'<section><h2>Does the queue recover while transmission continues?</h2><p>{recovered}/{len(recovery)} '
                        'runs demonstrated recovery with a complete drain and no loss. Queue depth and delay must first exceed their baseline ranges during overload. The baseline reference is the 95th percentile '
                        f'of baseline-window medians. Queue depth and delay must both return to the reference + '
                        f'{schedule["tolerance_ms"]:g} ms and stay there for {schedule["sustain_s"]:g} s across consecutive windows, using window medians. '
                        'Recovery must occur while transmission continues. Return time is the beginning of the first qualifying sequence; '
                        'confirmation occurs only at its end. Time resolution is limited by the window width.</p>'
                        f'<img src="{image}" alt="Temporary overload and recovery during transmission">'
                        '<table><tr><th>Run</th><th>Diagnosis</th><th>Return after overload removal (s)</th>'
                        f'<th>Confirmation (s)</th><th>Loss</th></tr>{"".join(cells)}</table></section>')
    return ''.join(sections)


def fmt(value, digits=1):
    return '—' if value is None else f'{value:.{digits}f}'


def report(root, rows, traces, cfg):
    rows = english_rows(rows)
    groups, primary = statistics(rows,cfg)
    images = plots(rows, traces, groups)
    saturation = saturation_plot(rows,traces) if any(not r.get('error') and
                  (r['axis']=='buffer' or (r['axis']=='load' and r['value']==3)) for r in rows) else None
    extras = local_sections(rows,traces,groups,cfg)+sampling_sections(rows,traces,groups,cfg)
    valid = [r for r in rows if not r.get('error')]
    load = [g for g in groups if g['axis']=='load' and g['n']]
    rising = [g['value'] for g in load if g['queue_slope_ms_s'][1] is not None and
              g['queue_slope_ms_s'][1]>10 and g['growing']==g['n']]
    stable = [g['value'] for g in load if g['queue_slope_ms_s'][2] is not None and
              g['queue_slope_ms_s'][2]<=10 and not g['drops'] and not g['censored']]
    upper = min(rising) if rising else None
    lower = max((v for v in stable if upper is None or v<upper), default=None)
    boundary = (f'The observed transition lies between loads {lower:g} and {upper:g}.' if lower is not None and upper is not None
                else 'The data do not bound the transition with sufficient confidence.')
    failures = sum(bool(r.get('error')) for r in rows)
    loss_runs = sum(r['lost'] is not None and r['lost']>0 for r in valid)
    capped = sum(not r['drained'] for r in valid)
    quick = []
    for g in load:
        quick.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in [
            f'{g["value"]:g}', fmt(g['queue_slope_ms_s'][0]), fmt(g['total_p99_ms'][0]),
            fmt(g['drain_s'][0],2), f'{g["growing"]}/{g["n"]}', str(g['censored'])])+'</tr>')
    effect, lo, hi = primary['effect']
    stats_text = (f'Prespecified primary comparison: loads ≥{cfg.get("primary_high_min",1.5):g} '
                  f'versus ≤{cfg.get("primary_low_max",.6):g}, paired by repetition. '
                  f'Difference in queue growth: {fmt(effect)} ms/s '
                  f'(95% bootstrap CI: {fmt(lo)} to {fmt(hi)}); '
                  f'two-sided exact sign test p={fmt(primary["p"],5)}, '
                  f'{primary["pairs"]} complete pairs, {primary["non_ties"]} non-tied. '
                  'The statistical unit is the run, never an individual sample. Exploratory intervals based on few repetitions are imprecise; '
                  'other contrasts and the boundary are descriptive, without multiple significance claims.')
    table = []
    for r in rows:
        if r.get('error'):
            table.append(f'<tr><td>{r["case"]}</td><td colspan="12">FAILURE: {html.escape(r["error"])}</td></tr>')
            continue
        cells = [r['case'], f'{r["axis"]}={r["value"]:g} / rep {r["repeat"]}', r['state'],
                 fmt(r['load'],2), fmt(r['queue_slope_ms_s']), fmt(r['added_delay_ms']),
                 fmt(r['total_p99_ms']), fmt(r['total_max_ms']), fmt(r['queue_peak_ms']),
                 fmt(r['drain_s'],2)+(' (incomplete)' if not r['drained'] else ''),
                 str(r['lost']) if r['lost'] is not None else f'unknown ({r["missing"]} unprocessed)',
                 f'{r["processed"]}/{r["expected"]}', fmt(r['rmse'],3)]
        table.append('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in cells)+'</tr>')
    components = []
    for g in load:
        components.append('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in [
            f'{g["value"]:g}', str(g['n']), fmt(g['batch_p50_ms'][0]), fmt(g['after_send_p50_ms'][0]),
            fmt(g['processing_p50_ms'][0]), fmt(g['total_p99_ms'][0])])+'</tr>')
    headers = ['Case','Parameter / repetition','Diagnosis','Load','Queue ms/s','Added delay ms',
               'Total p99 ms','Total maximum ms','Peak queue ms','Drain time s','Lost','Processed/sent','RMSE']
    attempted = len({r.get('run_case',r['case']) for r in rows})
    durations = ', '.join(f'{v:g}' for v in sorted({r['duration_s'] for r in rows}))
    machine = html.escape(json.dumps(cfg.get('machine',{}),ensure_ascii=False))
    extremes_section = f'''<section><h2>3. Buffer, rate and chunk size: extremes and intermediate values</h2><p>Buffer: fixed processing load 3. Rate: fixed processing cost of
{cfg['rate_sweep_processing_s_per_sample']*1e6:g} µs/sample and a 1 s buffer. Chunk: load 0.25 and buffer of {cfg['buffer_s']:g} s.
These panels vary different parameters and should be interpreted separately.
At low rates with a fixed chunk size, packets take longer to form; at high rates with fixed processing cost,
the consumer may fall behind. Latency therefore need not vary monotonically with input rate.
Queue plots and the delay breakdown distinguish these causes.</p><img src="{images[2]}" alt="Buffer, input-rate and chunk-size sweeps"></section>''' if images[2] else ''
    saturation_section = f'''<section><h2>What happens when the buffer fills up?</h2><p>Consumer processing cost is identical across these curves.
100% denotes configured receiver capacity; actual peaks depend on the implementation and queue-count sampling.
Missing samples appear as sequence-index gaps. A queue that flattens near capacity,
while sequence gaps increase, indicates saturation with discard, not healthy stability.
The final point includes total loss when draining completed.</p>
<p>A 1 s buffer stores one second of signal; it does not guarantee a one-second latency limit:
a consumer at load 3 needs about three seconds to process that content.</p>
<img src="{saturation}" alt="Buffer occupancy percentage and loss with the same overloaded consumer"></section>''' if saturation else ''
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>LSL — when the queue grows</title>
<style>body{{font:16px system-ui;color:#17233a;max-width:1250px;margin:32px auto;padding:0 20px;background:#f6f8fc}}
section,.card{{background:white;padding:24px;border-radius:12px;margin:18px 0}}h1{{font-size:32px}}h2{{font-size:23px}}
img{{max-width:100%;height:auto}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{flex:1;min-width:190px}}
strong.big{{font-size:28px;display:block}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{padding:9px;border-bottom:1px solid #dce3ef;text-align:left}}th{{background:#e8eef8;position:sticky;top:0}}
.scroll{{overflow:auto}}p,li{{line-height:1.6}}code{{background:#e8eef8;padding:2px 5px}}</style>
<h1>LSL: does the receiver keep up or let the queue grow?</h1>
<p>Controlled per-sample cost; real transmission between LSL processes. Acquisition duration: {durations} s, depending on the scenario.
Randomized order, seed {cfg['seed']}. {attempted} runs attempted; {len(rows)} stream measurements; {failures} technical failures.</p>
<p>Acquisition machine: {machine}. Python and operating system are recorded during acquisition; reanalysis on another machine does not change them.</p>
<div class="cards"><div class="card"><strong class="big">{sum(r['state']=='growing queue' for r in valid)}</strong>streams with growing queues</div>
<div class="card"><strong class="big">{loss_runs}</strong>streams with confirmed data loss</div>
<div class="card"><strong class="big">{capped}</strong>streams stopped with pending data</div></div>
<section><h2>1. The direct answer</h2><p><b>{boundary}</b> This range applies to this machine, duration and configuration;
it is not a universal capacity limit. Load 1 means the processing budget already consumes the entire interarrival interval;
reading, conversion and scheduling take additional time.</p>
<p>A growing queue means data arrive faster than they are processed. A larger buffer retains more data,
but does not speed up the consumer. A smaller buffer may bound latency by discarding samples: low delay with loss is not success.</p>
<div class="scroll"><table><tr><th>Load</th><th>Queue growth (ms/s)</th><th>Total p99 (ms)</th>
<th>Drain time (s)</th><th>Runs with growing queues</th><th>Incomplete</th></tr>{''.join(quick)}</table></div>
<p>Values above are run-level means. For incomplete runs, drain time is only the observed time, not the time required to finish.</p>
<img src="{images[0]}" alt="Load versus queue growth, total delay, queue depth and drain time"></section>
<section><h2>2. Watch the backlog build up</h2><p>A rising queue during transmission means backlog is accumulating.
A decline after the red line means pending data are being processed. The processed fraction trails the sent fraction when the consumer falls behind.
Each line is an actual run, not an average that hides peaks.</p><img src="{images[1]}" alt="Queue, delay and progress over time"></section>
{extremes_section}{saturation_section}{extras}
<section><h2>4. Where does total delay come from?</h2><p><b>Total = transmitter chunk formation and scheduling + transport/queue until read + batch processing.</b>
For each sample, total delay runs from acquisition timestamp to processing completion. p99 is the delay below which 99% of processed samples fall;
maximum is the worst observed case. Components below are run-level means of per-run medians;
they must not be added to obtain p99. If observation ends with pending data,
p99 and maximum cover processed samples only; they do not summarize the unprocessed tail.</p><div class="scroll"><table><tr><th>Load</th><th>n</th><th>TX chunk/scheduling ms</th>
<th>Transport/queue ms</th><th>Batch processing ms</th><th>Total p99 ms</th></tr>{''.join(components)}</table></div>
<p>Added delay is the difference between the last and first valid production windows, not the sum of all sample delays.
Queue ms/s measures how much additional signal time accumulates per wall-clock second. Drain time runs from the end of acquisition to the last processing completion;
when incomplete, it is only the observed time, not the time needed to recover.</p></section>
<section><h2>Nyquist and streams with different rates</h2><p>Acquisition rate must exceed twice the highest signal frequency,
with antialias filtering and a practical margin. This differs from the batch-reading frequency and display grid rate.
A fast shared grid does not create new measurements in a slower stream. Offline interpolation cannot undo aliasing or prove real-time availability.
Ideal rectangular marker pulses are not band-limited analog signals.</p>
<p>A shared consumer can delay EEG while processing EMG. In an application, separate reading from processing using bounded queues,
preserve LSL timestamps and define an explicit sample-age/discard policy. Independent readers cannot fix insufficient consumer capacity;
they only move the backlog elsewhere. This experiment keeps the shared consumer to measure competition between streams.</p></section>
<section><h2>5. Statistical evidence and criteria</h2><p>{stats_text}</p>
<p>Medians in windows of {cfg['window_s']:g} s, excluding the first window and the entire drain phase.
Sustained growth requires at least four windows, queue slope &gt;10 ms/s and total-delay growth &gt;50 ms.
For recovery cases, slope and delay growth are calculated during overload only;
the return diagnosis uses the sustained criterion shown in the recovery plot.
The boundary requires growth in every repetition and a slope CI lower bound above 10 ms/s;
the stable bound requires a CI upper bound ≤10 ms/s, with no loss or observation cutoffs.</p>
<p>Queue depth is the available-sample count reported by <code>samples_available()</code>, before reading and after processing.
Sampled queue peaks may underestimate peaks between reads. This metric measures neither every internal buffer in bytes nor clock drift.
Controlled processing cost is imposed by serial waiting; the experiment measures the consequence of insufficient capacity, not EEG-algorithm performance.
Drain observation limit: {cfg['max_drain_s']:g} s, allowing an already-started batch to finish. If data remain at the limit, total loss is unknown.
Latency uses the initial LSL clock correction and ends in the Python process, not at a hardware timestamp.
Technical failures are never included in averages as zero.</p></section>
<section><h2>6. All runs</h2><p>Files: <code>metrics.csv</code> (measurements), <code>timeline.csv</code> (queue per read),
<code>recordings.npz</code> (traces for reanalysis). This HTML embeds all plots, without separate figure folders.</p>
<div class="scroll"><table><tr>{''.join('<th>'+h+'</th>' for h in headers)}</tr>{''.join(table)}</table></div></section></html>'''
    (root/'summary.html').write_text(text, encoding='utf-8')
    return groups, primary, boundary


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_timeline(root, traces):
    timeline = []
    for case_id, t in traces.items():
        for read, finish, before, after, count in t['batches']:
            timeline.append(dict(case=case_id, read_s=read-float(t['start'])-float(t['correction']),
                                 processed_s=finish-float(t['start'])-float(t['correction']),
                                 queued_before=int(before), queued_after=int(after), batch_samples=int(count)))
    write_csv(root/'timeline.csv', timeline)


def reanalyze(recording, output=None):
    """Regenerate only from the saved run's configuration and timing, without LSL."""
    recording = Path(recording)
    with np.load(recording, allow_pickle=False) as data:
        cfg = json.loads(str(data['config_json']))
        rows = english_rows(json.loads(str(data['metrics_json'])))
        traces = {}
        for row in rows:
            if row.get('error'):
                continue
            prefix = row['case']+'__'
            traces[row['case']] = {key[len(prefix):]:data[key].copy() for key in data.files if key.startswith(prefix)}
    root = Path(output) if output else recording.parent
    if output:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError('Reanalysis output must be empty')
        import shutil
        shutil.copy2(recording, root/'recordings.npz')
    write_csv(root/'metrics.csv', rows)
    write_timeline(root, traces)
    _, primary, boundary = report(root, rows, traces, cfg)
    print(boundary)
    print('Primary comparison:', primary)
    print('Report:',root/'summary.html')


def experiment(config_path, output=None, plan_only=False, local=False):
    from helpers import prepare_results
    document = json.loads(Path(config_path).read_text())
    mode = 'local' if local else document.get('active_suite','quick')
    if mode not in ('local','quick'):
        raise ValueError('active_suite must be local or quick')
    if mode=='local':
        cfg,plan = local_plan(document)
    else:
        cfg = document['buffer_tests']
        plan = build_plan(cfg)
    for key,default in [('snapshot_seconds',2.),('reconstruction_rate_hz',16384),('delivery_max_age_s',.25)]:
        cfg.setdefault(key,document['acquisition'].get(key,default))
        if not math.isfinite(cfg[key]) or cfg[key]<=0:
            raise ValueError(f'{key} must be positive and finite')
    cfg.setdefault('playback_delay_s',document['acquisition'].get('playback_delay_s',.25))
    if not math.isfinite(cfg['playback_delay_s']) or cfg['playback_delay_s']<0:
        raise ValueError('playback_delay_s must be finite and nonnegative')
    print(f'{len(plan)} runs; {sum(c["duration_s"] for c in plan)/60:.1f} minutes of scheduled acquisition, '
          f'plus startup and up to {cfg["max_drain_s"]:g}s of draining per run.', flush=True)
    if plan_only:
        for r in plan:
            print(f'{r["case"]} {r["axis"]}={r["value"]:g} rep={r["repeat"]} load={r["load"]:g}', flush=True)
        # A plan preview never moves or rewrites previous experiment results.
        if output:
            root = Path(output)
            root.mkdir(parents=True, exist_ok=True)
            if any(root.iterdir()):
                raise ValueError('Plan output must be empty')
            write_csv(root/'plan.csv', plan)
        return
    root = prepare_results(output or Path(__file__).with_name('results'))
    cfg['machine'] = dict(system=platform.system(),release=platform.release(),architecture=platform.machine(),
                          python=platform.python_version(),cpu_count=__import__('os').cpu_count())
    cfg['started_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    rows, traces = [], {}
    archive = dict(config_json=np.array(json.dumps(cfg)), plan_json=np.array(json.dumps(plan)))
    for i, case in enumerate(plan, 1):
        print(f'[{i}/{len(plan)}] {case["case"]} {case["axis"]}={case["value"]:g} rep {case["repeat"]}', flush=True)
        try:
            recordings = record_multiple(case,cfg['max_drain_s']) if case.get('sensors') else [(case,record(case,cfg['max_drain_s']))]
            for spec,trace in recordings:
                row = summarize(spec, trace, cfg['window_s'])
                row.pop('sensors',None)
                if case['axis']=='recovery':
                    row.update(recovery_metrics(case,trace,cfg['window_s']))
                    if not row['overload_observed']:
                        row['state'] = 'overload not demonstrated'
                    elif row['recovered']:
                        row['state'] = 'queue recovered; drain incomplete' if not row['drained'] else ('recovered with data loss' if row['lost'] else 'recovered during transmission')
                    else:
                        row['state'] = 'recovery not demonstrated'
                retain_preview(spec,trace,cfg['snapshot_seconds'])
                traces[spec['case']] = trace
                # Preserve short measured previews, not all channels of every run.
                archive.update({spec['case']+'__'+key:value for key,value in trace.items() if key!='values'})
                print(f'  {spec["case"]}: {row["state"]}; queue {fmt(row["queue_slope_ms_s"])} ms/s; '
                      f'total p99 {row["total_p99_ms"]:.1f} ms; lost {row["lost"]}', flush=True)
                rows.append(row)
        except Exception as error:
            row = dict(case, state='technical failure', error=f'{type(error).__name__}: {error}')
            print('  FAILURE: '+row['error'], flush=True)
            rows.append(row)
        # Persist after every case so an interrupted long experiment remains inspectable.
        write_csv(root/'metrics.csv', rows)
        archive['metrics_json'] = np.array(json.dumps(rows, ensure_ascii=False, allow_nan=False))
        # An interrupted write must not destroy the last completed checkpoint.
        checkpoint = root/'.recordings.tmp.npz'
        np.savez_compressed(checkpoint, **archive)
        checkpoint.replace(root/'recordings.npz')
    write_timeline(root, traces)
    groups, primary, boundary = report(root, rows, traces, cfg)
    print(boundary, flush=True)
    print('Primary comparison:', primary, flush=True)
    print('Report:', root/'summary.html', flush=True)
    if any(row.get('error') for row in rows):
        raise RuntimeError('Technical failures occurred; inspect metrics.csv and summary.html')
