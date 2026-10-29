import csv
import json
from pathlib import Path

import numpy as np

from .reconstruction import temporal_reconstruction, windows, diagnose


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
        latency_change_ms=float((latency[-1]-latency[0])*1000))
    return metrics, dict(common=common, it=it, ir=ir, elapsed=elapsed,
                         latency=latency, queue=queue, timing_tx=timing_tx,
                         timing_rx=timing_rx, timestamp_error=timestamp_error,
                         read_error=read_error)


def plot_diagnostics(out, name, sensor, cfg, tx, rx, data, reconstructed, metrics, trends):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 1, figsize=(12, 18), constrained_layout=True)
    t = reconstructed['time']
    # Zoom where delivery is slowest, rather than hiding later problems behind a
    # first-three-seconds-only view. All series use the SAME absolute origin.
    worst = int(np.argmax(data['latency']))
    origin = tx['sample'][0]-t[0]
    center = tx['sample'][data['it'][worst]]-origin
    left = max(0., center-1.)
    right = left+max(2., min(4., metrics['latency_ms']['max']/1000+1.))
    view = (t >= left) & (t <= right)
    for key, label, style in [('tx', 'TX: aquisição simulada', '-'),
                              ('generated', 'TX: último valor calculado pelo Python', ':'),
                              ('rx_acquisition', 'RX: timestamp de aquisição corrigido', '--'),
                              ('rx_delivery', 'RX: último valor disponível na leitura', '-')]:
        axes[0].plot(t[view], reconstructed[key][view, 0], style, label=label,
                     alpha=.7 if key == 'rx_delivery' else 1.)
    axes[0].set(xlabel='Tempo desde a origem comum (s)', ylabel='Canal 0',
                title='Reconstrução temporal sem compensar o atraso — região de maior atraso')
    for key, label in [('tx', 'TX'), ('rx_acquisition', 'RX aquisição'), ('rx_delivery', 'RX leitura')]:
        axes[1].plot(t[view], (reconstructed[key]-reconstructed['tx'])[view, 0], label=label)
    axes[1].set(xlabel='Tempo desde a origem comum (s)', ylabel='Erro de amplitude no mesmo tempo',
                title='Erro temporal: valores intactos podem estar disponíveis no instante errado')
    elapsed = data['elapsed']
    axes[2].plot(elapsed, data['latency']*1000, label='Aquisição → leitura', lw=.7)
    axes[2].plot(elapsed, data['queue']*1000, label='Envio → leitura', alpha=.6, lw=.7)
    axes[2].axhline(metrics['limits']['latency_ms'], color='r', ls=':', label='Limite p99 de referência')
    axes[2].set(xlabel='Tempo nominal (s)', ylabel='Atraso (ms)')
    axes[3].plot(elapsed, data['timing_tx']*1000, label='TX vs taxa nominal')
    axes[3].plot(elapsed, data['timing_rx']*1000, '--', label='RX aquisição vs taxa nominal')
    axes[3].set(xlabel='Tempo nominal (s)', ylabel='Erro acumulado (ms)',
                title='Drift de aquisição acumulado (não é a soma dos atrasos)')
    axes[4].plot(elapsed, (data['latency']-data['latency'][0])*1000, label='Variação do atraso desde a 1ª amostra')
    axes[4].plot(elapsed, data['read_error']*1000, alpha=.6, label='Erro de entrega vs taxa nominal')
    axes[4].set(xlabel='Tempo nominal (s)', ylabel='Variação acumulada (ms)',
                title='Entrega: filas e pausas podem crescer sem alterar o sinal')
    valid = [w for w in trends if w['drift_ppm'] is not None]
    axes[5].plot([w['time'] for w in valid], [w['drift_ppm'] for w in valid], '.-', label='Drift local (janelas de 2 s)')
    axes[5].axhline(0, color='gray', lw=.5)
    axes[5].set(xlabel='Tempo nominal (s)', ylabel='Drift local (ppm)',
                title='Evolução do drift — estimativas curtas são sensíveis a ruído de relógio')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc='upper right', fontsize=8)
    flags = metrics['issues']
    status = '; '.join(f['message'] for f in flags) or 'Nenhum limite configurado excedido (não certifica sincronização física)'
    from textwrap import fill
    fig.suptitle(f'{name}: {sensor["rate"]} Hz / {sensor["channels"]} canais / chunk {cfg["chunk"]}\n'
                 + fill(status, 105), color='darkred' if flags else 'darkgreen', fontsize=11)
    fig.savefig(out/f'{name}_tempo.png', dpi=120)
    plt.close(fig)


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
    correction_path = out/'corrections.json'
    observations = json.loads(correction_path.read_text()) if correction_path.exists() else []
    # One fixed domain translation for every stream, not a fit to the signals.
    anchor_offset = next((o['offset'] for o in observations if o['stream'] != 'markers'), None)
    report['clock_anchor'] = dict(tx_origin=txmeta['start'], rx_minus_tx_initial=anchor_offset,
        method='Fixed initial LSL clock offset; no per-signal start alignment or delay removal')
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
        if data is not None:
            if anchor_offset is None:
                anchor_offset = float(rx['correction'][0])
                report['clock_anchor']['rx_minus_tx_initial'] = anchor_offset
            reconstructed, temporal = temporal_reconstruction(tx, rx, s['rate'], txmeta['start'], anchor_offset)
            metrics['temporal_reconstruction'] = temporal
            trends = windows(data['elapsed'], data['latency'], data['timing_rx'])
            metrics['windows'] = trends
            metrics['issues'], metrics['limits'] = diagnose(metrics, trends, cfg.get('analysis_limits'))
            np.savez_compressed(out/f'{name}_recon.npz', **reconstructed)
            plot_diagnostics(out, name, s, cfg, tx, rx, data, reconstructed, metrics, trends)
            with (out/f'{name}_janelas.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(trends[0]))
                writer.writeheader()
                writer.writerows(trends)
        else:
            metrics['issues'], metrics['limits'] = diagnose(metrics, [], cfg.get('analysis_limits'))
        rows.append(dict(sensor=name, rate=s['rate'], channels=s['channels'], chunk=cfg['chunk'],
                         expected=metrics['expected'], received=metrics['received'], missing=metrics['missing'],
                         p99_ms=metrics.get('latency_ms', {}).get('p99'), rmse=metrics.get('rmse'),
                         drift_ppm=metrics.get('rx_timing_drift_ppm'),
                         accumulated_ms=metrics.get('rx_accumulated_drift_ms'),
                         temporal_acquisition_rmse=metrics.get('temporal_reconstruction', {}).get('acquisition', {}).get('rmse'),
                         temporal_delivery_rmse=metrics.get('temporal_reconstruction', {}).get('delivery', {}).get('rmse'),
                         issues=';'.join(f['code'] for f in metrics['issues'])))
        if data is None:
            continue
        ir, it = data['ir'], data['it']
        fig, axes = plt.subplots(5, 1, figsize=(12, 15), constrained_layout=True)
        origin = txmeta['start']
        # Fixed translation preserves changes in correction and actual delays.
        corrected_source = rx['corrected'][ir]-anchor_offset
        read_source = rx['arrival'][ir]-anchor_offset
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
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        for name, events in marker_alignment.items():
            axes[0].plot([e['id'] for e in events], [e['sample_minus_marker_ms'] for e in events], '.-', label=name)
            axes[1].plot([e['id'] for e in events], [e['read_minus_marker_read_ms'] for e in events], '.-', label=name)
        axes[0].set(ylabel='Amostra − marcador (ms)', title='Alinhamento dos timestamps de aquisição corrigidos pelo LSL')
        axes[1].set(ylabel='Leitura do sinal − leitura do marcador (ms)', title='Disponibilidade no receptor: inclui filas e agrupamento')
        for ax in axes:
            ax.set_xlabel('ID do marcador sync')
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
        'Signals are synthetic, with a common coded pilot; units are illustrative.',
        'Temporal reconstruction uses each series timestamps on a shared grid, without delay compensation.',
        'Delivery reconstruction holds the latest available sample per read; within-chunk arrival times are not observable.',
        'Issue thresholds are illustrative configurable limits, not sensor requirements or physical certification.'
    ]
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Diagnóstico temporal', '',
             'Reconstrução em uma origem comum, sem remover atraso ou alinhar inícios individualmente.', '',
             '| Sensor | Perdas | RMSE por índice | RMSE temporal na leitura | Atraso p99 (ms) | Drift acumulado (ms) |',
             '| --- | --- | --- | --- | --- | --- |']
    def number(value):
        return f'{value:.4g}' if value is not None else 'indisponível'
    for row in rows:
        lines.append(f'| {row["sensor"]} | {row["missing"]} | {number(row["rmse"])} | '
                     f'{number(row["temporal_delivery_rmse"])} | {number(row["p99_ms"])} | {number(row["accumulated_ms"])} |')
    for name, metrics in report['sensors'].items():
        lines += ['', f'**{name}** — [{name}_tempo.png]({name}_tempo.png)']
        lines += ['- '+issue['message'] for issue in metrics['issues']] or ['- Nenhum limite configurado excedido.']
    lines += ['', f'Marcadores: {report["markers"]["received"]}/{report["markers"]["expected"]} recebidos; '
              f'{report["markers"]["missing"]} perdidos; {report["markers"]["duplicates"]} duplicados.',
              f'Conclusão: transmissor={txmeta["complete"]}, receptor={rxmeta["complete"]}.']
    lines += ['', 'Os limites estão em `report.json` e podem ser alterados com `analysis_limits` no JSON do experimento.',
              'RMSE temporal usa valores no mesmo instante; RMSE por índice mede apenas preservação de amplitude.',
              'Leitura é o fim de pull_chunk no Python, não chegada física ao sensor ou à interface de rede.',
              'Drift acumulado é diferença temporal desde a primeira amostra pareada, não soma dos atrasos.',
              'Latência crescente pode indicar filas; não é, por si só, drift de relógio.']
    (out/'diagnostico.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report
