"""Reconstrói pelos vetores de tempo, mede erros e gera gráficos e diagnóstico."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


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
        flag('loss', f"{metrics['missing']} amostras perdidas")
    if metrics['duplicates'] or metrics['reordered'] or metrics['unexpected']:
        flag('sequence', 'Duplicação, reordenação ou índices inesperados')
    if 'latency_ms' not in metrics:
        flag('no_samples', 'Sem amostras pareadas para avaliar tempo')
        return flags, limits
    if metrics['rmse'] > 0:
        flag('amplitude', 'Valores recebidos diferem dos transmitidos')
    if metrics['latency_ms']['p99'] > limits['latency_ms']:
        flag('latency', f"Atraso p99 > {limits['latency_ms']:g} ms")
    if metrics['latency_ms']['std'] > limits['jitter_ms']:
        flag('jitter', f"Jitter > {limits['jitter_ms']:g} ms")
    if abs(metrics.get('rx_timing_drift_ppm') or 0) > limits['drift_ppm']:
        flag('drift', f"Drift de aquisição > {limits['drift_ppm']:g} ppm em módulo")
    if abs(metrics['rx_accumulated_drift_ms']) > limits['accumulated_ms']:
        flag('accumulated_drift', f"Drift acumulado > {limits['accumulated_ms']:g} ms em módulo")
    if metrics.get('corrected_timestamp_reversals', 0):
        flag('timestamps', 'Timestamps corrigidos não são estritamente crescentes')
    timestamp_error = metrics['timestamp_error_ms']
    if max(abs(timestamp_error['min']), abs(timestamp_error['max'])) > limits['accumulated_ms']:
        flag('timestamp_mismatch', 'Timestamps recebidos diferem dos timestamps de origem além do limite temporal')
    if len(trends) >= 4:
        n = max(1, len(trends)//3)
        growth = (np.median([w['latency_median_ms'] for w in trends[-n:]]) -
                  np.median([w['latency_median_ms'] for w in trends[:n]]))
        if growth > limits['latency_growth_ms']:
            flag('latency_growth', f'Atraso mediano cresceu {growth:.2f} ms entre janelas iniciais e finais')
        first = [abs(w['drift_ppm']) for w in trends[:n] if w['drift_ppm'] is not None]
        last = [abs(w['drift_ppm']) for w in trends[-n:] if w['drift_ppm'] is not None]
        if first and last and np.median(last)-np.median(first) > limits['drift_ppm']:
            flag('growing_drift', 'Módulo do drift local aumentou entre janelas iniciais e finais')
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
    fig.savefig(out/f'{name}.png', dpi=120)
    plt.close(fig)


def analyze(directory, tx_directory=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(directory)
    source = Path(tx_directory) if tx_directory else out
    source = source/'dados' if (source/'dados').is_dir() else source
    recorded = out/'dados' if (out/'dados').is_dir() else out
    figures = out/'figuras'
    figures.mkdir(exist_ok=True)
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
    rows = []
    rxmarkers = json.loads((recorded/'rx_markers.json').read_text())
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
            np.savez_compressed(recorded/f'{name}_recon.npz', **reconstructed)
            plot_diagnostics(figures, name, s, cfg, tx, rx, data, reconstructed, metrics, trends)
            with (recorded/f'{name}_janelas.csv').open('w', newline='') as f:
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
        fig.savefig(figures/'sync.png', dpi=130)
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
    (recorded/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (out/'sumario.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Sumário da transmissão', '',
             'Reconstrução em uma origem comum, sem remover atraso ou alinhar inícios individualmente.', '',
             '| Sensor | Recebidas/previstas | Perdas | RMSE por índice | RMSE temporal na leitura | Atraso p99 (ms) | Drift (ppm) | Drift acumulado (ms) |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    def number(value):
        return f'{value:.4g}' if value is not None else 'indisponível'
    for row in rows:
        lines.append(f'| {row["sensor"]} | {row["received"]}/{row["expected"]} | {row["missing"]} | {number(row["rmse"])} | '
                     f'{number(row["temporal_delivery_rmse"])} | {number(row["p99_ms"])} | {number(row["drift_ppm"])} | {number(row["accumulated_ms"])} |')
    for name, metrics in report['sensors'].items():
        lines += ['', f'**{name}** — [Ver figura](figuras/{name}.png)']
        lines += ['- '+issue['message'] for issue in metrics['issues']] or ['- Nenhum limite configurado excedido.']
    if (figures/'sync.png').exists():
        lines += ['', '[Alinhamento dos marcadores](figuras/sync.png)']
    lines += ['', f'Marcadores: {report["markers"]["received"]}/{report["markers"]["expected"]} recebidos; '
              f'{report["markers"]["missing"]} perdidos; {report["markers"]["duplicates"]} duplicados.',
              f'Conclusão: transmissor={txmeta["complete"]}, receptor={rxmeta["complete"]}.']
    lines += ['', 'Os detalhes e limites estão em `dados/report.json` e podem ser alterados com `analysis_limits` em configuracao.py.',
              'RMSE temporal usa valores no mesmo instante; RMSE por índice mede apenas preservação de amplitude.',
              'Leitura é o fim de pull_chunk no Python, não chegada física ao sensor ou à interface de rede.',
              'Drift acumulado é diferença temporal desde a primeira amostra pareada, não soma dos atrasos.',
              'Latência crescente pode indicar filas; não é, por si só, drift de relógio.']
    (out/'sumario.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Analisar sinais e tempos gravados')
    parser.add_argument('directory', help='Pasta da aquisição do receptor ou do main')
    parser.add_argument('--tx-dir', help='Pasta do gerador, se gravada separadamente')
    args = parser.parse_args(argv)
    report = analyze(args.directory, args.tx_dir)
    print(json.dumps(report, indent=2))
    print(f'Gráficos e diagnóstico salvos em {args.directory}')


if __name__ == '__main__':
    main()
