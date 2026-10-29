"""Reconstruct from timestamps, never from sample positions or fitted delays."""
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
