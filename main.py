"""Executa gerador, receptor e análise; também organiza varreduras de parâmetros."""
import argparse
import itertools
import multiprocessing as mp
import sys
import time
import uuid
from datetime import datetime, timezone

from config import load, validate, fresh, save_json, wall_duration


def run(cfg, directory):
    from gerador import transmit
    from receptor import receive
    from analise import analyze
    out = fresh(directory)
    run_id = uuid.uuid4().hex[:12]
    save_json(out/'config.json', cfg)
    save_json(out/'run.json', dict(run_id=run_id, utc=datetime.now(timezone.utc).isoformat()))
    # Log arrays and Python lists use more than raw payload size.
    raw_mb = sum(s['rate']*s['channels']*cfg['duration']*4 for s in cfg['sensors'])/1e6
    print(f'{out}: {run_id}; signal payload {raw_mb:.1f} MB (RAM use is several times larger)', flush=True)
    context = mp.get_context('spawn')
    tx = context.Process(target=transmit, args=(cfg, run_id, str(out)), name='transmitter')
    rx = context.Process(target=receive, args=(cfg, run_id, str(out)), name='receiver')
    tx.start()
    rx.start()
    limit = time.monotonic()+120+max(wall_duration(cfg), cfg['duration']/(1+cfg['drift_ppm']*1e-6))+2*cfg['receiver_pause']
    try:
        while tx.is_alive() or rx.is_alive():
            if (tx.exitcode is not None and tx.exitcode != 0) or (rx.exitcode is not None and rx.exitcode != 0):
                raise RuntimeError(f'LSL process failed (TX={tx.exitcode}, RX={rx.exitcode}); see traceback above')
            if time.monotonic() > limit:
                raise TimeoutError('Experiment exceeded its time limit')
            time.sleep(.1)
        tx.join()
        rx.join()
        if tx.exitcode or rx.exitcode:
            raise RuntimeError(f'TX={tx.exitcode}, RX={rx.exitcode}')
    finally:
        for process in (tx, rx):
            if process.is_alive():
                process.terminate()
            process.join()
    report = analyze(out)
    for name, result in report['sensors'].items():
        print(f'{name}: received={result["received"]}/{result["expected"]}, '
              f'missing={result["missing"]}, p99={result.get("latency_ms", {}).get("p99")} ms, '
              f'RMSE por índice={result.get("rmse")}; '
              f'RMSE temporal na leitura={result.get("temporal_reconstruction", {}).get("delivery", {}).get("rmse")}', flush=True)
        for issue in result.get('issues', []):
            print(f'  [AVISO] {name}: {issue["message"]}', flush=True)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('--config', '--out'):
        argv.insert(0, 'run')
    parser = argparse.ArgumentParser(description='LSL signal/marker stress laboratory')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='Executar gerador, receptor e análise')
    p.add_argument('--config', default='configs/single.json')
    p.add_argument('--out')
    p = sub.add_parser('sweep')
    p.add_argument('--rates', nargs='+', type=float, default=[100, 250, 500, 1000, 1500, 2000])
    p.add_argument('--channels', nargs='+', type=int, default=[1, 8, 32])
    p.add_argument('--chunks', nargs='+', type=int, default=[1, 16, 64, 128])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--duration', type=float, default=10.)
    p.add_argument('--kind', choices=['EEG', 'EMG', 'IMU'], default='EEG')
    p.add_argument('--inlet-buffer', type=int, default=60)
    p.add_argument('--receiver-pause', type=float, default=0.)
    p.add_argument('--drift-ppm', type=float, default=0.)
    p.add_argument('--out')
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime('%y%m%d_%H%M%S')
    if args.command == 'sweep':
        if args.repeats < 1:
            parser.error('--repeats must be positive')
        root = fresh(args.out or f'runs/swp_{stamp}')
        jobs = list(itertools.product(args.rates, args.channels, args.chunks, range(args.repeats)))
        print(f'{len(jobs)} cases; acquisition time alone: {len(jobs)*args.duration/60:.1f} min', flush=True)
        # Validate the entire matrix before spending time on any case.
        cases = [(f'f{rate:g}c{channels}b{chunk}r{repeat+1}', validate(dict(
            sensors=[dict(name=args.kind.lower(), kind=args.kind, rate=rate, channels=channels)],
            duration=args.duration, chunk=chunk, inlet_buffer=args.inlet_buffer,
            receiver_pause=args.receiver_pause, drift_ppm=args.drift_ppm)))
            for rate, channels, chunk, repeat in jobs]
        import csv
        with (root/'sweep.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['case', 'status', 'missing', 'p99_ms', 'drift_ppm', 'error'])
            writer.writeheader()
            failures = 0
            for name, cfg in cases:
                try:
                    r = run(cfg, root/name)['sensors'][args.kind.lower()]
                    row = dict(case=name, status='ok', missing=r['missing'],
                               p99_ms=r.get('latency_ms', {}).get('p99'), drift_ppm=r.get('rx_timing_drift_ppm'))
                except Exception as e:
                    failures += 1
                    row = dict(case=name, status='failed', error=str(e))
                    print(f'{name}: {e}', file=sys.stderr, flush=True)
                writer.writerow(row)
                f.flush()
        if failures:
            raise RuntimeError(f'{failures} cases failed; see {root}/sweep.csv')
    else:
        cfg = load(args.config)
        run(cfg, args.out or f'runs/run_{stamp}')


if __name__ == '__main__':
    main()
