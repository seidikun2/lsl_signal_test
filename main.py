"""Execute este arquivo: gera, recebe, grava e analisa a transmissão."""
import argparse
import csv
import itertools
import json
import multiprocessing as mp
import runpy
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from codigo.config import validate, fresh, save_json, wall_duration


def run(cfg, directory):
    from codigo.gerador import transmit
    from codigo.receptor import receive
    from codigo.analise import analyze
    out = fresh(directory)
    data = out/'dados'
    data.mkdir()
    run_id = uuid.uuid4().hex[:12]
    save_json(data/'config.json', cfg)
    save_json(data/'run.json', dict(run_id=run_id, utc=datetime.now(timezone.utc).isoformat()))
    # Log arrays and Python lists use more than raw payload size.
    raw_mb = sum(s['rate']*s['channels']*cfg['duration']*4 for s in cfg['sensors'])/1e6
    print(f'{out}: {run_id}; signal payload {raw_mb:.1f} MB (RAM use is several times larger)', flush=True)
    context = mp.get_context('spawn')
    tx = context.Process(target=transmit, args=(cfg, run_id, str(data)), name='transmitter')
    rx = context.Process(target=receive, args=(cfg, run_id, str(data)), name='receiver')
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
    print(f'Resultados: {out}/sumario.md e {out}/figuras/', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description='Executar a simulação LSL')
    parser.add_argument('--config', default=str(Path(__file__).with_name('configuracao.py')))
    parser.add_argument('--out', help='Pasta de saída; por padrão cria uma pasta em resultados/')
    args = parser.parse_args()
    path = Path(args.config)
    settings = runpy.run_path(str(path)) if path.suffix == '.py' else {"CONFIG": json.loads(path.read_text())}
    cfg = validate(settings['CONFIG'])
    sweep = settings.get('VARREDURA', {})
    out = args.out or f'resultados/{datetime.now():%y%m%d_%H%M%S_%f}'
    if not sweep.get('ativar', False):
        run(cfg, out)
        return

    repeats = sweep['repeticoes']
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeticoes deve ser um inteiro positivo')
    kind = sweep['sensor'].upper()
    cases = []
    sensor = next((s for s in cfg['sensors'] if s['kind'].upper() == kind), {})
    for rate, channels, chunk, repeat in itertools.product(
            sweep['frequencias'], sweep['canais'], sweep['chunks'], range(repeats)):
        case = dict(cfg, chunk=chunk, online_rate=max(rate, cfg['online_rate']),
                    sensors=[dict(sensor, name=kind.lower(), kind=kind, rate=rate, channels=channels)])
        cases.append((f'f{rate:g}c{channels}b{chunk}r{repeat+1}', validate(case)))
    if not cases:
        raise ValueError('A varredura precisa ter frequencias, canais e chunks')
    root = fresh(out)
    print(f'{len(cases)} testes; {len(cases)*cfg["duration"]/60:.1f} min só de aquisição', flush=True)
    lines = ['# Sumário da varredura', '',
             '| Teste | Estado | Perdas | Atraso p99 (ms) | Drift (ppm) |',
             '| --- | --- | --- | --- | --- |']
    failures = 0
    with (root/'sumario.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['teste', 'estado', 'perdas', 'p99_ms', 'drift_ppm', 'erro'])
        writer.writeheader()
        for name, case in cases:
            try:
                result = run(case, root/name)['sensors'][kind.lower()]
                row = dict(teste=name, estado='concluído', perdas=result['missing'],
                           p99_ms=result.get('latency_ms', {}).get('p99'), drift_ppm=result.get('rx_timing_drift_ppm'))
                label = f'[{name}]({name}/sumario.md)'
            except Exception as error:
                failures += 1
                row = dict(teste=name, estado='falhou', erro=str(error))
                label = name
                print(f'{name}: {error}', flush=True)
            writer.writerow(row)
            f.flush()
            p99 = f'{row["p99_ms"]:.2f}' if row.get('p99_ms') is not None else '—'
            drift = f'{row["drift_ppm"]:.2f}' if row.get('drift_ppm') is not None else '—'
            lines.append(f'| {label} | {row["estado"]} | {row.get("perdas", "—")} | '
                         f'{p99} | {drift} |')
            (root/'sumario.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if failures:
        raise RuntimeError(f'{failures} testes falharam; consulte {root}/sumario.csv')


if __name__ == '__main__':
    main()
