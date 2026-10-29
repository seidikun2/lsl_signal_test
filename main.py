"""Run the EEG matrix and/or the incremental-stream experiment."""
import argparse
import csv
import itertools
import multiprocessing as mp
import runpy
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from lsl_lab.config import validate, fresh, save_json, wall_duration, sample_count


def run(cfg, directory):
    from lsl_lab.generator import transmit
    from lsl_lab.receiver import receive
    from lsl_lab.analysis import analyze
    out = fresh(directory)
    data = out/'data'
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
    limit = time.monotonic()+120+max(wall_duration(cfg), cfg['duration'])+2*cfg['receiver_pause']
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
              f'Sample RMSE={result.get("rmse")}; '
              f'Delivery-time RMSE={result.get("temporal_reconstruction", {}).get("delivery", {}).get("rmse")}', flush=True)
        for issue in result.get('issues', []):
            print(f'  [WARNING] {name}: {issue["message"]}', flush=True)
    print(f'Results: {out}/summary.md and {out}/figures/', flush=True)
    return report


def build_cases(settings):
    """Build the complete, validated plan before starting any recording."""
    tests = settings['TESTS']
    mode = tests['length_mode']
    if mode not in ('seconds', 'samples'):
        raise ValueError('length_mode must be seconds or samples')
    if mode == 'samples' and (not isinstance(tests['samples_per_stream'], int) or tests['samples_per_stream'] < 1):
        raise ValueError('samples_per_stream must be a positive integer')
    base = validate(dict(settings['CONFIG'], duration=tests['duration_s']))
    experiments = tests['run']
    if not experiments or len(experiments)!=len(set(experiments)) or set(experiments)-{'eeg','streams'}:
        raise ValueError('run must contain eeg and/or streams, without duplicates')
    cases = []
    def add(experiment, name, sensors, **overrides):
        cfg = dict(base, **overrides)
        cfg['sensors'] = [dict(s) for s in sensors]
        if mode == 'samples':
            for sensor in cfg['sensors']:
                sensor['samples'] = tests['samples_per_stream']
            cfg['duration'] = max(s['samples']/s['rate'] for s in cfg['sensors'])
        else:
            for sensor in cfg['sensors']:
                sensor.pop('samples', None)
        cfg['online_rate'] = max(s['rate'] for s in cfg['sensors'])
        cfg['experiment'], cfg['length_mode'] = experiment, mode
        cases.append((experiment, name, validate(cfg)))
    for experiment in experiments:
        options = tests[experiment]
        if not isinstance(options['repeats'], int) or options['repeats'] < 1:
            raise ValueError('repeats must be a positive integer')
        if experiment == 'eeg':
            template = next((s for s in base['sensors'] if s['kind'].upper()=='EEG'), {})
            for rate, channels, chunk, buffer, pause, repeat in itertools.product(
                    options['rates'], options['channels'], options['chunks'],
                    options['buffers_s'], options['pauses_s'], range(options['repeats'])):
                sensor = dict(template, name='eeg', kind='EEG', rate=rate, channels=channels)
                name = f'f{rate:g}_c{channels}_k{chunk}_b{buffer:g}_p{pause:g}_r{repeat+1}'
                add('eeg', name, [sensor], chunk=chunk, inlet_buffer=buffer, receiver_pause=pause)
        else:
            for count in range(1, len(base['sensors'])+1):
                for repeat in range(options['repeats']):
                    add('streams', f'n{count}_r{repeat+1}', base['sensors'][:count])
    if not cases or any(not any(e==experiment for e,_,_ in cases) for experiment in experiments):
        raise ValueError('Experiment parameter lists cannot be empty')
    return cases


def write_plan(root, plan):
    with (root/'plan.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(plan[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(plan)
    lines = ['# Experiment summary', '', 'The full case list and execution status are in [plan.csv](plan.csv).', '',
             '| Experiment | Planned | Completed | Failed | Not run | Results |', '| --- | --- | --- | --- | --- | --- |']
    for experiment in dict.fromkeys(p['experiment'] for p in plan):
        rows = [p for p in plan if p['experiment']==experiment]
        completed = sum(p['status']=='completed' for p in rows)
        failed = sum(p['status']=='failed' for p in rows)
        link = f'[{experiment}]({experiment}/summary.md)' if completed or failed else 'not run yet'
        lines.append(f'| {experiment} | {len(rows)} | {completed} | {failed} | {len(rows)-completed-failed} | {link} |')
    lines += ['', 'Recordings use the configured duration or exact sample count per stream.',
              'Startup, clock estimation, saving and plotting add time beyond acquisition.',
              'No delivery-delay compensation or online alignment is applied.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Run the LSL experiments configured in settings.py')
    parser.add_argument('--config', default=str(Path(__file__).with_name('settings.py')))
    parser.add_argument('--out', help='New output directory (default: results/timestamp)')
    args = parser.parse_args()
    settings = runpy.run_path(args.config)
    cases = build_cases(settings)
    root = fresh(args.out or f'results/{datetime.now():%y%m%d_%H%M%S_%f}')
    plan = [dict(experiment=e, case=name, status='planned', length_mode=cfg['length_mode'],
                 duration_s=cfg['duration'], streams=len(cfg['sensors']),
                 sensors=';'.join(s['name'] for s in cfg['sensors']),
                 rates_hz=';'.join(str(s['rate']) for s in cfg['sensors']),
                 channels=';'.join(str(s['channels']) for s in cfg['sensors']),
                 samples=';'.join(str(sample_count(cfg,s)) for s in cfg['sensors']),
                 chunk=cfg['chunk'], buffer_s=cfg['inlet_buffer'], pause_s=cfg['receiver_pause'],
                 marker_classes=';'.join(m['label'] for m in cfg['markers'])) for e,name,cfg in cases]
    write_plan(root, plan)
    print(f'{len(cases)} cases; {sum(wall_duration(c) for _,_,c in cases)/60:.1f} minutes acquisition, plus startup and analysis.', flush=True)
    from lsl_lab.analysis import case_result, write_comparison, write_eeg_matrix
    results = {e: [] for e,_,_ in cases}
    links = {e: {} for e in results}
    for index, (experiment, name, cfg) in enumerate(cases):
        directory = root/experiment
        directory.mkdir(exist_ok=True)
        try:
            report = run(cfg, directory/name)
            for sensor in cfg['sensors']:
                results[experiment].append(case_result(name,cfg,sensor,report['sensors'][sensor['name']]))
            links[experiment][name] = f'{name}/summary.md'
            plan[index]['status'] = 'completed'
        except Exception as error:
            plan[index]['status'] = 'failed'
            for sensor in cfg['sensors']:
                results[experiment].append(case_result(name,cfg,sensor,error=str(error)))
            print(f'[FAILED] {experiment}/{name}: {error}', flush=True)
        write_comparison(directory, results[experiment], f'{experiment.upper()} experiment', links[experiment], plot=False)
        write_plan(root, plan)
    for experiment, rows in results.items():
        write_comparison(root/experiment, rows, f'{experiment.upper()} experiment', links[experiment], plot=experiment!='eeg')
        if experiment == 'eeg':
            write_eeg_matrix(root/experiment, rows)
    print(f'Open {root}/summary.md', flush=True)
    if any(p['status']=='failed' for p in plan):
        raise RuntimeError('Some cases failed; inspect plan.csv and the experiment summaries')


if __name__ == '__main__':
    main()
