"""Run Test 1 (EEG matrix) and Test 2 (incremental streams)."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import itertools
import multiprocessing as mp
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from helpers import (validate, save_json, wall_duration, sample_count, read_settings,
                     prepare_results, pack_recording)


def run(cfg, output, case):
    """Use temporary IPC files; keep only one recording archive and snapshots."""
    import tempfile
    from transmission import transmit
    from reception import receive
    from analysis import analyze
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='lsl_recording_') as temporary:
        raw=Path(temporary)
        identity=uuid.uuid4().hex[:12]
        save_json(raw/'config.json',cfg)
        save_json(raw/'run.json',dict(case=case,run_id=identity,utc=datetime.now(timezone.utc).isoformat()))
        context=mp.get_context('spawn')
        tx=context.Process(target=transmit,args=(cfg,identity,temporary),name='transmission')
        rx=context.Process(target=receive,args=(cfg,identity,temporary),name='reception')
        tx.start(); rx.start()
        deadline=time.monotonic()+120+max(wall_duration(cfg),cfg['duration'])+2*cfg['receiver_pause']
        try:
            while tx.is_alive() or rx.is_alive():
                if any(p.exitcode is not None and p.exitcode!=0 for p in (tx,rx)):
                    raise RuntimeError(f'LSL process failed: TX={tx.exitcode}, RX={rx.exitcode}')
                if time.monotonic()>deadline: raise TimeoutError('Recording time limit exceeded')
                time.sleep(.05)
            if tx.exitcode or rx.exitcode: raise RuntimeError('An LSL process failed')
        finally:
            for process in (tx,rx):
                if process.is_alive(): process.terminate()
                process.join()
            # Preserve available raw data even if a process failed.
            pack_recording(raw,output/f'{case}_recording.npz')
        report=analyze(raw,output,case)
        report['elapsed_s']=time.monotonic()-started
        save_json(raw/'report.json',report)
        pack_recording(raw,output/f'{case}_recording.npz')
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


def make_plan(cases):
    plan=[]
    counters={1:0,2:0}
    for experiment,name,cfg in cases:
        number=1 if experiment=='eeg' else 2
        counters[number]+=1
        count=len(cfg['sensors'])
        low=wall_duration(cfg)+cfg['start_delay']+max(2.,cfg['receiver_pause']+1)+.5*count
        high=low+4+count
        plan.append(dict(case=f'T{number}_{counters[number]:02}',test=number,status='planned',
            length_mode=cfg['length_mode'],sensors=';'.join(s['name'] for s in cfg['sensors']),
            rates_hz=';'.join(str(s['rate']) for s in cfg['sensors']),channels=';'.join(str(s['channels']) for s in cfg['sensors']),
            samples=';'.join(str(sample_count(cfg,s)) for s in cfg['sensors']),chunk=cfg['chunk'],buffer_s=cfg['inlet_buffer'],
            pause_s=cfg['receiver_pause'],acquisition_s=float(wall_duration(cfg)),estimate_low_s=low,estimate_high_s=high,
            actual_s=None,error=''))
    return plan


def write_csv(path,rows):
    if not rows: return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def write_plan(root,plan,image=False,console=False):
    write_csv(root/'plan.csv',plan)
    low=sum(p['estimate_low_s'] for p in plan); high=sum(p['estimate_high_s'] for p in plan)
    if console:
        print('\nPLANNED TESTS — no recordings have started',flush=True)
        print('Case     Test  Streams / rates Hz / channels                 Acquire   Estimated total',flush=True)
        for p in plan:
            parameters=p['sensors']+' / '+p['rates_hz']+' / '+p['channels']
            print(f'{p["case"]:<8} {p["test"]:<5} {parameters:<50} {p["acquisition_s"]:5.1f}s   {p["estimate_low_s"]:.0f}–{p["estimate_high_s"]:.0f}s',flush=True)
        print(f'\n{len(plan)} cases; estimated total {low/60:.1f}–{high/60:.1f} minutes. Includes startup, clock estimation and analysis.',flush=True)
        print(f'Plan saved to {root}/plan.csv and plan.png. Estimates depend on this computer.\n',flush=True)
    if not image: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for start in range(0,len(plan),25):
        page=plan[start:start+25]
        cells=[]
        for p in page:
            description='\n'.join(f'{name}: {rate} Hz / {channels} ch' for name,rate,channels in zip(p['sensors'].split(';'),p['rates_hz'].split(';'),p['channels'].split(';')))
            description+=f'\nchunk {p["chunk"]}; buffer {p["buffer_s"]:g}s; pause {p["pause_s"]:g}s'
            cells.append([p['case'],p['test'],description,f'{p["acquisition_s"]:.1f}s',
                          f'{p["estimate_low_s"]:.0f}–{p["estimate_high_s"]:.0f}s',p['status'],
                          '—' if p['actual_s'] is None else f'{p["actual_s"]:.1f}s'])
        fig,ax=plt.subplots(figsize=(15,max(4,sum(.45+.12*len(p['sensors'].split(';')) for p in page)+1.5)))
        ax.axis('off')
        table=ax.table(cellText=cells,colLabels=['Case','Test','Conditions','Acquire','Estimate','Status','Actual'],
                       colWidths=[.08,.05,.43,.08,.12,.12,.08],cellLoc='left',loc='center')
        table.auto_set_font_size(False); table.set_fontsize(8)
        for (r,c),cell in table.get_celld().items():
            cell.set_height(.04 if r==0 else (.45+.12*len(page[r-1]['sensors'].split(';')))/(fig.get_figheight()-1))
            if r==0: cell.set_facecolor('#dce8f4')
            elif page[r-1]['status']=='failed': cell.set_facecolor('#ffe0da')
        ax.set_title(f'Complete test plan — estimated total {low/60:.1f}–{high/60:.1f} min\nHeuristic estimate; actual timing is recorded for every case',pad=15)
        fig.savefig(root/('plan.png' if start==0 else f'plan_{start//25+1:02}.png'),dpi=130,bbox_inches='tight')
        plt.close(fig)


def _image_data(path):
    import base64
    return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def _collect_normal(case,cfg,report,rows,events):
    from analysis import case_result
    for sensor in cfg['sensors']:
        metrics=report['sensors'][sensor['name']]
        rows.append(case_result(case,cfg,sensor,metrics))
        if sensor['kind'].upper()=='EEG':
            for event in metrics.get('responses',{}).get('events',[]):
                events.append(dict(case=case,test=1 if cfg['experiment']=='eeg' else 2,stream=sensor['name'],**event))


def _normal_views(root,work,plan,rows,events,assets,machine,finished=False):
    from analysis import write_eeg_matrix,write_stream_matrix,write_results_report
    maps={}
    if finished:
        directory=Path(work)/'maps';directory.mkdir(exist_ok=True)
        eeg=[r for r in rows if r['experiment']=='eeg']
        streams=[r for r in rows if r['experiment']=='streams']
        if eeg:write_eeg_matrix(directory,eeg,plan)
        if streams:write_stream_matrix(directory,streams)
        for path in directory.glob('*.png'):
            maps[path.stem]=_image_data(path)
    write_csv(root/'metrics.csv',rows)
    write_csv(root/'markers.csv',events)
    # Keep the four-file layout even when a failed run produced no rows.
    for name in ('metrics.csv','markers.csv'):
        if not (root/name).exists():(root/name).write_text('case,status\n')
    write_results_report(root,plan,rows,events,assets,maps,machine)


def normal_experiment(config,output=None,plan_only=False):
    import json,platform,tempfile,zipfile
    from analysis import case_result,write_results_report
    document=json.loads(Path(config).read_text())
    cases=build_cases(read_settings(config));plan=make_plan(cases)
    low=sum(p['estimate_low_s'] for p in plan);high=sum(p['estimate_high_s'] for p in plan)
    print(f'Test 1: {sum(p["test"]==1 for p in plan)} EEG cases. Test 2: {sum(p["test"]==2 for p in plan)} additional-signal cases.',flush=True)
    note='No added processing wait.' if all(c[2]['receiver_pause']==0 for c in cases) else 'An extra receiver pause is configured.'
    print(f'Estimated total: {low/60:.1f}–{high/60:.1f} minutes, including setup and recording. {note}',flush=True)
    if plan_only:
        for p in plan:print(f'{p["case"]}: {p["sensors"]}, {p["rates_hz"]} Hz, {p["channels"]} channels',flush=True)
        if output:
            root=Path(output);root.mkdir(parents=True,exist_ok=True)
            if any(root.iterdir()):raise ValueError('Plan output must be empty')
            write_csv(root/'plan.csv',plan)
            write_results_report(root,plan,[],[],{},{},{})
        return
    root=prepare_results(output or Path(__file__).with_name('results'))
    machine=dict(system=platform.system(),release=platform.release(),python=platform.python_version(),architecture=platform.machine(),started_utc=datetime.now(timezone.utc).isoformat())
    machine['label']=document.get('run_label','')
    archive=root/'recordings.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr('settings.json',json.dumps(document))
        bundle.writestr('plan.json',json.dumps(plan))
        bundle.writestr('machine.json',json.dumps(machine))
    rows=[];events=[];assets={};errors={}
    with tempfile.TemporaryDirectory(prefix='lsl_results_') as temporary:
        work=Path(temporary)
        _normal_views(root,work,plan,rows,events,assets,machine)
        for i,(_,_,cfg) in enumerate(cases):
            item=plan[i];case=item['case'];item['status']='running'
            directory=work/case;directory.mkdir()
            print(f'[{i+1}/{len(plan)}] {case}: {item["sensors"]}, {item["rates_hz"]} Hz, {item["channels"]} channels',flush=True)
            began=time.monotonic()
            try:
                report=run(cfg,directory,case)
                if not report['tx_complete'] or not report['rx_complete']:raise RuntimeError('Recording ended before both scripts completed')
                _collect_normal(case,cfg,report,rows,events)
                item['status']='completed'
            except Exception as error:
                item['status']='failed';item['error']=str(error);errors[case]=str(error)
                for sensor in cfg['sensors']:rows.append(case_result(case,cfg,sensor,error=str(error)))
                print(f'FAILED: {error}',flush=True)
            finally:
                recording=directory/f'{case}_recording.npz'
                if recording.exists():
                    with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_STORED) as bundle:bundle.write(recording,case+'.npz')
            item['actual_s']=time.monotonic()-began
            for path in directory.glob('*_snapshot.png'):
                assets[path.stem.removesuffix('_snapshot')]=_image_data(path)
            _normal_views(root,work,plan,rows,events,assets,machine)
            import shutil
            shutil.rmtree(directory)
            print(f'{case}: {item["status"]}',flush=True)
        with zipfile.ZipFile(archive,'a') as bundle:bundle.writestr('finished.json',json.dumps(dict(errors=errors)))
        _normal_views(root,work,plan,rows,events,assets,machine,finished=True)
    print(f'Results: {root/"summary.html"}',flush=True)
    if errors:raise RuntimeError('Some recordings failed; see the result table')


def reanalyze_normal(recording,output=None):
    import json,tempfile,zipfile,shutil
    from analysis import analyze,case_result
    from helpers import unpack_recording
    recording=Path(recording)
    root=Path(output) if output else recording.parent
    if output:
        root.mkdir(parents=True,exist_ok=True)
        if any(root.iterdir()):raise ValueError('Reanalysis output must be empty')
    rows=[];events=[];assets={}
    with zipfile.ZipFile(recording) as bundle,tempfile.TemporaryDirectory(prefix='lsl_reanalysis_') as temporary:
        work=Path(temporary)
        document=json.loads(bundle.read('settings.json'))
        plan=json.loads(bundle.read('plan.json'));machine=json.loads(bundle.read('machine.json'))
        by_id={p['case']:p for p in plan}
        errors=json.loads(bundle.read('finished.json')).get('errors',{}) if 'finished.json' in bundle.namelist() else {}
        names=[n for n in bundle.namelist() if n.endswith('.npz')]
        if len(names)!=len(set(names)):raise ValueError('Archive contains duplicate case names')
        for name in names:
            case=Path(name).stem
            if name!=case+'.npz' or case not in by_id:raise ValueError('Unexpected case in recording archive')
            item=by_id[case];directory=work/case;directory.mkdir()
            saved=directory/'recording.npz';saved.write_bytes(bundle.read(name))
            unpack_recording(saved,directory)
            cfg=json.loads((directory/'config.json').read_text())
            if json.loads((directory/'run.json').read_text())['case']!=case:raise ValueError('Recording case ID mismatch')
            try:
                report=analyze(directory,directory,case)
                if not report['tx_complete'] or not report['rx_complete']:raise RuntimeError('Incomplete recording')
                if case in errors:raise RuntimeError(errors[case])
                _collect_normal(case,cfg,report,rows,events)
                item['status']='completed'
            except Exception as error:
                item['status']='failed';item['error']=str(error)
                for sensor in cfg['sensors']:rows.append(case_result(case,cfg,sensor,error=str(error)))
            for path in directory.glob('*_snapshot.png'):
                assets[path.stem.removesuffix('_snapshot')]=_image_data(path)
            shutil.rmtree(directory)
        _normal_views(root,work,plan,rows,events,assets,machine,finished=True)
    if output:shutil.copy2(recording,root/'recordings.zip')
    print(f'Results: {root/"summary.html"}')


def main():
    import json
    parser=argparse.ArgumentParser(description='Record EEG and additional signals; show results before analysis')
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--out',help='Fresh output directory')
    parser.add_argument('--plan-only',action='store_true',help='Show the plan without transmitting or replacing results')
    parser.add_argument('--signals',action='store_true',help='Run normal signal and marker tests')
    parser.add_argument('--local',action='store_true',help='Run separate extra tests with deliberately slow processing')
    parser.add_argument('--reanalyze',help='Rebuild results from recordings.zip or the older buffering recordings.npz')
    args=parser.parse_args()
    if args.local and (args.signals or args.reanalyze):parser.error('--local cannot be combined with --signals or --reanalyze')
    if args.reanalyze:
        if args.signals or args.plan_only:parser.error('--reanalyze cannot be combined with --signals or --plan-only')
        if Path(args.reanalyze).suffix.lower()=='.zip':reanalyze_normal(args.reanalyze,args.out)
        else:
            from buffering import reanalyze
            reanalyze(args.reanalyze,args.out)
        return
    mode=json.loads(Path(args.config).read_text()).get('active_suite','normal')
    if args.signals or (not args.local and mode=='normal'):
        normal_experiment(args.config,args.out,args.plan_only)
    else:
        from buffering import experiment
        experiment(args.config,args.out,args.plan_only,local=args.local)


if __name__=='__main__':
    main()
