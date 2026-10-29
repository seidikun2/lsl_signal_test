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


def main():
    parser=argparse.ArgumentParser(description='Run the simulation configured in config.json')
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--out',help='Optional fresh output directory')
    parser.add_argument('--plan-only',action='store_true',help='Show and save the plan without transmitting')
    args=parser.parse_args()
    cases=build_cases(read_settings(args.config))
    root=prepare_results(args.out or Path(__file__).with_name('results'))
    plan=make_plan(cases)
    from analysis import case_result,write_metrics,write_eeg_matrix,write_flat_comparison,write_report
    rows=[]; event_rows=[]; marker_rows=[]
    write_plan(root,plan,image=True,console=True)
    write_report(root,plan,rows,event_rows,marker_rows)
    if args.plan_only: return
    for index,(_,_,cfg) in enumerate(cases):
        item=plan[index]; case=item['case']; directory=root/f'test_{item["test"]}'
        directory.mkdir(exist_ok=True)
        item['status']='running'
        write_plan(root,plan); write_report(root,plan,rows,event_rows,marker_rows)
        print(f'Running {case} ({index+1}/{len(cases)})...',flush=True)
        began=time.monotonic()
        try:
            report=run(cfg,directory,case)
            for sensor in cfg['sensors']:
                metric=report['sensors'][sensor['name']]
                rows.append(case_result(case,cfg,sensor,metric))
                for event in metric.get('responses',{}).get('events',[]):
                    event_rows.append(dict(case=case,test=item['test'],stream=sensor['name'],**event))
            for label,m in report['marker_classes'].items():
                marker_rows.append(dict(case=case,test=item['test'],**{'class':label},expected=m['expected'],received=m['received'],
                    missing=m['missing'],duplicates=m['duplicates'],expected_ms=m['interval_s']*1000,
                    tx_ms=m['tx_interval_ms']['median'],rx_acquisition_ms=m['rx_acquisition_interval_ms']['median'],
                    rx_delivery_ms=m['rx_delivery_interval_ms']['median']))
            item['status']='completed'
        except Exception as error:
            item['status']='failed'; item['error']=str(error)
            for sensor in cfg['sensors']: rows.append(case_result(case,cfg,sensor,error=str(error)))
            print(f'[FAILED] {case}: {error}',flush=True)
        item['actual_s']=time.monotonic()-began
        selected=[r for r in rows if r['experiment']==cfg['experiment']]
        write_metrics(directory,selected)
        write_csv(directory/'events.csv',[r for r in event_rows if r['test']==item['test']])
        write_csv(directory/'markers.csv',[r for r in marker_rows if r['test']==item['test']])
        write_plan(root,plan); write_report(root,plan,rows,event_rows,marker_rows)
        print(f'{case}: {item["status"]} in {item["actual_s"]:.1f}s',flush=True)
    for experiment,number in [('eeg',1),('streams',2)]:
        selected=[r for r in rows if r['experiment']==experiment]
        if selected:
            if experiment=='eeg': write_eeg_matrix(root/f'test_{number}',selected)
            else: write_flat_comparison(root/f'test_{number}',selected)
    write_plan(root,plan,image=True); write_report(root,plan,rows,event_rows,marker_rows)
    print(f'Open {root}/summary.html in a browser',flush=True)
    if any(p['status']=='failed' for p in plan): raise RuntimeError('Some cases failed; see the complete summary')


if __name__=='__main__':
    main()
