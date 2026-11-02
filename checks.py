from helpers import read_settings
"""Optional regression checks: python -m pytest checks.py -q"""


import numpy as np
import pytest

from analysis import compare
from helpers import validate
from transmission import signal


def test_loss_duplicates_order_drift_and_clock_offset():
    ids = np.arange(1000)
    ts = 100 + ids/1000/(1+200e-6)
    values = signal('EEG', ids, 1000, 4)
    tx = dict(ids=ids, sample=ts, generated=ts+.001, sent=ts+.002, values=values)
    selected = np.r_[np.arange(100), 101, 100, 101, np.arange(103, 1000)]
    c = np.full(len(selected), 7.)
    rx = dict(ids=selected, raw=ts[selected], corrected=ts[selected]+c,
              arrival=ts[selected]+c+.012, correction=c, values=values[selected])
    r, _ = compare(tx, rx, 1000)
    assert r['missing'] == 1  # sample 102
    assert r['duplicates'] == 1
    assert r['reordered'] == 1
    assert r['rmse'] == 0
    assert r['latency_ms']['median'] == pytest.approx(12.)
    assert r['send_to_read_ms']['median'] == pytest.approx(10.)
    assert r['rx_timing_drift_ppm'] == pytest.approx(-199.960008, abs=.001)


def test_signal_is_chunk_independent():
    for kind in ('EEG', 'EMG', 'IMU'):
        a = signal(kind, np.arange(200), 100, 3)
        b = np.concatenate([signal(kind, np.arange(i, min(i+17, 200)), 100, 3)
                            for i in range(0, 200, 17)])
        np.testing.assert_array_equal(a, b)


def test_amplitude_corruption_is_detected():
    ids = np.arange(10)
    ts = ids/100
    tx = dict(ids=ids, sample=ts, generated=ts, sent=ts, values=np.zeros((10, 2)))
    rx = dict(ids=ids, raw=ts, corrected=ts, arrival=ts+.01,
              correction=np.zeros(10), values=np.ones((10, 2)))
    r, _ = compare(tx, rx, 100)
    assert r['rmse'] == 1
    assert r['rmse_per_channel'] == [1, 1]


@pytest.mark.parametrize('change', [dict(chunk=0), dict(duration=-1), dict(inlet_buffer=.1),
                                  dict(receiver_pause=-1), dict(drift_ppm=float('nan'))])
def test_bad_config(change):
    with pytest.raises(ValueError):
        validate(dict(sensors=[dict(name='eeg', kind='EEG', rate=100, channels=2)], **change))


from pathlib import Path

import numpy as np
import pytest

from main import build_cases
from helpers import validate, marker_events, sample_count
from transmission import signal
from analysis import analyze_responses, analyze_marker_classes


def test_two_experiment_plan_and_fixed_stream_parameters():
    settings = read_settings(Path(__file__).with_name('config.json'))
    cases = build_cases(settings)
    eeg = [cfg for experiment, _, cfg in cases if experiment == 'eeg']
    streams = [cfg for experiment, _, cfg in cases if experiment == 'streams']
    assert len(eeg) == 72 and len(streams) == 9
    assert {(c['sensors'][0]['rate'], c['sensors'][0]['channels']) for c in eeg} == {
        (rate, channels) for rate in [500,512,731.5,1000,1500,2048] for channels in [16,32,64,128]}
    assert [len(c['sensors']) for c in streams] == [1,1,1,2,2,2,3,3,3]
    assert streams[0]['sensors'][0] == streams[3]['sensors'][0] == streams[6]['sensors'][0]
    assert streams[3]['sensors'][1] == streams[6]['sensors'][1]
    assert {c['receiver_poll_hz'] for _,_,c in cases}=={2048}
    assert {c['chunk'] for _,_,c in cases}=={1}
    assert [s['rate'] for s in streams[-1]['sensors']]==[512,2000,70]
    assert all(c['shared_sensor_events'] for c in streams)


def test_exact_sample_count_with_different_rates():
    settings = read_settings(Path(__file__).with_name('config.json'))
    settings['TESTS'].update(length_mode='samples', samples_per_stream=257)
    for _, _, cfg in build_cases(settings):
        assert all(sample_count(cfg, sensor) == 257 for sensor in cfg['sensors'])
        assert cfg['duration'] == max(257/s['rate'] for s in cfg['sensors'])


def event_fixture():
    cfg = validate(dict(duration=6, sensors=[dict(name='eeg',kind='EEG',rate=128,channels=2)]))
    sensor = cfg['sensors'][0]
    ids = np.arange(768)
    elapsed = ids/128
    events = marker_events(cfg)
    values = signal('EEG',ids,128,2,events,elapsed)
    tx = dict(ids=ids,sample=100+elapsed,values=values)
    rx = dict(ids=ids.copy(),raw=100+elapsed,corrected=107+elapsed,
              arrival=107+elapsed+.05,values=values.copy())
    txm = [dict(e,sample_time=100+e['nominal'],sent=100+e['nominal']) for e in events]
    rxm = [dict(e,raw=e['sample_time'],corrected=e['sample_time']+7,
                arrival=e['sample_time']+7+.01) for e in txm]
    return cfg, sensor, tx, rx, txm, rxm


def test_periodic_classes_and_waveform_response_detection(tmp_path):
    cfg,sensor,tx,rx,txm,rxm = event_fixture()
    assert len(txm) == 9 and len({e['id'] for e in txm}) == 9
    for label in ('A','B','C'):
        stamps = [e['sample_time'] for e in txm if e['label']==label]
        np.testing.assert_allclose(np.diff(stamps),2)
    result = analyze_responses(cfg,sensor,tx,rx,txm,rxm,tmp_path)
    assert result['expected'] == result['detected'] == 9
    for event in result['events']:
        assert 200-1e-6 <= event['tx_ms'] < 200+1000/128
        assert event['acquisition_error_ms'] == pytest.approx(0,abs=1e-8)
        assert event['delivery_error_ms'] == pytest.approx(40)
    assert result['delivery_mismatch'] == 9


def test_removed_response_is_not_inferred_from_schedule(tmp_path):
    cfg,sensor,tx,rx,txm,rxm = event_fixture()
    mask = (rx['raw'] >= 101.1) & (rx['raw'] < 101.18)
    rx['values'][mask] = signal('EEG',rx['ids'][mask],128,2)
    result = analyze_responses(cfg,sensor,tx,rx,txm,rxm,tmp_path)
    event = next(e for e in result['events'] if e['label']=='B' and e['sequence']==0)
    assert event['status'] == 'missing_response'
    assert event['rx_delivery_ms'] is None


def test_missing_first_response_sample_is_not_a_valid_onset(tmp_path):
    cfg,sensor,tx,rx,txm,rxm = event_fixture()
    rx = {key:value[rx['ids']!=64] for key,value in rx.items()}
    result = analyze_responses(cfg,sensor,tx,rx,txm,rxm,tmp_path)
    event = next(e for e in result['events'] if e['label']=='A' and e['sequence']==0)
    assert event['status'] == 'onset_lost'
    assert event['rx_acquisition_ms'] is None


def test_missing_marker_does_not_fake_period_drift(tmp_path):
    cfg,_,_,_,txm,rxm = event_fixture()
    rxm = [e for e in rxm if not (e['label']=='A' and e['sequence']==1)]
    rxm.append(dict(rxm[0]))
    result = analyze_marker_classes(cfg,txm,rxm,tmp_path)
    assert result['A']['missing'] == 1
    assert result['A']['duplicates'] == 1
    assert result['A']['rx_delivery_interval_ms']['median'] == pytest.approx(4000)
    assert result['A']['rx_delivery_interval_error_ms']['max'] == pytest.approx(0,abs=1e-8)


def test_response_after_stream_end_is_not_counted_as_lost(tmp_path):
    cfg,sensor,tx,rx,txm,rxm = event_fixture()
    keep = tx['ids'] < 32
    tx = {key:value[keep] for key,value in tx.items()}
    rx = {key:value[keep] for key,value in rx.items()}
    result = analyze_responses(cfg,sensor,tx,rx,txm,rxm,tmp_path)
    assert result['expected'] == 0 and result['missing'] == 0
    assert result['outside_recording'] == 9


def test_overlapping_classes_are_explicitly_ambiguous(tmp_path):
    cfg,sensor,tx,rx,txm,rxm = event_fixture()
    txm[1]['sample_time'] = txm[0]['sample_time']
    result = analyze_responses(cfg,sensor,tx,rx,txm,rxm,tmp_path)
    assert result['events'][0]['status'] == 'overlap'
    assert result['events'][1]['status'] == 'overlap'


def test_invalid_marker_spacing_is_rejected():
    with pytest.raises(ValueError,match='interval'):
        validate(dict(sensors=[dict(name='eeg',kind='EEG',rate=128,channels=16)],
                      markers=[dict(label='A',interval=0,first=.3,delay=.2,width=.08,amplitude=1)]))


def test_acquisition_timestamp_response_error_is_measured(tmp_path):
    cfg,sensor,tx,rx,txm,rxm = event_fixture()
    rx['corrected'] += .02
    result = analyze_responses(cfg,sensor,tx,rx,txm,rxm,tmp_path)
    assert result['acquisition_mismatch'] == 9
    assert all(e['acquisition_error_ms']==pytest.approx(20) for e in result['events'])


import numpy as np
import pytest

from analysis import compare
from analysis import temporal_reconstruction, windows, diagnose


def fixture(delay=.2, timestamp_shift=0):
    ids = np.arange(1000)
    t = ids/100
    values = t[:, None]
    ts = 100+t
    tx = dict(ids=ids, sample=ts, generated=ts, sent=ts, values=values)
    rx = dict(ids=ids, raw=ts+timestamp_shift, corrected=ts+7+timestamp_shift,
              arrival=ts+7+delay, correction=np.full(len(ids), 7.), values=values.copy())
    return tx, rx


def test_intact_delayed_values_fail_temporal_comparison():
    tx, rx = fixture(delay=.2)
    integrity, _ = compare(tx, rx, 100)
    assert integrity['rmse'] == 0
    rec, metrics = temporal_reconstruction(tx, rx, 100, origin=100., anchor_offset=7.)
    assert metrics['acquisition']['rmse'] == pytest.approx(0, abs=1e-10)
    assert metrics['delivery']['rmse'] == pytest.approx(.2, abs=.001)
    assert np.isnan(rec['rx_delivery'][0, 0])
    # Neither the RX clock's 7-s epoch offset nor its first arrival is treated as t=0.
    assert rec['time'][np.flatnonzero(np.isfinite(rec['rx_delivery'][:, 0]))[0]] == pytest.approx(.2)


def test_shifted_acquisition_timestamp_is_not_removed():
    tx, rx = fixture(delay=.3, timestamp_shift=.1)
    _, metrics = temporal_reconstruction(tx, rx, 100, 100., 7.)
    assert metrics['acquisition']['rmse'] == pytest.approx(.1, abs=1e-9)
    assert metrics['delivery']['rmse'] == pytest.approx(.3, abs=.001)


def test_chunks_do_not_invent_per_sample_arrival_times():
    from analysis import delivery_hold
    times = np.repeat([.04, .09], 5)
    ids = np.arange(10)
    values = ids[:, None].astype(float)
    reconstructed = delivery_hold(times, values, np.arange(10)/100, ids)[:, 0]
    assert np.isnan(reconstructed[:4]).all()
    np.testing.assert_array_equal(reconstructed[4:9], 4.)
    assert reconstructed[9] == 9


def test_no_interpolation_over_missing_samples():
    tx, rx = fixture(delay=0)
    keep = (rx['ids'] < 100) | (rx['ids'] >= 200)
    rx = {k: v[keep] for k, v in rx.items()}
    rec, _ = temporal_reconstruction(tx, rx, 100, 100., 7.)
    mask = (rec['time'] >= 1.) & (rec['time'] < 2.)
    assert np.isnan(rec['rx_acquisition'][mask]).all()


def test_growing_delay_and_nonlinear_drift_are_flagged():
    tx, rx = fixture(delay=0)
    t = tx['ids']/100
    # Acquisition drift grows with time; transport queue grows independently.
    rx['corrected'] += 50e-6*t**2
    rx['arrival'] += .02*t
    metrics, data = compare(tx, rx, 100)
    trends = windows(data['elapsed'], data['latency'], data['timing_rx'])
    flags, _ = diagnose(metrics, trends)
    codes = {f['code'] for f in flags}
    assert {'drift', 'growing_drift', 'accumulated_drift', 'latency_growth'} <= codes
    assert metrics['rmse'] == 0


def test_time_varying_clock_correction_is_preserved():
    tx, rx = fixture(delay=.2)
    change = np.linspace(0, .02, len(rx['ids']))
    rx['correction'] += change
    rx['corrected'] += change
    _, metrics = temporal_reconstruction(tx, rx, 100, 100., 7.)
    assert metrics['acquisition']['rmse'] > .005


def test_reordered_batch_keeps_latest_acquired_sample():
    from analysis import delivery_hold
    result = delivery_hold(np.array([1., 1., 2.]), np.array([[8.], [7.], [6.]]),
                           np.array([1., 2.]), np.array([8, 7, 6]))
    np.testing.assert_array_equal(result[:, 0], [8, 8])


def test_acquisition_drift_ramp_obeys_frequency_model():
    from helpers import sample_elapsed
    nominal = np.arange(0., 12., .002)
    sensor = dict(drift_ppm=200., drift_slope_ppm_s=1000.)
    elapsed = sample_elapsed(nominal, sensor)
    np.testing.assert_allclose(nominal, 1.0002*elapsed+.5*.001*elapsed**2, atol=1e-12)
    assert (np.diff(elapsed) > 0).all()
    assert np.diff(elapsed)[-1] < np.diff(elapsed)[0]
    fixed = sample_elapsed(nominal, dict(drift_ppm=200.))
    np.testing.assert_allclose(fixed, nominal/1.0002)


def test_reject_drift_ramp_that_stops_acquisition():
    from helpers import validate
    with pytest.raises(ValueError, match='reach zero'):
        validate(dict(duration=12, sensors=[dict(name='eeg', kind='EEG', rate=100,
                      channels=1, drift_slope_ppm_s=-100000)]))


def test_summary_distinguishes_constant_delay_from_acquisition_drift():
    from analysis import case_result
    from helpers import validate
    cfg = validate(dict(sensors=[dict(name='eeg', kind='EEG', rate=100, channels=1)]))
    tx, rx = fixture(delay=.2)
    metrics, data = compare(tx, rx, 100)
    metrics['issues'], metrics['limits'] = diagnose(metrics, windows(data['elapsed'], data['latency'], data['timing_rx']))
    row = case_result('C01', cfg, cfg['sensors'][0], metrics)
    assert row['latency_p99_ms'] == pytest.approx(200.)
    assert row['acquisition_drift_ppm'] == pytest.approx(0., abs=1e-6)
    assert row['delivery_error_ms'] == pytest.approx(0., abs=1e-6)
    assert row['drift_flag'] is False
    assert row['injected_drift_ppm'] == 0
    assert 'latency' in row['issues']


def test_growing_delivery_delay_is_not_reported_as_acquisition_clock_drift(tmp_path):
    import csv
    from analysis import case_result, write_metrics, write_flat_comparison, write_report
    from helpers import validate
    cfg = validate(dict(sensors=[dict(name='eeg', kind='EEG', rate=100, channels=1)]))
    tx, rx = fixture(delay=.2)
    rx['arrival'] += tx['ids']/100 * .02
    metrics, data = compare(tx, rx, 100)
    metrics['issues'], metrics['limits'] = diagnose(metrics, windows(data['elapsed'], data['latency'], data['timing_rx']))
    row = case_result('C01', cfg, cfg['sensors'][0], metrics)
    assert row['drift_flag'] is False
    assert row['delivery_error_ms'] == pytest.approx(199.8)
    assert row['delivery_trend_ppm'] == pytest.approx(20000.)
    assert 'delivery_variation' in row['issues']
    failed = case_result('C02', cfg, cfg['sensors'][0], error='stream unavailable')
    write_metrics(tmp_path,[row,failed])
    write_flat_comparison(tmp_path,[row,failed])
    write_report(tmp_path,[dict(case='C01',test=1,status='completed'),dict(case='C02',test=1,status='failed')],[row,failed],[],[])
    assert (tmp_path/'comparison_01.png').is_file()
    rows = list(csv.DictReader((tmp_path/'metrics.csv').open()))
    assert rows[1]['status'] == 'failed'
    assert rows[1]['acquisition_drift_ppm'] == ''  # missing, never a fictitious zero
    assert 'failed' in (tmp_path/'summary.html').read_text()


def test_recording_roundtrip_preserves_own_timestamps(tmp_path):
    import json
    from helpers import pack_recording,unpack_recording,save_json
    raw=tmp_path/'raw'; raw.mkdir()
    values=np.arange(6,dtype=np.float32).reshape(3,2)
    timestamps=np.array([10.,10.01,10.03])
    np.savez(raw/'rx_eeg-left__pilot.npz',values=values,raw=timestamps,arrival=timestamps+.2)
    save_json(raw/'config.json',{'sensors':['eeg-left__pilot']})
    pack_recording(raw,tmp_path/'case_recording.npz')
    restored=tmp_path/'restored'
    unpack_recording(tmp_path/'case_recording.npz',restored)
    with np.load(restored/'rx_eeg-left__pilot.npz') as f:
        np.testing.assert_array_equal(f['values'],values)
        np.testing.assert_array_equal(f['raw'],timestamps)
        np.testing.assert_array_equal(f['arrival'],timestamps+.2)


def test_startup_plan_lists_all_cases_without_transmitting(tmp_path):
    import subprocess,sys,csv
    output=tmp_path/'results'
    process=subprocess.run([sys.executable,str(Path(__file__).with_name('main.py')),'--signals','--plan-only','--out',str(output)],
                           cwd=Path(__file__).parent,capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stderr
    plan=list(csv.DictReader((output/'plan.csv').open()))
    assert len(plan)==81 and all(p['status']=='planned' for p in plan)
    assert all(float(p['estimate_low_s'])>=float(p['acquisition_s']) for p in plan)
    assert all(float(p['estimate_high_s'])>float(p['estimate_low_s']) for p in plan)
    assert 'T1_72' in process.stdout and 'T2_09' in process.stdout
    assert (output/'summary.html').is_file()
    assert not list(output.rglob('*recording.npz'))
    assert not list(output.rglob('recordings.zip'))


# Queue overload checks use whole executions and actual queue telemetry, not clock drift.
from buffering import build_plan as buffer_plan, fit_windows, summarize as buffer_summary, statistics, sign_test


def buffer_fixture(slow=False, missing=0, drained=True):
    rate = 100
    total = 400
    ids = np.arange(total-missing)
    sample = 10+np.arange(total)/rate
    read = sample[ids]+(.05+.5*ids/rate if slow else .2)
    processed = read+.03
    times = np.arange(.2, 4.2, .1)
    queue = times*50 if slow else np.full(len(times),20.)
    trace = dict(start=np.array(10.), ids=ids, values=signal('EEG',ids,rate,2),
                 raw=sample[ids], read=read, processed=processed, correction=np.array(0.),
                 sent=sample+.02, sample=sample, drained=np.array(drained), remaining=np.array(0 if drained else missing),
                 batches=np.column_stack((10+times,10+times+.03,queue,queue,np.full(len(times),10))))
    case = dict(case='B1', axis='load',value=1.5 if slow else .25,repeat=1,
                rate_hz=rate,channels=2,tx_chunk=10,read_chunk=10,buffer_s=8,duration_s=4,
                processing_s_per_sample=.003,load=.3,expected=total)
    return case, trace


def test_constant_large_delay_is_not_accumulating_backlog():
    case, trace = buffer_fixture()
    row = buffer_summary(case,trace,.5)
    assert row['total_p99_ms'] == pytest.approx(230)
    assert row['queue_slope_ms_s'] == pytest.approx(0,abs=1e-8)
    assert row['added_delay_ms'] == pytest.approx(0,abs=1e-8)
    assert row['state'] == 'no sustained growth'
    assert row['batch_p50_ms']+row['after_send_p50_ms']+row['processing_p50_ms'] == pytest.approx(230)


def test_growing_actual_queue_and_completion_delay_are_detected():
    case, trace = buffer_fixture(slow=True)
    row = buffer_summary(case,trace,.5)
    assert row['state'] == 'growing queue'
    assert row['queue_slope_ms_s'] > 400
    assert row['added_delay_ms'] > 500
    assert row['lost'] == row['rmse'] == 0


def test_unprocessed_tail_at_observation_limit_is_not_labeled_lost():
    case, trace = buffer_fixture(slow=True,missing=100,drained=False)
    row = buffer_summary(case,trace,.5)
    assert row['missing'] == row['remaining'] == 100
    assert row['lost'] is None
    assert row['state'] == 'observation limit'


def test_saturated_flat_queue_with_drops_is_not_reported_as_success():
    case, trace = buffer_fixture(missing=80)
    row = buffer_summary(case,trace,.5)
    assert row['lost'] == 80
    assert row['state'] == 'data loss observed'


def test_drain_phase_cannot_create_a_false_production_trend():
    times = np.arange(0,8,.05)
    queue = np.where(times<4,100,100+1000*(times-4))
    slope, growth, windows = fit_windows(times,queue,4,.5)
    assert slope == pytest.approx(0,abs=1e-8)
    assert growth == pytest.approx(0)
    assert np.max(windows[:,0])<4
    assert fit_windows(times[:5],queue[:5],4,.5)[0] is None


def test_statistical_unit_is_run_pair_and_incomplete_pairs_are_excluded():
    rows=[]
    for rep in range(1,7):
        for slow,value in [(False,.25),(True,1.5)]:
            case,trace=buffer_fixture(slow=slow)
            row=buffer_summary(case,trace,.5)
            row.update(repeat=rep,value=value)
            rows.append(row)
    groups, primary=statistics(rows)
    assert primary['pairs']==6
    assert primary['p']==pytest.approx(.03125)
    assert primary['effect'][1]>0
    assert all(g['n']==6 for g in groups)
    rows[-1]['error']='receiver failed'
    assert statistics(rows)[1]['pairs']==5
    assert sign_test([0,0])==(1.,0)


def test_buffer_plan_varies_one_axis_and_is_reproducibly_randomized():
    cfg=dict(duration_s=4, repeats=6, extra_repeats=3, rate_hz=512, channels=32,
             tx_chunk=16, read_chunk=32, buffer_s=8, loads=[.25,.6,.9,1.05,1.5,3],
             buffers_s=[1,2], rates_hz=[64,256,1024,4096,16384], tx_chunks=[1,128,512],
             rate_sweep_processing_s_per_sample=1/1024, max_drain_s=8, window_s=.5, seed=7301)
    plan=buffer_plan(cfg)
    assert plan==buffer_plan(cfg)
    assert len(plan)==66
    rates=[r for r in plan if r['axis']=='rate']
    assert len({r['processing_s_per_sample'] for r in rates})==1
    assert {r['rate_hz'] for r in rates}=={64,256,1024,4096,16384}
    loads=[r for r in plan if r['axis']=='load']
    assert len({(r['rate_hz'],r['buffer_s'],r['tx_chunk']) for r in loads})==1
    with pytest.raises(ValueError):
        buffer_plan(dict(cfg,read_chunk=0))


def test_reanalysis_uses_embedded_metadata_without_transmission(tmp_path,monkeypatch):
    import json
    import buffering
    case,trace=buffer_fixture()
    row=buffer_summary(case,trace,.5)
    row['state']='sem crescimento sustentado' # older recordings remain readable
    saved=dict(config_json=np.array(json.dumps({'saved_duration':4})),
               metrics_json=np.array(json.dumps([row])))
    saved.update({'B1__'+key:value for key,value in trace.items() if key!='values'})
    recording=tmp_path/'recordings.npz'
    np.savez_compressed(recording,**saved)
    def fail_if_transmitting(*args):
        raise AssertionError('Reanalysis must not acquire new data')
    monkeypatch.setattr(buffering,'record',fail_if_transmitting)
    def inspect_report(root,rows,traces,cfg):
        assert cfg=={'saved_duration':4}
        assert rows[0]['state']=='no sustained growth'
        assert rows[0]['total_p99_ms']==pytest.approx(230)
        np.testing.assert_array_equal(traces['B1']['raw'],trace['raw'])
        (root/'summary.html').write_text('report from saved data')
        return [],{},'saved data verified'
    monkeypatch.setattr(buffering,'report',inspect_report)
    output=tmp_path/'reanalysis'
    buffering.reanalyze(recording,output)
    assert {p.name for p in output.iterdir()}=={'recordings.npz','metrics.csv','timeline.csv','summary.html'}


def local_document():
    import json
    document=json.loads(Path(__file__).with_name('config.json').read_text())
    document['local_tests'].update(duration_s=4,repeats=2,loads=[.8,1.05],window_s=.5)
    document['local_tests'].update(rates_hz=[],stream_eeg_rates_hz=[])
    document['local_tests']['recovery'].update(baseline_s=2,overload_s=4,recovery_s=6,sustain_s=1)
    return document


def test_local_protocol_preserves_existing_sensors_and_has_shared_consumer_cost():
    from buffering import local_plan
    doc=local_document()
    cfg,plan=local_plan(doc)
    assert len(plan)==12
    for rep in (1,2):
        stages=sorted([r for r in plan if r['axis']=='streams' and r['repeat']==rep],key=lambda r:r['value'])
        assert [len(r['sensors']) for r in stages]==[1,2,3]
        assert stages[0]['sensors']==stages[1]['sensors'][:1]==stages[2]['sensors'][:1]
        assert stages[1]['sensors']==stages[2]['sensors'][:2]
        assert stages[2]['sensors']==doc['acquisition']['sensors']
        assert len({r['processing_s_per_sample'] for r in stages})==1
    assert cfg['profile']=='local'


def test_processing_overload_is_removed_by_wall_time_while_transmission_continues():
    from buffering import processing_cost
    case=dict(rate_hz=100,processing_s_per_sample=.006,recovery=dict(baseline_s=10,overload_s=10,baseline_load=.6,overload_load=1.5))
    assert processing_cost(case,9.9)==pytest.approx(.006)
    assert processing_cost(case,10)==pytest.approx(.015)
    assert processing_cost(case,19.9)==pytest.approx(.015)
    assert processing_cost(case,20)==pytest.approx(.006)


def test_recovery_requires_sustained_queue_and_delay_normalization_before_tx_stops():
    from buffering import recovery_metrics
    cfg=dict(baseline_s=10,overload_s=10,recovery_s=20,baseline_load=.6,overload_load=1.5,tolerance_ms=50,sustain_s=2)
    case=dict(duration_s=40,recovery=cfg)
    stamps=np.arange(1.5,46,1)
    q=np.where(stamps<10,0,np.where(stamps<20,(stamps-10)*200,np.maximum(0,2000-(stamps-20)*400)))
    trace=dict(queue_windows=np.column_stack((stamps,q)),delay_windows=np.column_stack((stamps,q+50)))
    result=recovery_metrics(case,trace,1)
    assert result['recovered']
    assert result['recovery_s']==5
    assert result['recovery_confirmed_s']==7
    trace['delay_windows'][:,1]=np.where(stamps<10,50,np.where(stamps<40,500,50))
    assert recovery_metrics(case,trace,1)['recovered'] is False
    trace['queue_windows'][:,1]=0
    trace['delay_windows'][:,1]=50
    result=recovery_metrics(case,trace,1)
    assert not result['overload_observed'] and not result['recovered']


def test_correlated_streams_are_not_pooled_as_independent_repetitions():
    rows=[]
    for rep in range(1,7):
        for stream in ('eeg','emg'):
            case,trace=buffer_fixture()
            row=buffer_summary(case,trace,.5)
            row.update(axis='streams',value=2,stream=stream,run_case=f'L{rep}',repeat=rep)
            rows.append(row)
    groups,primary=statistics(rows)
    assert len(groups)==2 and all(g['n']==6 for g in groups)
    assert primary['pairs']==0 and primary['p'] is None


def test_default_local_plan_preview_never_replaces_existing_results(tmp_path):
    import json
    from buffering import experiment
    doc=local_document()
    doc['active_suite']='local'
    config=tmp_path/'config.json'
    config.write_text(json.dumps(doc))
    out=tmp_path/'existing'
    out.mkdir()
    sentinel=out/'prior.txt'
    sentinel.write_text('preserve me')
    with pytest.raises(ValueError,match='empty'):
        experiment(config,out,plan_only=True)
    assert sentinel.read_text()=='preserve me'


def test_common_grid_preserves_native_rates_and_batch_causality():
    from analysis import temporal_reconstruction
    grids=[]
    for rate in (100,2000,16384):
        ids=np.arange(int(rate*.2)+1)
        times=10+ids/rate
        values=np.sin(2*np.pi*8*ids/rate)[:,None]
        # Every packet becomes available only when its last sample is read.
        reads=10+.02*(ids//max(1,int(rate*.02))+1)
        tx=dict(ids=ids,sample=times,values=values)
        rx=dict(ids=ids,corrected=times,arrival=reads,values=values)
        rec,_=temporal_reconstruction(tx,rx,rate,10,0,grid_rate=16384,until=.2,max_age_s=.1)
        grids.append(rec['time'])
        assert np.isnan(rec['rx_delivery'][rec['time']<.02]).all()
        between=(rec['time']>=.0201)&(rec['time']<.0399)
        assert np.ptp(rec['rx_delivery'][between,0])==0
        assert np.unique(times).size==len(times) # acquisition timestamps stay distinct
    np.testing.assert_array_equal(grids[0],grids[1])
    np.testing.assert_array_equal(grids[1],grids[2])
    assert np.diff(grids[0])==pytest.approx(1/16384)


def test_freshness_uses_acquisition_age_not_time_since_the_read():
    from analysis import delivery_hold
    grid=np.array([1.999,2.,2.1])
    received=delivery_hold(np.array([2.]),np.array([[7.]]),grid,np.array([0]),
                          acquisition_times=np.array([0.]),max_age_s=.25)
    assert np.isnan(received).all() # just delivered, but already two seconds old
    with pytest.raises(ValueError,match='acquisition timestamps'):
        delivery_hold(np.array([2.]),np.array([[7.]]),grid,np.array([0]),max_age_s=.25)


def test_declared_bandwidth_rejects_aliasing_without_using_display_grid_as_acquisition():
    sensor=dict(name='eeg',kind='EEG',rate=100,channels=2,bandwidth_hz=60)
    with pytest.raises(ValueError,match='Nyquist'):
        validate(dict(sensors=[sensor],reconstruction_rate_hz=16384))
    sensor['rate']=256
    assert validate(dict(sensors=[sensor]))['sensors'][0]['rate']==256


def test_optional_rate_sweep_keeps_other_sensor_parameters_fixed():
    from buffering import local_plan
    doc=local_document()
    doc['local_tests'].update(rates_hz=[500,2048],stream_eeg_rates_hz=[500,2048],rate_repeats=2)
    cfg,plan=local_plan(doc)
    assert len(plan)==20
    native=[r for r in plan if r['axis']=='rate_native']
    assert {r['rate_hz'] for r in native}=={500,2048}
    assert all(r['processing_s_per_sample']==0 for r in native)
    mixed=[r for r in plan if r['axis']=='streams_rate']
    for r in mixed:
        assert r['sensors'][0]['rate']==r['value']
        assert r['sensors'][1:]==doc['acquisition']['sensors'][1:]
    assert cfg['reconstruction_rate_hz']==2048


def test_short_preview_preserves_measured_corruption_and_drops_full_waveform():
    from buffering import retain_preview,waveform_preview
    case,trace=buffer_fixture()
    trace['values'][10,0]+=123 # preserve a measured value, never regenerate RX
    value=float(trace['values'][10,0])
    retain_preview(case,trace,.5)
    assert 'values' not in trace
    assert trace['preview_values'][10,0]==pytest.approx(value)
    assert trace['preview_positions'].max()<=51
    rec=waveform_preview(case,trace,dict(snapshot_seconds=.5,reconstruction_rate_hz=100,delivery_max_age_s=.25))
    assert rec['rx_acquisition'][10,0]==pytest.approx(value)


def test_buffered_playback_replays_every_sample_instead_of_only_batch_latest():
    from analysis import buffered_playback,delivery_hold
    ids=np.arange(10)
    acquired=ids/100
    arrival=np.repeat([.04,.09],5)
    values=ids[:,None].astype(float)
    grid=np.arange(20)/100
    replay=buffered_playback(acquired,values,arrival,ids,grid,100,.05)
    np.testing.assert_array_equal(replay[5:15,0],ids)
    assert np.isnan(replay[:5]).all()
    latest=delivery_hold(arrival,values,grid,ids)
    assert set(latest[np.isfinite(latest)])=={4.,9.}


def test_buffered_playback_does_not_use_future_arrivals_or_fill_lost_samples():
    from analysis import buffered_playback
    ids=np.array([0,1,3,4])
    acquired=ids/100
    values=ids[:,None].astype(float)
    grid=np.arange(20)/100
    replay=buffered_playback(acquired,values,np.full(4,.04),ids,grid,100,.05)
    assert np.isnan(replay[7,0]) # target .02 was lost
    assert replay[8,0]==3
    late=buffered_playback(acquired,values,np.full(4,.10),ids,grid,100,.05)
    assert np.isnan(late).all() # entire packet missed its playback deadline
    with pytest.raises(ValueError,match='delay'):
        buffered_playback(acquired,values,np.full(4,.04),ids,grid,100,-1)


def test_buffered_playback_does_not_silently_remove_the_configured_delay():
    from analysis import buffered_playback
    ids=np.arange(100)
    times=ids/100
    values=np.sin(2*np.pi*3*times)[:,None]
    grid=np.arange(120)/100
    output=buffered_playback(times,values,times+.01,ids,grid,100,.2)
    np.testing.assert_allclose(output[20:120],values,atol=1e-10)
    assert np.isnan(output[:20]).all()


def test_legacy_diagnostic_translation_preserves_metrics_and_growth_counts():
    from buffering import english_rows
    case,trace=buffer_fixture(slow=True)
    row=buffer_summary(case,trace,.5)
    row['state']='fila crescente'
    translated=english_rows([row])[0]
    assert translated['state']=='growing queue'
    assert row['state']=='fila crescente' # raw source metadata is left intact
    assert translated['queue_slope_ms_s']==row['queue_slope_ms_s']
    assert statistics([row])[0][0]['growing']==1


def test_report_and_every_plot_label_are_english(tmp_path,monkeypatch):
    import json
    import buffering
    from matplotlib.text import Text
    labels=[]
    def capture_labels(figure):
        labels.extend(t.get_text() for t in figure.findobj(Text))
        return 'data:image/png;base64,'
    monkeypatch.setattr(buffering,'figure_data',capture_labels)
    rows=[];traces={}
    for slow,case_id in [(False,'B1'),(True,'B2')]:
        case,trace=buffer_fixture(slow=slow)
        case['case']=case_id
        row=buffer_summary(case,trace,.5)
        rows.append(row);traces[case_id]=trace
    rows[-1]['state']='fila crescente'
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())['buffer_tests']
    buffering.report(tmp_path,rows,traces,cfg)
    markup=(tmp_path/'summary.html').read_text()
    assert 'lang="en"' in markup
    assert 'Statistical evidence and criteria' in markup
    assert 'growing queue' in markup
    assert 'Queue growth (ms per second)' in labels
    assert 'Total delay p99 (ms)' in labels
    for forbidden in ('aquisição','Atraso','Crescimento','fila crescente','execuções','amostras','Carga'):
        assert forbidden not in markup
        assert all(forbidden not in label for label in labels)


def test_time_counters_preserve_delay_instead_of_aligning_first_arrivals():
    from analysis import recorded_time_vectors
    tx,rx=fixture(delay=.2)
    tx['sender_time_s']=tx['sample']-100
    tx['sender_send_time_s']=tx['sender_time_s']+.01
    rx['sender_time_s']=tx['sender_time_s'].copy()
    rx['receiver_time_s']=rx['arrival']-107
    sent,carried,received=recorded_time_vectors(tx,rx,100,7)
    assert received[0]==pytest.approx(.2) # not reset to zero at first read
    np.testing.assert_array_equal(sent,carried)
    metrics,_=compare(tx,rx,100)
    assert metrics['counter_error_ms']==0
    assert metrics['counter_delay_ms']['median']==pytest.approx(200)
    assert metrics['counter_after_send_ms']['median']==pytest.approx(190)
    rx['sender_time_s'][3]+=.005
    assert compare(tx,rx,100)[0]['counter_error_ms']==pytest.approx(5)


def test_marker_result_contains_actual_sender_and_receiver_time_vectors(tmp_path):
    cfg,sensor,tx,rx,txm,rxm=event_fixture()
    result=analyze_responses(cfg,sensor,tx,rx,txm,rxm,tmp_path,origin=100,offset=7)
    first=result['events'][0]
    assert first['marker_sent_s']==pytest.approx(.3)
    assert first['signal_changed_sent_s']==pytest.approx(.5)
    assert first['marker_received_s']==pytest.approx(.31)
    assert first['signal_changed_received_s']==pytest.approx(.55)
    assert first['signal_sample_received_s']==pytest.approx(.5)
    assert first['delivery_error_ms']==pytest.approx(40)
    assert first['acquisition_error_ms']==pytest.approx(0,abs=1e-7)


def test_heatmap_does_not_turn_failed_or_unrun_cases_into_zero():
    from analysis import result_matrix
    rows=[dict(rate_hz=128,channels=16,status='completed',missing=0,expected=100),
          dict(rate_hz=256,channels=16,status='failed',missing=0,expected=100)]
    matrix,failed=result_matrix(rows,'missing_pct',[128,256,512],[16],'rate_hz','channels')
    assert matrix[0,0]==0
    assert np.isnan(matrix[0,1:]).all()
    np.testing.assert_array_equal(failed,[[False,True,False]])


def test_normal_report_shows_eeg_then_extra_signals_before_optional_analysis(tmp_path):
    from analysis import write_results_report
    plan=[dict(case='T1_01',test=1,status='completed'),dict(case='T2_01',test=2,status='completed')]
    write_results_report(tmp_path,plan,[],[],{},{},{})
    text=(tmp_path/'summary.html').read_text()
    assert text.index('Test 1 — EEG only')<text.index('Test 2 — adding signals')<text.index('After viewing the results')
    assert 'lang="en"' in text
    assert 'imshow' not in text # implementation details stay out of the report
    assert 'Nominal load' not in text and 'wall time' not in text.lower()
    assert text.index('Extra tests — slow processing and recovery')>text.index('Every recorded result')


def test_fixed_live_clock_preserves_delay_and_does_not_drop_recorded_samples():
    from analysis import live_view
    grids=[]
    for rate in (500,2048):
        ids=np.arange(int(rate*.1))
        source=ids/rate
        # Several samples may share a read; the full recording keeps them all.
        reads=np.ceil((source+.004)*2048)/2048
        tx=dict(ids=ids,sample=100+source,generated=100+source,sent=100+source,
                sender_time_s=source,values=ids[:,None])
        rx=dict(ids=ids,raw=100+source,corrected=107+source,arrival=107+reads,
                correction=np.full(len(ids),7.),sender_time_s=source,receiver_time_s=reads,values=ids[:,None])
        grid,sent,available=live_view(tx,rx,100,7,2048,.1,0)
        grids.append(grid)
        assert np.isnan(available[grid<.004]).all()
        present=np.isfinite(available)
        # No value can appear before its actual measured read.
        assert np.all(reads[available[present].astype(int)]<=grid[present]+1e-9)
        metrics,_=compare(tx,rx,rate)
        assert metrics['received']==metrics['expected']==len(ids)
        assert metrics['missing']==0 and metrics['rmse']==0
    np.testing.assert_array_equal(*grids)


def test_shared_event_sync_separates_native_sampling_from_added_timing_error(tmp_path):
    from analysis import stream_timing
    sensors=[dict(name='eeg',kind='EEG',rate=500,channels=1),
             dict(name='emg',kind='EMG',rate=2000,channels=1),
             dict(name='imu',kind='IMU',rate=100,channels=1)]
    cfg=validate(dict(duration=1,sensors=sensors,markers=[
        dict(label='A',interval=2,first=.303,delay=.2,width=.08,amplitude=1)]))
    events=marker_events(cfg)
    txm=[dict(e,sample_time=100+e['nominal'],sent=100+e['nominal']) for e in events]
    rxm=[dict(e,raw=e['sample_time'],corrected=e['sample_time']+7,arrival=e['sample_time']+7+.01) for e in txm]
    combined=[]
    for sensor in sensors:
        rate=sensor['rate'];ids=np.arange(rate);source=100+ids/rate
        values=signal(sensor['kind'],ids,rate,1,events,ids/rate)
        shift=.008 if sensor['name']=='imu' else 0
        tx=dict(ids=ids,sample=source,values=values)
        rx=dict(ids=ids,raw=source,corrected=source+7+shift,arrival=source+7+.02+shift,values=values)
        response=analyze_responses(cfg,sensor,tx,rx,txm,rxm,tmp_path,origin=100,offset=7)
        assert response['detected']==1
        combined.extend(dict(stream=sensor['name'],**e) for e in response['events'])
    measured=stream_timing(combined)
    imu=next(e for e in measured if e['stream']=='imu')
    assert imu['source_offset_ms']==pytest.approx(6)
    assert imu['stored_offset_ms']==pytest.approx(14)
    assert imu['stored_offset_error_ms']==pytest.approx(8)
    assert imu['read_offset_ms']==pytest.approx(14)
    combined[-1]['status']='onset_lost'
    assert stream_timing(combined)[-1]['stored_offset_error_ms'] is None


def test_reading_cadence_counts_empty_reads_and_reports_actual_gaps():
    tx,rx=fixture()
    rx['read_attempt_time_s']=np.r_[np.arange(10)/1000,.020]
    result,_=compare(tx,rx,100)
    assert result['receiver_measured_hz']==pytest.approx(500)
    assert result['read_gap_ms']['max']==pytest.approx(11)


def test_marker_colors_are_shared_between_sent_and_received_styles(tmp_path,monkeypatch):
    import matplotlib.figure
    from analysis import plot_snapshot,MARKER_COLORS
    cfg,sensor,tx,rx,txm,rxm=event_fixture()
    tx['sent']=tx['sample']+.001
    cfg['reconstruction_rate_hz']=2048
    response=analyze_responses(cfg,sensor,tx,rx,txm,rxm,tmp_path,origin=100,offset=7)
    figures=[]
    monkeypatch.setattr(matplotlib.figure.Figure,'savefig',lambda self,*a,**k:figures.append(self))
    plot_snapshot(tmp_path,'T1_01',sensor,cfg,tx,rx,txm,rxm,100,7,response['events'])
    figure=figures[0]
    assert '2048 Hz' in figure.axes[1].get_title()
    labels=[t.get_text() for t in figure.axes[3].get_yticklabels()]
    assert 'A — sent' in labels and 'A — received' in labels
    lines=figure.axes[3].lines
    for color in MARKER_COLORS[:3]:
        segments=[line for line in lines if line.get_color()==color and len(line.get_xdata())==2]
        assert {'-','--'}<={line.get_linestyle() for line in segments}
        assert all(line.get_linewidth()>=2.5 for line in segments)


@pytest.mark.parametrize('rates',[(512,2000,70),(511.7,1999.3,69.8)])
def test_unrelated_sampling_rates_preserve_timing_without_accumulating_delay(tmp_path,rates):
    from analysis import stream_timing
    sensors=[dict(name=name,kind=kind,rate=rate,channels=1)
             for name,kind,rate in zip(('eeg','emg','imu'),('EEG','EMG','IMU'),rates)]
    cfg=validate(dict(duration=12,sensors=sensors))
    events=marker_events(cfg)
    txm=[dict(e,sample_time=100+e['nominal'],sent=100+e['nominal']) for e in events]
    rxm=[dict(e,raw=e['sample_time'],corrected=e['sample_time']+7,arrival=e['sample_time']+7+.01) for e in txm]
    measured=[]
    for sensor in sensors:
        rate=sensor['rate'];ids=np.arange(int(12*rate));source=100+ids/rate
        values=signal(sensor['kind'],ids,rate,1,events,ids/rate)
        tx=dict(ids=ids,sample=source,generated=source,sent=source,values=values)
        rx=dict(ids=ids,raw=source,corrected=source+7,arrival=source+7+.01,
                correction=np.full(len(ids),7.),values=values)
        responses=analyze_responses(cfg,sensor,tx,rx,txm,rxm,tmp_path,origin=100,offset=7)
        assert responses['detected']==18
        measured.extend(dict(stream=sensor['name'],**e) for e in responses['events'])
        metrics,_=compare(tx,rx,rate)
        assert metrics['latency_change_ms']==pytest.approx(0,abs=1e-7)
        assert metrics['missing']==0
    for event in stream_timing(measured):
        stream_rate=next(s['rate'] for s in sensors if s['name']==event['stream'])
        assert -1000/rates[0]-1e-7<=event['source_offset_ms']<=1000/stream_rate+1e-7
        assert event['stored_offset_error_ms']==pytest.approx(0,abs=1e-7)


def write_xdf_fixture(path,streams):
    """Small XDF with real binary chunks, original stamps and clock offsets."""
    import struct
    from xml.sax.saxutils import escape
    def integer(value):return b'\x04'+struct.pack('<I',value)
    def chunk(tag,payload):
        content=struct.pack('<H',tag)+payload
        return integer(len(content))+content
    data=b'XDF:'+chunk(1,b'<info><version>1.0</version></info>')
    for identity,stream in enumerate(streams,1):
        channels=stream['values'].shape[1]
        strings=stream.get('type')=='Markers'
        channel_xml=''.join('<channel><label>'+escape(label)+'</label><unit>uV</unit></channel>' for label in stream.get('labels',[]))
        header=(f'<info><name>{escape(stream["name"])}</name><type>{stream.get("type","EEG")}</type>'
            f'<source_id>{stream.get("source_id","")}</source_id><uid>{stream.get("uid","")}</uid>'
            f'<channel_count>{channels}</channel_count><nominal_srate>{0 if strings else 512}</nominal_srate>'
            f'<channel_format>{"string" if strings else "float32"}</channel_format>'
            f'<desc><channels>{channel_xml}</channels></desc></info>').encode()
        data+=chunk(2,struct.pack('<I',identity)+header)
        data+=chunk(4,struct.pack('<Idd',identity,100.,7.))
        samples=struct.pack('<I',identity)+integer(len(stream['raw']))
        for stamp,values in zip(stream['raw'],stream['values']):
            samples+=b'\x08'+struct.pack('<d',stamp)
            if strings:
                for value in values:
                    encoded=str(value).encode();samples+=integer(len(encoded))+encoded
            else:samples+=np.asarray(values,dtype='<f4').tobytes()
        data+=chunk(3,samples)
    path.write_bytes(data)


def real_capture_fixture(tmp_path):
    import json
    raw=100+np.array([0,.00217,.00403,.00631,.00811])
    values=np.arange(10,dtype=np.float32).reshape(5,2)
    marker_raw=np.array([100.002,100.002,100.006])
    marker_values=np.array([['A'],['B'],['A']])
    meta=dict(mode='real LSL capture',start=107.,end=108.,status='completed',error='',marker_status='recorded',
        streams=dict(eeg=dict(name='test_eego',source_id='device',uid='same-session',channels=2,rate_hz=512,
            labels=['Fz','Cz'],units=['uV','uV']),
            marker=dict(name='PsychoPyMarkers',source_id='psychopy',uid='marker-session',channels=1,rate_hz=0,labels=[],units=[])))
    arrays=dict(eeg_raw=raw,eeg_values=values,eeg_corrected=raw+7,eeg_arrival=raw+7.02,
        marker_raw=marker_raw,marker_values=marker_values,marker_corrected=marker_raw+7,marker_arrival=marker_raw+7.01,
        metadata_json=np.array(json.dumps(meta)))
    recording=tmp_path/'recordings.npz';np.savez_compressed(recording,**arrays)
    streams=[dict(meta['streams']['eeg'],raw=raw,values=values),
             dict(meta['streams']['marker'],type='Markers',raw=marker_raw,values=marker_values)]
    return recording,meta,arrays,streams


def test_reference_matching_uses_overlap_and_preserves_missing_sample_evidence():
    from real_eeg import match_samples,compare_values
    paired=match_samples([1,2,4,5],[0,1,2,3,4,5,6])
    assert paired['captured_in_range']==4 and paired['reference_in_range']==5
    assert paired['unmatched_reference']==1 and paired['unmatched_captured']==0
    captured=np.array([[1.,10],[2,20],[4,40],[5,50]])
    reference=np.column_stack((np.arange(7),10*np.arange(7))).astype(float)
    reference[4,1]+=3
    rows=compare_values(captured,reference,paired,['Fz','Cz'])
    assert rows[0]['different_values']==0 and rows[1]['max_difference']==3
    with pytest.raises(ValueError,match='shared time range'):
        match_samples([1,2],[101,102])
    with pytest.raises(ValueError,match='strictly increasing'):
        match_samples([1,1],[1,2])


def test_real_reference_keeps_nonfinite_values_visible():
    from real_eeg import match_samples,compare_values
    pairs=match_samples([1,2],[1,2])
    rows=compare_values(np.array([[1.,np.nan],[2,np.inf]]),np.array([[1.,np.nan],[3,np.inf]]),pairs,['Fz','Cz'])
    assert rows[0]['different_values']==1 and rows[0]['max_difference']==1
    assert rows[1]['nonfinite_pairs']==2 and rows[1]['max_difference'] is None


def test_xdf_comparison_preserves_original_stamps_values_and_repeated_markers(tmp_path,monkeypatch):
    pytest.importorskip('pyxdf')
    import hashlib
    import real_eeg
    recording,meta,arrays,streams=real_capture_fixture(tmp_path)
    reference=tmp_path/'labrecorder.xdf';write_xdf_fixture(reference,streams)
    loaded,_=real_eeg.load_reference(reference,meta)
    np.testing.assert_array_equal(loaded['eeg']['raw'],arrays['eeg_raw'])
    np.testing.assert_array_equal(loaded['eeg']['values'],arrays['eeg_values'])
    original=hashlib.sha256(recording.read_bytes()).digest()
    monkeypatch.setattr(real_eeg,'_image',lambda fig:'data:image/png;base64,')
    result=real_eeg.compare(recording,reference)
    assert result['paired']['unmatched_captured']==result['paired']['unmatched_reference']==0
    assert all(r['different_values']==0 for r in result['rows'])
    assert len(result['markers']['captured'])==3 and result['markers']['different_values']==0
    assert np.max(np.abs(result['timestamp_difference_ms']))==0
    assert hashlib.sha256(recording.read_bytes()).digest()==original
    report=(tmp_path/'summary.html').read_text()
    assert 'lang="en"' in report and 'No resampling' in report
    assert 'original timestamps' in report and 'physical stimulus timing' in report


@pytest.mark.parametrize('reference_labels',[['Cz','Fz'],['Cz','']])
def test_reference_rejects_changed_channel_order_without_modifying_recording(tmp_path,reference_labels):
    pytest.importorskip('pyxdf')
    from real_eeg import compare
    recording,meta,arrays,streams=real_capture_fixture(tmp_path)
    streams[0]['labels']=reference_labels
    reference=tmp_path/'swapped.xdf';write_xdf_fixture(reference,streams)
    original=recording.read_bytes()
    with pytest.raises(ValueError,match='labels/order differ'):
        compare(recording,reference)
    assert recording.read_bytes()==original
    assert not (tmp_path/'summary.html').exists()


def test_xdf_reference_detects_a_changed_marker_value(tmp_path,monkeypatch):
    pytest.importorskip('pyxdf')
    import real_eeg
    recording,meta,arrays,streams=real_capture_fixture(tmp_path)
    streams[1]['values']=np.array([['A'],['C'],['A']])
    reference=tmp_path/'changed-marker.xdf';write_xdf_fixture(reference,streams)
    monkeypatch.setattr(real_eeg,'_image',lambda fig:'data:image/png;base64,')
    result=real_eeg.compare(recording,reference)
    assert result['markers']['different_values']==1


def test_real_eeg_plan_uses_device_name_without_connecting(tmp_path):
    import subprocess,sys,json
    process=subprocess.run([sys.executable,'main.py','--real-eeg','--plan-only','--duration','3'],
        cwd=Path(__file__).parent,capture_output=True,text=True,timeout=15)
    assert process.returncode==0,process.stderr
    plan=json.loads(process.stdout)
    assert plan['stream_name']=='EE225-020034-000905_on_DESKTOP-U197OKE'
    assert plan['duration_s']==3 and plan['buffer_s']==60


def test_real_capture_interrupt_saves_original_channels_and_timestamps(tmp_path,monkeypatch):
    import json
    import pylsl
    import real_eeg
    import matplotlib.pyplot as plt
    info=pylsl.StreamInfo('interrupt-check','EEG',2,512,'double64','interrupt-source')
    pulls=[];closed=[];clock=[100.]
    class Inlet:
        def __init__(self,selected,**kwargs):assert kwargs['processing_flags']==0
        def open_stream(self,**kwargs):pass
        def time_correction(self,**kwargs):return 7.
        def info(self,**kwargs):return info
        def flush(self):return 0
        def close_stream(self):closed.append(True)
        def pull_chunk(self,**kwargs):
            if pulls:raise KeyboardInterrupt
            pulls.append(True)
            return [[1.123456789,2.],[3.,4.]],[93.01,93.012]
    def tick():clock[0]+=.001;return clock[0]
    monkeypatch.setattr(pylsl,'StreamInlet',Inlet)
    monkeypatch.setattr(pylsl,'resolve_byprop',lambda prop,value,**kwargs:[info] if prop=='name' else [])
    monkeypatch.setattr(pylsl,'resolve_streams',lambda **kwargs:[])
    monkeypatch.setattr(pylsl,'local_clock',tick)
    monkeypatch.setattr(real_eeg.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(real_eeg,'_image',lambda figure:(plt.close(figure) or 'data:image/png;base64,'))
    cfg=json.loads(Path('config.json').read_text())['real_eeg'];cfg['stream_name']='interrupt-check'
    recording=real_eeg.capture(cfg,tmp_path/'capture')
    with np.load(recording,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['eeg_values'],[[1.123456789,2.],[3.,4.]])
        np.testing.assert_array_equal(data['eeg_raw'],[93.01,93.012])
        np.testing.assert_allclose(data['eeg_corrected'],[100.01,100.012])
        assert json.loads(str(data['metadata_json']))['status']=='stopped by user'
    assert closed


def test_real_capture_preserves_existing_results_before_any_connection(tmp_path,monkeypatch):
    import json
    import pylsl
    from real_eeg import capture
    sentinel=tmp_path/'previous.txt';sentinel.write_text('keep previous data')
    def unexpected(*args,**kwargs):raise AssertionError('Existing output must be checked before device discovery')
    monkeypatch.setattr(pylsl,'resolve_byprop',unexpected)
    cfg=json.loads(Path('config.json').read_text())['real_eeg']
    with pytest.raises(ValueError,match='empty'):
        capture(cfg,tmp_path)
    assert sentinel.read_text()=='keep previous data'


def test_xdf_reference_rejects_a_different_outlet_session(tmp_path):
    pytest.importorskip('pyxdf')
    from real_eeg import load_reference
    recording,meta,arrays,streams=real_capture_fixture(tmp_path)
    streams[0]['uid']='different-outlet-session'
    reference=tmp_path/'different-session.xdf';write_xdf_fixture(reference,streams)
    with pytest.raises(ValueError,match='matching eeg stream'):
        load_reference(reference,meta)


def test_real_marker_discovery_does_not_guess_between_multiple_outlets(tmp_path,monkeypatch):
    import json
    import pylsl
    from real_eeg import capture
    eeg=pylsl.StreamInfo('test_eego','EEG',2,512,'double64','device')
    markers=[pylsl.StreamInfo(name,'Markers',1,0,'string',name) for name in ('task1','task2')]
    monkeypatch.setattr(pylsl,'resolve_byprop',lambda *args,**kwargs:[eeg])
    monkeypatch.setattr(pylsl,'resolve_streams',lambda **kwargs:markers)
    cfg=json.loads(Path('config.json').read_text())['real_eeg']
    with pytest.raises(ValueError,match='Multiple marker streams'):
        capture(cfg,tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()
