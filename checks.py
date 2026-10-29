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
    assert len(eeg) == 20 and len(streams) == 3
    assert {(c['sensors'][0]['rate'], c['sensors'][0]['channels']) for c in eeg} == {
        (rate, channels) for rate in [128,256,512,1024,2048] for channels in [16,32,64,128]}
    assert [len(c['sensors']) for c in streams] == [1,2,3]
    assert streams[0]['sensors'][0] == streams[1]['sensors'][0] == streams[2]['sensors'][0]
    assert streams[1]['sensors'][1] == streams[2]['sensors'][1]


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
    process=subprocess.run([sys.executable,str(Path(__file__).with_name('main.py')),'--plan-only','--out',str(output)],
                           cwd=Path(__file__).parent,capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stderr
    plan=list(csv.DictReader((output/'plan.csv').open()))
    assert len(plan)==23 and all(p['status']=='planned' for p in plan)
    assert all(float(p['estimate_low_s'])>=float(p['acquisition_s']) for p in plan)
    assert all(float(p['estimate_high_s'])>float(p['estimate_low_s']) for p in plan)
    assert 'T1_20' in process.stdout and 'T2_03' in process.stdout
    assert (output/'plan.png').is_file() and (output/'summary.html').is_file()
    assert not list(output.rglob('*recording.npz'))
