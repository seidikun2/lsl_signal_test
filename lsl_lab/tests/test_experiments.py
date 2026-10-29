import runpy
from pathlib import Path

import numpy as np
import pytest

from main import build_cases
from lsl_lab.config import validate, marker_events, sample_count
from lsl_lab.generator import signal
from lsl_lab.analysis import analyze_responses, analyze_marker_classes


def test_two_experiment_plan_and_fixed_stream_parameters():
    settings = runpy.run_path(str(Path(__file__).parents[2]/'settings.py'))
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
    settings = runpy.run_path(str(Path(__file__).parents[2]/'settings.py'))
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
