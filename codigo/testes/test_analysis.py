import numpy as np
import pytest

from codigo.analise import compare
from codigo.config import validate
from codigo.receptor import OnlineAligner
from codigo.gerador import signal, pilot


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


def test_signal_is_chunk_independent_and_pilot_common():
    for kind in ('EEG', 'EMG', 'IMU'):
        a = signal(kind, np.arange(200), 100, 3)
        b = np.concatenate([signal(kind, np.arange(i, min(i+17, 200)), 100, 3)
                            for i in range(0, 200, 17)])
        np.testing.assert_array_equal(a, b)
    assert pilot(np.array([.49, .5, .52, .55]))[0] == 0
    assert abs(pilot(np.array([.5]))[0]) == 1


def test_amplitude_corruption_is_detected():
    ids = np.arange(10)
    ts = ids/100
    tx = dict(ids=ids, sample=ts, generated=ts, sent=ts, values=np.zeros((10, 2)))
    rx = dict(ids=ids, raw=ts, corrected=ts, arrival=ts+.01,
              correction=np.zeros(10), values=np.ones((10, 2)))
    r, _ = compare(tx, rx, 100)
    assert r['rmse'] == 1
    assert r['rmse_per_channel'] == [1, 1]


def test_online_aligns_different_rates_without_extrapolation():
    sensors = [dict(name='a', rate=100, channels=1), dict(name='b', rate=200, channels=1)]
    aligner = OnlineAligner(sensors, rate=200, lag=.02)
    for s in sensors:
        t = np.arange(20)/s['rate']
        aligner.add(s['name'], t, (2*t)[:, None])
    t, values = aligner.pull(.2)
    assert t[-1] <= .095 + 1e-9
    np.testing.assert_allclose(values['a'][:, 0], 2*t)
    np.testing.assert_allclose(values['a'], values['b'])
    assert aligner.pull(.2) is None


def test_online_marks_missing_interval():
    a = OnlineAligner([dict(name='a', rate=100, channels=1)], 100, .01)
    t = np.array([0., .01, .05, .06])
    a.add('a', t, t[:, None])
    grid, y = a.pull(.2)
    assert np.isnan(y['a'][(grid > .01) & (grid < .05)]).all()
    assert np.isfinite(y['a'][np.isclose(grid, .01)]).all()


def test_online_backward_clock_correction():
    a = OnlineAligner([dict(name='a', rate=100, channels=1)], 100, .01)
    a.add('a', np.array([0., .01, .02]), np.ones((3, 1)))
    a.add('a', np.array([.015, .025]), np.ones((2, 1))*2)
    assert (np.diff(a.data['a'][0]) > 0).all()


@pytest.mark.parametrize('change', [dict(chunk=0), dict(duration=-1), dict(inlet_buffer=.1),
                                  dict(receiver_pause=-1), dict(drift_ppm=float('nan'))])
def test_bad_config(change):
    with pytest.raises(ValueError):
        validate(dict(sensors=[dict(name='eeg', kind='EEG', rate=100, channels=2)], **change))
