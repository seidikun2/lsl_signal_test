import numpy as np
import pytest

from lsltest.analysis import compare
from lsltest.reconstruction import temporal_reconstruction, windows, diagnose


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
    from lsltest.reconstruction import delivery_hold
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
    from lsltest.reconstruction import delivery_hold
    result = delivery_hold(np.array([1., 1., 2.]), np.array([[8.], [7.], [6.]]),
                           np.array([1., 2.]), np.array([8, 7, 6]))
    np.testing.assert_array_equal(result[:, 0], [8, 8])


def test_acquisition_drift_ramp_obeys_frequency_model():
    from lsltest.signals import sample_elapsed
    nominal = np.arange(0., 12., .002)
    sensor = dict(drift_ppm=200., drift_slope_ppm_s=1000.)
    elapsed = sample_elapsed(nominal, sensor)
    np.testing.assert_allclose(nominal, 1.0002*elapsed+.5*.001*elapsed**2, atol=1e-12)
    assert (np.diff(elapsed) > 0).all()
    assert np.diff(elapsed)[-1] < np.diff(elapsed)[0]
    fixed = sample_elapsed(nominal, dict(drift_ppm=200.))
    np.testing.assert_allclose(fixed, nominal/1.0002)


def test_reject_drift_ramp_that_stops_acquisition():
    from lsltest.config import validate
    with pytest.raises(ValueError, match='reach zero'):
        validate(dict(duration=12, sensors=[dict(name='eeg', kind='EEG', rate=100,
                      channels=1, drift_slope_ppm_s=-100000)]))
