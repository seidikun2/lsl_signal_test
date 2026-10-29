"""Deterministic synthetic signals; no biological interpretation is implied."""
import numpy as np


def sample_elapsed(nominal, sensor):
    """Invert acquisition phase for a linearly changing frequency offset.

    nominal = scale * elapsed + 0.5 * slope * elapsed**2.
    Rationalized quadratic root also works for zero slope.
    """
    scale = 1+sensor['drift_ppm']*1e-6
    ramp = sensor.get('drift_slope_ppm_s', 0.)*1e-6
    discriminant = scale**2+2*ramp*np.asarray(nominal)
    if np.any(discriminant <= 0):
        raise ValueError('Acquisition rate would reach zero during the experiment')
    return 2*np.asarray(nominal)/(scale+np.sqrt(discriminant))


def pilot(t):
    """Common coded 40-ms pulse every second, beginning at t=0.5 s."""
    event = np.floor(t - 0.5).astype(np.int64)
    phase = t - 0.5 - event
    polarity = np.where(((event * 1103515245 + 12345) >> 4) & 1, 1., -1.)
    return np.where((t >= .5) & (phase < .04), polarity, 0.)


def signal(kind, indices, rate, channels):
    t = np.asarray(indices, dtype=float)[:, None] / rate
    c = np.arange(channels)[None, :]
    # Frequencies remain below Nyquist, even in the 100-Hz sweep.
    if kind.upper() == 'EEG':
        y = 20*np.sin(2*np.pi*(8 + c % 5)*t + .17*c)
        y += 8*np.sin(2*np.pi*min(22., .35*rate)*t + .11*c)
        amplitude = 35.
    elif kind.upper() == 'EMG':
        carrier = min(85., .37*rate)
        y = 70*np.sin(2*np.pi*(carrier-c % 5)*t + .23*c)
        y *= .2 + .8*np.sin(2*np.pi*.7*t)**2
        amplitude = 100.
    elif kind.upper() == 'IMU':
        y = np.sin(2*np.pi*(.7 + .1*(c % 3))*t + .4*c)
        amplitude = 2.
    else:
        raise ValueError(f'Unknown sensor: {kind}')
    return (y + amplitude*pilot(t)).astype(np.float32)
