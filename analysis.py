"""
analysis.py — Signal-analysis utilities for the 8-bit CDAC model.

Provides:
  • compute_fft_spectrum  — FFT of the DAC output and dBFS magnitude spectrum
  • compute_thd_sndr      — THD and SNDR from a sinusoidal output sequence
  • compute_inl_dnl       — INL and DNL from a ramp-driven output sequence
"""

import numpy as np


# ---------------------------------------------------------------------------
# FFT / THD / SNDR
# ---------------------------------------------------------------------------

def compute_fft_spectrum(vout: np.ndarray, vref: float,
                         window: bool = True) -> dict:
    """
    Compute the single-sided magnitude spectrum of a DAC output sequence.

    Parameters
    ----------
    vout : np.ndarray, shape (N,)
        Time-domain output voltage samples.
    vref : float
        Reference voltage used to express the spectrum in dBFS.
    window : bool
        When True a Hann window is applied before the FFT.

    Returns
    -------
    dict with keys:
        'freqs'     — normalised frequency axis  [0 … 0.5]
        'magnitude' — single-sided linear amplitude (V)
        'magnitude_dbfs' — amplitude in dBFS relative to full-scale sine
                           (full-scale sine amplitude = Vref / 2)
    """
    N = len(vout)
    if window:
        w = np.hanning(N)
        cg = np.mean(w)        # coherent gain
        vout_w = vout * w
    else:
        cg = 1.0
        vout_w = vout

    # Two-sided FFT → single-sided amplitude spectrum
    raw = np.fft.rfft(vout_w) / N
    magnitude = np.abs(raw) / cg
    magnitude[1:-1] *= 2          # double-sided → single-sided (not DC/Nyq)

    freqs = np.fft.rfftfreq(N)    # 0 … 0.5

    # Full-scale sine amplitude for 8-bit DAC = Vref/2 (code swing 0…255)
    full_scale_amplitude = vref / 2.0
    with np.errstate(divide='ignore'):
        magnitude_dbfs = 20.0 * np.log10(magnitude / full_scale_amplitude)

    return {
        'freqs': freqs,
        'magnitude': magnitude,
        'magnitude_dbfs': magnitude_dbfs,
    }


def compute_thd_sndr(vout: np.ndarray, vref: float,
                     fin_bin: int, n_harmonics: int = 9,
                     window: bool = True) -> dict:
    """
    Compute THD and SNDR for a sinusoidal DAC output sequence.

    Parameters
    ----------
    vout : np.ndarray, shape (N,)
        Output voltage time series.
    vref : float
        DAC reference voltage.
    fin_bin : int
        FFT bin index of the fundamental input frequency.
        Should equal  M  when coherent sampling is used with M cycles
        in N points.
    n_harmonics : int
        Number of harmonics (beyond the fundamental) to include in THD.
    window : bool
        Apply Hann window before the FFT.

    Returns
    -------
    dict with keys:
        'thd_db'            — THD in dB
        'sndr_db'           — SNDR in dB
        'enob'              — Effective Number Of Bits  = (SNDR - 1.76) / 6.02
        'fund_amplitude_db' — Fundamental amplitude in dBFS
        'harmonic_bins'     — List of harmonic bin indices used
        'spectrum'          — compute_fft_spectrum() result dict
    """
    N = len(vout)
    spec = compute_fft_spectrum(vout, vref, window=window)
    magnitude = spec['magnitude']

    fund_amp = magnitude[fin_bin]
    fund_power = fund_amp ** 2

    # Harmonic power (wrap around Nyquist)
    n_freq_bins = len(magnitude)
    harmonic_bins = []
    harmonic_power = 0.0
    for k in range(2, n_harmonics + 2):
        h_bin = (k * fin_bin) % (n_freq_bins - 1)
        harmonic_bins.append(h_bin)
        harmonic_power += magnitude[h_bin] ** 2

    # THD
    thd_linear = np.sqrt(harmonic_power / fund_power) if fund_power > 0 else 0.0
    thd_db = 20.0 * np.log10(thd_linear) if thd_linear > 0 else -np.inf

    # SNDR: power at fundamental vs. everything else (excluding DC bin 0)
    total_power = np.sum(magnitude[1:] ** 2)  # exclude DC
    noise_distortion_power = total_power - fund_power
    if noise_distortion_power > 0:
        sndr_db = 10.0 * np.log10(fund_power / noise_distortion_power)
    else:
        sndr_db = np.inf
    enob = (sndr_db - 1.76) / 6.02

    # Fundamental amplitude in dBFS
    full_scale_amplitude = vref / 2.0
    fund_amplitude_db = (20.0 * np.log10(fund_amp / full_scale_amplitude)
                         if fund_amp > 0 else -np.inf)

    return {
        'thd_db': thd_db,
        'sndr_db': sndr_db,
        'enob': enob,
        'fund_amplitude_db': fund_amplitude_db,
        'harmonic_bins': harmonic_bins,
        'spectrum': spec,
    }


# ---------------------------------------------------------------------------
# INL / DNL
# ---------------------------------------------------------------------------

def compute_inl_dnl(vout_ramp: np.ndarray, vref: float) -> dict:
    """
    Compute INL and DNL from the DAC output for a ramp (staircase) input.

    The ideal LSB step is  Vref / N_codes.  INL is computed using the
    endpoint method (best-fit straight line through the first and last
    output points).

    Parameters
    ----------
    vout_ramp : np.ndarray, shape (N_codes,)
        DAC output voltages for codes 0, 1, …, N_codes-1.
    vref : float
        Reference voltage.

    Returns
    -------
    dict with keys:
        'codes'         — code array (0 … N_codes-1)
        'dnl'           — DNL in LSB, shape (N_codes-1,)
        'inl'           — INL in LSB, shape (N_codes,)
        'dnl_max'       — peak-to-peak DNL magnitude (LSB)
        'inl_max'       — peak-to-peak INL magnitude (LSB)
        'lsb_ideal'     — ideal LSB voltage (V)
        'vtc'           — voltage transfer curve (Vout vs. code)
    """
    N = len(vout_ramp)
    lsb_ideal = vref / N

    # DNL: deviation of each step from the ideal LSB
    steps = np.diff(vout_ramp)
    dnl = steps / lsb_ideal - 1.0          # shape (N-1,)

    # INL: endpoint method — subtract a straight line from vout[0] to vout[-1]
    ideal_line = np.linspace(vout_ramp[0], vout_ramp[-1], N)
    inl = (vout_ramp - ideal_line) / lsb_ideal   # shape (N,)

    codes = np.arange(N)

    return {
        'codes': codes,
        'dnl': dnl,
        'inl': inl,
        'dnl_max': float(np.max(np.abs(dnl))),
        'inl_max': float(np.max(np.abs(inl))),
        'lsb_ideal': lsb_ideal,
        'vtc': vout_ramp,
    }
