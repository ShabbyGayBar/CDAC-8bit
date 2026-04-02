"""
__main__.py — 8-bit CDAC behavioral model: simulation and analysis.

Parts
-----
a) Ideal CDAC, sinusoidal input        → FFT spectrum + THD / SNDR
b) Mismatched CDAC, same input         → FFT spectrum + THD / SNDR (overlay with ideal)
c) Ideal & Mismatched CDAC, ramp input → Voltage Transfer Curve + INL / DNL (individual
                                          plots and overlay)

All plots are saved as SVG files to the  results/  directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend (no display required)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from cdac8bit.cdac_model import CDAC8bit
from cdac8bit.analysis import compute_fft_spectrum, compute_thd_sndr, compute_inl_dnl

# ---------------------------------------------------------------------------
# Global simulation parameters
# ---------------------------------------------------------------------------

VREF = 1.0          # Reference voltage (V)
N_FFT = 4096        # FFT / simulation length (power-of-2 for efficiency)
M_CYCLES = 29       # Number of input sine cycles in N_FFT points.
# 29 is prime → gcd(29, 4096) = 1 (coprime), so the
# signal frequency lands exactly on a single FFT bin
# with no spectral leakage (coherent sampling).
N_HARMONICS = 9     # Harmonics included in THD calculation

MISMATCH_SIGMA = 0.01   # 1 % relative capacitor mismatch (σ)
MISMATCH_SEED = 42      # Fixed seed for reproducibility

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: generate sinusoidal digital code sequence
# ---------------------------------------------------------------------------

def make_sine_codes(n_fft: int, m_cycles: int) -> np.ndarray:
    """
    Return integer digital codes for a full-scale coherent sine wave.

    The sequence has *m_cycles* complete cycles in *n_fft* samples so that
    no spectral leakage occurs in the FFT without windowing.

    Code range: 0 … 255  (8-bit, full scale)
    """
    n = np.arange(n_fft)
    phase = 2.0 * np.pi * m_cycles * n / n_fft
    # Full-scale: centre = 127.5, amplitude = 127.5  → codes in [0, 255]
    codes = np.round(127.5 + 127.5 * np.sin(phase)).astype(int)
    codes = np.clip(codes, 0, 255)
    return codes

# ---------------------------------------------------------------------------
# Part a) Ideal DAC — sinusoidal input, FFT + THD/SNDR
# ---------------------------------------------------------------------------

def part_a(dac_ideal: CDAC8bit, codes: np.ndarray) -> dict:
    """Analyse ideal DAC with sinusoidal input."""
    vout = dac_ideal.convert(codes)

    result = compute_thd_sndr(vout, VREF, fin_bin=M_CYCLES,
                               n_harmonics=N_HARMONICS, window=False)
    spec = result['spectrum']

    print("=" * 60)
    print("Part a)  Ideal 8-bit CDAC — sinusoidal input")
    print("=" * 60)
    print(f"  Fundamental bin  : {M_CYCLES}  (f_in = {M_CYCLES}/{N_FFT})")
    print(f"  THD              : {result['thd_db']:.2f} dB")
    print(f"  SNDR             : {result['sndr_db']:.2f} dB")
    print(f"  ENOB             : {result['enob']:.2f} bits")
    print(f"  Fund. level      : {result['fund_amplitude_db']:.2f} dBFS")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(spec['freqs'], spec['magnitude_dbfs'], lw=0.7, color='steelblue')
    ax.axvline(M_CYCLES / N_FFT, color='tomato', lw=0.8, ls='--',
               label=f'f_in = {M_CYCLES}/{N_FFT}')
    for hb in result['harmonic_bins']:
        ax.axvline(hb / N_FFT, color='orange', lw=0.6, ls=':')

    ax.set_xlim(0, 0.5)
    ax.set_ylim(-100, 10)
    ax.set_xlabel("Normalised frequency  (× f_s)")
    ax.set_ylabel("Amplitude (dBFS)")
    ax.set_title(
        f"THD = {result['thd_db']:.1f} dB   |   "
        f"SNDR = {result['sndr_db']:.1f} dB   |   "
        f"ENOB = {result['enob']:.2f} bits"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which='both', lw=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "part_a_fft_spectrum.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved: {path}")

    return result

# ---------------------------------------------------------------------------
# Part b) Mismatched DAC — sinusoidal input, FFT + THD/SNDR
# ---------------------------------------------------------------------------

def part_b(dac_mismatch: CDAC8bit, codes: np.ndarray) -> dict:
    """Analyse mismatched DAC with sinusoidal input and compare to ideal."""
    vout = dac_mismatch.convert(codes)

    result = compute_thd_sndr(vout, VREF, fin_bin=M_CYCLES,
                               n_harmonics=N_HARMONICS, window=False)
    spec = result['spectrum']

    print()
    print("=" * 60)
    print(f"Part b)  Mismatched CDAC (σ = {MISMATCH_SIGMA*100:.1f} %) — sinusoidal input")
    print("=" * 60)
    print(f"  Fundamental bin  : {M_CYCLES}")
    print(f"  THD              : {result['thd_db']:.2f} dB")
    print(f"  SNDR             : {result['sndr_db']:.2f} dB")
    print(f"  ENOB             : {result['enob']:.2f} bits")
    print(f"  Fund. level      : {result['fund_amplitude_db']:.2f} dBFS")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(spec['freqs'], spec['magnitude_dbfs'], lw=0.7, color='darkorange')
    ax.axvline(M_CYCLES / N_FFT, color='tomato', lw=0.8, ls='--',
               label=f'f_in = {M_CYCLES}/{N_FFT}')
    for hb in result['harmonic_bins']:
        ax.axvline(hb / N_FFT, color='purple', lw=0.6, ls=':')

    ax.set_xlim(0, 0.5)
    ax.set_ylim(-100, 10)
    ax.set_xlabel("Normalised frequency  (× f_s)")
    ax.set_ylabel("Amplitude (dBFS)")
    ax.set_title(
        f"CDAC Mismatch σ = {MISMATCH_SIGMA*100:.1f} %   |   "
        f"THD = {result['thd_db']:.1f} dB   |   "
        f"SNDR = {result['sndr_db']:.1f} dB   |   "
        f"ENOB = {result['enob']:.2f} bits"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which='both', lw=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "part_b_fft_spectrum_mismatch.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved: {path}")

    return result


def part_b_overlay(result_ideal: dict, result_mismatch: dict) -> None:
    """Overlay ideal and mismatched FFT spectra on the same axes.

    Parameters
    ----------
    result_ideal : dict
        Return value of :func:`part_a` (ideal DAC sinusoidal result).
    result_mismatch : dict
        Return value of :func:`part_b` (mismatched DAC sinusoidal result).
    """
    spec_ideal    = result_ideal['spectrum']
    spec_mismatch = result_mismatch['spectrum']

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(spec_mismatch['freqs'], spec_mismatch['magnitude_dbfs'],
            lw=0.8, color='darkorange', alpha=0.85,
            label=f"Mismatch (σ={MISMATCH_SIGMA*100:.1f} %) — SNDR {result_mismatch['sndr_db']:.1f} dB, ENOB {result_mismatch['enob']:.2f}")
    ax.plot(spec_ideal['freqs'], spec_ideal['magnitude_dbfs'],
            lw=0.8, color='steelblue', alpha=0.85,
            label=f"Ideal — SNDR {result_ideal['sndr_db']:.1f} dB, ENOB {result_ideal['enob']:.2f}")
    ax.axvline(M_CYCLES / N_FFT, color='tomato', lw=0.8, ls='--',
               label=f'f_in = {M_CYCLES}/{N_FFT}')

    ax.set_xlim(0, 0.5)
    ax.set_ylim(-100, 10)
    ax.set_xlabel("Normalised frequency  (× f_s)")
    ax.set_ylabel("Amplitude (dBFS)")
    ax.legend(fontsize=8)
    ax.grid(True, which='both', lw=0.3)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "part_b_overlay.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved: {path}")

# ---------------------------------------------------------------------------
# Part c) Ramp input — VTC + INL/DNL (works for any DAC instance)
# ---------------------------------------------------------------------------

_VTC_ZOOM_LO = 140   # first code shown in the VTC inset
_VTC_ZOOM_HI = 150  # last  code shown in the VTC inset (inclusive)


def _add_vtc_inset(ax_vtc, codes, lsb, *step_series):
    """
    Add a zoomed-in inset of the VTC and visually link it to the zoom region
    on the main curve using a bounding box and arrow annotation.
    """

    # --- Zoom region (codes) ---
    mask = (codes >= _VTC_ZOOM_LO) & (codes <= _VTC_ZOOM_HI)
    codes_zoom = codes[mask]
    ideal_zoom = codes_zoom * lsb * 1e3  # mV

    # --- Compute y-limits for the bounding box ---
    y_all = []
    for y_mV, *_ in step_series:
        y_all.append(y_mV[mask])
    y_all = np.concatenate(y_all)

    y_min = y_all.min()
    y_max = y_all.max()
    y_pad = 0.05 * (y_max - y_min)

    # --- Draw bounding box on parent VTC ---
    zoom_box = Rectangle(
        (_VTC_ZOOM_LO, y_min - y_pad),
        _VTC_ZOOM_HI - _VTC_ZOOM_LO,
        (y_max - y_min) + 2 * y_pad,
        edgecolor="black",
        facecolor="none",
        lw=1.0,
        linestyle="--",
        zorder=3,
    )
    ax_vtc.add_patch(zoom_box)

    # --- Create inset axes ---
    ax_ins = ax_vtc.inset_axes([0.7, 0.15, 0.2, 0.5])

    for y_mV, color, lw, alpha, _label in step_series:
        ax_ins.step(
            codes_zoom,
            y_mV[mask],
            where="post",
            color=color,
            lw=lw,
            alpha=alpha,
        )

    ax_ins.plot(
        codes_zoom,
        ideal_zoom,
        color="tomato",
        lw=0.8,
        ls="--",
        label="Ideal",
    )

    ax_ins.set_xlim(_VTC_ZOOM_LO, _VTC_ZOOM_HI)
    ax_ins.set_xlabel("Code", fontsize=6)
    ax_ins.set_ylabel("mV", fontsize=6)
    ax_ins.tick_params(labelsize=5)
    ax_ins.grid(True, lw=0.25)

    # --- Arrow linking zoom box to inset ---
    ax_vtc.annotate(
        "",
        xy=(0.65, 0.4),
        xycoords=ax_vtc.transAxes,  # inset anchor
        xytext=(
            (_VTC_ZOOM_LO + _VTC_ZOOM_HI) / 2,
            y_max + y_pad,
        ),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="->",
            lw=0.8,
            color="black",
        ),
    )


def part_c(dac: CDAC8bit, label: str, filename: str) -> dict:
    """Analyse *dac* with a ramp input: VTC, INL, DNL.

    Parameters
    ----------
    dac : CDAC8bit
        DAC instance to evaluate (ideal or mismatched).
    label : str
        Human-readable label used in plot titles and console output.
    filename : str
        Output SVG filename (without directory prefix).

    Returns
    -------
    dict
        Result from :func:`~cdac8bit.analysis.compute_inl_dnl` with keys:
        ``'codes'``, ``'dnl'``, ``'inl'``, ``'dnl_max'``, ``'inl_max'``,
        ``'lsb_ideal'``, ``'vtc'``.
    """
    codes_ramp = np.arange(CDAC8bit.N_CODES)          # 0 … 255
    vout_ramp = dac.convert(codes_ramp)

    result = compute_inl_dnl(vout_ramp, VREF)
    lsb = result['lsb_ideal']

    print()
    print("=" * 60)
    print(f"Part c)  {label} — ramp input (VTC, INL, DNL)")
    print("=" * 60)
    print(f"  Ideal LSB        : {lsb*1e3:.4f} mV  ({lsb/VREF*100:.4f} % of Vref)")
    print(f"  Peak |DNL|       : {result['dnl_max']:.4f} LSB")
    print(f"  Peak |INL|       : {result['inl_max']:.4f} LSB")

    return result


def part_c_plot(result_ideal: dict, result_mismatch: dict) -> None:
    """Overlay ideal and mismatched VTC / DNL / INL curves on the same axes.

    Parameters
    ----------
    result_ideal : dict
        Return value of :func:`part_c` for the ideal DAC.  Expected keys:
        ``'codes'``, ``'vtc'``, ``'dnl'``, ``'inl'``,
        ``'dnl_max'``, ``'inl_max'``, ``'lsb_ideal'``.
    result_mismatch : dict
        Return value of :func:`part_c` for the mismatched DAC (same keys).
    """
    codes = result_ideal['codes']
    lsb   = result_ideal['lsb_ideal']
    ideal_vout = codes * lsb

    # VTC
    fig, ax_vtc = plt.subplots(figsize=(9, 4))
    ax_vtc.step(codes, result_ideal['vtc'] * 1e3, where='post',
                color='steelblue', lw=1.0, alpha=0.85, label='Ideal CDAC output')
    ax_vtc.step(codes, result_mismatch['vtc'] * 1e3, where='post',
                color='darkorange', lw=1.0, alpha=0.85,
                label=f'Mismatched CDAC  (σ={MISMATCH_SIGMA*100:.1f} %)')
    ax_vtc.plot(codes, ideal_vout * 1e3, color='tomato', lw=0.8,
                ls='--', label='Ideal reference line')
    ax_vtc.set_xlabel("Digital code")
    ax_vtc.set_ylabel("Output voltage (mV)")
    ax_vtc.legend(fontsize=8)
    ax_vtc.grid(True, lw=0.3)
    _add_vtc_inset(ax_vtc, codes, lsb,
                   (result_ideal['vtc'] * 1e3,    'steelblue',  1.0, 0.85, 'Ideal'),
                   (result_mismatch['vtc'] * 1e3, 'darkorange', 1.0, 0.85, 'Mismatch'))

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "part_c_vtc.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved: {path}")

    # DNL
    fig, ax_dnl = plt.subplots(figsize=(9, 4))
    ax_dnl.step(codes[:-1], result_mismatch['dnl'], where='post',
                color='darkorange', lw=0.8, alpha=0.85,
                label=f"Mismatch (peak {result_mismatch['dnl_max']:.4f} LSB)")
    ax_dnl.axhline(0, color='k', lw=0.5)
    ax_dnl.axhline(1, color='tomato', lw=0.6, ls='--', label='+1 LSB')
    ax_dnl.axhline(-1, color='tomato', lw=0.6, ls='--', label='−1 LSB')
    ax_dnl.set_xlabel("Digital code")
    ax_dnl.set_ylabel("DNL (LSB)")
    ax_dnl.legend(fontsize=8)
    ax_dnl.grid(True, lw=0.3)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "part_c_dnl.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved: {path}")

    # INL
    fig, ax_inl = plt.subplots(figsize=(9, 4))
    ax_inl.plot(codes, result_mismatch['inl'], color='darkorange', lw=0.8, alpha=0.85,
                label=f"Mismatch (peak {result_mismatch['inl_max']:.4f} LSB)")
    ax_inl.axhline(0, color='k', lw=0.5)
    ax_inl.axhline(0.5, color='tomato', lw=0.6, ls='--', label='±0.5 LSB')
    ax_inl.axhline(-0.5, color='tomato', lw=0.6, ls='--')
    ax_inl.set_xlabel("Digital code")
    ax_inl.set_ylabel("INL (LSB)")
    ax_inl.legend(fontsize=8)
    ax_inl.grid(True, lw=0.3)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "part_c_inl.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → Saved: {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'='*60}")
    print("8-bit CDAC Behavioral Model  (Python)")
    print(f"{'='*60}")
    print(f"  Vref             : {VREF} V")
    print(f"  FFT length       : {N_FFT}")
    print(f"  Input frequency  : f_in = {M_CYCLES}/{N_FFT} × f_s")
    print(f"  Mismatch σ       : {MISMATCH_SIGMA*100:.1f} %")
    print()

    # Build DAC instances
    dac_ideal    = CDAC8bit(vref=VREF, mismatch_sigma=0.0)
    dac_mismatch = CDAC8bit(vref=VREF, mismatch_sigma=MISMATCH_SIGMA,
                             seed=MISMATCH_SEED)

    # Sinusoidal digital code sequence (shared by parts a & b)
    codes_sine = make_sine_codes(N_FFT, M_CYCLES)

    # Run analyses
    result_a = part_a(dac_ideal, codes_sine)
    result_b = part_b(dac_mismatch, codes_sine)

    print()
    part_b_overlay(result_a, result_b)

    result_c_ideal = part_c(dac_ideal, "Ideal 8-bit CDAC", "part_c_ideal_vtc_inl_dnl.svg")
    result_c_mismatch = part_c(
        dac_mismatch,
        f"Mismatched CDAC  (σ = {MISMATCH_SIGMA*100:.1f} %)",
        "part_c_mismatch_vtc_inl_dnl.svg",
    )
    print()
    part_c_plot(result_c_ideal, result_c_mismatch)

    print()
    print("All results saved to:", os.path.abspath(RESULTS_DIR))


if __name__ == "__main__":
    main()
