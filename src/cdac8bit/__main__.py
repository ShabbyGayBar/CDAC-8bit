"""
__main__.py — 8-bit CDAC behavioral model: simulation and analysis.

Parts
-----
a) Ideal CDAC, sinusoidal input  → FFT spectrum + THD / SNDR
b) Mismatched CDAC, same input   → FFT spectrum + THD / SNDR (comparison)
c) Ideal CDAC, ramp input        → Voltage Transfer Curve + INL / DNL

All plots are saved to the  results/  directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend (no display required)
import matplotlib.pyplot as plt

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
    ax.set_ylim(-160, 10)
    ax.set_xlabel("Normalised frequency  (× f_s)")
    ax.set_ylabel("Amplitude (dBFS)")
    ax.set_title(
        f"Part a)  Ideal CDAC — FFT Spectrum\n"
        f"THD = {result['thd_db']:.1f} dB   |   "
        f"SNDR = {result['sndr_db']:.1f} dB   |   "
        f"ENOB = {result['enob']:.2f} bits"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which='both', lw=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "part_a_fft_spectrum.png")
    fig.savefig(path, dpi=150)
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
    ax.set_ylim(-160, 10)
    ax.set_xlabel("Normalised frequency  (× f_s)")
    ax.set_ylabel("Amplitude (dBFS)")
    ax.set_title(
        f"Part b)  Mismatched CDAC (σ = {MISMATCH_SIGMA*100:.1f} %) — FFT Spectrum\n"
        f"THD = {result['thd_db']:.1f} dB   |   "
        f"SNDR = {result['sndr_db']:.1f} dB   |   "
        f"ENOB = {result['enob']:.2f} bits"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which='both', lw=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "part_b_fft_spectrum_mismatch.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {path}")

    return result


def part_b_comparison(result_ideal: dict, result_mismatch: dict) -> None:
    """Side-by-side FFT comparison of ideal vs mismatched DAC."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

    for ax, res, label, color in zip(
        axes,
        [result_ideal, result_mismatch],
        ["Ideal CDAC", f"Mismatched CDAC  (σ = {MISMATCH_SIGMA*100:.1f} %)"],
        ["steelblue", "darkorange"],
    ):
        spec = res['spectrum']
        ax.plot(spec['freqs'], spec['magnitude_dbfs'], lw=0.7, color=color)
        ax.axvline(M_CYCLES / N_FFT, color='tomato', lw=0.8, ls='--')
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-160, 10)
        ax.set_xlabel("Normalised frequency  (× f_s)")
        ax.set_ylabel("Amplitude (dBFS)")
        ax.set_title(
            f"{label}\n"
            f"THD = {res['thd_db']:.1f} dB   |   "
            f"SNDR = {res['sndr_db']:.1f} dB   |   "
            f"ENOB = {res['enob']:.2f} bits"
        )
        ax.grid(True, which='both', lw=0.3)

    fig.suptitle("FFT Spectrum Comparison: Ideal vs Mismatched CDAC", y=1.02)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "part_b_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved: {path}")

# ---------------------------------------------------------------------------
# Part c) Ideal DAC — ramp input, VTC + INL/DNL
# ---------------------------------------------------------------------------

def part_c(dac_ideal: CDAC8bit) -> dict:
    """Analyse ideal DAC with ramp input: VTC, INL, DNL."""
    codes_ramp = np.arange(CDAC8bit.N_CODES)          # 0 … 255
    vout_ramp = dac_ideal.convert(codes_ramp)

    result = compute_inl_dnl(vout_ramp, VREF)
    lsb = result['lsb_ideal']

    print()
    print("=" * 60)
    print("Part c)  Ideal 8-bit CDAC — ramp input (VTC, INL, DNL)")
    print("=" * 60)
    print(f"  Ideal LSB        : {lsb*1e3:.4f} mV  ({lsb/VREF*100:.4f} % of Vref)")
    print(f"  Peak |DNL|       : {result['dnl_max']:.4f} LSB")
    print(f"  Peak |INL|       : {result['inl_max']:.4f} LSB")

    # --- Voltage Transfer Curve (VTC) ---
    fig, (ax_vtc, ax_dnl, ax_inl) = plt.subplots(
        3, 1, figsize=(10, 9), constrained_layout=True
    )
    fig.suptitle("Part c)  Ideal 8-bit CDAC — Ramp Input Analysis")

    # VTC
    ideal_vout = codes_ramp * lsb   # ideal staircase
    ax_vtc.step(codes_ramp, vout_ramp * 1e3, where='post',
                color='steelblue', lw=1.0, label='CDAC output')
    ax_vtc.plot(codes_ramp, ideal_vout * 1e3, color='tomato', lw=0.8,
                ls='--', label='Ideal line')
    ax_vtc.set_xlabel("Digital code")
    ax_vtc.set_ylabel("Output voltage (mV)")
    ax_vtc.set_title("Voltage Transfer Curve (VTC)")
    ax_vtc.legend(fontsize=8)
    ax_vtc.grid(True, lw=0.3)

    # DNL
    ax_dnl.step(codes_ramp[:-1], result['dnl'], where='post',
                color='darkorange', lw=0.8)
    ax_dnl.axhline(0, color='k', lw=0.5)
    ax_dnl.axhline(1, color='tomato', lw=0.6, ls='--', label='+1 LSB')
    ax_dnl.axhline(-1, color='tomato', lw=0.6, ls='--', label='−1 LSB')
    ax_dnl.set_xlabel("Digital code")
    ax_dnl.set_ylabel("DNL (LSB)")
    ax_dnl.set_title(f"Differential Non-Linearity (DNL)   peak = {result['dnl_max']:.4f} LSB")
    ax_dnl.legend(fontsize=8)
    ax_dnl.grid(True, lw=0.3)

    # INL
    ax_inl.plot(codes_ramp, result['inl'], color='purple', lw=0.8)
    ax_inl.axhline(0, color='k', lw=0.5)
    ax_inl.axhline(0.5, color='tomato', lw=0.6, ls='--', label='±0.5 LSB')
    ax_inl.axhline(-0.5, color='tomato', lw=0.6, ls='--')
    ax_inl.set_xlabel("Digital code")
    ax_inl.set_ylabel("INL (LSB)")
    ax_inl.set_title(f"Integral Non-Linearity (INL)   peak = {result['inl_max']:.4f} LSB")
    ax_inl.legend(fontsize=8)
    ax_inl.grid(True, lw=0.3)

    path = os.path.join(RESULTS_DIR, "part_c_vtc_inl_dnl.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {path}")

    return result

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
    part_b_comparison(result_a, result_b)

    part_c(dac_ideal)

    print()
    print("All results saved to:", os.path.abspath(RESULTS_DIR))


if __name__ == "__main__":
    main()
