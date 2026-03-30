"""
cdac8bit — Behavioral model of an 8-bit capacitor-based DAC.

Modules
-------
cdac_model  — CDAC8bit class (ideal and mismatched)
analysis    — FFT, THD/SNDR, INL/DNL helper functions

Run the full simulation with::

    python -m cdac8bit
"""

from cdac8bit.cdac_model import CDAC8bit
from cdac8bit.analysis import (
    compute_fft_spectrum,
    compute_thd_sndr,
    compute_inl_dnl,
)

__all__ = [
    "CDAC8bit",
    "compute_fft_spectrum",
    "compute_thd_sndr",
    "compute_inl_dnl",
]
