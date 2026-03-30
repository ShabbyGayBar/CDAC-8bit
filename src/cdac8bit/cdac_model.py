"""
cdac_model.py — Behavioral model of an 8-bit Capacitor DAC (CDAC).

Architecture: Binary-weighted capacitor array
  - 8 binary-weighted capacitors:  bit k  → 2^k × C_unit  (k = 0 … 7)
  - 1 termination capacitor       :         1 × C_unit
  - Total (ideal):                          256 × C_unit

Ideal transfer function:
  Vout = Vref × D / 256   (D ∈ {0, 1, …, 255})

With capacitor mismatch the actual output becomes:
  Vout = Vref × Σ_k (bit_k × C_k_actual) / C_total_actual
"""

import numpy as np


class CDAC8bit:
    """Behavioral model of an 8-bit binary-weighted Capacitor DAC."""

    N_BITS = 8
    N_CODES = 2 ** N_BITS  # 256

    def __init__(self, vref: float = 1.0, mismatch_sigma: float = 0.0,
                 seed=None):
        """
        Initialise the CDAC model.

        Parameters
        ----------
        vref : float
            Reference voltage (V).  Output range is [0, Vref).
        mismatch_sigma : float
            Relative standard deviation of capacitor mismatch.
            0.0 → ideal DAC; 0.01 → 1 % σ mismatch.
        seed : int or None
            Random seed for reproducibility.
        """
        self.vref = float(vref)
        self.mismatch_sigma = float(mismatch_sigma)

        rng = np.random.default_rng(seed)

        # Nominal binary-weighted capacitor values (in units of C_unit)
        # Bit 0 (LSB) = 1 C_unit, …, Bit 7 (MSB) = 128 C_unit
        self.cap_nominal = 2.0 ** np.arange(self.N_BITS)  # shape (8,)

        # Apply mismatch
        if mismatch_sigma > 0.0:
            mismatch = 1.0 + mismatch_sigma * rng.standard_normal(self.N_BITS)
            self.cap_actual = self.cap_nominal * mismatch
            term_mismatch = 1.0 + mismatch_sigma * rng.standard_normal()
            self.cap_term = float(term_mismatch)
        else:
            self.cap_actual = self.cap_nominal.copy()
            self.cap_term = 1.0

        # Total capacitance (sum of all binary caps + termination cap)
        self.cap_total = float(np.sum(self.cap_actual) + self.cap_term)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert(self, digital_codes) -> np.ndarray:
        """
        Convert an array of digital codes to analog output voltages.

        Parameters
        ----------
        digital_codes : array-like of int
            Input digital codes in the range [0, 255].

        Returns
        -------
        np.ndarray
            Output voltages (V), same shape as *digital_codes*.
        """
        codes = np.asarray(digital_codes, dtype=int)
        scalar = codes.ndim == 0
        codes = np.atleast_1d(codes)

        # Decompose each code into 8 bits (bit 0 = LSB)
        bits = (codes[:, None] >> np.arange(self.N_BITS)) & 1  # (N, 8)

        vout = self.vref * (bits @ self.cap_actual) / self.cap_total

        return float(vout[0]) if scalar else vout

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def lsb_ideal(self) -> float:
        """Ideal LSB voltage = Vref / 2^N."""
        return self.vref / self.N_CODES

    @property
    def is_ideal(self) -> bool:
        """True when no mismatch is present."""
        return self.mismatch_sigma == 0.0

    def __repr__(self) -> str:
        return (
            f"CDAC8bit(vref={self.vref}, "
            f"mismatch_sigma={self.mismatch_sigma}, "
            f"cap_total={self.cap_total:.4f})"
        )
