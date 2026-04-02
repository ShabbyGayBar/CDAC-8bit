#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/unify:0.7.1": *
#import "@preview/wordometer:0.1.5": total-words, word-count

#show: ieee.with(
  title: [Behavioral Analysis of 8-bit CDACs],
  authors: (
    (
      name: "Brian Li",
      // department: [IME],
      organization: [University of Macau],
      location: [Macau, China],
      email: "brian.li@connect.um.edu.mo",
    ),
  ),
  abstract: [
    This report presents a behavioral analysis of an 8-bit capacitor DAC (CDAC) implemented in Python. The objective is to quantify dynamic performance (FFT, THD, SNDR, ENOB) under ideal conditions and under capacitor mismatch, then relate those dynamic effects to static nonlinearity metrics (VTC, INL, DNL).
  ],
  index-terms: ("DAC", "behavioral analysis", "mismatch", "linearity"),
  // bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

#show: word-count // Start word counting

#set figure(placement: top)

= Introduction

This report presents a behavioral analysis of an 8-bit capacitor DAC (CDAC) implemented in Python. The objective is to quantify dynamic performance (FFT, THD, SNDR, ENOB) under ideal conditions and under capacitor mismatch, then relate those dynamic effects to static nonlinearity metrics (VTC, INL, DNL).

The study is organized in three parts:
- ideal dynamic performance,
- mismatch-induced dynamic degradation,
- static transfer and linearity behavior.

== DAC Architecture

The modeled converter is an 8-bit binary-weighted CDAC with one termination capacitor.

- Binary array:
$C_k = 2^k C_u$, for $k=0,1,dots,7$
- Termination capacitor:
$C_t = 1 dot C_u$
- Ideal total capacitance:
$C_"tot,ideal"=sum_(k=0)^7 2^k C_u + C_t = 256C_u$

For ideal capacitors, the transfer is
$
  V_"out" = V_"ref" dot D/256, D in {0,1,dots,255}.
$

== Behavioral Modeling Assumptions

The following assumptions are used to match the simulation implementation:

- Reference voltage: $V_"ref"=#qty(1, "V")$
- Resolution: 8-bit ($256$ codes)
- FFT length: $N=4096$
- Coherent sine stimulus: $M=29$ cycles in $N$ samples ($f_"in"=29/4096dot f_s$)
- No windowing during reported FFT metrics (coherent sampling removes leakage)
- THD uses harmonics up to 9th order beyond the fundamental
- Mismatch case: Gaussian capacitor mismatch with $sigma=1.0\%$, fixed random seed

= Mismatch Modeling

== Definition of Mismatch

Capacitor mismatch is modeled as an independent Gaussian perturbation on each binary capacitor and the termination capacitor:
$
  C_(k,"actual") = C_(k,"nominal")(1+sigma epsilon_k), epsilon_k tilde.basic N(0,1), sigma=0.01.
$

The output becomes
$
  V_"out" = V_"ref" (sum_k b_k C_(k,"actual"))/C_("tot,actual").
$

= Dynamic Performance Comparison

This section reports the dynamic behavior of the ideal and mismatched 8-bit CDAC driven by a full-scale coherent sinusoidal code sequence.

== FFT Comparison

The FFT spectra of the ideal and mismatched CDAC outputs are overlaid in @fig:fft. The ideal spectrum shows a clean fundamental with low harmonic content, while the mismatched spectrum exhibits elevated harmonics and spurious tones due to nonlinearity.

#figure(
  image("../results/part_b_overlay.svg", width: 100%),
  caption: [Overlay of ideal and mismatched FFT spectra for direct comparison.],
) <fig:fft>

== Performance Metrics

#figure(
  table(
    columns: 3,
    table.header([Metric], [Ideal CDAC], [Mismatched CDAC]),
    [THD (#unit("dB"))], [-77.2], [-55.6],
    [SNDR (#unit("dB"))], [50.0], [47.8],
    [ENOB (#unit("bits"))], [8.01], [7.65],
  ),
  caption: [Summary of dynamic performance metrics for ideal and mismatched CDACs.],
) <tab:dynamic_metrics>

The measured THD, SNDR, and ENOB metrics are listed in @tab:dynamic_metrics.

The ideal CDAC achieves near-theoretical performance, while the mismatched one shows stronger harmonic content and a reduced effective resolution.

= Static Performance

== Voltage Transfer Characteristic

#figure(
  image("../results/part_c_vtc.svg", width: 100%),
  caption: [Voltage transfer characteristic (VTC).],
) <fig:vtc>

The voltage transfer characteristic (VTC) of the ideal and mismatched CDACs is shown in @fig:vtc, which shows code-dependent step variation caused by mismatch, while the ideal output remains uniformly spaced.

== Nonlinearity

#figure(
  image("../results/part_c_dnl.svg", width: 100%),
  caption: [DNL of mismatched CDAC across code range.],
) <fig:dnl>

The differential nonlinearity (DNL) of the mismatched CDAC is plotted in @fig:dnl, showing code-dependent step size variation with some codes exhibiting DNL excursions beyond $plus.minus 0.5$ LSB.

#figure(
  image("../results/part_c_inl.svg", width: 100%),
  caption: [INL of mismatched CDAC across code range.],
) <fig:inl>

The integral nonlinearity (INL) of the mismatched CDAC is plotted in @fig:inl, showing accumulated transfer deviation with some codes exceeding $plus.minus 0.5$ LSB.

Observed static nonlinearity peaks:
- Peak $|"DNL"| = 0.7297"LSB"$
- Peak $|"INL"| = 0.5627"LSB"$

= Conclusion

== Key Observations

- The ideal model achieves expected 8-bit-level dynamic performance (SNDR around #qty(50, "dB"), ENOB around $8$ bits).
- Introducing $1\%$ capacitor mismatch visibly raises harmonic distortion and degrades SNDR/ENOB.
- Static plots confirm nonuniform code step sizes and accumulated transfer deviation under mismatch.

== Sensitivity to Mismatch

This study indicates that CDAC dynamic and static linearity are sensitive to capacitor ratio errors even at moderate mismatch levels. A $1\%$ mismatch can noticeably degrade distortion metrics and produce measurable INL/DNL excursions. In practical designs, capacitor matching, layout symmetry, and calibration are therefore critical to preserving ENOB and linearity targets.

// #total-words

#heading(numbering: none, level: 1)[Disclaimer]

Generative AI was used to assist in debugging code errors and writing this report.
