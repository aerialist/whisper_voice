"""
Compute a smooth, minimum-phase inverse-EQ curve from parallel recordings.
Assumes each pair (y_throat, y_air) is time-aligned (or roughly; we’ll fine-align).

here’s a practical way to “calibrate” an inverse-EQ from your parallel recordings (throat + air), and then turn that curve into a minimum-phase, low-latency filter you can drop into your real-time ML (or DSP) pipeline.

⸻

What we’re doing (high level)
	1.	Estimate the average spectral ratio air/throat across many aligned, parallel utterances.
	•	Work in the log-magnitude domain (robust) and average per frequency bin over time/files.
	•	Weight frames (e.g., by voiced/VAD and energy) to ignore silence.
	2.	Smooth the curve in log-frequency (e.g., ~1/6–1/3 octave smoothing) and limit boost/cut (e.g., cap to +12–15 dB highs, avoid LF boosts).
	3.	Make it minimum-phase (so it’s causal with minimal group delay):
	•	Convert the desired |H(ω)| to a real cepstrum, zero the negative quefrency, double the positive part → min-phase cepstrum → exp(FFT) → impulse response.
	•	Window/truncate to a small FIR (e.g., 64–256 taps), or fit a few biquads (shelves/peaks) if you prefer IIR.
	4.	Bake the filter into your runtime as a fixed InverseEQ block right before the ML model (or as a pre-emphasis before DSP).

Notes & gotchas
	•	Frame weighting matters. If you include a lot of silence or breaths, the ratio is noisy. Use VAD or energy weighting (and maybe emphasize voiced frames) when averaging.
	•	Smoothing in log-frequency avoids “comb” artifacts and keeps the filter small/stable. 1/6–1/3 octave is a good start.
	•	Don’t over-boost the top end. Cap boosts (e.g., +12–15 dB) and gently taper >8–10 kHz to avoid hissing.
	•	Minimum-phase vs linear-phase. Linear-phase FIR adds ~N/2 latency; minimum-phase keeps delay minimal for the same magnitude match—perfect for real-time.
	•	IIR alternative. If you’d rather use biquads:
	•	Sample the target H_dB(f) on a grid.
	•	Fit a small set (e.g., 1–2 high shelves + 2–4 peaking EQs) by least squares on dB magnitude.
	•	Convert each to RBJ biquad coefficients, cascade → inherently min-phase with near-zero latency.

"""

import numpy as np
# import librosa, scipy.signal as sps

SR_FEAT   = 16000
NFFT      = 2048
HOP       = 160                  # 10 ms @16k
WIN       = 400                  # 25 ms @16k
F_LO, F_HI = 80, 8000            # band of interest
BOOST_MAX_DB = +14.0
CUT_MAX_DB   = -6.0

def fine_align(y_t, y_a):
    """
    Optional: GCC-PHAT or xcorr-based coarse alignment, then trim.
    Return time-aligned (y_t, y_a) at SR_FEAT.
    """
    # y_t = librosa.resample(y_t, sr_t, SR_FEAT); y_a = librosa.resample(y_a, sr_a, SR_FEAT)
    # shift = gcc_phat_offset(y_t, y_a); y_t, y_a = shift_signals(y_t, y_a, shift)
    return y_t, y_a

def stft_mag(y):
    # S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP, win_length=WIN, window="hann"))
    S = ...  # [F, T] magnitude
    return S

def vad_weight(S):
    """
    Simple per-frame weights to ignore silence: energy + voiced bias.
    """
    # energy per frame
    e = np.sqrt(np.maximum(1e-12, (S**2).sum(axis=0)))
    e /= (e.max() + 1e-9)
    # voiced proxy: low-harmonics dominance (bins ~70–300 Hz)
    freqs = np.linspace(0, SR_FEAT/2, S.shape[0])
    vband = (freqs >= 70) & (freqs <= 300)
    vscore = (S[vband, :].mean(axis=0) / (S.mean(axis=0) + 1e-9))
    vscore = np.clip(vscore, 0, 1)
    w = 0.7*vscore + 0.3*e
    return w  # [T]

def accumulate_ratio(pairs):
    """
    Accumulate robust average log-spectrum ratio: log(|A|) - log(|T|).
    Returns freq axis and mean_log_ratio[f].
    """
    num = None; den = None  # for weighted mean in log domain we keep sums
    count = None

    for (pth_t, pth_a) in pairs:
        y_t, sr_t = ..., ...
        y_a, sr_a = ..., ...
        # resample + align
        y_t, y_a = fine_align(y_t, y_a)

        S_t = stft_mag(y_t)  # [F,T]
        S_a = stft_mag(y_a)

        T = min(S_t.shape[1], S_a.shape[1])
        S_t = S_t[:, :T] + 1e-12
        S_a = S_a[:, :T] + 1e-12

        # per-frame log spectral ratio
        logR = np.log(S_a) - np.log(S_t)  # [F,T]

        w = vad_weight(S_t)[:T]           # [T]
        w = w / (w.mean() + 1e-9)

        # weighted sum/normalization across time
        if num is None:
            num = (logR * w[None, :]).sum(axis=1)
            count = w.sum()
        else:
            num += (logR * w[None, :]).sum(axis=1)
            count += w.sum()

    mean_log_ratio = num / (count + 1e-9)  # [F]
    freqs = np.linspace(0, SR_FEAT/2, len(mean_log_ratio))
    return freqs, mean_log_ratio

# Smooth in log-frequency and constrain
def log_freq_smooth(freqs, logH, octave_bw=1/6):
    """
    Smooth log-magnitude curve in LOG-frequency using a Gaussian kernel
    whose sigma is 'octave_bw' octaves.
    """
    # Build log-f axis
    f = np.maximum(1.0, freqs)
    lf = np.log2(f)

    # Regular grid in log-f (better smoothing behavior)
    lf_grid = np.linspace(lf.min(), lf.max(), len(lf))
    logH_grid = np.interp(lf_grid, lf, logH)

    # Gaussian in octaves
    sigma = octave_bw / np.sqrt(8*np.log(2))  # convert FWHM→sigma if you like
    # build kernel over, say, ±3*sigma
    kx = np.linspace(-3*sigma, +3*sigma, 201)
    kg = np.exp(-0.5*(kx/sigma)**2); kg /= kg.sum()

    logH_s = np.convolve(logH_grid, kg, mode='same')

    # map back to linear freq bins
    logH_out = np.interp(lf, lf_grid, logH_s)
    return logH_out

def constrain_band(freqs, logH, f_lo=F_LO, f_hi=F_HI, ref_hz=1000):
    """
    1) Zero out correction below f_lo and gently taper above f_hi.
    2) Re-anchor gain so that H(ref_hz)=0 dB (no net loudness change).
    3) Clamp boost/cut.
    """
    H_db = 20/np.log(10) * logH  # ln→dB
    # mute outside band with tapers
    taper_lo = 1.0 / (1.0 + np.exp( -(freqs - f_lo)/20.0 ))     # ~soft step
    taper_hi = 1.0 / (1.0 + np.exp(  (freqs - f_hi)/300.0 ))    # roll-off after f_hi
    w = taper_lo * taper_hi
    H_db *= w

    # anchor at reference
    ref = np.argmin(np.abs(freqs - ref_hz))
    H_db -= H_db[ref]

    # clamp extremes
    H_db = np.clip(H_db, CUT_MAX_DB, BOOST_MAX_DB)

    return (np.log(10)/20.0) * H_db  # back to ln(|H|)

# Minimum-phase FIR from magnitude
def minphase_fir_from_logmag(logH, n_taps=129):
    """
    Build a minimum-phase FIR that approximates |H| defined on the NFFT/2+1 grid.
    Steps:
      1) Make full-spectrum log magnitude (even real spectrum).
      2) Real cepstrum via IFFT(log |H|).
      3) Minimum-phase cepstrum: c_min[0]=c[0]; c_min[n>0]*=2; c_min[n<0]=0.
      4) Exp(FFT(c_min)) → complex spectrum; IFFT → impulse response.
      5) Window/truncate to n_taps.
    """
    # 1) mirror to full FFT size (positive+negative freqs)
    # Our logH is length K = NFFT//2 + 1 on [0..π]
    K = len(logH)
    N = (K - 1) * 2
    log_spec = np.zeros(N, dtype=np.float64)
    log_spec[:K] = logH
    log_spec[K:] = logH[1:-1][::-1]   # even symmetry (real-valued impulse)

    # 2) real cepstrum
    c = np.fft.ifft(log_spec).real

    # 3) minimum-phase cepstrum
    c_min = np.zeros_like(c)
    c_min[0] = c[0]
    c_min[1: N//2] = 2.0 * c[1: N//2]
    # c_min[N//2] left at 0 for even N

    # 4) spectrum → impulse response
    H_min = np.exp(np.fft.fft(c_min))        # complex spectrum with min-phase
    h = np.fft.ifft(H_min).real              # impulse response (long)

    # 5) take first n_taps (min-phase is already causal and front-loaded)
    # apply a mild window to control ripple
    win = np.kaiser(n_taps, beta=5.0)
    h_min = h[:n_taps] * win
    # normalize DC or reference gain
    h_min /= np.sum(h_min) + 1e-12
    return h_min  # FIR taps (min-phase, small latency)

# Putting it together (offline → saved filter)
def build_inverse_eq(pairs, octave_bw=1/6, n_taps=129):
    freqs, mean_logR = accumulate_ratio(pairs)       # ln(|A|) - ln(|T|)
    logH = mean_logR
    logH = log_freq_smooth(freqs, logH, octave_bw)
    logH = constrain_band(freqs, logH, F_LO, F_HI, ref_hz=1000)

    fir = minphase_fir_from_logmag(logH, n_taps=n_taps)
    np.save("inverse_eq_minphase.npy", fir)
    # optionally also save freqs/H_db for documentation/plots
    return fir

# Real-time usage (drop-in)
# Use your favorite fast FIR (overlap-save, or direct if short):
class InverseEQRuntime:
    def __init__(self, fir_taps):
        self.h = fir_taps.astype(np.float32)
        self.zi = np.zeros(len(self.h)-1, dtype=np.float32)  # FIR state

    def process(self, x):
        """
        Block FIR (direct form). For larger blocks, use overlap-save FFT FIR.
        """
        # y, self.zi = sps.lfilter(self.h, [1.0], x, zi=self.zi)  # real code
        y = ...  # replace with your vectorized FIR
        return y

# Place this right before your ML front end (feature extraction) or as the first step of your DSP chain:
inv_eq = InverseEQRuntime(np.load("inverse_eq_minphase.npy"))

def realtime_block(x_in):  # mono float32 block
    x_eq = inv_eq.process(x_in)
    # → proceed to throat→air MLP features, or your DSP stages
    return x_eq