"""
Pseudo-Python for TRADITIONAL DSP enhancement of a throat microphone
Goal: make throat-mic audio sound closer to an air mic in real time (≤100 ms)

here’s a clean, “traditional DSP” real-time pipeline in pseudo-Python. it shows how to open the throat-mic as input, process in short blocks (low latency), and play to an output device. the DSP chain uses only classic tools: HPF/DC removal, pre-emphasis, inverse-EQ (static + light dynamic), LPC-style formant enhancement, fricative/noise injection for unvoiced regions, optional Wiener denoise, AGC, and a safety limiter.

notes & tuning tips
	•	calibration: if you have parallel recordings (throat + air), compute an average inverse EQ curve (air/throat) and bake it into InverseEQ. Keep it smooth/min-phase so it’s stable and low latency.
	•	latency: blocksize 256–512 samples at 44.1 kHz keeps you in the ~6–12 ms per block range; total round-trip usually 15–40 ms depending on driver.
	•	voicing/fricatives: the simple V/UV detector and noise synth are placeholders. Even these small tricks help a lot for “s/ʃ/f/h” audibility.
	•	LPC step: classic formant “sharpening” is lightweight and improves vowel naturalness. Swap in a real LPC implementation when you wire this up.
	•	denoise: throat mics are naturally noise-rejecting, so keep Wiener conservative or off to avoid musical noise.
	•	Cortex-M4 port: replace NumPy/FFT bits with CMSIS-DSP (fixed-point), skip LPC if needed, and keep the chain: HPF → shelf EQ → multiband gain → light fricative noise → AGC/limiter.

Libraries you would actually use:
  numpy, scipy.signal, sounddevice (or PyAudio), collections.deque

This is STRUCTURE + math placeholders.
Replace `...` with your concrete implementation.
"""

import numpy as np
import sounddevice as sd
from scipy.signal import lfilter, iirfilter, butter, sosfilt, sosfilt_zi, get_window, fftconvolve

# -----------------------------
# Utility: stateful IIR filter
# -----------------------------
class IIR:
    def __init__(self, sos):
        self.sos = sos
        self.zi  = np.array([sosfilt_zi(s) for s in sos])  # per-section states

    def process(self, x):
        y = x
        for sec, zi in zip(self.sos, self.zi):
            y, zf = lfilter(sec[:3], sec[3:], y, zi=zi * y[0])
            zi[:] = zf
        return y

# -----------------------------
# Blocks: detectors & helpers
# -----------------------------
def db(x, eps=1e-12): return 20*np.log10(np.maximum(eps, x))

def short_time_energy(x):
    return np.sqrt(np.mean(x**2) + 1e-12)

def spectral_flatness(mag):
    gmean = np.exp(np.mean(np.log(mag + 1e-12)))
    amean = np.mean(mag + 1e-12)
    return gmean / amean

def zero_crossings(x):  # crude voicing hint
    return np.mean(np.abs(np.diff(np.signbit(x))))

def voiced_unvoiced_mask(block, fs):
    """
    Simple voiced/unvoiced decision using:
      - energy in F0 band (70–300 Hz)
      - spectral flatness
      - zero-crossing rate
    Returns scalar in [0,1]: 1=voiced, 0=unvoiced
    """
    N = len(block)
    win = get_window("hann", N)
    X = np.fft.rfft(block * win)
    f = np.fft.rfftfreq(N, 1/fs)
    mag = np.abs(X)

    # energy around plausible pitch region
    f0_band = (f >= 70) & (f <= 300)
    f0_energy = np.sum(mag[f0_band])

    sfm = spectral_flatness(mag)
    zcr = zero_crossings(block)

    # heuristic score (tune thresholds on your data)
    score = 0.0
    score += np.clip(f0_energy / (np.sum(mag) + 1e-9) * 3.0, 0, 1)  # strong low-harmonics → voiced
    score += np.clip(0.7 - sfm, 0, 0.7)                              # tonal → voiced
    score += np.clip(0.3 - zcr, 0, 0.3)                              # low ZCR → voiced
    return np.clip(score / 2.0, 0, 1)

# -----------------------------
# Blocks: excitation / noise injection for fricatives
# -----------------------------
class FricativeSynth:
    """
    Adds shaped noise when speech is unvoiced (s, f, sh).
    """
    def __init__(self, fs, band=(3000, 8000)):
        self.fs = fs
        self.sos = butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='bandpass', output='sos')
        self.filt = IIR(self.sos)

    def synth(self, N, level_db=-35):
        w = np.random.randn(N).astype(np.float32)
        w = self.filt.process(w)
        w = w * (10**(level_db/20.0))
        return w

# -----------------------------
# Blocks: static inverse EQ (shelf/tilt) + light dynamic EQ
# -----------------------------
class InverseEQ:
    """
    Fixed curve learned from calibration (throat vs air).
    Also adds a gentle high-shelf and spectral tilt fix.
    """
    def __init__(self, fs, shelf_gain_db=+12.0, shelf_f0=2500, shelf_Q=0.7):
        self.fs = fs
        # high-pass ~60 Hz to remove DC/cable rumble
        self.hp_sos = butter(2, 60/(fs/2), btype='highpass', output='sos')
        self.hp = IIR(self.hp_sos)

        # high-shelf biquad (pseudo; replace with RBJ cookbook design)
        self.shelf_sos = iirfilter(2, shelf_f0/(fs/2), btype='high', rs=None,
                                   ftype='butter', output='sos')  # placeholder high shelf proxy
        self.shelf = IIR(self.shelf_sos)
        self.shelf_gain = 10**(shelf_gain_db/20.0)

        # 4-band gentle dynamic EQ (static gains + envelope followers)
        edges = [0, 300, 1500, 3500, 8000]
        self.bands = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sos = butter(2, [max(1, lo)/(fs/2), hi/(fs/2)], btype='bandpass', output='sos')
            self.bands.append(dict(filt=IIR(sos), env=0.0, target_gain_db=0.0))

        # Example static target to brighten mids/highs
        target_db = [ -2, +3, +6, +8 ]  # tweak per your calibration
        for b, g in zip(self.bands, target_db):
            b["target_gain_db"] = g

        self.env_attack = 0.003  # 3 ms
        self.env_release= 0.150  # 150 ms

    def process(self, x):
        y = self.hp.process(x)
        # apply pseudo shelf (scale after filtering)
        y = self.shelf.process(y) * self.shelf_gain

        # split into bands, apply static gain + light upward compression in high bands
        out = np.zeros_like(y)
        eps = 1e-12
        for i, b in enumerate(self.bands):
            xb = b["filt"].process(y)
            # envelope follower (RMS)
            env_now = np.sqrt(np.mean(xb*xb) + eps)
            alpha = self.env_attack if env_now > b["env"] else self.env_release
            b["env"] = (1 - alpha)*b["env"] + alpha*env_now

            g_lin = 10**(b["target_gain_db"]/20.0)

            # add a touch of upward compression on top two bands
            if i >= 2:
                # if band is quiet, push it up a bit
                desired = 0.02  # target RMS
                lift = np.clip((desired / (b["env"]+eps)) - 1.0, 0.0, 3.0)
                g_lin *= (1.0 + 0.15*lift)

            out += xb * g_lin

        return out

# -----------------------------
# Blocks: LPC-style formant enhancement (classic trick)
# -----------------------------
def lpc_formant_enhance(x, order=14, preemph=0.97):
    """
    Simple LPC analysis/resynthesis to sharpen formants and reduce spectral tilt.
    Pseudo math: compute LPC, slightly move poles outward (or apply
    'formant emphasis' by leaky inverse filtering), then filter+de-emphasize.
    Replace with a real LPC implementation (e.g., librosa.lpc or your own).
    """
    # pre-emphasis
    x_pe = np.append(x[0], x[1:] - preemph * x[:-1])

    # ...compute LPC coeffs 'a' of order 'order' on x_pe...
    a = ...  # shape [order+1]

    # optional pole sharpening: a[1:] *= 0.9  # example (bring poles slightly closer to unit circle)
    # excitation by inverse filtering
    e = lfilter(a, [1.0], x_pe)

    # synthesis with slightly "sharpened" spectral envelope
    a_synth = ...  # adjusted coefficients
    y = lfilter([1.0], a_synth, e)

    # de-emphasis
    y[1:] = y[1:] + preemph * y[:-1]
    return y

# -----------------------------
# Blocks: simple Wiener denoiser (optional)
# -----------------------------
class Wiener:
    def __init__(self, NFFT, alpha=0.98):
        self.noise_psd = None
        self.alpha = alpha
        self.NFFT = NFFT
        self.win = get_window("hann", NFFT)

    def process(self, x):
        # STFT (single frame, minimal latency path)
        X = np.fft.rfft(x * self.win)
        psd = (np.abs(X)**2)

        # running noise estimate (assume throat mic already quiet; keep conservative)
        if self.noise_psd is None:
            self.noise_psd = psd.copy()
        else:
            self.noise_psd = self.alpha*self.noise_psd + (1-self.alpha)*psd

        H = np.maximum(0.15, 1.0 - self.noise_psd/(psd + 1e-12))  # floor to avoid musical noise
        Y = X * H
        y = np.fft.irfft(Y)
        return y[:len(x)]

# -----------------------------
# Blocks: AGC and Limiter
# -----------------------------
class AGC:
    def __init__(self, target_rms=0.07, atk=0.01, rel=0.300):
        self.level = 0.0
        self.atk = atk; self.rel = rel
        self.target = target_rms

    def process(self, x):
        rms = short_time_energy(x)
        alpha = self.atk if rms > self.level else self.rel
        self.level = (1-alpha)*self.level + alpha*rms
        gain = self.target / (self.level + 1e-9)
        gain = np.clip(gain, 0.5, 6.0)
        return x * gain

def soft_limiter(x, thresh=0.98):
    # soft clip around threshold
    y = x.copy()
    above = np.abs(y) > thresh
    y[above] = np.sign(y[above]) * (thresh + (1 - np.exp(-(np.abs(y[above])-thresh)*8)) / 8)
    return np.clip(y, -1.0, 1.0)

# -----------------------------
# The main DSP chain
# -----------------------------
class ThroatMicDSP:
    def __init__(self, fs, blocksize):
        self.fs = fs
        self.N   = blocksize
        self.eq  = InverseEQ(fs)
        self.fr  = FricativeSynth(fs)
        self.wnr = Wiener(NFFT=blocksize)  # optional
        self.agc = AGC()
        # state for overlap if you adopt STFT OLA; here we keep pure block for low latency

    def process_block(self, x):
        # 1) inverse EQ / spectral tilt fix
        y = self.eq.process(x)

        # 2) (optional) LPC formant enhancement (light)
        try:
            y = lpc_formant_enhance(y, order=14)
        except:
            pass  # keep running if not implemented

        # 3) voiced/unvoiced estimate and fricative injection
        vu = voiced_unvoiced_mask(y, self.fs)  # 0..1
        if vu < 0.35:
            # add a small amount of shaped noise; scale by block energy
            lvl = db(short_time_energy(y)) - 18  # relative level
            y = y + self.fr.synth(len(y), level_db=lvl)

        # 4) (optional) conservative Wiener denoise (throat mics are already isolated)
        # y = self.wnr.process(y)

        # 5) AGC and limiter
        y = self.agc.process(y)
        y = soft_limiter(y, thresh=0.98)
        return y.astype(np.float32)

# -----------------------------
# Real-time I/O (duplex stream)
# -----------------------------
def run_realtime(device_in=None, device_out=None, fs=44100, blocksize=512):
    """
    blocksize=512 @44.1 kHz ≈ 11.6 ms algorithmic chunk
    End-to-end latency ≈ I/O buffer (1–2 blocks) + processing (<1 ms) → ~15–30 ms typical
    """
    dsp = ThroatMicDSP(fs, blocksize)

    def callback(indata, outdata, frames, time, status):
        if status:
            # print(status)   # handle XRuns etc.
            pass
        x = indata[:, 0].copy()  # mono
        y = dsp.process_block(x)
        outdata[:, 0] = y  # mono out
        # if stereo output desired: outdata[:, 1] = y

    with sd.Stream(device=(device_in, device_out),
                   samplerate=fs,
                   blocksize=blocksize,
                   dtype='float32',
                   channels=1,  # mono in/out
                   latency='low',
                   callback=callback):
        print("Running… press Ctrl+C to stop")
        sd.sleep(10**9)  # keep alive; replace with your app loop

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    run_realtime(device_in=None, device_out=None, fs=44100, blocksize=512)