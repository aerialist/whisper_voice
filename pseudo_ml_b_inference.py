"""
Idea: stream throat audio → compute features each hop → stack context → MLP predicts air-like log-mel → convert to linear STFT magnitude via mel-pinv → combine with live throat phase (or do 3–5 Griffin-Lim iters) → ISTFT overlap-add → AGC/limiter → audio out.

Practical notes & options
	•	Resampling: Running features at 16 kHz keeps compute low while preserving speech. You can still output at 44.1 kHz (simple high-quality resampler both ways).
	•	Target representation: Mapping to air log-mel is robust and simple. If you prefer MFCCs, you’d predict air-MFCC, then rebuild a spectral envelope (e.g., inverse-DCT + liftering) before ISTFT.
	•	Phase: Using throat phase is the lowest-latency path and works surprisingly well for voiced regions. If you want cleaner fricatives, do 2–4 Griffin–Lim iterations per frame (trade a few ms extra CPU for quality).
	•	Context: ±2 frames (~±20 ms) is a good sweet spot for coarticulation without adding much latency (use causal right-context = 0 if you must keep <20 ms algorithmic delay).
	•	Normalization: Always apply the training set mean/std to inputs at inference; mismatch hurts.
	•	CPU-only: Set torch.set_num_threads(#) to your core count. This model easily runs real-time on a modern laptop CPU.
	•	Data: To stay speaker-independent, mix many speakers and conditions; augment with small EQ/noise on throat inputs so the model generalizes.
	•	For ASR: If your main goal is Whisper/STT, you can skip waveform resynthesis and feed the predicted mel (converted to 16k mel scale that Whisper expects) directly to the recognizer — lowest latency & compute.
"""

import numpy as np, sounddevice as sd, torch
from collections import deque

SR_IO      = 44100     # device rate
SR_FEAT    = 16000     # analysis rate
WIN_MS     = 25
HOP_MS     = 10
N_MELS     = 80
FFT        = 1024      # pick consistent with WIN_MS@SR_FEAT
HOP        = int(SR_FEAT * HOP_MS / 1000)
WIN        = int(SR_FEAT * WIN_MS / 1000)

class RealTimeEnhancer:
    def __init__(self, exp_dir="exp_mlp"):
        # load model + norm
        self.model = MLP(in_dim=..., out_dim=N_MELS)
        self.model.load_state_dict(torch.load(f"{exp_dir}/mlp.pt", map_location="cpu"))
        self.model.eval()
        stats = np.load(f"{exp_dir}/norm_stats.npz")
        self.mu, self.sigma = stats["mu"], stats["sigma"]

        # mel filter and pseudo-inverse for magnitude reconstruction
        self.Fmel = np.load(f"{exp_dir}/mel_filter.npy")       # [M, K]
        self.Fpinv = np.linalg.pinv(self.Fmel)                 # [K, M]

        # streaming buffers (analysis at SR_FEAT, synthesis at SR_FEAT)
        self.an_win = np.hanning(WIN).astype(np.float32)
        self.phase_prev = None
        self.ctx = deque(maxlen=1 + CTX_LEFT + CTX_RIGHT)
        for _ in range(1 + CTX_LEFT + CTX_RIGHT):
            self.ctx.append(np.zeros(N_MELS + (2 if ADD_F0 else 0), dtype=np.float32))

        # resamplers (throat I/O 44.1k → 16k → back to 44.1k)
        self.resamp_in  = ...
        self.resamp_out = ...

        # overlap-add synthesis buffer
        self.synth_tail = np.zeros(WIN, dtype=np.float32)

    # --- feature extraction (same as training) ---
    def throat_features(self, frame_16k):
        M_t = ...  # log-mel for current frame [1, 80]
        if ADD_F0:
            f0, vuv = estimate_f0_vuv(frame_16k, SR_FEAT)  # single frame versions or cached trackers
            feat = np.concatenate([M_t, f0[-1:], vuv[-1:]], axis=1)
        else:
            feat = M_t
        return feat.squeeze(0)  # [80(+2),]

    # --- single hop processing ---
    @torch.inference_mode()
    def process_hop(self, frame_44k1):
        # 1) downsample to 16k analysis
        x16 = self.resamp_in.process(frame_44k1)  # length ~HOP at 16k

        # 2) throat STFT for phase (keep a short sliding window)
        # X = stft(x16, WIN, HOP) last frame only
        mag_t, phase_t = ..., ...  # [K], [K] from last analysis frame

        # 3) features + context
        feat = self.throat_features(x16)          # [Din]
        self.ctx.append(feat)
        # wait until we have future context; or operate causal (pad zeros for right ctx)
        ctx_frames = list(self.ctx)[- (1 + CTX_LEFT + CTX_RIGHT):]
        X_ctx = np.concatenate(ctx_frames, axis=0)[None, :]      # [1, Dctx]
        Xn = (X_ctx - self.mu) / self.sigma
        # 4) predict air-like log-mel
        y_logmel = self.model(torch.from_numpy(Xn).float()).numpy()[0]  # [80]

        # 5) mel → linear magnitude (clip exponentials)
        M_air = np.exp(np.clip(y_logmel, -10, 10))                # [80]
        S_mag = self.Fpinv @ M_air                                # [K]
        S_mag = np.maximum(S_mag, 0.0)

        # 6) magnitude + phase → ISTFT (use throat phase, or few GL iters)
        # Use throat phase for minimal latency:
        S = S_mag * np.exp(1j * phase_t)
        y_frame = np.fft.irfft(S)[:WIN] * self.an_win            # time frame

        # 7) overlap-add to continuous stream
        out = self.synth_tail.copy()
        out[:WIN] += y_frame
        # keep tail for next OLA
        self.synth_tail = np.roll(self.synth_tail, -HOP)
        self.synth_tail[-HOP:] = 0.0

        # 8) light AGC/limiter
        out = self.agc_limit(out)

        # 9) upsample back to device rate
        y44 = self.resamp_out.process(out[:HOP])  # return one hop @44.1k worth
        return y44

    def agc_limit(self, x):
        rms = np.sqrt(np.mean(x*x) + 1e-12)
        gain = np.clip(0.08 / (rms + 1e-9), 0.5, 4.0)
        y = x * gain
        y = np.tanh(y * 1.2)  # soft limit
        return y

# -----------------------------
# Duplex stream
# -----------------------------
def run_realtime(exp_dir="exp_mlp", device_in=None, device_out=None,
                 fs_io=SR_IO, hop_io=480):  # ~10.9 ms @44.1k
    enh = RealTimeEnhancer(exp_dir)

    def callback(indata, outdata, frames, time, status):
        chunk = indata[:, 0]  # mono throat input @44.1k
        # split chunk into hops and process
        ptr = 0
        outbuf = np.zeros(frames, dtype=np.float32)
        while ptr < frames:
            n = min(hop_io, frames - ptr)
            y = enh.process_hop(chunk[ptr:ptr+n])  # returns ~n samples
            outbuf[ptr:ptr+len(y)] = y
            ptr += n
        outdata[:, 0] = outbuf

    with sd.Stream(device=(device_in, device_out),
                   samplerate=fs_io, channels=1, dtype='float32',
                   blocksize=hop_io, callback=callback, latency='low'):
        print("Real-time SI enhancement running… Ctrl+C to stop")
        sd.sleep(10**9)