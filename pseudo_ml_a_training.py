"""
Train a small feed-forward regressor:
inputs  = throat features (log-mel + F0 + V/UV) with ±context
targets = air-mic log-mel features

elow is a clear, speaker-independent classical ML pipeline:
(A) training a small MLP to map throat → air log-mel features, and
(B) low-latency, CPU-only real-time inference that takes the throat stream, predicts air-like mel frames, and resynthesizes audio with ISTFT (optionally a few Griffin–Lim phase refinement steps).

It’s pseudo-Python: structure, shapes, and logic are real; swap in your actual libraries (librosa/torchaudio/scipy), file I/O, and glue.
"""

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from collections import defaultdict
# use librosa or torchaudio in your real code
# import librosa

# -----------------------------
# Hyperparams
# -----------------------------
SR_IN        = 44100        # raw recordings
SR_FEAT      = 16000        # analysis rate (lower = faster; good for speech)
N_MELS       = 80
WIN_MS       = 25           # STFT window
HOP_MS       = 10
CTX_LEFT     = 2            # ±2 frames of context (≈ ±20 ms)
CTX_RIGHT    = 2
ADD_F0       = True         # add F0 + voicing flag
GL_ITERS     = 0            # not used in training (waveform not needed)
BATCH_FRAMES = 1024
EPOCHS       = 50

# -----------------------------
# Feature extraction helpers
# -----------------------------
def stft_logmel(y, sr, n_mels=N_MELS, win_ms=WIN_MS, hop_ms=HOP_MS):
    """
    Return log-mel spectrogram [T, n_mels], and also magnitude STFT and phase if needed.
    """
    # y = librosa.resample(y, orig_sr=sr, target_sr=SR_FEAT)
    # S = np.abs(librosa.stft(y, n_fft=round(sr*win_ms/1000), hop_length=round(sr*hop_ms/1000), window='hann'))**2
    # M = librosa.feature.melspectrogram(S=S, sr=SR_FEAT, n_mels=n_mels)
    # logM = np.log(np.maximum(1e-8, M))
    logM = ...
    return logM  # [T, n_mels]

def estimate_f0_vuv(y, sr):
    """
    Cheap F0 + voicing (e.g., pyin, RAPT, or autocorr).
    Returns f0_hz[T], vuv[T] (0/1).
    """
    f0_hz, vuv = ..., ...
    # Normalize F0 (log scale), clip to 50–400 Hz range, replace unvoiced f0 with 0
    f0_feat = np.log(np.clip(f0_hz, 50, 400)) / np.log(400)
    f0_feat[~(vuv > 0.5)] = 0.0
    return f0_feat[:, None], vuv[:, None]  # [T,1], [T,1]

def make_inputs_from_throat(y_throat):
    M_t = stft_logmel(y_throat, SR_FEAT)                 # [T, 80]
    feats = [M_t]
    if ADD_F0:
        f0, vuv = estimate_f0_vuv(y_throat, SR_FEAT)     # [T,1], [T,1]
        feats += [f0, vuv]
    X = np.concatenate(feats, axis=1)                     # [T, 80(+2)]
    return X

def stack_context(X, left=CTX_LEFT, right=CTX_RIGHT):
    """
    Frame context stacking for coarticulation.
    In:  X [T, D]  ->  Out: [T, D*(1+left+right)]
    """
    T, D = X.shape
    pads = [np.zeros((left, D)), X, np.zeros((right, D))]
    Xpad = np.concatenate(pads, axis=0)
    rows = [Xpad[t : t+1+left+right, :].reshape(-1) for t in range(left, left+T)]
    return np.stack(rows, axis=0)

# -----------------------------
# Dataset (paired files)
# -----------------------------
def load_paired_recordings(pairs_list):
    """
    pairs_list: list of (wav_throat_path, wav_air_path, speaker_id)
    Return dict with speaker-split for SI training/validation.
    """
    data = defaultdict(list)
    for p_th, p_air, spk in pairs_list:
        y_t, sr_t = ..., ...
        y_a, sr_a = ..., ...
        # resample to SR_FEAT, rough time align (they were recorded simultaneously)
        # y_t = librosa.resample(y_t, sr_t, SR_FEAT); y_a = librosa.resample(y_a, sr_a, SR_FEAT)
        # Optional fine alignment: GCC-PHAT / DTW on low-band envelopes
        # y_t, y_a = align_signals(y_t, y_a, SR_FEAT)
        X_t = make_inputs_from_throat(y_t)      # [T, 82]
        Y_a = stft_logmel(y_a, SR_FEAT)         # [T, 80]
        T = min(len(X_t), len(Y_a))
        data['all'].append(dict(X=X_t[:T], Y=Y_a[:T], spk=spk))
    return data

# -----------------------------
# Normalization (speaker-indep.)
# -----------------------------
class Scaler:
    def fit(self, X):
        self.mu = X.mean(axis=0); self.sigma = X.std(axis=0) + 1e-8
    def transform(self, X): return (X - self.mu) / self.sigma
    def inv(self, Xn): return Xn * self.sigma + self.mu

# -----------------------------
# Model: small MLP regressor
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),    nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x): return self.net(x)

# -----------------------------
# Training loop
# -----------------------------
def train_model(pairs_train, pairs_val, save_dir="exp_mlp"):
    ds = load_paired_recordings(pairs_train)
    # Concatenate all speakers to build speaker-indep. stats
    X_list, Y_list = [], []
    for item in ds['all']:
        X_list.append(item['X']); Y_list.append(item['Y'])
    X_all = np.concatenate(X_list, 0); Y_all = np.concatenate(Y_list, 0)

    # Stack context
    X_ctx = stack_context(X_all)                     # [T, Dctx]
    in_dim = X_ctx.shape[1]; out_dim = Y_all.shape[1]

    # Train/val split by speakers (already separated in pairs_val)
    scaler = Scaler(); scaler.fit(X_ctx)
    X_ctx_n = scaler.transform(X_ctx)

    # Torch tensors (minibatch by random slice of frames)
    device = torch.device("cpu")
    model  = MLP(in_dim, out_dim).to(device)
    opt    = optim.Adam(model.parameters(), lr=2e-3)
    crit   = nn.SmoothL1Loss()

    # Build a simple frame sampler
    def sampler(Xn, Y):
        T = len(Xn)
        while True:
            idx = np.random.randint(0, T, size=(BATCH_FRAMES,))
            yield torch.from_numpy(Xn[idx]).float(), torch.from_numpy(Y[idx]).float()

    gen = sampler(X_ctx_n, Y_all)
    best_val = 1e9; patience = 8; bad = 0

    for epoch in range(EPOCHS):
        model.train(); running = 0.0
        for _ in range(100):  # ~100 minibatches / epoch
            xb, yb = next(gen)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss)

        # --- validation on held-out speakers ---
        with torch.no_grad():
            val_losses = []
            for (pth, pair) in pairs_val:  # iterate files; compute per-frame loss
                # prepare X,Y for this file
                X_t = make_inputs_from_throat(...); Y_a = stft_logmel(..., SR_FEAT)
                T = min(len(X_t), len(Y_a))
                Xn = scaler.transform(stack_context(X_t[:T]))
                pred = model(torch.from_numpy(Xn).float()).numpy()
                val_losses.append(np.mean(np.abs(pred - Y_a[:T])))
            val = float(np.mean(val_losses))

        print(f"epoch {epoch}: train {running/100:.4f}  val {val:.4f}")
        if val < best_val:
            best_val = val; bad = 0
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/mlp.pt")
            np.savez(f"{save_dir}/norm_stats.npz", mu=scaler.mu, sigma=scaler.sigma)
        else:
            bad += 1
            if bad >= patience: break

    # Also save mel filterbank used at SR_FEAT for inference inversion
    # Fmel: [n_mels, n_fft_bins]  (use librosa.filters.mel)
    Fmel = ...
    np.save(f"{save_dir}/mel_filter.npy", Fmel)