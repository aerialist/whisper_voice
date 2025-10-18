import asyncio
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

TMP_DIR = Path('tmp')
TMP_DIR.mkdir(exist_ok=True)
MPL_DIR = TMP_DIR / 'matplotlib'
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_DIR.resolve()))
XDG_CACHE_DIR = TMP_DIR / 'xdg_cache'
XDG_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', str(XDG_CACHE_DIR.resolve()))
FONTCONFIG_CACHE_DIR = XDG_CACHE_DIR / 'fontconfig'
FONTCONFIG_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from nicegui import events, ui
from scipy.signal import ShortTimeFFT, istft, resample_poly, stft
from scipy.signal.windows import get_window

@dataclass
class SpectrogramState:
    prefix: str
    title: str
    audio_path: Path
    times: np.ndarray
    freqs: np.ndarray
    power_db: np.ndarray
    spectrogram_path: Path
    mel_bins: np.ndarray
    mel_power_db: np.ndarray
    mel_image_path: Path

    @property
    def max_time(self) -> float:
        return float(self.times[-1]) if self.times.size > 0 else 0.0

    @property
    def max_freq(self) -> float:
        return float(self.freqs[-1]) if self.freqs.size > 0 else 0.0

    @property
    def power_min(self) -> float:
        return float(self.power_db.min()) if self.power_db.size > 0 else 0.0

    @property
    def power_max(self) -> float:
        return float(self.power_db.max()) if self.power_db.size > 0 else 0.0

    @property
    def mel_power_min(self) -> float:
        return float(self.mel_power_db.min()) if self.mel_power_db.size > 0 else 0.0

    @property
    def mel_power_max(self) -> float:
        return float(self.mel_power_db.max()) if self.mel_power_db.size > 0 else 0.0


@dataclass
class AudioDataState:
    prefix: str
    title: str
    audio_path: Path
    sample_rate: int
    data: np.ndarray


FILTERED_PREFIX = 'audio_c'
FILTERED_TITLE = 'Filtered Audio'

PANEL_STATES: Dict[str, Optional[SpectrogramState]] = {'audio_a': None, 'audio_b': None, FILTERED_PREFIX: None}
PANEL_CONTAINERS: Dict[str, Dict[str, Any]] = {}
AUDIO_STATES: Dict[str, Optional[AudioDataState]] = {'audio_a': None, 'audio_b': None, FILTERED_PREFIX: None}

N_FFT_DEFAULT = 1024
HOP_DEFAULT = 256
NUM_MELS = 40
MEL_ALPHA = 0.05
GAIN_CLIP_DB = 12.0
EPSILON = 1e-12


def ensure_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float64)


def load_audio_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(audio_path)
    data = ensure_mono(data)
    return data, sample_rate


def compute_spectrogram_state(
    data: np.ndarray,
    sample_rate: int,
    audio_path: Path,
    prefix: str,
    title: str,
) -> SpectrogramState:
    data = ensure_mono(data)
    if data.size < 2:
        raise ValueError('Audio file is too short for spectrogram generation')
    window_size = min(N_FFT_DEFAULT, data.size)
    window = get_window('hann', window_size, fftbins=True)
    hop = max(1, window_size // 4)
    stft = ShortTimeFFT(window, hop, sample_rate, scale_to='psd')
    spectrogram = stft.spectrogram(data)
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    times = np.arange(spectrogram.shape[1]) * hop / sample_rate
    power_db = 10 * np.log10(spectrogram + EPSILON)
    mel_filter, mel_centers = build_mel_filterbank(NUM_MELS, sample_rate, window_size, f_max=sample_rate / 2)
    mel_power = mel_filter @ spectrogram
    mel_power_db = 10 * np.log10(mel_power + EPSILON)
    return SpectrogramState(
        prefix=prefix,
        title=title,
        audio_path=audio_path,
        times=times,
        freqs=freqs,
        power_db=power_db,
        spectrogram_path=audio_path.with_suffix('.spectrogram.png'),
        mel_bins=mel_centers,
        mel_power_db=mel_power_db,
        mel_image_path=audio_path.with_suffix('.logmel.png'),
    )


def render_spectrogram_image(
    state: SpectrogramState,
    time_limit: float,
    freq_limit: float,
    vmin: float,
    vmax: float,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 4))
    freqs_plot = state.freqs.copy()
    times_plot = state.times.copy()
    power_plot = state.power_db.T.copy()
    if freqs_plot.size > 0 and freq_limit < freqs_plot[-1]:
        cutoff_idx = np.searchsorted(freqs_plot, freq_limit, side='right')
        freqs_plot = freqs_plot[:cutoff_idx]
        power_plot = power_plot[:, :cutoff_idx] if power_plot.size else power_plot
        if freqs_plot.size == 0:
            freqs_plot = np.array([freq_limit])
            power_plot = np.full((power_plot.shape[0], 1), vmin)
        elif freqs_plot[-1] < freq_limit:
            freqs_plot = np.append(freqs_plot, freq_limit)
            last_col = power_plot[:, -1:] if power_plot.size else np.full((power_plot.shape[0], 1), vmin)
            power_plot = np.hstack([power_plot, last_col])
    if freqs_plot.size > 0 and freqs_plot[-1] < freq_limit:
        freqs_plot = np.append(freqs_plot, freq_limit)
        pad_column = np.full((power_plot.shape[0], 1), vmin)
        power_plot = np.hstack([power_plot, pad_column])
    if times_plot.size > 0 and times_plot[-1] < time_limit:
        times_plot = np.append(times_plot, time_limit)
        pad_row = np.full((1, power_plot.shape[1]), vmin)
        power_plot = np.vstack([power_plot, pad_row])
    mesh = axis.pcolormesh(freqs_plot, times_plot, power_plot, shading='auto', vmin=vmin, vmax=vmax)
    axis.set_xlabel('Frequency [Hz]')
    axis.set_ylabel('Time [s]')
    axis.set_title(f'{state.title} Spectrogram')
    axis.set_xlim(0, max(freq_limit, 1e-9))
    axis.set_ylim(0, max(time_limit, 1e-9))
    figure.colorbar(mesh, ax=axis, label='Power (dB)')
    figure.tight_layout()
    previous_path = state.spectrogram_path
    timestamp = int(time.time() * 1000)
    image_path = state.audio_path.parent / f'{state.audio_path.stem}_{timestamp}.spectrogram.png'
    figure.savefig(image_path, dpi=150)
    plt.close(figure)
    state.spectrogram_path = image_path
    if previous_path != image_path and previous_path.exists():
        try:
            previous_path.unlink(missing_ok=True)
        except OSError:
            pass
    return image_path


def render_log_mel_image(
    state: SpectrogramState,
    time_limit: float,
    freq_limit: float,
    vmin: float,
    vmax: float,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 4))
    mel_freqs = state.mel_bins.copy()
    times_plot = state.times.copy()
    mel_power = state.mel_power_db.T.copy()
    if mel_freqs.size > 0 and freq_limit < mel_freqs[-1]:
        cutoff_idx = np.searchsorted(mel_freqs, freq_limit, side='right')
        mel_freqs = mel_freqs[:cutoff_idx]
        mel_power = mel_power[:, :cutoff_idx] if mel_power.size else mel_power
        if mel_freqs.size == 0:
            mel_freqs = np.array([freq_limit])
            mel_power = np.full((mel_power.shape[0], 1), vmin)
        elif mel_freqs[-1] < freq_limit:
            mel_freqs = np.append(mel_freqs, freq_limit)
            last_col = mel_power[:, -1:] if mel_power.size else np.full((mel_power.shape[0], 1), vmin)
            mel_power = np.hstack([mel_power, last_col])
    if mel_freqs.size > 0 and mel_freqs[-1] < freq_limit:
        mel_freqs = np.append(mel_freqs, freq_limit)
        pad_column = np.full((mel_power.shape[0], 1), vmin)
        mel_power = np.hstack([mel_power, pad_column])
    if times_plot.size > 0 and times_plot[-1] < time_limit:
        times_plot = np.append(times_plot, time_limit)
        pad_row = np.full((1, mel_power.shape[1]), vmin)
        mel_power = np.vstack([mel_power, pad_row])
    mesh = axis.pcolormesh(mel_freqs, times_plot, mel_power, shading='auto', vmin=vmin, vmax=vmax)
    axis.set_xlabel('Frequency [Hz]')
    axis.set_ylabel('Time [s]')
    axis.set_title(f'{state.title} Log-Mel')
    axis.set_xlim(0, max(freq_limit, 1e-9))
    axis.set_ylim(0, max(time_limit, 1e-9))
    figure.colorbar(mesh, ax=axis, label='Power (dB)')
    figure.tight_layout()
    previous_path = state.mel_image_path
    timestamp = int(time.time() * 1000)
    image_path = state.audio_path.parent / f'{state.audio_path.stem}_{timestamp}.logmel.png'
    figure.savefig(image_path, dpi=150)
    plt.close(figure)
    state.mel_image_path = image_path
    if previous_path != image_path and previous_path.exists():
        try:
            previous_path.unlink(missing_ok=True)
        except OSError:
            pass
    return image_path


def aggregate_limits() -> Optional[tuple[float, float, float, float, float, float]]:
    states = [state for state in PANEL_STATES.values() if state is not None]
    if not states:
        return None
    time_limit = max(state.max_time for state in states)
    freq_limit = min(8000.0, max(state.max_freq for state in states))
    spec_vmin = min(state.power_min for state in states)
    spec_vmax = max(state.power_max for state in states)
    mel_vmin = min(state.mel_power_min for state in states)
    mel_vmax = max(state.mel_power_max for state in states)
    if np.isclose(spec_vmin, spec_vmax):
        spec_vmax = spec_vmin + 1.0
    if np.isclose(mel_vmin, mel_vmax):
        mel_vmax = mel_vmin + 1.0
    return time_limit, freq_limit, spec_vmin, spec_vmax, mel_vmin, mel_vmax


async def redraw_all_spectrograms() -> None:
    limits = aggregate_limits()
    if limits is None:
        return
    time_limit, freq_limit, spec_vmin, spec_vmax, mel_vmin, mel_vmax = limits
    for prefix, state in PANEL_STATES.items():
        if state is None:
            continue
        image_path = await asyncio.to_thread(render_spectrogram_image, state, time_limit, freq_limit, spec_vmin, spec_vmax)
        mel_image_path = await asyncio.to_thread(render_log_mel_image, state, time_limit, freq_limit, mel_vmin, mel_vmax)
        containers = PANEL_CONTAINERS.get(prefix)
        if containers:
            image_container = containers['image']
            image_container.clear()
            with image_container:
                ui.image(str(image_path)).props('alt=Spectrogram').classes('w-full max-w-3xl')
            mel_container = containers.get('mel')
            if mel_container is not None:
                mel_container.clear()
                with mel_container:
                    ui.image(str(mel_image_path)).props('alt=Log-mel').classes('w-full max-w-3xl')


def hz_to_mel(hz: Any) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: Any) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def build_mel_filterbank(
    num_mels: int,
    sample_rate: int,
    n_fft: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if f_max is None:
        f_max = sample_rate / 2
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, num_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)
    filterbank = np.zeros((num_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, num_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]
        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2)
        if right <= left:
            continue
        for k in range(left, center):
            filterbank[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            filterbank[m - 1, k] = (right - k) / max(right - center, 1)
    mel_centers = hz_points[1:-1]
    return filterbank, mel_centers


def resample_audio_to_rate(data: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return ensure_mono(data)
    gcd = math.gcd(source_sr, target_sr)
    up = target_sr // gcd
    down = source_sr // gcd
    resampled = resample_poly(data, up, down)
    return ensure_mono(resampled)


def apply_throat_to_normal_filter(
    normal: AudioDataState,
    throat: AudioDataState,
) -> tuple[Path, np.ndarray, int]:
    target_sr = normal.sample_rate
    throat_data = resample_audio_to_rate(throat.data, throat.sample_rate, target_sr)
    normal_data = resample_audio_to_rate(normal.data, normal.sample_rate, target_sr)

    n_fft = min(N_FFT_DEFAULT, max(throat_data.size, 2))
    hop = HOP_DEFAULT if n_fft == N_FFT_DEFAULT else max(1, n_fft // 4)
    window = get_window('hann', n_fft, fftbins=True)
    noverlap = n_fft - hop

    freqs, times, throat_stft = stft(
        throat_data,
        fs=target_sr,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    throat_power = (np.abs(throat_stft) ** 2).astype(np.float64)

    mel_filter, mel_centers = build_mel_filterbank(NUM_MELS, target_sr, n_fft)
    log_mel_throat = 10 * np.log10(mel_filter @ throat_power + EPSILON)
    mu_throat = np.zeros(NUM_MELS, dtype=np.float64)
    for frame in log_mel_throat.T:
        mu_throat = (1 - MEL_ALPHA) * mu_throat + MEL_ALPHA * frame

    _, _, normal_stft = stft(
        normal_data,
        fs=target_sr,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    normal_power = (np.abs(normal_stft) ** 2).astype(np.float64)
    log_mel_normal = 10 * np.log10(mel_filter @ normal_power + EPSILON)
    mu_air = log_mel_normal.mean(axis=1)

    diff_db = np.clip(mu_air - mu_throat, -GAIN_CLIP_DB, GAIN_CLIP_DB)
    smoothing_kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    diff_smoothed = np.convolve(diff_db, smoothing_kernel, mode='same')
    gains_db = np.interp(freqs, mel_centers, diff_smoothed, left=diff_smoothed[0], right=diff_smoothed[-1])
    gains = 10.0 ** (gains_db / 20.0)

    adjusted_stft = throat_stft * gains[:, None]
    _, filtered_audio = istft(
        adjusted_stft,
        fs=target_sr,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        input_onesided=True,
        boundary=None,
    )
    filtered_audio = np.real(filtered_audio)
    if filtered_audio.size == 0:
        raise ValueError('Filtered audio is empty')
    peak = np.max(np.abs(filtered_audio))
    if peak > 0:
        filtered_audio = filtered_audio / max(peak / 0.99, 1.0)
    output_path = TMP_DIR / f'{FILTERED_PREFIX}_{int(time.time() * 1000)}.wav'
    sf.write(output_path, filtered_audio, target_sr)
    return output_path, filtered_audio, target_sr


async def maybe_update_filtered_audio() -> None:
    normal_state = AUDIO_STATES.get('audio_a')
    throat_state = AUDIO_STATES.get('audio_b')
    containers = PANEL_CONTAINERS.get(FILTERED_PREFIX)
    if normal_state is None or throat_state is None or containers is None:
        return
    try:
        filtered_path, filtered_data, sample_rate = await asyncio.to_thread(
            apply_throat_to_normal_filter, normal_state, throat_state
        )
    except Exception as exc:
        ui.notify(f'Filter generation failed: {exc}', color='negative')
        return
    previous_audio = AUDIO_STATES.get(FILTERED_PREFIX)
    if previous_audio and previous_audio.audio_path.exists() and previous_audio.audio_path != filtered_path:
        try:
            previous_audio.audio_path.unlink(missing_ok=True)
        except OSError:
            pass
    previous_state = PANEL_STATES.get(FILTERED_PREFIX)
    if previous_state and previous_state.spectrogram_path.exists():
        try:
            previous_state.spectrogram_path.unlink(missing_ok=True)
        except OSError:
            pass
    if previous_state and previous_state.mel_image_path.exists():
        try:
            previous_state.mel_image_path.unlink(missing_ok=True)
        except OSError:
            pass
    AUDIO_STATES[FILTERED_PREFIX] = AudioDataState(
        FILTERED_PREFIX,
        FILTERED_TITLE,
        filtered_path,
        sample_rate,
        filtered_data,
    )
    audio_container = containers['audio']
    audio_container.clear()
    with audio_container:
        ui.audio(str(filtered_path)).props('controls autoplay')
    filtered_state = await asyncio.to_thread(
        compute_spectrogram_state,
        filtered_data,
        sample_rate,
        filtered_path,
        FILTERED_PREFIX,
        FILTERED_TITLE,
    )
    PANEL_STATES[FILTERED_PREFIX] = filtered_state
    await redraw_all_spectrograms()


def create_audio_panel(
    label: str,
    prefix: str,
    controls_parent: Any,
    spectrogram_parent: Any,
    logmel_parent: Any,
) -> None:
    """Render upload controls and register containers for spectrogram redraws."""

    async def handle_upload(event: events.UploadEventArguments) -> None:
        uploaded = event.file
        target = TMP_DIR / f'{prefix}_{uploaded.name}'
        await uploaded.save(target)
        ui.notify(f'Saved {target}')
        data, sample_rate = await asyncio.to_thread(load_audio_mono, target)
        AUDIO_STATES[prefix] = AudioDataState(prefix, label, target, sample_rate, data)
        state = await asyncio.to_thread(compute_spectrogram_state, data, sample_rate, target, prefix, label)
        PANEL_STATES[prefix] = state
        containers = PANEL_CONTAINERS.get(prefix)
        if containers:
            audio_container = containers['audio']
            audio_container.clear()
            with audio_container:
                ui.audio(str(target)).props('controls autoplay')
        await redraw_all_spectrograms()
        await maybe_update_filtered_audio()

    with controls_parent:
        with ui.card().classes('w-full flex-1 gap-4'):
            ui.label(label).classes('text-lg font-medium')
            ui.upload(
                label=f'Choose {label.lower()}',
                on_upload=handle_upload,
                auto_upload=True,
                max_files=1,
            ).props('accept=audio/*')
            audio_container = ui.column().classes('w-full')

    with spectrogram_parent:
        with ui.card().classes('w-full flex-1 gap-2'):
            ui.label(f'{label} Spectrogram').classes('text-lg font-medium')
            image_container = ui.column().classes('w-full')

    with logmel_parent:
        with ui.card().classes('w-full flex-1 gap-2'):
            ui.label(f'{label} Log-Mel').classes('text-lg font-medium')
            logmel_container = ui.column().classes('w-full')

    PANEL_CONTAINERS[prefix] = {'audio': audio_container, 'image': image_container, 'mel': logmel_container}


def create_filtered_panel(
    label: str,
    prefix: str,
    controls_parent: Any,
    spectrogram_parent: Any,
    logmel_parent: Any,
) -> None:
    """Set up containers for generated audio without upload controls."""

    with controls_parent:
        with ui.card().classes('w-full flex-1 gap-4'):
            ui.label(label).classes('text-lg font-medium')
            audio_container = ui.column().classes('w-full')
            with audio_container:
                ui.label('Waiting for generated audio...')

    with spectrogram_parent:
        with ui.card().classes('w-full flex-1 gap-2'):
            ui.label(f'{label} Spectrogram').classes('text-lg font-medium')
            image_container = ui.column().classes('w-full')
            with image_container:
                ui.label('Waiting for generated spectrogram...')

    with logmel_parent:
        with ui.card().classes('w-full flex-1 gap-2'):
            ui.label(f'{label} Log-Mel').classes('text-lg font-medium')
            logmel_container = ui.column().classes('w-full')
            with logmel_container:
                ui.label('Waiting for generated log-mel...')

    PANEL_CONTAINERS[prefix] = {'audio': audio_container, 'image': image_container, 'mel': logmel_container}


controls_row = ui.row().classes('w-full gap-6 items-start')
spectrogram_row = ui.row().classes('w-full gap-6 items-start')
logmel_row = ui.row().classes('w-full gap-6 items-start')

create_audio_panel('Audio A', 'audio_a', controls_row, spectrogram_row, logmel_row)
create_audio_panel('Audio B', 'audio_b', controls_row, spectrogram_row, logmel_row)
create_filtered_panel(FILTERED_TITLE, FILTERED_PREFIX, controls_row, spectrogram_row, logmel_row)

ui.run()
