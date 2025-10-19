import asyncio
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

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
FILTER_DIR = TMP_DIR / 'filters'
FILTER_DIR.mkdir(exist_ok=True)

import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib import pyplot as plt
from nicegui import context, events, ui
from scipy.signal import ShortTimeFFT, istft, resample_poly, stft
from scipy.signal.windows import get_window

if TYPE_CHECKING:
    from nicegui.client import Client

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


@dataclass
class FilterConfig:
    sample_rate: int
    n_fft: int
    hop: int
    num_mels: int
    mel_alpha: float
    gain_clip_db: float
    mel_centers_hz: np.ndarray
    mel_gain_db: np.ndarray


@dataclass
class FilterArtifacts:
    filtered_path: Path
    filtered_audio: np.ndarray
    sample_rate: int
    filter_config_path: Path
    filter_config: FilterConfig


FILTERED_PREFIX = 'audio_c'
FILTERED_TITLE = 'Filtered Audio'

PANEL_STATES: Dict[str, Optional[SpectrogramState]] = {'audio_a': None, 'audio_b': None, FILTERED_PREFIX: None}
PANEL_CONTAINERS: Dict[str, Dict[str, Any]] = {}
AUDIO_STATES: Dict[str, Optional[AudioDataState]] = {'audio_a': None, 'audio_b': None, FILTERED_PREFIX: None}
LAST_FILTER_CONFIG: Optional[FilterConfig] = None
LAST_FILTER_PATH: Optional[Path] = None

SAVED_FILTER_SELECT = None
MIC_SELECT = None
REALTIME_BUTTON = None
REALTIME_TASK: Optional[asyncio.Task[Any]] = None
REALTIME_STOP_EVENT: Optional[threading.Event] = None
REALTIME_CLIENT: Optional['Client'] = None

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


def save_filter_config(config: FilterConfig) -> Path:
    timestamp = int(time.time() * 1000)
    path = FILTER_DIR / f'filter_{timestamp}.npz'
    np.savez_compressed(
        path,
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop=config.hop,
        num_mels=config.num_mels,
        mel_alpha=config.mel_alpha,
        gain_clip_db=config.gain_clip_db,
        mel_centers_hz=config.mel_centers_hz,
        mel_gain_db=config.mel_gain_db,
    )
    metadata_path = path.with_suffix('.json')
    metadata = {
        'sample_rate': config.sample_rate,
        'n_fft': config.n_fft,
        'hop': config.hop,
        'num_mels': config.num_mels,
        'mel_alpha': config.mel_alpha,
        'gain_clip_db': config.gain_clip_db,
        'mel_centers_hz_min': float(config.mel_centers_hz.min()) if config.mel_centers_hz.size else 0.0,
        'mel_centers_hz_max': float(config.mel_centers_hz.max()) if config.mel_centers_hz.size else 0.0,
        'mel_gain_db_min': float(config.mel_gain_db.min()) if config.mel_gain_db.size else 0.0,
        'mel_gain_db_max': float(config.mel_gain_db.max()) if config.mel_gain_db.size else 0.0,
        'created_at': timestamp,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return path


def load_filter_config(path: Path) -> FilterConfig:
    with np.load(path) as data:
        return FilterConfig(
            sample_rate=int(data['sample_rate']),
            n_fft=int(data['n_fft']),
            hop=int(data['hop']),
            num_mels=int(data['num_mels']),
            mel_alpha=float(data['mel_alpha']),
            gain_clip_db=float(data['gain_clip_db']),
            mel_centers_hz=data['mel_centers_hz'],
            mel_gain_db=data['mel_gain_db'],
        )


def list_saved_filter_paths() -> list[Path]:
    dated_paths: list[tuple[float, Path]] = []
    for path in FILTER_DIR.glob('filter_*.npz'):
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        dated_paths.append((mtime, path))
    return [path for _, path in sorted(dated_paths, key=lambda item: item[0], reverse=True)]


def list_microphone_devices() -> list[tuple[str, int]]:
    devices: list[tuple[str, int]] = []
    try:
        device_infos = sd.query_devices()
    except Exception:
        return devices
    host_names: Dict[int, str] = {}
    try:
        hostapis = sd.query_hostapis()
        host_names = {idx: info.get('name', '') for idx, info in enumerate(hostapis)}
    except Exception:
        host_names = {}
    for index, info in enumerate(device_infos):
        max_inputs = info.get('max_input_channels') or info.get('maxInputChannels')
        if not max_inputs or max_inputs <= 0:
            continue
        name = info.get('name') or f'Device {index}'
        host_index = info.get('hostapi') if 'hostapi' in info else info.get('hostApi')
        host_name = host_names.get(host_index, '') if host_index is not None else ''
        default_rate = info.get('default_samplerate') if 'default_samplerate' in info else info.get('defaultSampleRate')
        rate_label = ''
        if isinstance(default_rate, (int, float)) and default_rate > 0:
            rate_label = f'{default_rate:.0f} Hz'
        label_parts = [name]
        if host_name:
            label_parts.append(host_name)
        if rate_label:
            label_parts.append(rate_label)
        label = ' - '.join(label_parts)
        devices.append((label, index))
    return devices


def selected_microphone_index() -> Optional[int]:
    if MIC_SELECT is None:
        return None
    value = MIC_SELECT.value
    if not value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def current_filter_path() -> Optional[Path]:
    selected_path: Optional[Path] = None
    if SAVED_FILTER_SELECT is not None:
        selected_value = SAVED_FILTER_SELECT.value
        if selected_value:
            candidate = Path(str(selected_value))
            if candidate.exists():
                selected_path = candidate
    if selected_path:
        return selected_path
    if LAST_FILTER_PATH and LAST_FILTER_PATH.exists():
        return LAST_FILTER_PATH
    return None


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
    mel_axis = hz_to_mel(mel_freqs)
    if mel_axis.size > 0 and mel_axis[0] > 0.0:
        num_rows = mel_power.shape[0]
        if mel_power.shape[1] > 0:
            first_col = mel_power[:, :1]
        else:
            first_col = np.full((num_rows, 1), vmin)
        mel_power = np.hstack([first_col, mel_power])
        mel_axis = np.insert(mel_axis, 0, 0.0)
    mesh = axis.pcolormesh(mel_axis, times_plot, mel_power, shading='auto', vmin=vmin, vmax=vmax)
    axis.set_xlabel('Mel')
    axis.set_ylabel('Time [s]')
    axis.set_title(f'{state.title} Log-Mel')
    mel_limit = hz_to_mel(freq_limit)
    mel_max = mel_axis[-1] if mel_axis.size > 0 else mel_limit
    axis.set_xlim(0, max(mel_max, 1e-9))
    axis.set_ylim(0, max(time_limit, 1e-9))
    secax = axis.secondary_xaxis('top', functions=(mel_to_hz, hz_to_mel))
    secax.set_xlabel('Frequency [Hz]')
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
) -> FilterArtifacts:
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

    filter_config = FilterConfig(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop=hop,
        num_mels=NUM_MELS,
        mel_alpha=MEL_ALPHA,
        gain_clip_db=GAIN_CLIP_DB,
        mel_centers_hz=mel_centers,
        mel_gain_db=diff_smoothed,
    )
    filter_path = save_filter_config(filter_config)

    output_path = TMP_DIR / f'{FILTERED_PREFIX}_{int(time.time() * 1000)}.wav'
    sf.write(output_path, filtered_audio, target_sr)
    return FilterArtifacts(
        filtered_path=output_path,
        filtered_audio=filtered_audio,
        sample_rate=target_sr,
        filter_config_path=filter_path,
        filter_config=filter_config,
    )


def apply_saved_filter_to_audio(
    source: AudioDataState,
    filter_path: Path,
) -> FilterArtifacts:
    config = load_filter_config(filter_path)
    target_sr = config.sample_rate
    source_data = resample_audio_to_rate(source.data, source.sample_rate, target_sr)
    n_fft = config.n_fft
    hop = config.hop
    window = get_window('hann', n_fft, fftbins=True)
    noverlap = n_fft - hop
    freqs, times, source_stft = stft(
        source_data,
        fs=target_sr,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    gains_db = np.interp(
        freqs,
        config.mel_centers_hz,
        config.mel_gain_db,
        left=config.mel_gain_db[0],
        right=config.mel_gain_db[-1],
    )
    gains = 10.0 ** (gains_db / 20.0)
    adjusted_stft = source_stft * gains[:, None]
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
    peak = np.max(np.abs(filtered_audio))
    if peak > 0:
        filtered_audio = filtered_audio / max(peak / 0.99, 1.0)
    output_path = TMP_DIR / f'{FILTERED_PREFIX}_{int(time.time() * 1000)}.wav'
    sf.write(output_path, filtered_audio, target_sr)
    return FilterArtifacts(
        filtered_path=output_path,
        filtered_audio=filtered_audio,
        sample_rate=target_sr,
        filter_config_path=filter_path,
        filter_config=config,
    )


def frequency_gains_from_config(config: FilterConfig) -> np.ndarray:
    freqs = np.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate)
    gains_db = np.interp(
        freqs,
        config.mel_centers_hz,
        config.mel_gain_db,
        left=config.mel_gain_db[0],
        right=config.mel_gain_db[-1],
    )
    return 10.0 ** (gains_db / 20.0)


async def save_audio_visualizations(audio_path: Path, title: str, prefix: str) -> Optional[tuple[Path, Path]]:
    if not audio_path.exists():
        return None
    data, sample_rate = await asyncio.to_thread(load_audio_mono, audio_path)
    state = await asyncio.to_thread(
        compute_spectrogram_state,
        data,
        sample_rate,
        audio_path,
        prefix,
        title,
    )
    time_limit = max(state.max_time, 1e-3)
    freq_limit = min(8000.0, max(state.max_freq, 1.0))
    spec_vmin = state.power_min
    spec_vmax = state.power_max if not np.isclose(state.power_min, state.power_max) else state.power_min + 1.0
    mel_vmin = state.mel_power_min
    mel_vmax = state.mel_power_max if not np.isclose(state.mel_power_min, state.mel_power_max) else state.mel_power_min + 1.0
    spectrogram_path = await asyncio.to_thread(
        render_spectrogram_image,
        state,
        time_limit,
        freq_limit,
        spec_vmin,
        spec_vmax,
    )
    mel_path = await asyncio.to_thread(
        render_log_mel_image,
        state,
        time_limit,
        freq_limit,
        mel_vmin,
        mel_vmax,
    )
    return spectrogram_path, mel_path


def realtime_filter_worker(filter_path: Path, device_index: int, stop_event: threading.Event) -> tuple[Path, Path]:
    config = load_filter_config(filter_path)
    hop = config.hop
    n_fft = config.n_fft
    window = get_window('hann', n_fft, fftbins=True)
    gains = frequency_gains_from_config(config)
    timestamp = int(time.time() * 1000)
    output_path = TMP_DIR / f'realtime_filtered_{timestamp}.wav'
    original_path = TMP_DIR / f'realtime_original_{timestamp}.wav'
    analysis_buffer = np.zeros(n_fft, dtype=np.float64)
    overlap = np.zeros(n_fft - hop, dtype=np.float64)

    input_kwargs = {
        'device': device_index,
        'channels': 1,
        'samplerate': config.sample_rate,
        'blocksize': hop,
        'dtype': 'float32',
    }
    output_kwargs = {
        'channels': 1,
        'samplerate': config.sample_rate,
        'blocksize': hop,
        'dtype': 'float32',
    }

    with sf.SoundFile(
        str(output_path),
        mode='w',
        samplerate=config.sample_rate,
        channels=1,
        subtype='FLOAT',
    ) as filtered_writer, sf.SoundFile(
        str(original_path),
        mode='w',
        samplerate=config.sample_rate,
        channels=1,
        subtype='FLOAT',
    ) as original_writer:
        try:
            with sd.InputStream(**input_kwargs) as input_stream, sd.OutputStream(**output_kwargs) as output_stream:
                while not stop_event.is_set():
                    try:
                        data, _ = input_stream.read(hop)
                    except Exception:
                        break
                    if data.size == 0:
                        continue
                    chunk = np.asarray(data[:, 0], dtype=np.float64)
                    if chunk.size < hop:
                        chunk = np.pad(chunk, (0, hop - chunk.size))
                    original_writer.write(chunk.astype(np.float32))
                    analysis_buffer = np.roll(analysis_buffer, -hop)
                    analysis_buffer[-hop:] = chunk
                    frame = analysis_buffer * window
                    spectrum = np.fft.rfft(frame)
                    filtered_spectrum = spectrum * gains
                    time_frame = np.fft.irfft(filtered_spectrum, n=n_fft)
                    output_frame = time_frame * window
                    if overlap.size:
                        output_frame[:overlap.size] += overlap
                    out_chunk = output_frame[:hop]
                    overlap = output_frame[hop:]
                    out_chunk = np.clip(out_chunk, -1.0, 1.0)
                    output_stream.write(out_chunk.astype(np.float32).reshape(-1, 1))
                    filtered_writer.write(out_chunk.astype(np.float32))
        except Exception:
            raise
    return output_path, original_path


async def finalize_realtime_session(
    filtered_path: Optional[Path],
    original_path: Optional[Path],
    filter_path: Path,
    error: Optional[BaseException],
) -> None:
    global REALTIME_TASK, REALTIME_STOP_EVENT, REALTIME_CLIENT
    client = REALTIME_CLIENT
    if client is not None:
        with client:
            if REALTIME_BUTTON is not None:
                REALTIME_BUTTON.text = 'realtime'
                REALTIME_BUTTON.props('color=primary')
            if error:
                ui.notify(f'Realtime processing failed: {error}', color='negative')
            else:
                audio_messages: list[str] = []
                visual_messages: list[str] = []
                if filtered_path is not None and filtered_path.exists():
                    audio_messages.append(f'Filtered: {saved_filter_display(filtered_path)}')
                    data, sample_rate = await asyncio.to_thread(load_audio_mono, filtered_path)
                    config = load_filter_config(filter_path)
                    artifacts = FilterArtifacts(
                        filtered_path=filtered_path,
                        filtered_audio=data,
                        sample_rate=sample_rate,
                        filter_config_path=filter_path,
                        filter_config=config,
                    )
                    await apply_filter_artifacts(artifacts, 'Realtime filtered output')
                    filtered_state = PANEL_STATES.get(FILTERED_PREFIX)
                    if filtered_state is not None:
                        visual_messages.append(f'Filtered spectrogram: {saved_filter_display(filtered_state.spectrogram_path)}')
                        visual_messages.append(f'Filtered logmel: {saved_filter_display(filtered_state.mel_image_path)}')
                if original_path is not None and original_path.exists():
                    audio_messages.append(f'Original: {saved_filter_display(original_path)}')
                    originals = await save_audio_visualizations(original_path, 'Realtime Original', 'realtime_original')
                    if originals is not None:
                        spectrogram_path, mel_path = originals
                        visual_messages.append(f'Original spectrogram: {saved_filter_display(spectrogram_path)}')
                        visual_messages.append(f'Original logmel: {saved_filter_display(mel_path)}')
                if audio_messages:
                    ui.notify('Realtime audio saved → ' + ' | '.join(audio_messages))
                if visual_messages:
                    ui.notify('Realtime visuals saved → ' + ' | '.join(visual_messages))
    else:
        if error:
            print(f'Realtime processing failed without UI context: {error}')
        else:
            if filtered_path is not None:
                await save_audio_visualizations(filtered_path, 'Realtime Filtered', 'realtime_filtered')
                print(f'Realtime filtered output saved (no UI context): {filtered_path}')
            if original_path is not None:
                await save_audio_visualizations(original_path, 'Realtime Original', 'realtime_original')
                print(f'Realtime original output saved (no UI context): {original_path}')
    REALTIME_TASK = None
    REALTIME_STOP_EVENT = None
    REALTIME_CLIENT = None


async def run_realtime_session(filter_path: Path, device_index: int, stop_event: threading.Event) -> None:
    filtered_path: Optional[Path] = None
    original_path: Optional[Path] = None
    error: Optional[BaseException] = None
    try:
        filtered_path, original_path = await asyncio.to_thread(realtime_filter_worker, filter_path, device_index, stop_event)
    except Exception as exc:
        error = exc
    await finalize_realtime_session(filtered_path, original_path, filter_path, error)


def saved_filter_display(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


async def update_saved_filter_select_options(selected_value: Optional[str] = None) -> None:
    global SAVED_FILTER_SELECT
    if SAVED_FILTER_SELECT is None:
        return
    options = {str(path): saved_filter_display(path) for path in list_saved_filter_paths()}
    SAVED_FILTER_SELECT.set_options(options)
    if not options:
        SAVED_FILTER_SELECT.value = None
        return
    keys = list(options.keys())
    if selected_value and selected_value in keys:
        SAVED_FILTER_SELECT.value = selected_value
    elif SAVED_FILTER_SELECT.value not in keys:
        SAVED_FILTER_SELECT.value = keys[0]


async def apply_filter_artifacts(artifacts: FilterArtifacts, status_message: str) -> None:
    containers = PANEL_CONTAINERS.get(FILTERED_PREFIX)
    if containers is None:
        return
    filtered_path = artifacts.filtered_path
    filtered_data = artifacts.filtered_audio
    sample_rate = artifacts.sample_rate
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
        ui.label(status_message).classes('text-sm text-gray-500')
        ui.audio(str(filtered_path)).props('controls autoplay')
        if LAST_FILTER_PATH:
            ui.label(f'Filter saved to: {saved_filter_display(LAST_FILTER_PATH)}').classes('text-sm')
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


async def generate_and_apply_filter() -> None:
    normal_state = AUDIO_STATES.get('audio_a')
    throat_state = AUDIO_STATES.get('audio_b')
    if normal_state is None or throat_state is None:
        ui.notify('Upload Reference Mic and Throat Mic before generating a filter.', color='warning')
        return
    try:
        artifacts = await asyncio.to_thread(apply_throat_to_normal_filter, normal_state, throat_state)
    except Exception as exc:
        ui.notify(f'Filter generation failed: {exc}', color='negative')
        return
    global LAST_FILTER_CONFIG, LAST_FILTER_PATH
    LAST_FILTER_CONFIG = artifacts.filter_config
    LAST_FILTER_PATH = artifacts.filter_config_path
    await apply_filter_artifacts(artifacts, 'Generated from Reference and Throat mic audio')
    if LAST_FILTER_PATH:
        ui.notify(f'Filter saved: {saved_filter_display(LAST_FILTER_PATH)}')
        await update_saved_filter_select_options(str(LAST_FILTER_PATH))


async def load_and_apply_selected_filter() -> None:
    throat_state = AUDIO_STATES.get('audio_b')
    if throat_state is None:
        ui.notify('Upload Throat Mic before loading a filter.', color='warning')
        return
    if SAVED_FILTER_SELECT is None:
        ui.notify('No saved filter selector available.', color='negative')
        return
    selected_value = SAVED_FILTER_SELECT.value
    if not selected_value:
        ui.notify('Choose a saved filter before loading.', color='warning')
        return
    selected_path = Path(str(selected_value))
    if not selected_path.exists():
        ui.notify('Selected filter file is missing. Refreshing saved filter list.', color='warning')
        await update_saved_filter_select_options()
        return
    try:
        artifacts = await asyncio.to_thread(apply_saved_filter_to_audio, throat_state, selected_path)
    except Exception as exc:
        ui.notify(f'Applying saved filter failed: {exc}', color='negative')
        return
    global LAST_FILTER_CONFIG, LAST_FILTER_PATH
    LAST_FILTER_CONFIG = artifacts.filter_config
    LAST_FILTER_PATH = selected_path
    await apply_filter_artifacts(artifacts, f'Loaded filter: {saved_filter_display(selected_path)}')
    await update_saved_filter_select_options(str(selected_path))


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

    async def handle_generate_click(_: events.ClickEventArguments) -> None:
        await generate_and_apply_filter()

    async def handle_load_click(_: events.ClickEventArguments) -> None:
        await load_and_apply_selected_filter()

    async def handle_realtime_click(_: events.ClickEventArguments) -> None:
        global REALTIME_TASK, REALTIME_STOP_EVENT, REALTIME_CLIENT
        if REALTIME_TASK and not REALTIME_TASK.done():
            if REALTIME_STOP_EVENT:
                REALTIME_STOP_EVENT.set()
            await REALTIME_TASK
            return
        filter_path = current_filter_path()
        if filter_path is None:
            ui.notify('Generate or select a filter before starting realtime processing.', color='warning')
            return
        device_index = selected_microphone_index()
        if device_index is None:
            ui.notify('Choose an input microphone before starting realtime processing.', color='warning')
            return
        try:
            REALTIME_CLIENT = context.client
        except RuntimeError:
            REALTIME_CLIENT = None
        stop_event = threading.Event()
        REALTIME_STOP_EVENT = stop_event
        if REALTIME_BUTTON is not None:
            REALTIME_BUTTON.text = 'stop realtime'
            REALTIME_BUTTON.props('color=negative')
        REALTIME_TASK = asyncio.create_task(run_realtime_session(filter_path, device_index, stop_event))

    with controls_parent:
        with ui.card().classes('w-full flex-1 gap-4'):
            ui.label(label).classes('text-lg font-medium')
            actions_row = ui.row().classes('w-full gap-3 items-center')
            with actions_row:
                ui.button('generate', on_click=handle_generate_click).props('color=primary')
                ui.button('load', on_click=handle_load_click)
                global REALTIME_BUTTON
                REALTIME_BUTTON = ui.button('realtime', on_click=handle_realtime_click).props('color=primary')
            saved_paths = list_saved_filter_paths()
            select_options = {str(path): saved_filter_display(path) for path in saved_paths}
            default_path = saved_paths[0] if saved_paths else None
            default_value = str(default_path) if default_path and str(default_path) in select_options else None
            global SAVED_FILTER_SELECT
            SAVED_FILTER_SELECT = ui.select(
                select_options,
                value=default_value,
                label='Saved filters',
            ).classes('w-full')
            mic_devices = list_microphone_devices()
            mic_options = {str(index): label for label, index in mic_devices}
            default_mic_value = next(iter(mic_options)) if mic_options else None
            global MIC_SELECT
            MIC_SELECT = ui.select(
                mic_options,
                value=default_mic_value,
                label='Input microphone',
            ).classes('w-full')
            if not mic_options:
                MIC_SELECT.props('disable')
            audio_container = ui.column().classes('w-full')
            with audio_container:
                ui.label('Press generate or load to create filtered audio.')

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

create_audio_panel('Reference Mic', 'audio_a', controls_row, spectrogram_row, logmel_row)
create_audio_panel('Throat Mic', 'audio_b', controls_row, spectrogram_row, logmel_row)
create_filtered_panel(FILTERED_TITLE, FILTERED_PREFIX, controls_row, spectrogram_row, logmel_row)

ui.run()
