import asyncio
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
from scipy.signal import ShortTimeFFT
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


PANEL_STATES: Dict[str, Optional[SpectrogramState]] = {'audio_a': None, 'audio_b': None}
PANEL_CONTAINERS: Dict[str, Dict[str, Any]] = {}


def compute_spectrogram_data(audio_path: Path, prefix: str, title: str) -> SpectrogramState:
    data, sample_rate = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.size < 2:
        raise ValueError('Audio file is too short for spectrogram generation')
    window_size = min(1024, data.size)
    window = get_window('hann', window_size, fftbins=True)
    hop = max(1, window_size // 4)
    stft = ShortTimeFFT(window, hop, sample_rate, scale_to='psd')
    spectrogram = stft.spectrogram(data)
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    times = np.arange(spectrogram.shape[1]) * hop / sample_rate
    power_db = 10 * np.log10(spectrogram + 1e-12)
    return SpectrogramState(
        prefix=prefix,
        title=title,
        audio_path=audio_path,
        times=times,
        freqs=freqs,
        power_db=power_db,
        spectrogram_path=audio_path.with_suffix('.spectrogram.png'),
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


def aggregate_limits() -> Optional[tuple[float, float, float, float]]:
    states = [state for state in PANEL_STATES.values() if state is not None]
    if not states:
        return None
    time_limit = max(state.max_time for state in states)
    freq_limit = min(8000.0, max(state.max_freq for state in states))
    vmin = min(state.power_min for state in states)
    vmax = max(state.power_max for state in states)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return time_limit, freq_limit, vmin, vmax


async def redraw_all_spectrograms() -> None:
    limits = aggregate_limits()
    if limits is None:
        return
    time_limit, freq_limit, vmin, vmax = limits
    for prefix, state in PANEL_STATES.items():
        if state is None:
            continue
        image_path = await asyncio.to_thread(render_spectrogram_image, state, time_limit, freq_limit, vmin, vmax)
        containers = PANEL_CONTAINERS.get(prefix)
        if containers:
            image_container = containers['image']
            image_container.clear()
            with image_container:
                ui.image(str(image_path)).props('alt=Spectrogram').classes('w-full max-w-3xl')


def create_audio_panel(label: str, prefix: str, controls_parent: Any, spectrogram_parent: Any) -> None:
    """Render upload controls and register containers for spectrogram redraws."""

    async def handle_upload(event: events.UploadEventArguments) -> None:
        uploaded = event.file
        target = TMP_DIR / f'{prefix}_{uploaded.name}'
        await uploaded.save(target)
        ui.notify(f'Saved {target}')
        state = await asyncio.to_thread(compute_spectrogram_data, target, prefix, label)
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

    PANEL_CONTAINERS[prefix] = {'audio': audio_container, 'image': image_container}


controls_row = ui.row().classes('w-full gap-6 items-start')
spectrogram_row = ui.row().classes('w-full gap-6 items-start')

create_audio_panel('Audio A', 'audio_a', controls_row, spectrogram_row)
create_audio_panel('Audio B', 'audio_b', controls_row, spectrogram_row)

ui.run()
