import asyncio
import os
from pathlib import Path

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

audio_container = None
image_container = None


def generate_spectrogram(audio_path: Path) -> Path:
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
    figure, axis = plt.subplots(figsize=(10, 4))
    mesh = axis.pcolormesh(times, freqs, power_db, shading='auto')
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Frequency [Hz]')
    axis.set_title('Spectrogram')
    figure.colorbar(mesh, ax=axis, label='Power (dB)')
    figure.tight_layout()
    image_path = audio_path.with_suffix('.spectrogram.png')
    figure.savefig(image_path, dpi=150)
    plt.close(figure)
    return image_path


async def handle_upload(event: events.UploadEventArguments) -> None:
    uploaded = event.file
    target = TMP_DIR / uploaded.name
    await uploaded.save(target)
    ui.notify(f'Saved {target}')
    spectrogram_path = await asyncio.to_thread(generate_spectrogram, target)
    if audio_container is not None:
        audio_container.clear()
        with audio_container:
            ui.audio(str(target)).props('controls autoplay')
    if image_container is not None:
        image_container.clear()
        with image_container:
            ui.image(str(spectrogram_path)).props('alt=Spectrogram').classes('w-full max-w-3xl')


with ui.column().classes('w-full gap-4'):
    ui.upload(
        label='Choose local audio',
        on_upload=handle_upload,
        auto_upload=True,
        max_files=1,
    ).props('accept=audio/*')
    audio_container = ui.column().classes('w-full')
    image_container = ui.column().classes('w-full')

ui.run()
