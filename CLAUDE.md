# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyQt6-based desktop application for throat microphone research. It records audio from microphones, converts speech to text using OpenAI's Whisper API, and displays real-time frequency spectrum visualization to help improve throat microphone audio quality.

## Development Setup

### Python Environment
- Uses Python virtual environment (`venv/`)
- Activate virtual environment before development: `source venv/bin/activate` (Unix/macOS) or `venv\Scripts\activate` (Windows)

### Platform-Specific Audio Dependencies

**For Windows:**
```bash
pip install -r requirements.txt
```

**For macOS:**
```bash
brew install portaudio
pip install -r requirements_mac.txt
```

Note: macOS uses `pyaudio` instead of `pyaudiowpatch` for better compatibility.

### Environment Variables
- Requires OpenAI API key in `.env` file: `OPENAI_API_KEY=your_key_here`
- Uses `python-dotenv` to load environment variables

## Running the Application

```bash
python push_to_text.py
```

## Code Architecture

### Core Components

1. **AudioRecorder** (`push_to_text.py:32`): Handles audio recording using PyAudio/PyAudioWPatch
   - Manages recording state and audio parameters
   - Saves recorded audio to WAV files
   - Provides real-time audio chunks for spectrum analysis

2. **AudioMonitor** (`push_to_text.py:127`): Real-time audio monitoring for spectrum visualization
   - Uses PyAudio callback-based streaming
   - Feeds audio data to spectrum plot updates

3. **MainWindow** (`push_to_text.py:254`): Main PyQt6 GUI application
   - Device selection with WASAPI enumeration
   - Real-time frequency spectrum plot using PyQtGraph
   - Push-to-talk recording interface
   - Transcription output display

4. **AudioProcessingWorker** (`push_to_text.py:265`): Unified worker for audio file operations
   - Handles saving, transcription, or combined save-and-transcribe operations
   - Uses QRunnable for non-blocking processing
   - Automatic cleanup of temporary audio files
   - Supports multiple processing modes in single worker class

### Audio Utilities (`util_audio.py`)
Cross-platform audio device management functions:
- **Windows**: WASAPI, MME, and DirectSound device enumeration
- **macOS**: Core Audio device enumeration  
- **Cross-platform**: Unified device interface with platform-specific optimizations
- Device capability detection and sample rate testing
- Automatic fallback to system defaults when platform APIs unavailable

### Audio Processing Utilities (`util_audio_processing.py`)
Reusable audio signal processing functions:
- **Data Conversion**: `bytes_to_audio_data()` - converts raw audio bytes to numpy arrays with channel handling
- **Spectrum Analysis**: `compute_audio_spectrum()` - FFT-based frequency analysis with windowing and dB conversion
- **Pipeline Function**: `process_audio_chunk_for_spectrum()` - complete bytes-to-spectrum processing

### Threading Architecture
- Main GUI thread handles UI updates
- **AudioProcessingWorker**: Unified QRunnable for all audio processing tasks (save/transcribe/both)
- **Subprocess isolation**: OpenAI API calls run in separate processes for UI responsiveness
- Thread-safe spectrum updates using Qt signals
- QThreadPool manages concurrent operations with automatic worker cleanup

## Key Technical Details

- **Audio Format**: 16-bit PCM, up to 2 channels, adaptive sample rates
- **Spectrum Analysis**: FFT with Hanning window, logarithmic frequency display (20Hz-20kHz)
- **Device Selection**: Platform-optimized (WASAPI on Windows, Core Audio on macOS) with fallback support
- **API Integration**: OpenAI Whisper-1 model with 60-second timeout
- **Real-time Processing**: 50ms timer intervals for audio chunks and spectrum updates