"""
Audio processing utilities for spectrum analysis and data conversion.
"""

import numpy as np


def bytes_to_audio_data(data_bytes, channels=1, dtype=np.int16):
    """
    Convert audio bytes to numpy array.
    
    Args:
        data_bytes: Raw audio data bytes
        channels: Number of audio channels (1 for mono, 2 for stereo)
        dtype: Numpy data type for the audio data
    
    Returns:
        numpy.ndarray: Audio data array, or None if conversion fails
    """
    if data_bytes is None:
        return None
    
    try:
        # Convert bytes to numpy array
        audio_data = np.frombuffer(data_bytes, dtype=dtype)
        
        # If stereo, take only the first channel
        if channels == 2:
            audio_data = audio_data[0::2]
        
        # Check data validity
        if len(audio_data) == 0:
            return None
            
        return audio_data
    except Exception:
        return None


def compute_audio_spectrum(audio_data, sample_rate, freq_min=20, freq_max=20000):
    """
    Compute frequency spectrum from audio data using FFT.
    
    Args:
        audio_data: Numpy array of audio samples
        sample_rate: Audio sample rate in Hz
        freq_min: Minimum frequency to include in output (Hz)
        freq_max: Maximum frequency to include in output (Hz)
    
    Returns:
        tuple: (frequencies, magnitudes_db) or (None, None) if computation fails
    """
    try:
        # Apply window function to reduce spectral leakage
        windowed_data = audio_data.astype(np.float64) * np.hanning(len(audio_data))
        
        # Compute FFT
        fft = np.fft.rfft(windowed_data)
        magnitude = np.abs(fft)
        
        # Normalize magnitude for 16-bit audio
        max_possible = 32767 * len(audio_data) / 2  # FFT scaling factor
        magnitude_normalized = magnitude / max_possible
        
        # Convert to dB (0 dB = maximum possible level)
        magnitude_db = 20 * np.log10(magnitude_normalized + 1e-10)
        
        # Frequency array
        freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
        
        # Filter to specified frequency range
        valid_indices = (freqs >= freq_min) & (freqs <= freq_max)
        freqs_filtered = freqs[valid_indices]
        magnitude_filtered = magnitude_db[valid_indices]
        
        if len(freqs_filtered) == 0:
            return None, None
            
        return freqs_filtered, magnitude_filtered
        
    except Exception:
        return None, None


def process_audio_chunk_for_spectrum(data_bytes, channels, sample_rate, freq_min=20, freq_max=20000):
    """
    Complete pipeline: convert audio bytes to spectrum data.
    
    Args:
        data_bytes: Raw audio data bytes
        channels: Number of audio channels
        sample_rate: Audio sample rate in Hz
        freq_min: Minimum frequency to include in output (Hz)
        freq_max: Maximum frequency to include in output (Hz)
    
    Returns:
        tuple: (frequencies, magnitudes_db) or (None, None) if processing fails
    """
    # Convert bytes to audio data
    audio_data = bytes_to_audio_data(data_bytes, channels)
    if audio_data is None:
        return None, None
    
    # Compute spectrum
    return compute_audio_spectrum(audio_data, sample_rate, freq_min, freq_max)