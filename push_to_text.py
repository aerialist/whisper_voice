import wave
import tempfile
import os
import sys
import threading
import time
import platform
from datetime import datetime
import json

# Platform-specific imports
if platform.system() == "Windows":
    import pyaudiowpatch as pyaudio
else:
    import pyaudio
import openai
from dotenv import load_dotenv
import numpy as np
import pyqtgraph as pg
import asyncio
# Note: avoid extra HTTP client deps; rely on OpenAI client's built-in timeout.

from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject, Qt
from PySide6.QtCore import QThread, Signal as _Signal, QObject as _QObject, Slot, QEvent, QTimer, QProcess
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMessageBox, QSizePolicy, QTextEdit,
    QScrollArea, QCheckBox, QFrame
)

VERSION = "20250810"

from util_audio import (
    get_platform_devices_info,
    get_device_info_by_id,
    get_default_input_device_cross_platform,
)

load_dotenv()

class AudioRecorder:
    def __init__(self, device_id, p, device_name=None):
        self.device_id = device_id
        self.p = p
        self.device_name = device_name or f"device_{device_id}"
        self.stream = None
        self.recording = False
        self.frames = []
        self.latest_chunk = None
        
    def start_recording(self):
        """Start recording audio"""
        try:
            device_info = get_device_info_by_id(self.device_id, self.p)
            if not device_info:
                return False, f"Device not found for ID: {self.device_id}"
            
            # Audio parameters
            self.sample_rate = int(device_info.get('defaultSampleRate', 44100))
            self.channels = min(int(device_info.get('maxInputChannels', 2)), 2)
            self.chunk = 1024
            
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=self.chunk
            )
            
            self.recording = True
            self.frames = []
            return True
        except Exception as e:
            return False, str(e)
    
    def record_chunk(self):
        """Record a chunk of audio data"""
        if self.stream and self.recording:
            try:
                # Drain as much as is available to avoid overflow/drops
                total_read = 0
                last_chunk = None
                bytes_per_frame = self.channels * self.p.get_sample_size(pyaudio.paInt16)
                vis_bytes = self.chunk * bytes_per_frame
                # Read at least once per call
                while True:
                    try:
                        avail = self.stream.get_read_available()
                    except Exception:
                        avail = self.chunk

                    # Determine how many frames to read this iteration
                    to_read = 0
                    if avail >= self.chunk:
                        # Read in multiples of chunk to keep timing smooth
                        to_read = (avail // self.chunk) * self.chunk
                    elif total_read == 0:
                        # Ensure we read at least one chunk per tick
                        to_read = self.chunk
                    else:
                        break

                    data = self.stream.read(to_read, exception_on_overflow=False)
                    self.frames.append(data)
                    total_read += to_read
                    # Store the last chunk-sized slice for visualization
                    if len(data) >= vis_bytes:
                        last_chunk = data[-vis_bytes:]
                    else:
                        last_chunk = data

                    # Avoid spending too long in one GUI tick
                    if total_read >= self.sample_rate // 5:  # cap ~200ms worth
                        break

                if last_chunk is not None:
                    self.latest_chunk = last_chunk
                return total_read > 0
            except Exception:
                return False
        return False
    
    def get_latest_audio_data(self):
        """Get latest audio chunk as numpy array for analysis"""
        if self.latest_chunk is None:
            return None
        
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(self.latest_chunk, dtype=np.int16)
            
            # If stereo, take only the first channel
            if self.channels == 2:
                audio_data = audio_data[0::2]
            
            # Debug: Check data validity
            if len(audio_data) == 0:
                return None
                
            return audio_data
        except Exception:
            return None
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        return self.frames
    
    def save_to_wav(self, filename):
        """Save recorded audio to WAV file"""
        if not self.frames:
            return False
        
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return True
        except Exception:
            return False
    
    def save_to_recordings_folder(self):
        """Save recorded audio to recordings folder with timestamp and device name"""
        if not self.frames:
            return False, "No audio data to save"
        
        try:
            # Create recordings directory if it doesn't exist
            recordings_dir = "recordings"
            if not os.path.exists(recordings_dir):
                os.makedirs(recordings_dir)
            
            # Generate filename with timestamp and device name
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            # Sanitize device name for filename
            safe_device_name = "".join(c for c in self.device_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_device_name = safe_device_name.replace(" ", "_")
            filename = f"{timestamp}_{safe_device_name}.wav"
            filepath = os.path.join(recordings_dir, filename)
            
            # Save the file
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            return True, filepath
        except Exception as e:
            return False, str(e)

class AudioMonitor:
    def __init__(self, device_id, p, callback):
        self.device_id = device_id
        self.p = p
        self.callback = callback
        self.stream = None
        self.monitoring = False
        self.latest_audio_data = None
        self.latest_sample_rate = None
        
    def start_monitoring(self):
        """Start monitoring audio for real-time visualization"""
        try:
            device_info = get_device_info_by_id(self.device_id, self.p)
            if not device_info:
                print(f"Device {self.device_id} not found for monitoring")
                return False
            
            self.sample_rate = int(device_info.get('defaultSampleRate', 44100))
            self.channels = min(int(device_info.get('maxInputChannels', 2)), 2)
            self.chunk = 1024
            
            print(f"Opening audio stream for monitoring - device: {self.device_id}, rate: {self.sample_rate}, channels: {self.channels}")
            
            # Add timeout and error handling for macOS microphone permission issues
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=self.chunk,
                stream_callback=self._audio_callback
            )
            
            print("Audio stream opened successfully")
            self.monitoring = True
            self.stream.start_stream()
            print("Audio monitoring started")
            return True
        except Exception as e:
            print(f"Audio monitoring failed: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for real-time processing"""
        if self.callback and self.monitoring:
            try:
                # Convert bytes to numpy array
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                
                # If stereo, take only the first channel
                if self.channels == 2:
                    audio_data = audio_data[0::2]
                
                # Store the data to be processed by timer on main thread
                self.latest_audio_data = audio_data
                self.latest_sample_rate = self.sample_rate
                
            except Exception as e:
                print(f"Monitor callback error: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

class TranscriptionWorker(QRunnable):
    class Signals(QObject):
        finished = Signal(str, str)  # transcription_text, file_path
        error = Signal(str, str)     # error_message, file_path
        progress = Signal(str)       # progress message

    def __init__(self, audio_file_path):
        super().__init__()
        self.audio_file_path = audio_file_path
        self.signals = self.Signals()
        self.finished = self.signals.finished
        self.error = self.signals.error
        self.progress = self.signals.progress
        # Configure sensible per-request timeouts to avoid long UI waits
        self._timeout_seconds = 45.0
        self._max_retries = 0

    def run(self):
        """Transcribe audio using OpenAI Whisper API (runs in thread pool)"""
        try:
            self.progress.emit("worker: start")
            transcript_text = self._transcribe_sync()
            self.finished.emit(transcript_text, self.audio_file_path)
        except Exception as e:
            self.error.emit(str(e), self.audio_file_path)
        finally:
            # Clean up the temporary file
            try:
                os.unlink(self.audio_file_path)
            except Exception:
                pass

    def _transcribe_sync(self):
        """Synchronous transcription method"""
        # Validate API key early for fast failure (avoids perceived freeze)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            raise RuntimeError("OPENAI_API_KEY is not set. Please configure your OpenAI API key.")

        # Create client with strict timeouts and no automatic retries to keep UI responsive
        client = openai.OpenAI(timeout=self._timeout_seconds, max_retries=self._max_retries)

        with open(self.audio_file_path, "rb") as audio_file:
            self.progress.emit("worker: request -> openai")
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

        self.progress.emit("worker: done")
        return transcript.text

class SaveAndTranscribeWorker(QRunnable):
    class Signals(QObject):
        finished = Signal(str)  # audio_file_path
        error = Signal(str, str)  # error_message, audio_file_path

    def __init__(self, recorder, temp_file_name):
        super().__init__()
        self.recorder = recorder
        self.temp_file_name = temp_file_name
        self.signals = self.Signals()
        self.finished = self.signals.finished
        self.error = self.signals.error

    def run(self):
        try:
            if self.recorder.save_to_wav(self.temp_file_name):
                self.finished.emit(self.temp_file_name)
            else:
                self.error.emit("Failed to save audio file", self.temp_file_name)
        except Exception as e:
            self.error.emit(str(e), self.temp_file_name)

class MainWindow(QMainWindow):
    # Add signals for recording control
    # device_id, audio_data, sample_rate
    spectrum_update_signal = Signal(int, object, int)

    def __init__(self):
        super().__init__()
        print("MainWindow __init__ started")
        self.setWindowTitle("Push to Text")
        self.setMinimumSize(800, 700)

        # Initialize PyAudio with error handling
        print("Initializing PyAudio...")
        try:
            self.p = pyaudio.PyAudio()
            print("PyAudio initialized successfully")
        except Exception as e:
            print(f"PyAudio initialization failed: {e}")
            raise

        # Recording/monitoring state
        self.is_recording = False
        self.recorders = {}
        self.monitors = {}
        self.device_colors = {}
        self.spectrum_curves = {}
        self.device_checkboxes = []
        self.tempfile_to_device = {}
        self.pending_transcriptions = 0
        self.session_results = []
        self._active_workers = set()
        self._procs = {}

        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.record_chunks)

        # Connect the spectrum update signal to the actual update method
        self.spectrum_update_signal.connect(self.update_spectrum_safe)

        # Thread pool for transcription - increase max threads
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)

        # Timer for processing audio data from monitoring thread
        self.spectrum_timer = QTimer()
        self.spectrum_timer.timeout.connect(self.process_audio_data)
        self.spectrum_timer.start(50)

        # Heartbeat timer to show UI is alive while background tasks run
        self.processing_heartbeat = QTimer()
        self.processing_heartbeat.setInterval(1000)
        self.processing_heartbeat.timeout.connect(self._processing_heartbeat_tick)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Device selection section (checkboxes)
        device_layout = QVBoxLayout()
        device_label = QLabel("Audio Input Devices:")
        device_layout.addWidget(device_label)
        self.device_scroll = QScrollArea()
        self.device_scroll.setWidgetResizable(True)
        self.device_scroll.setMinimumHeight(120)
        container = QWidget()
        self.device_list_layout = QVBoxLayout(container)
        self.device_list_layout.setContentsMargins(0, 0, 0, 0)
        self.device_list_layout.setSpacing(4)
        self.device_scroll.setWidget(container)
        device_layout.addWidget(self.device_scroll)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        device_layout.addWidget(line)
        main_layout.addLayout(device_layout)

        # Frequency spectrum plot
        spectrum_label = QLabel("Real-time Frequency Response:")
        main_layout.addWidget(spectrum_label)

        self.spectrum_widget = pg.PlotWidget()
        self.spectrum_widget.setLabel('left', 'Magnitude (dB)')
        self.spectrum_widget.setLabel('bottom', 'Frequency (Hz)')
        self.spectrum_widget.setTitle('Audio Frequency Spectrum')
        self.spectrum_widget.setMinimumHeight(200)
        self.spectrum_widget.setLogMode(x=True, y=False)
        self.spectrum_widget.setXRange(np.log10(20), np.log10(20000))
        self.spectrum_widget.setYRange(-110, 0)
        try:
            self.legend = self.spectrum_widget.addLegend()
        except Exception:
            self.legend = None
        main_layout.addWidget(self.spectrum_widget)

        # Push to Text button
        self.push_to_text_button = QPushButton("Push to Text")
        self.push_to_text_button.setMinimumHeight(40)
        self.push_to_text_button.setCheckable(True)
        self.push_to_text_button.clicked.connect(self.toggle_recording)
        main_layout.addWidget(self.push_to_text_button)

        # Output text area
        output_label = QLabel("Output Text:")
        main_layout.addWidget(output_label)
        self.output_text = QTextEdit()
        self.output_text.setMinimumHeight(150)
        self.output_text.setReadOnly(True)
        main_layout.addWidget(self.output_text)

        # Log text area
        log_label = QLabel("Log:")
        main_layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setMinimumHeight(100)
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

        # Populate device checkboxes with error handling
        print("Populating device list...")
        try:
            self.populate_device_checkboxes()
            print("Device list populated successfully")
        except Exception as e:
            print(f"Device list population failed: {e}")
            raise

        # Start audio monitoring for visualization with error handling
        print("Starting audio monitoring...")
        try:
            QTimer.singleShot(1000, self.delayed_audio_monitoring)
            print("Audio monitoring will start in 1 second...")
        except Exception as e:
            print(f"Audio monitoring setup failed: {e}")

        # Log initialization
        self.log_message("Application initialized")
        self.log_message("Hold down 'Push to Text' button to record audio")
        print("MainWindow __init__ completed")

    def populate_device_checkboxes(self):
        """Populate the device list with checkboxes for input devices"""
        try:
            devices_info = get_platform_devices_info(self.p)
            input_devices = devices_info.get("input", [])

            # Clear existing checkboxes
            while self.device_list_layout.count():
                item = self.device_list_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
            self.device_checkboxes.clear()

            if not input_devices:
                lbl = QLabel("No input devices found")
                self.device_list_layout.addWidget(lbl)
                self.log_message("No input devices found")
                return
            
            # Add devices to combo box
            default_device = get_default_input_device_cross_platform(self.p) or {}
            default_name = default_device.get("name", "")

            for idx, device in enumerate(input_devices):
                device_name = device.get("name", "Unknown Device")
                device_index = device.get("index", -1)
                cb = QCheckBox(device_name)
                cb.setToolTip(f"Index: {device_index} | Host API: {device.get('hostApiName', '')} | DefaultRate: {int(device.get('defaultSampleRate', 44100))} Hz")
                # Pre-check the default device
                if device_name == default_name:
                    cb.setChecked(True)
                cb.stateChanged.connect(self.on_device_selection_changed)
                self.device_list_layout.addWidget(cb)
                self.device_checkboxes.append((cb, device_index, device_name))

            self.device_list_layout.addStretch()
            self.log_message(f"Found {len(input_devices)} input devices")
            
        except Exception as e:
            lbl = QLabel("Error loading devices")
            self.device_list_layout.addWidget(lbl)
            self.log_message(f"Error loading audio devices: {str(e)}")
    
    def on_device_selection_changed(self):
        """Handle device selection change"""
        self.restart_audio_monitoring()
    
    def delayed_audio_monitoring(self):
        """Start audio monitoring after a delay to avoid blocking initialization"""
        print("Attempting delayed audio monitoring...")
        self.start_audio_monitoring()
    
    def start_audio_monitoring(self):
        """Start audio monitoring for real-time visualization"""
        # Stop and clear existing monitors and curves
        if self.monitors:
            print("Stopping existing audio monitors...")
            for m in self.monitors.values():
                try:
                    m.stop_monitoring()
                except Exception:
                    pass
            self.monitors.clear()

        # Determine selected device IDs
        selected = self.get_selected_device_ids()
        selected_ids = set([d for d, _ in selected])

        # Remove curves (and legend entries) for devices no longer selected
        for device_id in list(self.spectrum_curves.keys()):
            if device_id not in selected_ids:
                curve = self.spectrum_curves.pop(device_id, None)
                if curve is not None:
                    try:
                        self.spectrum_widget.removeItem(curve)
                    except Exception:
                        pass
                    if getattr(self, 'legend', None):
                        try:
                            self.legend.removeItem(curve)
                        except Exception:
                            pass
        # Create monitoring for each selected device
        if not selected:
            print("No devices selected for monitoring")
            return

        for i, (device_id, device_name) in enumerate(selected):
            print(f"Starting monitoring with device ID: {device_id}")
            # Assign color if not exists
            if device_id not in self.device_colors:
                self.device_colors[device_id] = pg.intColor(len(self.device_colors))
            # Ensure curve exists
            if device_id not in self.spectrum_curves:
                self.spectrum_curves[device_id] = self.spectrum_widget.plot(pen=self.device_colors[device_id], width=2, name=device_name)
            monitor = AudioMonitor(device_id, self.p, self.update_spectrum)
            if monitor.start_monitoring():
                self.monitors[device_id] = monitor
                print(f"Audio monitoring started for {device_name}")
            else:
                self.log_message(f"Failed to start audio monitoring for {device_name}")
    
    def restart_audio_monitoring(self):
        """Restart audio monitoring with new device"""
        # Stop any existing monitors
        if self.monitors:
            for m in self.monitors.values():
                try:
                    m.stop_monitoring()
                except Exception:
                    pass
            self.monitors.clear()
        self.start_audio_monitoring()
    
    def process_audio_data(self):
        """Process audio data from monitoring thread - called by timer on main thread"""
        # Iterate over monitors and process available data
        for device_id, mon in list(self.monitors.items()):
            if hasattr(mon, 'latest_audio_data') and mon.latest_audio_data is not None:
                audio_data = mon.latest_audio_data
                sample_rate = mon.latest_sample_rate
                mon.latest_audio_data = None
                self.update_spectrum_safe(device_id, audio_data, sample_rate)
    
    def update_spectrum(self, device_id, audio_data, sample_rate):
        """Update spectrum - called from audio thread, emit signal for thread safety"""
        self.spectrum_update_signal.emit(device_id, audio_data, sample_rate)
    
    @Slot(int, object, int)
    def update_spectrum_safe(self, device_id, audio_data, sample_rate):
        """Update the frequency spectrum plot - runs on main GUI thread"""
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
            
            # Filter to 20..20000 Hz
            valid_indices = (freqs >= 20) & (freqs <= 20000)
            freqs_filtered = freqs[valid_indices]
            magnitude_filtered = magnitude_db[valid_indices]
            
            if len(freqs_filtered) > 0:
                if device_id not in self.spectrum_curves:
                    color = self.device_colors.get(device_id, pg.intColor(len(self.device_colors)))
                    device_name = self.get_device_name_by_id(device_id) or f"Device {device_id}"
                    self.spectrum_curves[device_id] = self.spectrum_widget.plot(pen=color, width=2, name=device_name)
                self.spectrum_curves[device_id].setData(freqs_filtered, magnitude_filtered)
        
        except Exception as e:
            print(f"Spectrum update error: {e}")

    def record_chunks(self):
        """Record chunks from all active recorders"""
        if not self.is_recording:
            return
        selected_ids = set([d for d, _ in self.get_selected_device_ids()])
        for device_id, rec in list(self.recorders.items()):
            # If device is no longer selected, remove its curve and skip plotting
            if device_id not in selected_ids:
                if device_id in self.spectrum_curves:
                    curve = self.spectrum_curves.pop(device_id)
                    try:
                        self.spectrum_widget.removeItem(curve)
                    except Exception:
                        pass
                    if getattr(self, 'legend', None):
                        try:
                            self.legend.removeItem(curve)
                        except Exception:
                            pass
                # Still drain audio to keep buffers healthy, but don't plot
                try:
                    rec.record_chunk()
                except Exception:
                    pass
                continue
            success = rec.record_chunk()
            if success:
                audio_data = rec.get_latest_audio_data()
                if audio_data is not None:
                    self.update_spectrum_safe(device_id, audio_data, rec.sample_rate)
    
    def toggle_recording(self):
        """Toggle recording on button click"""
        if self.push_to_text_button.isChecked():
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording when button is toggled on"""
        if self.is_recording:
            return

        selected = self.get_selected_device_ids()
        if not selected:
            self.log_message("No input devices selected")
            self.push_to_text_button.setChecked(False)
            return

        # Reset session results for a new recording session
        self.session_results = []

        # Stop monitoring to avoid device conflicts
        if self.monitors:
            for m in self.monitors.values():
                try:
                    m.stop_monitoring()
                except Exception:
                    pass
            self.monitors.clear()
            self.log_message("Stopped audio monitoring during recording")

        # Create and start recorders for each selected device
        self.recorders.clear()
        start_failures = []
        for device_id, device_name in selected:
            rec = AudioRecorder(device_id, self.p, device_name)
            result = rec.start_recording()
            if result is True:
                self.recorders[device_id] = rec
                # Assign colors/curves if not present
                if device_id not in self.device_colors:
                    self.device_colors[device_id] = pg.intColor(len(self.device_colors))
                if device_id not in self.spectrum_curves:
                    self.spectrum_curves[device_id] = self.spectrum_widget.plot(pen=self.device_colors[device_id], width=2, name=device_name)
            else:
                err = result[1] if isinstance(result, tuple) else "Unknown error"
                start_failures.append((device_name, err))

        if self.recorders:
            self.is_recording = True
            self.push_to_text_button.setText("Recording... (Click again to stop)")
            self.push_to_text_button.setStyleSheet("background-color: #ff4444; color: white;")
            self.push_to_text_button.repaint()
            QApplication.processEvents()
            # Use a conservative timer interval; recorders drain available data internally
            interval_ms = 30
            self.recording_timer.start(interval_ms)
            self.log_message(
                f"Recording timer set to {interval_ms} ms across {len(self.recorders)} devices"
            )
            names = ", ".join([rec.device_name for rec in self.recorders.values()])
            self.log_message(f"Recording started with devices: {names}")
            if start_failures:
                for name, err in start_failures:
                    self.log_message(f"Failed to start device '{name}': {err}")
        else:
            if start_failures:
                for name, err in start_failures:
                    self.log_message(f"Failed to start device '{name}': {err}")
            self.log_message("Failed to start recording on any device")
            self.push_to_text_button.setChecked(False)
            self.start_audio_monitoring()

    def stop_recording(self):
        """Stop recording when button is toggled off"""
        if not self.is_recording:
            return

        self.recording_timer.stop()
        self.is_recording = False

        # UI update first
        self.push_to_text_button.setText("Processing...")
        self.push_to_text_button.setStyleSheet("background-color: #ffaa00; color: white;")
        self.push_to_text_button.setEnabled(False)
        self.push_to_text_button.repaint()
        QApplication.processEvents()

        QTimer.singleShot(0, self._after_stop_recording)

    def _after_stop_recording(self):
        # Stop all recorders and gather frames
        any_frames = False
        stopped_recorders = []
        for device_id, rec in list(self.recorders.items()):
            frames = rec.stop_recording()
            stopped_recorders.append(rec)
            if frames:
                any_frames = True
        self.recorders.clear()
        self.log_message("Recording stopped")

    # Don't restart audio monitoring during transcription; resume after finalize

        if not any_frames:
            self.log_message("No audio data recorded")
            self.reset_button()
            return

        # Save and transcribe each recorder
        self.pending_transcriptions = 0
        for rec in stopped_recorders:
            if not rec.frames:
                continue
            # Save to recordings folder first
            success, result = rec.save_to_recordings_folder()
            if success:
                self.log_message(f"Audio saved to: {result}")
            else:
                self.log_message(f"Failed to save recording for {rec.device_name}: {result}")
            # Create temp file for transcription
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            # Map temp file to device, id, and final saved recording path
            self.tempfile_to_device[temp_file.name] = (rec.device_name, rec.device_id, result if success else None)
            # Save WAV in background
            worker = SaveAndTranscribeWorker(rec, temp_file.name)
            worker.finished.connect(self.transcribe_audio)
            worker.error.connect(self._on_save_audio_error)
            # Retain until completion
            self._active_workers.add(worker)
            worker.finished.connect(lambda *_, w=worker: self._active_workers.discard(w))
            worker.error.connect(lambda *_, w=worker: self._active_workers.discard(w))
            self.thread_pool.start(worker)
            self.pending_transcriptions += 1

        if self.pending_transcriptions > 0:
            self.processing_heartbeat.start()

    def _on_save_audio_error(self, msg, temp_file_name):
        self.log_message(msg)
        self.reset_button()
        try:
            os.unlink(temp_file_name)
        except Exception:
            pass

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio file using OpenAI Whisper API"""
        self.log_message("Starting transcription with OpenAI Whisper...")
        # Prefer subprocess isolation to guarantee the UI process never blocks
        try:
            proc = QProcess(self)
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(lambda proc=proc: self._on_proc_output(proc, audio_file_path))
            proc.finished.connect(lambda code, status, proc=proc: self._on_proc_finished(proc, code, status, audio_file_path))
            self._procs[audio_file_path] = proc
            proc.start(sys.executable, [os.path.join(os.path.dirname(__file__), 'transcribe_subprocess.py'), audio_file_path])
            self.log_message("Transcription dispatched to subprocess")
        except Exception as _proc_err:
            # Fallback to thread worker
            self.log_message(f"Subprocess start failed, falling back to thread: {_proc_err}")
            worker = TranscriptionWorker(audio_file_path)
            worker.finished.connect(self.on_transcription_finished)
            worker.error.connect(self.on_transcription_error)
            worker.progress.connect(self._on_worker_progress)
            self._active_workers.add(worker)
            worker.finished.connect(lambda *_, w=worker: self._active_workers.discard(w))
            worker.error.connect(lambda *_, w=worker: self._active_workers.discard(w))
            self.thread_pool.start(worker)
            self.log_message("Transcription dispatched to background thread")
        # Ensure heartbeat is running even if save stage didn't start it
        try:
            if not self.processing_heartbeat.isActive():
                self.processing_heartbeat.start()
        except Exception:
            pass

    def _on_proc_output(self, proc: QProcess, audio_file_path: str):
        try:
            data = proc.readAllStandardOutput().data().decode('utf-8', errors='ignore').strip()
            if not data:
                return
            # Buffer until newline JSON
            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    self.log_message(f"subprocess: {line}")
                    continue
                if 'text' in obj:
                    self.on_transcription_finished(obj['text'], audio_file_path)
                elif 'error' in obj:
                    self.on_transcription_error(obj['error'], audio_file_path)
        except Exception as e:
            self.log_message(f"Subprocess read error: {e}")

    def _on_proc_finished(self, proc: QProcess, code: int, status, audio_file_path: str):
        # Clean up stored proc
        try:
            self._procs.pop(audio_file_path, None)
        except Exception:
            pass
        # If no result was produced, surface an error
        if audio_file_path in self.tempfile_to_device:
            self.on_transcription_error(f"Subprocess finished without result (code {code})", audio_file_path)

    def _on_worker_progress(self, message: str):
        """Log background worker progress messages."""
        self.log_message(message)
    
    def on_transcription_finished(self, transcription_text, file_path):
        """Handle successful transcription"""
        self.log_message("Transcription completed successfully")
        # Attempt to delete temporary file here (subprocess path)
        try:
            os.unlink(file_path)
            self.log_message("Temporary audio file deleted")
        except Exception:
            pass
        # Prefix device name if available
        prefix = ""
        device_name = None
        device_id = None
        saved_recording_path = None
        if file_path in self.tempfile_to_device:
            device_name, device_id, saved_recording_path = self.tempfile_to_device.pop(file_path)
            prefix = f"[{device_name}] "
        self.output_text.append(prefix + transcription_text)
        self.output_text.append("")  # Add blank line

        # Build session result entry for JSON summary
        try:
            device_info = get_device_info_by_id(device_id, self.p) if device_id is not None else None
        except Exception:
            device_info = None
        result_entry = {
            "device_info": device_info,
            "transcription": transcription_text,
            "transcription provider": "openai.whisper-1",
            "wave file name": os.path.basename(saved_recording_path) if saved_recording_path else None,
        }
        self.session_results.append(result_entry)

        self.pending_transcriptions = max(0, self.pending_transcriptions - 1)
        if self.pending_transcriptions == 0:
            self._finalize_session_and_reset()
    
    def on_transcription_error(self, error_message, file_path):
        """Handle transcription error"""
        self.log_message(f"Transcription error: {error_message}")
        self.log_message("Temporary audio file deleted")
        # Avoid a blocking modal dialog so the UI stays responsive.
        # If needed, we could surface a non-blocking toast/notification here.
        self.pending_transcriptions = max(0, self.pending_transcriptions - 1)
        if self.pending_transcriptions == 0:
            self._finalize_session_and_reset()

        # Clean up temp file
        try:
            os.unlink(file_path)
        except Exception:
            pass
    
    def reset_button(self):
        """Reset the Push to Text button to its default state"""
        self.push_to_text_button.setChecked(False)
        self.push_to_text_button.setText("Push to Text")
        self.push_to_text_button.setStyleSheet("")
        self.push_to_text_button.setEnabled(True)

    def _finalize_session_and_reset(self):
        """Save session JSON summary if available, then reset the UI button."""
        try:
            if self.session_results:
                recordings_dir = "recordings"
                if not os.path.exists(recordings_dir):
                    os.makedirs(recordings_dir)
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                json_path = os.path.join(recordings_dir, f"{ts}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(self.session_results, f, ensure_ascii=False, indent=2)
                self.log_message(f"Saved transcription summary: {json_path}")
            else:
                self.log_message("No transcriptions to summarize.")
        except Exception as e:
            self.log_message(f"Failed to save transcription summary: {e}")
        finally:
            try:
                self.processing_heartbeat.stop()
            except Exception:
                pass
            self.reset_button()
            # Now that transcription work is done, resume audio monitoring
            self.start_audio_monitoring()

    def _processing_heartbeat_tick(self):
        """Emit a periodic log while transcriptions are pending."""
        if self.pending_transcriptions > 0:
            self.log_message(f"Transcription pending: {self.pending_transcriptions}")
        else:
            try:
                self.processing_heartbeat.stop()
            except Exception:
                pass

    def closeEvent(self, event):
        """Clean up when closing the application"""
        if self.is_recording:
            self.stop_recording()
        
        if self.monitors:
            for m in self.monitors.values():
                try:
                    m.stop_monitoring()
                except Exception:
                    pass
        
        if hasattr(self, 'p'):
            self.p.terminate()
        event.accept()

    def get_selected_device_ids(self):
        """Get a list of (device_id, device_name) for selected devices"""
        selected = []
        for cb, device_id, device_name in self.device_checkboxes:
            if cb.isChecked():
                selected.append((device_id, device_name))
        return selected
    
    def log_message(self, message):
        """Add a message to the log text area"""
        import datetime as _dt
        timestamp = _dt.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def get_device_name_by_id(self, device_id):
        """Lookup a device name by its device_id from the checkbox list."""
        for cb, did, name in self.device_checkboxes:
            if did == device_id:
                return name
        return None

    def test_plot(self):
        """Test if the plot widget is working"""
        try:
            # Create some test data - a simple declining curve
            test_freqs = np.logspace(np.log10(100), np.log10(10000), 50)
            test_magnitudes = -20 - 10 * np.log10(test_freqs/1000)
            
            print(f"Testing plot with {len(test_freqs)} points")
            # Create a temporary curve for test
            temp_curve = self.spectrum_widget.plot(pen='y', width=2)
            temp_curve.setData(test_freqs, test_magnitudes)
            print("Test plot should now be visible")
        except Exception as e:
            print(f"Test plot error: {e}")
            import traceback
            traceback.print_exc()
    
if __name__ == "__main__":
    print("Starting application...")
    
    try:
        app = QApplication([])
        print("QApplication created successfully")
        
        # macOS-specific application setup
        if platform.system() == "Darwin":
            print("Applying macOS-specific application setup...")
            app.setApplicationName("Push to Text")
            app.setApplicationDisplayName("Push to Text")
            app.setQuitOnLastWindowClosed(True)
        
        window = MainWindow()
        
        # macOS-specific window visibility fixes
        if platform.system() == "Darwin":
            # Set window flags to ensure it appears
            window.setWindowFlags(window.windowFlags() | Qt.WindowStaysOnTopHint)
            window.show()
            window.setWindowFlags(window.windowFlags() & ~Qt.WindowStaysOnTopHint)
            window.show()
            window.raise_()
            window.activateWindow()
            app.processEvents()
            
            # Force application to front
            import subprocess
            subprocess.run(['osascript', '-e', 'tell application "System Events" to set frontmost of first process whose unix id is {} to true'.format(os.getpid())], check=False)
        else:
            window.show()
        
        app.exec()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

