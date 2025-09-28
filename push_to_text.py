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
    QScrollArea, QCheckBox, QFrame, QComboBox
)

VERSION = "20250811"

from util_audio import (
    get_platform_devices_info,
    get_device_info_by_id,
    get_default_input_device_cross_platform,
    get_default_devices,
)
from util_audio_processing import (
    bytes_to_audio_data,
    compute_audio_spectrum,
)

load_dotenv()

class AudioDevice:
    def __init__(self, device_id, p, device_name=None):
        self.device_id = device_id
        self.p = p
        self.device_name = device_name or f"device_{device_id}"
        self.stream = None
        self.mode = None  # 'recording' or 'monitoring'
        
        # Recording state
        self.frames = []
        self.latest_chunk = None
        
        # Monitoring state
        self.callback = None
        self.latest_audio_data = None
        self.latest_sample_rate = None
        
        # Shared audio parameters
        self.sample_rate = None
        self.channels = None
        self.chunk = 1024
        
    def _initialize_audio_params(self):
        """Initialize audio parameters from device info"""
        device_info = get_device_info_by_id(self.device_id, self.p)
        if not device_info:
            return False, f"Device not found for ID: {self.device_id}"
        
        self.sample_rate = int(device_info.get('defaultSampleRate', 44100))
        self.channels = min(int(device_info.get('maxInputChannels', 2)), 2)
        return True, None
        
    def start_recording(self):
        """Start recording audio"""
        try:
            success, error = self._initialize_audio_params()
            if not success:
                return False, error
            
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=self.chunk
            )
            
            self.mode = 'recording'
            self.frames = []
            return True
        except Exception as e:
            return False, str(e)
    
    def start_monitoring(self, callback):
        """Start monitoring audio for real-time visualization"""
        try:
            success, error = self._initialize_audio_params()
            if not success:
                print(f"Device {self.device_id} not found for monitoring")
                return False
            
            self.callback = callback
            
            print(f"Opening audio stream for monitoring - device: {self.device_id}, rate: {self.sample_rate}, channels: {self.channels}")
            
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
            self.mode = 'monitoring'
            self.stream.start_stream()
            print("Audio monitoring started")
            return True
        except Exception as e:
            print(f"Audio monitoring failed: {e}")
            return False
    
    def record_chunk(self):
        """Record a chunk of audio data (only in recording mode)"""
        if self.stream and self.mode == 'recording':
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
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for real-time processing (monitoring mode)"""
        if self.callback and self.mode == 'monitoring':
            try:
                # Convert bytes to numpy array using utility function
                audio_data = bytes_to_audio_data(in_data, self.channels)
                
                if audio_data is not None:
                    # Store the data to be processed by timer on main thread
                    self.latest_audio_data = audio_data
                    self.latest_sample_rate = self.sample_rate
                
            except Exception as e:
                print(f"Monitor callback error: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def get_latest_audio_data(self):
        """Get latest audio chunk as numpy array for analysis"""
        # For recording mode, use latest_chunk
        if self.mode == 'recording' and self.latest_chunk is not None:
            return bytes_to_audio_data(self.latest_chunk, self.channels)
        
        # For monitoring mode, return None as data is handled by callback
        return None
    
    def stop(self):
        """Stop recording or monitoring"""
        if self.stream:
            if self.mode == 'monitoring':
                self.mode = None
            elif self.mode == 'recording':
                self.mode = None
                
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        return self.frames if hasattr(self, 'frames') else []
    
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
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
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

class AudioProcessingWorker(QRunnable):
    class Signals(QObject):
        finished = Signal(str, str)  # transcription_text, file_path
        error = Signal(str, str)     # error_message, file_path
        progress = Signal(str)       # progress message
        saved = Signal(str)          # temp_file_path (for save-only mode)

    def __init__(self, audio_device=None, audio_file_path=None, temp_file_name=None, mode='transcribe'):
        super().__init__()
        self.audio_device = audio_device
        self.audio_file_path = audio_file_path
        self.temp_file_name = temp_file_name
        self.mode = mode  # 'save', 'transcribe', or 'save_and_transcribe'
        
        self.signals = self.Signals()
        self.finished = self.signals.finished
        self.error = self.signals.error
        self.progress = self.signals.progress
        self.saved = self.signals.saved
        
        # Configure sensible per-request timeouts to avoid long UI waits
        self._timeout_seconds = 45.0
        self._max_retries = 0

    def run(self):
        """Process audio based on the specified mode"""
        try:
            if self.mode == 'save':
                self._save_audio()
            elif self.mode == 'transcribe':
                self._transcribe_audio()
            elif self.mode == 'save_and_transcribe':
                self._save_and_transcribe()
        except Exception as e:
            file_path = self.audio_file_path or self.temp_file_name
            self.error.emit(str(e), file_path or "unknown")

    def _save_audio(self):
        """Save audio to file"""
        if self.audio_device.save_to_wav(self.temp_file_name):
            self.saved.emit(self.temp_file_name)
        else:
            self.error.emit("Failed to save audio file", self.temp_file_name)

    def _transcribe_audio(self):
        """Transcribe existing audio file"""
        self.progress.emit("worker: start")
        transcript_text = self._transcribe_sync()
        self.finished.emit(transcript_text, self.audio_file_path)
        
        # Clean up the temporary file
        try:
            os.unlink(self.audio_file_path)
        except Exception:
            pass

    def _save_and_transcribe(self):
        """Save audio then transcribe it"""
        # First save the audio
        if not self.audio_device.save_to_wav(self.temp_file_name):
            self.error.emit("Failed to save audio file", self.temp_file_name)
            return
        
        # Then transcribe it
        self.progress.emit("worker: start transcription")
        transcript_text = self._transcribe_sync_from_file(self.temp_file_name)
        self.finished.emit(transcript_text, self.temp_file_name)
        
        # Clean up the temporary file
        try:
            os.unlink(self.temp_file_name)
        except Exception:
            pass

    def _transcribe_sync(self):
        """Synchronous transcription method for existing file"""
        return self._transcribe_sync_from_file(self.audio_file_path)

    def _transcribe_sync_from_file(self, file_path):
        """Synchronous transcription method for specified file"""
        # Validate API key early for fast failure (avoids perceived freeze)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            raise RuntimeError("OPENAI_API_KEY is not set. Please configure your OpenAI API key.")

        # Create client with strict timeouts and no automatic retries to keep UI responsive
        client = openai.OpenAI(timeout=self._timeout_seconds, max_retries=self._max_retries)

        with open(file_path, "rb") as audio_file:
            self.progress.emit("worker: request -> openai")
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

        self.progress.emit("worker: done")
        return transcript.text

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
        self.audio_devices = {}
        self.device_colors = {}
        self.spectrum_curves = {}
        self.device_checkboxes = []
        self.tempfile_to_device = {}
        self.pending_transcriptions = 0
        self.session_results = []
        self._active_workers = set()
        self._procs = {}
        
        # Air monitoring state
        self.is_air_monitoring = False
        self.output_device_id = None
        self.output_stream = None
        self.mixed_audio_buffer = None
        self.mixing_sample_rate = 44100
        self.mixing_channels = 2

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

        # Device selection section
        device_layout = QVBoxLayout()
        
        # Input devices section
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
        
        # Output device and air monitor controls in horizontal layout
        output_layout = QHBoxLayout()
        
        # Audio Output Device selection
        output_label = QLabel("Audio Output Device:")
        output_layout.addWidget(output_label)
        
        self.output_device_combo = QComboBox()
        self.output_device_combo.setMinimumWidth(200)
        output_layout.addWidget(self.output_device_combo)
        
        # Air monitor checkbox
        self.air_monitor_checkbox = QCheckBox("Air Monitor")
        self.air_monitor_checkbox.setChecked(False)
        self.air_monitor_checkbox.stateChanged.connect(self.on_air_monitor_changed)
        output_layout.addWidget(self.air_monitor_checkbox)
        
        # Test the connection
        print(f"Air monitor checkbox created, signal connected: {self.air_monitor_checkbox.stateChanged}")
        
        output_layout.addStretch()  # Push controls to the left
        device_layout.addLayout(output_layout)
        
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

        # Populate device checkboxes and output devices with error handling
        print("Populating device lists...")
        try:
            self.populate_device_checkboxes()
            self.populate_output_devices()
            print("Device lists populated successfully")
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
        self.log_message("Air monitor checkbox initialized")
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
    
    def populate_output_devices(self):
        """Populate the output device combo box"""
        try:
            devices_info = get_platform_devices_info(self.p)
            output_devices = devices_info.get("output", [])
            
            self.output_device_combo.clear()
            
            if not output_devices:
                self.output_device_combo.addItem("No output devices found", -1)
                self.log_message("No output devices found")
                return
            
            # Get default output device
            default_devices = get_default_devices(self.p)
            default_output = default_devices.get("output")
            default_name = default_output.get("name", "") if default_output else ""
            default_index = -1
            
            for device in output_devices:
                device_name = device.get("name", "Unknown Device")
                device_index = device.get("index", -1)
                tooltip = f"Index: {device_index} | Host API: {device.get('hostApiName', '')} | DefaultRate: {int(device.get('defaultSampleRate', 44100))} Hz"
                
                self.output_device_combo.addItem(device_name, device_index)
                self.output_device_combo.setItemData(self.output_device_combo.count() - 1, tooltip, Qt.ToolTipRole)
                
                # Mark the default device
                if device_name == default_name:
                    default_index = self.output_device_combo.count() - 1
            
            # Select the default device
            if default_index >= 0:
                self.output_device_combo.setCurrentIndex(default_index)
            
            self.log_message(f"Found {len(output_devices)} output devices")
            
        except Exception as e:
            self.output_device_combo.clear()
            self.output_device_combo.addItem("Error loading output devices", -1)
            self.log_message(f"Error loading output audio devices: {str(e)}")
    
    def on_device_selection_changed(self):
        """Handle device selection change"""
        self.restart_audio_monitoring()
    
    def on_air_monitor_changed(self, state):
        """Handle air monitor checkbox state change"""
        print(f"Air monitor checkbox changed: state={state}, Qt.Checked={Qt.Checked}")
        
        # state is 2 for checked, 0 for unchecked
        is_checked = state == 2
        self.log_message(f"Air monitor checkbox changed to: {'ON' if is_checked else 'OFF'}")
        
        if is_checked:
            print("Starting air monitoring...")
            self.start_air_monitoring()
        else:
            print("Stopping air monitoring...")
            self.stop_air_monitoring()
    
    def start_air_monitoring(self):
        """Start air monitoring - mix input audio and output to selected device"""
        if self.is_air_monitoring:
            return
        
        # Get selected output device
        output_device_id = self.output_device_combo.currentData()
        if output_device_id is None or output_device_id == -1:
            self.log_message("No valid output device selected for air monitoring")
            self.air_monitor_checkbox.setChecked(False)
            return
        
        # Get selected input devices
        selected_inputs = self.get_selected_device_ids()
        if not selected_inputs:
            self.log_message("No input devices selected for air monitoring")
            self.air_monitor_checkbox.setChecked(False)
            return
        
        try:
            # Open output stream
            self.output_device_id = output_device_id
            print(f"Opening output stream: device_id={output_device_id}, rate={self.mixing_sample_rate}, channels={self.mixing_channels}")
            
            self.output_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.mixing_channels,
                rate=self.mixing_sample_rate,
                output=True,
                output_device_index=output_device_id,
                frames_per_buffer=1024
            )
            
            # Start the output stream
            self.output_stream.start_stream()
            
            self.is_air_monitoring = True
            self.log_message(f"Air monitoring started - Output device: {self.output_device_combo.currentText()}")
            print(f"Air monitoring started successfully")
            
        except Exception as e:
            self.log_message(f"Failed to start air monitoring: {str(e)}")
            print(f"Failed to start air monitoring: {str(e)}")
            self.air_monitor_checkbox.setChecked(False)
    
    def stop_air_monitoring(self):
        """Stop air monitoring"""
        if not self.is_air_monitoring:
            return
        
        self.is_air_monitoring = False
        
        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception:
                pass
            self.output_stream = None
        
        self.output_device_id = None
        self.log_message("Air monitoring stopped")
    
    def mix_audio_streams(self, input_streams_data):
        """Mix multiple input audio streams by averaging"""
        if not input_streams_data:
            print("Mix: no input streams data")
            return None
        
        print(f"Mix: processing {len(input_streams_data)} input streams")
        
        # Convert all audio data to the same format and sample rate
        normalized_streams = []
        target_length = 0
        
        for i, (audio_data, sample_rate, channels) in enumerate(input_streams_data):
            if audio_data is None:
                print(f"Mix: stream {i} has no audio data")
                continue
            
            print(f"Mix: stream {i} - {len(audio_data)} samples, {sample_rate}Hz, {channels} channels")
            
            # Ensure audio_data is float for processing
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Resample if necessary (simplified - in practice would need proper resampling)
            if sample_rate != self.mixing_sample_rate:
                # Simple approach: repeat or subsample
                ratio = self.mixing_sample_rate / sample_rate
                if ratio != 1.0:
                    new_length = int(len(audio_data) * ratio)
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data) - 1, new_length),
                        np.arange(len(audio_data)),
                        audio_data
                    )
                    print(f"Mix: resampled stream {i} from {len(input_streams_data[i][0])} to {len(audio_data)} samples")
            
            # Convert to stereo if needed
            if channels == 1 and self.mixing_channels == 2:
                # Duplicate mono to stereo
                audio_data = np.repeat(audio_data, 2)
                print(f"Mix: converted stream {i} from mono to stereo")
            elif channels == 2 and self.mixing_channels == 1:
                # Mix stereo to mono
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                print(f"Mix: converted stream {i} from stereo to mono")
            
            normalized_streams.append(audio_data)
            target_length = max(target_length, len(audio_data))
        
        if not normalized_streams:
            print("Mix: no valid normalized streams")
            return None
        
        print(f"Mix: target length {target_length}, {len(normalized_streams)} streams")
        
        # Pad shorter streams to target length
        for i in range(len(normalized_streams)):
            if len(normalized_streams[i]) < target_length:
                padding = target_length - len(normalized_streams[i])
                normalized_streams[i] = np.pad(normalized_streams[i], (0, padding), 'constant')
        
        # Average the streams
        mixed_audio = np.mean(normalized_streams, axis=0)
        
        # Apply some gain to make it audible
        mixed_audio = mixed_audio * 2.0
        
        # Ensure the output doesn't clip
        mixed_audio = np.clip(mixed_audio, -32767, 32767)
        
        result = mixed_audio.astype(np.int16)
        print(f"Mix: output {len(result)} samples, range [{result.min()}, {result.max()}]")
        
        return result
    
    def delayed_audio_monitoring(self):
        """Start audio monitoring after a delay to avoid blocking initialization"""
        print("Attempting delayed audio monitoring...")
        self.start_audio_monitoring()
    
    def start_audio_monitoring(self):
        """Start audio monitoring for real-time visualization"""
        # Stop existing monitoring devices
        self._stop_devices_by_mode('monitoring')
        
        # Get selected devices
        selected = self.get_selected_device_ids()
        if not selected:
            print("No devices selected for monitoring")
            return

        # Manage spectrum curves for selected devices
        self._manage_spectrum_curves_for_devices(selected)

        # Start monitoring for each selected device
        for device_id, device_name in selected:
            print(f"Starting monitoring with device ID: {device_id}")
            
            device = AudioDevice(device_id, self.p, device_name)
            if device.start_monitoring(self.update_spectrum):
                self.audio_devices[device_id] = device
                print(f"Audio monitoring started for {device_name}")
            else:
                self.log_message(f"Failed to start audio monitoring for {device_name}")
    
    def restart_audio_monitoring(self):
        """Restart audio monitoring with new device"""
        # Simply restart monitoring - start_audio_monitoring handles cleanup
        self.start_audio_monitoring()
    
    def process_audio_data(self):
        """Process audio data from monitoring thread - called by timer on main thread"""
        input_streams_data = []
        
        # Iterate over audio devices and process available data
        for device_id, device in list(self.audio_devices.items()):
            if device.mode == 'monitoring' and device.latest_audio_data is not None:
                audio_data = device.latest_audio_data
                sample_rate = device.latest_sample_rate
                device.latest_audio_data = None
                
                # Update spectrum visualization
                self.update_spectrum_safe(device_id, audio_data, sample_rate)
                
                # Collect for air monitoring if enabled
                if self.is_air_monitoring:
                    print(f"Air monitor: collecting from device {device_id}, {len(audio_data)} samples")
                    input_streams_data.append((audio_data, sample_rate, device.channels))
        
        # Mix and output audio if air monitoring is enabled
        if self.is_air_monitoring and input_streams_data and self.output_stream:
            try:
                mixed_audio = self.mix_audio_streams(input_streams_data)
                if mixed_audio is not None:
                    # Write to output stream
                    audio_bytes = mixed_audio.tobytes()
                    if len(audio_bytes) > 0:
                        self.output_stream.write(audio_bytes, exception_on_underflow=False)
                        print(f"Air monitor: wrote {len(audio_bytes)} bytes to output")
                else:
                    print("Air monitor: mixed_audio is None")
            except Exception as e:
                print(f"Air monitoring output error: {e}")
                self.log_message(f"Air monitoring output error: {e}")
    
    def update_spectrum(self, device_id, audio_data, sample_rate):
        """Update spectrum - called from audio thread, emit signal for thread safety"""
        self.spectrum_update_signal.emit(device_id, audio_data, sample_rate)
    
    @Slot(int, object, int)
    def update_spectrum_safe(self, device_id, audio_data, sample_rate):
        """Update the frequency spectrum plot - runs on main GUI thread"""
        try:
            # Compute spectrum using utility function
            freqs_filtered, magnitude_filtered = compute_audio_spectrum(audio_data, sample_rate)
            
            if freqs_filtered is not None and magnitude_filtered is not None:
                # Ensure curve exists for this device
                self._ensure_spectrum_curve(device_id)
                # Update the curve with new data
                self.spectrum_curves[device_id].setData(freqs_filtered, magnitude_filtered)
        
        except Exception as e:
            print(f"Spectrum update error: {e}")

    def record_chunks(self):
        """Record chunks from all active recorders"""
        if not self.is_recording:
            return
            
        selected_ids = set([d for d, _ in self.get_selected_device_ids()])
        for device_id, device in list(self.audio_devices.items()):
            if device.mode != 'recording':
                continue
                
            # If device is no longer selected, remove its curve and skip plotting
            if device_id not in selected_ids:
                self._remove_device_curve(device_id)
                # Still drain audio to keep buffers healthy, but don't plot
                try:
                    device.record_chunk()
                except Exception:
                    pass
                continue
                
            success = device.record_chunk()
            if success:
                audio_data = device.get_latest_audio_data()
                if audio_data is not None:
                    self.update_spectrum_safe(device_id, audio_data, device.sample_rate)
    
    def toggle_recording(self):
        """Toggle recording on button click"""
        # Use actual recording state instead of button state to avoid sync issues
        if not self.is_recording:
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
            # Reset button to unchecked state since recording failed to start
            self._update_button_state(checked=False)
            return

        # Reset session results for a new recording session
        self.session_results = []

        # Stop monitoring to avoid device conflicts
        self._stop_devices_by_mode('monitoring')
        self.log_message("Stopped audio monitoring during recording")

        # Create and start recorders for each selected device
        start_failures = []
        for device_id, device_name in selected:
            device = AudioDevice(device_id, self.p, device_name)
            result = device.start_recording()
            if result is True:
                self.audio_devices[device_id] = device
                # Ensure visualization is set up  
                self._ensure_spectrum_curve(device_id)
            else:
                err = result[1] if isinstance(result, tuple) else "Unknown error"
                start_failures.append((device_name, err))

        recording_devices = [d for d in self.audio_devices.values() if d.mode == 'recording']
        if recording_devices:
            self.is_recording = True
            self._update_button_state(
                text="Recording... (Click again to stop)",
                style="background-color: #ff4444; color: white;",
                checked=True,
                force_refresh=True
            )
            # Use a conservative timer interval; recorders drain available data internally
            interval_ms = 30
            self.recording_timer.start(interval_ms)
            self.log_message(
                f"Recording timer set to {interval_ms} ms across {len(recording_devices)} devices"
            )
            names = ", ".join([device.device_name for device in recording_devices])
            self.log_message(f"Recording started with devices: {names}")
            if start_failures:
                for name, err in start_failures:
                    self.log_message(f"Failed to start device '{name}': {err}")
        else:
            if start_failures:
                for name, err in start_failures:
                    self.log_message(f"Failed to start device '{name}': {err}")
            self.log_message("Failed to start recording on any device")
            # Reset button to unchecked state since recording failed to start
            self._update_button_state(checked=False)
            self.start_audio_monitoring()

    def stop_recording(self):
        """Stop recording when button is toggled off"""
        if not self.is_recording:
            return

        self.recording_timer.stop()
        self.is_recording = False

        # UI update first
        self._update_button_state(
            text="Processing...",
            style="background-color: #ffaa00; color: white;",
            enabled=False,
            checked=False,
            force_refresh=True
        )

        QTimer.singleShot(0, self._after_stop_recording)

    def _after_stop_recording(self):
        # Stop all recorders and gather frames
        any_frames = False
        stopped_devices = []
        for device_id in list(self.audio_devices.keys()):
            device = self.audio_devices[device_id]
            if device.mode == 'recording':
                frames = device.stop()
                stopped_devices.append(device)
                if frames:
                    any_frames = True
                del self.audio_devices[device_id]
        self.log_message("Recording stopped")

    # Don't restart audio monitoring during transcription; resume after finalize

        if not any_frames:
            self.log_message("No audio data recorded")
            self.reset_button()
            return

        # Save and transcribe each recorded device
        self.pending_transcriptions = 0
        for device in stopped_devices:
            if not device.frames:
                continue
            # Save to recordings folder first
            success, result = device.save_to_recordings_folder()
            if success:
                self.log_message(f"Audio saved to: {result}")
            else:
                self.log_message(f"Failed to save recording for {device.device_name}: {result}")
            # Create temp file for transcription
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            # Map temp file to device, id, and final saved recording path
            self.tempfile_to_device[temp_file.name] = (device.device_name, device.device_id, result if success else None)
            # Save WAV and transcribe in single worker
            worker = AudioProcessingWorker(audio_device=device, temp_file_name=temp_file.name, mode='save_and_transcribe')
            worker.finished.connect(self.on_transcription_finished)
            worker.error.connect(self.on_transcription_error)
            worker.progress.connect(self._on_worker_progress)
            # Retain until completion
            self._active_workers.add(worker)
            worker.finished.connect(lambda *_, w=worker: self._active_workers.discard(w))
            worker.error.connect(lambda *_, w=worker: self._active_workers.discard(w))
            self.thread_pool.start(worker)
            self.pending_transcriptions += 1

        if self.pending_transcriptions > 0:
            self._ensure_heartbeat_active()


    def transcribe_audio(self, audio_file_path):
        """Transcribe audio file using OpenAI Whisper API"""
        self.log_message("Starting transcription with OpenAI Whisper...")
        # Prefer subprocess isolation to guarantee the UI process never blocks
        self._start_transcription_subprocess(audio_file_path)
        
        # Ensure heartbeat is running
        self._ensure_heartbeat_active()

    def _start_transcription_subprocess(self, audio_file_path):
        """Start transcription subprocess with fallback to thread worker"""
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
            self._start_transcription_thread(audio_file_path)

    def _start_transcription_thread(self, audio_file_path):
        """Start transcription in thread worker"""
        worker = AudioProcessingWorker(audio_file_path=audio_file_path, mode='transcribe')
        worker.finished.connect(self.on_transcription_finished)
        worker.error.connect(self.on_transcription_error)
        worker.progress.connect(self._on_worker_progress)
        self._active_workers.add(worker)
        worker.finished.connect(lambda *_, w=worker: self._active_workers.discard(w))
        worker.error.connect(lambda *_, w=worker: self._active_workers.discard(w))
        self.thread_pool.start(worker)
        self.log_message("Transcription dispatched to background thread")

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
    
    def _update_button_state(self, text="Push to Text", style="", enabled=True, checked=False, force_refresh=False):
        """Unified method to update Push to Text button state"""
        self.push_to_text_button.setText(text)
        self.push_to_text_button.setStyleSheet(style)
        self.push_to_text_button.setEnabled(enabled)
        self.push_to_text_button.setChecked(checked)
        
        if force_refresh:
            self.push_to_text_button.repaint()
            QApplication.processEvents()

    def _ensure_heartbeat_active(self):
        """Ensure processing heartbeat timer is running"""
        try:
            if not self.processing_heartbeat.isActive():
                self.processing_heartbeat.start()
        except Exception:
            pass

    def _stop_heartbeat(self):
        """Stop processing heartbeat timer safely"""
        try:
            self.processing_heartbeat.stop()
        except Exception:
            pass

    def reset_button(self):
        """Reset the Push to Text button to its default state"""
        self._update_button_state()

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
            self._stop_heartbeat()
            self.reset_button()
            # Now that transcription work is done, resume audio monitoring
            self.start_audio_monitoring()

    def _processing_heartbeat_tick(self):
        """Emit a periodic log while transcriptions are pending."""
        if self.pending_transcriptions > 0:
            self.log_message(f"Transcription pending: {self.pending_transcriptions}")
        else:
            self._stop_heartbeat()

    def closeEvent(self, event):
        """Clean up when closing the application"""
        if self.is_recording:
            self.stop_recording()
        
        # Stop air monitoring
        if self.is_air_monitoring:
            self.stop_air_monitoring()
        
        # Stop all audio devices
        for device in list(self.audio_devices.values()):
            try:
                device.stop()
            except Exception:
                pass
        self.audio_devices.clear()
        
        if hasattr(self, 'p'):
            self.p.terminate()
        event.accept()

    def get_selected_device_ids(self):
        """Get a list of (device_id, device_name) for selected devices"""
        return [(device_id, device_name) for cb, device_id, device_name in self.device_checkboxes if cb.isChecked()]
    
    def log_message(self, message):
        """Add a message to the log text area"""
        import datetime as _dt
        timestamp = _dt.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def get_device_name_by_id(self, device_id):
        """Lookup a device name by its device_id from the checkbox list."""
        return next((name for cb, did, name in self.device_checkboxes if did == device_id), None)

    def _stop_devices_by_mode(self, mode):
        """Stop all devices operating in the specified mode."""
        for device_id in list(self.audio_devices.keys()):
            device = self.audio_devices[device_id]
            if device.mode == mode:
                try:
                    device.stop()
                except Exception:
                    pass
                del self.audio_devices[device_id]

    def _cleanup_unselected_curves(self, selected_devices):
        """Remove visualization curves for devices that are no longer selected."""
        selected_ids = set([d for d, _ in selected_devices])
        for device_id in list(self.spectrum_curves.keys()):
            if device_id not in selected_ids:
                self._remove_device_curve(device_id)

    def _manage_spectrum_curves_for_devices(self, selected_devices):
        """Comprehensive spectrum curve management for selected devices."""
        # Clean up curves for unselected devices
        self._cleanup_unselected_curves(selected_devices)
        # Ensure curves exist for selected devices
        for device_id, device_name in selected_devices:
            self._ensure_device_visualization(device_id, device_name)

    def _clear_all_spectrum_curves(self):
        """Remove all spectrum curves and clear related data."""
        for device_id in list(self.spectrum_curves.keys()):
            self._remove_device_curve(device_id)
        self.device_colors.clear()

    def _remove_device_curve(self, device_id):
        """Remove a specific device's visualization curve and legend entry."""
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

    def _ensure_device_color(self, device_id):
        """Assign color to device if not already assigned."""
        if device_id not in self.device_colors:
            self.device_colors[device_id] = pg.intColor(len(self.device_colors))
        return self.device_colors[device_id]

    def _ensure_spectrum_curve(self, device_id):
        """Ensure spectrum curve exists for device."""
        if device_id not in self.spectrum_curves:
            color = self._ensure_device_color(device_id)
            device_name = self.get_device_name_by_id(device_id) or f"Device {device_id}"
            self.spectrum_curves[device_id] = self.spectrum_widget.plot(
                pen=color, 
                width=2, 
                name=device_name
            )

    def _ensure_device_visualization(self, device_id, device_name):
        """Ensure device has assigned color and spectrum curve."""
        self._ensure_device_color(device_id)
        self._ensure_spectrum_curve(device_id)

    def test_plot(self):
        """Test if the plot widget is working"""
        try:
            # Create some test data - a simple declining curve
            test_freqs = np.logspace(np.log10(100), np.log10(10000), 50)
            test_magnitudes = -20 - 10 * np.log10(test_freqs/1000)
            
            print(f"Testing plot with {len(test_freqs)} points")
            # Create a temporary curve for test
            temp_curve = self.spectrum_widget.plot(pen='y', width=2, name="Test Signal")
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

