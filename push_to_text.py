import wave
import tempfile
import os
import threading
import time
import platform
from datetime import datetime

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
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject, Qt
from PySide6.QtCore import QThread, Signal, QObject, Slot, QEvent, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QMessageBox, QSizePolicy, QTextEdit
)

VERSION = "20250807"

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
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
                self.latest_chunk = data
                return True
            except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
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
                
                # Call callback in a thread-safe way
                # Don't call the callback directly from audio thread - can cause Qt issues
                # Instead, we'll store the data and let a timer process it
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
    
    def __init__(self, audio_file_path):
        super().__init__()
        self.audio_file_path = audio_file_path
        self.signals = self.Signals()
        self.finished = self.signals.finished
        self.error = self.signals.error
    
    def run(self):
        """Transcribe audio using OpenAI Whisper API"""
        try:
            # Use a separate thread executor to ensure the API call doesn't block
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._transcribe_sync)
                transcript_text = future.result(timeout=60)  # 60 second timeout
            
            self.finished.emit(transcript_text, self.audio_file_path)
            
        except Exception as e:
            self.error.emit(str(e), self.audio_file_path)
        finally:
            # Clean up the temporary file
            try:
                os.unlink(self.audio_file_path)
            except:
                pass
    
    def _transcribe_sync(self):
        """Synchronous transcription method"""
        client = openai.OpenAI()
        
        with open(self.audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
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
    spectrum_update_signal = Signal(np.ndarray, int)

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
        
        # Recording state
        self.is_recording = False
        self.recorder = None
        self.monitor = None
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.record_chunk)
        
        # Connect the spectrum update signal to the actual update method
        self.spectrum_update_signal.connect(self.update_spectrum_safe)
        
        # Thread pool for transcription - increase max threads
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)  # Allow multiple concurrent operations
        
        # Timer for processing audio data from monitoring thread
        self.spectrum_timer = QTimer()
        self.spectrum_timer.timeout.connect(self.process_audio_data)
        self.spectrum_timer.start(50)  # Process every 50ms
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Device selection section
        device_layout = QHBoxLayout()
        device_label = QLabel("Audio Input Device:")
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
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
        
        # Create spectrum plot line
        self.spectrum_curve = self.spectrum_widget.plot(pen='y', width=2)
        
        main_layout.addWidget(self.spectrum_widget)
        
        # Push to Text button
        self.push_to_text_button = QPushButton("Push to Text")
        self.push_to_text_button.setMinimumHeight(40)
        self.push_to_text_button.setCheckable(True)  # Make it a toggle button
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
        
        # Populate device dropdown with error handling
        print("Populating device combo...")
        try:
            self.populate_device_combo()
            print("Device combo populated successfully")
        except Exception as e:
            print(f"Device combo population failed: {e}")
            raise
        
        # Start audio monitoring for visualization with error handling
        print("Starting audio monitoring...")
        try:
            # Use a timer to delay audio monitoring to avoid blocking initialization
            QTimer.singleShot(1000, self.delayed_audio_monitoring)
            print("Audio monitoring will start in 1 second...")
        except Exception as e:
            print(f"Audio monitoring setup failed: {e}")
            # Don't raise here, monitoring can fail but app should still work
        
        # Log initialization
        self.log_message("Application initialized")
        self.log_message("Hold down 'Push to Text' button to record audio")
        print("MainWindow __init__ completed")

    def populate_device_combo(self):
        """Populate the device combo box with input devices"""
        try:
            devices_info = get_platform_devices_info(self.p)
            input_devices = devices_info.get("input", [])
            
            self.device_combo.clear()
            
            if not input_devices:
                self.device_combo.addItem("No input devices found")
                self.log_message("No input devices found")
                return
            
            # Add devices to combo box
            for device in input_devices:
                device_name = device.get("name", "Unknown Device")
                device_index = device.get("index", -1)
                self.device_combo.addItem(device_name, device_index)
            
            # Try to select default input device
            default_device = get_default_input_device_cross_platform(self.p)
            if default_device:
                default_name = default_device.get("name", "")
                for i in range(self.device_combo.count()):
                    if self.device_combo.itemText(i) == default_name:
                        self.device_combo.setCurrentIndex(i)
                        break
            
            self.log_message(f"Found {len(input_devices)} input devices")
            
        except Exception as e:
            self.device_combo.addItem("Error loading devices")
            self.log_message(f"Error loading audio devices: {str(e)}")
    
    def on_device_changed(self):
        """Handle device selection change"""
        self.restart_audio_monitoring()
    
    def delayed_audio_monitoring(self):
        """Start audio monitoring after a delay to avoid blocking initialization"""
        print("Attempting delayed audio monitoring...")
        self.start_audio_monitoring()
    
    def start_audio_monitoring(self):
        """Start audio monitoring for real-time visualization"""
        # Stop any existing monitor first
        if self.monitor:
            print("Stopping existing audio monitor...")
            self.monitor.stop_monitoring()
            self.monitor = None
        
        device_id = self.get_selected_device_id()
        if device_id is None:
            print("No device selected for monitoring")
            return
        
        print(f"Starting monitoring with device ID: {device_id}")
        self.monitor = AudioMonitor(device_id, self.p, self.update_spectrum)
        if self.monitor.start_monitoring():
            print("Audio monitoring started successfully")
            self.log_message("Audio monitoring started")
        else:
            print("Failed to start audio monitoring")
            self.log_message("Failed to start audio monitoring")
    
    def restart_audio_monitoring(self):
        """Restart audio monitoring with new device"""
        if self.monitor:
            self.monitor.stop_monitoring()
        self.start_audio_monitoring()
    
    def process_audio_data(self):
        """Process audio data from monitoring thread - called by timer on main thread"""
        if self.monitor and hasattr(self.monitor, 'latest_audio_data') and self.monitor.latest_audio_data is not None:
            # Get the latest data from the monitor
            audio_data = self.monitor.latest_audio_data
            sample_rate = self.monitor.latest_sample_rate
            
            # Clear the data to avoid processing the same data multiple times
            self.monitor.latest_audio_data = None
            
            # Update spectrum on main thread
            self.update_spectrum_safe(audio_data, sample_rate)
    
    def update_spectrum(self, audio_data, sample_rate):
        """Update spectrum - called from audio thread, emit signal for thread safety"""
        # Emit signal to update on the main thread
        self.spectrum_update_signal.emit(audio_data, sample_rate)
    
    @Slot(np.ndarray, int)
    def update_spectrum_safe(self, audio_data, sample_rate):
        """Update the frequency spectrum plot - runs on main GUI thread"""
        try:
            # Apply window function to reduce spectral leakage
            windowed_data = audio_data.astype(np.float64) * np.hanning(len(audio_data))
            
            # Compute FFT
            fft = np.fft.rfft(windowed_data)
            magnitude = np.abs(fft)
            
            # Normalize magnitude to the maximum possible value for 16-bit audio
            # For 16-bit audio, max value is 32767
            max_possible = 32767 * len(audio_data) / 2  # FFT scaling factor
            magnitude_normalized = magnitude / max_possible
            
            # Convert to dB with proper reference (0 dB = maximum possible level)
            magnitude_db = 20 * np.log10(magnitude_normalized + 1e-10)
            
            # Create frequency array
            freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
            
            # Filter out DC and very low frequencies, and limit to audible range
            valid_indices = (freqs >= 20) & (freqs <= 20000)
            freqs_filtered = freqs[valid_indices]
            magnitude_filtered = magnitude_db[valid_indices]
            
            if len(freqs_filtered) > 0:
                # Update the plot on the main thread
                self.spectrum_curve.setData(freqs_filtered, magnitude_filtered)
        
        except Exception as e:
            print(f"Spectrum update error: {e}")

    def record_chunk(self):
        """Record a chunk of audio data"""
        if self.recorder and self.is_recording:
            success = self.recorder.record_chunk()
            if success:
                # Get the latest audio data for spectrum display
                audio_data = self.recorder.get_latest_audio_data()
                if audio_data is not None:
                    self.update_spectrum(audio_data, self.recorder.sample_rate)
    
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

        device_id = self.get_selected_device_id()
        if device_id is None:
            self.log_message("No device selected")
            self.push_to_text_button.setChecked(False)
            return

        # Stop monitoring to avoid device conflicts
        if self.monitor:
            self.monitor.stop_monitoring()
            self.log_message("Stopped audio monitoring during recording")

        device_name = self.device_combo.currentText()
        self.recorder = AudioRecorder(device_id, self.p, device_name)
        result = self.recorder.start_recording()

        if result is True:
            self.is_recording = True
            self.push_to_text_button.setText("Recording... (Click again to stop)")
            self.push_to_text_button.setStyleSheet("background-color: #ff4444; color: white;")
            self.push_to_text_button.repaint()
            QApplication.processEvents()
            self.recording_timer.start(50)
            device_name = self.device_combo.currentText()
            self.log_message(f"Recording started with device: {device_name}")
        else:
            error_msg = result[1] if isinstance(result, tuple) else "Failed to start recording"
            self.log_message(f"Failed to start recording: {error_msg}")
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
        frames = self.recorder.stop_recording()
        self.log_message("Recording stopped")

        # Restart audio monitoring for visualization
        self.start_audio_monitoring()

        if not frames:
            self.log_message("No audio data recorded")
            self.reset_button()
            return

        # Save to recordings folder first
        success, result = self.recorder.save_to_recordings_folder()
        if success:
            self.log_message(f"Audio saved to: {result}")
        else:
            self.log_message(f"Failed to save recording: {result}")

        # Also create temp file for transcription
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.close()

        # Save WAV in background
        worker = SaveAndTranscribeWorker(self.recorder, temp_file.name)
        worker.finished.connect(self.transcribe_audio)
        worker.error.connect(self._on_save_audio_error)
        self.thread_pool.start(worker)

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
        
        # Use QRunnable for transcription to avoid blocking
        worker = TranscriptionWorker(audio_file_path)
        worker.finished.connect(self.on_transcription_finished)
        worker.error.connect(self.on_transcription_error)
        
        # Run transcription in thread pool (non-blocking)
        self.thread_pool.start(worker)
    
    def on_transcription_finished(self, transcription_text, file_path):
        """Handle successful transcription"""
        self.log_message("Transcription completed successfully")
        self.log_message("Temporary audio file deleted")
        self.output_text.append(transcription_text)
        self.output_text.append("")  # Add blank line
        self.reset_button()
    
    def on_transcription_error(self, error_message, file_path):
        """Handle transcription error"""
        self.log_message(f"Transcription error: {error_message}")
        self.log_message("Temporary audio file deleted")
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("Transcription Error")
        error_msg.setText(f"Failed to transcribe audio:\n{error_message}")
        error_msg.exec()
        self.reset_button()
        
        # Clean up temp file
        try:
            os.unlink(file_path)
        except:
            pass
    
    def reset_button(self):
        """Reset the Push to Text button to its default state"""
        self.push_to_text_button.setChecked(False)
        self.push_to_text_button.setText("Push to Text")
        self.push_to_text_button.setStyleSheet("")
        self.push_to_text_button.setEnabled(True)

    def closeEvent(self, event):
        """Clean up when closing the application"""
        if self.is_recording:
            self.stop_recording()
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        if hasattr(self, 'p'):
            self.p.terminate()
        event.accept()

    def get_selected_device_id(self):
        """Get the device ID of the currently selected device"""
        current_index = self.device_combo.currentIndex()
        if current_index >= 0:
            return self.device_combo.itemData(current_index)
        return None
    
    def log_message(self, message):
        """Add a message to the log text area"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def test_plot(self):
        """Test if the plot widget is working"""
        try:
            # Create some test data - a simple declining curve
            test_freqs = np.logspace(np.log10(100), np.log10(10000), 50)
            test_magnitudes = -20 - 10 * np.log10(test_freqs/1000)
            
            print(f"Testing plot with {len(test_freqs)} points")
            self.spectrum_curve.setData(test_freqs, test_magnitudes)
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

