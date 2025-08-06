import wave
import tempfile
import os
import threading
import time
import pyaudiowpatch as pyaudio
import openai
from dotenv import load_dotenv

from PySide6.QtCore import QThread, Signal, QRunnable, QThreadPool, QObject, Slot, QEvent, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QMessageBox, QSizePolicy, QTextEdit
)

VERSION = "20250806"

from util_audio import (
    get_wasapi_devices_info,
    get_device_info_by_id,
    get_wasapi_default_input_device,
)

load_dotenv()

class AudioRecorder:
    def __init__(self, device_id, p):
        self.device_id = device_id
        self.p = p
        self.stream = None
        self.recording = False
        self.frames = []
        
    def start_recording(self):
        """Start recording audio"""
        try:
            device_info = get_device_info_by_id(self.device_id, self.p)  # Fixed parameter order
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
                return True
            except Exception as e:
                return False
        return False
    
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

class TranscriptionWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, audio_file_path):
        super().__init__()
        self.audio_file_path = audio_file_path
    
    @Slot()
    def transcribe(self):
        """Transcribe audio using OpenAI Whisper API"""
        try:
            client = openai.OpenAI()
            
            with open(self.audio_file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            self.finished.emit(transcript.text)
            
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Push to Text")
        self.setMinimumSize(600, 400)
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Recording state
        self.is_recording = False
        self.recorder = None
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.record_chunk)
        
        # Thread pool for transcription
        self.thread_pool = QThreadPool()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Device selection section
        device_layout = QHBoxLayout()
        device_label = QLabel("WASAPI Input Device:")
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        main_layout.addLayout(device_layout)
        
        # Push to Text button
        self.push_to_text_button = QPushButton("Push to Text")
        self.push_to_text_button.setMinimumHeight(40)
        self.push_to_text_button.pressed.connect(self.start_recording)
        self.push_to_text_button.released.connect(self.stop_recording)
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
        
        # Populate device dropdown
        self.populate_device_combo()
        
        # Log initialization
        self.log_message("Application initialized")
        self.log_message("Hold down 'Push to Text' button to record audio")

    def populate_device_combo(self):
        """Populate the device combo box with WASAPI input devices"""
        try:
            devices_info = get_wasapi_devices_info(self.p)
            input_devices = devices_info.get("input", [])
            
            self.device_combo.clear()
            
            if not input_devices:
                self.device_combo.addItem("No WASAPI input devices found")
                self.log_message("No WASAPI input devices found")
                return
            
            # Add devices to combo box
            for device in input_devices:
                device_name = device.get("name", "Unknown Device")
                device_index = device.get("index", -1)
                self.device_combo.addItem(device_name, device_index)
            
            # Try to select default input device
            default_device = get_wasapi_default_input_device(self.p)
            if default_device:
                default_name = default_device.get("name", "")
                for i in range(self.device_combo.count()):
                    if self.device_combo.itemText(i) == default_name:
                        self.device_combo.setCurrentIndex(i)
                        break
            
            self.log_message(f"Found {len(input_devices)} WASAPI input devices")
            
        except Exception as e:
            self.device_combo.addItem("Error loading devices")
            self.log_message(f"Error loading WASAPI devices: {str(e)}")
    
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
    
    def start_recording(self):
        """Start recording when button is pressed"""
        if self.is_recording:
            return
        
        device_id = self.get_selected_device_id()
        if device_id is None:
            self.log_message("No device selected")
            return
        
        self.recorder = AudioRecorder(device_id, self.p)
        result = self.recorder.start_recording()
        
        if result is True:
            self.is_recording = True
            self.push_to_text_button.setText("Recording... (Release to stop)")
            self.push_to_text_button.setStyleSheet("background-color: #ff4444; color: white;")
            self.recording_timer.start(50)  # Record chunks every 50ms
            device_name = self.device_combo.currentText()
            self.log_message(f"Recording started with device: {device_name}")
        else:
            error_msg = result[1] if isinstance(result, tuple) else "Failed to start recording"
            self.log_message(f"Failed to start recording: {error_msg}")
    
    def record_chunk(self):
        """Record a chunk of audio data"""
        if self.recorder and self.is_recording:
            self.recorder.record_chunk()
    
    def stop_recording(self):
        """Stop recording when button is released"""
        if not self.is_recording:
            return
        
        self.recording_timer.stop()
        self.is_recording = False
        self.push_to_text_button.setText("Processing...")
        self.push_to_text_button.setStyleSheet("background-color: #ffaa00; color: white;")
        self.push_to_text_button.setEnabled(False)
        
        # Stop recording and get frames
        frames = self.recorder.stop_recording()
        self.log_message("Recording stopped")
        
        if not frames:
            self.log_message("No audio data recorded")
            self.reset_button()
            return
        
        # Save to temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.close()
        
        if self.recorder.save_to_wav(temp_file.name):
            self.log_message(f"Audio saved to temporary file: {temp_file.name}")
            self.transcribe_audio(temp_file.name)
        else:
            self.log_message("Failed to save audio file")
            self.reset_button()
            os.unlink(temp_file.name)
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio file using OpenAI Whisper API"""
        self.log_message("Starting transcription with OpenAI Whisper...")
        
        # Create worker thread for transcription
        worker = TranscriptionWorker(audio_file_path)
        worker.finished.connect(self.on_transcription_finished)
        worker.error.connect(self.on_transcription_error)
        
        # Run transcription in separate thread
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.transcribe)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(lambda: self.cleanup_transcription(thread, worker, audio_file_path))
        thread.start()
    
    def on_transcription_finished(self, transcription_text):
        """Handle successful transcription"""
        self.log_message("Transcription completed successfully")
        self.output_text.append(transcription_text)
        self.output_text.append("")  # Add blank line
        self.reset_button()
    
    def on_transcription_error(self, error_message):
        """Handle transcription error"""
        self.log_message(f"Transcription error: {error_message}")
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("Transcription Error")
        error_msg.setText(f"Failed to transcribe audio:\n{error_message}")
        error_msg.exec()
        self.reset_button()
    
    def cleanup_transcription(self, thread, worker, audio_file_path):
        """Clean up transcription thread and temporary file"""
        thread.deleteLater()
        worker.deleteLater()
        
        # Delete temporary audio file
        try:
            os.unlink(audio_file_path)
            self.log_message("Temporary audio file deleted")
        except Exception as e:
            self.log_message(f"Failed to delete temporary file: {e}")
    
    def reset_button(self):
        """Reset the Push to Text button to its default state"""
        self.push_to_text_button.setText("Push to Text")
        self.push_to_text_button.setStyleSheet("")
        self.push_to_text_button.setEnabled(True)

    def closeEvent(self, event):
        """Clean up when closing the application"""
        if self.is_recording:
            self.stop_recording()
        
        if hasattr(self, 'p'):
            self.p.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

