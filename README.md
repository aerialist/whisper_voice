## Push to Text

This app is to record audio from microphone(s) and convert it to text using OpenAI's Whisper model. The app's purpose is to help development of throat microphone to improve its audio quality similar to normal microphones.

## For MacOS

use pyaudio instead of pyaudiowpatch

brew install portaudio
pip install pyaudio

## TODO

[] Compare transcription models: whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe
[] Apply FIR filter realtime
[x] Add air monitor checkbox