import os
import sys
import json
from dotenv import load_dotenv
import openai


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing argument: audio_file_path"}))
        return 2

    audio_path = sys.argv[1]
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        print(json.dumps({"error": "OPENAI_API_KEY is not set"}))
        return 3

    try:
        client = openai.OpenAI(timeout=45, max_retries=0)
        with open(audio_path, "rb") as f:
            t = client.audio.transcriptions.create(model="whisper-1", file=f)
        print(json.dumps({"text": t.text}))
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
