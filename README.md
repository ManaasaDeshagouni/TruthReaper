# 🔍 TruthReaper

## 💡 Project Overview
TruthReaper is an AI-powered real-time voice-based **lie and stress detection system** that analyzes human speech for hesitation, disfluency, and behavioral patterns. It is designed to assist in scenarios like virtual interrogations, mock interviews, or psychological research by identifying potential indicators of deception and emotional tension.

The system uses:
- 🎙️ Live or pre-recorded voice input
- 🤖 Whisper (offline, Hugging Face version) for transcription
- 🧠 Custom logic to auto-label speakers as [OFFICER] or [ACCUSED]
- ⏱️ Pause-based auto-stopping during real-time recording

---

## ✅ Current Features

### 🎧 Dual Mode Transcriber
Choose between:
1. **Real-Time Recording**: Record voice using mic (max 2 minutes), stops automatically after 15s of silence
2. **Pre-Recorded Audio**: Use your `.wav` file (interviews, conversations, etc.)

### 📝 Whisper Transcription
- Fully offline transcription using Hugging Face’s `whisper-base`
- Automatically resamples any input audio to 16kHz

### 🗣 Speaker Labeling
- Automatically tags each sentence as:
  - `[OFFICER]` if it ends with a `?`
  - `[ACCUSED]` otherwise
- Saves output as `recordings/marked_transcript.txt`

### 🔇 Silence Detection (Real-Time Mode)
- Records up to 2 mins OR
- Stops early if there's **15 seconds of silence**

---

## 🗂 Project Structure
```
TruthReaper/
├── audio_processor.py                # Original prototype (kept for backup)
├── truth_reaper_transcriber.py       # 🔥 Dual-mode transcriber with auto-tagging
├── models/
│   └── whisper-base/                 # Hugging Face Whisper model files
├── recordings/
│   ├── realtime.wav                  # Recorded real-time input
│   ├── interview1.wav               # Sample input (optional)
│   └── marked_transcript.txt        # Final output transcript
├── env/                              # Virtual environment (add to .gitignore)
├── requirements.txt                 # Required packages
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/TruthReaper.git
cd TruthReaper
```

### 2. Set up virtual environment (optional but recommended)
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Whisper Base Model from Hugging Face
Go to: https://huggingface.co/openai/whisper-base/tree/main  
Place the following files in `models/whisper-base/`:
- config.json
- merges.txt
- preprocessor_config.json
- pytorch_model.bin
- tokenizer.json
- vocab.json

---

## 🚀 How to Run the Project

### 1. Run the main tool
```bash
python3 truth_reaper_transcriber.py
```

### 2. Choose mode
```
[1] Real-Time Live Recording (max 2 mins, stops if 15s silence)
[2] Use Pre-Recorded Interview Audio
```

### 3. Output
You’ll get a clear, labeled transcription in:
```
recordings/marked_transcript.txt
```

---

## 🧪 What’s Next (To Be Developed)
- `pause_analyzer.py`: Analyze silent gaps, hesitation patterns
- `emotion_detector.py`: Extract pitch, tone, and stress markers using Librosa
- `lie_predictor.py`: Combine behavioral and audio features to predict deception (Random Forest → LSTM → Transformer upgrade)
- Web Interface (Flask/Streamlit) for cleaner UX

---

## 🙋 Team Handoff Notes
- Everything is modular, commented, and extensible
- Use real voice input for best results
- Model accuracy will improve with clean, controlled input
- Avoid background noise in live testing

Feel free to contact the current lead (Manasa) for guidance or merge approvals 😎

---

## ⚠️ Setup Notes & Common Pitfalls
> These issues were faced during development. Follow these tips to avoid them:

### 🔁 Python Version Compatibility
- Use **Python 3.10 or 3.9**
- Avoid Python 3.12 (some packages like `pyaudio` or Whisper’s dependencies break)

### 🎙 PyAudio Installation Issues
- If `pyaudio` fails:
  ```bash
  brew install portaudio
  pip install pyaudio
  ```
- On Linux:
  ```bash
  sudo apt install portaudio19-dev python3-pyaudio
  ```

### 🔇 Whisper Transcription Problems
- Avoid using `tiny` model — it's not accurate
- Use `whisper-base` from Hugging Face
- If you get:
  ```
  ValueError: You have explicitly specified forced_decoder_ids
  ```
  then add this line **right after loading the model**:
  ```python
  model.config.forced_decoder_ids = None
  model.config.suppress_tokens = []
  ```

### 📉 Sample Rate Errors (44100 vs 16000)
- If Whisper throws:
  ```
  sampling rate must be 16000
  ```
  that means your mic/audio file is at 44100 Hz.
- This is already handled with `torchaudio.transforms.Resample()` in the code.

### 🔐 SSL Errors on Whisper Download
- If model download fails with SSL cert errors:
  - Use a **VPN**
  - OR **download the model manually** from Hugging Face:
    https://huggingface.co/openai/whisper-base/tree/main

### 📦 Required Python Packages
Make sure `requirements.txt` includes:
```txt
transformers
torch
torchaudio
pyaudio
librosa
audioop
```
Then install:
```bash
pip install -r requirements.txt
```

✅ If anything breaks — ping the last committer. They went through hell so you don’t have to 🙃

---

## ✨ Built with
- 🤖 [Hugging Face Transformers](https://huggingface.co/transformers/)
- 🔊 [Torchaudio](https://pytorch.org/audio/)
- 🎧 [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
- ❤️ Collaboration, fire, and vision

---

Let’s build AI that doesn’t just hear you — it **reads your truth.** 😈🖤
