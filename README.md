# TruthReaper – Real-Time Deception Detection from Speech and Text

**Authors:**  
- Manasa Deshagouni  
- Dheeraj Kumar Alla  
**Institution:** San Jose State University  
**Course:** CS286 – Advanced Topics in Computer Science  
**Project Type:** Final Research Submission

---

## 📌 Overview

TruthReaper is a **dual-track deception detection system** designed to classify spoken statements as **truthful** or **deceptive**. The system leverages both acoustic and semantic signals from speech through two complementary machine learning pipelines:

---

### 🔁 Track 1 – Hybrid LSTM (Manasa)

- Extracts **sequential time-series features** from `.wav` audio:
  - `pitch_seq`, `energy_seq`, `hesitation_seq`, `disfluency_seq`
- Computes 12 **summary features** like:
  - pause duration, disfluency rate, pitch variance, etc.
- Combines both using a **Bidirectional LSTM**
- Trains using **weighted cross-entropy**
- Augments training data using a **conditional GAN**
- Supports **real-time voice-based prediction** via microphone + Whisper



---

## 📊 Evaluation

- **Dataset**: Real-Life Trial (RLT) + self-recorded `.wav` clips
- **Cross-validation**: 5-Fold
- **Track 1** – LSTM:
  - F1 Score: **0.798**, Accuracy: 80.4%

---

## 🗂 Folder Structure
TruthReaper/
├── analysis/                       # Stores evaluation plots and prediction reports
├── clips/                          # Place your raw .wav data here (truthful, deception folders)
├── env/ / venv/                    # Virtual environment folders (optional)
├── models/                         # Whisper + saved model weights
├── recordings/                     # Real-time recorded clips (auto-created)
├── *.py                            # All training, inference, and feature scripts
├── *.json                          # Processed datasets and synthetic data
├── README.md                       # Project guide (this file)
├── requirements.txt                # Dependency list

---

## 📦 File Descriptions

### 🔄 Feature Extraction
- `batch_feature_extractor.py` – Main extractor for all audio features (sequential + summary)
- `pause_anlyzer.py` – Identifies hesitation/pause segments in speech
- `disfluency_extractor.py` – Uses Whisper transcription to find fillers, stutters, repetitions
- `emotion_analyzer.py` – Extracts average energy and pitch variance for emotion cues
- `audio_processor.py` / `feature_extractor.py` – Older feature modules (optional)

---

### 🤖 Modeling & Training
- `sequence_lstm_trainer.py` – Base LSTM model trainer
- `k_fold_trainer.py` – Performs 5-fold cross-validation and saves results
- `truthreaper_hybrid_lstm.pt` – Final trained BiLSTM model

---

### 🧪 Synthetic Data
- `gan_trainer.py` – Trains a GAN for synthetic time-series generation
- `gan_generator.py` – Internal generator class
- `generator_truth.pt`, `generator_lie.pt` – Trained GAN models
- `synthetic_full_truth.json`, `synthetic_full_lie.json` – GAN-generated sample datasets
- `merge_datasets.py` – Combines real and synthetic samples into one JSON

---

### 🎤 Inference & Real-Time
- `truth_reaper_transcriber.py` – Record audio + transcribe + predict (full pipeline)
- `truth_recorder.py`, `lie_recorder.py` – Save mic input directly to respective folders
- `test-input-01.txt` – Test transcripts for evaluation
- `video_to_audio_converter.py` – Extracts audio from video files for labeling

---

## 📦 Installation

Create a virtual environment (optional) and install dependencies:

```bash
pip install -r requirements.txt

📂 Dataset Setup

🔺 The dataset is not provided in this repo. You must download it manually.

	1.	Download from:
https://archive.ics.uci.edu/ml/datasets/Real+Life+Trial+Dataset
	2.	Place your .wav files in this structure:
  /clips/
├── truthful/
│   ├── trial_truth_001.wav
│   └── ...
└── deception/
    ├── trial_lie_001.wav
    └── ...

🚀 How to Run the Project

✅ 1. Feature Extraction:
python3 batch_feature_extractor.py --limit 100

✔️ Generates sequence_dataset.json with audio features

✅ 2. (Optional) Generate Synthetic Data:
python3 gan_trainer.py --label truth --epochs 5000
python3 gan_trainer.py --label lie --epochs 5000

then merge them:
python3 merge_datasets.py
✔️ Creates sequence_dataset_combined.json

✅ 3. Train the LSTM Model (Track 1):
python3 k_fold_trainer.py

✔️ Performs 5-fold CV
✔️ Saves truthreaper_hybrid_lstm.pt
✔️ Saves evaluation plot as kfold_metrics.png

✅ 4. Real-Time Prediction (Microphone)
python3 truth_reaper_transcriber.py

	•	Records your voice
	•	Predicts “truth” or “lie”
	•	Logs to /analysis/reports/

❗ Notes
	•	Whisper ASR is downloaded automatically via Huggingface (whisper-base)
	•	Project assumes single-speaker English voice recordings

📚 References
	•	Whisper: https://github.com/openai/whisper
	•	RLT Dataset: https://archive.ics.uci.edu/ml/datasets/Real+Life+Trial+Dataset
	•	Librosa: https://librosa.org

🙏 Acknowledgments

We thank Prof. Amith Kamath Belman for valuable feedback and research guidance, and acknowledge the use of OpenAI Whisper and Huggingface Transformers in this project.
