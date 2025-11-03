# ViToSA: Vietnamese Audio-based Toxic Spans Detection  

---

## Overview

**ViToSA-Pipeline** provides an end-to-end Python interface for:
- Transcribing Vietnamese audio into text using ViToSA PhoWhisper ASR.
- Detecting and masking toxic words with ViToSA PhoBERT TSD.
- Optionally processing full directories of `.wav` files.

It is built on top of the **ViToSA 1.0 dataset**, the first Vietnamese dataset for audio-based toxic span detection, developed by researchers from the University of Information Technology (VNU-HCM).

---

## About the ViToSA Dataset

**Title:** ViToSA: Audio-Based Toxic Spans Detection on Vietnamese Speech Utterances  
**Version:** 1.0 (August 2025)  
**Authors:** Huy Ba Do, Vy Le-Phuong Huynh, and Luan Thanh Nguyen  
**Institution:** University of Information Technology, VNU-HCM  
**Published:** Interspeech 2025  

If you use this dataset or models derived from it, please cite:

```bibtex
@inproceedings{do25b_interspeech,
  title     = {{ViToSA: Audio-Based Toxic Spans Detection on Vietnamese Speech Utterances}},
  author    = {Huy Ba Do and Vy Le-Phuong Huynh and Luan Thanh Nguyen},
  year      = {2025},
  booktitle = {Interspeech 2025},
  pages     = {4013--4017},
  doi       = {10.21437/Interspeech.2025-1958},
  issn      = {2958-1796}
}
```

### Abstract

Toxic speech in online platforms is a growing concern for user safety. While textual toxicity detection is well-studied, audio-based toxicity detection—especially for low-resource languages like Vietnamese—remains underexplored.

The ViToSA dataset introduces ~11,000 speech samples (≈25 hours) with human-annotated toxic spans, enabling ASR + Toxic Speech Detection (TSD) pipelines for Vietnamese. Experiments show that fine-tuning ASR models on ViToSA improves WER, and that text-based toxic span detection (TSD) models outperform baseline toxicity classifiers.

### Dataset Summary

| Split | # Examples | Notes |
|-------|------------|-------|
| Train | 8,641 | toxic only |
| Validation | 2,161 | toxic only |
| Test | 2,000 | balanced 50/50 toxic / non-toxic |

Each entry includes:
- `file_name`: audio file name
- `audio`: waveform (float32)
- `transcript`: human transcription
- `toxicity`: binary annotation

**Modalities:** Audio + Text  
**Language:** Vietnamese  
**License:** CC-BY-NC-ND 4.0 (non-commercial, no derivatives)

---

## ViToSA Pipeline Features

✅ Automatic GPU / CPU selection  
✅ Robust Hugging Face model handling  
✅ Toxic word masking with "***"  

---

## Installation

```bash
pip install vitosa-pipeline
```

---

## Usage

### 1. Full Pipeline (ASR + TSD)

Transcribe and censor toxic words in an audio file:

```python
from vitosa_pipeline import ViToSA

vitosa = ViToSA()
print(vitosa.pipeline("example.wav"))
```

### 2. Individual Components

#### 2.1. ASR Only - Transcribe Audio

Transcribe an utterance with ViToSA-PhoWhisper:

```python
from vitosa_pipeline import ViToSA

vitosa = ViToSA()
print(vitosa.asr("path_to_audio.wav"))
```

#### 2.2. TSD Only - Detect Toxic Spans

Detect and mask toxic words with ViToSA-PhoBERT:

```python
from vitosa_pipeline import ViToSA

vitosa = ViToSA()
transcript = "your_text_here"

# Return censored text with *** masking toxic words
print(vitosa.tsd(transcript))

# Return binary labels (0 = non-toxic, 1 = toxic word)
print(vitosa.tsd(transcript, return_labels=True))
```

---

## Terms of Use

- Research and educational use only (non-commercial).  
- Redistribution of raw data is prohibited.  
- Cite the ViToSA paper if used in research.  
- The dataset and models may contain explicit language. Use responsibly.  

---

## Contact

For more information: luannt@uit.edu.vn  

---

MIT License © 2025 Luan Thanh Nguyen