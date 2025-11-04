import os
import glob
import torch
import torchaudio
import warnings
import huggingface_hub
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# ===== Setup and Safety =====
warnings.filterwarnings("ignore", category=UserWarning)
huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 120
huggingface_hub.constants.HF_HUB_DOWNLOAD_RETRIES = 10

# ===== Constants =====
ASR_REPO = "UIT-ViToSA/vitosa-phowhisper-asr"
TSD_REPO = "UIT-ViToSA/vitosa-phobert-tsd"


class ViToSA:
    """
    ViToSA: Vietnamese Toxic Speech Analysis Pipeline
    Combines:
      - ASR (speech → text)
      - TSD (text → toxicity detection)
    """

    def __init__(self, device: str = None, cache_dir: str = None, verbose: bool = True):
        """
        Initialize ViToSA models.

        Args:
            device: Optional ('cuda', 'cpu', or 'mps'). Auto-detects if None.
            cache_dir: Optional local directory to store downloaded models.
            verbose: Whether to print progress info.
        """
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        if verbose:
            print(f"[ViToSA] Using device: {self.device}")

        # === Load ASR ===
        if verbose:
            print("[ViToSA] Loading ASR model...")
        self.proc = WhisperProcessor.from_pretrained(ASR_REPO, cache_dir=cache_dir)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(
            ASR_REPO, cache_dir=cache_dir
        ).to(self.device)

        # === Load TSD ===
        if verbose:
            print("[ViToSA] Loading TSD model...")
        self.tokenizer = AutoTokenizer.from_pretrained(TSD_REPO, cache_dir=cache_dir)
        self.tsd_model = AutoModelForTokenClassification.from_pretrained(
            TSD_REPO, cache_dir=cache_dir, num_labels=2
        ).to(self.device)
        self.tsd_model.eval()

    # === ASR ===
    def asr(self, audio_filepath: str) -> str:
        """Transcribe Vietnamese audio using ViToSA ASR."""
        speech_array, sampling_rate = torchaudio.load(audio_filepath)

        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(speech_array)
            sampling_rate = 16000

        input_features = self.proc(
            speech_array[0], sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.asr_model.generate(input_features)
            transcription = self.proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return transcription.strip()

    # === TSD ===
    def tsd(self, text: str, return_labels: bool = False, mask_token: str = "***"):
        """
        Token-level Toxic Speech Detection.

        Args:
            text: Input sentence
            return_labels: If True, return list of labels (0 or 1)
            mask_token: Replacement for toxic words (default: "***")
        Returns:
            - str (masked text)
            - list[int] if return_labels=True
        """

        text = text.split()

        enc = self.tokenizer(list(text), is_split_into_words=True,
        padding='max_length', truncation=True,
        max_length=len(list(text)), return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.tsd_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
        labels = logits.argmax(-1)[0].cpu().tolist()

        if return_labels:
            return labels

        result = ' '.join(['***' if l == 1 else s for s, l in zip(text, labels)])

        # Replace toxic words
        return result

    # === PIPELINE ===
    def pipeline(self, audio_filepath: str, mask_token: str = "***") -> str:
        """
        Full ASR → TSD pipeline.
        Args:
            audio_filepath: Path to .wav file
            mask_token: Replacement for toxic words (default: "***")
        Returns:
            str - Transcription with toxic words masked.
        """
        transcription = self.asr(audio_filepath)
        masked_text = self.tsd(transcription, return_labels=False, mask_token=mask_token)
        return masked_text

    # === Batch Processing ===
    def process_folder(self, folder_path: str, output_dir: str = None, mask_token: str = "***"):
        """
        Run the full ASR→TSD pipeline on all .wav files in a folder.

        Args:
            folder_path: Directory containing .wav files
            output_dir: Directory to save results (default: same as input)
            mask_token: Replacement for toxic words (default: "***")
        """
        output_dir = output_dir or folder_path
        os.makedirs(output_dir, exist_ok=True)

        audio_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
        if not audio_files:
            print("[ViToSA] No .wav files found in:", folder_path)
            return

        for path in audio_files:
            print(f"\n🎧 Processing: {os.path.basename(path)}")
            try:
                result = self.pipeline(path, mask_token=mask_token)
                out_file = os.path.join(
                    output_dir, os.path.basename(path).replace(".wav", "_result.txt")
                )
                with open(out_file, "w") as f:
                    f.write(result)
                print(f"✅ Saved result → {out_file}")
            except Exception as e:
                print(f"❌ Failed on {path}: {e}")
