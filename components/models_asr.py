from typing import Tuple
from components.models_base import ASRModel, register_model

@register_model("asr", "nemo")
class NeMoASR(ASRModel):
    """NeMo RNNT 0.6B - Fast streaming ASR"""

    def load(self):
        from nemo.collections.asr.models import EncDecRNNTBPEModel
        print("🎤 Loading NeMo RNNT 0.6B...")
        self.model = (
            EncDecRNNTBPEModel
            .from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")
            .cuda()
            .eval()
        )

    def transcribe(self, audio_bytes: bytes) -> Tuple[str, float]:
        import time
        import io
        import numpy as np
        from scipy.io import wavfile

        t0 = time.time()

        # Parse WAV in-memory — single parse, no temp file
        with io.BytesIO(audio_bytes) as f:
            sr, data = wavfile.read(f)
        audio_duration = len(data) / sr
        print(f"   📊 Audio: {sr}Hz, {audio_duration:.2f}s")

        # Convert to float32 tensor for NeMo (in-memory, no disk I/O)
        if data.dtype == np.int16:
            audio_float = data.astype(np.float32) / 32768.0
        elif data.dtype == np.float32:
            audio_float = data
        else:
            audio_float = data.astype(np.float32)

        try:
            # Try in-memory transcription first (NeMo >= 2.0)
            import torch
            audio_tensor = torch.tensor(audio_float).unsqueeze(0).cuda()
            audio_len = torch.tensor([len(audio_float)], dtype=torch.long).cuda()
            result = self.model.transcribe(audio_tensor, audio_len)
            if result and len(result) > 0:
                hypothesis = result[0]
                text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)
            else:
                text = ""
        except (TypeError, AttributeError):
            # Fallback: some NeMo versions only accept file paths
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            try:
                result = self.model.transcribe([temp_path])
                if result and len(result) > 0:
                    hypothesis = result[0]
                    text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)
                else:
                    text = ""
            finally:
                os.unlink(temp_path)

        return text.strip(), time.time() - t0

    @property
    def model_name(self) -> str:
        return "NeMo RNNT 0.6B"

@register_model("asr", "faster-whisper")
class FasterWhisperASR(ASRModel):
    """Faster-Whisper distil-large-v3 - Optimized CTranslate2 ASR"""

    def load(self):
        from faster_whisper import WhisperModel
        print("🎤 Loading Faster-Whisper distil-large-v3...")
        self.model = WhisperModel(
            "distil-large-v3",
            device="cuda",
            compute_type="float16"
        )

    def transcribe(self, audio_bytes: bytes) -> Tuple[str, float]:
        import tempfile
        import os
        import time

        t0 = time.time()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            segments, info = self.model.transcribe(
                temp_path,
                beam_size=5,
                language="en",
                vad_filter=True
            )
            text = " ".join([segment.text for segment in segments])
        finally:
            os.unlink(temp_path)

        return text.strip(), time.time() - t0

    @property
    def model_name(self) -> str:
        return "Faster-Whisper distil-large-v3"
