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
        import tempfile
        import os
        import time
        from scipy.io import wavfile
        import io

        t0 = time.time()

        try:
            with io.BytesIO(audio_bytes) as f:
                sr, data = wavfile.read(f)
            audio_duration = len(data) / sr
            print(f"   📊 Audio: {sr}Hz, {audio_duration:.2f}s")
        except Exception as e:
            print(f"   ⚠️  Audio validation error: {e}")

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

@register_model("asr", "whisper")
class WhisperASR(ASRModel):
    """OpenAI Whisper - High accuracy ASR"""

    def load(self):
        import whisper
        print("🎤 Loading Whisper Large-v3...")
        self.model = whisper.load_model("large-v3", device="cuda")

    def transcribe(self, audio_bytes: bytes) -> Tuple[str, float]:
        import tempfile
        import os
        import time

        t0 = time.time()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            result = self.model.transcribe(temp_path)
            text = result["text"]
        finally:
            os.unlink(temp_path)

        return text.strip(), time.time() - t0

    @property
    def model_name(self) -> str:
        return "Whisper Large-v3"

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
