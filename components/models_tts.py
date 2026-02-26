import os
from typing import Tuple
from components.models_base import TTSModel, register_model

@register_model("tts", "chatterbox")
class ChatterboxTTS(TTSModel):
    """ChatterboxTTS Turbo 350M - Low latency TTS"""

    def load(self):
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        print("🔊 Loading ChatterboxTTS Turbo...")
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    def synthesize(self, text: str) -> Tuple[bytes, float, float]:
        import io
        import time
        import numpy as np
        from scipy.io import wavfile

        t0 = time.time()

        text = text[:300]

        audio_tensor = self.model.generate(text)
        audio_np = audio_tensor.cpu().numpy().squeeze()

        if audio_np.dtype in [np.float32, np.float64]:
            max_val = np.abs(audio_np).max()
            if max_val > 1.0:
                audio_np = audio_np / max_val
            audio_np = (audio_np * 32767).astype(np.int16)

        buffer = io.BytesIO()
        sample_rate = 24000
        wavfile.write(buffer, sample_rate, audio_np)

        audio_duration = len(audio_np) / sample_rate

        return buffer.getvalue(), audio_duration, time.time() - t0

    @property
    def model_name(self) -> str:
        return "ChatterboxTTS Turbo"

class _InworldTTSBase(TTSModel):
    """
    Inworld TTS-1.5 Base — shared implementation for Max and Mini.

    Key latency path:
      - Use streaming endpoint (/voice:stream) → first PCM chunk in <250ms (Max) or <130ms (Mini)
      - Each SSE line is base64-encoded LINEAR16 PCM at 24 kHz
      - Yield chunks immediately; don't accumulate before sending to client
    """

    _model_id: str = "inworld-tts-1.5-max"
    _chunker_min_chars: int = 8
    _chunker_max_chars: int = 80

    def load(self):
        import os
        import requests
        print(f"🔊 Initialising Inworld TTS ({self._model_id})...")
        self.api_key = os.getenv("INWORLD_API_KEY")
        if not self.api_key:
            raise ValueError("INWORLD_API_KEY is required. Add it to your Modal secret.")
        self.voice_id = os.getenv("INWORLD_VOICE_ID", "Ashley")
        self.sample_rate = 24000
        self.stream_endpoint = "https://api.inworld.ai/tts/v1/voice:stream"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        })
        print(f"✅ Inworld {self._model_id} ready  (voice={self.voice_id})")

    def _stream_raw_pcm(self, text: str):
        import base64
        import json

        payload = {
            "text": text[:2000],
            "voiceId": self.voice_id,
            "modelId": self._model_id,
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": self.sample_rate,
            },
        }

        prev_chunk = None
        with self.session.post(
            self.stream_endpoint, json=payload, stream=True, timeout=30
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                audio_b64 = (
                    data.get("result", {}).get("audioContent")
                    or data.get("audioContent")
                )
                if not audio_b64:
                    continue
                pcm = base64.b64decode(audio_b64)
                if prev_chunk is not None:
                    yield prev_chunk, False
                prev_chunk = pcm
        if prev_chunk is not None:
            yield prev_chunk, True

    @staticmethod
    def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    @property
    def supports_streaming(self) -> bool:
        return True

    def synthesize_stream(self, text: str):
        for pcm_chunk, is_last in self._stream_raw_pcm(text):
            wav_chunk = self._pcm_to_wav(pcm_chunk, self.sample_rate)
            yield (wav_chunk, self.sample_rate, is_last)

    def synthesize(self, text: str) -> Tuple[bytes, float, float]:
        import time

        t0 = time.time()
        all_pcm = bytearray()
        for pcm_chunk, _ in self._stream_raw_pcm(text):
            all_pcm.extend(pcm_chunk)
        all_pcm = bytes(all_pcm)
        wav = self._pcm_to_wav(all_pcm, self.sample_rate)
        duration = len(all_pcm) / (self.sample_rate * 2)
        return wav, duration, time.time() - t0

@register_model("tts", "inworld-tts-1.5-max")
class InworldMaxTTS(_InworldTTSBase):
    """
    Inworld TTS-1.5 Max
    - P90 TTFA: <250 ms
    - 30 % more expressive than prior generation
    - 40 % lower word-error-rate
    - Recommended for most voice applications
    """
    _model_id = "inworld-tts-1.5-max"
    _chunker_min_chars = 8
    _chunker_max_chars = 90

    @property
    def model_name(self) -> str:
        return "Inworld TTS-1.5 Max"

@register_model("tts", "inworld-tts-1.5-mini")
class InworldMiniTTS(_InworldTTSBase):
    """
    Inworld TTS-1.5 Mini
    - P90 TTFA: <130 ms  (4× faster than prior gen)
    - Optimised for hyper-latency-sensitive applications
    - Trade a little expressiveness for blazing speed
    """
    _model_id = "inworld-tts-1.5-mini"
    _chunker_min_chars = 5
    _chunker_max_chars = 60

    @property
    def model_name(self) -> str:
        return "Inworld TTS-1.5 Mini"
