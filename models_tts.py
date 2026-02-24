import os
from typing import Tuple
from models_base import TTSModel, register_model

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

@register_model("tts", "parler")
class ParlerTTS(TTSModel):
    """Parler-TTS Mini v1 - Expressive text-to-speech with voice descriptions"""

    def load(self):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        print("🔊 Loading Parler-TTS Mini v1...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-mini-v1"
        ).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

        self.voice_description = (
            "A cool, confident speaker with a deep, smooth, and engaging tone. "
            "The recording quality is excellent with minimal background noise."
        )

    def synthesize(self, text: str, voice_description: str = None) -> Tuple[bytes, float, float]:
        import io
        import time
        import numpy as np
        from scipy.io import wavfile
        import torch

        t0 = time.time()

        text = text[:300]
        description = voice_description or self.voice_description

        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to("cuda")
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda")

        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )

        audio_np = generation.cpu().numpy().squeeze()
        sample_rate = self.model.config.sampling_rate

        if audio_np.dtype in [np.float32, np.float64]:
            max_val = np.abs(audio_np).max()
            if max_val > 1.0:
                audio_np = audio_np / max_val
            audio_np = (audio_np * 32767).astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_np)

        audio_duration = len(audio_np) / sample_rate

        return buffer.getvalue(), audio_duration, time.time() - t0

    @property
    def model_name(self) -> str:
        return "Parler-TTS Mini v1"

@register_model("tts", "vibevoice")
class VibeVoiceTTS(TTSModel):
    """Microsoft VibeVoice-Realtime-0.5B - Ultra-low latency real-time TTS (~300ms first speech)"""

    def load(self):
        import torch
        import copy
        from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        from huggingface_hub import hf_hub_download

        print("🔊 Loading VibeVoice-Realtime-0.5B...")

        model_path = "microsoft/VibeVoice-Realtime-0.5B"

        self.processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=5)

        voice_paths = [
            "/root/vibevoice_voices/streaming_model/en-Carter_man.pt",
            "/root/vibevoice_voices/streaming_model/en-Davis_man.pt",
            "/root/vibevoice_voices/streaming_model/en-Mike_man.pt",
            "/root/vibevoice_voices/streaming_model/en-Emma_woman.pt",
        ]
        voice_file = None
        for path in voice_paths:
            if os.path.exists(path):
                voice_file = path
                break

        if voice_file is None:
            import glob
            available = glob.glob("/root/vibevoice_voices/**/*.pt", recursive=True)
            raise FileNotFoundError(f"Voice preset not found. Checked: {voice_paths}. Available: {available}")

        print(f"   Using voice: {voice_file}")
        self.voice_preset = torch.load(voice_file, map_location="cuda", weights_only=False)
        self.copy = copy

        print("✅ VibeVoice-Realtime-0.5B loaded")

    def synthesize(self, text: str) -> Tuple[bytes, float, float]:
        import io
        import time
        import numpy as np
        from scipy.io import wavfile
        import torch

        t0 = time.time()

        text = text[:500].replace("'", "'").replace('"', '"').replace('"', '"')

        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=self.voice_preset,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.5,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
                all_prefilled_outputs=self.copy.deepcopy(self.voice_preset),
            )

        audio_tensor = outputs.speech_outputs[0]
        audio_np = audio_tensor.float().cpu().numpy().squeeze()
        sample_rate = 24000

        if audio_np.dtype in [np.float32, np.float64]:
            max_val = np.abs(audio_np).max()
            if max_val > 1.0:
                audio_np = audio_np / max_val
            audio_np = (audio_np * 32767).astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_np)

        audio_duration = len(audio_np) / sample_rate

        return buffer.getvalue(), audio_duration, time.time() - t0

    @property
    def model_name(self) -> str:
        return "VibeVoice-Realtime-0.5B"

@register_model("tts", "orpheus")
class OrpheusTTS(TTSModel):
    """Canopy Labs Orpheus-3B - Human-like expressive TTS with emotion tags (~200ms streaming latency)"""

    def load(self):
        print("🔊 Loading Orpheus TTS 3B...")
        from orpheus_tts import OrpheusModel

        self.model = OrpheusModel(
            model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
            max_model_len=2048
        )

        self.default_voice = "tara"

        print("✅ Orpheus TTS 3B loaded")

    def synthesize(self, text: str, voice: str = None) -> Tuple[bytes, float, float]:
        import io
        import time
        import wave

        t0 = time.time()

        text = text[:500].replace("'", "'").replace('"', '"').replace('"', '"')
        voice = voice or self.default_voice

        audio_chunks = []
        syn_tokens = self.model.generate_speech(
            prompt=text,
            voice=voice,
        )

        for audio_chunk in syn_tokens:
            audio_chunks.append(audio_chunk)

        audio_data = b''.join(audio_chunks)

        buffer = io.BytesIO()
        sample_rate = 24000
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        audio_duration = len(audio_data) / (sample_rate * 2)

        return buffer.getvalue(), audio_duration, time.time() - t0

    @property
    def model_name(self) -> str:
        return "Orpheus TTS 3B"

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
        all_pcm = b""
        for pcm_chunk, _ in self._stream_raw_pcm(text):
            all_pcm += pcm_chunk
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
