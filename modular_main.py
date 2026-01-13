"""
Modular Speech-to-Speech Pipeline - Production Ready with Client Support

Architecture:
- Base classes for ASR, LLM, TTS
- Concrete implementations for different models
- Config-based model selection
- Compatible with real-time VAD client
- Easy model swapping

Add a new model in 3 steps:
1. Create a class that inherits from base (ASRModel/LLMModel/TTSModel)
2. Implement the required methods
3. Register it in MODEL_REGISTRY

Usage:
    # Deploy with default models (nemo, phi3, chatterbox)
    modal deploy modular_main.py

    # Deploy with custom models
    ASR_MODEL=whisper LLM_MODEL=llama modal deploy modular_main.py

    # Run client
    python client.py

    # Test locally
    modal run modular_main.py --audio-path input.wav

    $env:ASR_MODEL="nemo"; $env:LLM_MODEL="gpt4omini"; $env:TTS_MODEL="chatterbox"; modal deploy modular_main.py

"""
import modal
import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

# Audio Compression Utilities (Inlined for Modal compatibility)

def compress_wav_to_mp3(wav_bytes: bytes, bitrate: int = 64) -> bytes:
    """Compress WAV bytes to MP3 for smaller network transfer."""
    import io
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    buffer = io.BytesIO()
    audio.export(buffer, format="mp3", bitrate=f"{bitrate}k")
    return buffer.getvalue()

def decompress_mp3_to_wav(audio_bytes: bytes) -> bytes:
    """Convert any audio format (MP3, M4A, WebM, OGG, etc.) to WAV for processing."""
    import io
    from pydub import AudioSegment
    
    # Try to detect format from magic bytes
    header = audio_bytes[:12]
    
    try:
        if header[:3] == b'ID3' or header[:2] in (b'\xff\xfb', b'\xff\xfa', b'\xff\xf3'):
            # MP3
            audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        elif header[:4] == b'OggS':
            # OGG/WebM
            audio = AudioSegment.from_ogg(io.BytesIO(audio_bytes))
        elif header[:4] == b'fLaC':
            # FLAC
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="flac")
        elif b'ftyp' in header[:12]:
            # M4A/MP4
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="m4a")
        elif header[:4] == b'RIFF':
            # Already WAV
            return audio_bytes
        else:
            # Try auto-detect
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        # Last resort: try raw format detection
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    
    # Convert to WAV (16-bit PCM, mono, 16kHz for ASR)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()

# Configuration & Registry

class ModelConfig:
    """Configuration for model selection"""
    def __init__(self):
        self.asr = os.getenv("ASR_MODEL", "nemo")
        self.llm = os.getenv("LLM_MODEL", "phi3")
        self.tts = os.getenv("TTS_MODEL", "parler")

    def __str__(self):
        return f"ASR={self.asr}, LLM={self.llm}, TTS={self.tts}"

# Model Registry
MODEL_REGISTRY = {
    "asr": {},
    "llm": {},
    "tts": {}
}

# Default System Prompt for all LLM models
CAR_RENTAL_SYSTEM_PROMPT = """You are a friendly and professional customer service representative for a car rental business. You help customers with inquiries about vehicle availability, pricing, reservations, policies, and general questions.

BUSINESS DETAILS:
- Company: premium car rentals
- Vehicle types available: Economy cars, sedans, SUVs, vans
- Daily rates: Economy $45/day, Sedan $65/day, SUV $95/day, Van $120/day
- Weekly discounts: 7-day rentals get ~15% discount
- Mileage: 1,000 miles included per week, $0.25 per additional mile
- Hours: 7am - 8pm daily
- Minimum age: 21 years old (under 25 incurs $15/day young driver fee)
- Insurance options: Basic CDW $15/day, Full coverage $25/day
- Additional driver: $10/day or $50/week
- Extras: GPS $10/day, child seat $8/day

POLICIES:
- Payment: Major credit cards accepted (debit cards require credit check and $200 deposit)
- Cancellation: Free cancellation up to 24 hours before pickup, $50 fee within 24 hours
- Late return: 1-hour grace period, then charged for additional day
- Fuel policy: Return with same fuel level or pay $6/gallon plus $15 refueling fee
- Modification: Free changes if made 24+ hours in advance, $25 fee within 24 hours
- Security deposit: $500 hold if declining insurance coverage
- Cleaning fee: $50 if vehicle returned excessively dirty
- No smoking, pets must be in carriers

YOUR BEHAVIOR:
- Keep responses conversational and natural, like you're speaking out loud
- Use short, clear sentences that work well for text-to-speech
- Be warm, helpful, and patient
- Ask clarifying questions when needed (dates, vehicle type, customer needs)
- Provide relevant information without overwhelming the customer
- Handle one topic at a time in multi-turn conversations
- If customer asks about something you don't have information on, acknowledge it and offer to check or have a manager call back
- Use casual, friendly language but remain professional
- Avoid excessive formatting, bullet points, or lists - speak in natural paragraphs
- When discussing prices, be clear and include all relevant fees
- Always confirm important details (dates, times, vehicle type) before proceeding

CONVERSATION FLOW:
1. Greet and understand what the customer needs
2. Ask follow-up questions to clarify details
3. Provide relevant information clearly
4. Offer next steps or ask if they need anything else
5. Close warmly and professionally

COMMON SCENARIOS YOU HANDLE:
- Checking vehicle availability for specific dates
- Explaining rental requirements and policies
- Providing pricing quotes with all fees
- Making, modifying, or canceling reservations
- Explaining insurance options
- Answering questions about pickup/return process
- Handling special requests (additional drivers, equipment, etc.)

TONE: Friendly, helpful, professional, conversational, patient

Remember: You're having a voice conversation, so speak naturally like you would on the phone. Keep it simple and easy to understand when spoken aloud."""

def register_model(model_type: str, name: str):
    """Decorator to register models"""
    def decorator(cls):
        MODEL_REGISTRY[model_type][name] = cls
        return cls
    return decorator

# Base Model Classes

class ASRModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> Tuple[str, float]:
        """Returns (transcription, processing_time)"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

class LLMModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        """Returns (response, processing_time)"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

class TTSModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def synthesize(self, text: str) -> Tuple[bytes, float, float]:
        """Returns (audio_bytes, audio_duration, processing_time)"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

# ASR IMPLEMENTATIONS
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

        # Validate audio
        try:
            with io.BytesIO(audio_bytes) as f:
                sr, data = wavfile.read(f)
            audio_duration = len(data) / sr
            print(f"   📊 Audio: {sr}Hz, {audio_duration:.2f}s")
        except Exception as e:
            print(f"   ⚠️  Audio validation error: {e}")

        # Transcribe
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


# LLM IMPLEMENTATIONS
@register_model("llm", "phi3")
class Phi3LLM(LLMModel):
    """Microsoft Phi-3-Mini 3.8B - Fast efficient LLM"""

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("🤖 Loading Phi-3-Mini...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        import torch
        import time
        import re

        t0 = time.time()

        if not user_input or len(user_input.strip()) < 2:
            return "I didn't catch that. Could you please repeat?", 0.0

        system = system_prompt or "You are a helpful voice assistant. Answer directly and completely."

        prompt = f"""<|system|>
{system}<|end|>
<|user|>
{user_input}<|end|>
<|assistant|>"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split(user_input)[-1] if user_input in full_response else full_response
        response = re.sub(r'<\|[^|]*\|>', '', response).strip()

        # Truncate for voice
        if len(response) > 300:
            response = response[:300].rsplit(' ', 1)[0] + '...'

        return response, time.time() - t0

    @property
    def model_name(self) -> str:
        return "Phi-3-Mini 3.8B"


@register_model("llm", "llama")
class LlamaLLM(LLMModel):
    """Meta Llama 3.2 3B - High quality responses"""

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("🤖 Loading Llama 3.2 3B...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        self.model.eval()

    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        import torch
        import time

        t0 = time.time()

        if not user_input.strip():
            return "I didn't catch that.", 0.0

        messages = [
            {"role": "system", "content": system_prompt or CAR_RENTAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
            )

        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

        if len(response) > 300:
            response = response[:300].rsplit(' ', 1)[0] + '...'

        return response, time.time() - t0

    @property
    def model_name(self) -> str:
        return "Llama 3.2 3B"


@register_model("llm", "gpt4omini")
class GPT4oMiniLLM(LLMModel):
    def load(self):
        import os
        print("🤖 Initializing OpenAI API (GPT-4o Mini)...")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Run: modal secret create api-keys OPENAI_API_KEY=sk-xxx")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, timeout=10.0)
        print("✅ OpenAI API ready (GPT-4o Mini)")

    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        import time
        t0 = time.time()

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=150,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt or CAR_RENTAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ]
            )
            text = response.choices[0].message.content[:300]
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            text = "I'm having trouble connecting. Please try again."

        return text, time.time() - t0

    @property
    def model_name(self) -> str:
        return "GPT-4o Mini"


@register_model("llm", "llama31-groq")
class Llama31GroqLLM(LLMModel):
    """Meta Llama-3.1-8B-Instruct via Groq API - Ultra-fast inference"""

    def load(self):
        import os
        print("🤖 Initializing Groq API (Llama-3.1-8B-Instruct)...")
        # Check multiple possible key names
        api_key = (
            os.environ.get("GROQ_API_KEY") or
            os.environ.get("GROQ_KEY") or
            os.environ.get("groq_api_key") or
            os.environ.get("api_keys") or
            os.environ.get("groq-secret")
        )
        if not api_key:
            # List available env vars for debugging
            groq_vars = [k for k in os.environ.keys() if 'groq' in k.lower()]
            raise ValueError(f"GROQ_API_KEY not found. Available groq-related vars: {groq_vars}. "
                           "Run: modal secret create groq-secret GROQ_API_KEY=your_key")

        from groq import Groq
        self.client = Groq(api_key=api_key)
        print("✅ Groq API ready (Llama-3.1-8B-Instruct)")

    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        import time
        t0 = time.time()

        if not user_input.strip():
            return "I didn't catch that.", 0.0

        system = system_prompt or "You are a helpful voice assistant. Keep responses concise and natural for speech."

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=150,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_input}
                ]
            )
            text = response.choices[0].message.content
            if len(text) > 300:
                text = text[:300].rsplit(' ', 1)[0] + '...'
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            text = "I'm having trouble connecting. Please try again."

        return text.strip(), time.time() - t0

    @property
    def model_name(self) -> str:
        return "Llama-3.1-8B-Instruct (Groq)"


@register_model("llm", "qwen3")
class Qwen3LLM(LLMModel):
    """Qwen3-1.7B - Fast and efficient small LLM"""

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("🤖 Loading Qwen3-1.7B...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B-Instruct",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        self.model.eval()
        print("✅ Qwen3-1.7B loaded")

    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        import torch
        import time

        t0 = time.time()

        if not user_input.strip():
            return "I didn't catch that.", 0.0

        system = system_prompt or "You are a helpful voice assistant. Keep responses concise and natural for speech."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode for faster responses
        )

        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Truncate for voice output
        if len(response) > 300:
            response = response[:300].rsplit(' ', 1)[0] + '...'

        return response.strip(), time.time() - t0

    @property
    def model_name(self) -> str:
        return "Qwen3-1.7B"


# TTS IMPLEMENTATIONS
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

        text = text[:300]  # Safety limit

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

        # Default voice description - cool voice
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

        text = text[:300]  # Safety limit
        description = voice_description or self.voice_description

        # Tokenize inputs
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to("cuda")
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda")

        # Generate audio
        with torch.no_grad():
            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )

        audio_np = generation.cpu().numpy().squeeze()
        sample_rate = self.model.config.sampling_rate  # 44100 Hz for Parler

        # Convert to int16 for WAV
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

        # Load processor
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

        # Load model with SDPA attention (PyTorch native, no flash-attn needed)
        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",  # SDPA is built into PyTorch, no extra deps
        )
        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=5)  # Fast inference

        # Load voice preset (Carter - natural male voice)
        # Available English voices: en-Carter_man, en-Frank_man, en-Davis_man, en-Mike_man, en-Emma_woman, en-Grace_woman
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
            # List what's available for debugging
            import glob
            available = glob.glob("/root/vibevoice_voices/**/*.pt", recursive=True)
            raise FileNotFoundError(f"Voice preset not found. Checked: {voice_paths}. Available: {available}")

        print(f"   Using voice: {voice_file}")
        self.voice_preset = torch.load(voice_file, map_location="cuda", weights_only=False)
        self.copy = copy  # Store for deep copy in generate

        print("✅ VibeVoice-Realtime-0.5B loaded")

    def synthesize(self, text: str) -> Tuple[bytes, float, float]:
        import io
        import time
        import numpy as np
        from scipy.io import wavfile
        import torch

        t0 = time.time()

        # Clean and limit text
        text = text[:500].replace("'", "'").replace('"', '"').replace('"', '"')

        # Prepare inputs with cached voice prompt
        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=self.voice_preset,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to GPU
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to("cuda")

        # Generate audio
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

        # Extract audio
        audio_tensor = outputs.speech_outputs[0]
        # Convert bfloat16 to float32 (NumPy doesn't support bfloat16)
        audio_np = audio_tensor.float().cpu().numpy().squeeze()
        sample_rate = 24000  # VibeVoice outputs at 24kHz

        # Convert to int16 for WAV
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

        # Load Orpheus model - uses vLLM under the hood
        self.model = OrpheusModel(
            model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
            max_model_len=2048
        )

        # Available voices: tara, leah, jess, leo, dan, mia, zac, zoe
        self.default_voice = "tara"  # Natural female voice

        print("✅ Orpheus TTS 3B loaded")

    def synthesize(self, text: str, voice: str = None) -> Tuple[bytes, float, float]:
        import io
        import time
        import wave

        t0 = time.time()

        # Clean and limit text
        text = text[:500].replace("'", "'").replace('"', '"').replace('"', '"')
        voice = voice or self.default_voice

        # Generate audio using Orpheus streaming API
        audio_chunks = []
        syn_tokens = self.model.generate_speech(
            prompt=text,
            voice=voice,
        )

        # Collect all chunks
        for audio_chunk in syn_tokens:
            audio_chunks.append(audio_chunk)

        # Combine audio data
        audio_data = b''.join(audio_chunks)

        # Create WAV file
        buffer = io.BytesIO()
        sample_rate = 24000
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        audio_duration = len(audio_data) / (sample_rate * 2)  # 2 bytes per sample

        return buffer.getvalue(), audio_duration, time.time() - t0

    @property
    def model_name(self) -> str:
        return "Orpheus TTS 3B"


# Modal App Setup
app = modal.App("speech-to-speech")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git", "build-essential", "wget")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pydub>=0.25.0",
    )
    .pip_install("nemo-toolkit[asr]>=1.0.0")
    # Upgrade transformers for Qwen3 support (requires >=4.51.0) - force rebuild v2
    .pip_install("transformers>=4.51.0", "accelerate>=0.26.0")
    .pip_install("chatterbox-tts>=0.1.0")
    # Faster-Whisper for optimized ASR
    .pip_install("faster-whisper>=1.0.0")
    # Parler-TTS for expressive speech synthesis
    .pip_install("parler-tts>=0.2.0")
    # VibeVoice for ultra-low latency TTS (~300ms first speech)
    # Using SDPA attention (built into PyTorch) instead of flash-attn to avoid CUDA compilation
    .pip_install("diffusers>=0.25.0", "soundfile")
    .run_commands(
        # Clone VibeVoice repo and install
        "git clone --depth 1 https://github.com/microsoft/VibeVoice.git /tmp/vibevoice && "
        "cd /tmp/vibevoice && pip install -e . && "
        # Download experimental voices using their script
        "cd /tmp/vibevoice/demo && bash download_experimental_voices.sh && "
        # Copy voices to persistent location
        "mkdir -p /root/vibevoice_voices && "
        "cp -r /tmp/vibevoice/demo/voices/* /root/vibevoice_voices/"
    )
    # Orpheus   for human-like expressive speech with emotion tags
    .pip_install("orpheus-speech", "vllm==0.7.3")
    .pip_install("openai", "groq")  # For OpenAI and Groq API models
    .pip_install("fastapi", "uvicorn")
)


# Modular Pipeline Service

# Capture environment variables at deploy time to pass to container
_ASR_MODEL = os.getenv("ASR_MODEL", "nemo")
_LLM_MODEL = os.getenv("LLM_MODEL", "phi3")
_TTS_MODEL = os.getenv("TTS_MODEL", "parler")

@app.cls(
    image=image,
    gpu="A10G",
    # timeout=600,
    # min_containers=1,
    # max_containers=1,
    scaledown_window=300,
    secrets=[
        modal.Secret.from_dict({
            "ASR_MODEL": _ASR_MODEL,
            "LLM_MODEL": _LLM_MODEL,
            "TTS_MODEL": _TTS_MODEL,
        }),
        modal.Secret.from_name("hf-secret"),
        modal.Secret.from_name("api-keys"), 
     ],
)
class SpeechToSpeechService:
    """
    Modular Speech-to-Speech Pipeline

    Change models via environment variables:
        ASR_MODEL=whisper LLM_MODEL=llama TTS_MODEL=chatterbox
    """
    
    # Class-level caches for loaded models
    loaded_asr: dict = {}
    loaded_llm: dict = {}
    loaded_tts: dict = {}

    @modal.enter()
    def load_models(self):
        """Load all models on container startup"""
        import torch

        # Get configuration
        self.config = ModelConfig()

        print("=" * 70)
        print(f"🚀 MODULAR PIPELINE - Configuration: {self.config}")
        print("=" * 70)

        # Validate and load models
        asr_class = MODEL_REGISTRY["asr"].get(self.config.asr)
        llm_class = MODEL_REGISTRY["llm"].get(self.config.llm)
        tts_class = MODEL_REGISTRY["tts"].get(self.config.tts)

        if not asr_class:
            raise ValueError(f"ASR '{self.config.asr}' not found. Available: {list(MODEL_REGISTRY['asr'].keys())}")
        if not llm_class:
            raise ValueError(f"LLM '{self.config.llm}' not found. Available: {list(MODEL_REGISTRY['llm'].keys())}")
        if not tts_class:
            raise ValueError(f"TTS '{self.config.tts}' not found. Available: {list(MODEL_REGISTRY['tts'].keys())}")

        # Load models
        self.asr = asr_class()
        self.asr.load()
        self.loaded_asr[self.config.asr] = self.asr
        print(f"✅ ASR: {self.asr.model_name}")

        self.llm = llm_class()
        self.llm.load()
        self.loaded_llm[self.config.llm] = self.llm
        print(f"✅ LLM: {self.llm.model_name}")

        self.tts = tts_class()
        self.tts.load()
        self.loaded_tts[self.config.tts] = self.tts
        print(f"✅ TTS: {self.tts.model_name}")

        # Check VRAM
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n📊 VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")
        print("=" * 70)

    def _split_sentences(self, text: str) -> list:
        """Split text into sentences for streaming TTS."""
        import re
        # Split on sentence boundaries but keep short fragments together
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Merge very short sentences to avoid tiny audio chunks
        merged = []
        buffer = ""
        for s in sentences:
            if len(buffer) + len(s) < 50:  # Merge if combined < 50 chars
                buffer = (buffer + " " + s).strip() if buffer else s
            else:
                if buffer:
                    merged.append(buffer)
                buffer = s
        if buffer:
            merged.append(buffer)
        return merged if merged else [text]

    @modal.method()
    def process_streaming(self, audio_bytes: bytes, system_prompt: Optional[str] = None):
        """
        Streaming speech-to-speech pipeline.
        Yields audio chunks as sentences are synthesized for lower perceived latency.
        """
        import time
        from scipy.io import wavfile
        import io

        t_start = time.time()

        # Handle compressed input
        if not audio_bytes:
            print("❌ Input audio bytes are empty")
            return {"error": "Empty input audio"}

        if not audio_bytes.startswith(b'RIFF'):
            try:
                audio_bytes = decompress_mp3_to_wav(audio_bytes)
            except Exception:
                pass

        # Step 1: ASR
        print(f"🎤 [{self.asr.model_name}] Transcribing...")
        transcription, asr_time = self.asr.transcribe(audio_bytes)
        print(f"   ✓ {asr_time:.2f}s: {transcription}")

        # Step 2: LLM
        print(f"🤖 [{self.llm.model_name}] Generating...")
        response, llm_time = self.llm.generate(transcription, system_prompt)
        print(f"   ✓ {llm_time:.2f}s: {response}")

        # Yield metadata first
        yield {
            "type": "metadata",
            "transcription": transcription,
            "response": response,
            "asr_time": asr_time,
            "llm_time": llm_time,
        }

        # Step 3: Streaming TTS - sentence by sentence
        sentences = self._split_sentences(response)
        print(f"🔊 [{self.tts.model_name}] Streaming {len(sentences)} chunks...")

        total_tts_time = 0
        total_duration = 0

        for i, sentence in enumerate(sentences):
            t0 = time.time()
            audio_chunk, chunk_duration, chunk_time = self.tts.synthesize(sentence)
            audio_chunk = compress_wav_to_mp3(audio_chunk)
            total_tts_time += chunk_time
            total_duration += chunk_duration

            print(f"   ✓ Chunk {i+1}/{len(sentences)}: {chunk_time:.2f}s")

            yield {
                "type": "audio",
                "audio": audio_chunk,
                "chunk_index": i,
                "total_chunks": len(sentences),
                "chunk_duration": chunk_duration,
                "compressed": True,
            }

        # Yield final metrics
        total_time = time.time() - t_start
        print(f"\n{'='*70}")
        print(f"Streaming complete: {total_time:.2f}s total, {len(sentences)} chunks")
        print(f"{'='*70}\n")

        yield {
            "type": "done",
            "metrics": {
                "asr_time": asr_time,
                "llm_time": llm_time,
                "tts_time": total_tts_time,
                "total_time": total_time,
                "output_duration": total_duration,
                "chunks": len(sentences),
            }
        }

    @modal.method()
    def process(self, audio_bytes: bytes, system_prompt: Optional[str] = None,
                asr_model: Optional[str] = None, llm_model: Optional[str] = None, 
                tts_model: Optional[str] = None) -> Dict:
        """
        Complete speech-to-speech pipeline with compression support.
        Compatible with real-time VAD client.
        
        Optionally specify models to use:
            asr_model: "nemo", "whisper", "faster-whisper"
            llm_model: "phi3", "llama", "gpt4omini", "llama31-groq", "qwen3"
            tts_model: "chatterbox", "parler", "vibevoice", "orpheus"
        """
        import time
        from scipy.io import wavfile
        import io

        t_start = time.time()
        
        # Use specified models or defaults
        asr = self.asr
        llm = self.llm
        tts = self.tts
        
        # Load different model if requested
        if asr_model and asr_model != self.config.asr:
            if asr_model in self.loaded_asr:
                asr = self.loaded_asr[asr_model]
            elif asr_model in MODEL_REGISTRY["asr"]:
                asr = MODEL_REGISTRY["asr"][asr_model]()
                asr.load()
                self.loaded_asr[asr_model] = asr
        
        if llm_model and llm_model != self.config.llm:
            if llm_model in self.loaded_llm:
                llm = self.loaded_llm[llm_model]
            elif llm_model in MODEL_REGISTRY["llm"]:
                llm = MODEL_REGISTRY["llm"][llm_model]()
                llm.load()
                self.loaded_llm[llm_model] = llm
        
        if tts_model and tts_model != self.config.tts:
            if tts_model in self.loaded_tts:
                tts = self.loaded_tts[tts_model]
            elif tts_model in MODEL_REGISTRY["tts"]:
                tts = MODEL_REGISTRY["tts"][tts_model]()
                tts.load()
                self.loaded_tts[tts_model] = tts

        # Handle compressed input
        input_compressed = False
        if not audio_bytes:
            print("❌ Input audio bytes are empty")
            return {"error": "Empty input audio"}

        original_size = len(audio_bytes)

        if not audio_bytes.startswith(b'RIFF'):
            try:
                print(f"📦 Decompressing input: {original_size} bytes")
                audio_bytes = decompress_mp3_to_wav(audio_bytes)
                print(f"📦 Converted: {len(audio_bytes)} bytes")
                input_compressed = True
            except Exception as e:
                print(f"⚠️  Decompression failed, assuming WAV: {e}")

        # Get input duration
        try:
            with io.BytesIO(audio_bytes) as f:
                sr, data = wavfile.read(f)
                input_duration = len(data) / sr
        except:
            input_duration = 0.0

        # Step 1: ASR
        print(f"🎤 [{asr.model_name}] Transcribing...")
        transcription, asr_time = asr.transcribe(audio_bytes)
        print(f"   ✓ {asr_time:.2f}s: {transcription}")

        # Step 2: LLM
        print(f"🤖 [{llm.model_name}] Generating...")
        response, llm_time = llm.generate(transcription, system_prompt)
        print(f"   ✓ {llm_time:.2f}s: {response}")

        # Step 3: TTS
        print(f"🔊 [{tts.model_name}] Synthesizing...")
        audio_response, output_duration, tts_time = tts.synthesize(response)
        print(f"   ✓ {tts_time:.2f}s: {output_duration:.1f}s audio")

        # Compress output
        original_audio_size = len(audio_response)
        audio_response = compress_wav_to_mp3(audio_response)
        compressed_size = len(audio_response)
        print(f"📦 Compressed output: {original_audio_size} → {compressed_size} bytes")

        total_time = time.time() - t_start

        # Print metrics
        print(f"\n{'='*70}")
        print(f"Pipeline: {asr.model_name} → {llm.model_name} → {tts.model_name}")
        print(f"Total: {total_time:.2f}s (ASR:{asr_time:.2f}s LLM:{llm_time:.2f}s TTS:{tts_time:.2f}s)")
        print(f"{'='*70}\n")

        return {
            "audio": audio_response,
            "transcription": transcription,
            "response": response,
            "compressed": True,
            "models": {
                "asr": asr.model_name,
                "llm": llm.model_name,
                "tts": tts.model_name,
            },
            "metrics": {
                "asr_time": asr_time,
                "llm_time": llm_time,
                "tts_time": tts_time,
                "total_time": total_time,
                "total_pipeline": total_time,  # For backward compat
                "input_duration": input_duration,
                "output_duration": output_duration,
                "input_chars": len(transcription),
                "output_chars": len(response),
            }
        }

# Backward Compatible Wrapper (for existing client.py
@app.function(image=image, timeout=600)
def process_speech(audio_bytes: bytes) -> dict:
    """Wrapper for backward compatibility with client.py"""
    service = SpeechToSpeechService()
    return service.process.remote(audio_bytes)

 

@app.function(image=image, timeout=600)
def process_speech_streaming(audio_bytes: bytes):
    """Streaming wrapper - yields audio chunks for lower perceived latency"""
    service = SpeechToSpeechService()
    for chunk in service.process_streaming.remote_gen(audio_bytes):
        yield chunk


# Web API endpoint to list available models (lightweight - no GPU needed)
@app.function(image=image, timeout=60, gpu=None)
@modal.fastapi_endpoint(method="GET")
def get_models() -> dict:
    """
    Get list of available models for frontend dropdowns.
    This is a lightweight endpoint that doesn't require GPU.
    """
    config = ModelConfig()
    return {
        "asr": list(MODEL_REGISTRY["asr"].keys()),
        "llm": list(MODEL_REGISTRY["llm"].keys()),
        "tts": list(MODEL_REGISTRY["tts"].keys()),
        "current": {
            "asr": config.asr,
            "llm": config.llm,
            "tts": config.tts,
        }
    }


# Web API endpoint for frontend
@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def process_web(data: dict) -> dict:
    """
    Web endpoint for frontend - accepts JSON with base64 audio.
    
    Request: {
        "audio_base64": "...",
        "asr_model": "faster-whisper",  # optional
        "llm_model": "llama31-groq",     # optional
        "tts_model": "chatterbox"        # optional
    }
    Response: {"audio_base64": "...", "transcription": "...", "response": "...", ...}
    """
    import base64
    
    # Debug: log what we received
    print(f"📥 Received request data keys: {list(data.keys()) if data else 'None'}")
    print(f"📥 Data type: {type(data)}")
    
    # Decode input with validation
    audio_base64 = data.get("audio_base64", "") if data else ""
    
    if not audio_base64:
        print(f"❌ audio_base64 is empty. Full data: {str(data)[:500]}")
        raise ValueError(f"No audio data received. 'audio_base64' is empty or missing. Received keys: {list(data.keys()) if data else 'None'}")
    
    print(f"📥 Received audio_base64: {len(audio_base64)} chars")
    
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 audio: {e}")
    
    if len(audio_bytes) < 100:
        raise ValueError(f"Audio data too small: {len(audio_bytes)} bytes. Expected valid audio file.")
    
    print(f"📥 Decoded audio: {len(audio_bytes)} bytes")
    
    system_prompt = data.get("system_prompt")

    
    asr_model = data.get("asr_model")
    tts_model = data.get("tts_model")
    import os
    cfg = ModelConfig()
    llm_requested = data.get("llm_model")
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_KEY") or os.environ.get("groq_api_key")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if llm_requested == "gpt4omini":
        if openai_key:
            llm_model = "gpt4omini"
        elif groq_key:
            llm_model = "llama31-groq"
        else:
            llm_model = cfg.llm
    elif llm_requested == "llama31-groq":
        if groq_key:
            llm_model = "llama31-groq"
        elif openai_key:
            llm_model = "gpt4omini"
        else:
            llm_model = cfg.llm
    else:
        llm_model = llm_requested
    
    # Process with optional model selection
    service = SpeechToSpeechService()
    try:
        result = service.process.remote(
            audio_bytes, 
            system_prompt,
            asr_model=asr_model,
            llm_model=llm_model,
            tts_model=tts_model
        )
    except Exception as e:
        # Fallback if API key missing or other runtime error
        print(f"⚠️ Error processing request (likely missing API key): {e}. Falling back to default models.")
        result = service.process.remote(
            audio_bytes, 
            system_prompt
        )
    
    # Encode output for JSON
    audio_out = result["audio"]
    # Decompress to WAV for browser playback
    if result.get("compressed"):
        audio_out = decompress_mp3_to_wav(audio_out)
    
    return {
        "audio_base64": base64.b64encode(audio_out).decode("utf-8"),
        "transcription": result["transcription"],
        "response": result["response"],
        "models": result["models"],
        "asr_time": result["metrics"]["asr_time"],
        "llm_time": result["metrics"]["llm_time"],
        "tts_time": result["metrics"]["tts_time"],
        "total_time": result["metrics"]["total_time"],
        "output_duration": result["metrics"]["output_duration"],
    }


# Local Testing
@app.local_entrypoint()
def main(audio_path: str):
    """
    Local testing entrypoint

    Usage:
        modal run modular_main.py --audio-path input.wav
    """
    print(f"📂 Reading {audio_path}...")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    config = ModelConfig()
    print(f"🎯 Using configuration: {config}\n")

    service = SpeechToSpeechService()
    result = service.process.remote(audio_bytes)

    # Save output
    output_path = "output.wav"
    audio_out = decompress_mp3_to_wav(result["audio"]) if result.get("compressed") else result["audio"]
    with open(output_path, "wb") as f:
        f.write(audio_out)

    print(f"\n✅ Output saved to {output_path}")
    print(f"   Models: {result['models']}")
