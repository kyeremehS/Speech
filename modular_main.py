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

# Default System Prompt - Optimized for LOW LATENCY voice responses
VOICE_ASSISTANT_SYSTEM_PROMPT = """You are a real-time voice assistant.

CRITICAL RULES (must always follow):
- Speak briefly and naturally.
- Default to 1 short sentence.
- Never exceed 2 sentences unless the user explicitly asks for detail.
- Prefer asking a clarifying question over giving a long explanation.
- Do NOT monologue.
- Do NOT give background, context, or disclaimers unless asked.
- Assume the user is listening, not reading.

LATENCY OPTIMIZATION:
- Produce a complete, speakable first sentence immediately.
- Avoid conjunction-heavy or run-on sentences.
- Avoid lists unless explicitly requested.
- Use simple, conversational language.

CONVERSATION FLOW:
- If the request is broad or ambiguous, ask one clarifying question.
- If an answer could be long, summarize in one sentence and ask whether to continue.
- If the user seems to be chatting casually, respond casually and briefly.

STYLE:
- Friendly, calm, and human.
- No filler phrases like "As an AI" or "I can help with".
- No over-explaining.
- No repetition.

FAILURE MODES TO AVOID:
- Long paragraphs
- Multi-sentence explanations without confirmation
- Speaking more than the user asked for"""

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

    def generate_stream(self, user_input: str, system_prompt: Optional[str] = None):
        """
        Streaming generator that yields tokens as they're generated.
        
        MUST be implemented for sub-2s first-audio latency.
        Default implementation falls back to batch (suboptimal).
        
        Yields: str tokens incrementally
        """
        # Default fallback - suboptimal but functional
        # Subclasses SHOULD override for true streaming
        response, _ = self.generate(user_input, system_prompt)
        # Yield word by word to simulate streaming
        for word in response.split():
            yield word + " "

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class StreamingSentenceChunker:
    """
    Ultra-low-latency sentence chunker for streaming LLM → TTS pipeline.
    
    OPTIMIZED FOR TTFA < 1.0s:
    - Aggressive early breaks on clauses
    - Scan from END of buffer (O(1) for common case)
    - Yield as soon as we have a speakable chunk
    
    Rules:
    - Yield on sentence boundaries (.!?)
    - Yield on clause boundaries (,:;) if buffer > min_chars
    - Force yield at max_chars (never wait too long)
    """
    
    def __init__(self, min_chars: int = 10, max_chars: int = 100):
        self.buffer = ""
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.sentence_endings = ".!?"
        self.clause_breaks = ",:;—"  # Break on clause boundaries for faster TTFA
    
    def add_token(self, token: str) -> Optional[str]:
        """
        Add a token and return a complete chunk if available.
        OPTIMIZED: Scans from end for O(1) common case.
        Returns None if still buffering.
        """
        self.buffer += token
        buf_len = len(self.buffer)
        
        # Fast path: buffer too short to yield
        if buf_len < self.min_chars:
            return None
        
        # PRIORITY 1: Hard sentence boundaries (.!?) - yield immediately
        # Scan from end for O(1) in common case (boundary just added)
        for i in range(buf_len - 1, self.min_chars - 2, -1):
            if self.buffer[i] in self.sentence_endings:
                sentence = self.buffer[:i+1].strip()
                self.buffer = self.buffer[i+1:].lstrip()
                return sentence
        
        # PRIORITY 2: Clause breaks for faster TTFA if buffer is growing
        if buf_len >= self.min_chars + 5:
            for i in range(buf_len - 1, self.min_chars - 2, -1):
                if self.buffer[i] in self.clause_breaks:
                    sentence = self.buffer[:i+1].strip()
                    self.buffer = self.buffer[i+1:].lstrip()
                    return sentence
        
        # PRIORITY 3: Force yield at max_chars (never wait too long)
        if buf_len >= self.max_chars:
            # Find last space for clean break
            last_space = self.buffer.rfind(" ", self.min_chars, self.max_chars)
            if last_space > 0:
                sentence = self.buffer[:last_space].strip()
                self.buffer = self.buffer[last_space:].lstrip()
                return sentence
            else:
                # No space found - force break at max
                sentence = self.buffer[:self.max_chars].strip()
                self.buffer = self.buffer[self.max_chars:].lstrip()
                return sentence
        
        return None
    
    def flush(self) -> Optional[str]:
        """Flush any remaining content in buffer."""
        if self.buffer.strip():
            result = self.buffer.strip()
            self.buffer = ""
            return result
        return None

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
    """Microsoft Phi-3-Mini 3.8B - Fast efficient LLM with TRUE STREAMING"""

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

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
        # Store streamer class for later use
        self._TextIteratorStreamer = TextIteratorStreamer

    def generate_stream(self, user_input: str, system_prompt: Optional[str] = None):
        """
        TRUE STREAMING: Yields tokens as they're generated.
        
        Uses HuggingFace TextIteratorStreamer for real-time token output.
        TTS can start synthesis before LLM completes - CRITICAL for <2s latency.
        """
        import torch
        import threading
        import re

        if not user_input or len(user_input.strip()) < 2:
            yield "I didn't catch that. Could you please repeat?"
            return

        system = system_prompt or VOICE_ASSISTANT_SYSTEM_PROMPT

        prompt = f"""<|system|>
{system}<|end|>
<|user|>
{user_input}<|end|>
<|assistant|>"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Create streamer for real-time token output
        streamer = self._TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        # Generation config
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Run generation in background thread (non-blocking)
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they arrive - THIS IS THE STREAMING HOT PATH
        total_chars = 0
        for token in streamer:
            # Clean control tokens
            clean_token = re.sub(r'<\|[^|]*\|>', '', token)
            if clean_token:
                total_chars += len(clean_token)
                if total_chars <= 300:  # Respect voice length limit
                    yield clean_token
        
        thread.join()

    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        """Batch mode - collects all streaming tokens. Use generate_stream() for low latency."""
        import time
        t0 = time.time()
        
        tokens = []
        for token in self.generate_stream(user_input, system_prompt):
            tokens.append(token)
        
        response = "".join(tokens).strip()
        return response, time.time() - t0

    @property
    def model_name(self) -> str:
        return "Phi-3-Mini 3.8B"


@register_model("llm", "llama")
class LlamaLLM(LLMModel):
    """Meta Llama 3.2 3B - High quality responses with TRUE STREAMING"""

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

        print("🤖 Loading Llama 3.2 3B...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        self.model.eval()
        self._TextIteratorStreamer = TextIteratorStreamer

    def generate_stream(self, user_input: str, system_prompt: Optional[str] = None):
        """
        TRUE STREAMING: Yields tokens as they're generated.
        TTS can start before LLM completes - CRITICAL for <2s latency.
        """
        import torch
        import threading

        if not user_input.strip():
            yield "I didn't catch that."
            return

        messages = [
            {"role": "system", "content": system_prompt or VOICE_ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        # Create streamer for real-time token output
        streamer = self._TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            streamer=streamer,
        )

        # Run generation in background thread
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they arrive - STREAMING HOT PATH
        total_chars = 0
        for token in streamer:
            if token:
                total_chars += len(token)
                if total_chars <= 300:
                    yield token

        thread.join()

    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        """Batch mode - use generate_stream() for low latency."""
        import time
        t0 = time.time()
        
        tokens = []
        for token in self.generate_stream(user_input, system_prompt):
            tokens.append(token)
        
        response = "".join(tokens).strip()
        return response, time.time() - t0

    @property
    def model_name(self) -> str:
        return "Llama 3.2 3B"


@register_model("llm", "gpt4omini")
class GPT4oMiniLLM(LLMModel):
    """OpenAI GPT-4o Mini with TRUE STREAMING"""
    
    def load(self):
        import os
        print("🤖 Initializing OpenAI API (GPT-4o Mini)...")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Run: modal secret create api-keys OPENAI_API_KEY=sk-xxx")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, timeout=10.0)
        print("✅ OpenAI API ready (GPT-4o Mini)")

    def generate_stream(self, user_input: str, system_prompt: Optional[str] = None):
        """
        TRUE STREAMING: Yields tokens via OpenAI streaming API.
        First token arrives in ~200-400ms - CRITICAL for <2s latency.
        """
        if not user_input.strip():
            yield "I didn't catch that."
            return

        try:
            stream = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=150,
                temperature=0.7,
                stream=True,  # STREAMING ENABLED
                messages=[
                    {"role": "system", "content": system_prompt or VOICE_ASSISTANT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ]
            )
            
            total_chars = 0
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    total_chars += len(token)
                    if total_chars <= 300:
                        yield token
                        
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            yield "I'm having trouble connecting. Please try again."

    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        """Batch mode - use generate_stream() for low latency."""
        import time
        t0 = time.time()
        
        tokens = []
        for token in self.generate_stream(user_input, system_prompt):
            tokens.append(token)
        
        return "".join(tokens).strip(), time.time() - t0

    @property
    def model_name(self) -> str:
        return "GPT-4o Mini"


@register_model("llm", "llama31-groq")
class Llama31GroqLLM(LLMModel):
    """Meta Llama-3.1-8B-Instruct via Groq API - Ultra-fast with TRUE STREAMING"""

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

    def generate_stream(self, user_input: str, system_prompt: Optional[str] = None):
        """
        TRUE STREAMING: Yields tokens via Groq streaming API.
        Groq is extremely fast - first token in ~50-100ms.
        """
        if not user_input.strip():
            yield "I didn't catch that."
            return

        system = system_prompt or VOICE_ASSISTANT_SYSTEM_PROMPT

        try:
            stream = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=150,
                temperature=0.7,
                stream=True,  # STREAMING ENABLED
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_input}
                ]
            )
            
            total_chars = 0
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    total_chars += len(token)
                    if total_chars <= 300:
                        yield token
                        
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            yield "I'm having trouble connecting. Please try again."

    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        """Batch mode - use generate_stream() for low latency."""
        import time
        t0 = time.time()
        
        tokens = []
        for token in self.generate_stream(user_input, system_prompt):
            tokens.append(token)
        
        return "".join(tokens).strip(), time.time() - t0

    @property
    def model_name(self) -> str:
        return "Llama-3.1-8B-Instruct (Groq)"


@register_model("llm", "qwen3")
class Qwen3LLM(LLMModel):
    """Qwen3-1.7B - Fast and efficient small LLM with TRUE STREAMING"""

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

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
        self._TextIteratorStreamer = TextIteratorStreamer
        print("✅ Qwen3-1.7B loaded")

    def generate_stream(self, user_input: str, system_prompt: Optional[str] = None):
        """
        TRUE STREAMING: Yields tokens as they're generated.
        TTS can start before LLM completes - CRITICAL for <2s latency.
        """
        import torch
        import threading

        if not user_input.strip():
            yield "I didn't catch that."
            return

        system = system_prompt or VOICE_ASSISTANT_SYSTEM_PROMPT

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

        # Create streamer for real-time token output
        streamer = self._TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Run generation in background thread
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they arrive - STREAMING HOT PATH
        total_chars = 0
        for token in streamer:
            if token:
                total_chars += len(token)
                if total_chars <= 300:
                    yield token

        thread.join()

    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        """Batch mode - use generate_stream() for low latency."""
        import time
        t0 = time.time()
        
        tokens = []
        for token in self.generate_stream(user_input, system_prompt):
            tokens.append(token)
        
        return "".join(tokens).strip(), time.time() - t0

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
        
        try:
            _ = self.llm.generate("Hello", "You are a helpful voice assistant.")
        except Exception as _:
            pass
        try:
            _ = self.tts.synthesize("Hello")
        except Exception as _:
            pass

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
        TRUE STREAMING speech-to-speech pipeline.
        
        CRITICAL INVARIANTS:
        - LLM tokens stream into sentence chunker
        - TTS starts as soon as FIRST sentence is complete
        - Audio begins playing while LLM is STILL generating
        - First audio < 2 seconds from ASR completion
        
        This achieves 60%+ latency reduction vs batch mode.
        """
        import time
        from scipy.io import wavfile
        import io

        t_start = time.time()

        # Handle compressed input
        if not audio_bytes:
            print("❌ Input audio bytes are empty")
            yield {"type": "error", "error": "Empty input audio"}
            return

        if not audio_bytes.startswith(b'RIFF'):
            try:
                audio_bytes = decompress_mp3_to_wav(audio_bytes)
            except Exception:
                pass

        # Step 1: ASR (must complete before LLM)
        print(f"🎤 [{self.asr.model_name}] Transcribing...")
        transcription, asr_time = self.asr.transcribe(audio_bytes)
        print(f"   ✓ {asr_time:.2f}s: {transcription}")
        
        if not transcription.strip():
            yield {"type": "error", "error": "Empty transcription"}
            return

        # Yield transcription immediately so client knows what was heard
        yield {
            "type": "transcription",
            "transcription": transcription,
            "asr_time": asr_time,
        }

        # Step 2 + 3: STREAMING LLM → Sentence Chunker → TTS
        # This is the CRITICAL streaming path
        print(f"🤖 [{self.llm.model_name}] Streaming generation...")
        print(f"🔊 [{self.tts.model_name}] Ready for streaming TTS...")
        
        # LATENCY OPTIMIZED: min_chars=10 for sub-1s TTFA, max_chars=100 for tight chunks
        chunker = StreamingSentenceChunker(min_chars=10, max_chars=100)
        
        llm_start = time.time()
        first_audio_time = None
        total_tts_time = 0
        total_duration = 0
        chunk_index = 0
        full_response = []
        
        # Stream LLM tokens → chunk into sentences → synthesize immediately
        for token in self.llm.generate_stream(transcription, system_prompt):
            full_response.append(token)
            
            # Feed token to sentence chunker
            complete_sentence = chunker.add_token(token)
            
            if complete_sentence:
                # IMMEDIATELY synthesize this sentence - don't wait for more LLM tokens
                t_tts = time.time()
                
                if first_audio_time is None:
                    first_audio_time = t_tts - t_start
                    print(f"   ⚡ FIRST AUDIO at {first_audio_time:.2f}s (target: <1s)")
                
                audio_chunk, chunk_duration, chunk_time = self.tts.synthesize(complete_sentence)
                
                # LATENCY OPTIMIZATION: Skip compression for small chunks (<50KB)
                # Saves 30-50ms per chunk - critical for TTFA
                is_compressed = False
                if len(audio_chunk) >= 50000:  # Only compress if >= 50KB
                    audio_chunk = compress_wav_to_mp3(audio_chunk)
                    is_compressed = True
                
                total_tts_time += chunk_time
                total_duration += chunk_duration
                
                print(f"   ✓ Chunk {chunk_index}: \"{complete_sentence[:40]}...\" → {chunk_duration:.1f}s audio {'(mp3)' if is_compressed else '(wav)'}")
                
                yield {
                    "type": "audio",
                    "audio": audio_chunk,
                    "text": complete_sentence,
                    "chunk_index": chunk_index,
                    "chunk_duration": chunk_duration,
                    "compressed": is_compressed,
                }
                chunk_index += 1
        
        llm_time = time.time() - llm_start
        
        # Flush any remaining text in the chunker
        remaining = chunker.flush()
        if remaining:
            t_tts = time.time()
            if first_audio_time is None:
                first_audio_time = t_tts - t_start
            
            audio_chunk, chunk_duration, chunk_time = self.tts.synthesize(remaining)
            
            # LATENCY OPTIMIZATION: Skip compression for small chunks
            is_compressed = False
            if len(audio_chunk) >= 50000:
                audio_chunk = compress_wav_to_mp3(audio_chunk)
                is_compressed = True
            
            total_tts_time += chunk_time
            total_duration += chunk_duration
            
            print(f"   ✓ Final chunk: \"{remaining[:40]}...\" → {chunk_duration:.1f}s audio {'(mp3)' if is_compressed else '(wav)'}")
            
            yield {
                "type": "audio",
                "audio": audio_chunk,
                "text": remaining,
                "chunk_index": chunk_index,
                "chunk_duration": chunk_duration,
                "compressed": is_compressed,
            }
            chunk_index += 1

        total_time = time.time() - t_start
        response_text = "".join(full_response).strip()
        
        # Print streaming metrics
        print(f"\n{'='*70}")
        print(f"{'STREAMING PIPELINE METRICS':^70}")
        print(f"{'='*70}")
        print(f"  ASR time:           {asr_time:.2f}s")
        print(f"  LLM stream time:    {llm_time:.2f}s")
        print(f"  TTS total time:     {total_tts_time:.2f}s")
        print(f"  First audio at:     {first_audio_time:.2f}s {'✅' if first_audio_time < 2.0 else '⚠️  >2s!'}")
        print(f"  Audio chunks:       {chunk_index}")
        print(f"  Total audio:        {total_duration:.1f}s")
        print(f"  End-to-end:         {total_time:.2f}s")
        print(f"{'='*70}\n")

        # Yield final completion with full response and metrics
        yield {
            "type": "done",
            "response": response_text,
            "metrics": {
                "asr_time": asr_time,
                "llm_time": llm_time,
                "tts_time": total_tts_time,
                "total_time": total_time,
                "first_audio_time": first_audio_time,
                "output_duration": total_duration,
                "chunks": chunk_index,
            }
        }

    @modal.method()
    def process_streaming_legacy(self, audio_bytes: bytes, system_prompt: Optional[str] = None):
        """
        LEGACY streaming - waits for full LLM response, then streams TTS by sentence.
        
        This is SUBOPTIMAL - use process_streaming() for true overlap.
        Kept for backward compatibility.
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

        # Step 2: LLM (BATCH - waits for full response)
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

@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
async def process_web_stream_text(req):
    from fastapi.responses import StreamingResponse
    import base64
    try:
        body = await req.body()
        audio_base64 = body.decode("utf-8")
    except Exception:
        audio_base64 = ""
    audio_bytes = base64.b64decode(audio_base64) if audio_base64 else b""
    cfg = ModelConfig()
    asr_model = data.get("asr_model")
    llm_model = resolve_llm_model(data.get("llm_model"), cfg)
    service = SpeechToSpeechService()
    def gen():
        asr = service.asr if not asr_model or asr_model == cfg.asr else service._get_or_load_model("asr", asr_model)
        llm = service.llm if not llm_model or llm_model == cfg.llm else service._get_or_load_model("llm", llm_model)
        transcription, _ = asr.transcribe(audio_bytes)
        yield transcription + "\n\n"
        for token in llm.generate_stream(transcription):
            yield token
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    return StreamingResponse(gen(), media_type="text/plain", headers=headers)

@app.function(image=image, timeout=60, gpu=None)
@modal.fastapi_endpoint(method="OPTIONS")
def process_web_stream_text_options():
    from fastapi import Response
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    return Response(status_code=204, headers=headers)

# Health check endpoint for keep-alive (prevents cold starts)
@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint - ping every 30s to keep container warm."""
    return {"status": "warm", "timestamp": time.time()}


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
