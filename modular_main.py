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

def decompress_mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Decompress MP3 bytes back to WAV for processing."""
    import io
    from pydub import AudioSegment
    
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()

# Configuration & Registry

class ModelConfig:
    """Configuration for model selection"""
    def __init__(self):
        self.asr = os.getenv("ASR_MODEL", "nemo")
        self.llm = os.getenv("LLM_MODEL", "phi3")
        self.tts = os.getenv("TTS_MODEL", "chatterbox")
    
    def __str__(self):
        return f"ASR={self.asr}, LLM={self.llm}, TTS={self.tts}"

# Model Registry
MODEL_REGISTRY = {
    "asr": {},
    "llm": {},
    "tts": {}
}

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
            {"role": "system", "content": system_prompt or "You are a helpful voice assistant."},
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
                    {"role": "system", "content": system_prompt or "You are a helpful voice assistant."},
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

# Modal App Setup
app = modal.App("speech-to-speech")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git", "build-essential")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pydub>=0.25.0",
    )
    .pip_install("nemo-toolkit[asr]>=1.0.0")
    .pip_install("transformers>=4.36.0", "accelerate>=0.20.0")
    .pip_install("chatterbox-tts>=0.1.0")
    # Add more model dependencies as needed
    # .pip_install("openai-whisper")  # Uncomment for Whisper
    .pip_install("openai")  # Uncomment for OpenAI GPT models
)


# Modular Pipeline Service

# Capture environment variables at deploy time to pass to container
_ASR_MODEL = os.getenv("ASR_MODEL", "nemo")
_LLM_MODEL = os.getenv("LLM_MODEL", "phi3")
_TTS_MODEL = os.getenv("TTS_MODEL", "chatterbox")

@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=300,
    secrets=[modal.Secret.from_dict({
        "ASR_MODEL": _ASR_MODEL,
        "LLM_MODEL": _LLM_MODEL,
        "TTS_MODEL": _TTS_MODEL,
    }), modal.Secret.from_name("api-keys"), modal.Secret.from_name("hf-secret")],
)
class SpeechToSpeechService:
    """
    Modular Speech-to-Speech Pipeline
    
    Change models via environment variables:
        ASR_MODEL=whisper LLM_MODEL=llama TTS_MODEL=chatterbox
    """
    
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
        print(f"✅ ASR: {self.asr.model_name}")
        
        self.llm = llm_class()
        self.llm.load()
        print(f"✅ LLM: {self.llm.model_name}")
        
        self.tts = tts_class()
        self.tts.load()
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
    def process(self, audio_bytes: bytes, system_prompt: Optional[str] = None) -> Dict:
        """
        Complete speech-to-speech pipeline with compression support.
        Compatible with real-time VAD client.
        """
        import time
        from scipy.io import wavfile
        import io
        
        t_start = time.time()
        
        # Handle compressed input
        input_compressed = False
        original_size = len(audio_bytes)
        
        if not audio_bytes.startswith(b'RIFF'):
            try:
                print(f"📦 Decompressing input: {original_size} bytes")
                audio_bytes = decompress_mp3_to_wav(audio_bytes)
                print(f"📦 Decompressed: {len(audio_bytes)} bytes")
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
        print(f"🎤 [{self.asr.model_name}] Transcribing...")
        transcription, asr_time = self.asr.transcribe(audio_bytes)
        print(f"   ✓ {asr_time:.2f}s: {transcription}")
        
        # Step 2: LLM
        print(f"🤖 [{self.llm.model_name}] Generating...")
        response, llm_time = self.llm.generate(transcription, system_prompt)
        print(f"   ✓ {llm_time:.2f}s: {response}")
        
        # Step 3: TTS
        print(f"🔊 [{self.tts.model_name}] Synthesizing...")
        audio_response, output_duration, tts_time = self.tts.synthesize(response)
        print(f"   ✓ {tts_time:.2f}s: {output_duration:.1f}s audio")
        
        # Compress output
        original_audio_size = len(audio_response)
        audio_response = compress_wav_to_mp3(audio_response)
        compressed_size = len(audio_response)
        print(f"📦 Compressed output: {original_audio_size} → {compressed_size} bytes")
        
        total_time = time.time() - t_start
        
        # Print metrics
        print(f"\n{'='*70}")
        print(f"Pipeline: {self.asr.model_name} → {self.llm.model_name} → {self.tts.model_name}")
        print(f"Total: {total_time:.2f}s (ASR:{asr_time:.2f}s LLM:{llm_time:.2f}s TTS:{tts_time:.2f}s)")
        print(f"{'='*70}\n")
        
        return {
            "audio": audio_response,
            "transcription": transcription,
            "response": response,
            "compressed": True,
            "models": {
                "asr": self.asr.model_name,
                "llm": self.llm.model_name,
                "tts": self.tts.model_name,
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