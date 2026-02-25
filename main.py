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

from __future__ import annotations

import os
from typing import Dict, Optional

import modal

try:
    from fastapi import Request
except Exception:  # pragma: no cover
    Request = object

from components import models_asr, models_llm, models_tts
from components.models_base import MODEL_REGISTRY, ModelConfig, StreamingSentenceChunker
from helper.utility import compress_wav_to_mp3, decompress_mp3_to_wav, ensure_wav_bytes

ENABLE_LEGACY_WEB = os.getenv("ENABLE_LEGACY_WEB", "0") == "1"
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
    .pip_install("transformers>=4.51.0", "accelerate>=0.26.0")
    .pip_install("chatterbox-tts>=0.1.0")
    .pip_install("faster-whisper>=1.0.0")
    .pip_install("parler-tts>=0.2.0")
    .pip_install("requests>=2.31.0")
    .pip_install("diffusers>=0.25.0", "soundfile")
)


def add_local_dir(image_obj, local_dir: str, remote_dir: str):
    if hasattr(image_obj, "copy_local_dir"):
        return image_obj.copy_local_dir(local_dir, remote_dir)
    if hasattr(image_obj, "add_local_dir"):
        return image_obj.add_local_dir(local_dir, remote_dir)
    raise AttributeError("No supported local dir method on modal.Image")


image = (
    image.run_commands(
        "git clone --depth 1 https://github.com/microsoft/VibeVoice.git /tmp/vibevoice && "
        "cd /tmp/vibevoice && pip install -e . && "
        "cd /tmp/vibevoice/demo && bash download_experimental_voices.sh && "
        "mkdir -p /root/vibevoice_voices && "
        "cp -r /tmp/vibevoice/demo/voices/* /root/vibevoice_voices/"
    )
    .pip_install("orpheus-speech", "vllm==0.7.3")
    .pip_install("openai", "groq")
    .pip_install("fastapi", "uvicorn")
)
image = add_local_dir(image, "helper", "/root/helper")
image = add_local_dir(image, "components", "/root/components")


# Modular Pipeline Service

# Capture environment variables at deploy time to pass to container
_ASR_MODEL = os.getenv("ASR_MODEL", "nemo")
_LLM_MODEL = os.getenv("LLM_MODEL", "phi3")
_TTS_MODEL = os.getenv("TTS_MODEL", "parler")


def split_sentences(text: str) -> list:
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    merged = []
    buffer = ""
    for s in sentences:
        if len(buffer) + len(s) < 50:
            buffer = (buffer + " " + s).strip() if buffer else s
        else:
            if buffer:
                merged.append(buffer)
            buffer = s
    if buffer:
        merged.append(buffer)
    return merged if merged else [text]


def get_model_class(model_type: str, model_name: str):
    return MODEL_REGISTRY[model_type].get(model_name)


def require_model_class(model_type: str, model_name: str):
    model_class = get_model_class(model_type, model_name)
    if not model_class:
        raise ValueError(
            f"{model_type.upper()} '{model_name}' not found. Available: {list(MODEL_REGISTRY[model_type].keys())}"
        )
    return model_class


def load_model(model_class):
    model = model_class()
    model.load()
    return model


def get_or_load_model(model_type: str, model_name: str, cache: dict):
    if model_name in cache:
        return cache[model_name]
    model_class = get_model_class(model_type, model_name)
    if not model_class:
        return None
    model = load_model(model_class)
    cache[model_name] = model
    return model


def normalize_audio_bytes(audio_bytes: bytes, log_decompress: bool = False):
    if not audio_bytes:
        return None, False, 0
    original_size = len(audio_bytes)
    if audio_bytes.startswith(b"RIFF"):
        return audio_bytes, False, original_size
    try:
        if log_decompress:
            print(f"📦 Decompressing input: {original_size} bytes")
        audio_bytes = decompress_mp3_to_wav(audio_bytes)
        if log_decompress:
            print(f"📦 Converted: {len(audio_bytes)} bytes")
        return audio_bytes, True, original_size
    except Exception as e:
        if log_decompress:
            print(f"⚠️  Decompression failed, assuming WAV: {e}")
        return audio_bytes, False, original_size


def normalize_binary_audio(
    audio_bytes: bytes, sample_rate: int, channels: int, sample_width: int
):
    if not audio_bytes:
        return None
    if audio_bytes.startswith(b"RIFF"):
        return audio_bytes
    return ensure_wav_bytes(
        audio_bytes,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


def get_wav_duration(audio_bytes: bytes) -> float:
    import io

    from scipy.io import wavfile

    try:
        with io.BytesIO(audio_bytes) as f:
            sr, data = wavfile.read(f)
            return len(data) / sr
    except Exception:
        return 0.0


def compress_if_large(audio_bytes: bytes, threshold: int = 50_000):
    if len(audio_bytes) < threshold:
        return audio_bytes, False
    return compress_wav_to_mp3(audio_bytes), True


def stream_tts_sentence(tts, sentence: str, using_tts_stream: bool, state: dict):
    import time

    t_tts = time.time()
    if state["first_audio_time"] is None:
        state["first_audio_time"] = t_tts - state["t_start"]
        print(f"   ⚡ FIRST AUDIO at {state['first_audio_time']:.2f}s")

    if using_tts_stream:
        sub_index = 0
        chunk_start = time.time()
        for wav_chunk, sample_rate, is_last in tts.synthesize_stream(sentence):
            chunk_duration = (len(wav_chunk) - 44) / (sample_rate * 2)
            state["total_duration"] += chunk_duration
            wav_chunk, compressed = compress_if_large(wav_chunk)
            yield {
                "type": "audio",
                "audio": wav_chunk,
                "text": sentence if sub_index == 0 else "",
                "chunk_index": state["chunk_index"],
                "sub_index": sub_index,
                "chunk_duration": chunk_duration,
                "compressed": compressed,
                "is_last_sub": is_last,
            }
            sub_index += 1
        state["total_tts_time"] += time.time() - chunk_start
    else:
        audio_chunk, chunk_duration, chunk_time = tts.synthesize(sentence)
        state["total_tts_time"] += chunk_time
        state["total_duration"] += chunk_duration
        audio_chunk, compressed = compress_if_large(audio_chunk)
        yield {
            "type": "audio",
            "audio": audio_chunk,
            "text": sentence,
            "chunk_index": state["chunk_index"],
            "chunk_duration": chunk_duration,
            "compressed": compressed,
        }

    print(f'   ✓ Chunk {state["chunk_index"]}: "{sentence[:40]}..."')
    state["chunk_index"] += 1


def stream_tts_sentence_raw(tts, sentence: str, using_tts_stream: bool, state: dict):
    import time

    t_tts = time.time()
    if state["first_audio_time"] is None:
        state["first_audio_time"] = t_tts - state["t_start"]
        print(f"   ⚡ FIRST AUDIO at {state['first_audio_time']:.2f}s")
    if using_tts_stream:
        sub_index = 0
        chunk_start = time.time()
        for wav_chunk, sample_rate, is_last in tts.synthesize_stream(sentence):
            chunk_duration = (len(wav_chunk) - 44) / (sample_rate * 2)
            state["total_duration"] += chunk_duration
            yield {
                "type": "audio",
                "audio": wav_chunk,
                "text": sentence if sub_index == 0 else "",
                "chunk_index": state["chunk_index"],
                "sub_index": sub_index,
                "chunk_duration": chunk_duration,
                "compressed": False,
                "is_last_sub": is_last,
            }
            sub_index += 1
        state["total_tts_time"] += time.time() - chunk_start
    else:
        audio_chunk, chunk_duration, chunk_time = tts.synthesize(sentence)
        state["total_tts_time"] += chunk_time
        state["total_duration"] += chunk_duration
        yield {
            "type": "audio",
            "audio": audio_chunk,
            "text": sentence,
            "chunk_index": state["chunk_index"],
            "chunk_duration": chunk_duration,
            "compressed": False,
        }
    print(f'   ✓ Chunk {state["chunk_index"]}: "{sentence[:40]}..."')
    state["chunk_index"] += 1


@app.cls(
    image=image,
    gpu="A10G",
    # timeout=600,
    # min_containers=1,
    # max_containers=1,
    scaledown_window=300,
    secrets=[
        modal.Secret.from_dict(
            {
                "ASR_MODEL": _ASR_MODEL,
                "LLM_MODEL": _LLM_MODEL,
                "TTS_MODEL": _TTS_MODEL,
            }
        ),
        modal.Secret.from_name("hf-secret"),
        modal.Secret.from_name("api-keys"),
        modal.Secret.from_name("groq-secret"),
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
        asr_class = require_model_class("asr", self.config.asr)
        llm_class = require_model_class("llm", self.config.llm)
        tts_class = require_model_class("tts", self.config.tts)

        self.asr = load_model(asr_class)
        self.loaded_asr[self.config.asr] = self.asr
        print(f"✅ ASR: {self.asr.model_name}")

        self.llm = load_model(llm_class)
        self.loaded_llm[self.config.llm] = self.llm
        print(f"✅ LLM: {self.llm.model_name}")

        self.tts = load_model(tts_class)
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

    @modal.method()
    def process_streaming(
        self, audio_bytes: bytes, system_prompt: Optional[str] = None
    ):
        """
        TRUE STREAMING speech-to-speech pipeline.

        When TTS supports streaming (Inworld), audio chunks begin
        flowing to the client BEFORE a full sentence has even been
        synthesised — shaving another 100-200 ms off perceived latency.

        Yield contract (unchanged):
          {"type": "transcription", ...}
          {"type": "audio", "audio": <bytes>, "compressed": False, ...}  (N times)
          {"type": "done", "metrics": {...}}
        """
        import time

        t_start = time.time()

        audio_bytes, _, _ = normalize_audio_bytes(audio_bytes)
        if not audio_bytes:
            yield {"type": "error", "error": "Empty input audio"}
            return

        print(f"🎤 [{self.asr.model_name}] Transcribing...")
        transcription, asr_time = self.asr.transcribe(audio_bytes)
        print(f"   ✓ {asr_time:.2f}s: {transcription}")

        if not transcription.strip():
            yield {"type": "error", "error": "Empty transcription"}
            return

        yield {
            "type": "transcription",
            "transcription": transcription,
            "asr_time": asr_time,
        }

        print(f"🤖 [{self.llm.model_name}] Streaming generation...")

        min_chars = getattr(self.tts, "_chunker_min_chars", 10)
        max_chars = getattr(self.tts, "_chunker_max_chars", 100)
        chunker = StreamingSentenceChunker(min_chars=min_chars, max_chars=max_chars)

        using_tts_stream = self.tts.supports_streaming
        print(
            f"🔊 [{self.tts.model_name}] {'True-stream' if using_tts_stream else 'Batch'} TTS"
        )

        llm_start = time.time()
        state = {
            "t_start": t_start,
            "first_audio_time": None,
            "total_tts_time": 0,
            "total_duration": 0,
            "chunk_index": 0,
        }
        full_response = []

        for token in self.llm.generate_stream(transcription, system_prompt):
            full_response.append(token)
            complete_sentence = chunker.add_token(token)
            if complete_sentence:
                yield from stream_tts_sentence(
                    self.tts, complete_sentence, using_tts_stream, state
                )

        llm_time = time.time() - llm_start

        remaining = chunker.flush()
        if remaining:
            yield from stream_tts_sentence(self.tts, remaining, using_tts_stream, state)

        total_time = time.time() - t_start
        response_text = "".join(full_response).strip()

        print(f"\n{'=' * 70}")
        print(f"{'STREAMING PIPELINE METRICS':^70}")
        print(f"{'=' * 70}")
        print(f"  ASR time:        {asr_time:.2f}s")
        print(f"  LLM stream time: {llm_time:.2f}s")
        print(f"  TTS total time:  {state['total_tts_time']:.2f}s")
        print(
            f"  First audio at:  {state['first_audio_time']:.2f}s "
            f"{'✅' if state['first_audio_time'] and state['first_audio_time'] < 1.5 else '⚠️  >1.5s!'}"
        )
        print(f"  Chunks sent:     {state['chunk_index']}")
        print(f"  Total audio:     {state['total_duration']:.1f}s")
        print(f"  End-to-end:      {total_time:.2f}s")
        print(f"{'=' * 70}\n")

        yield {
            "type": "done",
            "response": response_text,
            "metrics": {
                "asr_time": asr_time,
                "llm_time": llm_time,
                "tts_time": state["total_tts_time"],
                "total_time": total_time,
                "first_audio_time": state["first_audio_time"],
                "output_duration": state["total_duration"],
                "chunks": state["chunk_index"],
            },
        }

    @modal.method()
    def process_streaming_raw(
        self,
        audio_bytes: bytes,
        system_prompt: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ):
        import time

        t_start = time.time()
        audio_bytes = normalize_binary_audio(
            audio_bytes, sample_rate, channels, sample_width
        )
        if not audio_bytes:
            yield {"type": "error", "error": "Empty input audio"}
            return
        print(f"🎤 [{self.asr.model_name}] Transcribing...")
        transcription, asr_time = self.asr.transcribe(audio_bytes)
        print(f"   ✓ {asr_time:.2f}s: {transcription}")
        if not transcription.strip():
            yield {"type": "error", "error": "Empty transcription"}
            return
        yield {
            "type": "transcription",
            "transcription": transcription,
            "asr_time": asr_time,
        }
        print(f"🤖 [{self.llm.model_name}] Streaming generation...")
        min_chars = getattr(self.tts, "_chunker_min_chars", 10)
        max_chars = getattr(self.tts, "_chunker_max_chars", 100)
        chunker = StreamingSentenceChunker(min_chars=min_chars, max_chars=max_chars)
        using_tts_stream = self.tts.supports_streaming
        print(
            f"🔊 [{self.tts.model_name}] {'True-stream' if using_tts_stream else 'Batch'} TTS"
        )
        llm_start = time.time()
        state = {
            "t_start": t_start,
            "first_audio_time": None,
            "total_tts_time": 0,
            "total_duration": 0,
            "chunk_index": 0,
        }
        full_response = []
        for token in self.llm.generate_stream(transcription, system_prompt):
            full_response.append(token)
            complete_sentence = chunker.add_token(token)
            if complete_sentence:
                yield from stream_tts_sentence_raw(
                    self.tts, complete_sentence, using_tts_stream, state
                )
        llm_time = time.time() - llm_start
        remaining = chunker.flush()
        if remaining:
            yield from stream_tts_sentence_raw(
                self.tts, remaining, using_tts_stream, state
            )
        total_time = time.time() - t_start
        response_text = "".join(full_response).strip()
        yield {
            "type": "done",
            "response": response_text,
            "metrics": {
                "asr_time": asr_time,
                "llm_time": llm_time,
                "tts_time": state["total_tts_time"],
                "total_time": total_time,
                "first_audio_time": state["first_audio_time"],
                "output_duration": state["total_duration"],
                "chunks": state["chunk_index"],
            },
        }

    @modal.method()
    def process_streaming_legacy(
        self, audio_bytes: bytes, system_prompt: Optional[str] = None
    ):
        """
        LEGACY streaming - waits for full LLM response, then streams TTS by sentence.

        This is SUBOPTIMAL - use process_streaming() for true overlap.
        Kept for backward compatibility.
        """
        import time

        t_start = time.time()

        audio_bytes, _, _ = normalize_audio_bytes(audio_bytes)
        if not audio_bytes:
            print("❌ Input audio bytes are empty")
            return {"error": "Empty input audio"}

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
        sentences = split_sentences(response)
        print(f"🔊 [{self.tts.model_name}] Streaming {len(sentences)} chunks...")

        total_tts_time = 0
        total_duration = 0

        for i, sentence in enumerate(sentences):
            t0 = time.time()
            audio_chunk, chunk_duration, chunk_time = self.tts.synthesize(sentence)
            audio_chunk = compress_wav_to_mp3(audio_chunk)
            total_tts_time += chunk_time
            total_duration += chunk_duration

            print(f"   ✓ Chunk {i + 1}/{len(sentences)}: {chunk_time:.2f}s")

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
        print(f"\n{'=' * 70}")
        print(f"Streaming complete: {total_time:.2f}s total, {len(sentences)} chunks")
        print(f"{'=' * 70}\n")

        yield {
            "type": "done",
            "metrics": {
                "asr_time": asr_time,
                "llm_time": llm_time,
                "tts_time": total_tts_time,
                "total_time": total_time,
                "output_duration": total_duration,
                "chunks": len(sentences),
            },
        }

    @modal.method()
    def process(
        self,
        audio_bytes: bytes,
        system_prompt: Optional[str] = None,
        asr_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        tts_model: Optional[str] = None,
    ) -> Dict:
        """
        Complete speech-to-speech pipeline with compression support.
        Compatible with real-time VAD client.

        Optionally specify models to use:
            asr_model: "nemo", "whisper", "faster-whisper"
            llm_model: "phi3", "llama", "gpt4omini", "llama31-groq", "qwen3"
            tts_model: "chatterbox", "parler", "vibevoice", "orpheus", "inworld-tts-1.5-max"
        """
        import time

        t_start = time.time()

        asr = (
            get_or_load_model("asr", asr_model or self.config.asr, self.loaded_asr)
            or self.asr
        )
        llm = (
            get_or_load_model("llm", llm_model or self.config.llm, self.loaded_llm)
            or self.llm
        )
        tts = (
            get_or_load_model("tts", tts_model or self.config.tts, self.loaded_tts)
            or self.tts
        )

        audio_bytes, _, _ = normalize_audio_bytes(audio_bytes, log_decompress=True)
        if not audio_bytes:
            print("❌ Input audio bytes are empty")
            return {"error": "Empty input audio"}

        # Get input duration
        input_duration = get_wav_duration(audio_bytes)

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
        print(f"\n{'=' * 70}")
        print(f"Pipeline: {asr.model_name} → {llm.model_name} → {tts.model_name}")
        print(
            f"Total: {total_time:.2f}s (ASR:{asr_time:.2f}s LLM:{llm_time:.2f}s TTS:{tts_time:.2f}s)"
        )
        print(f"{'=' * 70}\n")

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
            },
        }

    @modal.method()
    def process_raw(
        self,
        audio_bytes: bytes,
        system_prompt: Optional[str] = None,
        asr_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        tts_model: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> Dict:
        import time

        t_start = time.time()
        asr = (
            get_or_load_model("asr", asr_model or self.config.asr, self.loaded_asr)
            or self.asr
        )
        llm = (
            get_or_load_model("llm", llm_model or self.config.llm, self.loaded_llm)
            or self.llm
        )
        tts = (
            get_or_load_model("tts", tts_model or self.config.tts, self.loaded_tts)
            or self.tts
        )
        audio_bytes = normalize_binary_audio(
            audio_bytes, sample_rate, channels, sample_width
        )
        if not audio_bytes:
            print("❌ Input audio bytes are empty")
            return {"error": "Empty input audio"}
        input_duration = get_wav_duration(audio_bytes)
        print(f"🎤 [{asr.model_name}] Transcribing...")
        transcription, asr_time = asr.transcribe(audio_bytes)
        print(f"   ✓ {asr_time:.2f}s: {transcription}")
        print(f"🤖 [{llm.model_name}] Generating...")
        response, llm_time = llm.generate(transcription, system_prompt)
        print(f"   ✓ {llm_time:.2f}s: {response}")
        print(f"🔊 [{tts.model_name}] Synthesizing...")
        audio_response, output_duration, tts_time = tts.synthesize(response)
        print(f"   ✓ {tts_time:.2f}s: {output_duration:.1f}s audio")
        total_time = time.time() - t_start
        return {
            "audio": audio_response,
            "transcription": transcription,
            "response": response,
            "compressed": False,
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
                "total_pipeline": total_time,
                "input_duration": input_duration,
                "output_duration": output_duration,
                "input_chars": len(transcription),
                "output_chars": len(response),
            },
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


@app.function(image=image, timeout=600)
def process_speech_raw(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> dict:
    service = SpeechToSpeechService()
    return service.process_raw.remote(
        audio_bytes,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


@app.function(image=image, timeout=600)
def process_speech_streaming_raw(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
):
    service = SpeechToSpeechService()
    for chunk in service.process_streaming_raw.remote_gen(
        audio_bytes,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    ):
        yield chunk


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
        },
    }


if ENABLE_LEGACY_WEB:

    @app.function(
        image=image,
        timeout=30,
        gpu=None,
        secrets=[modal.Secret.from_name("api-keys")],
    )
    @modal.fastapi_endpoint(method="POST")
    def get_tts_token(data: dict) -> dict:
        import base64
        import hashlib
        import hmac
        import json
        import os
        import time

        api_key = os.getenv("INWORLD_API_KEY")
        if not api_key:
            raise ValueError("INWORLD_API_KEY not set")

        try:
            decoded = base64.b64decode(api_key).decode("utf-8")
            key_id, key_secret = decoded.split(":", 1)
        except Exception:
            raise ValueError(
                "INWORLD_API_KEY must be base64-encoded 'key_id:key_secret'. "
                "Check your Inworld console."
            )

        now = int(time.time())
        header = (
            base64.urlsafe_b64encode(
                json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        payload = (
            base64.urlsafe_b64encode(
                json.dumps(
                    {
                        "iss": key_id,
                        "sub": key_id,
                        "iat": now,
                        "exp": now + 60,
                    }
                ).encode()
            )
            .rstrip(b"=")
            .decode()
        )

        sig_input = f"{header}.{payload}".encode()
        signature = (
            base64.urlsafe_b64encode(
                hmac.new(key_secret.encode(), sig_input, hashlib.sha256).digest()
            )
            .rstrip(b"=")
            .decode()
        )

        token = f"{header}.{payload}.{signature}"
        data = data or {}

        return {
            "token": token,
            "endpoint": "https://api.inworld.ai/tts/v1/voice:stream",
            "voice_id": data.get("voice_id", os.getenv("INWORLD_VOICE_ID", "Ashley")),
            "model_id": data.get("model_id", "inworld-tts-1.5-max"),
            "expires_in": 60,
        }


if ENABLE_LEGACY_WEB:

    @app.function(image=image, timeout=600)
    @modal.fastapi_endpoint(method="POST")
    def process_web(data: dict) -> dict:
        import base64

        print(f"📥 Received request data keys: {list(data.keys()) if data else 'None'}")
        print(f"📥 Data type: {type(data)}")
        audio_base64 = data.get("audio_base64", "") if data else ""
        if not audio_base64:
            print(f"❌ audio_base64 is empty. Full data: {str(data)[:500]}")
            raise ValueError(
                f"No audio data received. 'audio_base64' is empty or missing. Received keys: {list(data.keys()) if data else 'None'}"
            )
        print(f"📥 Received audio_base64: {len(audio_base64)} chars")
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {e}")
        if len(audio_bytes) < 100:
            raise ValueError(
                f"Audio data too small: {len(audio_bytes)} bytes. Expected valid audio file."
            )
        print(f"📥 Decoded audio: {len(audio_bytes)} bytes")
        system_prompt = data.get("system_prompt")
        asr_model = data.get("asr_model")
        tts_model = data.get("tts_model")
        import os

        cfg = ModelConfig()
        llm_requested = data.get("llm_model")
        groq_key = (
            os.environ.get("GROQ_API_KEY")
            or os.environ.get("GROQ_KEY")
            or os.environ.get("groq_api_key")
        )
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
        service = SpeechToSpeechService()
        try:
            result = service.process.remote(
                audio_bytes,
                system_prompt,
                asr_model=asr_model,
                llm_model=llm_model,
                tts_model=tts_model,
            )
        except Exception as e:
            print(
                f"⚠️ Error processing request (likely missing API key): {e}. Falling back to default models."
            )
            result = service.process.remote(audio_bytes, system_prompt)
        audio_out = result["audio"]
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
async def process_web_binary(request: Request):
    import os

    from fastapi import Request, Response

    body = await request.body()
    if not body:
        return Response(
            content=b"Empty audio", status_code=400, media_type="text/plain"
        )
    content_type = request.headers.get("content-type", "")
    audio_format = (request.headers.get("x-audio-format") or "").lower()
    is_pcm = (
        audio_format in {"pcm16", "pcm_s16le", "pcm", "raw"}
        or content_type == "audio/l16"
    )
    if not body.startswith(b"RIFF") and not is_pcm:
        return Response(
            content=b"Unsupported audio format",
            status_code=400,
            media_type="text/plain",
        )
    sample_rate = int(request.headers.get("x-sample-rate", "16000"))
    channels = int(request.headers.get("x-channels", "1"))
    sample_width = int(request.headers.get("x-sample-width", "2"))
    system_prompt = request.headers.get("x-system-prompt")
    asr_model = request.headers.get("x-asr-model")
    tts_model = request.headers.get("x-tts-model")
    llm_requested = request.headers.get("x-llm-model")
    cfg = ModelConfig()
    groq_key = (
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("GROQ_KEY")
        or os.environ.get("groq_api_key")
    )
    openai_key = os.environ.get("OPENAI_API_KEY")
    if llm_requested == "gpt4omini":
        llm_model = (
            "gpt4omini" if openai_key else ("llama31-groq" if groq_key else cfg.llm)
        )
    elif llm_requested == "llama31-groq":
        llm_model = (
            "llama31-groq" if groq_key else ("gpt4omini" if openai_key else cfg.llm)
        )
    else:
        llm_model = llm_requested
    service = SpeechToSpeechService()
    try:
        result = service.process_raw.remote(
            body,
            system_prompt,
            asr_model=asr_model,
            llm_model=llm_model,
            tts_model=tts_model,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )
    except Exception as e:
        print(f"⚠️ Error processing request: {e}. Falling back to default models.")
        result = service.process_raw.remote(
            body,
            system_prompt,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )
    audio_out = result.get("audio", b"")
    return Response(content=audio_out, media_type="application/octet-stream")


@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST")
async def process_web_stream_binary(request: Request):
    import struct

    from fastapi import Request, Response
    from fastapi.responses import StreamingResponse

    body = await request.body()
    if not body:
        return Response(
            content=b"Empty audio", status_code=400, media_type="text/plain"
        )
    content_type = request.headers.get("content-type", "")
    audio_format = (request.headers.get("x-audio-format") or "").lower()
    is_pcm = (
        audio_format in {"pcm16", "pcm_s16le", "pcm", "raw"}
        or content_type == "audio/l16"
    )
    if not body.startswith(b"RIFF") and not is_pcm:
        return Response(
            content=b"Unsupported audio format",
            status_code=400,
            media_type="text/plain",
        )
    sample_rate = int(request.headers.get("x-sample-rate", "16000"))
    channels = int(request.headers.get("x-channels", "1"))
    sample_width = int(request.headers.get("x-sample-width", "2"))
    system_prompt = request.headers.get("x-system-prompt")
    service = SpeechToSpeechService()

    def gen():
        try:
            for chunk in service.process_streaming_raw.remote_gen(
                body,
                system_prompt,
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
            ):
                if chunk.get("type") == "audio":
                    audio = chunk.get("audio", b"")
                    yield struct.pack(">I", len(audio)) + audio
                elif chunk.get("type") == "error":
                    break
            yield struct.pack(">I", 0)
        except Exception:
            yield struct.pack(">I", 0)

    return StreamingResponse(gen(), media_type="application/octet-stream")


if ENABLE_LEGACY_WEB:

    @app.function(image=image, timeout=600)
    @modal.asgi_app()
    def process_web_stream_text():
        import base64
        import os

        from fastapi import FastAPI, Request, Response
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse

        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.options("/")
        def _options():
            return Response(status_code=204)

        @app.post("/")
        async def _post(request: Request):
            content_type = request.headers.get("content-type", "")
            audio_base64 = ""
            data = {}
            if content_type.startswith("application/json"):
                try:
                    data = await request.json()
                    audio_base64 = (
                        data.get("audio_base64", "") if isinstance(data, dict) else ""
                    )
                except Exception:
                    audio_base64 = ""
            else:
                try:
                    body = await request.body()
                    audio_base64 = body.decode("utf-8")
                except Exception:
                    audio_base64 = ""
            if not audio_base64:
                return StreamingResponse(
                    iter(["Error: No audio data received"]), media_type="text/plain"
                )
            try:
                audio_bytes = base64.b64decode(audio_base64)
            except Exception as e:
                return StreamingResponse(
                    iter([f"Error: Invalid base64 audio data - {e}"]),
                    media_type="text/plain",
                )
            cfg = ModelConfig()
            asr_model = request.headers.get("x-asr-model") or (
                data.get("asr_model") if isinstance(data, dict) else None
            )
            llm_requested = request.headers.get("x-llm-model") or (
                data.get("llm_model") if isinstance(data, dict) else None
            )
            groq_key = (
                os.environ.get("GROQ_API_KEY")
                or os.environ.get("GROQ_KEY")
                or os.environ.get("groq_api_key")
            )
            openai_key = os.environ.get("OPENAI_API_KEY")
            if llm_requested == "gpt4omini":
                llm_model = (
                    "gpt4omini"
                    if openai_key
                    else ("llama31-groq" if groq_key else cfg.llm)
                )
            elif llm_requested == "llama31-groq":
                llm_model = (
                    "llama31-groq"
                    if groq_key
                    else ("gpt4omini" if openai_key else cfg.llm)
                )
            else:
                llm_model = llm_requested
            service = SpeechToSpeechService()

            def gen():
                try:
                    asr = service.asr
                    if asr_model and asr_model != service.config.asr:
                        if asr_model in service.loaded_asr:
                            asr = service.loaded_asr[asr_model]
                        elif asr_model in MODEL_REGISTRY["asr"]:
                            asr_class = MODEL_REGISTRY["asr"][asr_model]
                            asr = asr_class()
                            asr.load()
                            service.loaded_asr[asr_model] = asr
                    llm = service.llm
                    if llm_model and llm_model != service.config.llm:
                        if llm_model in service.loaded_llm:
                            llm = service.loaded_llm[llm_model]
                        elif llm_model in MODEL_REGISTRY["llm"]:
                            llm_class = MODEL_REGISTRY["llm"][llm_model]
                            llm = llm_class()
                            llm.load()
                            service.loaded_llm[llm_model] = llm
                    transcription, _ = asr.transcribe(audio_bytes)
                    yield transcription + "\n\n"
                    for token in llm.generate_stream(transcription):
                        yield token
                except Exception as e:
                    yield f"\n\nError: {str(e)}"

            return StreamingResponse(gen(), media_type="text/plain")

        return app


# Health check endpoint for keep-alive (prevents cold starts)
@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint - ping every 30s to keep container warm."""
    import time

    return {"status": "warm", "timestamp": time.time()}
