"""
Speech-to-Speech Pipeline on Modal (Single Container)

All models run in ONE container for minimal latency:
- ASR: NeMo RNNT (nvidia/nemotron-speech-streaming-en-0.6b)
- LLM: Phi-3-Mini (microsoft/Phi-3-mini-4k-instruct)
- TTS: ChatterboxTTS

Usage:
    modal deploy main.py                           # Deploy
    modal run main.py --audio-path input.wav       # Local test
    python client.py                               # Microphone client
"""
import modal

# =============================================================================
# Modal App & Image Configuration
# =============================================================================

app = modal.App("speech-to-speech")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "git",
        "build-essential",
    )
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    )
    .pip_install(
        "nemo-toolkit[asr]>=1.0.0",
    )
    .pip_install(
        "transformers>=4.36.0",
        "accelerate>=0.20.0",
    )
    .pip_install(
        "chatterbox-tts>=0.1.0",
    )
    .pip_install(
        "pydub>=0.25.0",
        "lameenc>=1.2.0",
    )
)

# Audio Compression Utilities (Inlined to avoid module import issues on Modal)

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

# Unified Speech-to-Speech Service (ALL MODELS IN ONE CONTAINER)
@app.cls(
    image=image,
    gpu="A10G",
    min_containers=1,
    timeout=1800,
    secrets=[modal.Secret.from_name("hf-secret")],
)
class SpeechToSpeechService:
    """
    Complete speech-to-speech pipeline in a single container.
    
    Benefits:
    - No inter-container network latency
    - Single cold start instead of three
    - Lower cost (one GPU instead of three)
    
    Memory footprint (~15GB on 24GB A10G):
    - NeMo RNNT: ~2-3GB
    - Phi-3-Mini: ~8GB (fp16)
    - ChatterboxTTS: ~2-3GB
    """

    @modal.enter()
    def load_all_models(self):
        """Load all models ONCE when container starts."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from nemo.collections.asr.models import EncDecRNNTBPEModel
        from chatterbox.tts import ChatterboxTTS
        
        # Store compression utilities for later use (now inlined above)
        self.decompress_audio = decompress_mp3_to_wav
        self.compress_audio = compress_wav_to_mp3
        
        print("=" * 60)
        print("🚀 Loading all models into single container...")
        print("=" * 60)
        
        # 1. ASR Model
        print("\n🎤 [1/3] Loading NeMo RNNT...")
        self.asr_model = (
            EncDecRNNTBPEModel
            .from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")
            .cuda()
            .eval()
        )
        print("✅ ASR model loaded")
        
        # 2. LLM Model
        print("\n🤖 [2/3] Loading Phi-3-Mini...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        self.llm_model.eval()
        print("✅ LLM model loaded")
        
        # 3. TTS Model
        print("\n🔊 [3/3] Loading ChatterboxTTS Turbo...")
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        
        self.tts_model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        print("✅ TTS model loaded")
        
        # Check VRAM usage
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n📊 VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")
        print("=" * 60)
        print("✅ All models loaded - ready for inference!")
        print("=" * 60)

    # =========================================================================
    # ASR: Speech-to-Text
    # =========================================================================
    
    def _transcribe(self, audio_bytes: bytes) -> tuple:
        """Transcribe audio to text. Returns (text, time)."""
        import tempfile
        import os
        import time
        from scipy.io import wavfile
        import io
        
        t0 = time.time()
        
        # Validate audio before processing
        try:
            with io.BytesIO(audio_bytes) as f:
                sr, data = wavfile.read(f)
            audio_duration = len(data) / sr
            print(f"   📊 Audio stats: {sr}Hz, {len(data)} samples, {audio_duration:.2f}s")
            
            # Check if audio has actual content
            import numpy as np
            audio_rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
            print(f"   📊 Audio RMS level: {audio_rms:.1f}")
            if audio_rms < 50:  # Very quiet audio threshold
                print(f"   ⚠️  Audio appears very quiet (RMS={audio_rms:.1f})")
        except Exception as e:
            print(f"   ⚠️  Could not validate audio: {e}")
        
        # NeMo requires file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            result = self.asr_model.transcribe([temp_path])
            print(f"   📊 ASR raw result type: {type(result)}, len: {len(result) if result else 0}")
            
            if result:
                hypothesis = result[0]
                print(f"   📊 Hypothesis type: {type(hypothesis)}")
                
                if hasattr(hypothesis, 'text'):
                    text = hypothesis.text
                elif hasattr(hypothesis, 'words'):
                    text = ' '.join(hypothesis.words)
                elif isinstance(hypothesis, str):
                    text = hypothesis
                else:
                    text = str(hypothesis)
                    print(f"   📊 Used str() fallback: {text[:100]}..." if len(text) > 100 else f"   📊 Used str() fallback: {text}")
            else:
                text = ""
                print("   ⚠️  ASR returned empty result!")
        except Exception as e:
            print(f"   ❌ ASR error: {e}")
            text = ""
        finally:
            os.unlink(temp_path)
        
        # Log empty transcription warning
        if not text.strip():
            print(f"   ⚠️  Empty transcription from {audio_duration:.1f}s of audio")
        
        elapsed = time.time() - t0
        return text.strip(), elapsed

    # =========================================================================
    # LLM: Text Generation (Voice Assistant Mode)
    # =========================================================================
    
    def _generate_response(self, user_input: str) -> tuple:
        """Generate short voice response. Returns (response, time)."""
        import torch
        import time
        import re
        
        t0 = time.time()
        
        # Handle empty/very short input
        if not user_input or len(user_input.strip()) < 2:
            print(f"   ⚠️  Empty or too short input, using default response")
            return "I didn't catch that. Could you please repeat?", 0.0
        
        # Phi-3 chat format with system instruction
        voice_prompt = f"""<|system|>
You are a helpful assistant. Answer directly and completely. If asked to count or list, provide the full count or list.<|end|>
<|user|>
{user_input}<|end|>
<|assistant|>"""
        
        inputs = self.tokenizer(
            voice_prompt,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.3,  # Low temp for more predictable responses
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response (after the user message)
        if user_input in full_response:
            response = full_response.split(user_input)[-1]
        else:
            response = full_response
        
        # Clean up
        response = response.strip()
        response = re.sub(r'<\|[^|]*\|>', '', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Truncate for speech (max 200 chars to allow for lists)
        response = self._truncate_for_speech(response, max_chars=200)
        
        elapsed = time.time() - t0
        return response, elapsed
    
    def _truncate_for_speech(self, text: str, max_chars: int = 300) -> str:
        """Keep text within limits for TTS - preserves lists and counting."""
        if not text:
            return "I understand."
        
        # Just truncate by character limit, don't split by sentence
        # This preserves lists like "one, two, three..."
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(' ', 1)[0] + '...'
        
        return text

    # =========================================================================
    # TTS: Text-to-Speech
    # =========================================================================
    
    def _synthesize(self, text: str) -> tuple:
        """Synthesize text to speech. Returns (audio_bytes, duration, time)."""
        import io
        import time
        import numpy as np
        from scipy.io import wavfile
        
        t0 = time.time()
        
        # Safety truncation for TTS (allow longer for lists)
        if len(text) > 300:
            text = text[:300].rsplit(' ', 1)[0] + '...'
        
        audio_tensor = self.tts_model.generate(text)
        audio_np = audio_tensor.cpu().numpy()
        
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        
        if audio_np.dtype in [np.float32, np.float64]:
            max_val = np.abs(audio_np).max()
            if max_val > 1.0:
                audio_np = audio_np / max_val
            audio_np = (audio_np * 32767).astype(np.int16)
        elif audio_np.dtype != np.int16:
            audio_np = audio_np.astype(np.int16)
        
        buffer = io.BytesIO()
        sample_rate = 24000
        wavfile.write(buffer, sample_rate, audio_np)
        audio_bytes = buffer.getvalue()
        
        audio_duration = len(audio_np) / sample_rate
        elapsed = time.time() - t0
        
        return audio_bytes, audio_duration, elapsed

    # =========================================================================
    # Main Pipeline Method (Called Remotely)
    # =========================================================================
    
    @modal.method()
    def process(self, audio_bytes: bytes) -> dict:
        """
        Complete speech-to-speech pipeline with audio compression support.
        
        All processing happens IN-PROCESS (no network calls between models).
        Automatically detects and handles compressed audio input.
        
        Returns dict with audio, transcription, response, and detailed metrics.
        """
        import time
        from scipy.io import wavfile
        import io
        
        pipeline_start = time.time()
        
        # Handle compressed audio input
        input_compressed = False
        original_size = len(audio_bytes)
        compression_ratio = 1.0
        
        # Try to detect if input is compressed (MP3)
        if not audio_bytes.startswith(b'RIFF'):  # Not WAV format
            try:
                print(f"📦 Received compressed audio: {original_size} bytes")
                decompressed_bytes = self.decompress_audio(audio_bytes)
                print(f"📦 Decompressed to WAV: {len(decompressed_bytes)} bytes")
                compression_ratio = len(decompressed_bytes) / original_size
                audio_bytes = decompressed_bytes
                input_compressed = True
            except Exception as e:
                print(f"⚠️  Failed to decompress audio, assuming WAV: {e}")
        
        # Get input audio duration
        try:
            with io.BytesIO(audio_bytes) as f:
                sr, data = wavfile.read(f)
                input_duration = len(data) / sr
        except:
            input_duration = 0.0
        
        # Step 1: ASR (in-process, no network)
        print("🎤 [1/3] Transcribing...")
        transcription, asr_time = self._transcribe(audio_bytes)
        print(f"   ✓ {asr_time:.2f}s: {transcription}")
        
        # Step 2: LLM (in-process, no network)
        print("🤖 [2/3] Generating response...")
        response, llm_time = self._generate_response(transcription)
        print(f"   ✓ {llm_time:.2f}s: {response}")
        
        # Step 3: TTS (in-process, no network)
        print("🔊 [3/3] Synthesizing...")
        audio_response, output_duration, tts_time = self._synthesize(response)
        print(f"   ✓ {tts_time:.2f}s: {len(response)} chars → {output_duration:.1f}s audio")
        
        # Compress audio response for network transmission
        original_audio_size = len(audio_response)
        audio_response = self.compress_audio(audio_response)
        compressed_audio_size = len(audio_response)
        audio_compression_ratio = original_audio_size / compressed_audio_size if compressed_audio_size > 0 else 1.0
        print(f"📦 Compressed response: {original_audio_size} → {compressed_audio_size} bytes ({audio_compression_ratio:.1f}x)")
        
        total_time = time.time() - pipeline_start
        
        # Print metrics
        print(f"\n" + "="*60)
        print(f"{'PIPELINE METRICS (SINGLE CONTAINER)':^60}")
        print("="*60)
        print(f"{'Component':<15} | {'Time (s)':<10} | {'%':<8} | {'Details':<20}")
        print("-"*60)
        print(f"{'ASR':<15} | {asr_time:<10.3f} | {asr_time/total_time*100:>6.1f}% | {input_duration:.1f}s → {len(transcription)} chars")
        print(f"{'LLM':<15} | {llm_time:<10.3f} | {llm_time/total_time*100:>6.1f}% | {len(transcription.split())}→{len(response.split())} words")
        print(f"{'TTS':<15} | {tts_time:<10.3f} | {tts_time/total_time*100:>6.1f}% | {len(response)} chars → {output_duration:.1f}s")
        print("-"*60)
        print(f"{'TOTAL':<15} | {total_time:<10.3f} | {'100%':<8} | RTF: {total_time/max(input_duration, 0.1):.2f}x")
        print("="*60 + "\n")
        
        return {
            "audio": audio_response,
            "transcription": transcription,
            "response": response,
            "compressed": True,  # Signal that audio is compressed
            "metrics": {
                "asr_time": asr_time,
                "llm_time": llm_time,
                "tts_time": tts_time,
                "total_time": total_time,
                "input_duration": input_duration,
                "output_duration": output_duration,
                "input_chars": len(transcription),
                "output_chars": len(response),
                "input_compression_ratio": compression_ratio if input_compressed else 1.0,
                "output_compression_ratio": audio_compression_ratio,
            }
        }

# =============================================================================
# Remote Function Wrapper (for client.py compatibility)
# =============================================================================

@app.function(image=image, timeout=600)
def process_speech(audio_bytes: bytes) -> dict:
    """Wrapper function for backward compatibility with client.py"""
    service = SpeechToSpeechService()
    return service.process.remote(audio_bytes)

# =============================================================================
# Local Entrypoint for Testing
# =============================================================================

@app.local_entrypoint()
def main(audio_path: str):
    """
    Local testing entrypoint.
    
    Usage:
        modal run main.py --audio-path input.wav
    """
    import time
    
    print(f"📂 Reading audio from {audio_path}...")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    print(f"   Loaded {len(audio_bytes)} bytes")
    
    # Use the service directly
    service = SpeechToSpeechService()
    
    print("\n🚀 Running speech-to-speech pipeline (single container)...")
    t0 = time.time()
    result = service.process.remote(audio_bytes)
    total_time = time.time() - t0
    
    # Save output
    output_path = "output.wav"
    with open(output_path, "wb") as f:
        f.write(result["audio"])
    
    print(f"\n✅ Complete!")
    print(f"   Network + pipeline time: {total_time:.2f}s")
    print(f"   Output saved to {output_path}")
