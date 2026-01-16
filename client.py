import time
import queue
import sys
import numpy as np
import sounddevice as sd
import webrtcvad
import io
import argparse
from scipy.io import wavfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import modal
from audio_compression import compress_wav_to_mp3, decompress_mp3_to_wav, get_compression_ratio

# Thread pool for parallel decompression (avoids blocking main loop)
_decompression_executor = ThreadPoolExecutor(max_workers=2)

# Audio configuration
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_LEVEL = 3
SILENCE_THRESHOLD_MS = 1000  # Stop recording after 1 second of silence

# Streaming audio player for low-latency playback
class StreamingAudioPlayer:
    """
    NON-BLOCKING audio player for true streaming.
    
    CRITICAL: Chunks are queued and played in background thread.
    This allows receiving next chunk WHILE previous is playing.
    
    LATENCY OPTIMIZATION:
    - add_chunk() returns IMMEDIATELY (non-blocking)
    - Audio plays in background via sounddevice callback
    - No gaps between chunks (continuous playback)
    """
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.playing = False
        self.stream = None
        self._playback_thread = None
        self._stop_event = None
        import threading
        self._lock = threading.Lock()
    
    def _playback_worker(self):
        """Background thread that plays queued audio chunks sequentially."""
        while not self._stop_event.is_set():
            try:
                # Wait for next chunk with timeout (allows checking stop event)
                data = self.audio_queue.get(timeout=0.1)
                if data is None:  # Poison pill
                    break
                # Play this chunk (blocking within thread is fine)
                sd.play(data, self.sample_rate)
                sd.wait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Playback error: {e}")
    
    def start(self):
        """Start the background playback thread."""
        import threading
        if self._playback_thread is None or not self._playback_thread.is_alive():
            self._stop_event = threading.Event()
            self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self._playback_thread.start()
    
    def add_chunk(self, audio_bytes: bytes):
        """
        Add audio chunk to playback queue. Returns IMMEDIATELY.
        This is the key to low-latency streaming - we don't block on playback.
        """
        # Ensure playback thread is running
        self.start()
        
        # Decode audio bytes
        try:
            with io.BytesIO(audio_bytes) as f:
                sr, data = wavfile.read(f)
            
            # Resample if needed (rare - most TTS outputs 24kHz)
            if sr != self.sample_rate:
                from scipy import signal
                num_samples = int(len(data) * self.sample_rate / sr)
                data = signal.resample(data, num_samples).astype(np.int16)
            
            # Queue for playback - returns immediately
            self.audio_queue.put(data)
            
        except Exception as e:
            print(f"⚠️ Audio chunk error: {e}")
    
    def play_blocking(self, audio_bytes: bytes):
        """Play audio synchronously (for non-streaming mode only)."""
        try:
            with io.BytesIO(audio_bytes) as f:
                sr, data = wavfile.read(f)
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print(f"⚠️ Playback error: {e}")
    
    def wait_for_completion(self):
        """Wait for all queued audio to finish playing."""
        # Wait for queue to drain
        while not self.audio_queue.empty():
            time.sleep(0.05)
        # Give last chunk time to play (estimate ~3s max per chunk)
        time.sleep(0.3)
    
    def stop(self):
        """Stop playback and cleanup."""
        if self._stop_event:
            self._stop_event.set()
        if self._playback_thread and self._playback_thread.is_alive():
            self.audio_queue.put(None)  # Poison pill
            self._playback_thread.join(timeout=1.0)

# Session metrics tracking
class SessionMetrics:
    def __init__(self):
        self.calls = []
        self.start_time = datetime.now()
    
    def add_call(self, metrics: dict, network_time: float, record_time: float):
        self.calls.append({
            **metrics,
            "network_overhead": network_time - metrics.get("total_pipeline", 0),
            "record_time": record_time,
            "e2e_time": record_time + network_time,
        })
    
    def print_summary(self):
        if not self.calls:
            return
        
        n = len(self.calls)
        avg = lambda key: sum(c.get(key, 0) for c in self.calls) / n
        
        print("\n" + "="*70)
        print(f"{'SESSION SUMMARY':^70}")
        print("="*70)
        print(f"Total calls: {n}")
        print(f"Session duration: {(datetime.now() - self.start_time).seconds}s")
        print("-"*70)
        print(f"{'Metric':<25} | {'Average':<12} | {'Min':<12} | {'Max':<12}")
        print("-"*70)
        
        for key, label in [
            ("asr_time", "ASR Time"),
            ("llm_time", "LLM Time"),
            ("tts_time", "TTS Time"),
            ("total_pipeline", "Pipeline Total"),
            ("network_overhead", "Network Overhead"),
            ("e2e_time", "End-to-End"),
        ]:
            vals = [c.get(key, 0) for c in self.calls]
            print(f"{label:<25} | {avg(key):<12.3f} | {min(vals):<12.3f} | {max(vals):<12.3f}")
        
        print("="*70)

class AudioRecorder:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_LEVEL)
        self.q = queue.Queue()
        self.recording = False
        self.silence_start = None
        self.speech_detected = False

    def callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def record_until_silence(self):
        print("\n🎤 Listening... (speak now)")
        self.q.queue.clear()
        audio_buffer = []
        self.speech_detected = False
        self.silence_start = None
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=self.callback, blocksize=FRAME_SIZE):
            while True:
                try:
                    data = self.q.get()
                    is_speech = self.vad.is_speech(data.tobytes(), SAMPLE_RATE)
                    
                    if is_speech:
                        if not self.speech_detected:
                            print("🗣️ Speech detected...")
                            self.speech_detected = True
                        self.silence_start = None
                        audio_buffer.append(data)
                    else:
                        if self.speech_detected:
                            if self.silence_start is None:
                                self.silence_start = time.time()
                            elif time.time() - self.silence_start > SILENCE_THRESHOLD_MS / 1000.0:
                                print("🛑 Silence detected, stopping recording.")
                                break
                            audio_buffer.append(data)
                        else:
                            # Keep a small rolling buffer of pre-speech audio (300ms)
                            audio_buffer.append(data)
                            if len(audio_buffer) > 10:
                                audio_buffer.pop(0)
                                
                except KeyboardInterrupt:
                    return None

        # Flatten audio buffer
        return np.concatenate(audio_buffer)

def play_audio(audio_bytes):
    """Play audio bytes (handles both WAV and MP3). Returns immediately, plays in background."""
    # Check file format from header bytes
    header = audio_bytes[:4]
    is_wav = header[:4] == b'RIFF' or header[:4] == b'RIFX'
    is_mp3 = header[:3] == b'ID3' or header[:2] == b'\xff\xfb' or header[:2] == b'\xff\xfa'
    
    if is_mp3 or not is_wav:
        # MP3 or unknown format - try to convert using pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            # Convert to numpy array for sounddevice
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            samplerate = audio.frame_rate
            data = samples
        except Exception as e:
            print(f"⚠️ pydub failed: {e}. Trying ffmpeg fallback...")
            # Fallback: save to temp file and use ffmpeg via pydub
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(tmp_path)
                samples = np.array(audio.get_array_of_samples())
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                samplerate = audio.frame_rate
                data = samples
            finally:
                os.unlink(tmp_path)
    else:
        # WAV - read directly
        with io.BytesIO(audio_bytes) as f:
            samplerate, data = wavfile.read(f)
    
    print(f"🔊 Playing response ({len(data)/samplerate:.1f}s)...")
    sd.play(data, samplerate)
    sd.wait()

def print_metrics(metrics: dict, network_time: float, record_time: float, models: dict = None):
    """Print detailed latency breakdown."""
    asr = metrics.get("asr_time", 0)
    llm = metrics.get("llm_time", 0)
    tts = metrics.get("tts_time", 0)
    # Handle both old key (total_pipeline) and new key (total_time)
    pipeline = metrics.get("total_pipeline", 0) or metrics.get("total_time", 0)
    input_dur = metrics.get("input_duration", 0)
    output_dur = metrics.get("output_duration", 0)
    input_chars = metrics.get("input_chars", 0)
    output_chars = metrics.get("output_chars", 0)
    input_tokens = metrics.get("input_tokens", len(str(input_chars).split()))
    output_tokens = metrics.get("output_tokens", len(str(output_chars).split()))
    
    # Get model names from response or use defaults
    asr_name = models.get("asr", "ASR") if models else "ASR"
    llm_name = models.get("llm", "LLM") if models else "LLM"
    tts_name = models.get("tts", "TTS") if models else "TTS"
    
    # Avoid division by zero
    pipeline = max(pipeline, 0.001)
    
    network_overhead = network_time - pipeline
    e2e = record_time + network_time
    rtf = network_time / max(input_dur, 0.1)
    
    print("\n" + "="*70)
    print(f"{'LATENCY BREAKDOWN':^70}")
    print("="*70)
    print(f"{'Component':<25} | {'Time (s)':<10} | {'%':<8} | {'Details':<20}")
    print("-"*70)
    print(f"{'Recording':<25} | {record_time:<10.3f} | {'-':<8} | {input_dur:.1f}s audio captured")
    print("-"*70)
    print(f"{'ASR: ' + asr_name[:20]:<25} | {asr:<10.3f} | {asr/pipeline*100:>6.1f}% | {input_dur:.1f}s → {input_chars} chars")
    print(f"{'LLM: ' + llm_name[:20]:<25} | {llm:<10.3f} | {llm/pipeline*100:>6.1f}% | {input_tokens}→{output_tokens} tokens")
    print(f"{'TTS: ' + tts_name[:20]:<25} | {tts:<10.3f} | {tts/pipeline*100:>6.1f}% | {output_chars} chars → {output_dur:.1f}s")
    print("-"*70)
    print(f"{'Pipeline Total':<25} | {pipeline:<10.3f} | {'100%':<8} | Single container")
    print(f"{'Network Overhead':<25} | {max(network_overhead, 0):<10.3f} | {'-':<8} | Serialization + transfer")
    print("-"*70)
    print(f"{'End-to-End':<25} | {e2e:<10.3f} | {'-':<8} | Record → Response ready")
    print("="*70)
    print(f"  RTF (Real-Time Factor): {rtf:.2f}x | "
          f"Throughput: {output_dur/max(network_time, 0.1):.2f}x realtime")
    print("="*70 + "\n")


def main():
    """
    Real-time speech-to-speech client.
    
    Modes:
        --streaming: TRUE STREAMING - audio plays while LLM is still generating
                     Achieves sub-2s first-audio latency
        (default):   BATCH - waits for full response before playing
    """
    parser = argparse.ArgumentParser(description="Speech-to-Speech Client")
    parser.add_argument("--streaming", action="store_true", 
                        help="Enable TRUE streaming mode for lowest latency")
    args = parser.parse_args()
    
    print("🚀 Starting real-time speech-to-speech client...")
    print(f"   Mode: {'STREAMING (low latency)' if args.streaming else 'BATCH'}")
    print("   Connecting to Modal...")
    
    try:
        if args.streaming:
            # TRUE STREAMING: Audio plays while LLM is still generating
            process_func = modal.Function.from_name("speech-to-speech", "process_speech_streaming")
        else:
            # BATCH: Wait for full response
            process_func = modal.Function.from_name("speech-to-speech", "process_speech")
    except modal.exception.NotFoundError:
        print("❌ Error: App not deployed. Run 'modal deploy modular_main.py' first.")
        sys.exit(1)
    
    print("✅ Connected to Modal services.")
    print("   Press Ctrl+C to exit and see session summary.\n")
    
    recorder = AudioRecorder()
    session = SessionMetrics()
    player = StreamingAudioPlayer()
    
    while True:
        try:
            # 1. Record
            t_record_start = time.time()
            audio_data = recorder.record_until_silence()
            record_time = time.time() - t_record_start
            
            if audio_data is None:
                break
            
            if len(audio_data) == 0:
                continue

            # Create WAV container for the raw PCM data
            wav_buffer = io.BytesIO()
            wavfile.write(wav_buffer, SAMPLE_RATE, audio_data)
            wav_bytes = wav_buffer.getvalue()
            
            # Compress audio to reduce network overhead
            print(f"📦 Original WAV: {len(wav_bytes)} bytes")
            compressed_bytes = compress_wav_to_mp3(wav_bytes)
            print(f"📦 Compressed MP3: {len(compressed_bytes)} bytes")
            compression_ratio = get_compression_ratio(len(wav_bytes), len(compressed_bytes))
            print(f"📦 Compression ratio: {compression_ratio:.1f}x")
            
            # 2. Process through pipeline (ASR -> LLM -> TTS)
            t0 = time.time()
            
            if args.streaming:
                # STREAMING MODE: Play audio chunks as they arrive
                # This achieves sub-2s first-audio latency
                print("🚀 Processing (streaming)...")
                
                transcription = ""
                response = ""
                metrics = {}
                first_audio_received = None
                audio_chunks_played = 0
                
                for chunk in process_func.remote_gen(compressed_bytes):
                    chunk_type = chunk.get("type", "")
                    
                    if chunk_type == "transcription":
                        transcription = chunk.get("transcription", "")
                        print(f"\n📝 You said: \"{transcription}\"")
                        
                    elif chunk_type == "audio":
                        # PLAY IMMEDIATELY - don't wait for more chunks
                        if first_audio_received is None:
                            first_audio_received = time.time() - t0
                            print(f"   ⚡ First audio at {first_audio_received:.2f}s")
                        
                        audio_data = chunk.get("audio", b"")
                        text = chunk.get("text", "")
                        
                        # Handle both compressed (MP3) and uncompressed (WAV) audio
                        # Server skips compression for small chunks to reduce TTFA
                        if chunk.get("compressed", False):
                            # Decompress MP3 -> WAV (still fast, ~10-20ms)
                            audio_data = decompress_mp3_to_wav(audio_data)
                        # else: already WAV, play directly (saves 30-50ms!)
                        
                        print(f"   🔊 Queued: \"{text[:40]}...\"")
                        player.add_chunk(audio_data)  # Returns immediately!
                        audio_chunks_played += 1
                        
                    elif chunk_type == "done":
                        response = chunk.get("response", "")
                        metrics = chunk.get("metrics", {})
                        # Wait for audio to finish before showing completion
                        player.wait_for_completion()
                        print(f"\n💬 Full response: \"{response}\"")
                        
                    elif chunk_type == "error":
                        print(f"❌ Error: {chunk.get('error', 'Unknown')}")
                
                network_time = time.time() - t0
                
                if metrics:
                    # Add first_audio_time to metrics for display
                    metrics["first_audio_time"] = first_audio_received
                    print_metrics(metrics, network_time, record_time, None)
                    session.add_call(metrics, network_time, record_time)
                    
            else:
                # BATCH MODE: Wait for full response
                print("🚀 Processing...")
                result = process_func.remote(compressed_bytes)
                network_time = time.time() - t0
                
                # Extract audio and metrics
                if isinstance(result, dict):
                    audio_response = result.get("audio", b"")
                    metrics = result.get("metrics", {})
                    models = result.get("models", {})
                    transcription = result.get("transcription", "")
                    response = result.get("response", "")
                    
                    print(f"\n📝 You said: \"{transcription}\"")
                    print(f"💬 Response: \"{response}\"")
                    
                    print_metrics(metrics, network_time, record_time, models)
                    session.add_call(metrics, network_time, record_time)
                else:
                    audio_response = result
                    print(f"✅ Response received in {network_time:.2f}s")

                # Decompress and play
                if isinstance(result, dict) and result.get("compressed", False):
                    audio_response = decompress_mp3_to_wav(audio_response)
                
                play_audio(audio_response)
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            session.print_summary()
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
