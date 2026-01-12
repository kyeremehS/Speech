"""
Speech-to-Speech Pipeline Benchmarking Tool

Measures:
1. Latency - End-to-end and per-stage timing
2. Accent Understanding - WER (Word Error Rate) with ground truth
3. Real-time Factor (RTF) - Processing time vs audio duration

Usage:
    # Quick latency test with sample audio
    python benchmark.py --quick
    
    # Full benchmark with accent test dataset
    python benchmark.py --accent-test
    
    # Custom audio file
    python benchmark.py --audio path/to/audio.wav --expected "expected transcription"
    
    # Multiple iterations for statistical significance
    python benchmark.py --audio input.wav --iterations 10
"""

import modal
import time
import argparse
import json
import os
import wave
import struct
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from pathlib import Path

# Try to import optional dependencies
try:
    from jiwer import wer, cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    print("⚠️  Install jiwer for WER calculation: pip install jiwer")


@dataclass
class BenchmarkResult:
    """Single benchmark run result"""
    audio_file: str
    audio_duration: float
    asr_time: float
    llm_time: float
    tts_time: float
    total_time: float
    transcription: str
    response: str
    output_duration: float
    expected_text: Optional[str] = None
    wer_score: Optional[float] = None
    cer_score: Optional[float] = None
    
    @property
    def rtf(self) -> float:
        """Real-time factor (< 1 means faster than real-time)"""
        return self.total_time / self.audio_duration if self.audio_duration > 0 else 0
    
    @property
    def latency_breakdown(self) -> Dict[str, float]:
        return {
            "asr": self.asr_time,
            "llm": self.llm_time,
            "tts": self.tts_time,
            "total": self.total_time
        }


@dataclass  
class BenchmarkSummary:
    """Aggregated benchmark statistics"""
    num_runs: int
    avg_total_time: float
    avg_asr_time: float
    avg_llm_time: float
    avg_tts_time: float
    avg_rtf: float
    min_total_time: float
    max_total_time: float
    p50_total_time: float
    p95_total_time: float
    avg_wer: Optional[float] = None
    avg_cer: Optional[float] = None
    model_config: Optional[Dict[str, str]] = None


def generate_test_audio(duration: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate a simple sine wave test audio (for latency testing only)"""
    num_samples = int(duration * sample_rate)
    frequency = 440  # A4 note
    
    # Generate sine wave
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        # Add some variation to make it more speech-like
        value = int(16000 * math.sin(2 * math.pi * frequency * t) * 
                   (0.5 + 0.5 * math.sin(2 * math.pi * 2 * t)))
        samples.append(struct.pack('<h', max(-32768, min(32767, value))))
    
    # Create WAV file in memory
    import io
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b''.join(samples))
    
    return buffer.getvalue()


def get_audio_duration(audio_bytes: bytes) -> float:
    """Get duration of WAV audio in seconds"""
    import io
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return frames / rate


def calculate_accuracy(transcription: str, expected: str) -> Dict[str, float]:
    """Calculate WER and CER if jiwer is available"""
    if not HAS_JIWER or not expected:
        return {"wer": None, "cer": None}
    
    # Normalize text
    trans_norm = transcription.lower().strip()
    exp_norm = expected.lower().strip()
    
    return {
        "wer": wer(exp_norm, trans_norm),
        "cer": cer(exp_norm, trans_norm)
    }


def run_single_benchmark(
    service,
    audio_bytes: bytes,
    audio_file: str = "test",
    expected_text: Optional[str] = None,
    warmup: bool = False
) -> tuple:
    """Run a single benchmark iteration, returns (BenchmarkResult, model_config)"""
    
    audio_duration = get_audio_duration(audio_bytes)
    
    if warmup:
        print("   🔥 Warmup run (not counted)...")
    
    t_start = time.time()
    result = service.process.remote(audio_bytes)
    total_time = time.time() - t_start
    
    # Extract model configuration from response
    models_info = result.get("models", {})
    model_config = {
        "asr_model": models_info.get("asr", "unknown"),
        "llm_model": models_info.get("llm", "unknown"),
        "tts_model": models_info.get("tts", "unknown")
    }
    
    # Calculate accuracy if expected text provided
    accuracy = calculate_accuracy(result["transcription"], expected_text)
    
    return (BenchmarkResult(
        audio_file=audio_file,
        audio_duration=audio_duration,
        asr_time=result["metrics"]["asr_time"],
        llm_time=result["metrics"]["llm_time"],
        tts_time=result["metrics"]["tts_time"],
        total_time=total_time,
        transcription=result["transcription"],
        response=result["response"],
        output_duration=result["metrics"]["output_duration"],
        expected_text=expected_text,
        wer_score=accuracy["wer"],
        cer_score=accuracy["cer"]
    ), model_config)


def run_benchmark(
    audio_bytes: bytes,
    audio_file: str = "test",
    iterations: int = 5,
    expected_text: Optional[str] = None,
    warmup: bool = True
) -> tuple:
    """Run multiple benchmark iterations, returns (List[BenchmarkResult], model_config)"""
    
    # Connect to Modal service
    print("🔌 Connecting to Modal service...")
    SpeechToSpeechService = modal.Cls.from_name("speech-to-speech", "SpeechToSpeechService")
    service = SpeechToSpeechService()
    
    results = []
    model_config = None
    
    # Warmup run (not counted)
    if warmup:
        _, model_config = run_single_benchmark(service, audio_bytes, audio_file, expected_text, warmup=True)
    
    # Actual benchmark runs
    print(f"\n📊 Running {iterations} benchmark iterations...")
    for i in range(iterations):
        print(f"   Run {i+1}/{iterations}...", end=" ", flush=True)
        result, model_config = run_single_benchmark(service, audio_bytes, audio_file, expected_text)
        results.append(result)
        print(f"✓ {result.total_time:.2f}s (RTF: {result.rtf:.2f})")
    
    return results, model_config


def calculate_summary(results: List[BenchmarkResult], model_config: Optional[Dict[str, str]] = None) -> BenchmarkSummary:
    """Calculate aggregate statistics from benchmark results"""
    import statistics
    
    total_times = [r.total_time for r in results]
    total_times_sorted = sorted(total_times)
    
    wer_scores = [r.wer_score for r in results if r.wer_score is not None]
    cer_scores = [r.cer_score for r in results if r.cer_score is not None]
    
    n = len(results)
    p50_idx = int(n * 0.5)
    p95_idx = min(int(n * 0.95), n - 1)
    
    return BenchmarkSummary(
        num_runs=n,
        avg_total_time=statistics.mean(total_times),
        avg_asr_time=statistics.mean([r.asr_time for r in results]),
        avg_llm_time=statistics.mean([r.llm_time for r in results]),
        avg_tts_time=statistics.mean([r.tts_time for r in results]),
        avg_rtf=statistics.mean([r.rtf for r in results]),
        min_total_time=min(total_times),
        max_total_time=max(total_times),
        p50_total_time=total_times_sorted[p50_idx],
        p95_total_time=total_times_sorted[p95_idx],
        avg_wer=statistics.mean(wer_scores) if wer_scores else None,
        avg_cer=statistics.mean(cer_scores) if cer_scores else None,
        model_config=model_config
    )


def print_results(results: List[BenchmarkResult], summary: BenchmarkSummary):
    """Print formatted benchmark results"""
    
    print("\n" + "=" * 70)
    print("📈 BENCHMARK RESULTS")
    print("=" * 70)
    
    # Model Configuration
    if summary.model_config:
        print("\n🔧 MODEL CONFIGURATION")
        print("-" * 50)
        print(f"   ASR Model: {summary.model_config.get('asr_model', 'unknown')}")
        print(f"   LLM Model: {summary.model_config.get('llm_model', 'unknown')}")
        print(f"   TTS Model: {summary.model_config.get('tts_model', 'unknown')}")
    
    # Latency Results
    print("\n⏱️  LATENCY METRICS")
    print("-" * 50)
    print(f"   Total Runs:        {summary.num_runs}")
    print(f"   Avg Total Time:    {summary.avg_total_time:.3f}s")
    print(f"   Min Total Time:    {summary.min_total_time:.3f}s")
    print(f"   Max Total Time:    {summary.max_total_time:.3f}s")
    print(f"   P50 (Median):      {summary.p50_total_time:.3f}s")
    print(f"   P95:               {summary.p95_total_time:.3f}s")
    print(f"   Real-Time Factor:  {summary.avg_rtf:.2f}x")
    
    # Stage Breakdown
    print("\n📊 STAGE BREAKDOWN (averages)")
    print("-" * 50)
    total_avg = summary.avg_asr_time + summary.avg_llm_time + summary.avg_tts_time
    print(f"   ASR:  {summary.avg_asr_time:.3f}s ({100*summary.avg_asr_time/total_avg:.1f}%)")
    print(f"   LLM:  {summary.avg_llm_time:.3f}s ({100*summary.avg_llm_time/total_avg:.1f}%)")
    print(f"   TTS:  {summary.avg_tts_time:.3f}s ({100*summary.avg_tts_time/total_avg:.1f}%)")
    
    # Find bottleneck
    stages = {"ASR": summary.avg_asr_time, "LLM": summary.avg_llm_time, "TTS": summary.avg_tts_time}
    bottleneck = max(stages, key=stages.get)
    print(f"\n   ⚠️  Bottleneck: {bottleneck} ({stages[bottleneck]:.3f}s)")
    
    # Accuracy Results (if available)
    if summary.avg_wer is not None:
        print("\n🎯 ACCURACY METRICS")
        print("-" * 50)
        print(f"   Word Error Rate (WER):  {summary.avg_wer:.2%}")
        print(f"   Char Error Rate (CER):  {summary.avg_cer:.2%}")
        
        # Interpret WER
        if summary.avg_wer < 0.05:
            quality = "Excellent"
        elif summary.avg_wer < 0.10:
            quality = "Good"
        elif summary.avg_wer < 0.20:
            quality = "Fair"
        else:
            quality = "Needs Improvement"
        print(f"   Quality:                {quality}")
    
    # Sample transcription
    print("\n📝 SAMPLE OUTPUT (last run)")
    print("-" * 50)
    last = results[-1]
    print(f"   Input Audio:    {last.audio_duration:.2f}s")
    print(f"   Transcription:  {last.transcription[:80]}{'...' if len(last.transcription) > 80 else ''}")
    if last.expected_text:
        print(f"   Expected:       {last.expected_text[:80]}{'...' if len(last.expected_text) > 80 else ''}")
    print(f"   Response:       {last.response[:80]}{'...' if len(last.response) > 80 else ''}")
    print(f"   Output Audio:   {last.output_duration:.2f}s")
    
    print("\n" + "=" * 70)


def save_results(results: List[BenchmarkResult], summary: BenchmarkSummary, output_file: str):
    """Save results to JSON file"""
    data = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")


# Accent test dataset (you can expand this)
ACCENT_TEST_SAMPLES = [
    {
        "name": "US English",
        "url": "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav",
        "expected": "The birch canoe slid on the smooth planks"
    },
    {
        "name": "UK English", 
        "url": "https://www.voiptroubleshooter.com/open_speech/british/OSR_uk_000_0010_8k.wav",
        "expected": "The birch canoe slid on the smooth planks"
    },
]


def run_accent_benchmark():
    """Run benchmark with various accent samples"""
    print("🌍 Accent Understanding Benchmark")
    print("=" * 70)
    print("\nℹ️  For comprehensive accent testing, prepare audio samples with:")
    print("   - Ground truth transcriptions")
    print("   - Various accents (US, UK, Indian, Australian, etc.)")
    print("   - Different noise levels")
    print("   - Various speaking speeds")
    print("\n📁 Place test files in ./benchmark_data/ folder with format:")
    print("   audio_file.wav + audio_file.txt (ground truth)")
    
    benchmark_dir = Path("./benchmark_data")
    if not benchmark_dir.exists():
        benchmark_dir.mkdir()
        print(f"\n✨ Created {benchmark_dir} - add your test files there")
        
        # Create sample structure
        readme = """# Benchmark Test Data

Add your audio files here for accent testing:

## File Format
- audio_sample.wav - The audio file (16kHz mono recommended)
- audio_sample.txt - Ground truth transcription

## Recommended Test Categories
1. **Accents**: US, UK, Indian, Australian, Nigerian, etc.
2. **Noise Levels**: Clean, light noise, heavy noise
3. **Speaking Speed**: Slow, normal, fast
4. **Audio Length**: Short (2-5s), Medium (5-15s), Long (15-30s)

## Sample Sources
- LibriSpeech: https://www.openslr.org/12
- Common Voice: https://commonvoice.mozilla.org/
- VoxForge: http://www.voxforge.org/
"""
        (benchmark_dir / "README.md").write_text(readme)
        return
    
    # Find audio files with transcriptions
    all_results = []
    
    for wav_file in benchmark_dir.glob("*.wav"):
        txt_file = wav_file.with_suffix(".txt")
        expected = txt_file.read_text().strip() if txt_file.exists() else None
        
        print(f"\n📂 Testing: {wav_file.name}")
        if expected:
            print(f"   Expected: {expected[:50]}...")
        
        audio_bytes = wav_file.read_bytes()
        results, model_config = run_benchmark(
            audio_bytes, 
            str(wav_file.name), 
            iterations=3,
            expected_text=expected,
            warmup=True
        )
        all_results.extend(results)
        last_model_config = model_config
    
    if all_results:
        summary = calculate_summary(all_results, last_model_config if 'last_model_config' in dir() else None)
        print_results(all_results, summary)
        save_results(all_results, summary, "benchmark_accent_results.json")
    else:
        print("\n⚠️  No test files found. Add .wav files to ./benchmark_data/")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Speech-to-Speech Pipeline")
    parser.add_argument("--audio", type=str, help="Path to audio file to benchmark")
    parser.add_argument("--expected", type=str, help="Expected transcription for WER calculation")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations (default: 5)")
    parser.add_argument("--quick", action="store_true", help="Quick latency test with synthetic audio")
    parser.add_argument("--accent-test", action="store_true", help="Run accent understanding benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup run")
    
    args = parser.parse_args()
    
    print("🎙️  Speech-to-Speech Pipeline Benchmark")
    print("=" * 70)
    
    if args.accent_test:
        run_accent_benchmark()
        return
    
    # Determine audio source
    if args.audio:
        print(f"📂 Loading audio: {args.audio}")
        with open(args.audio, "rb") as f:
            audio_bytes = f.read()
        audio_name = Path(args.audio).name
    elif args.quick:
        print("🔊 Generating synthetic test audio (3 seconds)...")
        audio_bytes = generate_test_audio(duration=3.0)
        audio_name = "synthetic_test.wav"
        print("   ⚠️  Note: Synthetic audio won't produce meaningful transcriptions")
    else:
        # Try to find a sample audio
        sample_files = list(Path(".").glob("*.wav")) + list(Path("./audio").glob("*.wav"))
        if sample_files:
            audio_path = sample_files[0]
            print(f"📂 Using found audio: {audio_path}")
            audio_bytes = audio_path.read_bytes()
            audio_name = audio_path.name
        else:
            print("❌ No audio file specified. Use --audio or --quick")
            print("\nUsage examples:")
            print("  python benchmark.py --audio input.wav --iterations 10")
            print("  python benchmark.py --quick")
            print("  python benchmark.py --accent-test")
            return
    
    # Run benchmark
    results, model_config = run_benchmark(
        audio_bytes,
        audio_name,
        iterations=args.iterations,
        expected_text=args.expected,
        warmup=not args.no_warmup
    )
    
    # Calculate and print summary
    summary = calculate_summary(results, model_config)
    print_results(results, summary)
    
    # Save results
    save_results(results, summary, args.output)


if __name__ == "__main__":
    main()
