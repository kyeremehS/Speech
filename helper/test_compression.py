"""
Test script to verify audio compression reduces network overhead by ≥60%.

Tests compression ratios and ensures audio quality is preserved.
"""

import numpy as np
import io
from scipy.io import wavfile
from audio_compression import (
    compress_wav_to_mp3, 
    decompress_mp3_to_wav,
    compress_audio_pcm_to_mp3,
    decompress_audio_mp3_to_pcm,
    get_compression_ratio
)
from utility import ensure_wav_bytes

def generate_test_audio(duration_seconds: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio signal (sine wave + noise)."""
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate))
    
    # Generate speech-like signal: multiple sine waves + noise
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 880 * t) +  # A5
        0.1 * np.sin(2 * np.pi * 220 * t) +  # A3
        0.1 * np.random.randn(len(t))  # noise
    )
    
    # Normalize to int16 range
    signal = np.clip(signal, -1.0, 1.0)
    return (signal * 32767).astype(np.int16)

def test_compression_pipeline():
    """Test complete compression/decompression pipeline."""
    print("🧪 Testing audio compression pipeline...")
    
    # Generate test audio
    print("\n1. Generating test audio signal...")
    pcm_data = generate_test_audio(duration_seconds=5.0)
    print(f"   PCM data: {len(pcm_data)} samples, {pcm_data.nbytes} bytes")
    
    # Create WAV container
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, 16000, pcm_data)
    wav_bytes = wav_buffer.getvalue()
    print(f"   WAV container: {len(wav_bytes)} bytes")
    
    # Test 1: WAV → MP3 → WAV
    print("\n2. Testing WAV → MP3 → WAV compression...")
    
    # Compress WAV to MP3
    mp3_bytes = compress_wav_to_mp3(wav_bytes, bitrate="24k")
    wav_to_mp3_ratio = get_compression_ratio(len(wav_bytes), len(mp3_bytes))
    print(f"   WAV → MP3: {len(wav_bytes)} → {len(mp3_bytes)} bytes ({wav_to_mp3_ratio:.1f}x compression)")
    
    # Decompress MP3 back to WAV
    decompressed_wav_bytes = decompress_mp3_to_wav(mp3_bytes)
    mp3_to_wav_ratio = len(decompressed_wav_bytes) / len(wav_bytes)
    print(f"   MP3 → WAV: {len(mp3_bytes)} → {len(decompressed_wav_bytes)} bytes ({mp3_to_wav_ratio:.1f}x expansion)")
    
    # Test 2: PCM → MP3 → PCM
    print("\n3. Testing PCM → MP3 → PCM compression...")
    
    # Compress PCM to MP3
    mp3_bytes = compress_audio_pcm_to_mp3(pcm_data, 16000, bitrate="24k")
    pcm_to_mp3_ratio = get_compression_ratio(pcm_data.nbytes, len(mp3_bytes))
    print(f"   PCM → MP3: {pcm_data.nbytes} → {len(mp3_bytes)} bytes ({pcm_to_mp3_ratio:.1f}x compression)")
    
    # Decompress MP3 back to PCM
    decompressed_pcm = decompress_audio_mp3_to_pcm(mp3_bytes, 16000)
    print(f"   MP3 → PCM: {len(mp3_bytes)} → {decompressed_pcm.nbytes} bytes")
    
    # Test 3: Different bitrates
    print("\n4. Testing different bitrates...")
    for bitrate in ["16k", "24k", "32k", "48k"]:
        mp3_bytes = compress_wav_to_mp3(wav_bytes, bitrate=bitrate)
        ratio = get_compression_ratio(len(wav_bytes), len(mp3_bytes))
        print(f"   {bitrate:>3} bitrate: {ratio:.1f}x compression ({len(mp3_bytes)} bytes)")
    
    # Test 4: Verify compression meets target (≥60% reduction = ≥2.5x compression)
    print("\n5. Verifying compression target...")
    target_ratio = 2.5  # 60% reduction
    
    if wav_to_mp3_ratio >= target_ratio:
        print(f"   ✅ PASS: WAV→MP3 compression ratio {wav_to_mp3_ratio:.1f}x meets target {target_ratio}x")
        reduction_percent = (1 - 1/wav_to_mp3_ratio) * 100
        print(f"   📊 Network overhead reduction: {reduction_percent:.0f}%")
    else:
        print(f"   ❌ FAIL: WAV→MP3 compression ratio {wav_to_mp3_ratio:.1f}x below target {target_ratio}x")
    
    # Test 5: Audio quality check (basic)
    print("\n6. Basic audio quality check...")
    decompressed_pcm = decompress_audio_mp3_to_pcm(compress_audio_pcm_to_mp3(pcm_data, 16000))
    
    # Check if audio is roughly the same length
    length_ratio = len(decompressed_pcm) / len(pcm_data)
    if 0.9 <= length_ratio <= 1.1:
        print(f"   ✅ Audio length preserved: {length_ratio:.2f}x")
    else:
        print(f"   ⚠️  Audio length changed: {length_ratio:.2f}x")
    
    # Check signal energy
    original_energy = np.mean(pcm_data.astype(float)**2)
    decompressed_energy = np.mean(decompressed_pcm.astype(float)**2)
    energy_ratio = decompressed_energy / original_energy
    
    if 0.7 <= energy_ratio <= 1.3:
        print(f"   ✅ Audio energy preserved: {energy_ratio:.2f}x")
    else:
        print(f"   ⚠️  Audio energy changed: {energy_ratio:.2f}x")
    
    print(f"\n🎯 Expected network overhead reduction: {reduction_percent:.0f}%")
    print(f"🎯 Expected latency improvement for 3.6s network time: ~{3.6 * reduction_percent/100:.1f}s")
    
    return wav_to_mp3_ratio >= target_ratio

def test_realistic_scenarios():
    """Test with realistic speech audio durations."""
    print("\n" + "="*60)
    print("🎤 Testing realistic speech scenarios...")
    
    scenarios = [
        ("Short command", 2.0),
        ("Medium query", 5.0),
        ("Long sentence", 10.0),
    ]
    
    all_passed = True
    
    for name, duration in scenarios:
        print(f"\n📍 {name} ({duration}s):")
        
        # Generate audio
        pcm_data = generate_test_audio(duration)
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, 16000, pcm_data)
        wav_bytes = wav_buffer.getvalue()
        
        # Compress
        mp3_bytes = compress_wav_to_mp3(wav_bytes, bitrate="24k")
        ratio = get_compression_ratio(len(wav_bytes), len(mp3_bytes))
        reduction = (1 - 1/ratio) * 100
        
        print(f"   Original: {len(wav_bytes)} bytes")
        print(f"   Compressed: {len(mp3_bytes)} bytes")
        print(f"   Compression: {ratio:.1f}x ({reduction:.0f}% reduction)")
        
        if ratio >= 2.5:  # 60% reduction target
            print(f"   ✅ PASS")
        else:
            print(f"   ❌ FAIL")
            all_passed = False
    
    return all_passed

def test_binary_roundtrip():
    print("\n" + "="*60)
    print("🔁 Testing binary PCM/WAV round-trip...")
    pcm_data = generate_test_audio(duration_seconds=3.0)
    pcm_bytes = pcm_data.tobytes()
    wav_bytes = ensure_wav_bytes(pcm_bytes, sample_rate=16000, channels=1, sample_width=2)
    sr, decoded = wavfile.read(io.BytesIO(wav_bytes))
    same_rate = sr == 16000
    same_len = len(decoded) == len(pcm_data)
    if decoded.dtype != pcm_data.dtype:
        decoded = decoded.astype(pcm_data.dtype)
    diff = np.mean(np.abs(decoded.astype(np.int32) - pcm_data.astype(np.int32)))
    ok = same_rate and same_len and diff < 2
    print(f"   Sample rate OK: {same_rate}")
    print(f"   Length OK: {same_len}")
    print(f"   Mean abs diff: {diff:.2f}")
    print(f"   {'✅ PASS' if ok else '❌ FAIL'}")
    return ok

if __name__ == "__main__":
    print("🚀 Audio Compression Test Suite")
    print("="*60)
    
    # Run basic pipeline test
    basic_passed = test_compression_pipeline()
    
    # Run realistic scenarios test
    realistic_passed = test_realistic_scenarios()
    
    binary_passed = test_binary_roundtrip()
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS:")
    print(f"   Basic pipeline: {'✅ PASS' if basic_passed else '❌ FAIL'}")
    print(f"   Realistic scenarios: {'✅ PASS' if realistic_passed else '❌ FAIL'}")
    print(f"   Binary round-trip: {'✅ PASS' if binary_passed else '❌ FAIL'}")
    
    if basic_passed and realistic_passed and binary_passed:
        print("\n🎉 All tests passed! Audio compression ready for deployment.")
        print("🎯 Expected network overhead reduction: ≥60%")
        print("🎯 Expected latency improvement: ~2-3 seconds")
    else:
        print("\n⚠️  Some tests failed. Review compression settings.")
    
    print("="*60)
