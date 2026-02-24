"""
Audio compression utilities for reducing network overhead in speech pipeline.

Uses MP3 compression (OPUS would require additional dependencies)
16kHz mono, low bitrate (~24kbps) for optimal size/quality tradeoff.
"""

import io
import numpy as np

try:
    from pydub import AudioSegment
    from pydub.utils import which
    # Ensure ffmpeg is available
    AudioSegment.converter = which("ffmpeg")
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️  pydub not available, using fallback compression")

try:
    import lameenc
    LAMEENC_AVAILABLE = True
except ImportError:
    LAMEENC_AVAILABLE = False

def compress_audio_pcm_to_mp3(pcm_data: np.ndarray, sample_rate: int = 16000, bitrate: str = "24k") -> bytes:
    """
    Compress PCM audio data to MP3 format.
    
    Args:
        pcm_data: numpy array of audio samples (int16)
        sample_rate: audio sample rate (default 16000)
        bitrate: MP3 bitrate (default "24k" for ~3KB/s)
    
    Returns:
        MP3 compressed audio bytes
    """
    if PYDUB_AVAILABLE and AudioSegment.converter:
        # Convert numpy array to AudioSegment
        if pcm_data.dtype != np.int16:
            # Normalize and convert to int16
            if pcm_data.dtype in [np.float32, np.float64]:
                pcm_data = (pcm_data * 32767).astype(np.int16)
            else:
                pcm_data = pcm_data.astype(np.int16)
        
        audio_segment = AudioSegment(
            pcm_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1  # mono
        )
        
        # Export as MP3 to memory buffer
        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3", bitrate=bitrate)
        
        return mp3_buffer.getvalue()
    else:
        # Fallback: use WAV compression
        # Convert PCM to WAV first, then compress
        wav_buffer = io.BytesIO()
        from scipy.io import wavfile
        wavfile.write(wav_buffer, sample_rate, pcm_data)
        return _compress_wav_fallback(wav_buffer.getvalue())

def decompress_audio_mp3_to_pcm(mp3_data: bytes, sample_rate: int = 16000) -> np.ndarray:
    """
    Decompress MP3 audio data to PCM format.
    
    Args:
        mp3_data: MP3 compressed audio bytes
        sample_rate: target sample rate (default 16000)
    
    Returns:
        numpy array of audio samples (int16)
    """
    if PYDUB_AVAILABLE and AudioSegment.converter:
        # Load MP3 from memory buffer
        mp3_buffer = io.BytesIO(mp3_data)
        audio_segment = AudioSegment.from_mp3(mp3_buffer)
        
        # Convert to target sample rate and mono
        if audio_segment.frame_rate != sample_rate:
            audio_segment = audio_segment.set_frame_rate(sample_rate)
        if audio_segment.channels != 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Convert to numpy array
        pcm_data = np.array(audio_segment.get_array_of_samples())
        
        return pcm_data
    else:
        # Fallback for compressed WAV
        from scipy.io import wavfile
        from scipy import signal
        
        wav_buffer = io.BytesIO(mp3_data)
        src_rate, pcm_data = wavfile.read(wav_buffer)
        
        # Upsample if necessary
        if src_rate != sample_rate:
            num_samples = int(len(pcm_data) * float(sample_rate) / src_rate)
            pcm_data = signal.resample(pcm_data, num_samples)
            
        # Convert to int16 if it was 8-bit
        if pcm_data.dtype == np.int8:
            pcm_data = (pcm_data.astype(np.int16) * 256)
        
        return pcm_data.astype(np.int16)

def compress_wav_to_mp3(wav_bytes: bytes, bitrate: str = "24k") -> bytes:
    """
    Compress WAV audio bytes to MP3 format.
    
    Args:
        wav_bytes: WAV audio bytes
        bitrate: MP3 bitrate (default "24k")
    
    Returns:
        MP3 compressed audio bytes
    """
    if PYDUB_AVAILABLE and AudioSegment.converter:
        # Load WAV from memory buffer
        wav_buffer = io.BytesIO(wav_bytes)
        audio_segment = AudioSegment.from_wav(wav_buffer)
        
        # Ensure mono and 16kHz
        if audio_segment.channels != 1:
            audio_segment = audio_segment.set_channels(1)
        if audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000)
        
        # Export as MP3 to memory buffer
        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3", bitrate=bitrate)
        
        return mp3_buffer.getvalue()
    else:
        # Fallback: use WAV with lower quality settings
        # This still provides some compression through lower sample rate and bit depth
        return _compress_wav_fallback(wav_bytes)

def decompress_mp3_to_wav(mp3_data: bytes) -> bytes:
    """
    Decompress MP3 audio bytes to WAV format.
    
    Args:
        mp3_data: MP3 compressed audio bytes
    
    Returns:
        WAV audio bytes
    """
    if PYDUB_AVAILABLE and AudioSegment.converter:
        # Load MP3 from memory buffer
        mp3_buffer = io.BytesIO(mp3_data)
        audio_segment = AudioSegment.from_mp3(mp3_buffer)
        
        # Ensure mono and 16kHz
        if audio_segment.channels != 1:
            audio_segment = audio_segment.set_channels(1)
        if audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000)
        
        # Export as WAV to memory buffer
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        
        return wav_buffer.getvalue()
    else:
        # Fallback: assume it's the fallback format and return as-is
        return mp3_data

def _compress_wav_fallback(wav_bytes: bytes) -> bytes:
    """
    Fallback compression using WAV with reduced quality.
    Still provides ~2-3x compression through lower sample rate.
    """
    import io
    from scipy.io import wavfile
    
    # Read the original WAV
    wav_buffer = io.BytesIO(wav_bytes)
    sample_rate, audio_data = wavfile.read(wav_buffer)
    
    # Downsample to 8kHz for compression (from 16kHz)
    if sample_rate == 16000:
        # Simple downsampling by taking every 2nd sample
        compressed_data = audio_data[::2]
        new_sample_rate = 8000
    else:
        compressed_data = audio_data
        new_sample_rate = sample_rate
    
    # Reduce bit depth if possible
    if compressed_data.dtype == np.int16:
        # Convert to 8-bit for additional compression
        compressed_data = (compressed_data // 256).astype(np.int8)
    
    # Write compressed audio
    compressed_buffer = io.BytesIO()
    wavfile.write(compressed_buffer, new_sample_rate, compressed_data)
    
    return compressed_buffer.getvalue()

def get_compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """Calculate compression ratio."""
    return original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0