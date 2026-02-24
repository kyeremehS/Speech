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
    
    header = audio_bytes[:12]
    
    try:
        if header[:3] == b'ID3' or header[:2] in (b'\xff\xfb', b'\xff\xfa', b'\xff\xf3'):
            audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        elif header[:4] == b'OggS':
            audio = AudioSegment.from_ogg(io.BytesIO(audio_bytes))
        elif header[:4] == b'fLaC':
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="flac")
        elif b'ftyp' in header[:12]:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="m4a")
        elif header[:4] == b'RIFF':
            return audio_bytes
        else:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()
