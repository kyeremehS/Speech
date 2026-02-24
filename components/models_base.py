import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class ModelConfig:
    """Configuration for model selection"""
    def __init__(self):
        self.asr = os.getenv("ASR_MODEL", "nemo")
        self.llm = os.getenv("LLM_MODEL", "phi3")
        self.tts = os.getenv("TTS_MODEL", "parler")

    def __str__(self):
        return f"ASR={self.asr}, LLM={self.llm}, TTS={self.tts}"

MODEL_REGISTRY = {
    "asr": {},
    "llm": {},
    "tts": {}
}

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
        response, _ = self.generate(user_input, system_prompt)
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
        self.clause_breaks = ",:;—"
    
    def add_token(self, token: str) -> Optional[str]:
        """
        Add a token and return a complete chunk if available.
        OPTIMIZED: Scans from end for O(1) common case.
        Returns None if still buffering.
        """
        self.buffer += token
        buf_len = len(self.buffer)
        
        if buf_len < self.min_chars:
            return None
        
        for i in range(buf_len - 1, self.min_chars - 2, -1):
            if self.buffer[i] in self.sentence_endings:
                sentence = self.buffer[:i+1].strip()
                self.buffer = self.buffer[i+1:].lstrip()
                return sentence
        
        if buf_len >= self.min_chars + 5:
            for i in range(buf_len - 1, self.min_chars - 2, -1):
                if self.buffer[i] in self.clause_breaks:
                    sentence = self.buffer[:i+1].strip()
                    self.buffer = self.buffer[i+1:].lstrip()
                    return sentence
        
        if buf_len >= self.max_chars:
            last_space = self.buffer.rfind(" ", self.min_chars, self.max_chars)
            if last_space > 0:
                sentence = self.buffer[:last_space].strip()
                self.buffer = self.buffer[last_space:].lstrip()
                return sentence
            else:
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

    def synthesize_stream(self, text: str):
        """
        Optional: yield raw audio bytes incrementally.
        
        Override this for sub-250ms TTFA (e.g. Inworld streaming API).
        Each yielded item: (pcm_chunk: bytes, sample_rate: int, is_last: bool)
        
        Default: falls back to synthesize() — one big chunk.
        """
        audio_bytes, duration, proc_time = self.synthesize(text)
        yield (audio_bytes, 24000, True)

    @property
    def supports_streaming(self) -> bool:
        """Return True if this model implements true streaming synthesis."""
        return False

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
