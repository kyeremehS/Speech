from typing import Optional, Tuple
from models_base import LLMModel, register_model, VOICE_ASSISTANT_SYSTEM_PROMPT

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
        
        streamer = self._TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

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

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        total_chars = 0
        for token in streamer:
            clean_token = re.sub(r'<\|[^|]*\|>', '', token)
            if clean_token:
                total_chars += len(clean_token)
                if total_chars <= 300:
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

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

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
                stream=True,
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
        api_key = (
            os.environ.get("GROQ_API_KEY") or
            os.environ.get("GROQ_KEY") or
            os.environ.get("groq_api_key") or
            os.environ.get("api_keys") or
            os.environ.get("groq-secret")
        )
        if not api_key:
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
                stream=True,
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
            enable_thinking=False
        )

        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

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

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

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
