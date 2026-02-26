from typing import Optional, Tuple
from components.models_base import LLMModel, register_model, VOICE_ASSISTANT_SYSTEM_PROMPT

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
                temperature=0.6,
                stream=True,
                stop=["\n\n", "User:", "Human:"],
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
            temperature=0.6,
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
