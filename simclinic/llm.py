"""Chat backends for the 7-8B instruction-tuned baselines. Both are callables: messages -> reply text.

* OllamaChat: local quantised models through the Ollama server (`ollama serve`, `ollama pull <tag>`).
* HFChat: Hugging Face checkpoints (gated: accept the licence and `huggingface-cli login` first), optionally in
  4-bit with bitsandbytes so an 8B model fits on a 12-16 GB GPU.
Weights are never modified; these models are only prompted.
"""
import json
import time
import urllib.error
import urllib.request

MODELS = {
    "llama3-8b": {"ollama": "llama3:8b-instruct-q4_K_M", "hf": "meta-llama/Meta-Llama-3-8B-Instruct"},
    "mistral-7b": {"ollama": "mistral:7b-instruct-v0.2-q4_K_M", "hf": "mistralai/Mistral-7B-Instruct-v0.2"},
}


class OllamaChat:
    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.0,
                 max_tokens: int = 400, seed: int = 0, num_ctx: int = 8192, timeout: int = 600):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        # num_ctx is set explicitly: Ollama's default window can be smaller than a few-shot prompt plus a
        # 10-question dialogue, and it truncates silently.
        self.options = {"temperature": temperature, "num_predict": max_tokens, "seed": seed, "num_ctx": num_ctx}

    def __call__(self, messages: list, max_tokens: int | None = None, schema: dict | None = None,
                 **options) -> str:
        """`schema` constrains the reply to a JSON schema; extra keyword arguments override sampling options
        (e.g. temperature, seed) for this call."""
        options = self.options | ({"num_predict": max_tokens} if max_tokens else {}) | options
        payload = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if schema:
            payload["format"] = schema
        body = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())["message"]["content"]
            except urllib.error.HTTPError as e:
                # A 5xx is usually transient (e.g. the GPU briefly out of memory while another job loads a
                # model); one failure should not end a multi-hour run.
                if e.code < 500 or attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))

    def unload(self):
        """Drop the model from memory now; the server otherwise keeps it on the GPU for minutes after the
        last request. Does nothing if it is not loaded."""
        body = json.dumps({"model": self.model, "keep_alive": 0}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=self.timeout).read()


class HFChat:
    def __init__(self, model_id: str, load_in_4bit: bool = True, max_new_tokens: int = 400,
                 temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["dtype"] = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    def _template(self, messages):
        try:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                      return_dict=True)
        except Exception:
            # Mistral-7B-Instruct-v0.2 has no system role: fold the system prompt into the first user turn.
            system, rest = messages[0]["content"], messages[1:]
            rest = [{**rest[0], "content": f"{system}\n\n{rest[0]['content']}"}] + rest[1:]
            return self.tokenizer.apply_chat_template(rest, add_generation_prompt=True, return_tensors="pt",
                                                      return_dict=True)

    def __call__(self, messages: list, max_tokens: int | None = None) -> str:
        import torch

        enc = self._template(messages).to(self.model.device)
        sample = self.temperature > 0
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_tokens or self.max_new_tokens, do_sample=sample,
                                      temperature=self.temperature if sample else None,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
