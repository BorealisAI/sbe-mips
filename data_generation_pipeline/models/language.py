# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, List
try:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokenInputs
except ImportError:
    vllm = None
import numpy as np
import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    BitsAndBytesConfig = None  # type: ignore[assignment]


@dataclass
class GenerationConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    k: int
    seed: int


class BaseTextPolicy:
    model_id: str

    def generate_one(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        raise NotImplementedError

    def logprob(self, prompt: str, response: str) -> tuple[float, int]:
        raise NotImplementedError

    def generate_batch(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        raise NotImplementedError

    def logprob_batch(self, prompt: str, response: str) -> tuple[float, int]:
        raise NotImplementedError


class DummyTextPolicy(BaseTextPolicy):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def _render(self, prompt: str) -> str:
        key = abs(hash((self.model_id, prompt))) % 7
        variants = [
            "I would start by identifying the core objective, then outline 3 practical steps.",
            "A concise plan: define success, collect constraints, and execute in small iterations.",
            "Here is a structured answer: summary, rationale, and concrete next actions.",
            "I recommend balancing safety, usefulness, and clarity in this response.",
            "You can approach this with a checklist: scope, assumptions, and verification.",
            "A practical approach is to break this into diagnosis, options, and recommendation.",
            "The best response here is short, specific, and aligned with the user intent.",
        ]
        return variants[key]

    def generate_one(self, prompt: str, gen_cfg: GenerationConfig) -> str:  # noqa: ARG002
        return self._render(prompt)

    def logprob(self, prompt: str, response: str) -> tuple[float, int]:
        n_tokens = max(1, len(response.split()))
        if response == self._render(prompt):
            return -0.25 * n_tokens, n_tokens
        drift = abs(hash((self.model_id, prompt, response))) % 1000
        return -1.5 * n_tokens - drift / 3000.0, n_tokens
    
    def generate_batch(self, prompts, gen_cfg):
        return [self.generate_one(prompt, gen_cfg) for prompt in prompts]
    
    def logprob_batch(self, prompts, responses):
        return [self.logprob(prompt, response) for prompt, response in zip(prompts, responses)]


class HFTextPolicy(BaseTextPolicy):
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        torch_dtype: str = "float32",
        use_chat_template: bool = True,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.use_chat_template = use_chat_template

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_cfg = None
        if load_in_4bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError(
                    "load_in_4bit requested but BitsAndBytesConfig is unavailable in this transformers version."
                )
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "auto" else None,
            quantization_config=quant_cfg,
        )
        if device != "auto":
            self.model.to(device)
        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            chat = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    @torch.no_grad()
    def generate_one(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        formatted_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        torch.manual_seed(gen_cfg.seed)
        output = self.model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.temperature > 0,
            temperature=max(gen_cfg.temperature, 1e-5),
            top_p=gen_cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        response_ids = output[0, prompt_len:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    @torch.no_grad()
    def logprob(self, prompt: str, response: str) -> tuple[float, int]:
        formatted_prompt = self._format_prompt(prompt)
        prompt_ids = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        full_ids = self.tokenizer(
            formatted_prompt + response,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        prompt_len = prompt_ids.shape[1]
        total_len = full_ids.shape[1]
        if total_len <= 1 or total_len <= prompt_len:
            return 0.0, 0

        input_ids = full_ids[:, :-1].to(self.model.device)
        target_ids = full_ids[:, 1:].to(self.model.device)

        logits = self.model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        start = max(prompt_len - 1, 0)
        end = total_len - 1
        response_log_probs = token_log_probs[:, start:end]
        n_tokens = int(response_log_probs.shape[1])
        return float(response_log_probs.sum().item()), n_tokens
    
    @torch.no_grad()
    def generate_batch(self, prompts: List[str], gen_cfg: GenerationConfig):
        responses = []
        for i, prompt in enumerate(prompts):
            for k in range(gen_cfg.k):
                response = self.generate_one(prompt, replace(gen_cfg, seed=gen_cfg.seed + i * gen_cfg.k + k))
                responses.append(response)
        
        return responses

    @torch.no_grad()
    def logprob_batch(self, prompts: List[str], responses: List[str]) -> List[tuple[float, int]]:
        results = []
        for prompt, response in zip(prompts, responses):
            score, n_tokens = self.logprob(prompt, response)
            results.append((score, n_tokens))
        
        return results

        
        

class VLLMTextPolicy(BaseTextPolicy):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: str = "auto",
        use_chat_template: bool = True,
        load_in_4bit: bool = False,
        tensor_parallel_size: int = 1,  # multi-GPU
        gpu_memory_utilization: float = 0.9,  # GPU memory portion to use
    ) -> None:

        self.model_id = model_id
        self.device = device
        self.use_chat_template = use_chat_template

        dtype_map = {
            "float16": "float16",
            "bfloat16": "bfloat16",
            "float32": "float32",
            "auto": "auto",
        }
        dtype = dtype_map.get(torch_dtype, "auto")

        llm_kwargs = {
            "model": model_id,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,  # Often needed for custom models
            "max_model_len": 16384, 
            "max_num_batched_tokens": 4096,
        }

        if load_in_4bit:
            # vLLM supports AWQ and GPTQ quantization
            # For 4-bit, you typically need a pre-quantized model
            llm_kwargs["quantization"] = "awq"  # or gptq depending on the model

        if vllm is None:
            raise ImportError("Install vllm or set runtime.text_framework=hf")
        # initialize vllm
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _format_prompt(self, prompt: str) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            chat = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _calculate_logprobs_from_vllm_output(
        self,
        vllm_output: vllm.RequestOutput,
        response_start: int,
    ) -> tuple[float, int]:
        # Use the exact tokenization vLLM actually scored
        full_ids = vllm_output.prompt_token_ids
        prompt_logprobs = vllm_output.prompt_logprobs

        total_logprob = 0.0
        n_tokens = 0

        for tok_id, lp_dict in zip(full_ids[response_start:], prompt_logprobs[response_start:]):
            if lp_dict is None:
                continue
            if tok_id not in lp_dict:
                continue

            entry = lp_dict[tok_id]
            logprob = entry if isinstance(entry, float) else entry.logprob
            total_logprob += logprob
            n_tokens += 1

        return total_logprob, n_tokens

    def generate_one(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        formatted_prompt = self._format_prompt(prompt)
        sampling_params = SamplingParams(
            max_tokens=gen_cfg.max_new_tokens,
            temperature=(
                max(gen_cfg.temperature, 1e-5)
                if gen_cfg.temperature > 0
                else 0.0  # similar to do_sample in HF
            ),
            top_p=gen_cfg.top_p,
            seed=gen_cfg.seed,
            stop_token_ids=[self.tokenizer.eos_token_id],
        )

        outputs = self.llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

    def generate_batch(self, prompts: List[str], gen_cfg: GenerationConfig) -> List[str]:
        """Generate responses for a batch of prompts using vLLM.
        This method formats prompts, sets up sampling parameters, and calls vLLM's generate function.
        It then extracts and returns the generated texts. Based on the values of k in gen_cfg, it can generate multiple responses per prompt.
        Responses are returned in the same order as the input prompts, with k responses flattened into a single list.

        args:
            prompts: List of input prompt strings.
            gen_cfg: GenerationConfig object containing generation parameters like temperature, top_p, max_new_tokens, k, and seed.
        returns:
            List of generated response strings, with k responses per prompt flattened into a single list.
        """
        formatted_prompts = []
        for prompt in prompts:
            formatted_prompts.append(self._format_prompt(prompt))

        sampling_params = SamplingParams(
            max_tokens=gen_cfg.max_new_tokens,
            temperature=(
                max(gen_cfg.temperature, 1e-5)
                if gen_cfg.temperature > 0
                else 0.0  # similar to do_sample in HF
            ),
            top_p=gen_cfg.top_p,
            seed=gen_cfg.seed,
            stop_token_ids=[self.tokenizer.eos_token_id],
            n=gen_cfg.k,  # generate k responses per prompt
        )

        generations = self.llm.generate(formatted_prompts, sampling_params, use_tqdm=False)
        generations = [[o.outputs[i].text for i in range(len(o.outputs))] for o in generations]

        outputs = []
        for gen_list in generations:
            outputs.extend(gen_list)  # flatten the list of lists
        return outputs

    def logprob(self, prompt: str, response: str) -> tuple[float, int]:
        formatted_prompt = self._format_prompt(prompt)

        # Tokenize prompt and response separately, then concatenate token ids.
        # This is the safer pattern to ensure we know exactly where the response tokens start in the full token sequence.
        prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        full_ids = prompt_ids + response_ids
        response_start = len(prompt_ids)

        sampling_params = SamplingParams(
            prompt_logprobs=1,
            max_tokens=1,
            temperature=0.0,
            detokenize=False,
        )

        outputs = self.llm.generate(
            [TokenInputs(prompt_token_ids=full_ids)],
            sampling_params,
            use_tqdm=False,
        )

        return self._calculate_logprobs_from_vllm_output(outputs[0], response_start)

    def logprob_batch(self, prompts: List[str], responses: List[str]) -> List[tuple[float, int]]:
        formatted_prompts = []
        token_prompts = []
        response_starts = []

        for prompt, response in zip(prompts, responses):
            formatted_prompt = self._format_prompt(prompt)
            formatted_prompts.append(formatted_prompt)

            prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)

            full_ids = prompt_ids + response_ids
            response_starts.append(len(prompt_ids))
            token_prompts.append(TokenInputs(prompt_token_ids=full_ids))

        sampling_params = SamplingParams(
            prompt_logprobs=1,
            max_tokens=1,
            temperature=0.0,
            detokenize=False,
        )

        outputs = self.llm.generate(token_prompts, sampling_params, use_tqdm=False)

        logprobs = []
        for out, response_start in zip(outputs, response_starts):
            logprobs.append(
                self._calculate_logprobs_from_vllm_output(out, response_start)
            )

        return logprobs

class PolicyFactory:
    @staticmethod
    def build(model_id: str, config: dict[str, Any]) -> BaseTextPolicy:
        runtime_cfg = config.get("runtime", {})
        model_cfg = config.get("models", {})
        offline_mode = bool(runtime_cfg.get("offline_mode", False))
        text_framework = runtime_cfg.get("text_framework", "hf")
        if offline_mode or model_id.startswith("dummy/"):
            return DummyTextPolicy(model_id)
        if text_framework == 'hf':
            return HFTextPolicy(
                model_id=model_id,
                device=str(runtime_cfg.get("device", "cuda")),
                torch_dtype=str(runtime_cfg.get("dtype", "float32")),
                use_chat_template=bool(model_cfg.get("use_chat_template", True)),
                load_in_4bit=bool(runtime_cfg.get("load_in_4bit", False)),
            )
        elif "vllm" == text_framework:
            return VLLMTextPolicy(
                model_id=model_id,
                device=str(runtime_cfg.get("device", "cuda")),
                torch_dtype=str(runtime_cfg.get("dtype", "float32")),
                use_chat_template=bool(model_cfg.get("use_chat_template", True)),
                load_in_4bit=bool(runtime_cfg.get("load_in_4bit", False)),
                tensor_parallel_size=int(runtime_cfg.get("vllm_tensor_parallel_size", 1)),
                gpu_memory_utilization=float(runtime_cfg.get("vllm_gpu_memory_utilization", 0.9))
            )
