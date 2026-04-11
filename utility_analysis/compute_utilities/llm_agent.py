import asyncio
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


class LLMAgent(ABC):

    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, retry_times: int = 3, accepts_system_message: bool = True):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_outputs = "Sorry, I can not satisfy that request."
        self.retry_times = retry_times
        self.accepts_system_message = accepts_system_message

    @abstractmethod
    def _completions(self, messages) -> str:
        raise NotImplementedError

    def _completions_batch(self, messages) -> List[str]:
        raise NotImplementedError

    async def _async_completions(self, messages) -> str:
        raise NotImplementedError

    async def _completions_stream(self, messages: List[Dict]) -> str:
        raise NotImplementedError

    async def completions_stream(self, messages: List[Dict]) -> str:
        try:
            response = self._completions_stream(messages)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}") from e

    def completions(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = self._completions(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}") from e

    def completions_batch(self, messages: List[Dict], **kwargs) -> List[str]:
        try:
            response = self._completions_batch(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}") from e

    async def async_completions(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = await self._async_completions(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}") from e


class vLLMAgent(LLMAgent):

    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/public_models', trust_remote_code=False, accepts_system_message=True, tokenizer_path=None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )

        additional_kwargs = {}
        if "deepseek" in model.lower():
            additional_kwargs["max_model_len"] = 8192
            additional_kwargs["dtype"] = "float16"
            additional_kwargs["enforce_eager"] = True

        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_path if tokenizer_path is not None else model,
            trust_remote_code=trust_remote_code,
            download_dir=cache_dir,
            tensor_parallel_size=torch.cuda.device_count(),
            **additional_kwargs
        )

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _completions(self, messages: Union[List[Dict], List[List[Dict]]], batch_size: int = 1) -> Union[str, List[str]]:
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        prompts = []
        for message_set in messages_list:
            prompt = self._messages_to_prompt(message_set)
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        result_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text
            result_texts.append(generated_text.strip())

        return result_texts[0] if len(result_texts) == 1 else result_texts

    def _completions_batch(self, messages_list: List[List[Dict]], batch_size: int = 1) -> List[str]:
        return self._completions(messages_list, batch_size)

    async def _completions_stream(self, messages: List[Dict]):
        prompt = self._messages_to_prompt(messages)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs_generator = self.llm.generate([prompt], sampling_params, stream=True)

        for request_output in outputs_generator:
            for token_output in request_output.outputs:
                for token in token_output.tokens:
                    yield token.text

    async def _async_completions(self, messages: List[Dict]) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._completions, messages)
        return result


FEW_SHOT_PROMPT = ""


class vLLMAgentBaseModel(LLMAgent):

    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/public_models', trust_remote_code=False, accepts_system_message=False, tokenizer_path=None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )

        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_path if tokenizer_path is not None else model,
            trust_remote_code=trust_remote_code,
            download_dir=cache_dir,
            tensor_parallel_size=torch.cuda.device_count()
        )

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _format_messages(self, messages: Union[List[Dict], List[List[Dict]]]) -> List[str]:
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        formatted_messages = []
        for msg_list in messages_list:
            user_part = "".join(
                f"{msg['content']}\n\nAnswer:"
                for msg in msg_list
                if msg['role'] == 'user'
            )
            final_prompt = f"{FEW_SHOT_PROMPT}{user_part}"
            formatted_messages.append(final_prompt)

        return formatted_messages

    def _completions(self, messages: Union[List[Dict], List[List[Dict]]], batch_size: int = 1) -> Union[str, List[str]]:
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        prompts = self._format_messages(messages_list)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        result_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text
            result_texts.append(generated_text.strip())

        return result_texts[0] if len(result_texts) == 1 else result_texts

    def _completions_batch(self, messages_list: List[List[Dict]], batch_size: int = 1) -> List[str]:
        return self._completions(messages_list, batch_size)

    async def _completions_stream(self, messages: List[Dict]):
        prompt = self._format_messages([messages])[0]

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs_generator = self.llm.generate([prompt], sampling_params, stream=True)

        for request_output in outputs_generator:
            for token_output in request_output.outputs:
                for token in token_output.tokens:
                    yield token.text

    async def _async_completions(self, messages: List[Dict]) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._completions, messages)
        return result


class HuggingFaceAgentLogitsPrediction(LLMAgent):

    def __init__(
        self,
        model="meta-llama/Llama-2-7b-chat-hf",
        max_tokens=2048,
        temperature=0.0,
        cache_dir='/data/public_models',
        trust_remote_code=False,
        accepts_system_message=False
    ):
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model = model
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.accepts_system_message = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
            ).to(self.device)

        self.llm.eval()

        self.completions_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def update_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.completions_kwargs["max_tokens"] = max_tokens

    def _format_messages(self, messages: Union[List[Dict], List[List[Dict]]]) -> List[str]:
        if isinstance(messages[0], dict):
            messages_list = [messages]
        else:
            messages_list = messages

        formatted_messages = []
        for msg_list in messages_list:
            user_part = "".join(
                f"{msg['content']}\n\nAnswer:"
                for msg in msg_list
                if msg['role'] == 'user'
            )
            final_prompt = f"{FEW_SHOT_PROMPT}{user_part}"
            formatted_messages.append(final_prompt)

        return formatted_messages

    def _completions(
        self,
        messages: Union[List[Dict], List[List[Dict]]],
        batch_size: int = 1,
        options: List[str] = None
    ) -> List[Dict[str, float]]:
        if options is None:
            options = ['A', 'B']
        formatted_messages = self._format_messages(messages)
        results = []

        option_tokens = [" " + opt for opt in options]
        option_ids = self.tokenizer.encode(option_tokens, add_special_tokens=False)

        for i in range(0, len(formatted_messages), batch_size):
            batch_messages = formatted_messages[i:i + batch_size]

            inputs = self.tokenizer(
                batch_messages,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                truncation=False
            ).to(self.device)

            with torch.no_grad():
                outputs = self.llm(**inputs)
                logits = outputs.logits[:, -1, :]

            for logits_seq in logits:
                masked_logits = torch.full_like(logits_seq, -1e4)
                for option_id in option_ids:
                    if option_id < logits_seq.shape[0]:
                        masked_logits[option_id] = logits_seq[option_id]

                full_dist = F.softmax(masked_logits, dim=0)
                subset_dist = full_dist[option_ids]
                subset_dist = subset_dist / subset_dist.sum()

                distribution_dict = {
                    options[idx]: float(subset_dist[idx]) for idx in range(len(options))
                }
                results.append(distribution_dict)

        return results


class HuggingFaceAgent(LLMAgent):
    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", max_tokens=2048, temperature=0.0, cache_dir='/data/public_models', trust_remote_code=False, batch_size=512, accepts_system_message=True, tokenizer_path=None):
        super().__init__(temperature=temperature, max_tokens=max_tokens, accepts_system_message=accepts_system_message)
        self.model_name = model
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path is not None else model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _completions(self, messages: List[Dict]) -> str:
        return self._completions_batch([messages])[0]

    def _completions_batch(self, messages_list: List[List[Dict]], **kwargs) -> List[str]:
        from accelerate.utils import find_executable_batch_size

        prompts = [self._messages_to_prompt(messages) for messages in messages_list]
        all_outputs = []

        @find_executable_batch_size(starting_batch_size=self.batch_size)
        def _process_batch(batch_size):
            nonlocal all_outputs
            print(f"\nProcessing with batch size: {batch_size}", flush=True)

            for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
                batch_prompts = prompts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=2048
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=self.max_tokens,
                        do_sample=self.temperature > 0,
                        temperature=self.temperature if self.temperature > 0 else 1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                for j, output in enumerate(outputs):
                    prompt_length = len(inputs["input_ids"][j])
                    decoded = self.tokenizer.decode(
                        output[prompt_length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    all_outputs.append(decoded.strip())

        _process_batch()
        return all_outputs

    async def _async_completions(self, messages: List[Dict]) -> str:
        return self._completions(messages)

    async def _completions_stream(self, messages: List[Dict]) -> str:
        raise NotImplementedError("Streaming not implemented for HuggingFace models")
