# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import json
import os
import gc
import time
import uuid
from typing import Any, Dict, Optional
from copy import deepcopy
import logging
import threading

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer

from sllm.serve.backends.backend_utils import SllmBackend, BackendStatus
from sllm.serve.logger import init_logger
from sllm_store.transformers import load_model

# logger = init_logger(__name__)
logger = logging.getLogger("ray")

class DeletingException(Exception):
    pass

class InferenceStatus(BaseStreamer):
    def __init__(self, status: BackendStatus):
        super().__init__()
        self.status = status
        self.intermediate = []

    def put(self, value):
        value = value.flatten().tolist()
        self.intermediate.extend(value)
        logger.warn(f"Intermediate output: {self.intermediate}")
        if self.status == BackendStatus.DELETING:
            raise DeletingException("Backend is deleting")

    def end(self):
        logger.error("Inference completed")

    def get(self):
        return deepcopy(self.intermediate)
    
    def delete(self):
        logger.info("Deleting intermediate output")
        self.intermediate = []


class TransformersBackend(SllmBackend):
    def __init__(self, backend_config: Optional[Dict[str, Any]] = None) -> None:
        self.backend_config = backend_config
        logger.info(
            f"Initializing TransformersBackend with config: {backend_config}"
        )
        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.inf_status = InferenceStatus(self.status)
        self.status_lock = threading.Lock()
        self.model_name = backend_config.get("pretrained_model_name_or_path")
        self.model = None
        self.tokenizer = None

    def convert_str_to_json(self, json_str):
        try:
            # Parse the JSON string and return the corresponding Python object
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON string: {e}")
            return None

    def init_backend(self) -> None:
        with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            device_map = self.backend_config.get("device_map", "auto")
            torch_dtype = self.backend_config.get("torch_dtype", torch.float16)
            torch_dtype = getattr(torch, torch_dtype)
            hf_model_class = self.backend_config.get("hf_model_class", None)
            if torch_dtype is None:
                logger.warning(
                    f"Invalid torch_dtype: {torch_dtype}. Using torch.float16"
                )
                torch_dtype = torch.float16
            if hf_model_class is None:
                logger.error(
                    f"hf_model_class cannot be None. Please provide a valid model class"
                )
                raise ValueError(
                    "hf_model_class cannot be None. Please provide a valid model class"
                )

            storage_path = os.getenv("STORAGE_PATH", "./models")
            model_path = os.path.join("transformers", self.model_name)
            self.model = load_model(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                storage_path=storage_path,
                hf_model_class=hf_model_class,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.status = BackendStatus.RUNNING

    def _tokenize(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to("cuda:0")

    def _encoder_tokenize(self, query: str, max_length: int):
        return self.tokenizer(
            query,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda:0")

    def encode(self, request_data: Optional[Dict[str, Any]]):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        def last_token_pool(
            last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            left_padding = (
                attention_mask[:, -1].sum() == attention_mask.shape[0]
            )
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    sequence_lengths,
                ]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f"Instruct: {task_description}\nQuery: {query}"

        model_name = request_data.get("model", "dummy-model")
        task_instruct = request_data.get("task_instruct", "")
        max_length = request_data.get("max_length", 4096)
        query = request_data.get("query", [])

        if not query:
            return {"error": "Missing query in request data"}

        query = [get_detailed_instruct(task_instruct, q) for q in query]

        batch_dict = self._encoder_tokenize(query, max_length)
        with torch.no_grad():
            output = self.model(**batch_dict, output_hidden_states=True)
        embeddings = last_token_pool(
            output.hidden_states[-1], batch_dict["attention_mask"]
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        query_tokens = sum([len(self.tokenizer.tokenize(q)) for q in query])
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embeddings[i].tolist(),
                }
                for i in range(len(embeddings))
            ],
            "model": model_name,
            "usage": {
                "query_tokens": query_tokens,
                "total_tokens": query_tokens,
            },
        }

        return response

    def generate(self, request_data: Optional[Dict[str, Any]]):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        assert self.model is not None

        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)

        # Combine messages to form the prompt
        prompt = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        if not prompt:
            return {"error": "Missing prompt in request data"}

        inputs = self._tokenize(prompt)

        # Generate response
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, temperature=temperature,
                    streamer=self.inf_status
                )
        except DeletingException:
            logger.info("Backend is shutting down. Aborting request")
            output_text = "Backend is shutting down. Please try again later."
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            output_text = "Failed to generate response. Please try again later."
        else:
            output_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

        # Simulate token counts for the response
        prompt_tokens = len(self.tokenizer.tokenize(prompt))
        completion_tokens = len(self.tokenizer.tokenize(output_text))
        total_tokens = prompt_tokens + completion_tokens

        # Generate response compatible with OpenAI's API
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

        self.inf_status.delete()

        return response

    def shutdown(self):
        """Abort all requests and shutdown the backend."""
        with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING
            if self.inf_status:
                self.inf_status.status = BackendStatus.DELETING

        while self.inf_status and len(self.inf_status.get()) > 0:
            logger.info("Waiting for all requests to finish")
            time.sleep(1)

        if self.model is not None:
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while self.inf_status and len(self.inf_status.get()) > 0:
            logger.info("Waiting for all requests to finish")
            time.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        self.shutdown()

    def get_current_tokens(self):
        """Return a list of all ongoing request tokens."""
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
        # TODO: debug this code
        status = self.inf_status.get()
        logger.fatal(f"Current tokens: {status}")
        return status

    def resume_kv_cache(self, request_datas):
        logger.error("Not implemented")
        raise NotImplementedError
