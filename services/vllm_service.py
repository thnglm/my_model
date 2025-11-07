# import os
# import json
# from uuid import uuid4
# from typing_extensions import AsyncGenerator
# from vllm import SamplingParams
# from vllm.engine.arg_utils import AsyncEngineArgs
# from vllm.engine.async_llm_engine import AsyncLLMEngine
# from vllm.sampling_params import RequestOutputKind
# from vllm.outputs import CompletionOutput
# from vllm.config.compilation import CompilationConfig


# os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/runpod-volume/torch_cache")
# os.environ.setdefault("TRITON_CACHE_DIR", "/runpod-volume/triton_cache")
# os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


# class vLLMService:

#     @staticmethod
#     async def init_resource() -> AsyncLLMEngine:
#         cache_dir = os.getenv("VLLM_TORCH_COMPILE_DIR", "/cache/vllm_compile")
#         use_eager = os.getenv("VLLM_EAGER", "0") == "1"
#         if not use_eager:
#             comp_cfg = CompilationConfig(
#                 level=3,
#                 cache_dir="/runpod-volume/vllm_cache"
#             )
#             args = AsyncEngineArgs(
#                 model="/models_dir/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
#                 tokenizer="/models_dir/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
#                 trust_remote_code=True,
#                 # quantization='gptq',
#                 # dtype="float16",
#                 max_model_len=4096,
#                 gpu_memory_utilization=0.9,
#                 # kv_cache_dtype="fp8",
#                 max_num_batched_tokens=4096,
#                 # enable_chunked_prefill=True,
#                 # max_num_partial_prefills=4,
#                 enable_prefix_caching=True,
#                 swap_space=4,
#                 enforce_eager=use_eager,
#                 compilation_config=comp_cfg,
#                 max_seq_len_to_capture=4096
#             )
#         else:
#             args = AsyncEngineArgs(
#                 model="/models_dir/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
#                 tokenizer="/models_dir/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
#                 trust_remote_code=True,
#                 # quantization='gptq',
#                 # dtype="float16",
#                 max_model_len=4096,
#                 gpu_memory_utilization=0.9,
#                 # kv_cache_dtype="fp8",
#                 max_num_batched_tokens=4096,
#                 # enable_chunked_prefill=True,
#                 # max_num_partial_prefills=4,
#                 enable_prefix_caching=True,
#                 swap_space=4,
#                 enforce_eager=use_eager,
#                 max_seq_len_to_capture=4096
#             )
#         engine = AsyncLLMEngine.from_engine_args(args)
#         async for _ in engine.generate(
#             prompt="hello",
#             sampling_params=SamplingParams(
#                 max_tokens=1,
#                 temperature=0.0
#             ),
#             request_id=str(uuid4())
#         ):
#             continue
#         return engine

#     @staticmethod
#     async def generate_answer(
#         llm_engine: AsyncLLMEngine,
#         prompt: str
#     ) -> AsyncGenerator[str, None]:
#         num_tokens: int = 0
#         tokens: list[str] = ["",]
#         answer_generator = llm_engine.generate(
#             prompt=prompt,
#             sampling_params=SamplingParams(
#                 # n=1,
#                 # best_of=1,
#                 max_tokens=1024,
#                 temperature=0.5,
#                 top_p=0.85,
#                 top_k=25,
#                 repetition_penalty=1.1,
#                 output_kind=RequestOutputKind.DELTA
#                 # stop=["<|im_end|>"]
#             ),
#             request_id=str(uuid4())
#         )
#         async for request_output in answer_generator:
#             completion_output: CompletionOutput = request_output.outputs[0]
#             generated_text: str = completion_output.text
#             if generated_text:
#                 tokens.append(generated_text)
#                 num_tokens += 1
#                 text = tokens[(-1 * (num_tokens > 1))]
#                 answer_part = {
#                     "type": "answer_part",
#                     "data": text
#                 }
#                 # print(f"data: {json.dumps(answer_part)}\n\n")
#                 yield f"data: {json.dumps(answer_part)}\n\n"
#             if request_output.finished:
#                 stream_end = {
#                     "type": "done"
#                 }
#                 yield f"data: {json.dumps(stream_end)}"
#                 return
import os
import json
from uuid import uuid4
from typing_extensions import AsyncGenerator
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind
from vllm.outputs import CompletionOutput
from vllm.config.compilation import CompilationConfig

class vLLMService:
    @staticmethod
    async def init_resource() -> AsyncLLMEngine:
        """Initialize vLLM engine with proper cache directories"""
        
        # ✅ Lấy cache directory từ environment (được set trong deploy_model.py)
        cache_dir = os.getenv("VLLM_TORCH_COMPILE_DIR", "/cache/vllm_compile")
        use_eager = os.getenv("VLLM_EAGER", "0") == "1"
        
        print(f"📁 vLLM compile cache directory: {cache_dir}")
        
        if not use_eager:
            comp_cfg = CompilationConfig(
                level=2,  # Giảm từ 3 xuống 2 để ổn định
                cache_dir=cache_dir  # ✅ Dùng path đã mount volume
            )
            args = AsyncEngineArgs(
                model="/models_dir/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
                tokenizer="/models_dir/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
                trust_remote_code=True,
                max_model_len=4096,
                gpu_memory_utilization=0.85,
                max_num_batched_tokens=4096,
                enable_prefix_caching=True,
                swap_space=4,
                enforce_eager=use_eager,
                compilation_config=comp_cfg,
                max_seq_len_to_capture=4096
            )
        else:
            args = AsyncEngineArgs(
                model="/models_dir/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
                tokenizer="/models_dir/language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16",
                trust_remote_code=True,
                max_model_len=4096,
                gpu_memory_utilization=0.85,
                max_num_batched_tokens=4096,
                enable_prefix_caching=True,
                swap_space=4,
                enforce_eager=use_eager,
                max_seq_len_to_capture=4096
            )
        
        engine = AsyncLLMEngine.from_engine_args(args)
        
        # Warmup
        print("🔥 Warming up vLLM engine...")
        async for _ in engine.generate(
            prompt="hello",
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            request_id=str(uuid4())
        ):
            continue
        
        print("✅ vLLM engine ready!")
        return engine

    @staticmethod
    async def generate_answer(
        llm_engine: AsyncLLMEngine,
        prompt: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        num_tokens: int = 0
        tokens: list[str] = [""]
        answer_generator = llm_engine.generate(
            prompt=prompt,
            sampling_params=SamplingParams(
                max_tokens=1024,
                temperature=0.5,
                top_p=0.85,
                top_k=25,
                repetition_penalty=1.1,
                output_kind=RequestOutputKind.DELTA
            ),
            request_id=str(uuid4())
        )
        async for request_output in answer_generator:
            completion_output: CompletionOutput = request_output.outputs[0]
            generated_text: str = completion_output.text
            if generated_text:
                tokens.append(generated_text)
                num_tokens += 1
                text = tokens[(-1 * (num_tokens > 1))]
                answer_part = {
                    "type": "answer_part",
                    "data": text
                }
                yield f"data: {json.dumps(answer_part)}\n\n"
            if request_output.finished:
                stream_end = {"type": "done"}
                yield f"data: {json.dumps(stream_end)}"
                return
