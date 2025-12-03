# API Reference

<cite>
**Referenced Files in This Document**   
- [llm.py](file://vllm/entrypoints/llm.py)
- [async_llm.py](file://vllm/v1/engine/async_llm.py)
- [sampling_params.py](file://vllm/sampling_params.py)
- [outputs.py](file://vllm/outputs.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [LLM Class](#llm-class)
3. [AsyncLLMEngine](#asyncllmengine)
4. [SamplingParams](#samplingparams)
5. [RequestOutput and CompletionOutput](#requestoutput-and-completionoutput)
6. [Usage Examples](#usage-examples)
7. [Error Handling and Performance](#error-handling-and-performance)

## Introduction
This document provides comprehensive API documentation for vLLM's programmatic interfaces. The vLLM library offers two primary interfaces for interacting with language models: the synchronous `LLM` class for offline inference and the asynchronous `AsyncLLMEngine` for online serving. These interfaces enable efficient text generation with advanced sampling capabilities, streaming support, and fine-grained control over generation parameters.

The API design follows a clean separation between model initialization, request processing, and output handling. The `LLM` class provides a simple, high-level interface for batch processing, while `AsyncLLMEngine` offers more granular control for real-time applications with streaming capabilities. Both interfaces share common components like `SamplingParams` for controlling generation behavior and `RequestOutput` for representing results.

**Section sources**
- [llm.py](file://vllm/entrypoints/llm.py#L90-L1761)
- [async_llm.py](file://vllm/v1/engine/async_llm.py#L54-L875)

## LLM Class
The `LLM` class is the primary interface for offline inference in vLLM, designed for generating text from prompts with various sampling parameters. It encapsulates a tokenizer, language model, and GPU memory space for intermediate states (KV cache), providing an intelligent batching mechanism and efficient memory management.

### Initialization Parameters
The `LLM` class is initialized with several parameters that configure the model and inference behavior:

- **model**: The name or path of a HuggingFace Transformers model
- **tokenizer**: The name or path of a HuggingFace Transformers tokenizer
- **tokenizer_mode**: Tokenizer mode ("auto" for fast tokenizer if available, "slow" for always using slow tokenizer)
- **skip_tokenizer_init**: Skip initialization of tokenizer and detokenizer when valid token IDs are provided
- **trust_remote_code**: Trust remote code from HuggingFace when downloading model and tokenizer
- **tensor_parallel_size**: Number of GPUs to use for distributed execution with tensor parallelism
- **dtype**: Data type for model weights and activations (float32, float16, bfloat16, or "auto")
- **quantization**: Method used to quantize model weights ("awq", "gptq", "fp8", or None)
- **gpu_memory_utilization**: Ratio of GPU memory to reserve for model weights, activations, and KV cache (0-1)
- **kv_cache_memory_bytes**: Size of KV Cache per GPU in bytes (overrides gpu_memory_utilization when specified)
- **swap_space**: Size (GiB) of CPU memory per GPU to use as swap space for requests with best_of > 1
- **cpu_offload_gb**: Size (GiB) of CPU memory to use for offloading model weights
- **enforce_eager**: Whether to enforce eager execution (disabling CUDA graph)
- **seed**: Seed to initialize the random number generator for sampling

### Generate Methods
The `LLM` class provides several methods for text generation:

- **generate**: Generates outputs for a list of prompts with specified sampling parameters. Returns a list of RequestOutput objects containing the generated text and metadata.
- **chat**: Processes chat conversations with system, user, and assistant messages, applying the appropriate chat template for the model.
- **classify**: Performs classification tasks using a pooling model.
- **embed**: Generates embeddings from input text using a pooling model.
- **score**: Computes similarity scores between text pairs.

The generate method supports both single prompts and batch processing of multiple prompts, automatically handling batching for optimal performance.

### Context Manager Usage
The `LLM` class can be used as a context manager, ensuring proper resource cleanup:

```python
with LLM(model="meta-llama/Llama-3.2-1B-Instruct") as llm:
    outputs = llm.generate("Hello, my name is")
    print(outputs[0].outputs[0].text)
```

When used as a context manager, the LLM instance automatically shuts down and releases GPU memory when exiting the context block.

**Section sources**
- [llm.py](file://vllm/entrypoints/llm.py#L90-L1761)

## AsyncLLMEngine
The `AsyncLLMEngine` (aliased as `AsyncLLM`) provides an asynchronous interface for online serving of language models, enabling efficient handling of concurrent requests with streaming capabilities.

### Asynchronous Request Handling
The `AsyncLLMEngine` processes requests asynchronously using Python's asyncio framework, allowing for non-blocking operations and efficient handling of multiple concurrent requests. The primary method for text generation is `generate`, which returns an asynchronous generator that yields `RequestOutput` objects as they become available.

```python
async for output in engine.generate(prompt, sampling_params, request_id):
    # Process output as it arrives
    pass
```

This asynchronous pattern enables real-time applications to start processing results immediately without waiting for the complete generation to finish.

### Streaming Capabilities
The engine supports three output kinds through the `output_kind` parameter in `SamplingParams`:

- **CUMULATIVE**: Returns the entire output so far in every RequestOutput (default)
- **DELTA**: Returns only the new tokens generated since the last output
- **FINAL_ONLY**: Does not return intermediate outputs, only the final result

The DELTA mode is particularly useful for streaming applications where you want to display tokens as they are generated, providing a more interactive user experience.

### Lifecycle Management
The `AsyncLLMEngine` provides comprehensive lifecycle management methods:

- **shutdown**: Cleans up the background process and IPC resources
- **pause_generation**: Pauses generation to allow model weight updates, with options to wait for in-flight requests or clear caches
- **resume_generation**: Resumes generation after being paused
- **is_paused**: Checks whether the engine is currently paused
- **reset_prefix_cache**: Resets the prefix cache
- **reset_mm_cache**: Resets the multi-modal cache
- **sleep/wake_up**: Controls the sleep state of the engine for power management

These methods enable dynamic management of the engine during operation, supporting use cases like model updates, maintenance, and power optimization.

**Section sources**
- [async_llm.py](file://vllm/v1/engine/async_llm.py#L54-L875)
- [protocol.py](file://vllm/engine/protocol.py#L21-L189)

## SamplingParams
The `SamplingParams` class controls the text generation behavior, following the OpenAI text completion API parameters with additional support for beam search.

### Configurable Parameters
The class provides numerous parameters to fine-tune the generation process:

- **n**: Number of outputs to return for the request (default: 1)
- **presence_penalty**: Penalizes new tokens based on whether they appear in generated text (range: -2.0 to 2.0, default: 0.0)
- **frequency_penalty**: Penalizes new tokens based on their frequency in generated text (range: -2.0 to 2.0, default: 0.0)
- **repetition_penalty**: Penalizes new tokens based on whether they appear in prompt and generated text (range: > 0.0, default: 1.0)
- **temperature**: Controls randomness of sampling (range: >= 0.0, default: 1.0; 0.0 means greedy sampling)
- **top_p**: Controls cumulative probability of top tokens to consider (range: (0, 1], default: 1.0)
- **top_k**: Controls number of top tokens to consider (range: >= -1, default: 0; 0 or -1 means consider all tokens)
- **min_p**: Minimum probability for a token to be considered relative to the most likely token (range: [0, 1], default: 0.0)
- **seed**: Random seed for generation (default: None)
- **stop**: String(s) that stop generation when generated (default: None)
- **stop_token_ids**: Token IDs that stop generation when generated (default: None)
- **ignore_eos**: Whether to ignore the EOS token and continue generating (default: False)
- **max_tokens**: Maximum number of tokens to generate per output sequence (default: 16)
- **min_tokens**: Minimum number of tokens to generate before EOS or stop_token_ids can be generated (default: 0)
- **logprobs**: Number of log probabilities to return per output token (default: None; -1 returns all vocab_size log probabilities)
- **prompt_logprobs**: Number of log probabilities to return per prompt token (default: None; -1 returns all vocab_size log probabilities)
- **flat_logprobs**: Whether to return logprobs in flattened format for better performance (default: False)
- **detokenize**: Whether to detokenize the output (default: True)
- **skip_special_tokens**: Whether to skip special tokens in the output (default: True)
- **spaces_between_special_tokens**: Whether to add spaces between special tokens in the output (default: True)
- **logits_processors**: Functions that modify logits based on previously generated tokens (default: None)
- **include_stop_str_in_output**: Whether to include stop strings in output text (default: False)
- **truncate_prompt_tokens**: If set to -1, uses model's truncation size; if set to integer k, uses only the last k tokens from the prompt (default: None)
- **output_kind**: Controls output format (CUMULATIVE, DELTA, or FINAL_ONLY) (default: CUMULATIVE)

### Effects on Generation
These parameters significantly affect the quality and characteristics of generated text:

- **Temperature**: Lower values make the model more deterministic, while higher values make it more random.
- **Top-p and Top-k**: These parameters control the diversity of generated text by limiting the token selection to a subset of the most probable tokens.
- **Penalties**: Presence, frequency, and repetition penalties help prevent repetition and encourage diversity in the generated text.
- **Stop conditions**: Stop strings and token IDs allow for controlled termination of generation based on content.
- **Log probabilities**: Enabling logprobs provides insight into the model's confidence in its token selections.

The parameters are validated during initialization, with appropriate error messages for invalid values, ensuring robust operation.

**Section sources**
- [sampling_params.py](file://vllm/sampling_params.py#L110-L580)

## RequestOutput and CompletionOutput
The output classes in vLLM provide structured representations of generation results, with `RequestOutput` containing the overall request information and `CompletionOutput` representing individual completion results.

### RequestOutput
The `RequestOutput` class represents the output data of a completion request to the LLM with the following fields:

- **request_id**: Unique ID of the request
- **prompt**: Prompt string of the request
- **prompt_token_ids**: Token IDs of the prompt
- **prompt_logprobs**: Log probabilities to return per prompt token
- **outputs**: List of CompletionOutput objects representing the output sequences
- **finished**: Whether the whole request is finished
- **metrics**: Metrics associated with the request
- **lora_request**: LoRA request used to generate the output
- **encoder_prompt**: Encoder prompt string (for encoder/decoder models)
- **encoder_prompt_token_ids**: Token IDs of the encoder prompt
- **num_cached_tokens**: Number of tokens with prefix cache hit
- **kv_transfer_params**: Parameters for remote K/V transfer

### CompletionOutput
The `CompletionOutput` class represents the output data of one completion output of a request with these fields:

- **index**: Index of the output in the request
- **text**: Generated output text
- **token_ids**: Token IDs of the generated output text
- **cumulative_logprob**: Cumulative log probability of the generated output text
- **logprobs**: Log probabilities of the top probability words at each position
- **finish_reason**: Reason why the sequence is finished
- **stop_reason**: Stop string or token ID that caused completion to stop
- **lora_request**: LoRA request used to generate the output

The `finished` method indicates whether the completion has finished, and the `__repr__` method provides a string representation of the object.

These classes enable comprehensive access to generation results, including text, token-level information, probabilities, and metadata, supporting various use cases from simple text generation to detailed analysis of model behavior.

**Section sources**
- [outputs.py](file://vllm/outputs.py#L22-L189)

## Usage Examples
This section provides practical examples demonstrating both synchronous and asynchronous usage patterns of vLLM's API.

### Synchronous Usage
The following example demonstrates basic synchronous usage with the LLM class:

```python
from vllm import LLM, SamplingParams

# Initialize the LLM
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate text from a prompt
outputs = llm.generate("Hello, my name is", sampling_params)

# Print the generated text
for output in outputs:
    print(output.outputs[0].text)
```

For batch processing multiple prompts:

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is"
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### Asynchronous Usage
The following example demonstrates asynchronous streaming with AsyncLLM:

```python
import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

async def main():
    # Create AsyncLLM engine
    engine_args = AsyncEngineArgs(model="meta-llama/Llama-3.2-1B-Instruct")
    engine = AsyncLLM.from_engine_args(engine_args)
    
    try:
        # Configure sampling parameters for streaming
        sampling_params = SamplingParams(
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
            output_kind=RequestOutputKind.DELTA
        )
        
        # Stream response
        async for output in engine.generate(
            request_id="request-1",
            prompt="The future of artificial intelligence is",
            sampling_params=sampling_params
        ):
            for completion in output.outputs:
                if completion.text:
                    print(completion.text, end="", flush=True)
            
            if output.finished:
                print("\nGeneration complete!")
                break
                
    finally:
        engine.shutdown()

# Run the async function
asyncio.run(main())
```

These examples illustrate the core usage patterns of vLLM's API, from simple synchronous generation to advanced asynchronous streaming scenarios.

**Section sources**
- [llm.py](file://vllm/entrypoints/llm.py#L90-L1761)
- [async_llm.py](file://vllm/v1/engine/async_llm.py#L54-L875)
- [examples/offline_inference/basic/generate.py](file://examples/offline_inference/basic/generate.py#L39-L65)
- [examples/offline_inference/async_llm_streaming.py](file://examples/offline_inference/async_llm_streaming.py#L22-L63)

## Error Handling and Performance
This section addresses common issues like request timeouts and provides performance considerations for different API usage patterns.

### Common Issues and Solutions
**Request Timeouts**: When dealing with long-running requests, configure appropriate timeout settings in your client code. For streaming applications, implement heartbeat mechanisms to maintain connection stability.

**Memory Management**: Monitor GPU memory utilization and adjust parameters like `gpu_memory_utilization` and `kv_cache_memory_bytes` based on your workload. For memory-constrained environments, consider using quantized models or enabling CPU offload.

**Batch Size Optimization**: Adjust the batch size based on your hardware capabilities and latency requirements. Larger batches improve throughput but may increase latency. Use the `max_num_batched_tokens` parameter to control the maximum number of tokens processed in a single batch.

**Context Length**: Be aware of the model's maximum context length (`max_model_len`) and use `truncate_prompt_tokens` to handle long prompts when necessary.

### Performance Considerations
**Synchronous vs Asynchronous**: Use synchronous APIs for batch processing and offline inference where you can wait for complete results. Use asynchronous APIs for real-time applications requiring low latency and streaming capabilities.

**Caching**: Enable prefix caching to improve performance for requests with similar prefixes. This can significantly reduce computation for chat applications where conversation history is maintained.

**Model Quantization**: Consider using quantized models (AWQ, GPTQ, FP8) for improved inference speed and reduced memory footprint, especially on consumer hardware.

**Tensor Parallelism**: For multi-GPU setups, use tensor parallelism (`tensor_parallel_size`) to distribute the model across multiple GPUs, enabling inference with larger models.

**CUDA Graphs**: By default, vLLM uses CUDA graphs for improved performance. Disable with `enforce_eager=True` only when necessary for debugging or specific use cases.

These considerations help optimize the performance of vLLM applications for different deployment scenarios, from local development to production-scale serving.

**Section sources**
- [llm.py](file://vllm/entrypoints/llm.py#L90-L1761)
- [async_llm.py](file://vllm/v1/engine/async_llm.py#L54-L875)
- [sampling_params.py](file://vllm/sampling_params.py#L110-L580)