# Request and Response Formats

<cite>
**Referenced Files in This Document**   
- [outputs.py](file://vllm/outputs.py)
- [logprobs.py](file://vllm/logprobs.py)
- [sampling_params.py](file://vllm/sampling_params.py)
- [sequence.py](file://vllm/sequence.py)
- [generate.py](file://examples/offline_inference/basic/generate.py)
- [async_llm_streaming.py](file://examples/offline_inference/async_llm_streaming.py)
- [openai_completion_client.py](file://examples/online_serving/openai_completion_client.py)
</cite>

## Table of Contents
1. [RequestOutput Class Structure](#requestoutput-class-structure)
2. [CompletionOutput Fields](#completionoutput-fields)
3. [Finish Reason Usage](#finish-reason-usage)
4. [Logprobs Structure and Interpretation](#logprobs-structure-and-interpretation)
5. [Response Data Processing Examples](#response-data-processing-examples)
6. [Streaming Response Handling](#streaming-response-handling)
7. [Common Issues and Solutions](#common-issues-and-solutions)

## RequestOutput Class Structure

The RequestOutput class represents the complete output data of a completion request to the LLM. It contains comprehensive information about the request, including the prompt, generated outputs, and timing metrics.

Key attributes of the RequestOutput class include:
- **request_id**: Unique identifier for the request
- **prompt**: The input prompt string
- **prompt_token_ids**: Token IDs of the prompt
- **prompt_logprobs**: Log probabilities for prompt tokens (when requested)
- **outputs**: List of CompletionOutput objects containing generated sequences
- **finished**: Boolean indicating if the entire request is complete
- **metrics**: Timing information including arrival time, first token time, and completion time
- **num_cached_tokens**: Number of tokens that benefited from prefix caching

The class provides an `add` method to merge subsequent RequestOutput instances, which is particularly useful for handling streaming responses by aggregating partial outputs.

**Section sources**
- [outputs.py](file://vllm/outputs.py#L83-L189)

## CompletionOutput Fields

The CompletionOutput class represents a single output sequence generated for a request. Each RequestOutput can contain multiple CompletionOutput instances when generating multiple responses (n > 1).

Key fields of the CompletionOutput class include:

- **index**: Position of this output in the request (0-based)
- **text**: The generated output text as a string
- **token_ids**: List of token IDs corresponding to the generated text
- **cumulative_logprob**: Cumulative log probability of the generated output text
- **logprobs**: Log probabilities of the top-k most likely tokens at each position (when logprobs are requested)
- **finish_reason**: String indicating why the sequence generation terminated
- **stop_reason**: Specific stop string or token ID that caused termination

The `finished()` method provides a convenient way to check if the completion has reached a terminal state by checking if finish_reason is not None.

**Section sources**
- [outputs.py](file://vllm/outputs.py#L22-L61)

## Finish Reason Usage

The finish_reason field in CompletionOutput indicates why sequence generation was terminated. Different finish reasons trigger different application logic and help clients understand the context of response completion.

Available finish reasons include:

- **length**: Generation stopped due to reaching the maximum token limit (max_tokens parameter)
- **stop**: Generation stopped due to encountering a stop sequence (either string or token ID)
- **abort**: Generation was aborted due to an error or external cancellation

Applications should handle these different finish reasons appropriately:
- When finish_reason is "length", consider increasing max_tokens to allow for more complete responses
- When finish_reason is "stop", the response is naturally terminated and typically complete
- When finish_reason is "abort", investigate potential issues and potentially retry the request

The finish_reason is crucial for determining whether a response is complete or truncated, enabling applications to make informed decisions about follow-up actions.

**Section sources**
- [outputs.py](file://vllm/outputs.py#L33-L36)
- [test_logger.py](file://tests/test_logger.py#L337)
- [test_output_processor.py](file://tests/v1/engine/test_output_processor.py#L1101)

## Logprobs Structure and Interpretation

When logprobs are requested in the sampling parameters, the system returns probability information for token selection at each generation step. This information is valuable for understanding model confidence and analyzing generation quality.

The logprobs structure depends on the flat_logprobs setting:
- When flat_logprobs=False: Returns a list of dictionaries mapping token IDs to Logprob objects
- When flat_logprobs=True: Returns a FlatLogprobs object that optimizes memory usage by flattening the data structure

Each Logprob object contains:
- **logprob**: The log probability of the chosen token
- **rank**: The vocabulary rank of the chosen token (1-based)
- **decoded_token**: The decoded string representation of the token

For each generated token position, the system returns log probabilities for the top-k most likely tokens (as specified by the logprobs parameter), plus the probability of the actually selected token. This allows for analysis of the model's confidence and alternative token choices at each step.

**Section sources**
- [logprobs.py](file://vllm/logprobs.py#L1-L207)
- [sampling_params.py](file://vllm/sampling_params.py#L176-L183)

## Response Data Processing Examples

The following examples demonstrate how to parse and utilize response data for different applications:

### Basic Response Processing
```python
# From generate.py example
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
```

### Multi-output Processing
When generating multiple responses (n > 1), iterate through all outputs:
```python
for output in outputs:
    print(f"Prompt: {output.prompt}")
    for i, completion in enumerate(output.outputs):
        print(f"Response {i+1}: {completion.text}")
        print(f"Finished due to: {completion.finish_reason}")
```

### Metrics Analysis
Utilize timing metrics for performance monitoring:
```python
for output in outputs:
    metrics = output.metrics
    if metrics:
        ttft = metrics.first_token_time - metrics.arrival_time
        print(f"Time to first token: {ttft:.2f}s")
```

**Section sources**
- [generate.py](file://examples/offline_inference/basic/generate.py#L44-L59)

## Streaming Response Handling

Streaming responses enable real-time processing of generated tokens as they become available. vLLM supports different output kinds through the RequestOutputKind enum:

- **CUMULATIVE**: Returns the complete output sequence in each update
- **DELTA**: Returns only the newly generated tokens since the last update
- **FINAL_ONLY**: Only returns the complete output upon completion

### Streaming Implementation
```python
# From async_llm_streaming.py example
async for output in engine.generate(request_id, prompt, sampling_params):
    for completion in output.outputs:
        new_text = completion.text  # Only new tokens in DELTA mode
        if new_text:
            print(new_text, end="", flush=True)
    
    if output.finished:
        print("\nGeneration complete!")
        break
```

When aggregating partial outputs, use the RequestOutput.add() method with aggregate=True to properly combine text and token IDs from streaming chunks. This ensures that the final output accurately represents the complete generated sequence.

**Section sources**
- [async_llm_streaming.py](file://examples/offline_inference/async_llm_streaming.py#L46-L59)
- [sampling_params.py](file://vllm/sampling_params.py#L102-L108)

## Common Issues and Solutions

### Truncated Responses
Truncated responses occur when generation stops due to the "length" finish reason. This indicates the model reached the max_tokens limit before naturally concluding.

**Solution**: Increase the max_tokens parameter in sampling parameters:
```python
sampling_params = SamplingParams(
    max_tokens=512,  # Increase from default 16
    temperature=0.7
)
```

### Handling Special Tokens
When skip_special_tokens=False, generated text may include special tokens (e.g., EOS, BOS). Applications should be prepared to handle these tokens appropriately.

### Streaming Artifacts
In streaming mode with RequestOutputKind.DELTA, ensure proper text reconstruction by concatenating all delta chunks. Be aware that some deltas may be empty, especially when processing special tokens.

### Memory Optimization
For high-throughput applications, consider enabling flat_logprobs=True to reduce garbage collection overhead when logprobs are requested.

**Section sources**
- [sampling_params.py](file://vllm/sampling_params.py#L171-L172)
- [outputs.py](file://vllm/outputs.py#L145-L174)
- [logprobs.py](file://vllm/logprobs.py#L189-L194)