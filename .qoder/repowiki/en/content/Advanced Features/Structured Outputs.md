# Structured Outputs

<cite>
**Referenced Files in This Document**   
- [structured_outputs.py](file://vllm/sampling_params.py#L33-L46)
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)
- [backend_guidance.py](file://vllm/v1/structured_output/backend_guidance.py)
- [backend_types.py](file://vllm/v1/structured_output/backend_types.py)
- [request.py](file://vllm/v1/structured_output/request.py)
- [structured_outputs.py](file://vllm/config/structured_outputs.py)
- [gpu_input_batch.py](file://vllm/v1/worker/gpu_input_batch.py)
- [structured_outputs.py](file://examples/offline_inference/structured_outputs.py)
- [api_server.py](file://vllm/entrypoints/openai/api_server.py)
- [serving_chat.py](file://vllm/entrypoints/openai/serving_chat.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)

## Introduction
The vLLM structured outputs system enables constrained text generation to produce valid JSON outputs according to specified schemas. This documentation provides a comprehensive analysis of the implementation details, architecture, and integration points of the structured output engine. The system supports multiple backend libraries including xgrammar, outlines, and lm-format-enforcer to validate generated tokens against target schemas during the sampling process. This ensures reliable generation of structured data for applications requiring predictable output formats.

## Project Structure
The structured outputs functionality in vLLM is organized across multiple directories with clear separation of concerns. The core implementation resides in the `vllm/v1/structured_output` directory, which contains backend-specific implementations and shared types. Configuration options are defined in `vllm/config/structured_outputs.py`, while integration with the API server is handled in `vllm/entrypoints/openai/`. Example implementations can be found in the examples directory, demonstrating various use cases for structured output constraints.

```mermaid
graph TD
A[Structured Outputs] --> B[vllm/v1/structured_output]
A --> C[vllm/config]
A --> D[vllm/entrypoints/openai]
A --> E[examples/offline_inference]
B --> F[backend_xgrammar.py]
B --> G[backend_outlines.py]
B --> H[backend_lm_format_enforcer.py]
B --> I[backend_guidance.py]
B --> J[backend_types.py]
C --> K[structured_outputs.py]
D --> L[api_server.py]
D --> M[serving_chat.py]
E --> N[structured_outputs.py]
```

**Diagram sources**
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)
- [backend_guidance.py](file://vllm/v1/structured_output/backend_guidance.py)
- [backend_types.py](file://vllm/v1/structured_output/backend_types.py)
- [structured_outputs.py](file://vllm/config/structured_outputs.py)
- [api_server.py](file://vllm/entrypoints/openai/api_server.py)
- [serving_chat.py](file://vllm/entrypoints/openai/serving_chat.py)
- [structured_outputs.py](file://examples/offline_inference/structured_outputs.py)

**Section sources**
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)
- [backend_guidance.py](file://vllm/v1/structured_output/backend_guidance.py)
- [backend_types.py](file://vllm/v1/structured_output/backend_types.py)
- [structured_outputs.py](file://vllm/config/structured_outputs.py)
- [api_server.py](file://vllm/entrypoints/openai/api_server.py)
- [serving_chat.py](file://vllm/entrypoints/openai/serving_chat.py)
- [structured_outputs.py](file://examples/offline_inference/structured_outputs.py)

## Core Components
The structured outputs system in vLLM consists of several core components that work together to constrain text generation. The `StructuredOutputsParams` class defines the parameters for structured output constraints, including JSON schema, regex patterns, choice lists, and custom grammars. Multiple backend implementations handle the actual validation and token restriction during generation. The system integrates with the sampling process through logits processors that modify the probability distribution of tokens based on the current state of the structured output constraint.

**Section sources**
- [structured_outputs.py](file://vllm/sampling_params.py#L33-L46)
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)
- [backend_guidance.py](file://vllm/v1/structured_output/backend_guidance.py)

## Architecture Overview
The structured outputs architecture in vLLM follows a modular design with clear separation between the configuration layer, backend implementations, and integration points. The system uses a pluggable backend architecture that allows different constraint validation libraries to be used interchangeably. When a request with structured output constraints is received, the system selects an appropriate backend based on the constraint type and configuration. The selected backend compiles the constraint into a finite state machine that guides the token generation process.

```mermaid
graph LR
A[Client Request] --> B[API Server]
B --> C[Structured Output Manager]
C --> D{Constraint Type}
D --> |JSON Schema| E[xgrammar Backend]
D --> |Regex| F[Outlines Backend]
D --> |Choice List| G[lm-format-enforcer Backend]
D --> |Custom Grammar| H[Guidance Backend]
E --> I[Token Bitmask]
F --> I
G --> I
H --> I
I --> J[Sampling Process]
J --> K[Generated Tokens]
K --> L[Validation]
L --> M[Response]
```

**Diagram sources**
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)
- [backend_guidance.py](file://vllm/v1/structured_output/backend_guidance.py)
- [gpu_input_batch.py](file://vllm/v1/worker/gpu_input_batch.py)

## Detailed Component Analysis

### Backend Implementations
The structured outputs system in vLLM supports multiple backend libraries, each with its own strengths and capabilities. The choice of backend can be specified in the configuration or automatically selected based on the constraint type and other factors.

#### xgrammar Backend
The xgrammar backend provides high-performance validation for JSON schemas, regex patterns, and custom grammars. It compiles constraints into efficient finite state machines that can be used to validate token sequences during generation. The backend supports advanced features like jump-forward decoding for improved performance.

```mermaid
classDiagram
class XgrammarBackend {
+compile_grammar(request_type, grammar_spec)
+allocate_token_bitmask(max_num_seqs)
+destroy()
}
class XgrammarGrammar {
+accept_tokens(request_id, tokens)
+validate_tokens(tokens)
+rollback(num_tokens)
+fill_bitmask(bitmask, idx)
+is_terminated()
+reset()
}
XgrammarBackend --> XgrammarGrammar : creates
```

**Diagram sources**
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)

#### Outlines Backend
The outlines backend uses the outlines_core library to provide structured output capabilities. It converts JSON schemas to regex patterns and uses a guide-based approach to constrain token generation. This backend is particularly effective for complex JSON schema validation.

```mermaid
classDiagram
class OutlinesBackend {
+compile_grammar(request_type, grammar_spec)
+allocate_token_bitmask(max_num_seqs)
+destroy()
}
class OutlinesGrammar {
+accept_tokens(request_id, tokens)
+validate_tokens(tokens)
+rollback(num_tokens)
+fill_bitmask(bitmask, idx)
+is_terminated()
+reset()
}
OutlinesBackend --> OutlinesGrammar : creates
```

**Diagram sources**
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)

#### lm-format-enforcer Backend
The lm-format-enforcer backend uses the lm-format-enforcer library to enforce structured output constraints. It provides a flexible framework for defining custom parsers and validators for various output formats.

```mermaid
classDiagram
class LMFormatEnforcerBackend {
+compile_grammar(request_type, grammar_spec)
+allocate_token_bitmask(max_num_seqs)
+destroy()
}
class LMFormatEnforcerGrammar {
+accept_tokens(request_id, tokens)
+validate_tokens(tokens)
+rollback(num_tokens)
+fill_bitmask(bitmask, batch_index)
+is_terminated()
+reset()
}
LMFormatEnforcerBackend --> LMFormatEnforcerGrammar : creates
```

**Diagram sources**
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)

#### Guidance Backend
The guidance backend uses the guidance library to provide structured output capabilities. It supports complex grammar definitions and provides fine-grained control over the generation process.

```mermaid
classDiagram
class GuidanceBackend {
+compile_grammar(request_type, grammar_spec)
+allocate_token_bitmask(max_num_seqs)
+destroy()
}
class GuidanceGrammar {
+accept_tokens(request_id, tokens)
+validate_tokens(tokens)
+rollback(num_tokens)
+fill_bitmask(bitmask, idx)
+is_terminated()
+reset()
}
GuidanceBackend --> GuidanceGrammar : creates
```

**Diagram sources**
- [backend_guidance.py](file://vllm/v1/structured_output/backend_guidance.py)

### Integration with Sampling Process
The structured outputs system integrates with the sampling process through a logits processor that modifies the probability distribution of tokens based on the current state of the structured output constraint. When a request with structured output constraints is processed, the system creates a grammar bitmask that indicates which tokens are allowed at each position in the sequence.

```mermaid
sequenceDiagram
participant Client
participant API_Server
participant Structured_Output_Manager
participant Backend
participant Sampler
participant Model
Client->>API_Server : Send request with structured output constraint
API_Server->>Structured_Output_Manager : Parse constraint and select backend
Structured_Output_Manager->>Backend : Compile constraint to grammar
Backend-->>Structured_Output_Manager : Return compiled grammar
Structured_Output_Manager->>Sampler : Provide grammar for token validation
loop Generate tokens
Model->>Sampler : Generate logits
Sampler->>Backend : Validate tokens against grammar
Backend-->>Sampler : Return allowed tokens
Sampler->>Model : Sample from allowed tokens
end
Model-->>Client : Return structured output
```

**Diagram sources**
- [backend_types.py](file://vllm/v1/structured_output/backend_types.py)
- [gpu_input_batch.py](file://vllm/v1/worker/gpu_input_batch.py)
- [structured_outputs.py](file://vllm/sampling_params.py#L33-L46)

### Configuration and Validation
The structured outputs system provides comprehensive configuration options through the `StructuredOutputsConfig` class. These options allow fine-tuning of the behavior for different use cases and constraints.

```mermaid
flowchart TD
A[Request with Structured Output Constraint] --> B{Validate Constraint}
B --> |Valid| C[Select Backend]
C --> D{Backend Support}
D --> |Supported| E[Compile Constraint]
D --> |Not Supported| F[Fallback or Error]
E --> G[Generate Token Bitmask]
G --> H[Integrate with Sampling Process]
H --> I[Validate Generated Tokens]
I --> J{Valid Sequence?}
J --> |Yes| K[Return Output]
J --> |No| L[Error Recovery]
L --> M[Retry or Return Error]
```

**Diagram sources**
- [structured_outputs.py](file://vllm/config/structured_outputs.py)
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)

**Section sources**
- [structured_outputs.py](file://vllm/config/structured_outputs.py)
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)
- [backend_guidance.py](file://vllm/v1/structured_output/backend_guidance.py)

## Dependency Analysis
The structured outputs system in vLLM has dependencies on several external libraries and internal components. The primary external dependencies are the backend libraries: xgrammar, outlines_core, lm-format-enforcer, and guidance. These libraries provide the core constraint validation capabilities. The system also depends on the tokenizer, logits processor, and API server components within vLLM.

```mermaid
graph TD
A[Structured Outputs] --> B[xgrammar]
A --> C[outlines_core]
A --> D[lm-format-enforcer]
A --> E[guidance]
A --> F[Tokenizer]
A --> G[Logits Processor]
A --> H[API Server]
A --> I[Sampling Process]
B --> J[Regex Automata]
C --> K[Regex Engine]
D --> L[Character Level Parsing]
E --> M[Grammar Engine]
```

**Diagram sources**
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)
- [backend_guidance.py](file://vllm/v1/structured_output/backend_guidance.py)
- [gpu_input_batch.py](file://vllm/v1/worker/gpu_input_batch.py)

## Performance Considerations
The structured outputs system in vLLM is designed to minimize performance overhead while ensuring reliable constraint validation. The use of compiled finite state machines and efficient bitmask operations allows for fast token validation during generation. The system also supports speculative decoding to further improve throughput for structured output requests.

**Section sources**
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)
- [backend_guidance.py](file://vllm/v1/structured_output/backend_guidance.py)

## Troubleshooting Guide
Common issues with structured outputs in vLLM include schema validation errors, unsupported constraint types, and performance bottlenecks. The system provides detailed error messages to help diagnose and resolve these issues. For schema validation errors, ensure that the JSON schema is valid and supported by the selected backend. For performance issues, consider using the xgrammar backend which provides optimized validation for common constraint types.

**Section sources**
- [backend_xgrammar.py](file://vllm/v1/structured_output/backend_xgrammar.py)
- [backend_outlines.py](file://vllm/v1/structured_output/backend_outlines.py)
- [backend_lm_format_enforcer.py](file://vllm/v1/structured_output/backend_lm_format_enforcer.py)
- [backend_guidance.py](file://vllm/v1/structured_output/backend_guidance.py)
- [api_server.py](file://vllm/entrypoints/openai/api_server.py)

## Conclusion
The structured outputs system in vLLM provides a robust and flexible framework for generating valid JSON outputs according to specified schemas. By supporting multiple backend libraries and integrating seamlessly with the sampling process, the system enables reliable structured data generation for a wide range of applications. The modular architecture allows for easy extension and customization, making it suitable for both simple and complex use cases.