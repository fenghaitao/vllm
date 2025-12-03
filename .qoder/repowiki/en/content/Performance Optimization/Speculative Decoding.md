# Speculative Decoding

<cite>
**Referenced Files in This Document**   
- [speculative.py](file://vllm/config/speculative.py)
- [eagle.py](file://vllm/v1/spec_decode/eagle.py)
- [medusa.py](file://vllm/v1/spec_decode/medusa.py)
- [ngram_proposer.py](file://vllm/v1/spec_decode/ngram_proposer.py)
- [suffix_decoding.py](file://vllm/v1/spec_decode/suffix_decoding.py)
- [metrics.py](file://vllm/v1/spec_decode/metrics.py)
- [utils.py](file://vllm/v1/spec_decode/utils.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Domain Model](#domain-model)
5. [Token Verification Process](#token-verification-process)
6. [Configuration Options](#configuration-options)
7. [Performance Metrics](#performance-metrics)
8. [Common Issues and Solutions](#common-issues-and-solutions)

## Introduction
Speculative decoding is a performance optimization technique in vLLM that accelerates text generation by using a smaller draft model to predict tokens, which are then verified by a larger target model. This approach significantly increases generation throughput by reducing the number of expensive forward passes required by the target model. The implementation supports multiple speculative methods including draft models, n-gram proposers, Medusa, Eagle, and suffix decoding, each with specific use cases and performance characteristics.

## Architecture Overview
The speculative decoding architecture in vLLM consists of two main components: the draft model (proposer) and the target model (verifier). The draft model generates a sequence of speculative tokens based on the current context, and the target model verifies these tokens in a single forward pass. When tokens are accepted, the generation process advances by multiple steps, dramatically improving throughput. The system is designed to handle various speculative methods through a unified interface, allowing different proposer implementations to be used interchangeably.

```mermaid
graph TD
A[Input Sequence] --> B[Draft Model/Proposer]
B --> C[Speculative Tokens]
C --> D[Target Model Verification]
D --> E{Tokens Accepted?}
E --> |Yes| F[Advance by Multiple Tokens]
E --> |No| G[Rollback and Regenerate]
F --> H[Output Sequence]
G --> B
H --> I[Final Output]
```

**Diagram sources**
- [eagle.py](file://vllm/v1/spec_decode/eagle.py#L57-L800)
- [medusa.py](file://vllm/v1/spec_decode/medusa.py#L18-L74)

## Core Components
The speculative decoding system in vLLM is built around several key components that work together to enable efficient token prediction and verification. The main components include the SpeculativeConfig class that manages configuration parameters, various proposer implementations (EagleProposer, MedusaProposer, NgramProposer, SuffixDecodingProposer), and the metrics system that tracks performance. The draft model worker is responsible for generating speculative tokens, while the target model performs the verification step. The system is designed to be modular, allowing different speculative methods to be implemented and used based on the specific requirements.

**Section sources**
- [speculative.py](file://vllm/config/speculative.py#L52-L644)
- [eagle.py](file://vllm/v1/spec_decode/eagle.py#L57-L800)

## Domain Model
The domain model for speculative decoding in vLLM centers around the SpeculativeConfig class, which encapsulates all configuration parameters for the speculative decoding process. This includes the number of speculative tokens, the draft model specification, and method-specific parameters. The DraftModelWorker concept is implemented through various proposer classes that generate speculative tokens. The SpeculativeConfig class also manages the relationship between the draft and target models, including tensor parallelism configuration and model length constraints. The system supports multiple speculative methods through the SpeculativeMethod enum, allowing flexible configuration based on use case requirements.

```mermaid
classDiagram
class SpeculativeConfig {
+num_speculative_tokens : int
+model : str
+method : SpeculativeMethod
+draft_tensor_parallel_size : int
+quantization : str
+max_model_len : int
+revision : str
+code_revision : str
+disable_by_batch_size : int
+prompt_lookup_max : int
+prompt_lookup_min : int
+speculative_token_tree : str
+target_model_config : ModelConfig
+target_parallel_config : ParallelConfig
+draft_model_config : ModelConfig
+draft_parallel_config : ParallelConfig
+compute_hash() : str
+__post_init__() : void
+_verify_args() : void
+use_eagle() : bool
}
class EagleProposer {
+vllm_config : VllmConfig
+speculative_config : SpeculativeConfig
+draft_model_config : ModelConfig
+method : str
+device : torch.device
+dtype : torch.dtype
+max_model_len : int
+block_size : int
+dp_rank : int
+num_speculative_tokens : int
+max_num_tokens : int
+hidden_size : int
+inputs_embeds_size : int
+supports_mm_inputs : bool
+attn_metadata_builder : AttentionMetadataBuilder
+draft_indexer_metadata_builder : AttentionMetadataBuilder
+attn_layer_names : list[str]
+indexer_layer_names : list[str]
+eagle3_use_aux_hidden_state : bool
+use_cuda_graph : bool
+input_ids : torch.Tensor
+positions : torch.Tensor
+hidden_states : torch.Tensor
+arange : torch.Tensor
+inputs_embeds : torch.Tensor
+backup_next_token_ids : CpuGpuBuffer
+allowed_attn_types : tuple
+tree_choices : list[tuple[int, ...]]
+cu_drafts_per_level : list[int]
+child_drafts_per_level : list[int]
+tree_draft_pos_offsets : torch.Tensor
+propose() : torch.Tensor
+prepare_next_token_ids_cpu() : torch.Tensor
+prepare_next_token_ids_padded() : tuple[torch.Tensor, torch.Tensor]
+prepare_inputs_padded() : tuple[CommonAttentionMetadata, torch.Tensor]
+propose_tree() : list[torch.Tensor]
}
class MedusaProposer {
+vllm_config : VllmConfig
+device : torch.device
+max_num_tokens : int
+hidden_size : int
+dtype : torch.dtype
+model : nn.Module
+propose() : list[list[int]]
+load_model() : void
+dummy_run() : void
}
class NgramProposer {
+min_n : int
+max_n : int
+k : int
+max_model_len : int
+valid_ngram_draft : np.ndarray
+valid_ngram_num_drafts : np.ndarray
+num_tokens_threshold : int
+num_numba_thread_available : int
+batch_propose() : list[list[int]]
+propose() : list[list[int]]
+load_model() : void
}
class SuffixDecodingProposer {
+num_speculative_tokens : int
+max_tree_depth : int
+max_spec_factor : float
+min_token_prob : float
+max_model_len : int
+suffix_cache : SuffixDecodingCache
+propose() : list[list[int]]
+load_model() : void
}
SpeculativeConfig --> EagleProposer : "configures"
SpeculativeConfig --> MedusaProposer : "configures"
SpeculativeConfig --> NgramProposer : "configures"
SpeculativeConfig --> SuffixDecodingProposer : "configures"
```

**Diagram sources**
- [speculative.py](file://vllm/config/speculative.py#L52-L644)
- [eagle.py](file://vllm/v1/spec_decode/eagle.py#L57-L800)
- [medusa.py](file://vllm/v1/spec_decode/medusa.py#L18-L74)
- [ngram_proposer.py](file://vllm/v1/spec_decode/ngram_proposer.py#L11-L292)
- [suffix_decoding.py](file://vllm/v1/spec_decode/suffix_decoding.py#L7-L102)

## Token Verification Process
The token verification process in vLLM's speculative decoding system follows a multi-step approach. First, the draft model generates a sequence of speculative tokens based on the current context. These tokens are then passed to the target model, which verifies them in a single forward pass. The verification process compares the logits produced by the target model with the speculative tokens to determine acceptance. When tokens are accepted, the generation process advances by the number of accepted tokens plus one (the bonus token). If tokens are rejected, the system rolls back to the last accepted token and continues generation from that point. This process is optimized through CUDA kernels and memory management techniques to minimize overhead.

```mermaid
sequenceDiagram
participant Client as "Client"
participant Scheduler as "Scheduler"
participant DraftModel as "Draft Model"
participant TargetModel as "Target Model"
Client->>Scheduler : Submit Request
Scheduler->>DraftModel : Get Speculative Tokens
DraftModel->>Scheduler : Return Speculative Tokens
Scheduler->>TargetModel : Verify Tokens
TargetModel->>Scheduler : Return Verification Results
alt Tokens Accepted
Scheduler->>Scheduler : Advance by Multiple Tokens
Scheduler->>Client : Return Accepted Tokens
else Tokens Rejected
Scheduler->>Scheduler : Rollback to Last Accepted Token
Scheduler->>DraftModel : Generate New Speculative Tokens
end
Scheduler->>Client : Final Output
```

**Diagram sources**
- [eagle.py](file://vllm/v1/spec_decode/eagle.py#L219-L525)
- [medusa.py](file://vllm/v1/spec_decode/medusa.py#L37-L50)

## Configuration Options
vLLM provides extensive configuration options for speculative decoding through the SpeculativeConfig class. Key parameters include num_speculative_tokens (the number of tokens to speculate), model (the draft model name), method (the speculative method to use), and draft_tensor_parallel_size (tensor parallelism for the draft model). Advanced options include disable_by_batch_size (to disable speculation under high load), prompt_lookup_max/min (for n-gram proposer), and various suffix decoding parameters. The configuration system automatically validates parameters and ensures compatibility between draft and target models. Users can also configure quantization, revision, and code revision for the draft model to optimize performance and memory usage.

**Section sources**
- [speculative.py](file://vllm/config/speculative.py#L60-L148)

## Performance Metrics
The speculative decoding system in vLLM tracks comprehensive performance metrics to monitor effectiveness and identify optimization opportunities. Key metrics include mean acceptance length (average number of tokens accepted per speculation), acceptance throughput (tokens accepted per second), draft throughput (tokens speculated per second), and per-position acceptance rates. These metrics are exposed through both logging and Prometheus monitoring, allowing users to track performance over time. The system also calculates draft acceptance rate as the ratio of accepted tokens to drafted tokens, providing insight into the efficiency of the speculative process. These metrics help users tune configuration parameters and select appropriate draft models for their use cases.

```mermaid
flowchart TD
A[Start] --> B[Gather Metrics]
B --> C{Calculate}
C --> D[Mean Acceptance Length]
C --> E[Acceptance Throughput]
C --> F[Draft Throughput]
C --> G[Draft Acceptance Rate]
C --> H[Per-Position Acceptance Rates]
D --> I[Log Metrics]
E --> I
F --> I
G --> I
H --> I
I --> J[Prometheus Export]
J --> K[Monitoring Dashboard]
```

**Diagram sources**
- [metrics.py](file://vllm/v1/spec_decode/metrics.py#L16-L226)

## Common Issues and Solutions
Common issues in speculative decoding include low acceptance rates due to draft model inaccuracy, compatibility issues between draft and target models, and performance degradation under high load. Solutions include careful selection of draft models that are well-matched to the target model, tuning the number of speculative tokens based on acceptance rates, and using the disable_by_batch_size parameter to disable speculation when system load is high. For n-gram proposers, adjusting the prompt_lookup_max/min parameters can improve performance. The system also provides detailed metrics to help diagnose issues and optimize configuration. In cases where speculative decoding is not beneficial, it can be disabled selectively for specific requests based on their characteristics.

**Section sources**
- [speculative.py](file://vllm/config/speculative.py#L96-L103)
- [utils.py](file://vllm/v1/spec_decode/utils.py#L9-L17)