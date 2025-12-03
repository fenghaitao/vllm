# CUDA Graphs

<cite>
**Referenced Files in This Document**   
- [cuda_graph.py](file://vllm/compilation/cuda_graph.py)
- [cudagraph_utils.py](file://vllm/v1/worker/gpu/cudagraph_utils.py)
- [eagle_cudagraph.py](file://vllm/v1/worker/gpu/spec_decode/eagle_cudagraph.py)
- [compilation.py](file://vllm/config/compilation.py)
- [cudagraph_dispatcher.py](file://vllm/v1/cudagraph_dispatcher.py)
- [monitor.py](file://vllm/compilation/monitor.py)
- [parallel_state.py](file://vllm/distributed/parallel_state.py)
- [pynccl_allocator.py](file://vllm/distributed/device_communicators/pynccl_allocator.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Domain Model](#domain-model)
3. [Graph Capture and Replay Mechanism](#graph-capture-and-replay-mechanism)
4. [Integration with GPU Model Runner](#integration-with-gpu-model-runner)
5. [Memory Management and Graph Pooling](#memory-management-and-graph-pooling)
6. [Configuration Options](#configuration-options)
7. [Conclusion](#conclusion)

## Introduction

CUDA Graphs in vLLM represent a critical performance optimization technique that captures sequences of CUDA operations into a single graph to reduce kernel launch overhead. This optimization is particularly important in large language model inference where minimizing latency and maximizing throughput are essential. By capturing repetitive computation patterns into static graphs, vLLM can significantly reduce the CPU overhead associated with launching individual CUDA kernels, leading to improved performance and more efficient GPU utilization.

The implementation leverages PyTorch's CUDA Graph API to capture and replay computation graphs, with vLLM providing an abstraction layer that manages the complexity of graph capture, storage, and execution. This documentation provides a comprehensive analysis of the CUDA Graphs implementation in vLLM, covering the domain model, capture and replay mechanisms, integration with the GPU model runner, memory management strategies, and configuration options.

**Section sources**
- [cuda_graph.py](file://vllm/compilation/cuda_graph.py#L1-L209)
- [compilation.py](file://vllm/config/compilation.py#L1-L200)

## Domain Model

The CUDA Graphs implementation in vLLM is built around several key components that work together to capture, store, and execute computation graphs. The domain model includes the CudaGraphRunner, CudaGraphPool, and related classes that manage the graph lifecycle.

The `CUDAGraphWrapper` class serves as the primary interface for CUDA Graph functionality, wrapping a callable to add graph capturing and replaying capabilities. It maintains a cache of captured graphs indexed by batch descriptors, allowing for efficient lookup and replay of previously captured computation patterns. The wrapper is designed to be transparent to the underlying callable, providing attribute access to the wrapped object through `__getattr__`.

```mermaid
classDiagram
class CUDAGraphWrapper {
+runnable : Callable
+vllm_config : VllmConfig
+runtime_mode : CUDAGraphMode
+graph_pool : Any
+concrete_cudagraph_entries : dict[BatchDescriptor, CUDAGraphEntry]
+__call__(*args, **kwargs) : Any
+unwrap() : Callable
}
class CUDAGraphEntry {
+batch_descriptor : BatchDescriptor
+cudagraph : torch.cuda.CUDAGraph | None
+output : Any | None
+input_addresses : list[int] | None
}
class CUDAGraphOptions {
+debug_log_enable : bool
+gc_disable : bool
+weak_ref_output : bool
}
class CudagraphDispatcher {
+vllm_config : VllmConfig
+cudagraph_keys : dict[CUDAGraphMode, set[BatchDescriptor]]
+keys_initialized : bool
+initialize_cudagraph_keys(cudagraph_mode : CUDAGraphMode, uniform_decode_query_len : int) : void
+dispatch(num_tokens : int, uniform_decode : bool, has_lora : bool, use_cascade_attn : bool) : tuple[CUDAGraphMode, BatchDescriptor]
}
CUDAGraphWrapper --> CUDAGraphEntry : "contains"
CUDAGraphWrapper --> CUDAGraphOptions : "uses"
CudagraphDispatcher --> BatchDescriptor : "manages"
CUDAGraphWrapper --> CudagraphDispatcher : "uses for dispatching"
```

**Diagram sources **
- [cuda_graph.py](file://vllm/compilation/cuda_graph.py#L43-L209)
- [cudagraph_dispatcher.py](file://vllm/v1/cudagraph_dispatcher.py#L12-L184)

**Section sources**
- [cuda_graph.py](file://vllm/compilation/cuda_graph.py#L25-L209)
- [cudagraph_dispatcher.py](file://vllm/v1/cudagraph_dispatcher.py#L12-L184)

## Graph Capture and Replay Mechanism

The graph capture and replay mechanism in vLLM follows a well-defined workflow that ensures efficient execution of captured computation patterns. The process begins with the initialization of a `CUDAGraphWrapper` with a specific runtime mode (FULL or PIECEWISE) and a reference to the callable to be wrapped.

During execution, the wrapper receives a runtime mode and batch descriptor from the forward context, which it uses to determine whether to capture a new graph or replay an existing one. If the runtime mode matches the wrapper's mode and a graph exists for the given batch descriptor, the wrapper replays the graph. Otherwise, it captures a new graph by executing the wrapped callable within a CUDA graph context.

```mermaid
sequenceDiagram
participant Context as ForwardContext
participant Wrapper as CUDAGraphWrapper
participant Graph as CUDA Graph
participant Model as Model
Context->>Wrapper : Execute with batch_descriptor
Wrapper->>Wrapper : Check runtime_mode match
alt Mode matches and graph exists
Wrapper->>Graph : replay()
Graph-->>Wrapper : Execution result
else Need to capture
Wrapper->>Wrapper : Validate capture allowed
Wrapper->>Wrapper : Prepare input addresses
Wrapper->>Graph : torch.cuda.graph()
Graph->>Model : Execute wrapped callable
Model-->>Graph : Output
Graph-->>Wrapper : Captured graph
Wrapper->>Wrapper : Cache graph and output
end
Wrapper-->>Context : Return output
```

**Diagram sources **
- [cuda_graph.py](file://vllm/compilation/cuda_graph.py#L111-L208)

**Section sources**
- [cuda_graph.py](file://vllm/compilation/cuda_graph.py#L111-L208)
- [monitor.py](file://vllm/compilation/monitor.py#L48-L62)

## Integration with GPU Model Runner

The integration of CUDA Graphs with the GPU model runner in vLLM is designed to optimize the model execution loop by capturing and replaying computation patterns for different sequence lengths. The `CudaGraphManager` class plays a central role in this integration, managing the capture and execution of graphs for various batch sizes.

The manager determines which graph sizes to capture based on the configuration and model constraints, creating a mapping from sequence lengths to appropriate graph sizes. During inference, the manager selects the appropriate graph based on the current batch size and sequence length, ensuring that the most efficient computation pattern is used.

```mermaid
flowchart TD
A[Model Execution] --> B{Needs Capture?}
B --> |Yes| C[Capture Graphs]
B --> |No| D{Run with CUDA Graphs?}
C --> E[Initialize CudaGraphManager]
E --> F[Capture Graphs for Different Sizes]
F --> G[Store Graphs in Manager]
D --> |Yes| H[Select Appropriate Graph]
H --> I[Replay Graph]
I --> J[Return Results]
D --> |No| K[Execute Eagerly]
K --> J
```

**Diagram sources **
- [cudagraph_utils.py](file://vllm/v1/worker/gpu/cudagraph_utils.py#L24-L154)

**Section sources**
- [cudagraph_utils.py](file://vllm/v1/worker/gpu/cudagraph_utils.py#L24-L154)
- [eagle_cudagraph.py](file://vllm/v1/worker/gpu/spec_decode/eagle_cudagraph.py#L21-L116)

## Memory Management and Graph Pooling

Memory management and graph pooling are critical aspects of the CUDA Graphs implementation in vLLM, addressing the potential memory overhead from maintaining multiple graph instances. The system employs several strategies to optimize memory usage and prevent excessive memory consumption.

The `CudaGraphPool` concept is implemented through PyTorch's graph pool handle, which allows for efficient memory allocation and reuse across graph instances. The global graph pool is accessed through the platform interface, ensuring consistent behavior across different execution environments.

```mermaid
classDiagram
class CudaGraphManager {
+vllm_config : VllmConfig
+device : torch.device
+cudagraph_sizes : dict[int, int]
+graphs : dict[int, torch.cuda.CUDAGraph]
+pool : torch.cuda.graph_pool_handle
+hidden_states : torch.Tensor | None
+capture_graph(num_tokens : int, model : nn.Module, ...) : None
+run(num_tokens : int) : torch.Tensor
}
class EagleCudaGraphManager {
+vllm_config : VllmConfig
+device : torch.device
+cudagraph_sizes : dict[int, int]
+graphs : dict[int, torch.cuda.CUDAGraph]
+pool : torch.cuda.graph_pool_handle
+capture_graph(num_tokens : int, generate_fn : Callable, ...) : None
+run(num_tokens : int) : None
}
class CudaGraphPool {
+get_global_graph_pool() : Any
+graph_pool_handle() : Any
}
CudaGraphManager --> CudaGraphPool : "uses"
EagleCudaGraphManager --> CudaGraphPool : "uses"
CudaGraphManager --> torch.cuda.CUDAGraph : "creates"
EagleCudaGraphManager --> torch.cuda.CUDAGraph : "creates"
```

**Diagram sources **
- [cudagraph_utils.py](file://vllm/v1/worker/gpu/cudagraph_utils.py#L24-L58)
- [eagle_cudagraph.py](file://vllm/v1/worker/gpu/spec_decode/eagle_cudagraph.py#L21-L58)
- [pynccl_allocator.py](file://vllm/distributed/device_communicators/pynccl_allocator.py#L62-L191)

**Section sources**
- [cudagraph_utils.py](file://vllm/v1/worker/gpu/cudagraph_utils.py#L52-L58)
- [pynccl_allocator.py](file://vllm/distributed/device_communicators/pynccl_allocator.py#L62-L191)

## Configuration Options

vLLM provides several configuration options for enabling and tuning CUDA Graphs, allowing users to optimize performance based on their specific use cases and hardware constraints. These options are defined in the `CompilationConfig` class and can be set through the vLLM configuration system.

The primary configuration options include:
- `cudagraph_mode`: Controls the CUDA Graph mode (NONE, PIECEWISE, FULL, FULL_DECODE_ONLY, FULL_AND_PIECEWISE)
- `cudagraph_capture_sizes`: Specifies the batch sizes for which graphs should be captured
- `cudagraph_specialize_lora`: Determines whether to specialize graphs for LoRA activation cases
- `max_cudagraph_capture_size`: Sets the maximum size for graph capture

```mermaid
erDiagram
COMPILATION_CONFIG {
string cudagraph_mode
list[int] cudagraph_capture_sizes
bool cudagraph_specialize_lora
int max_cudagraph_capture_size
}
CUDAGRAPH_MODE {
int NONE
int PIECEWISE
int FULL
int FULL_DECODE_ONLY
int FULL_AND_PIECEWISE
}
COMPILATION_CONFIG ||--o{ CUDAGRAPH_MODE : "defines"
```

**Diagram sources **
- [compilation.py](file://vllm/config/compilation.py#L47-L88)

**Section sources**
- [compilation.py](file://vllm/config/compilation.py#L47-L88)
- [vllm.py](file://vllm/config/vllm.py#L670-L692)

## Conclusion

The CUDA Graphs implementation in vLLM provides a powerful performance optimization mechanism that significantly reduces kernel launch overhead in large language model inference. By capturing sequences of CUDA operations into static graphs, vLLM can achieve substantial performance improvements, particularly in scenarios with repetitive computation patterns.

The system's design balances performance optimization with memory efficiency through careful graph pooling and memory management strategies. The integration with the GPU model runner ensures that graphs are captured and replayed appropriately for different sequence lengths, while the configuration options provide flexibility for tuning the optimization to specific use cases.

Future work could explore additional optimization opportunities, such as dynamic graph capture for previously unseen batch sizes, more sophisticated graph pooling strategies, and enhanced support for mixed precision computations within captured graphs.

**Section sources**
- [cuda_graph.py](file://vllm/compilation/cuda_graph.py#L1-L209)
- [cudagraph_utils.py](file://vllm/v1/worker/gpu/cudagraph_utils.py#L1-L260)
- [compilation.py](file://vllm/config/compilation.py#L1-L200)