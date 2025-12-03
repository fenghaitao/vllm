# Tensor Parallelism

<cite>
**Referenced Files in This Document**   
- [parallel_state.py](file://vllm/distributed/parallel_state.py)
- [linear.py](file://vllm/model_executor/layers/linear.py)
- [communication_op.py](file://vllm/distributed/communication_op.py)
- [pynccl.py](file://vllm/distributed/device_communicators/pynccl.py)
- [custom_all_reduce.py](file://vllm/distributed/device_communicators/custom_all_reduce.py)
- [utils.py](file://vllm/distributed/utils.py)
- [parallel.py](file://vllm/config/parallel.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Tensor Parallelism Fundamentals](#tensor-parallelism-fundamentals)
3. [Model Weight Partitioning](#model-weight-partitioning)
4. [Communication Patterns](#communication-patterns)
5. [Attention and Linear Layer Implementation](#attention-and-linear-layer-implementation)
6. [Parallel State Management](#parallel-state-management)
7. [Configuration and Usage](#configuration-and-usage)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting Common Issues](#troubleshooting-common-issues)
10. [Conclusion](#conclusion)

## Introduction
Tensor parallelism in vLLM enables efficient inference on large language models that exceed the memory capacity of a single GPU by distributing model weights across multiple GPUs within a single layer. This approach allows for the processing of models with billions of parameters by partitioning the computational workload and model parameters across a tensor parallel group. The implementation leverages PyTorch distributed and custom CUDA communicators to manage the communication patterns between GPUs, ensuring efficient data exchange during the forward and backward passes. This document provides a comprehensive overview of the tensor parallelism implementation in vLLM, detailing the mechanisms for model weight partitioning, communication patterns, and the interfaces that manage tensor parallel groups and ranks.

**Section sources**
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L8-L24)
- [linear.py](file://vllm/model_executor/layers/linear.py#L4-L9)

## Tensor Parallelism Fundamentals

Tensor parallelism in vLLM is a technique that partitions the model weights across multiple GPUs, allowing for the inference of models larger than single GPU memory capacity. This is achieved by dividing the model's layers into smaller segments that are distributed across the available GPUs. Each GPU processes a portion of the input data and communicates with other GPUs to exchange necessary information for the computation. The key to this approach is the efficient management of communication patterns between GPUs, which is handled by the custom CUDA communicators and PyTorch distributed. The tensor parallel group is responsible for coordinating the communication between GPUs, ensuring that the model weights are correctly partitioned and that the results are aggregated appropriately.

```mermaid
graph TD
A[Input Data] --> B[GPU 1: Process Segment 1]
A --> C[GPU 2: Process Segment 2]
A --> D[GPU 3: Process Segment 3]
B --> E[All-Reduce Communication]
C --> E
D --> E
E --> F[Aggregated Output]
```

**Diagram sources** 
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L278-L389)
- [communication_op.py](file://vllm/distributed/communication_op.py#L12-L28)

**Section sources**
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L278-L389)
- [communication_op.py](file://vllm/distributed/communication_op.py#L12-L28)

## Model Weight Partitioning

In vLLM, model weights are partitioned across multiple GPUs using a column-parallel approach, where the weight matrix is divided along its second dimension. This is particularly effective for linear layers, where the weight matrix is split into smaller segments that are distributed across the tensor parallel group. The `ColumnParallelLinear` class in vLLM is responsible for managing this partitioning, ensuring that each GPU receives the appropriate segment of the weight matrix. The partitioning is done in such a way that the input to each GPU is replicated, and the output is gathered from all GPUs to form the complete output. This approach allows for efficient computation while minimizing the communication overhead between GPUs.

The partitioning process is managed by the `divide` function in the `utils.py` file, which ensures that the numerator is divisible by the denominator and returns the division value. This function is used to calculate the size of each partition, ensuring that the weight matrix is evenly distributed across the GPUs. The `split_tensor_along_last_dim` function is also used to split a tensor along its last dimension, which is essential for the partitioning of the weight matrix.

```mermaid
classDiagram
class ColumnParallelLinear {
+input_size : int
+output_size : int
+bias : bool
+gather_output : bool
+skip_bias_add : bool
+params_dtype : torch.dtype
+quant_config : QuantizationConfig
+output_sizes : list[int]
+prefix : str
+return_bias : bool
+disable_tp : bool
+__init__(input_size : int, output_size : int, bias : bool, gather_output : bool, skip_bias_add : bool, params_dtype : torch.dtype, quant_config : QuantizationConfig, output_sizes : list[int], prefix : str, return_bias : bool, disable_tp : bool)
+weight_loader(param : Parameter, loaded_weight : torch.Tensor)
+forward(input_ : torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]
+extra_repr() -> str
}
class LinearBase {
+input_size : int
+output_size : int
+skip_bias_add : bool
+params_dtype : torch.dtype
+quant_config : QuantizationConfig
+prefix : str
+return_bias : bool
+disable_tp : bool
+tp_rank : int
+tp_size : int
+__init__(input_size : int, output_size : int, skip_bias_add : bool, params_dtype : torch.dtype, quant_config : QuantizationConfig, prefix : str, return_bias : bool, disable_tp : bool)
+update_param_tp_status()
}
LinearBase <|-- ColumnParallelLinear : "extends"
```

**Diagram sources** 
- [linear.py](file://vllm/model_executor/layers/linear.py#L413-L584)
- [utils.py](file://vllm/distributed/utils.py#L60-L64)

**Section sources**
- [linear.py](file://vllm/model_executor/layers/linear.py#L413-L584)
- [utils.py](file://vllm/distributed/utils.py#L60-L64)

## Communication Patterns

The communication patterns in vLLM's tensor parallelism implementation are designed to minimize the overhead of data exchange between GPUs. The primary communication operations are all-reduce, all-gather, and reduce-scatter, which are used to aggregate and distribute data across the tensor parallel group. The `tensor_model_parallel_all_reduce` function in the `communication_op.py` file is responsible for performing an all-reduce operation on the input tensor across the model parallel group. This operation is essential for aggregating the partial results from each GPU and forming the complete output.

The `tensor_model_parallel_all_gather` function is used to gather the input tensor across the model parallel group, ensuring that each GPU has access to the complete output. This is particularly useful for layers where the output needs to be replicated across all GPUs. The `tensor_model_parallel_reduce_scatter` function is used to reduce and scatter the input tensor across the model parallel group, which is useful for distributing the workload across multiple GPUs.

The custom CUDA communicators in vLLM are optimized for these communication patterns, providing efficient implementations of the all-reduce, all-gather, and reduce-scatter operations. The `PyNcclCommunicator` class in the `pynccl.py` file is responsible for managing the NCCL communicator, which is used to perform the communication operations. The `CustomAllreduce` class in the `custom_all_reduce.py` file provides a custom implementation of the all-reduce operation, which is optimized for the specific hardware and network topology.

```mermaid
sequenceDiagram
participant GPU1 as GPU 1
participant GPU2 as GPU 2
participant GPU3 as GPU 3
participant Communicator as Communicator
GPU1->>Communicator : Send Partial Result
GPU2->>Communicator : Send Partial Result
GPU3->>Communicator : Send Partial Result
Communicator->>GPU1 : All-Reduce Operation
Communicator->>GPU2 : All-Reduce Operation
Communicator->>GPU3 : All-Reduce Operation
GPU1->>GPU1 : Receive Aggregated Result
GPU2->>GPU2 : Receive Aggregated Result
GPU3->>GPU3 : Receive Aggregated Result
```

**Diagram sources** 
- [communication_op.py](file://vllm/distributed/communication_op.py#L12-L28)
- [pynccl.py](file://vllm/distributed/device_communicators/pynccl.py#L150-L181)
- [custom_all_reduce.py](file://vllm/distributed/device_communicators/custom_all_reduce.py#L105-L136)

**Section sources**
- [communication_op.py](file://vllm/distributed/communication_op.py#L12-L28)
- [pynccl.py](file://vllm/distributed/device_communicators/pynccl.py#L150-L181)
- [custom_all_reduce.py](file://vllm/distributed/device_communicators/custom_all_reduce.py#L105-L136)

## Attention and Linear Layer Implementation

The implementation of attention and linear layers in vLLM is designed to support tensor parallelism by partitioning the model weights and managing the communication patterns between GPUs. The `QKVParallelLinear` class in the `linear.py` file is responsible for managing the linear transformation of the query, key, and value vectors in the attention layer. The weight matrix is concatenated along the output dimension and parallelized along the head dimension. When the number of key/value heads is smaller than the number of query heads, the key/value head may be replicated while the query heads are partitioned.

The `ColumnParallelLinear` class is used for the linear layers, where the weight matrix is divided along its second dimension. The `ReplicatedLinear` class is used for layers where the weight matrix is not partitioned, and the same weights are used across all GPUs. The `MergedColumnParallelLinear` class is used for layers where the weight matrix is concatenated along the output dimension, and the different partitions are sharded separately.

The `weight_loader` method in the `QKVParallelLinear` class is responsible for loading the weights from the checkpoint and partitioning them across the GPUs. The `weight_loader_v2` method is used for models where the QKV layers are already fused on disk, and the weights need to be split and loaded into the appropriate partitions.

```mermaid
classDiagram
class QKVParallelLinear {
+hidden_size : int
+head_size : int
+total_num_heads : int
+total_num_kv_heads : int
+bias : bool
+skip_bias_add : bool
+params_dtype : torch.dtype
+quant_config : QuantizationConfig
+prefix : str
+return_bias : bool
+disable_tp : bool
+__init__(hidden_size : int, head_size : int, total_num_heads : int, total_num_kv_heads : int, bias : bool, skip_bias_add : bool, params_dtype : torch.dtype, quant_config : QuantizationConfig, prefix : str, return_bias : bool, disable_tp : bool)
+_get_shard_offset_mapping(loaded_shard_id : str)
+_get_shard_size_mapping(loaded_shard_id : str)
+_load_fused_module_from_checkpoint(param : BasevLLMParameter, loaded_weight : torch.Tensor)
+weight_loader_v2(param : BasevLLMParameter, loaded_weight : torch.Tensor, loaded_shard_id : str | None)
+weight_loader(param : Parameter, loaded_weight : torch.Tensor, loaded_shard_id : str | None)
}
class ColumnParallelLinear {
+input_size : int
+output_size : int
+bias : bool
+gather_output : bool
+skip_bias_add : bool
+params_dtype : torch.dtype
+quant_config : QuantizationConfig
+output_sizes : list[int]
+prefix : str
+return_bias : bool
+disable_tp : bool
+__init__(input_size : int, output_size : int, bias : bool, gather_output : bool, skip_bias_add : bool, params_dtype : torch.dtype, quant_config : QuantizationConfig, output_sizes : list[int], prefix : str, return_bias : bool, disable_tp : bool)
+weight_loader(param : Parameter, loaded_weight : torch.Tensor)
+forward(input_ : torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]
+extra_repr() -> str
}
class ReplicatedLinear {
+input_size : int
+output_size : int
+bias : bool
+skip_bias_add : bool
+params_dtype : torch.dtype
+quant_config : QuantizationConfig
+prefix : str
+return_bias : bool
+disable_tp : bool
+__init__(input_size : int, output_size : int, bias : bool, skip_bias_add : bool, params_dtype : torch.dtype, quant_config : QuantizationConfig, prefix : str, return_bias : bool, disable_tp : bool)
+weight_loader(param : Parameter, loaded_weight : torch.Tensor)
+forward(x : torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]
+extra_repr() -> str
}
class MergedColumnParallelLinear {
+input_size : int
+output_sizes : list[int]
+bias : bool
+gather_output : bool
+skip_bias_add : bool
+params_dtype : torch.dtype
+quant_config : QuantizationConfig
+prefix : str
+return_bias : bool
+disable_tp : bool
+__init__(input_size : int, output_sizes : list[int], bias : bool, gather_output : bool, skip_bias_add : bool, params_dtype : torch.dtype, quant_config : QuantizationConfig, prefix : str, return_bias : bool, disable_tp : bool)
+weight_loader(param : Parameter, loaded_weight : torch.Tensor, loaded_shard_id : int | None)
+_load_fused_module_from_checkpoint(param : BasevLLMParameter, loaded_weight : torch.Tensor)
+forward(input_ : torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]
+extra_repr() -> str
}
ColumnParallelLinear <|-- QKVParallelLinear : "extends"
LinearBase <|-- ColumnParallelLinear : "extends"
LinearBase <|-- ReplicatedLinear : "extends"
ColumnParallelLinear <|-- MergedColumnParallelLinear : "extends"
```

**Diagram sources** 
- [linear.py](file://vllm/model_executor/layers/linear.py#L866-L1237)
- [linear.py](file://vllm/model_executor/layers/linear.py#L413-L584)
- [linear.py](file://vllm/model_executor/layers/linear.py#L296-L341)
- [linear.py](file://vllm/model_executor/layers/linear.py#L586-L642)

**Section sources**
- [linear.py](file://vllm/model_executor/layers/linear.py#L866-L1237)
- [linear.py](file://vllm/model_executor/layers/linear.py#L413-L584)
- [linear.py](file://vllm/model_executor/layers/linear.py#L296-L341)
- [linear.py](file://vllm/model_executor/layers/linear.py#L586-L642)

## Parallel State Management

The management of tensor parallel groups and ranks in vLLM is handled by the `GroupCoordinator` class in the `parallel_state.py` file. This class is responsible for creating and managing the process groups for CPU and device communication, as well as the device communicator for GPU communication. The `GroupCoordinator` class also manages the tensor parallel group, which is used to coordinate the communication between GPUs.

The `ensure_model_parallel_initialized` function in the `parallel_state.py` file is responsible for initializing the model parallel groups, ensuring that the tensor parallel group is correctly set up. The `destroy_model_parallel` function is used to destroy the model parallel groups when they are no longer needed. The `get_tp_group` function is used to retrieve the tensor parallel group, which is used for communication operations.

The `GroupCoordinator` class also provides methods for performing communication operations, such as `all_reduce`, `all_gather`, and `reduce_scatter`. These methods are used to perform the communication operations between GPUs, ensuring that the model weights are correctly partitioned and that the results are aggregated appropriately.

```mermaid
classDiagram
class GroupCoordinator {
+rank : int
+ranks : list[int]
+world_size : int
+local_rank : int
+rank_in_group : int
+cpu_group : ProcessGroup
+device_group : ProcessGroup
+device_communicator : DeviceCommunicatorBase | None
+mq_broadcaster : Any | None
+__init__(group_ranks : list[list[int]], local_rank : int, torch_distributed_backend : str | Backend, use_device_communicator : bool, use_message_queue_broadcaster : bool, group_name : str | None)
+create_mq_broadcaster(writer_rank : int, external_writer_handle : Any, blocking : bool)
+create_single_reader_mq_broadcasters(reader_rank_in_group : int, blocking : bool)
+first_rank : int
+last_rank : int
+is_first_rank : bool
+is_last_rank : bool
+next_rank : int
+prev_rank : int
+graph_capture(graph_capture_context : GraphCaptureContext | None)
+all_reduce(input_ : torch.Tensor) -> torch.Tensor
+_all_reduce_out_place(input_ : torch.Tensor) -> torch.Tensor
+all_gather(input_ : torch.Tensor, dim : int) -> torch.Tensor
+_all_gather_out_place(input_ : torch.Tensor, dim : int) -> torch.Tensor
+all_gatherv(input_ : torch.Tensor | list[torch.Tensor], dim : int, sizes : list[int] | None) -> torch.Tensor
+reduce_scatter(input_ : torch.Tensor, dim : int) -> torch.Tensor
+reduce_scatterv(input_ : torch.Tensor, dim : int, sizes : list[int] | None) -> torch.Tensor
+_reduce_scatter_out_place(input_ : torch.Tensor, dim : int) -> torch.Tensor
+gather(input_ : torch.Tensor, dst : int, dim : int) -> torch.Tensor | None
+broadcast(input_ : torch.Tensor, src : int)
+broadcast_object(obj : Any | None, src : int)
+broadcast_object_list(obj_list : list[Any], src : int, group : ProcessGroup | None)
+send_object(obj : Any, dst : int) -> None
+recv_object(src : int) -> Any
+broadcast_tensor_dict(tensor_dict : dict[str, torch.Tensor | Any] | None, src : int, group : ProcessGroup | None, metadata_group : ProcessGroup | None) -> dict[str, torch.Tensor | Any] | None
+send_tensor_dict(tensor_dict : dict[str, torch.Tensor | Any], dst : int | None, all_gather_group : GroupCoordinator | None, all_gather_tensors : dict[str, bool] | None) -> dict[str, torch.Tensor | Any] | None
+barrier()
+send(tensor : torch.Tensor, dst : int | None) -> None
+recv(size : torch.Size, dtype : torch.dtype, src : int | None) -> torch.Tensor
+destroy()
+prepare_communication_buffer_for_model(model : torch.nn.Module)
+dispatch(hidden_states : torch.Tensor, router_logits : torch.Tensor, is_sequence_parallel : bool) -> tuple[torch.Tensor, torch.Tensor]
}
class GroupCoordinator
```

**Diagram sources** 
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L278-L1011)

**Section sources**
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L278-L1011)

## Configuration and Usage

The configuration of tensor parallelism in vLLM is managed through the `tensor_parallel_size` parameter, which specifies the number of tensor parallel groups. This parameter is used to determine the number of GPUs that will be used for tensor parallelism and to partition the model weights accordingly. The `tensor_parallel_size` parameter can be set via command-line arguments or API parameters, allowing for flexible configuration of the tensor parallelism setup.

The `parallel.py` file in the `vllm/config` directory contains the configuration for the distributed execution, including the `tensor_parallel_size` parameter. The `ParallelConfig` class in this file is responsible for managing the configuration of the distributed execution, including the tensor parallel size, pipeline parallel size, and data parallel size. The `compute_hash` method in the `ParallelConfig` class is used to compute a hash that uniquely identifies the configuration, which is used for validation and debugging purposes.

The usage of tensor parallelism in vLLM is straightforward, with the `tensor_parallel_size` parameter being the primary configuration option. When setting up a model for inference, the `tensor_parallel_size` parameter should be set to the desired number of tensor parallel groups. The model weights will be automatically partitioned across the available GPUs, and the communication patterns will be managed by the custom CUDA communicators and PyTorch distributed.

```mermaid
flowchart TD
A[Set tensor_parallel_size] --> B[Initialize Model Parallel Groups]
B --> C[Partition Model Weights]
C --> D[Set Up Communication Patterns]
D --> E[Perform Inference]
E --> F[Aggregate Results]
F --> G[Return Output]
```

**Diagram sources** 
- [parallel.py](file://vllm/config/parallel.py#L74-L81)
- [parallel.py](file://vllm/config/parallel.py#L504-L508)

**Section sources**
- [parallel.py](file://vllm/config/parallel.py#L74-L81)
- [parallel.py](file://vllm/config/parallel.py#L504-L508)

## Performance Considerations

The performance of tensor parallelism in vLLM is influenced by several factors, including the communication overhead, memory imbalances, and the efficiency of the custom CUDA communicators. The communication overhead is primarily determined by the bandwidth and latency of the interconnect between GPUs, with high-bandwidth, low-latency interconnects such as NVLink providing better performance. The memory imbalances can occur when the model weights are not evenly distributed across the GPUs, leading to some GPUs being underutilized while others are overloaded.

To address these issues, vLLM provides several optimizations, including the use of optimized all-reduce operations and proper GPU topology configuration. The `CustomAllreduce` class in the `custom_all_reduce.py` file provides a custom implementation of the all-reduce operation, which is optimized for the specific hardware and network topology. The `PyNcclCommunicator` class in the `pynccl.py` file is responsible for managing the NCCL communicator, which is used to perform the communication operations.

The proper configuration of the GPU topology is also essential for achieving optimal performance. The `tensor_parallel_size` parameter should be set to a value that is a multiple of the number of GPUs in each node, ensuring that the model weights are evenly distributed across the GPUs. The use of high-bandwidth, low-latency interconnects such as NVLink is also recommended to minimize the communication overhead.

```mermaid
flowchart TD
A[High Communication Overhead] --> B[Optimize All-Reduce Operations]
B --> C[Use Custom All-Reduce]
C --> D[Minimize Communication Latency]
D --> E[Improve Performance]
F[Memory Imbalances] --> G[Evenly Distribute Model Weights]
G --> H[Use Proper GPU Topology]
H --> I[Ensure Even Load Distribution]
I --> J[Improve Performance]
```

**Diagram sources** 
- [custom_all_reduce.py](file://vllm/distributed/device_communicators/custom_all_reduce.py#L105-L136)
- [pynccl.py](file://vllm/distributed/device_communicators/pynccl.py#L150-L181)

**Section sources**
- [custom_all_reduce.py](file://vllm/distributed/device_communicators/custom_all_reduce.py#L105-L136)
- [pynccl.py](file://vllm/distributed/device_communicators/pynccl.py#L150-L181)

## Troubleshooting Common Issues

Common issues with tensor parallelism in vLLM include communication overhead, memory imbalances, and configuration errors. The communication overhead can be minimized by using optimized all-reduce operations and proper GPU topology configuration. The memory imbalances can be addressed by ensuring that the model weights are evenly distributed across the GPUs and that the GPU topology is properly configured.

Configuration errors can occur when the `tensor_parallel_size` parameter is not set correctly, leading to issues with the initialization of the model parallel groups. The `ensure_model_parallel_initialized` function in the `parallel_state.py` file should be used to ensure that the model parallel groups are correctly initialized. The `destroy_model_parallel` function should be used to destroy the model parallel groups when they are no longer needed.

To troubleshoot these issues, it is recommended to use the `compute_hash` method in the `ParallelConfig` class to compute a hash that uniquely identifies the configuration. This hash can be used for validation and debugging purposes, ensuring that the configuration is correct and that the model parallel groups are properly initialized.

**Section sources**
- [parallel_state.py](file://vllm/distributed/parallel_state.py#L37-L389)
- [parallel.py](file://vllm/config/parallel.py#L445-L490)

## Conclusion
Tensor parallelism in vLLM is a powerful technique for enabling inference on large language models that exceed the memory capacity of a single GPU. By partitioning the model weights across multiple GPUs and managing the communication patterns between them, vLLM can efficiently process models with billions of parameters. The implementation leverages PyTorch distributed and custom CUDA communicators to ensure efficient data exchange, with optimizations for communication overhead and memory imbalances. The configuration and usage of tensor parallelism are straightforward, with the `tensor_parallel_size` parameter being the primary configuration option. With proper configuration and optimization, tensor parallelism in vLLM can provide significant performance improvements for large-scale inference tasks.