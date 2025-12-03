# Version History

<cite>
**Referenced Files in This Document**   
- [RELEASE.md](file://RELEASE.md)
- [README.md](file://README.md)
- [vllm/version.py](file://vllm/version.py)
- [setup.py](file://setup.py)
- [pyproject.toml](file://pyproject.toml)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Release Versioning and Cadence](#release-versioning-and-cadence)
3. [Major Releases and Milestones](#major-releases-and-milestones)
4. [Patch Release Series](#patch-release-series)
5. [Community Contributions and Impact](#community-contributions-and-impact)
6. [Performance Benchmarks and Optimizations](#performance-benchmarks-and-optimizations)
7. [Deprecation and Breaking Changes](#deprecation-and-breaking-changes)
8. [Future Roadmap](#future-roadmap)

## Introduction

vLLM has established itself as a leading high-throughput and memory-efficient inference and serving engine for large language models (LLMs). Since its initial release in June 2023, vLLM has evolved rapidly through a consistent release cadence, introducing groundbreaking features like PagedAttention, speculative decoding, and comprehensive quantization support. The project has transitioned from academic research at UC Berkeley's Sky Computing Lab to a community-driven open-source project under the PyTorch Foundation, reflecting its growing importance in the AI ecosystem. This document chronicles the evolution of vLLM through its release history, highlighting key technical advancements, performance improvements, and community contributions that have shaped its development.

**Section sources**
- [README.md](file://README.md#L67-L70)
- [RELEASE.md](file://RELEASE.md#L3-L4)

## Release Versioning and Cadence

vLLM employs a "right-shifted" versioning scheme that differs from traditional semantic versioning. In this model, patch releases (e.g., 0.7.1, 0.7.2) are released bi-weekly and include both new features and bug fixes, rather than being limited to backward-compatible bug fixes as in standard semver. This accelerated release cadence enables rapid delivery of innovations to users while maintaining stability through rigorous testing protocols.

The versioning scheme includes four components:
- **Major**: Reserved for major architectural milestones and incompatible API changes
- **Minor**: For major feature additions
- **Patch**: For features and backward-compatible bug fixes released bi-weekly
- **Post1**: For critical backward-compatible bug fixes released 1-3 days after patch releases

The project maintains a predictable release schedule with patch releases every two weeks, as evidenced by the documented cadence for 2025 which shows sequential patch versions from 0.7.0 through 0.7.23 across the year. Each release is built from a dedicated release branch, with cherry-picks limited to regression fixes, critical fixes for severe issues, fixes to new features, documentation improvements, and release branch-specific changes—explicitly excluding new feature work to ensure stability.

**Section sources**
- [RELEASE.md](file://RELEASE.md#L7-L16)
- [RELEASE.md](file://RELEASE.md#L19-L32)

## Major Releases and Milestones

### v0.1.0 - Initial Release (June 2023)
The initial release of vLLM introduced the revolutionary PagedAttention mechanism, which dramatically improved memory efficiency in LLM serving by applying the concept of virtual memory and paging to attention keys and values. This innovation enabled vLLM to achieve state-of-the-art serving throughput while reducing memory waste from memory fragmentation by up to 8x compared to Hugging Face Transformers. The release marked vLLM as a production-ready solution, with FastChat-vLLM integration powering the LMSYS Chatbot Arena since April 2023.

### v0.5.0 - Quantization Expansion (Late 2023)
This release significantly expanded vLLM's quantization capabilities, adding support for GPTQ, AWQ, and INT4/INT8 quantization methods. These additions made vLLM more accessible to organizations with limited GPU resources by enabling efficient inference on smaller hardware. The release also included optimizations for continuous batching and CUDA/HIP graph execution, further enhancing throughput performance.

### v0.8.0 - Llama 3.1 Support and FP8 Quantization (July 2024)
A major milestone that added official support for Meta's Llama 3.1 with FP8 quantization and pipeline parallelism. This release demonstrated vLLM's commitment to supporting cutting-edge models from leading AI organizations. The FP8 quantization support provided significant memory savings while maintaining model accuracy, enabling larger models to be served efficiently.

### v1.0 Alpha - Architectural Overhaul (January 2025)
The alpha release of vLLM V1 represented a major architectural upgrade with a 1.7x speedup over previous versions. Key improvements included a clean codebase, optimized execution loop, zero-overhead prefix caching, and enhanced multimodal support. This release laid the foundation for future performance improvements and feature expansion, representing a significant evolution in the project's architecture.

### PyTorch Foundation Integration (May 2025)
vLLM's acceptance as a hosted project under the PyTorch Foundation marked a significant milestone in its development, providing increased resources, visibility, and integration with the broader PyTorch ecosystem. This move solidified vLLM's position as a critical component of the modern AI infrastructure stack.

**Section sources**
- [README.md](file://README.md#L59-L60)
- [README.md](file://README.md#L77-L78)
- [README.md](file://README.md#L53-L54)
- [README.md](file://README.md#L33-L34)
- [README.md](file://README.md#L32-L33)

## Patch Release Series

### 0.7.x Series (2025)
The 0.7.x series represents vLLM's current release train with bi-weekly patch releases from 0.7.0 through 0.7.23 planned throughout 2025. This series focuses on incremental improvements, bug fixes, and performance optimizations while maintaining backward compatibility. The predictable cadence allows users to plan upgrades and benefit from continuous enhancements without disruptive changes.

### 0.9.x Series
The 0.9.x series included significant performance improvements and new features, with documented comparisons between versions like v0.9.1 and v0.9.2 showing measurable throughput gains. These releases refined the speculative decoding implementation and improved support for mixture-of-experts (MoE) models like Mixtral.

### 0.8.x Series
The 0.8.x series introduced several important changes, including a shift in the source of default sampling parameters from vLLM's neutral defaults to the model creator's generation_config.json, aligning more closely with Hugging Face conventions. This series also added SHA256 support for enhanced security in prefix caching, available since v0.8.3 with a performance impact of approximately 100-200ns per token.

### 0.5.x Series
The 0.5.x series addressed critical stability issues, including a bug in versions 0.5.2, 0.5.3, and 0.5.3.post1 related to the zmq library that could cause vLLM to hang depending on machine configuration. This was resolved in subsequent releases, demonstrating the project's commitment to reliability.

**Section sources**
- [RELEASE.md](file://RELEASE.md#L21-L32)
- [docs/usage/troubleshooting.md](file://docs/usage/troubleshooting.md#L26)
- [docs/usage/troubleshooting.md](file://docs/usage/troubleshooting.md#L324)
- [docs/design/prefix_caching.md](file://docs/design/prefix_caching.md#L25)

## Community Contributions and Impact

vLLM's development has been significantly shaped by community contributions from both academia and industry. The project has received support and contributions from major technology organizations including NVIDIA, AMD, Google Cloud, AWS, Meta, and IBM, as evidenced by the numerous meetups hosted with these partners. These collaborations have directly influenced vLLM's roadmap, with features like FP8 quantization and pipeline parallelism developed in partnership with Meta for Llama 3.1 support.

The project's compute resources for development and testing are supported by a diverse group of organizations including Alibaba Cloud, AMD, Anyscale, AWS, Google Cloud, Intel, NVIDIA, and others, reflecting broad industry recognition of vLLM's importance. Financial support from a16z, Dropbox, and Sequoia Capital has enabled the project to maintain its open-source development and research efforts.

Community engagement is facilitated through multiple channels including a dedicated Slack workspace, user forum, and regular meetups in cities worldwide. The project has hosted events in San Francisco, New York, Zurich, Beijing, Shanghai, Singapore, and other locations, creating a global community of vLLM users and contributors. These events have focused on practical applications of vLLM for inference optimization, distributed inference, and scaling challenges.

**Section sources**
- [README.md](file://README.md#L47-L57)
- [README.md](file://README.md#L127-L157)
- [README.md](file://README.md#L18-L27)

## Performance Benchmarks and Optimizations

vLLM conducts rigorous end-to-end performance validation before each release using the vllm-benchmark workflow on PyTorch CI. This process compares current results against previous releases to verify no performance regressions have occurred. The current benchmark coverage includes key models like Llama3, Llama4, and Mixtral on hardware platforms including NVIDIA H100 and AMD MI300x.

The performance validation process involves:
1. Running benchmarks on the release branch with the RC commit hash
2. Comparing results against the previous release on the vLLM benchmark dashboard
3. Verifying throughput, latency, and memory utilization metrics

Key performance optimizations across releases include:
- **PagedAttention**: Reduced memory fragmentation by up to 8x compared to standard Transformers
- **Speculative decoding**: Accelerated generation by using a draft model to speculate token sequences
- **CUDA/HIP graph optimization**: Improved kernel launch efficiency
- **Continuous batching**: Enhanced throughput by processing multiple requests simultaneously
- **Quantization support**: Enabled efficient inference with GPTQ, AWQ, and FP8 methods

These optimizations have collectively contributed to vLLM's reputation for achieving state-of-the-art serving throughput, making it a preferred choice for production LLM serving.

**Section sources**
- [RELEASE.md](file://RELEASE.md#L58-L90)
- [README.md](file://README.md#L73-L80)

## Deprecation and Breaking Changes

vLLM follows a structured deprecation policy to manage breaking changes while minimizing disruption to users. Features are deprecated with clear removal timelines, typically spanning three minor releases:

1. **Deprecation phase**: The feature is marked as deprecated with a clear removal version
2. **Off-by-default phase**: The feature is disabled by default and throws errors when used
3. **Removal phase**: The feature is completely removed from the codebase

Several features are scheduled for removal in upcoming releases:
- The `--cuda-graph-sizes` flag is deprecated and will be removed in v0.13.0 or v1.0.0
- The `use_v1` parameter in `Platform.get_attn_backend_cls` is deprecated
- The `_Backend` class in `vllm.attention` is deprecated in favor of the registry pattern

The project also removed deprecated API fields in v0.12.0, requiring users to update their code to use the new `structured_outputs` parameter. This structured approach to deprecation allows users to plan migrations and maintain stable production systems while benefiting from ongoing improvements to the codebase.

**Section sources**
- [vllm/engine/arg_utils.py](file://vllm/engine/arg_utils.py#L1094)
- [docs/design/plugin_system.md](file://docs/design/plugin_system.md#L154-L155)
- [docs/features/structured_outputs.md](file://docs/features/structured_outputs.md#L9)

## Future Roadmap

Based on the documented release cadence and community discussions, vLLM's future development will focus on several key areas:

1. **Performance optimization**: Continued improvements to inference speed and memory efficiency, building on the 1.7x speedup achieved in the V1 alpha release
2. **Expanded hardware support**: Enhanced support for diverse hardware platforms including AMD, Intel, and specialized AI accelerators
3. **Multimodal capabilities**: Further development of support for vision-language models and other multimodal architectures
4. **Distributed inference**: Advancements in tensor, pipeline, data, and expert parallelism for large-scale model serving
5. **Developer experience**: Improvements to the OpenAI-compatible API server and tooling for easier integration

The bi-weekly release cadence will continue to deliver incremental improvements while major architectural changes are developed in the V1 branch. The project's integration with the PyTorch Foundation is expected to accelerate development and broaden its impact across the AI community.

**Section sources**
- [RELEASE.md](file://RELEASE.md#L19-L32)
- [README.md](file://README.md#L88-L89)