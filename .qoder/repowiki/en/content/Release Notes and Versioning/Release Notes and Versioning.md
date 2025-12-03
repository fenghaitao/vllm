# Release Notes and Versioning

<cite>
**Referenced Files in This Document**   
- [README.md](file://README.md)
- [RELEASE.md](file://RELEASE.md)
- [version.py](file://vllm/version.py)
- [pyproject.toml](file://pyproject.toml)
- [common.txt](file://requirements/common.txt)
- [cuda.txt](file://requirements/cuda.txt)
- [rocm.txt](file://requirements/rocm.txt)
- [deprecation_policy.md](file://docs/contributing/deprecation_policy.md)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Versioning Strategy](#versioning-strategy)
3. [Release Cadence](#release-cadence)
4. [Version History and Release Notes](#version-history-and-release-notes)
5. [Breaking Changes and Upgrade Guide](#breaking-changes-and-upgrade-guide)
6. [Deprecation Policy](#deprecation-policy)
7. [Compatibility Requirements](#compatibility-requirements)
8. [Common Upgrade Issues and Solutions](#common-upgrade-issues-and-solutions)

## Introduction

vLLM is a high-throughput and memory-efficient inference and serving engine for large language models (LLMs). The project follows a structured release management process to communicate changes, improvements, and breaking changes between versions. This document provides comprehensive information about vLLM's versioning strategy, release cadence, compatibility requirements, and upgrade guidance.

The release process serves as a key milestone for the development team to communicate with the community about newly available features, improvements, and upcoming changes that could affect users. Releases offer a reliable version of the codebase packaged into a binary format accessible via PyPI.

**Section sources**
- [README.md](file://README.md#L67-L100)
- [RELEASE.md](file://RELEASE.md#L3-L4)

## Versioning Strategy

vLLM uses a "right-shifted" versioning scheme that differs from traditional semantic versioning. In this scheme:

- **Major version**: Represents major architectural milestones and incompatible API changes, similar to PyTorch 2.0
- **Minor version**: Introduces major features and is used for significant changes including deprecations and removals
- **Patch version**: Contains features and backwards-compatible bug fixes (unlike semver where patch releases contain only bug fixes)
- **Post release (post1 or patch-1)**: Contains backwards-compatible bug fixes, either explicit or implicit post-release

This versioning approach allows vLLM to deliver new features in patch releases while maintaining a predictable deprecation and removal cycle through minor versions. The project aims to balance continued innovation with respect for users' reliance on existing functionality.

The versioning strategy is particularly important for managing the lifecycle of deprecated features, which follows a structured "deprecation pipeline" spanning multiple minor releases.

**Section sources**
- [RELEASE.md](file://RELEASE.md#L5-L13)
- [deprecation_policy.md](file://docs/contributing/deprecation_policy.md#L14-L19)

## Release Cadence

vLLM follows a regular release cadence to ensure users receive updates and improvements in a predictable manner:

- **Patch releases**: Released bi-weekly (every two weeks)
- **Post releases**: Released 1-3 days after patch releases and use the same branch as the patch release
- Post releases are optional and used when critical fixes need to be made

The release cadence for 2025 includes two patch releases per month, with version numbers incrementing sequentially. For example, January 2025 includes version 0.7.0, February includes 0.7.1, 0.7.2, and 0.7.3, and so on throughout the year.

Each release is built from a dedicated release branch. The release branch cut is performed 1-2 days before the release goes live. For post releases, the previously cut release branch is reused. Release builds are triggered by pushing to an RC tag (e.g., vX.Y.Z-rc1), allowing multiple release candidates to be built and tested for each release.

The final tag (vX.Y.Z) does not trigger a build but is used for release notes and assets. After the branch cut, the team monitors the main branch for any reverts and applies these reverts to the release branch to maintain stability.

**Section sources**
- [RELEASE.md](file://RELEASE.md#L14-L43)

## Version History and Release Notes

vLLM maintains a comprehensive version history with detailed release notes for each version. The release process includes several key components:

### Release Branch Management
Each release is built from a dedicated release branch with specific criteria for cherry-picking changes after the branch cut:

- Regression fixes addressing functional or performance regressions against the most recent release
- Critical fixes for severe issues such as silent incorrectness, backwards compatibility problems, crashes, deadlocks, or large memory leaks
- Fixes to new features introduced in the most recent release
- Documentation improvements
- Release branch specific changes (e.g., version identifiers or CI fixes)

No feature work is allowed for cherry picks after the branch cut to ensure the team has sufficient time to complete thorough testing on a stable code base. All PRs considered for cherry-picks must be merged on trunk, with the only exception being release branch specific changes.

### Performance Validation
Before each release, vLLM performs end-to-end performance validation to ensure no regressions are introduced. This validation uses the vllm-benchmark workflow on PyTorch CI with current coverage including:

- Models: Llama3, Llama4, and Mixtral
- Hardware: NVIDIA H100 and AMD MI300x

The performance validation process involves running benchmarks on the release branch and comparing results against the previous release to verify no performance regressions have occurred.

**Section sources**
- [RELEASE.md](file://RELEASE.md#L44-L90)

## Breaking Changes and Upgrade Guide

When upgrading vLLM, users may encounter breaking changes that require specific actions. The project follows a structured deprecation pipeline to minimize unexpected disruptions.

### Deprecation Pipeline
The deprecation process consists of three clearly defined stages spanning multiple minor releases:

1. **Deprecated (Still On By Default)**: The feature is marked as deprecated with a clear removal version stated in the deprecation warning (e.g., "This will be removed in v0.10.0"). Communication occurs through help strings, log output, API responses, documentation, and release notes.

2. **Deprecated (Off By Default)**: The feature is disabled by default but can still be re-enabled via a CLI flag or environment variable. It throws an error when used without re-enabling, providing a temporary escape hatch while signaling imminent removal.

3. **Removed**: The feature is completely removed from the codebase.

### Example Upgrade Scenario
For example, if a feature is deprecated in v0.9.0:
- v0.9.0: Feature is deprecated with clear removal version listed
- v0.10.0: Feature is off by default, throws an error when used, and can be re-enabled for legacy use
- v0.11.0: Feature is removed

When upgrading, users should:
1. Check the release notes for deprecated features in the target version
2. Update code to remove usage of deprecated APIs, CLI flags, or configuration options
3. Test the application thoroughly before deploying in production
4. Consult the documentation for alternative approaches to replaced functionality

**Section sources**
- [deprecation_policy.md](file://docs/contributing/deprecation_policy.md#L34-L73)
- [deprecation_policy.md](file://docs/contributing/deprecation_policy.md#L64-L73)

## Deprecation Policy

vLLM has a formal deprecation policy to ensure users receive clear and sufficient notice when features are deprecated. The policy applies to:

- CLI flags
- Environment variables
- Configuration files
- APIs in the OpenAI-compatible API server
- Public Python APIs for the vllm library

### Key Guidelines
- **No removals in patch releases**: Removing deprecated features in patch (.Z) releases is disallowed to avoid surprising users
- **Grace period for existing deprecations**: Any feature deprecated before this policy will have its grace period start from the policy's implementation date, not retroactively
- **Documentation is critical**: Every stage of the deprecation pipeline must be clearly documented for users

The deprecation policy is a living document that may evolve as the needs of the project and its users change. Community feedback is welcome and encouraged as the process is refined.

Features are deprecated using appropriate mechanisms such as the `@typing_extensions.deprecated` decorator for Python APIs, deprecation warnings in help strings, and clear communication in release notes and documentation.

**Section sources**
- [deprecation_policy.md](file://docs/contributing/deprecation_policy.md#L21-L88)

## Compatibility Requirements

vLLM supports a wide range of hardware, software, and dependency configurations. Understanding these compatibility requirements is essential for successful deployment and upgrades.

### Python Compatibility
vLLM supports Python versions 3.10 through 3.13, as specified in the pyproject.toml file:
```
requires-python = ">=3.10,<3.14"
```

### PyTorch Compatibility
vLLM has specific PyTorch version requirements that vary by hardware platform:
- **CUDA (NVIDIA GPUs)**: torch==2.9.0, torchaudio==2.9.0, torchvision==0.24.0
- **ROCm (AMD GPUs)**: No specific torch version pinned, allowing flexibility with AMD's ROCm platform

### Hardware Support
vLLM supports multiple hardware platforms:
- NVIDIA GPUs (via CUDA)
- AMD GPUs (via ROCm)
- CPUs (AMD, Intel, PowerPC, Arm)
- TPUs
- Specialized hardware plugins (Intel Gaudi, IBM Spyre, Huawei Ascend)

### Dependency Management
The project uses a modular requirements system with different dependency files for various configurations:
- **common.txt**: Core dependencies used across all platforms
- **cuda.txt**: Dependencies specific to NVIDIA GPUs
- **rocm.txt**: Dependencies specific to AMD GPUs
- **cpu.txt**: Dependencies specific to CPU-only deployments

Key dependencies include:
- transformers >= 4.56.0, < 5
- fastapi[standard] >= 0.115.0
- pydantic >= 2.12.0
- flashinfer-python==0.5.3 (for CUDA)
- conch-triton-kernels==1.2.1 (for ROCm)

### Environment Variables
vLLM uses numerous environment variables to control behavior, with over 100 configurable options. These include settings for:
- Caching and memory management
- Distributed inference
- Performance optimization
- Logging and debugging
- Hardware-specific configurations

**Section sources**
- [pyproject.toml](file://pyproject.toml#L9)
- [common.txt](file://requirements/common.txt#L10)
- [cuda.txt](file://requirements/cuda.txt#L8-L11)
- [rocm.txt](file://requirements/rocm.txt#L9)
- [envs.py](file://vllm/envs.py#L1)

## Common Upgrade Issues and Solutions

When upgrading vLLM, users may encounter several common issues. This section addresses these problems and provides solutions.

### Issue 1: Dependency Conflicts
**Problem**: Conflicting PyTorch versions when upgrading between vLLM versions with different torch requirements.

**Solution**: 
1. Create a clean virtual environment
2. Install vLLM according to the target version's requirements
3. Verify compatibility with other project dependencies
4. Use pip's dependency resolver or conda for complex dependency management

### Issue 2: Deprecated API Usage
**Problem**: Code breaks due to use of deprecated APIs, CLI flags, or configuration options.

**Solution**:
1. Review release notes for the target version
2. Search codebase for deprecated features mentioned in release notes
3. Update code to use recommended alternatives
4. Test thoroughly in a staging environment

### Issue 3: Performance Regressions
**Problem**: Slower inference performance after upgrade.

**Solution**:
1. Compare performance using vLLM's benchmarking tools
2. Check for changes in default configuration values
3. Review environment variable settings that may affect performance
4. Consult the performance validation results in the release notes

### Issue 4: Hardware-Specific Issues
**Problem**: Issues specific to certain hardware platforms (CUDA, ROCm, CPU).

**Solution**:
1. Ensure hardware-specific dependencies are correctly installed
2. Verify compatibility between vLLM version and hardware drivers
3. Check environment variables specific to the hardware platform
4. Consult platform-specific documentation

### General Upgrade Best Practices
1. Always read release notes before upgrading
2. Test upgrades in a non-production environment first
3. Backup configurations and models before upgrading
4. Monitor logs and metrics after deployment
5. Have a rollback plan in case of issues

**Section sources**
- [RELEASE.md](file://RELEASE.md#L58-L90)
- [deprecation_policy.md](file://docs/contributing/deprecation_policy.md#L34-L57)
- [envs.py](file://vllm/envs.py#L1)