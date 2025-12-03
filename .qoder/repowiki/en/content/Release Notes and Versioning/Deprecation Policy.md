# Deprecation Policy

<cite>
**Referenced Files in This Document**   
- [func_utils.py](file://vllm/utils/func_utils.py)
- [selector.py](file://vllm/attention/selector.py)
- [datasets.py](file://vllm/benchmarks/datasets.py)
- [logger.py](file://vllm/logger.py)
- [observability.py](file://vllm/config/observability.py)
- [version.py](file://vllm/version.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Deprecation Process and Timeline](#deprecation-process-and-timeline)
3. [Deprecation Mechanisms](#deprecation-mechanisms)
4. [Criteria for Deprecating Features](#criteria-for-deprecating-features)
5. [Communication of Deprecation Notices](#communication-of-deprecation-notices)
6. [User Guidance for Migration](#user-guidance-for-migration)
7. [Examples of Past Deprecation Cycles](#examples-of-past-deprecation-cycles)
8. [Balancing Innovation and Backward Compatibility](#balancing-innovation-and-backward-compatibility)
9. [Conclusion](#conclusion)

## Introduction
vLLM maintains a structured deprecation policy to ensure smooth transitions when phasing out features and APIs. This policy balances the need for innovation with the importance of backward compatibility, providing users with clear timelines, warning mechanisms, and migration paths. The deprecation process is designed to minimize disruption while allowing the project to evolve and incorporate new technologies and architectural improvements.

## Deprecation Process and Timeline
The vLLM deprecation process follows a standardized timeline to ensure users have adequate time to adapt to changes. When a feature or API is marked for deprecation, it typically follows a three-phase approach:

1. **Announcement Phase**: The feature is marked as deprecated with clear warnings in logs, documentation, and runtime messages. During this phase, the feature remains fully functional.
2. **Transition Phase**: The deprecated feature continues to work but with prominent warnings. Alternative approaches or replacement features are provided and documented.
3. **Removal Phase**: After a sufficient grace period (typically spanning multiple releases), the deprecated feature is removed from the codebase.

The exact timeline varies based on the impact of the change, but major deprecations are typically announced several releases in advance, with removal occurring in a subsequent major version. For example, the `use_v1` parameter deprecation in attention backends includes a warning that it will be removed in "v0.13.0 or v1.0.0, whichever is soonest."

**Section sources**
- [selector.py](file://vllm/attention/selector.py#L160-L164)
- [version.py](file://vllm/version.py#L15-L32)

## Deprecation Mechanisms
vLLM employs several technical mechanisms to implement deprecation warnings and manage the transition process:

### Decorator-Based Deprecation
The primary mechanism for deprecating function parameters is through specialized decorators in `func_utils.py`. These decorators provide a clean, consistent way to mark deprecated arguments:

```python
def deprecate_args(
    start_index: int,
    is_deprecated: bool | Callable[[], bool] = True,
    additional_message: str | None = None,
) -> Callable[[F], F]:
    # Implementation details
    pass

def deprecate_kwargs(
    *kws: str,
    is_deprecated: bool | Callable[[], bool] = True,
    additional_message: str | None = None,
) -> Callable[[F], F]:
    # Implementation details
    pass
```

These decorators can be applied to functions to automatically generate deprecation warnings when specific arguments are used. The `is_deprecated` parameter can be a boolean or a callable, allowing for dynamic deprecation based on runtime conditions.

### Class-Level Deprecation
For entire classes or modules, vLLM uses the `@deprecated` decorator from `typing_extensions`. For example, the `SonnetDataset` class is marked as deprecated:

```python
@deprecated(
    "SonnetDataset is deprecated and will be removed in a future version.",
)
class SonnetDataset(BenchmarkDataset):
    # Class implementation
    pass
```

This approach provides a clear indication that the entire class should no longer be used and will be removed in future versions.

### Runtime Environment Deprecation
Some deprecations are tied to environment variables or configuration settings. For instance, the `_VLLM_V1` suffix in the `VLLM_ATTENTION_BACKEND` environment variable is deprecated:

```python
if backend_by_env_var.endswith("_VLLM_V1"):
    logger.warning(
        "The suffix '_VLLM_V1' in the environment variable "
        "VLLM_ATTENTION_BACKEND is no longer necessary as "
        "V0 backends have been deprecated. "
        "Please remove this suffix from your "
        "environment variable setting.",
    )
    backend_by_env_var = backend_by_env_var.removesuffix("_VLLM_V1")
```

This allows for a smooth transition by automatically handling the deprecated syntax while warning users to update their configurations.

**Section sources**
- [func_utils.py](file://vllm/utils/func_utils.py#L48-L127)
- [datasets.py](file://vllm/benchmarks/datasets.py#L2073-L2076)
- [selector.py](file://vllm/attention/selector.py#L139-L145)

## Criteria for Deprecating Features
vLLM follows specific criteria when deciding to deprecate features, ensuring that deprecations are justified and necessary:

### Technical Obsolescence
Features are deprecated when they are superseded by more efficient or capable alternatives. For example, V0 attention backends were deprecated in favor of V1 implementations that offer better performance and maintainability.

### API Design Improvements
When better API designs emerge, older patterns may be deprecated to encourage adoption of more intuitive or consistent interfaces. This includes simplifying complex parameter combinations or improving type safety.

### Maintenance Burden
Features that are difficult to maintain, have low usage, or create technical debt may be deprecated to allow the team to focus on higher-priority areas. This is particularly important for experimental features that didn't gain traction.

### Security and Stability
Features with security vulnerabilities or stability issues may be deprecated as part of a responsible disclosure process, with replacements provided to maintain functionality.

### Ecosystem Alignment
As the broader ML ecosystem evolves, vLLM may deprecate features that conflict with standard practices or that are no longer supported by dependent libraries.

## Communication of Deprecation Notices
vLLM employs multiple channels to communicate deprecation notices, ensuring users are informed through various touchpoints:

### Runtime Warnings
The primary communication method is through runtime warnings using Python's `warnings` module. These warnings are designed to be visible without being overly disruptive:

```python
warnings.warn(
    DeprecationWarning(msg),
    stacklevel=3,
)
```

The `stacklevel=3` parameter ensures that the warning points to the user's code rather than the internal implementation, making it easier to identify the source of the deprecated usage.

### Logging System
vLLM's logging system includes specialized methods for deprecation notices, such as `logger.warning_once()`, which prevents repetitive warnings for the same deprecation:

```python
logger.warning_once(
    "use_v1 parameter for get_attn_backend_cls is deprecated and will "
    "be removed in v0.13.0 or v1.0.0, whichever is soonest. Please "
    "remove it from your plugin code."
)
```

This approach ensures users see the warning but aren't overwhelmed by repeated messages.

### Documentation
Deprecation notices are prominently featured in the official documentation, typically in dedicated sections for migration guides and release notes. The documentation includes:
- Clear timelines for removal
- Rationale for the deprecation
- Step-by-step migration instructions
- Examples of the new recommended approaches

### Hidden Metrics Management
For metrics that are deprecated but still needed temporarily, vLLM provides a configuration option to control their visibility:

```python
show_hidden_metrics_for_version: str | None = None
"""Enable deprecated Prometheus metrics that have been hidden since the
specified version. For example, if a previously deprecated metric has been
hidden since the v0.7.0 release, you use
`--show-hidden-metrics-for-version=0.7` as a temporary escape hatch while
you migrate to new metrics."""
```

This allows users to continue using deprecated metrics during migration while encouraging eventual adoption of the new metrics.

**Section sources**
- [func_utils.py](file://vllm/utils/func_utils.py#L76-L79)
- [selector.py](file://vllm/attention/selector.py#L160-L164)
- [observability.py](file://vllm/config/observability.py#L23-L29)

## User Guidance for Migration
vLLM provides comprehensive guidance to help users identify and migrate away from deprecated components:

### Identifying Deprecated Components
Users can identify deprecated components through several methods:

1. **Runtime Warnings**: Monitor application logs for deprecation warnings during normal operation.
2. **Static Analysis**: Use tools to scan code for deprecated imports or function calls.
3. **Documentation Review**: Check the migration guides and release notes for information about deprecated features.

### Migration Strategies
The recommended migration process includes:

1. **Audit Current Usage**: Identify all instances of deprecated components in the codebase.
2. **Update Dependencies**: Ensure all vLLM dependencies are up to date to access the latest features.
3. **Implement Alternatives**: Replace deprecated components with the recommended alternatives.
4. **Test Thoroughly**: Validate that the migration doesn't affect functionality or performance.
5. **Monitor for Warnings**: Continue monitoring logs to ensure no deprecated components remain.

### Example Migration
For example, when migrating from the deprecated `SonnetDataset` class, users should:

1. Replace imports of `SonnetDataset` with alternative benchmark datasets.
2. Update any dataset-specific configuration parameters.
3. Verify that benchmark results remain consistent after the migration.

The `show_hidden_metrics_for_version` configuration parameter provides a valuable tool for gradual migration, allowing users to maintain compatibility with monitoring systems while updating their metric collection processes.

## Examples of Past Deprecation Cycles
Several notable deprecation cycles illustrate vLLM's approach:

### Attention Backend Deprecation
The transition from V0 to V1 attention backends involved:
- Adding deprecation warnings for the `_VLLM_V1` suffix in environment variables
- Providing a clear timeline for removal (v0.13.0 or v1.0.0)
- Maintaining backward compatibility during the transition period
- Documenting the performance benefits of the new implementation

### Benchmark Script Migration
Several benchmark scripts were deprecated in favor of CLI-based alternatives:

```python
print("""DEPRECATED: This script has been moved to the vLLM CLI.
Please use one of the following commands:
vLLM benchmark throughput
vLLM benchmark latency""")
```

This deprecation included:
- Clear messaging about the new location of functionality
- Specific instructions for the replacement commands
- Maintenance of the same core functionality through the new interface

### Dataset Class Deprecation
The `SonnetDataset` class deprecation demonstrates the class-level deprecation pattern:
- Using the `@deprecated` decorator with a clear message
- Maintaining functionality during the transition period
- Planning for complete removal in a future version

## Balancing Innovation and Backward Compatibility
vLLM's deprecation policy reflects a careful balance between innovation and backward compatibility:

### Innovation Drivers
The deprecation process enables important innovations by:
- Removing technical debt that hinders new development
- Allowing adoption of more efficient algorithms and data structures
- Enabling cleaner, more maintainable code architecture
- Facilitating integration with new hardware and software platforms

### Compatibility Considerations
To minimize disruption, vLLM considers:
- The user base impact of each deprecation
- The availability of viable migration paths
- The timing of deprecations relative to other changes
- The provision of adequate notice and support

### Community Engagement
The project maintains open communication with users through:
- Detailed release notes explaining deprecations
- Responsive issue tracking for migration questions
- Documentation that evolves with the codebase
- Community forums for discussing migration challenges

This balanced approach ensures that vLLM can continue to innovate while respecting the needs of its user community.

## Conclusion
vLLM's deprecation policy provides a structured, transparent approach to phasing out features and APIs. By combining technical mechanisms like decorators and runtime warnings with clear communication and migration guidance, the project ensures that users can adapt to changes with minimal disruption. The policy balances the need for innovation with the importance of backward compatibility, enabling vLLM to evolve while maintaining trust with its user community. As the project continues to grow, this deprecation framework will remain essential for managing technical debt and driving forward progress.