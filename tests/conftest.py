"""Pytest configuration and fixtures.

Torch is a required dependency and must be present for most tests.
The "function '_has_torch_function' already has a docstring" error occurs when
torch is imported multiple times during pytest-cov instrumentation.

This conftest ensures pytest hooks are set up properly, but doesn't import torch
directly to avoid conflicts with pytest plugins that may load torch early.
"""
