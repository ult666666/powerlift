"""Shared pytest configuration."""

from hypothesis import settings

# Derandomized property tests keep CI deterministic; no deadline because the
# first SciPy call in a process can be slow.
settings.register_profile("default", derandomize=True, deadline=None, max_examples=100)
settings.load_profile("default")
