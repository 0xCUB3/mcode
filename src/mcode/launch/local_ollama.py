"""Local Ollama target.

Public surface mirrors local_vllm. Model pull is a separate phase so the user
sees progress when a model is being fetched the first time.
"""

from __future__ import annotations
