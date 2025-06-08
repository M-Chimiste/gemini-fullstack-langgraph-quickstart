import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from inference.llm import LLMModelFactory


@dataclass
class NodeModelConfig:
    """Configuration for a single graph node."""
    model_type: str
    model_name: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    num_ctx: int | None = None


class ModelRouter:
    """Router that returns LLM instances per node."""

    def __init__(self, configs: Dict[str, NodeModelConfig]):
        self._configs = configs
        self._factory = LLMModelFactory()
        self._cache: Dict[str, Any] = {}

    def get_model(self, node: str):
        if node not in self._configs:
            raise KeyError(f"No model configuration for node '{node}'")
        cfg = self._configs[node]
        key = f"{cfg.model_type}:{cfg.model_name}"
        if key not in self._cache:
            kwargs = {"model_name": cfg.model_name}
            if cfg.max_new_tokens is not None:
                kwargs["max_new_tokens"] = cfg.max_new_tokens
            if cfg.temperature is not None:
                kwargs["temperature"] = cfg.temperature
            if cfg.num_ctx is not None:
                kwargs["num_ctx"] = cfg.num_ctx
            self._cache[key] = self._factory.create_model(cfg.model_type, **kwargs)
        return self._cache[key]


def load_model_router() -> ModelRouter:
    """Load router configuration from environment variables."""
    config_json = os.getenv("MODEL_ROUTER_CONFIG")
    user_config: Dict[str, Dict[str, Any]] = {}
    if config_json:
        try:
            user_config = json.loads(config_json)
        except json.JSONDecodeError:
            pass

    default_config = {
        "generate_query": {
            "model_type": os.getenv("QUERY_GENERATOR_PROVIDER", "gemini"),
            "model_name": os.getenv("QUERY_GENERATOR_MODEL", "gemini-2.0-flash"),
            "temperature": 1.0,
        },
        "reflection": {
            "model_type": os.getenv("REFLECTION_PROVIDER", "gemini"),
            "model_name": os.getenv("REFLECTION_MODEL", "gemini-2.5-flash-preview-04-17"),
            "temperature": 1.0,
        },
        "finalize_answer": {
            "model_type": os.getenv("ANSWER_PROVIDER", "gemini"),
            "model_name": os.getenv("ANSWER_MODEL", "gemini-2.5-pro-preview-05-06"),
            "temperature": 0.0,
        },
    }

    for node, cfg in user_config.items():
        if node in default_config:
            default_config[node].update(cfg)
        else:
            default_config[node] = cfg

    node_configs = {n: NodeModelConfig(**c) for n, c in default_config.items()}
    return ModelRouter(node_configs)
