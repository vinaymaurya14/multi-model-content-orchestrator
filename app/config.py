"""Application configuration using pydantic-settings.

All settings are loaded from environment variables with sensible defaults.
The application works out-of-the-box in demo mode (no API keys required).
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the orchestrator."""

    # ---- General ----
    app_name: str = "Multi-Model Content Orchestrator"
    app_version: str = "1.0.0"
    debug: bool = False
    demo_mode: bool = True
    default_provider: str = "mock"
    log_level: str = "info"
    host: str = "0.0.0.0"
    port: int = 8002

    # ---- Provider API Keys (all optional) ----
    openai_api_key: Optional[str] = None
    openai_org_id: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    hf_api_token: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # ---- Cost Controls ----
    monthly_cost_limit: float = 10.0
    cost_alert_threshold: float = 0.8

    # ---- Generation Defaults ----
    default_max_tokens: int = 1024
    default_temperature: float = 0.7

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # -- Derived helpers --
    @property
    def available_real_providers(self) -> list[str]:
        """Return a list of provider names whose API keys are configured."""
        providers: list[str] = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.hf_api_token:
            providers.append("huggingface")
        # Ollama does not need an API key; availability is checked at runtime.
        providers.append("ollama")
        return providers


# Singleton settings instance
settings = Settings()
