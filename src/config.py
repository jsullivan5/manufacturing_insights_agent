from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict # Must use pydantic_settings for env vars
from typing import Dict, Optional, Any
from decimal import Decimal, ROUND_HALF_UP
import os
from dotenv import load_dotenv

# Ensure .env is loaded once at the start of this module, before Settings class is defined.
# Pydantic's BaseSettings will also attempt to load it based on model_config.
load_dotenv(override=True) # override=True ensures .env takes precedence over existing env vars if any conflict for a var it sets.

class Settings(BaseSettings): # Inherit from BaseSettings
    """
    Centralized configuration settings for the Manufacturing Copilot LLM Orchestrator.
    Uses pydantic-settings for robust environment variable and .env file handling.
    """
    # --- API Keys & Paths ---
    # Field name `openai_api_key` with `env_prefix="MCP_"` will look for `MCP_OPENAI_API_KEY`
    openai_api_key: Optional[str] = Field(None, description="OpenAI API Key. Loaded from MCP_OPENAI_API_KEY.")
    tag_glossary_path: str = "data/tag_glossary.csv"
    tool_schemas_cache_path: str = ".cache/tool_schemas.json"
    evidence_log_dir: str = Field("logs/evidence_store", description="Directory to store evidence logs.")

    # --- LLM Model Configuration ---
    planning_model_name: str = Field("gpt-4o-mini")
    final_report_model_name: str = Field("gpt-4o")
    planning_temperature: float = 0.1
    temperature_final_report: float = 0.3
    max_tokens_planning: int = Field(1500, description="Max tokens for planning model. Ensure this is well within model limits (e.g., gpt-4o-mini 128k context).")
    max_tokens_final_report: int = Field(2000, description="Max tokens for final report model.")
    max_tokens_final_report_evidence_summary: int = Field(4000, description="Max tokens to allocate for summarizing evidence in the final report prompt.") 
    default_tool_openai_schema_version: str = "0.1" 

    # --- Token Pricing (USD per token) ---
    token_prices_usd_per_token: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=lambda: {
            "gpt-4o-mini": {"input": Decimal("0.15") / Decimal("1000000"), "output": Decimal("0.60") / Decimal("1000000")},
            "gpt-4o": {"input": Decimal("5.00") / Decimal("1000000"), "output": Decimal("15.00") / Decimal("1000000")},
        }
    )

    # --- Orchestrator Behavior ---
    max_steps: int = 7
    confidence_threshold: float = 0.9
    default_numeric_tag: str = Field("FREEZER01.TEMP.INTERNAL_C")
    min_initial_tag_similarity_threshold: float = 0.25
    default_window_hours: int = 2
    default_timezone: str = "UTC"

    # --- Confidence Scoring & Staleness ---
    default_confidence_decay_base: float = 0.98
    min_confidence_delta_for_staleness: float = 0.005
    max_stale_steps: int = 5

    # --- Guardrails ---
    max_runaway_cost_usd: Decimal = Field(Decimal("0.50"), description="Max USD cost before stopping a run.")
    max_window_age_days: int = 30

    # --- Logging ---
    log_level: str = "INFO"

    # model_config is the Pydantic V2 way to configure settings loading
    model_config = SettingsConfigDict(
        env_prefix="MCP_", 
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False # Default, but good to be explicit if prefix case is a concern
    )

    @field_validator("evidence_log_dir")
    @classmethod
    def create_evidence_log_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v

    def get_token_price(self, model_name: str, direction: str) -> Decimal:
        model_prices = self.token_prices_usd_per_token.get(model_name)
        if model_prices is None: raise ValueError(f"Token prices not configured for model: {model_name}")
        price = model_prices.get(direction)
        if price is None: raise ValueError(f"Token price not for model {model_name}, direction {direction}")
        return price.quantize(Decimal('1e-10'), rounding=ROUND_HALF_UP)

# Example of how to load settings (orchestrator would do this)
# if __name__ == "__main__":
#     # Create a dummy .env file for testing
#     # with open(".env", "w") as f:
#     #     f.write("MCP_OPENAI_API_KEY=your_actual_openai_key_here\n") # Note MCP_ prefix
#     #     f.write("MCP_DEFAULT_NUMERIC_TAG=OVERRIDE.TEMP.SENSOR\n")
#     #     f.write("MCP_PLANNING_MODEL_NAME=gpt-3.5-turbo\n")
#
#     settings = Settings()
#     print("Loaded Settings:")
#     print(f"  OpenAI API Key: {'Loaded' if settings.openai_api_key else 'Not set (expected from env: MCP_OPENAI_API_KEY)'}")
#     print(f"  Planning Model: {settings.planning_model_name}")
#     print(f"  Default Tool OpenAI Schema Version: {settings.default_tool_openai_schema_version}")
#     print(f"  Default Numeric Tag: {settings.default_numeric_tag}")
#     print(f"  Evidence Log Dir: {settings.evidence_log_dir}")
#     print(f"  Token Price for {settings.planning_model_name} input: {settings.get_token_price(settings.planning_model_name, 'input')}")
#     print(f"  Max Runaway Cost: {settings.max_runaway_cost_usd}")
#     print(f"  Default Timezone: {settings.default_timezone}")
#
#     # Clean up dummy .env
#     # if os.path.exists(".env"):
#     #     os.remove(".env") 