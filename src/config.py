"""Central configuration.

Every tunable knob lives here and is populated from the environment (or a
``.env`` file). Nothing in the rest of the codebase reads ``os.environ``
directly, which keeps the agent reproducible: an evaluation run is fully
described by the settings object it was given.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Provider(str, Enum):
    """Supported LLM backends."""

    #: OpenAI-compatible endpoint. Default: higher rate limits than the
    #: Gemini free tier, and serves both chat and embedding models.
    NVIDIA = "nvidia"
    #: Genuine OpenAI. Same wire protocol as NVIDIA, but it rejects the
    #: vendor extensions NVIDIA accepts - see PROVIDER_DEFAULTS.
    OPENAI = "openai"
    GEMINI = "gemini"


#: Per-provider defaults, applied when the corresponding setting is left blank.
#:
#: "OpenAI-compatible" is a family resemblance, not a guarantee: the endpoints
#: agree on the message/tool-call shape but differ on parameter names and on
#: which vendor extensions they tolerate. Rather than sniff the model name at
#: runtime, those differences are declared here as overridable settings.
PROVIDER_DEFAULTS: dict[Provider, dict[str, Any]] = {
    Provider.NVIDIA: {
        # A fast reasoning model: tool-call turns land in ~1s versus ~50s for
        # the larger models on the same endpoint, which is the difference
        # between a usable agent and one nobody waits for.
        "model_name": "nvidia/nemotron-3.5-lightning-30b-a3b",
        # Asymmetric retrieval model: encodes questions and passages
        # differently, which suits question -> abstract matching.
        "embedding_model": "nvidia/nv-embedqa-e5-v5",
        # NVIDIA's retriever models *require* input_type; OpenAI rejects it.
        "embedding_input_type": True,
        "max_tokens_field": "max_tokens",
        "send_temperature": True,
    },
    Provider.OPENAI: {
        "model_name": "gpt-5",
        "embedding_model": "text-embedding-3-small",
        "embedding_input_type": False,
        # Newer OpenAI models reject `max_tokens` and require this instead.
        "max_tokens_field": "max_completion_tokens",
        # OpenAI reasoning models accept only the default temperature, so it
        # is omitted rather than risking a 400. Set SEND_TEMPERATURE=true for
        # a chat model that supports it.
        "send_temperature": False,
    },
    Provider.GEMINI: {
        # The Flash tier is what a free key can reach; Pro returns 429
        # without a billing account.
        "model_name": "gemini-3.7-flash",
        "embedding_model": "gemini-embedding-2",
        # Unused by the Gemini adapter, which has its own wire format.
        "embedding_input_type": False,
        "max_tokens_field": "max_tokens",
        "send_temperature": True,
    },
}


class TransportMode(str, Enum):
    """How an external dependency (LLM or PubMed) is reached.

    ``LIVE``    - real network calls.
    ``RECORD``  - real network calls, persisted to ``fixtures/`` as cassettes.
    ``REPLAY``  - no network; responses are served from recorded cassettes.

    ``REPLAY`` is what makes the evaluation suite deterministic and runnable
    without any API key.
    """

    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider -----------------------------------------------------
    # Two backends are implemented. The agent code above src/llm/base.py is
    # provider-agnostic; this selects which adapter is built.
    provider: Provider = Provider.NVIDIA

    nvidia_api_key: SecretStr | None = Field(
        default=None, description="NVIDIA NIM key (OpenAI-compatible endpoint)."
    )
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    openai_api_key: SecretStr | None = Field(
        default=None, description="OpenAI API key."
    )
    openai_base_url: str = "https://api.openai.com/v1"

    gemini_api_key: SecretStr | None = Field(
        default=None, description="Google AI Studio key. Not required in REPLAY mode."
    )
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Left unset, each provider's defaults below are applied.
    model_name: str = ""
    embedding_model: str = ""
    temperature: float = 0.1
    # Reasoning tokens are drawn from the same budget as the answer, so this
    # needs real headroom or a long synthesis truncates mid-JSON.
    max_output_tokens: int = 16384

    # Vendor extension, off by default: these parameters are accepted by
    # NVIDIA-hosted reasoning models and rejected outright by OpenAI.
    enable_thinking: bool = False
    reasoning_budget: int = 8192

    # --- Dialect capabilities (filled per provider; override via env) ------
    embedding_input_type: bool = True
    max_tokens_field: str = "max_tokens"
    send_temperature: bool = True

    # --- Transports -------------------------------------------------------
    llm_mode: TransportMode = TransportMode.LIVE
    pubmed_mode: TransportMode = TransportMode.LIVE

    # --- NCBI E-utilities -------------------------------------------------
    # NCBI asks every client to identify itself; unidentified traffic is the
    # first to get throttled.
    ncbi_api_key: SecretStr | None = None
    ncbi_tool: str = "pubmed-evidence-agent"
    ncbi_email: str = "agent@example.com"
    ncbi_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # --- Budgets ----------------------------------------------------------
    # These are the agent's circuit breakers. Without them a confused
    # researcher node can loop on tool calls until the quota is gone.
    max_research_iterations: int = 4
    max_tool_calls: int = 10
    max_revisions: int = 1
    max_articles_per_search: int = 12
    max_evidence_items: int = 8
    evidence_char_budget: int = 24_000

    # --- Reliability ------------------------------------------------------
    # Reasoning models writing a structured answer over a full evidence set
    # routinely exceed 90s; a short timeout turns a slow success into two
    # wasted attempts plus backoff.
    request_timeout: float = 240.0
    max_retries: int = 4
    retry_base_delay: float = 1.0
    # A 429 can ask for a wait of tens of seconds (the Gemini free tier allows
    # only 5 requests/minute). Capped so a stuck quota cannot hang a run.
    max_retry_delay: float = 65.0

    # The deterministic emergency and personal-advice guards always run. This
    # flag controls only the *optional* LLM scope classifier, which costs an
    # extra request per question - material when the quota is 5/minute.
    use_llm_scope_check: bool = False

    # --- Paths ------------------------------------------------------------
    fixtures_dir: Path = REPO_ROOT / "fixtures"
    traces_dir: Path = REPO_ROOT / "traces"
    memory_db: Path = REPO_ROOT / "traces" / "memory.sqlite3"

    # --- Observability ----------------------------------------------------
    log_level: str = "INFO"

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("fixtures_dir", "traces_dir")
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    def model_post_init(self, _context: Any) -> None:
        """Fill in provider-specific defaults for anything left blank.

        Dialect capabilities are only defaulted when the user has not set them
        explicitly, so an unusual endpoint stays configurable from ``.env``.
        """
        defaults = PROVIDER_DEFAULTS[self.provider]
        if not self.model_name:
            object.__setattr__(self, "model_name", defaults["model_name"])
        if not self.embedding_model:
            object.__setattr__(self, "embedding_model", defaults["embedding_model"])

        explicit = self.model_fields_set
        for key in ("embedding_input_type", "max_tokens_field", "send_temperature"):
            if key not in explicit:
                object.__setattr__(self, key, defaults[key])

    @property
    def api_key(self) -> SecretStr | None:
        """The key for the selected provider."""
        return {
            Provider.NVIDIA: self.nvidia_api_key,
            Provider.OPENAI: self.openai_api_key,
            Provider.GEMINI: self.gemini_api_key,
        }[self.provider]

    @property
    def base_url(self) -> str:
        """The base URL for the selected provider."""
        return {
            Provider.NVIDIA: self.nvidia_base_url,
            Provider.OPENAI: self.openai_base_url,
            Provider.GEMINI: self.gemini_base_url,
        }[self.provider]

    @property
    def requires_api_key(self) -> bool:
        """True when the configured mode will actually hit the LLM API."""
        return self.llm_mode in (TransportMode.LIVE, TransportMode.RECORD)

    def ensure_dirs(self) -> None:
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
