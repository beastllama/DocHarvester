from typing import Optional, List, Literal, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
import os
import json


class Settings(BaseSettings):
    # Application settings
    app_name: str = "DocHarvester"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # API settings
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost:5173", "http://localhost:5174"]
    
    # Database settings
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/docharvester",
        description="PostgreSQL connection URL"
    )
    
    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    
    # Celery settings
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    
    # LLM settings - Enhanced for large context models
    llm_provider: Literal["OPENAI", "AZURE_OPENAI", "LOCAL"] = "LOCAL"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_organization_id: Optional[str] = os.getenv("OPENAI_ORGANIZATION_ID", None)
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    llm_model: str = "gpt-4o"  # Default to GPT-4o for large context
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4000  # Increased for large context models
    
    # Large context model options
    available_openai_models: List[str] = [
        "gpt-4o",              # 128k context, latest and most capable
        "gpt-4o-mini",         # 128k context, cost-effective
        "gpt-4-turbo",         # 128k context
        "gpt-4-turbo-preview", # 128k context
        "gpt-4",               # 8k context
        "gpt-3.5-turbo",       # 16k context
        "gpt-3.5-turbo-16k"    # 16k context
    ]
    
    # Available local models for selection
    # Local models verified on the Ollama registry. Pull with: ollama pull <name>
    available_local_models: List[str] = [
        "gemma4:e2b",  # Gemma 4, smallest. Low memory
        "gemma4:e4b",  # Gemma 4, balanced. Default
        "hermes3:8b",  # Nous Research Hermes 3. Strong tool use and instruction following
    ]
    
    # Model cost and performance tiers for recommendations
    model_tiers: Dict[str, Dict[str, any]] = {
        # OpenAI Models
        "gpt-4o": {
            "tier": "premium",
            "cost_per_1k": 0.005,  # Input cost
            "context_window": 128000,
            "recommended_for": ["complex_reasoning", "large_documents", "production"]
        },
        "gpt-4o-mini": {
            "tier": "efficient", 
            "cost_per_1k": 0.00015,
            "context_window": 128000,
            "recommended_for": ["entity_extraction", "summarization", "cost_effective"]
        },
        "gpt-4-turbo": {
            "tier": "high_performance",
            "cost_per_1k": 0.01,
            "context_window": 128000,
            "recommended_for": ["detailed_analysis", "large_documents"]
        },
        # Local Models
        "gemma:2b": {
            "tier": "lightweight",
            "cost_per_1k": 0,
            "context_window": 2048,
            "memory_gb": 4,
            "recommended_for": ["development", "memory_efficient", "fast"]
        },
        "gemma:7b": {
            "tier": "balanced",
            "cost_per_1k": 0,
            "context_window": 4096,
            "memory_gb": 8,
            "recommended_for": ["balanced", "good_quality"]
        }
    }
    
    # Embedding settings
    # Embeddings. LOCAL uses Ollama. OPENAI and AZURE_OPENAI use the cloud models below.
    embedding_provider: Literal["LOCAL", "OPENAI", "AZURE_OPENAI"] = "LOCAL"
    embedding_model: str = "text-embedding-3-small"  # OpenAI and Azure model
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
    # One size for every provider. The database column is built from it.
    # Changing it, or changing the embedding model, means re-embedding: scripts/reembed.py
    embedding_dimension: int = 768
    
    # Chunk settings
    chunk_size: int = 1500
    chunk_overlap: int = 200
    
    # Coverage settings
    coverage_config_path: str = "coverage.yml"
    
    # Document processing
    max_file_size_mb: int = 50
    allowed_file_types: List[str] = [
        ".txt", ".md", ".pdf", ".docx", ".html", 
        ".json", ".yml", ".yaml", ".csv"
    ]
    allowed_extensions: List[str] = [  # Alias for allowed_file_types
        ".txt", ".md", ".pdf", ".docx", ".html", 
        ".json", ".yml", ".yaml", ".csv"
    ]
    
    # Worker settings
    worker_batch_size: int = 20
    worker_timeout_seconds: int = 3600  # 1 hour
    max_concurrent_tasks: int = 4
    
    # Security
    secret_key: str = Field(
        default="",
        description="Secret key for JWT tokens. Required. Set SECRET_KEY in .env"
    )
    access_token_expire_minutes: int = 30

    # Address other machines use to reach this backend, e.g. https://beastllama.tail1234.ts.net
    # Set by scripts/connect-tailscale. Shown in the API tokens page so the Hermes config is right.
    public_url: str = ""

    # Local folder connector may only read inside this root
    ingest_root: str = os.getenv("INGEST_ROOT", "/app/uploads")
    
    # Knowledge Graph settings
    enable_knowledge_graph: bool = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "docharvester123")
    
    # Local LLM settings
    use_local_llm: bool = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "gemma4:e4b")
    
    # Dynamic LLM switching (persisted setting)
    
    # Entity extraction settings
    entity_extraction_enabled: bool = os.getenv("ENTITY_EXTRACTION_ENABLED", "true").lower() == "true"
    entity_extraction_timeout: int = int(os.getenv("ENTITY_EXTRACTION_TIMEOUT", "120"))
    
    # Wiki generation settings
    wiki_generation_enabled: bool = os.getenv("WIKI_GENERATION_ENABLED", "true").lower() == "true"
    
    @model_validator(mode='after')
    def require_real_secret(self):
        """Refuse to start with a missing or placeholder JWT secret"""
        if not self.secret_key or self.secret_key.startswith("your-"):
            raise ValueError(
                "SECRET_KEY is not set. Add a real random value to .env, "
                "for example: openssl rand -hex 32"
            )
        return self

    @field_validator('allowed_file_types', 'allowed_extensions', mode='before')
    @classmethod
    def parse_allowed_file_types(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v.split(',')
        return v
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v.split(',')
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    def get_model_context_window(self, model: str) -> int:
        """Get context window size for different models"""
        context_windows = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16384,
            "gpt-3.5-turbo-16k": 16384,
            "gemma:2b": 2048,
            "gemma:7b": 4096,
            "llama3:8b": 8192,
            "mistral:7b": 8192
        }
        return context_windows.get(model, 4096)
    
    def is_large_context_model(self, model: str) -> bool:
        """Check if model supports large context (>32k tokens)"""
        return self.get_model_context_window(model) >= 32000
    
    def get_model_tier_info(self, model: str) -> Dict[str, any]:
        """Get tier information for a model"""
        return self.model_tiers.get(model, {
            "tier": "unknown",
            "cost_per_1k": 0,
            "context_window": 4096,
            "recommended_for": ["general"]
        })
    
    def get_models_by_tier(self, tier: str) -> List[str]:
        """Get models by performance/cost tier"""
        return [
            model for model, info in self.model_tiers.items()
            if info.get("tier") == tier
        ]
    
    def get_recommended_models_for_task(self, task: str) -> Dict[str, List[str]]:
        """Get models recommended for a specific task, grouped by provider"""
        openai_models = []
        local_models = []
        
        for model, info in self.model_tiers.items():
            if task in info.get("recommended_for", []):
                if model in self.available_openai_models:
                    openai_models.append(model)
                elif model in self.available_local_models:
                    local_models.append(model)
        
        return {
            "openai": openai_models,
            "local": local_models
        }


settings = Settings() 