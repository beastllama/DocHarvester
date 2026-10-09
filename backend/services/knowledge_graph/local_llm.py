"""Local LLM Service for Knowledge Graph Operations"""
import os
from typing import Dict, List, Optional, Any, Union
import httpx
import json
from enum import Enum
import asyncio
import time
from pathlib import Path

from backend.config import settings
from backend.services import runtime_settings


class LLMError(RuntimeError):
    """A local or cloud LLM call failed. Callers must not treat this as content."""


class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


class LocalLLMService:
    """Service for interacting with local and cloud LLMs"""
    
    # File to persist provider preference

    
    # Optimized models for different tasks - using smaller, faster models
    RECOMMENDED_MODELS = {
        "entity_extraction": "gemma4:e4b",  # Fast and efficient for structured output
        "relationship_mapping": "gemma4:e4b",  # Consistent model for reliability  
        "summarization": "gemma4:e4b",  # Good for summarization tasks
        "general": "gemma4:e4b",  # Default - reliable and fast
        "wiki_generation": "gemma4:e4b"  # For wiki content generation
    }
    
    # OpenAI models optimized for different tasks
    RECOMMENDED_OPENAI_MODELS = {
        "entity_extraction": "gpt-4o-mini",      # Fast and cost-effective for structured tasks
        "relationship_mapping": "gpt-4o",        # More capable for complex reasoning
        "summarization": "gpt-4o-mini",          # Good for summarization
        "general": "gpt-4o",                     # Best overall performance
        "large_document": "gpt-4o",              # For processing large documents
        "wiki_generation": "gpt-4o"              # Best for comprehensive content generation
    }
    
    def __init__(self):
        # Use container hostname when running in Docker, localhost otherwise
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_organization_id = os.getenv("OPENAI_ORGANIZATION_ID", None)
        
        # Active provider is shared with the workers through the database (env default until switched)
        self.current_provider = runtime_settings.get_provider(runtime_settings.LLM_KEY)
        if self.current_provider != "LOCAL" and not self.openai_api_key:
            print("⚠️ LLM provider is set to OPENAI but OPENAI_API_KEY is empty")
        self.use_local_llm = self.current_provider == "LOCAL"
        self.default_model = settings.local_llm_model
        
        print(f"🔧 LocalLLMService initialized:")
        print(f"   - LLM_PROVIDER: {self.current_provider}")
        print(f"   - USE_LOCAL_LLM: {self.use_local_llm}")
        print(f"   - LOCAL_LLM_MODEL: {self.default_model}")
        print(f"   - OpenAI API Key configured: {'Yes' if self.openai_api_key else 'No'}")
        print(f"   - Ollama URL: {self.ollama_url}")
        
        # Initialize HTTP client with optimized timeouts
        self.client = None
        self._init_client()
        
        # Add response caching for repeated queries
        self._response_cache = {}
        self._cache_max_size = 100
    

        
    def _init_client(self):
        """Initialize the HTTP client with optimized settings"""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,    # Faster connection timeout
                    read=120.0,      # Reduced from 300s to 120s
                    write=30.0,
                    pool=10.0
                ),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                http2=False
            )
    
    def _get_cache_key(self, prompt: str, model: str, task_type: str) -> str:
        """Generate cache key for responses"""
        import hashlib
        content = f"{prompt[:200]}:{model}:{task_type}"  # Use first 200 chars for key
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cache_response(self, cache_key: str, response: Any):
        """Cache response with size limit"""
        if len(self._response_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
    
    def _get_cached_response(self, cache_key: str, max_age: int = 300) -> Optional[Any]:
        """Get cached response if available and not expired"""
        if cache_key in self._response_cache:
            entry = self._response_cache[cache_key]
            if time.time() - entry["timestamp"] < max_age:
                return entry["response"]
            else:
                # Remove expired entry
                del self._response_cache[cache_key]
        return None

    def get_best_model_for_task(self, task: str, provider: Optional[str] = None) -> str:
        """Get the best model for a specific task"""
        target_provider = provider or self.current_provider
        
        if target_provider == "OPENAI":
            return self.RECOMMENDED_OPENAI_MODELS.get(task, self.RECOMMENDED_OPENAI_MODELS["general"])
        else:
            return self.RECOMMENDED_MODELS.get(task, self.default_model)

    def switch_provider(self, provider: str) -> bool:
        """Switch between LOCAL and OPENAI LLM providers"""
        provider = provider.upper()
        
        if provider not in ["LOCAL", "OPENAI"]:
            print(f"❌ Invalid provider: {provider}. Must be LOCAL or OPENAI")
            return False
        
        if provider == "OPENAI":
            if not self.openai_api_key:
                print("❌ Cannot switch to OPENAI: No API key configured")
                return False
            self.use_local_llm = False
            self.current_provider = "OPENAI"
            # Persist the change to file
            runtime_settings.set_provider(runtime_settings.LLM_KEY, "OPENAI")
            print(f"✅ Switched to OpenAI provider")
        else:  # LOCAL
            self.use_local_llm = True
            self.current_provider = "LOCAL"
            # Persist the change to file
            runtime_settings.set_provider(runtime_settings.LLM_KEY, "LOCAL")
            print(f"✅ Switched to LOCAL provider (Ollama)")
        
        return True

    async def validate_openai_connection(self) -> Dict[str, Any]:
        """Validate OpenAI API connection and return available models"""
        if not self.openai_api_key:
            return {"valid": False, "error": "No API key configured"}
            
        try:
            from openai import AsyncOpenAI
            
            client_kwargs = {"api_key": self.openai_api_key}
            if self.openai_organization_id:
                client_kwargs["organization"] = self.openai_organization_id
                
            client = AsyncOpenAI(**client_kwargs)
            
            # Test with a simple completion using the fastest model
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
                max_tokens=10,
                timeout=30  # 30 second timeout for validation
            )
            
            if response.choices[0].message.content:
                # Get available models
                models = await client.models.list()
                available_models = [
                    m.id for m in models.data 
                    if m.id in settings.available_openai_models
                ]
                
                return {
                    "valid": True,
                    "available_models": available_models,
                    "test_response": response.choices[0].message.content,
                    "organization_id": self.openai_organization_id
                }
            else:
                return {"valid": False, "error": "No response from OpenAI"}
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                return {
                    "valid": False, 
                    "error": "API quota exceeded - please check your billing",
                    "error_type": "quota_exceeded",
                    "suggestion": "Check your OpenAI billing dashboard or switch to local LLM"
                }
            elif "401" in error_msg or "authentication" in error_msg.lower():
                return {
                    "valid": False,
                    "error": "Authentication failed - check API key and organization ID",
                    "error_type": "auth_failed"
                }
            else:
                return {"valid": False, "error": error_msg, "error_type": "unknown"}

    async def validate_ollama_connection(self) -> Dict[str, Any]:
        """Validate Ollama connection and return available models"""
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            
            return {
                "valid": True,
                "available_models": [m["name"] for m in models],
                "ollama_url": self.ollama_url
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}
        
    async def ensure_model_available(self, model_name: str) -> bool:
        """Ensure the specified Ollama model is available locally"""
        if not self.use_local_llm:
            return True
            
        try:
            # Check if model exists
            response = await self.client.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            
            if not any(m["name"] == model_name for m in models):
                print(f"🔽 Pulling model {model_name}...")
                pull_response = await self.client.post(
                    f"{self.ollama_url}/api/pull",
                    json={"name": model_name},
                    timeout=httpx.Timeout(600.0)  # 10 minutes for model download
                )
                pull_response.raise_for_status()
                print(f"✅ Model {model_name} pulled successfully")
                
            return True
        except Exception as e:
            print(f"❌ Error ensuring model availability: {e}")
            return False

    async def extract_entities(
        self,
        text: str,
        entity_types: List[Dict[str, Any]],
        lens_type: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Extract entities from text using current LLM provider with optimizations"""
        
        task_type = "entity_extraction"
        model = self.get_best_model_for_task(task_type)
        
        # Check cache first
        cache_key = self._get_cache_key(text[:500], model, task_type) if use_cache else None
        if cache_key:
            cached = self._get_cached_response(cache_key, max_age=1800)  # 30 min cache
            if cached:
                print(f"🎯 Using cached entity extraction")
                return cached
        
        # Optimize entity schema for prompt
        entity_schema = "\n".join([
            f"- {et['name']}: {', '.join(et['properties'][:3])}"  # Limit properties
            for et in entity_types[:5]  # Limit entity types
        ])
        
        # Optimize text length and chunk if needed
        max_text_length = 4000 if self.use_local_llm else 16000
        
        if len(text) > max_text_length:
            # Process in chunks for very long text
            chunks = [text[i:i+max_text_length] for i in range(0, len(text), max_text_length)]
            all_entities = []
            all_relationships = []
            
            for i, chunk in enumerate(chunks[:3]):  # Limit to 3 chunks max
                print(f"🔍 Processing chunk {i+1}/{min(len(chunks), 3)}")
                chunk_result = await self._extract_entities_from_chunk(
                    chunk, entity_schema, lens_type, model, task_type
                )
                if isinstance(chunk_result, dict):
                    all_entities.extend(chunk_result.get("entities", []))
                    all_relationships.extend(chunk_result.get("relationships", []))
            
            result = {
                "entities": all_entities,
                "relationships": all_relationships
            }
        else:
            result = await self._extract_entities_from_chunk(
                text, entity_schema, lens_type, model, task_type
            )
        
        # Cache the result
        if cache_key and isinstance(result, dict):
            self._cache_response(cache_key, result)
        
        return result

    async def _extract_entities_from_chunk(
        self,
        text: str,
        entity_schema: str,
        lens_type: Optional[str],
        model: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Extract entities from a single chunk of text"""
        
        prompt = f"""Extract entities from the following text. Focus on the most important entities only.

Entity Types:
{entity_schema}

Context: This is a {lens_type or 'general'} document.

Text:
{text}

Return a JSON object with this structure:
{{
    "entities": [
        {{
            "type": "EntityType",
            "name": "entity name",
            "properties": {{"key": "value"}},
            "confidence": 0.95
        }}
    ],
    "relationships": [
        {{
            "source": "entity1",
            "target": "entity2", 
            "type": "relationship_type"
        }}
    ]
}}

Only return valid JSON, no additional text."""

        if self.use_local_llm:
            return await self._query_ollama(prompt, model, task_type)
        else:
            return await self._query_openai(prompt, task_type)

    async def map_relationships(
        self,
        entities: List[Dict[str, Any]],
        context: str,
        existing_relationships: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """Map relationships between entities with optimizations"""
        
        task_type = "relationship_mapping"
        model = self.get_best_model_for_task(task_type)
        
        # Limit entity count for performance
        max_entities = 15 if self.use_local_llm else 30
        entities = entities[:max_entities]
        
        entities_text = "\n".join([
            f"- {e['type']}: {e['name']}"
            for e in entities
        ])
        
        context_length = 1500 if self.use_local_llm else 4000
        
        prompt = f"""Identify the most important relationships between these entities:

Entities:
{entities_text}

Context:
{context[:context_length]}

Return a JSON array of relationships:
[
    {{
        "source": "entity_name_1",
        "target": "entity_name_2", 
        "type": "uses|manages|contains|depends_on",
        "strength": 0.9
    }}
]

Focus on direct, meaningful relationships. Return only JSON."""

        if self.use_local_llm:
            result = await self._query_ollama(prompt, model, task_type)
            return result if isinstance(result, list) else result.get("relationships", [])
        else:
            result = await self._query_openai(prompt, task_type)
            return result if isinstance(result, list) else result.get("relationships", [])

    async def generate_summary(
        self,
        text: str,
        max_length: int = 200,
        focus: Optional[str] = None
    ) -> str:
        """Generate a summary of the text with optimizations"""
        
        task_type = "summarization"
        model = self.get_best_model_for_task(task_type)
        
        # Check cache
        cache_key = self._get_cache_key(text[:300] + str(max_length), model, task_type)
        cached = self._get_cached_response(cache_key, max_age=3600)  # 1 hour cache
        if cached:
            return cached
        
        max_text_length = 3000 if self.use_local_llm else 12000
        
        prompt = f"""Summarize the following text in {max_length} characters or less.
{f"Focus on: {focus}" if focus else ""}

Text:
{text[:max_text_length]}

Summary:"""

        if self.use_local_llm:
            result = await self._query_ollama(prompt, model, task_type, json_response=False)
        else:
            result = await self._query_openai(prompt, task_type, json_response=False)
            
        # Cache the result
        self._cache_response(cache_key, result)
        return result

    async def query_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        json_mode: bool = False,
        task_type: str = "general",
        use_cache: bool = False
    ) -> str:
        """General purpose LLM query method for wiki generation and other tasks"""
        
        model = self.get_best_model_for_task(task_type)
        
        print(f"🔧 query_llm called with provider: {self.current_provider}, model: {model}")
        
        # Check cache if enabled
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(prompt[:300], model, task_type)
            cached = self._get_cached_response(cache_key, max_age=600)  # 10 min cache
            if cached:
                print(f"🎯 Using cached response")
                return cached
        
        if self.use_local_llm:
            try:
                result = await self._query_ollama(
                    prompt=prompt,
                    model=model,
                    task_type=task_type,
                    json_response=json_mode,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Handle different response types
                if isinstance(result, dict):
                    if "error" in result:
                        raise Exception(f"Local LLM error: {result['error']}")
                    # If it's a JSON response and we're expecting JSON, return the dict as string
                    response = json.dumps(result) if json_mode else str(result.get("response", ""))
                else:
                    response = str(result)
                    
                # Cache the result
                if cache_key:
                    self._cache_response(cache_key, response)
                    
                return response
                    
            except Exception as e:
                print(f"❌ Local LLM query failed: {e}")
                raise Exception(f"Local LLM failed: {e}")
                
        # Use OpenAI
        try:
            response = await self._query_openai(
                prompt, task_type, json_response=json_mode, 
                max_tokens=max_tokens, temperature=temperature
            )
            
            # Cache the result
            if cache_key:
                self._cache_response(cache_key, response)
                
            return response
        except Exception as e:
            print(f"❌ OpenAI query failed: {e}")
            raise Exception(f"OpenAI LLM failed: {e}")

    async def _query_ollama(
        self,
        prompt: str,
        model: str,
        task_type: str,
        json_response: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Any:
        """Query Ollama. Failures raise LLMError. They are never returned as document text."""
        # Ensure client is initialized
        if self.client is None:
            self._init_client()

        # Optimize parameters based on task type for much faster responses
        if task_type in ["entity_extraction", "relationship_mapping"]:
            temperature = 0.1  # More deterministic for structured tasks
            max_tokens = min(max_tokens, 500)
        elif task_type == "wiki_generation":
            temperature = 0.3
            max_tokens = min(max_tokens, 800)
        else:
            max_tokens = min(max_tokens, 600)

        options = {
            "temperature": temperature,
            "top_p": 0.7,
            "num_predict": max_tokens,
            "num_ctx": 8192,  # Prompts are larger than 1024 tokens
            "repeat_penalty": 1.2,
            "top_k": 20,
        }
        # Stop tokens only for structured output. For markdown they cut the page short.
        if json_response:
            options["stop"] = ["</json>", "```"]

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # Gemma 4 and other thinking models otherwise spend the budget on hidden reasoning
            "options": options,
        }
        if json_response:
            payload["format"] = "json"

        print(f"🔍 Ollama request: {self.ollama_url}/api/generate model={model}")
        start_time = time.time()

        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                # A cold model load can take half a minute on CPU, so the read timeout is generous
                timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=30.0),
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama timed out while generating with {model}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed for {model}: {exc}") from exc

        result = response.json()
        generated_text = (result.get("response") or "").strip()
        print(f"⏱️ Ollama response time: {time.time() - start_time:.2f}s")

        if not generated_text:
            raise LLMError(
                f"{model} returned an empty response (done_reason={result.get('done_reason')}). "
                f"Try a larger max_tokens or check the model."
            )

        if not json_response:
            return generated_text

        try:
            return json.loads(generated_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}|\[.*\]', generated_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {"error": "Failed to parse JSON response", "raw": generated_text}

    async def _query_openai(
        self,
        prompt: str,
        task_type: str = "general",
        json_response: bool = False,
        max_tokens: int = 4000,
        temperature: float = 0.7
    ) -> str:
        """Query OpenAI API with support for large context models"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")
            
        try:
            from openai import AsyncOpenAI
            
            client_kwargs = {"api_key": self.openai_api_key}
            if self.openai_organization_id:
                client_kwargs["organization"] = self.openai_organization_id
                
            client = AsyncOpenAI(**client_kwargs)
            
            model = self.get_best_model_for_task(task_type, "OPENAI")
            
            # Optimize parameters based on task
            if task_type in ["entity_extraction", "relationship_mapping"]:
                temperature = 0.2  # More deterministic
                max_tokens = min(max_tokens, 2000)
            elif task_type == "wiki_generation":
                temperature = 0.7  # Balanced creativity
                max_tokens = min(max_tokens, 4000)
            
            system_message = f"You are a {task_type} assistant."
            if json_response:
                system_message += " Always respond with valid JSON only."
            
            start_time = time.time()
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60  # 60 second timeout
            )
            
            elapsed_time = time.time() - start_time
            print(f"⏱️ OpenAI response time: {elapsed_time:.2f}s")
            
            content = response.choices[0].message.content
            
            if json_response:
                try:
                    # Validate JSON
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
                    # Try to extract JSON
                    import re
                    json_match = re.search(r'\{.*\}|\[.*\]', content, re.DOTALL)
                    if json_match:
                        return json_match.group()
                    raise ValueError(f"Invalid JSON response: {content}")
            
            return content
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ OpenAI API error: {error_msg}")
            
            # Handle quota exceeded gracefully
            if "429" in error_msg or "quota" in error_msg.lower():
                raise Exception(
                    "OpenAI quota exceeded. Please check your billing or switch to local LLM. "
                    "You can switch providers in the admin settings."
                )
            elif "401" in error_msg:
                raise Exception(
                    "OpenAI authentication failed. Please check your API key and organization ID."
                )
            else:
                raise
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()


# Specialized prompts for logistics domain
LOGISTICS_PROMPTS = {
    "route_optimization": """Analyze the following logistics data and suggest route optimizations:
{data}

Consider: distance, time, cost, capacity, and constraints.""",
    
    "equipment_analysis": """Analyze equipment utilization and status:
{data}

Identify: underutilized assets, maintenance needs, and optimization opportunities.""",
    
    "supply_chain_insights": """Extract supply chain insights from:
{data}

Focus on: bottlenecks, dependencies, risks, and improvement areas."""
} 