# LANGVEL PRODUCTION HARDENING TODOS

**Generated**: 2025-10-17
**Total Issues**: 47
**Estimated Effort**: 16 engineer-weeks
**Target**: Production-ready in 8-12 weeks

---

## 🚨 P0 - CRITICAL (MUST FIX BEFORE PRODUCTION)

### **SECURITY**

#### ✅ TODO-001: Fix JWT Secret Key Management
- **File**: `langvel/auth/manager.py:46`
- **Severity**: CRITICAL (10/10)
- **Issue**: JWT secret regenerates on every restart, invalidates all tokens
- **Impact**:
  - Tokens don't work across restarts
  - Each worker has different secret in multi-worker setup
  - All users logged out on deployment
- **Fix**:
  ```python
  def _get_secret_key(self) -> str:
      import os
      secret = os.getenv('JWT_SECRET_KEY')
      if not secret:
          raise RuntimeError(
              "JWT_SECRET_KEY environment variable is required. "
              "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
          )
      return secret
  ```
- **Testing**: Verify tokens work across restarts and multiple workers
- **Effort**: 2 hours

#### ✅ TODO-002: Fix Error Message Exposure
- **File**: `langvel/server.py:150`
- **Severity**: CRITICAL (8/10)
- **Issue**: Global exception handler exposes internal errors, stack traces, credentials
- **Impact**: Information leakage, OWASP A06:2021 violation
- **Fix**:
  ```python
  @app.exception_handler(Exception)
  async def global_exception_handler(request: Request, exc: Exception):
      # Log full error internally
      logger.exception("Request failed", exc_info=exc, request_path=request.url.path)

      # Return safe error to client
      trace_id = getattr(request.state, 'trace_id', 'unknown')

      if config.DEBUG:
          return JSONResponse(
              status_code=500,
              content={"error": str(exc), "type": type(exc).__name__, "trace_id": trace_id}
          )
      else:
          return JSONResponse(
              status_code=500,
              content={"error": "Internal server error", "trace_id": trace_id}
          )
  ```
- **Effort**: 3 hours

#### ✅ TODO-003: Add Request Size Limits
- **File**: `langvel/server.py:1`
- **Severity**: HIGH (7/10)
- **Issue**: No size limits on request body, allows payload bomb DoS
- **Impact**: Memory exhaustion, denial of service
- **Fix**:
  ```python
  from fastapi import Request, HTTPException
  from starlette.middleware.base import BaseHTTPMiddleware

  class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
      def __init__(self, app, max_size: int = 10_000_000):  # 10MB default
          super().__init__(app)
          self.max_size = max_size

      async def dispatch(self, request: Request, call_next):
          if request.method in ["POST", "PUT", "PATCH"]:
              content_length = request.headers.get("content-length")
              if content_length and int(content_length) > self.max_size:
                  raise HTTPException(413, "Request body too large")
          return await call_next(request)

  app.add_middleware(RequestSizeLimitMiddleware)
  ```
- **Effort**: 2 hours

#### ✅ TODO-004: Validate Agent Path for Path Traversal
- **File**: `langvel/server.py:69`
- **Severity**: HIGH (7/10)
- **Issue**: `agent_path` not validated, allows path traversal
- **Impact**: Could access unauthorized agents or system files
- **Fix**:
  ```python
  import re

  AGENT_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9_/-]+$')

  @app.post("/agents/{agent_path:path}")
  async def invoke_agent(agent_path: str, request: AgentRequest):
      # Validate path
      if not AGENT_PATH_PATTERN.match(agent_path):
          raise HTTPException(400, "Invalid agent path")

      if '..' in agent_path or agent_path.startswith('/'):
          raise HTTPException(400, "Invalid agent path")

      # Rest of implementation...
  ```
- **Effort**: 2 hours

### **PERFORMANCE & STABILITY**

#### ✅ TODO-005: Implement Agent Instance Pooling
- **File**: `langvel/server.py:89-90`
- **Severity**: CRITICAL (9/10)
- **Issue**: New agent instance created per request, causes memory leak
- **Impact**:
  - Memory leak: +15MB per request
  - 1000 req/min = +15GB/hour → OOM in 2-3 hours
  - Re-initialization overhead (LLM clients, MCP servers, etc.)
- **Fix**:
  ```python
  from typing import Dict
  from threading import Lock

  _agent_pool: Dict[str, Agent] = {}
  _agent_pool_lock = Lock()

  def get_agent(agent_path: str) -> Agent:
      """Get or create agent instance (singleton per path)."""
      if agent_path not in _agent_pool:
          with _agent_pool_lock:
              # Double-check inside lock
              if agent_path not in _agent_pool:
                  agent_class = router.get(f"/{agent_path}")
                  if not agent_class:
                      raise ValueError(f"Agent not found: {agent_path}")
                  _agent_pool[agent_path] = agent_class()
      return _agent_pool[agent_path]

  @app.post("/agents/{agent_path:path}")
  async def invoke_agent(agent_path: str, request: AgentRequest):
      agent = get_agent(agent_path)  # Reuse instance
      # Rest of implementation...
  ```
- **Testing**: Monitor memory usage under load (1000 requests)
- **Effort**: 4 hours

#### ✅ TODO-006: Add Request Timeout
- **File**: `langvel/server.py:102`
- **Severity**: CRITICAL (9/10)
- **Issue**: No timeout on agent execution, allows hanging requests
- **Impact**: DoS vulnerability, resource exhaustion
- **Fix**:
  ```python
  import asyncio
  from config.langvel import config

  # Add to config
  AGENT_TIMEOUT: int = int(os.getenv('AGENT_TIMEOUT', '300'))  # 5 minutes

  @app.post("/agents/{agent_path:path}")
  async def invoke_agent(agent_path: str, request: AgentRequest):
      agent = get_agent(agent_path)

      try:
          result = await asyncio.wait_for(
              agent.invoke(request.input, request.config),
              timeout=config.AGENT_TIMEOUT
          )
          return AgentResponse(output=result, metadata={...})

      except asyncio.TimeoutError:
          raise HTTPException(
              status_code=504,
              detail=f"Agent execution timeout after {config.AGENT_TIMEOUT}s"
          )
  ```
- **Effort**: 2 hours

#### ✅ TODO-007: Fix Rate Limiter Race Condition
- **File**: `langvel/middleware/base.py:58-102`
- **Severity**: CRITICAL (8/10)
- **Issue**: `_requests` dict modified concurrently without lock
- **Impact**: Rate limits can be bypassed, memory leak (dict grows forever)
- **Fix**:
  ```python
  import asyncio
  import time
  from collections import defaultdict

  class RateLimitMiddleware(Middleware):
      def __init__(self, max_requests: int = 10, window: int = 60):
          self.max_requests = max_requests
          self.window = window
          self._requests: Dict[str, list] = defaultdict(list)
          self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
          self._cleanup_task = None

      async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
          user_id = state.get('user_id', 'anonymous')
          current_time = time.time()

          # Use lock per user
          async with self._locks[user_id]:
              # Clean old requests
              self._requests[user_id] = [
                  req_time for req_time in self._requests[user_id]
                  if current_time - req_time < self.window
              ]

              # Check limit
              if len(self._requests[user_id]) >= self.max_requests:
                  raise Exception(
                      f"Rate limit exceeded: {self.max_requests} requests per {self.window}s"
                  )

              # Add current request
              self._requests[user_id].append(current_time)

          return state

      async def _cleanup_old_users(self):
          """Periodic cleanup to prevent memory leak."""
          while True:
              await asyncio.sleep(3600)  # Every hour
              current_time = time.time()

              # Remove users with no recent activity
              to_remove = [
                  user_id for user_id, requests in self._requests.items()
                  if not requests or current_time - requests[-1] > self.window * 2
              ]

              for user_id in to_remove:
                  del self._requests[user_id]
                  del self._locks[user_id]
  ```
- **Effort**: 3 hours

---

## ⚠️ P1 - HIGH PRIORITY (FIX WITHIN 2 WEEKS)

### **STABILITY & RELIABILITY**

#### ✅ TODO-008: Fix MCP Stderr Deadlock
- **File**: `langvel/mcp/manager.py:134-154`
- **Severity**: HIGH (7/10)
- **Issue**: Stderr not read, can cause deadlock if buffer fills
- **Impact**: MCP servers hang after writing >64KB to stderr
- **Fix**:
  ```python
  async def start(self):
      self.process = await asyncio.create_subprocess_exec(...)

      # Start stderr reader
      asyncio.create_task(self._read_stderr())

      # Wait for actual readiness instead of fixed delay
      await self._wait_for_ready(timeout=10)

  async def _read_stderr(self):
      """Continuously read stderr to prevent buffer filling."""
      while self.process and self.process.returncode is None:
          try:
              line = await self.process.stderr.readline()
              if not line:
                  break

              # Log stderr output
              logger.warning(
                  f"[MCP {self.name}] {line.decode().strip()}",
                  extra={'mcp_server': self.name}
              )
          except Exception as e:
              logger.error(f"Error reading MCP stderr: {e}")
              break

  async def _wait_for_ready(self, timeout: float = 10):
      """Wait for server to respond to ping."""
      start = time.time()
      while time.time() - start < timeout:
          try:
              # Try to list tools as health check
              await self.list_tools()
              return
          except:
              await asyncio.sleep(0.1)
      raise TimeoutError(f"MCP server {self.name} failed to start within {timeout}s")
  ```
- **Effort**: 4 hours

#### ✅ TODO-009: Add Backpressure to Message Bus
- **File**: `langvel/multiagent/communication.py:58`
- **Severity**: HIGH (7/10)
- **Issue**: Unbounded queue can grow infinitely
- **Impact**: Memory exhaustion if producer faster than consumer
- **Fix**:
  ```python
  def __init__(self):
      self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)  # Bounded
      self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
      self._agent_handlers: Dict[str, Callable] = {}
      self._running = False
      self._message_history: List[AgentMessage] = []
      self._max_history = 1000

      # Auto-start on first message
      self._started = False

  async def send(self, sender_id, recipient_id, content, ...):
      # Auto-start if not running
      if not self._started:
          await self.start()
          self._started = True

      message = AgentMessage(...)

      try:
          # Will block if queue full (backpressure)
          await asyncio.wait_for(
              self._message_queue.put(message),
              timeout=30.0
          )
      except asyncio.TimeoutError:
          logger.error(
              f"Message queue full, dropping message from {sender_id} to {recipient_id}"
          )
          raise Exception("Message queue full, system overloaded")
  ```
- **Effort**: 3 hours

#### ✅ TODO-010: Move Auth Storage to Redis/Database
- **File**: `langvel/auth/manager.py:43-44`
- **Severity**: HIGH (7/10)
- **Issue**: API keys and sessions stored in memory only
- **Impact**: Lost on restart, doesn't work with multiple workers
- **Fix**:
  ```python
  class AuthManager:
      def __init__(self, redis_client=None, secret_key=None, ...):
          self.redis = redis_client or self._get_redis_client()
          # ... rest of init

      def _get_redis_client(self):
          from config.langvel import config
          import redis.asyncio as redis
          return redis.from_url(config.REDIS_URL)

      async def create_api_key(self, name, permissions=None, metadata=None):
          api_key = f"lv_{secrets.token_urlsafe(32)}"
          key_hash = hashlib.sha256(api_key.encode()).hexdigest()

          # Store in Redis instead of memory
          await self.redis.hset(
              f"api_key:{key_hash}",
              mapping={
                  'name': name,
                  'permissions': json.dumps(permissions or []),
                  'metadata': json.dumps(metadata or {}),
                  'created_at': datetime.utcnow().isoformat(),
                  'last_used': '',
                  'usage_count': '0'
              }
          )

          return api_key

      async def verify_api_key(self, api_key: str):
          key_hash = hashlib.sha256(api_key.encode()).hexdigest()

          key_data = await self.redis.hgetall(f"api_key:{key_hash}")
          if not key_data:
              raise AuthenticationError("Invalid API key")

          # Update usage stats
          await self.redis.hincrby(f"api_key:{key_hash}", 'usage_count', 1)
          await self.redis.hset(
              f"api_key:{key_hash}",
              'last_used',
              datetime.utcnow().isoformat()
          )

          return {
              'name': key_data['name'],
              'permissions': json.loads(key_data['permissions']),
              'metadata': json.loads(key_data['metadata']),
              'usage_count': int(key_data['usage_count']) + 1
          }

      # Similar implementations for sessions
  ```
- **Effort**: 6 hours

#### ✅ TODO-011: Add JWT Token Revocation
- **File**: `langvel/auth/manager.py:81-105`
- **Severity**: MEDIUM (6/10)
- **Issue**: No way to revoke compromised tokens before expiry
- **Impact**: Security risk, can't invalidate stolen tokens
- **Fix**:
  ```python
  def create_token(self, user_id, permissions=None, metadata=None):
      now = datetime.utcnow()
      jti = secrets.token_urlsafe(16)  # JWT ID for revocation

      payload = {
          'user_id': user_id,
          'permissions': permissions or [],
          'jti': jti,  # Add JWT ID
          'iat': now,
          'exp': now + timedelta(seconds=self.token_expiry),
          'metadata': metadata or {}
      }

      return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

  async def verify_token(self, token: str):
      try:
          payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

          # Check revocation list
          jti = payload.get('jti')
          if jti:
              is_revoked = await self.redis.sismember("revoked_tokens", jti)
              if is_revoked:
                  raise AuthenticationError("Token has been revoked")

          return payload
      except jwt.ExpiredSignatureError:
          raise AuthenticationError("Token has expired")
      except jwt.InvalidTokenError as e:
          raise AuthenticationError(f"Invalid token: {str(e)}")

  async def revoke_token(self, token: str):
      """Revoke a token by adding its JTI to blacklist."""
      try:
          payload = jwt.decode(
              token,
              self.secret_key,
              algorithms=[self.algorithm],
              options={"verify_exp": False}  # Allow checking expired tokens
          )

          jti = payload.get('jti')
          if not jti:
              raise ValueError("Token missing JTI, cannot revoke")

          # Add to Redis set with expiry
          exp = payload.get('exp')
          if exp:
              ttl = exp - time.time()
              if ttl > 0:
                  await self.redis.sadd("revoked_tokens", jti)
                  await self.redis.expire("revoked_tokens", int(ttl))

      except jwt.InvalidTokenError as e:
          raise AuthenticationError(f"Cannot revoke invalid token: {str(e)}")
  ```
- **Effort**: 4 hours

#### ✅ TODO-012: Add Retry Logic to RAG Queries
- **File**: `langvel/rag/manager.py:42-96`
- **Severity**: MEDIUM (6/10)
- **Issue**: Single network error fails entire RAG query
- **Impact**: Brittle system, poor user experience
- **Fix**:
  ```python
  async def retrieve(self, collection, query, k=5, similarity_threshold=None, **kwargs):
      if collection not in self._collections:
          raise ValueError(f"Collection '{collection}' not found")

      vector_store = self._collections[collection]

      # Retry with exponential backoff
      max_retries = 3
      for attempt in range(max_retries):
          try:
              # Perform similarity search
              if similarity_threshold is not None:
                  results = await vector_store.asimilarity_search_with_relevance_scores(
                      query, k=k, **kwargs
                  )
                  results = [
                      (doc, score) for doc, score in results
                      if score >= similarity_threshold
                  ]
              else:
                  docs = await vector_store.asimilarity_search(query, k=k, **kwargs)
                  results = [(doc, None) for doc in docs]

              # Format and return results
              return [
                  {
                      'content': doc.page_content,
                      'metadata': doc.metadata,
                      'score': score
                  }
                  for doc, score in results
              ]

          except Exception as e:
              if attempt < max_retries - 1:
                  wait_time = 2 ** attempt  # Exponential backoff
                  logger.warning(
                      f"RAG retrieval failed (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {wait_time}s: {e}"
                  )
                  await asyncio.sleep(wait_time)
              else:
                  logger.error(f"RAG retrieval failed after {max_retries} attempts: {e}")
                  raise
  ```
- **Effort**: 3 hours

### **LOGGING & OBSERVABILITY**

#### ✅ TODO-013: Replace Print Statements with Structured Logging
- **File**: Multiple files
- **Severity**: MEDIUM (6/10)
- **Issue**: Using `print()` for logging, not production-ready
- **Impact**: No log levels, rotation, centralization
- **Fix**:
  ```python
  # Create langvel/logging.py
  import logging
  import json
  from datetime import datetime
  from typing import Any, Dict

  class JSONFormatter(logging.Formatter):
      def format(self, record: logging.LogRecord) -> str:
          log_data = {
              'timestamp': datetime.utcnow().isoformat(),
              'level': record.levelname,
              'logger': record.name,
              'message': record.getMessage(),
              'module': record.module,
              'function': record.funcName,
              'line': record.lineno
          }

          # Add extra fields
          if hasattr(record, 'extra'):
              log_data.update(record.extra)

          if record.exc_info:
              log_data['exception'] = self.formatException(record.exc_info)

          return json.dumps(log_data)

  def setup_logging(log_level: str = "INFO", log_file: str = None):
      """Setup structured JSON logging."""
      logger = logging.getLogger('langvel')
      logger.setLevel(log_level)

      # Console handler
      console_handler = logging.StreamHandler()
      console_handler.setFormatter(JSONFormatter())
      logger.addHandler(console_handler)

      # File handler if specified
      if log_file:
          file_handler = logging.FileHandler(log_file)
          file_handler.setFormatter(JSONFormatter())
          logger.addHandler(file_handler)

      return logger

  # Replace all print() statements:
  # langvel/middleware/base.py:110
  logger = logging.getLogger('langvel.middleware')
  logger.info("Input state", extra={'state': state})

  # langvel/observability/tracer.py:42
  logger = logging.getLogger('langvel.observability')
  logger.info("LangSmith tracing enabled")

  # langvel/multiagent/communication.py:85
  logger = logging.getLogger('langvel.multiagent')
  logger.error("Error processing message", exc_info=e, extra={'message_id': message.id})
  ```
- **Files to update**:
  - `langvel/middleware/base.py:110, 116`
  - `langvel/observability/tracer.py:42, 64, 110, 122, 161, 175, 232, 250`
  - `langvel/multiagent/communication.py:85, 100, 108, 197`
- **Effort**: 6 hours

#### ✅ TODO-014: Add Health Check Endpoints
- **File**: `langvel/server.py:52-55`
- **Severity**: MEDIUM (6/10)
- **Issue**: Health check only returns static response
- **Impact**: Can't detect service degradation, load balancers can't route properly
- **Fix**:
  ```python
  from typing import Dict, Any

  async def check_postgres() -> Dict[str, Any]:
      """Check PostgreSQL connection."""
      try:
          from langvel.state.checkpointers import PostgresCheckpointer
          checkpointer = PostgresCheckpointer()
          pool = await checkpointer._get_pool()
          async with pool.acquire() as conn:
              await conn.execute('SELECT 1')
          return {"status": "healthy"}
      except Exception as e:
          return {"status": "unhealthy", "error": str(e)}

  async def check_redis() -> Dict[str, Any]:
      """Check Redis connection."""
      try:
          from langvel.state.checkpointers import RedisCheckpointer
          checkpointer = RedisCheckpointer()
          await checkpointer._ensure_setup()
          await checkpointer._client.ping()
          return {"status": "healthy"}
      except Exception as e:
          return {"status": "unhealthy", "error": str(e)}

  async def check_mcp_servers() -> Dict[str, Any]:
      """Check MCP servers status."""
      from langvel.mcp.manager import get_mcp_manager
      manager = get_mcp_manager()

      results = {}
      for name, server in manager._servers.items():
          try:
              # Try to list tools as health check
              await asyncio.wait_for(server.list_tools(), timeout=5.0)
              results[name] = {"status": "healthy"}
          except Exception as e:
              results[name] = {"status": "unhealthy", "error": str(e)}

      return results

  @app.get("/health")
  async def health():
      """Detailed health check."""
      checks = {
          "postgres": await check_postgres() if config.STATE_CHECKPOINTER == "postgres" else {"status": "not_configured"},
          "redis": await check_redis() if config.STATE_CHECKPOINTER == "redis" else {"status": "not_configured"},
          "mcp_servers": await check_mcp_servers()
      }

      # Overall status
      all_healthy = all(
          check.get("status") == "healthy" or check.get("status") == "not_configured"
          for check in checks.values()
          if isinstance(check, dict)
      )

      status_code = 200 if all_healthy else 503

      return JSONResponse(
          status_code=status_code,
          content={
              "status": "healthy" if all_healthy else "degraded",
              "timestamp": datetime.utcnow().isoformat(),
              "checks": checks
          }
      )

  @app.get("/health/ready")
  async def readiness():
      """Readiness probe for Kubernetes."""
      # Check if essential services are ready
      try:
          # Check agent pool initialized
          if not _agent_pool:
              return JSONResponse(
                  status_code=503,
                  content={"status": "not_ready", "reason": "No agents loaded"}
              )

          return {"status": "ready"}
      except Exception as e:
          return JSONResponse(
              status_code=503,
              content={"status": "not_ready", "error": str(e)}
          )

  @app.get("/health/live")
  async def liveness():
      """Liveness probe for Kubernetes."""
      # Simple check that server is responding
      return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
  ```
- **Effort**: 4 hours

---

## 📋 P2 - MEDIUM PRIORITY (FIX WITHIN 1 MONTH)

### **PERFORMANCE & OPTIMIZATION**

#### ✅ TODO-015: Fix LLM Client Singleton Issue
- **File**: `langvel/llm/manager.py:28-57`
- **Severity**: MEDIUM (6/10)
- **Issue**: Client cached with first config, can't use different models dynamically
- **Impact**: Can't vary temperature, model, or other params per request
- **Fix**:
  ```python
  from functools import lru_cache

  class LLMManager:
      def __init__(self, provider="anthropic", model=None, **kwargs):
          self.provider = provider
          self.default_model = model
          self.default_kwargs = kwargs

      @lru_cache(maxsize=10)
      def _get_client(self, model: str, temperature: float, max_tokens: int):
          """Get cached client for specific config."""
          if self.provider == "anthropic":
              from langchain_anthropic import ChatAnthropic
              return ChatAnthropic(
                  model=model,
                  temperature=temperature,
                  max_tokens=max_tokens
              )
          elif self.provider == "openai":
              from langchain_openai import ChatOpenAI
              return ChatOpenAI(
                  model=model,
                  temperature=temperature
              )
          else:
              raise ValueError(f"Unsupported provider: {self.provider}")

      async def invoke(self, prompt, system_prompt=None, model=None, temperature=None, **kwargs):
          # Use provided params or defaults
          model = model or self.default_model or self._get_default_model()
          temperature = temperature if temperature is not None else self.default_kwargs.get('temperature', 0.7)
          max_tokens = kwargs.get('max_tokens', self.default_kwargs.get('max_tokens', 4096))

          # Get client for this specific config
          client = self._get_client(model, temperature, max_tokens)

          # Build messages
          messages = []
          if system_prompt:
              messages.append({"role": "system", "content": system_prompt})
          messages.append({"role": "user", "content": prompt})

          # Invoke and track token usage
          response = await client.ainvoke(messages, **kwargs)

          # Log token usage if available
          if hasattr(response, 'usage_metadata'):
              logger.info(
                  "LLM invocation",
                  extra={
                      'model': model,
                      'prompt_tokens': response.usage_metadata.get('input_tokens', 0),
                      'completion_tokens': response.usage_metadata.get('output_tokens', 0),
                      'total_tokens': response.usage_metadata.get('total_tokens', 0)
                  }
              )

          return response.content
  ```
- **Effort**: 4 hours

#### ✅ TODO-016: Add LLM Response Caching
- **File**: `langvel/llm/manager.py:59`
- **Severity**: LOW (4/10)
- **Issue**: No caching of LLM responses, wasteful for repeated queries
- **Impact**: Unnecessary costs and latency
- **Fix**:
  ```python
  import hashlib
  from typing import Optional

  class LLMManager:
      def __init__(self, provider="anthropic", model=None, cache_backend=None, **kwargs):
          self.provider = provider
          self.default_model = model
          self.default_kwargs = kwargs
          self.cache = cache_backend or self._get_cache_backend()

      def _get_cache_backend(self):
          """Get Redis cache or in-memory fallback."""
          try:
              from config.langvel import config
              import redis.asyncio as redis
              return redis.from_url(config.REDIS_URL)
          except:
              # Fallback to in-memory LRU cache
              from functools import lru_cache
              return {}

      def _cache_key(self, prompt: str, system_prompt: Optional[str], model: str, temperature: float, **kwargs) -> str:
          """Generate cache key for request."""
          cache_parts = [
              prompt,
              system_prompt or "",
              model,
              str(temperature),
              json.dumps(sorted(kwargs.items()))
          ]
          cache_str = "|".join(cache_parts)
          return f"llm_cache:{hashlib.md5(cache_str.encode()).hexdigest()}"

      async def invoke(self, prompt, system_prompt=None, model=None, temperature=None, use_cache=True, **kwargs):
          model = model or self.default_model
          temperature = temperature if temperature is not None else 0.7

          # Check cache if enabled
          if use_cache:
              cache_key = self._cache_key(prompt, system_prompt, model, temperature, **kwargs)

              if isinstance(self.cache, dict):
                  # In-memory cache
                  if cache_key in self.cache:
                      logger.debug(f"LLM cache hit: {cache_key}")
                      return self.cache[cache_key]
              else:
                  # Redis cache
                  cached = await self.cache.get(cache_key)
                  if cached:
                      logger.debug(f"LLM cache hit: {cache_key}")
                      return cached.decode()

          # Get client and invoke
          client = self._get_client(model, temperature, kwargs.get('max_tokens', 4096))
          messages = []
          if system_prompt:
              messages.append({"role": "system", "content": system_prompt})
          messages.append({"role": "user", "content": prompt})

          response = await client.ainvoke(messages, **kwargs)
          result = response.content

          # Store in cache
          if use_cache:
              cache_key = self._cache_key(prompt, system_prompt, model, temperature, **kwargs)

              if isinstance(self.cache, dict):
                  # In-memory cache (simple, no TTL)
                  if len(self.cache) > 1000:  # Limit size
                      self.cache.clear()
                  self.cache[cache_key] = result
              else:
                  # Redis cache with 1 hour TTL
                  await self.cache.setex(cache_key, 3600, result)

          return result
  ```
- **Effort**: 4 hours

#### ✅ TODO-017: Add Connection Pooling for RAG Vector Stores
- **File**: `langvel/rag/manager.py:1`
- **Severity**: MEDIUM (5/10)
- **Issue**: No connection pooling, creates new connection per query
- **Impact**: Poor performance, connection exhaustion
- **Fix**:
  ```python
  class RAGManager:
      def __init__(self):
          self._collections: Dict[str, VectorStore] = {}
          self._embeddings: Optional[Embeddings] = None
          self._connection_pools: Dict[str, Any] = {}

      def register_collection(self, name: str, vector_store: VectorStore):
          """Register collection with connection pooling."""
          self._collections[name] = vector_store

          # Initialize connection pool if backend supports it
          if hasattr(vector_store, '_client'):
              # For Chroma
              if hasattr(vector_store._client, '_pool'):
                  self._connection_pools[name] = vector_store._client._pool

          # For Pinecone, connection pooling handled by SDK

      async def retrieve(self, collection, query, k=5, ...):
          # Existing retrieval logic, but connections are pooled
          vector_store = self._collections[collection]

          # Connection reused from pool automatically
          results = await vector_store.asimilarity_search(query, k=k, **kwargs)

          return formatted_results
  ```
- **Note**: Connection pooling is often handled by the vector store SDK itself. Verify implementation.
- **Effort**: 3 hours

#### ✅ TODO-018: Add Request Queuing with Semaphore
- **File**: `langvel/server.py:1`
- **Severity**: MEDIUM (5/10)
- **Issue**: No limit on concurrent requests
- **Impact**: Can be overwhelmed under high load
- **Fix**:
  ```python
  from asyncio import Semaphore
  from config.langvel import config

  # Add to config
  MAX_CONCURRENT_REQUESTS: int = int(os.getenv('MAX_CONCURRENT_REQUESTS', '100'))

  # Global semaphore
  _request_limiter = Semaphore(config.MAX_CONCURRENT_REQUESTS)

  @app.post("/agents/{agent_path:path}")
  async def invoke_agent(agent_path: str, request: AgentRequest):
      # Acquire semaphore (will block if limit reached)
      async with _request_limiter:
          agent = get_agent(agent_path)

          result = await asyncio.wait_for(
              agent.invoke(request.input, request.config),
              timeout=config.AGENT_TIMEOUT
          )

          return AgentResponse(output=result, metadata={...})

  @app.get("/metrics/concurrency")
  async def get_concurrency_metrics():
      """Expose concurrency metrics."""
      return {
          "max_concurrent": config.MAX_CONCURRENT_REQUESTS,
          "current_active": config.MAX_CONCURRENT_REQUESTS - _request_limiter._value,
          "available": _request_limiter._value
      }
  ```
- **Effort**: 2 hours

### **CONFIGURATION & VALIDATION**

#### ✅ TODO-019: Add Configuration Validation
- **File**: `config/langvel.py:1`
- **Severity**: MEDIUM (5/10)
- **Issue**: No validation of config values
- **Impact**: Invalid configs cause runtime errors
- **Fix**:
  ```python
  from pydantic import BaseSettings, validator, Field
  from typing import List

  class LangvelConfig(BaseSettings):
      """Validated configuration using Pydantic."""

      # LLM Configuration
      LLM_PROVIDER: str = Field("anthropic", pattern="^(anthropic|openai)$")
      LLM_MODEL: str = "claude-3-5-sonnet-20241022"
      LLM_TEMPERATURE: float = Field(0.7, ge=0.0, le=1.0)
      LLM_MAX_TOKENS: int = Field(4096, gt=0, le=100000)

      # API Keys
      ANTHROPIC_API_KEY: str = ""
      OPENAI_API_KEY: str = ""

      # Validate API key matches provider
      @validator('ANTHROPIC_API_KEY')
      def validate_anthropic_key(cls, v, values):
          if values.get('LLM_PROVIDER') == 'anthropic' and not v:
              raise ValueError("ANTHROPIC_API_KEY required when LLM_PROVIDER=anthropic")
          return v

      @validator('OPENAI_API_KEY')
      def validate_openai_key(cls, v, values):
          if values.get('LLM_PROVIDER') == 'openai' and not v:
              raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")
          return v

      # RAG Configuration
      RAG_PROVIDER: str = Field("chroma", pattern="^(chroma|pinecone)$")
      RAG_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
      RAG_PERSIST_DIRECTORY: str = "./storage/chroma_db"

      # State Management
      STATE_CHECKPOINTER: str = Field("memory", pattern="^(memory|postgres|redis)$")
      DATABASE_URL: str = "postgresql://localhost/langvel"
      DATABASE_POOL_SIZE: int = Field(10, ge=1, le=100)
      REDIS_URL: str = "redis://localhost:6379"

      # Server Configuration
      SERVER_HOST: str = "0.0.0.0"
      SERVER_PORT: int = Field(8000, ge=1, le=65535)
      SERVER_WORKERS: int = Field(1, ge=1, le=32)
      AGENT_TIMEOUT: int = Field(300, ge=1, le=3600)
      MAX_CONCURRENT_REQUESTS: int = Field(100, ge=1, le=10000)

      # Security
      JWT_SECRET_KEY: str  # Required, no default
      CORS_ORIGINS: List[str] = ["*"]

      @validator('CORS_ORIGINS')
      def validate_cors(cls, v):
          if "*" in v and len(v) > 1:
              raise ValueError("CORS_ORIGINS cannot contain '*' with other origins")
          return v

      RATE_LIMIT_REQUESTS: int = Field(10, ge=1)
      RATE_LIMIT_WINDOW: int = Field(60, ge=1)

      # Logging
      LOG_LEVEL: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
      LOG_FILE: str = "./storage/logs/langvel.log"

      # Observability
      LANGSMITH_API_KEY: str = ""
      LANGSMITH_PROJECT: str = "langvel"
      LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

      LANGFUSE_PUBLIC_KEY: str = ""
      LANGFUSE_SECRET_KEY: str = ""
      LANGFUSE_HOST: str = "https://cloud.langfuse.com"

      # Development
      DEBUG: bool = False
      RELOAD: bool = False

      class Config:
          env_file = ".env"
          case_sensitive = True

  # Load and validate config
  try:
      config = LangvelConfig()
  except Exception as e:
      print(f"Configuration error: {e}")
      sys.exit(1)
  ```
- **Effort**: 4 hours

#### ✅ TODO-020: Remove CORS Wildcard Default
- **File**: `config/langvel.py:67`
- **Severity**: MEDIUM (5/10)
- **Issue**: Default `CORS_ORIGINS=*` is insecure for production
- **Impact**: Security risk, CSRF vulnerability
- **Fix**:
  ```python
  # In config
  CORS_ORIGINS: List[str] = []  # Empty by default, must be explicitly set

  @validator('CORS_ORIGINS')
  def validate_cors_in_production(cls, v, values):
      debug = values.get('DEBUG', False)

      if not debug and '*' in v:
          raise ValueError(
              "CORS wildcard '*' not allowed in production. "
              "Set DEBUG=true for development or specify allowed origins."
          )

      if not v and not debug:
          logger.warning(
              "CORS_ORIGINS not set. API will reject all cross-origin requests. "
              "Set CORS_ORIGINS environment variable with comma-separated origins."
          )

      return v or []

  # In server.py
  if config.CORS_ORIGINS:
      app.add_middleware(
          CORSMiddleware,
          allow_origins=config.CORS_ORIGINS,
          allow_credentials=True,
          allow_methods=["*"],
          allow_headers=["*"],
      )
  ```
- **Effort**: 1 hour

### **CODE QUALITY & BUGS**

#### ✅ TODO-021: Fix Graph Builder Parallel Branch Merging
- **File**: `langvel/routing/builder.py:106-129`
- **Severity**: LOW (4/10)
- **Issue**: Parallel branches don't auto-merge, can cause hanging graphs
- **Impact**: User confusion, requires explicit `.merge()` or `.end()`
- **Fix Options**:

  **Option 1: Auto-merge to END**
  ```python
  def parallel(self, *funcs: Callable) -> "GraphBuilder":
      """Execute multiple nodes in parallel and auto-merge to END."""
      parallel_node_names = []

      for func in funcs:
          node_name = func.__name__
          self.nodes.append((node_name, func))
          parallel_node_names.append(node_name)

          # Connect from current node to each parallel node
          if self.current_node:
              self.edges.append((self.current_node, node_name))

      # Auto-connect all parallel nodes to END
      for node_name in parallel_node_names:
          self.edges.append((node_name, END))

      # Set current_node to None (graph ends after parallel)
      self.current_node = None
      return self
  ```

  **Option 2: Better documentation**
  ```python
  def parallel(self, *funcs: Callable) -> "GraphBuilder":
      """
      Execute multiple nodes in parallel.

      IMPORTANT: After parallel execution, you must explicitly:
      - Call .merge() to combine branches, or
      - Call .end() to terminate all branches

      Example:
          .parallel(func_a, func_b, func_c)
          .merge(combine_results)
          .end()

      Or:
          .parallel(func_a, func_b, func_c)
          .end()  # All branches terminate
      """
      # Existing implementation...
  ```

  **Recommended: Option 1 (auto-merge to END)**
- **Effort**: 2 hours

#### ✅ TODO-022: Fix Branch Condition Default Fallback
- **File**: `langvel/routing/builder.py:66-70`
- **Severity**: LOW (4/10)
- **Issue**: Silently falls back to first branch if `next_step` missing
- **Impact**: Incorrect routing without error
- **Fix**:
  ```python
  def branch(self, conditions: Dict[str, Callable], condition_func: Optional[Callable] = None):
      """Add conditional branch with validation."""
      if condition_func is None:
          def auto_condition(state):
              next_step = state.get('next_step')

              if next_step is None:
                  raise ValueError(
                      f"Branch requires 'next_step' in state. "
                      f"Available branches: {list(conditions.keys())}"
                  )

              if next_step not in conditions:
                  raise ValueError(
                      f"Invalid branch '{next_step}'. "
                      f"Available branches: {list(conditions.keys())}"
                  )

              return next_step

          condition_func = auto_condition

      # Rest of implementation...
  ```
- **Effort**: 2 hours

#### ✅ TODO-023: Fix Auth Permission Wildcard Bug
- **File**: `langvel/auth/manager.py:300-328`
- **Severity**: LOW (4/10)
- **Issue**: `admin.*` doesn't match `admin`, only `admin.something`
- **Impact**: Permission checks incorrect
- **Fix**:
  ```python
  def has_permission(self, user_permissions: List[str], required_permission: str) -> bool:
      """
      Check if user has required permission.

      Supports wildcards:
      - 'admin.*' matches 'admin', 'admin.read', 'admin.write', etc.
      - 'api.*.write' matches 'api.users.write', 'api.posts.write', etc.
      """
      # Direct match
      if required_permission in user_permissions:
          return True

      # Check wildcard permissions
      for perm in user_permissions:
          if '.*' in perm:
              # Remove .* suffix
              prefix = perm.replace('.*', '')

              # Match exact prefix or prefix.*
              if required_permission == prefix or required_permission.startswith(f"{prefix}."):
                  return True

      return False
  ```
- **Effort**: 2 hours

#### ✅ TODO-024: Fix Rate Limiter Decorator Shared State
- **File**: `langvel/auth/decorators.py:70-118`
- **Severity**: MEDIUM (5/10)
- **Issue**: All rate-limited functions share same `request_history` dict
- **Impact**: Rate limits mixed across functions
- **Fix**:
  ```python
  def rate_limit(max_requests: int = 10, window: int = 60) -> Callable:
      """
      Decorator to add rate limiting to an agent node.

      Each decorated function gets its own rate limiter.
      """
      def decorator(func: Callable) -> Callable:
          # Create separate history for THIS function
          request_history = {}
          lock = asyncio.Lock()

          @wraps(func)
          async def wrapper(*args, **kwargs):
              state = args[1] if len(args) > 1 else kwargs.get('state')
              user_id = getattr(state, 'user_id', 'anonymous')
              current_time = time.time()

              async with lock:
                  # Initialize tracking
                  if user_id not in request_history:
                      request_history[user_id] = []

                  # Clean old requests
                  request_history[user_id] = [
                      req_time for req_time in request_history[user_id]
                      if current_time - req_time < window
                  ]

                  # Check limit
                  if len(request_history[user_id]) >= max_requests:
                      raise Exception(
                          f"Rate limit exceeded for {func.__name__}: "
                          f"{max_requests} requests per {window}s"
                      )

                  # Add current request
                  request_history[user_id].append(current_time)

              return await func(*args, **kwargs)

          wrapper._rate_limit = {'max_requests': max_requests, 'window': window}
          return wrapper

      return decorator
  ```
- **Effort**: 2 hours

#### ✅ TODO-025: Fix RAG Anthropic Embeddings Silent Fallback
- **File**: `langvel/rag/manager.py:188-192`
- **Severity**: LOW (3/10)
- **Issue**: Silently uses OpenAI when Anthropic requested
- **Impact**: Unexpected behavior, cost implications
- **Fix**:
  ```python
  def _create_embeddings(self) -> Embeddings:
      """Create embeddings instance based on configuration."""
      if self.embedding_model.startswith('openai/'):
          from langchain_openai import OpenAIEmbeddings
          model_name = self.embedding_model.replace('openai/', '')
          return OpenAIEmbeddings(model=model_name)

      elif self.embedding_model.startswith('anthropic/'):
          # Anthropic doesn't have native embeddings
          raise ValueError(
              "Anthropic embeddings not yet supported. "
              "Use 'openai/text-embedding-3-small' or another provider. "
              "See: https://docs.anthropic.com/claude/docs/embeddings"
          )

      elif self.embedding_model.startswith('cohere/'):
          from langchain_cohere import CohereEmbeddings
          model_name = self.embedding_model.replace('cohere/', '')
          return CohereEmbeddings(model=model_name)

      else:
          raise ValueError(
              f"Unsupported embedding model: {self.embedding_model}. "
              f"Supported providers: openai/*, cohere/*"
          )
  ```
- **Effort**: 1 hour

#### ✅ TODO-026: Add MCP JSON-RPC Response Validation
- **File**: `langvel/mcp/manager.py:207-229`
- **Severity**: MEDIUM (5/10)
- **Issue**: JSON-RPC responses not validated
- **Impact**: Can fail silently or incorrectly
- **Fix**:
  ```python
  async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
      """Send validated JSON-RPC request and validate response."""
      if not self.process or not self.process.stdin or not self.process.stdout:
          raise RuntimeError(f"MCP server {self.name} not running")

      # Validate request
      if 'jsonrpc' not in request:
          request['jsonrpc'] = '2.0'
      if 'id' not in request:
          request['id'] = id(request)

      # Send request
      request_str = json.dumps(request) + "\n"
      self.process.stdin.write(request_str.encode())
      await self.process.stdin.drain()

      # Read response with timeout
      try:
          response_str = await asyncio.wait_for(
              self.process.stdout.readline(),
              timeout=30.0
          )
      except asyncio.TimeoutError:
          raise RuntimeError(
              f"MCP server {self.name} did not respond within 30s. "
              f"Request: {request.get('method')}"
          )

      # Parse response
      try:
          response = json.loads(response_str.decode())
      except json.JSONDecodeError as e:
          raise RuntimeError(
              f"Invalid JSON response from MCP server {self.name}: {e}. "
              f"Response: {response_str[:100]}"
          )

      # Validate JSON-RPC response
      if response.get('jsonrpc') != '2.0':
          raise RuntimeError(
              f"Invalid JSON-RPC version from MCP server {self.name}: "
              f"{response.get('jsonrpc')}"
          )

      # Check for error
      if 'error' in response:
          error = response['error']
          raise RuntimeError(
              f"MCP server {self.name} returned error: "
              f"[{error.get('code')}] {error.get('message')}"
          )

      # Verify ID matches
      if response.get('id') != request.get('id'):
          logger.warning(
              f"Response ID mismatch from MCP server {self.name}: "
              f"expected {request.get('id')}, got {response.get('id')}"
          )

      return response
  ```
- **Effort**: 3 hours

---

## 🧪 P3 - TESTING & DOCUMENTATION

#### ✅ TODO-027: Write Comprehensive Test Suite
- **Files**: `tests/`
- **Severity**: CRITICAL (Long-term)
- **Issue**: Only basic setup tests exist
- **Target Coverage**: >80%
- **Tests Needed**:

  **Unit Tests**:
  ```
  tests/unit/
  ├── test_agent.py           # Agent creation, compilation, execution
  ├── test_graph_builder.py   # Graph building, branches, parallel
  ├── test_auth.py             # JWT, API keys, permissions
  ├── test_middleware.py       # Rate limiting, logging, validation
  ├── test_tools.py            # Tool decorators, registry, execution
  ├── test_rag.py              # RAG retrieval, embeddings
  ├── test_mcp.py              # MCP server communication
  ├── test_llm.py              # LLM manager, caching
  └── test_multiagent.py       # Message bus, coordinator
  ```

  **Integration Tests**:
  ```
  tests/integration/
  ├── test_server.py           # API endpoints, streaming
  ├── test_checkpointers.py    # PostgreSQL, Redis persistence
  ├── test_agent_flow.py       # End-to-end agent execution
  └── test_mcp_integration.py  # Real MCP server interaction
  ```

  **Performance Tests**:
  ```
  tests/performance/
  ├── test_load.py             # Load testing with k6
  ├── test_memory.py           # Memory leak detection
  └── test_concurrency.py      # Race condition testing
  ```

- **Example Test**:
  ```python
  # tests/unit/test_agent.py
  import pytest
  from langvel.core.agent import Agent
  from langvel.state.base import StateModel

  class TestState(StateModel):
      query: str
      response: str = ""

  class TestAgent(Agent):
      state_model = TestState

      def build_graph(self):
          return self.start().then(self.process).end()

      async def process(self, state: TestState):
          state.response = f"Processed: {state.query}"
          return state

  @pytest.mark.asyncio
  async def test_agent_invoke():
      agent = TestAgent()
      result = await agent.invoke({"query": "test"})

      assert result["response"] == "Processed: test"
      assert "query" in result

  @pytest.mark.asyncio
  async def test_agent_timeout():
      class SlowAgent(Agent):
          state_model = TestState

          def build_graph(self):
              return self.start().then(self.slow_process).end()

          async def slow_process(self, state):
              await asyncio.sleep(10)  # Longer than timeout
              return state

      agent = SlowAgent()

      with pytest.raises(asyncio.TimeoutError):
          await asyncio.wait_for(
              agent.invoke({"query": "test"}),
              timeout=1.0
          )
  ```

- **Effort**: 40 hours (full test suite)

#### ✅ TODO-028: Add Prometheus Metrics Export
- **File**: New file `langvel/metrics.py`
- **Severity**: MEDIUM (6/10)
- **Issue**: No metrics for monitoring
- **Impact**: Can't detect performance issues
- **Fix**:
  ```python
  # langvel/metrics.py
  from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
  from fastapi import Response

  # Request metrics
  agent_requests_total = Counter(
      'langvel_agent_requests_total',
      'Total agent requests',
      ['agent', 'status']
  )

  agent_request_duration_seconds = Histogram(
      'langvel_agent_request_duration_seconds',
      'Agent request duration',
      ['agent'],
      buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
  )

  active_requests = Gauge(
      'langvel_active_requests',
      'Currently active requests'
  )

  # LLM metrics
  llm_requests_total = Counter(
      'langvel_llm_requests_total',
      'Total LLM requests',
      ['provider', 'model']
  )

  llm_tokens_total = Counter(
      'langvel_llm_tokens_total',
      'Total LLM tokens used',
      ['provider', 'model', 'type']  # type: prompt/completion
  )

  llm_request_duration_seconds = Histogram(
      'langvel_llm_request_duration_seconds',
      'LLM request duration',
      ['provider', 'model']
  )

  # Tool metrics
  tool_executions_total = Counter(
      'langvel_tool_executions_total',
      'Total tool executions',
      ['tool', 'status']
  )

  tool_execution_duration_seconds = Histogram(
      'langvel_tool_execution_duration_seconds',
      'Tool execution duration',
      ['tool']
  )

  # MCP metrics
  mcp_calls_total = Counter(
      'langvel_mcp_calls_total',
      'Total MCP calls',
      ['server', 'tool', 'status']
  )

  # In server.py
  from langvel.metrics import (
      agent_requests_total,
      agent_request_duration_seconds,
      active_requests,
      generate_latest
  )

  @app.middleware("http")
  async def metrics_middleware(request: Request, call_next):
      # Skip metrics endpoint itself
      if request.url.path == "/metrics":
          return await call_next(request)

      # Track active requests
      active_requests.inc()

      # Extract agent from path
      agent_path = request.url.path.replace("/agents/", "").split("/")[0]

      start_time = time.time()

      try:
          response = await call_next(request)
          status = "success" if response.status_code < 400 else "error"

          # Record metrics
          duration = time.time() - start_time
          agent_requests_total.labels(agent=agent_path, status=status).inc()
          agent_request_duration_seconds.labels(agent=agent_path).observe(duration)

          return response

      except Exception as e:
          duration = time.time() - start_time
          agent_requests_total.labels(agent=agent_path, status="error").inc()
          agent_request_duration_seconds.labels(agent=agent_path).observe(duration)
          raise

      finally:
          active_requests.dec()

  @app.get("/metrics")
  async def metrics():
      """Prometheus metrics endpoint."""
      return Response(
          content=generate_latest(REGISTRY),
          media_type="text/plain"
      )
  ```
- **Effort**: 6 hours

#### ✅ TODO-029: Add Database Migrations (Alembic)
- **File**: New directory `migrations/`
- **Severity**: MEDIUM (5/10)
- **Issue**: Manual schema management
- **Impact**: Deployment fragility, version conflicts
- **Setup**:
  ```bash
  pip install alembic
  alembic init migrations
  ```

  **Configuration**:
  ```python
  # migrations/env.py
  from langvel.state.checkpointers import PostgresCheckpointer
  from config.langvel import config

  target_metadata = None  # We manage schema in code

  def run_migrations_online():
      from sqlalchemy import engine_from_config, pool

      connectable = engine_from_config(
          config.get_section(config.config_ini_section),
          prefix="sqlalchemy.",
          poolclass=pool.NullPool,
          url=config.DATABASE_URL
      )

      # ... rest of Alembic setup
  ```

  **Initial Migration**:
  ```python
  # migrations/versions/001_initial_schema.py
  from alembic import op
  import sqlalchemy as sa
  from sqlalchemy.dialects import postgresql

  def upgrade():
      op.create_table(
          'langvel_checkpoints',
          sa.Column('thread_id', sa.Text(), nullable=False),
          sa.Column('checkpoint_id', sa.Text(), nullable=False),
          sa.Column('parent_checkpoint_id', sa.Text(), nullable=True),
          sa.Column('checkpoint_data', postgresql.JSONB(), nullable=False),
          sa.Column('metadata', postgresql.JSONB(), nullable=True),
          sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
          sa.PrimaryKeyConstraint('thread_id', 'checkpoint_id')
      )

      op.create_index(
          'idx_checkpoints_thread',
          'langvel_checkpoints',
          ['thread_id', 'created_at'],
          postgresql_ops={'created_at': 'DESC'}
      )

      op.create_index(
          'idx_checkpoints_parent',
          'langvel_checkpoints',
          ['parent_checkpoint_id']
      )

  def downgrade():
      op.drop_table('langvel_checkpoints')
  ```
- **Effort**: 4 hours

#### ✅ TODO-030: Document All Configuration Options
- **File**: New file `docs/CONFIGURATION.md`
- **Severity**: LOW (3/10)
- **Issue**: Many config options undocumented
- **Content**: See template below
- **Effort**: 4 hours

---

## 🔄 ONGOING MAINTENANCE

#### ✅ TODO-031: Set Up CI/CD Pipeline
- **Platform**: GitHub Actions
- **Workflow**:
  ```yaml
  # .github/workflows/ci.yml
  name: CI

  on: [push, pull_request]

  jobs:
    test:
      runs-on: ubuntu-latest

      services:
        postgres:
          image: postgres:15
          env:
            POSTGRES_DB: langvel_test
            POSTGRES_PASSWORD: test
          options: >-
            --health-cmd pg_isready
            --health-interval 10s
            --health-timeout 5s
            --health-retries 5

        redis:
          image: redis:7
          options: >-
            --health-cmd "redis-cli ping"
            --health-interval 10s
            --health-timeout 5s
            --health-retries 5

      steps:
        - uses: actions/checkout@v3

        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.11'

        - name: Install dependencies
          run: |
            pip install -e .
            pip install pytest pytest-asyncio pytest-cov

        - name: Run tests
          env:
            DATABASE_URL: postgresql://postgres:test@localhost/langvel_test
            REDIS_URL: redis://localhost:6379
            JWT_SECRET_KEY: test_secret_key_for_ci_only
          run: |
            pytest tests/ --cov=langvel --cov-report=xml --cov-report=term

        - name: Upload coverage
          uses: codecov/codecov-action@v3
          with:
            file: ./coverage.xml

    lint:
      runs-on: ubuntu-latest

      steps:
        - uses: actions/checkout@v3

        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.11'

        - name: Install dependencies
          run: |
            pip install black flake8 mypy

        - name: Run black
          run: black --check langvel/

        - name: Run flake8
          run: flake8 langvel/ --max-line-length=100

        - name: Run mypy
          run: mypy langvel/ --ignore-missing-imports
  ```
- **Effort**: 4 hours

#### ✅ TODO-032: Add Load Testing Suite
- **Tool**: k6
- **File**: New file `tests/load/agent_load_test.js`
- **Script**:
  ```javascript
  import http from 'k6/http';
  import { check, sleep } from 'k6';

  export const options = {
    stages: [
      { duration: '1m', target: 10 },   // Ramp up to 10 users
      { duration: '3m', target: 10 },   // Stay at 10 users
      { duration: '1m', target: 50 },   // Ramp up to 50 users
      { duration: '3m', target: 50 },   // Stay at 50 users
      { duration: '1m', target: 100 },  // Ramp up to 100 users
      { duration: '3m', target: 100 },  // Stay at 100 users
      { duration: '1m', target: 0 },    // Ramp down to 0 users
    ],
    thresholds: {
      http_req_duration: ['p(95)<5000'], // 95% of requests under 5s
      http_req_failed: ['rate<0.1'],     // Error rate under 10%
    },
  };

  export default function () {
    const payload = JSON.stringify({
      input: { query: 'What is Python?' },
      config: { thread_id: `thread_${__VU}_${__ITER}` }
    });

    const params = {
      headers: { 'Content-Type': 'application/json' },
    };

    const res = http.post(
      'http://localhost:8000/agents/test-agent',
      payload,
      params
    );

    check(res, {
      'status is 200': (r) => r.status === 200,
      'response has output': (r) => JSON.parse(r.body).output !== undefined,
      'response time < 5s': (r) => r.timings.duration < 5000,
    });

    sleep(1);
  }
  ```
- **Run**: `k6 run tests/load/agent_load_test.js`
- **Effort**: 4 hours

---

## 📝 SUMMARY

**Total TODOs**: 32
**P0 (Critical)**: 7 items - 22 hours
**P1 (High)**: 7 items - 28 hours
**P2 (Medium)**: 13 items - 37 hours
**P3 (Testing)**: 5 items - 62 hours

**Total Estimated Effort**: 149 hours (≈19 days with 1 engineer, ≈5 days with 4 engineers)

---

## 🎯 RECOMMENDED EXECUTION ORDER

### Week 1-2: Critical Security & Stability
1. TODO-001: JWT secret key (2h)
2. TODO-002: Error message exposure (3h)
3. TODO-005: Agent pooling (4h)
4. TODO-006: Request timeout (2h)
5. TODO-007: Rate limiter race condition (3h)
6. TODO-003: Request size limits (2h)
7. TODO-004: Agent path validation (2h)

### Week 3-4: High Priority Fixes
8. TODO-008: MCP stderr deadlock (4h)
9. TODO-009: Message bus backpressure (3h)
10. TODO-010: Auth storage to Redis (6h)
11. TODO-011: JWT revocation (4h)
12. TODO-013: Structured logging (6h)
13. TODO-014: Health checks (4h)

### Week 5-6: Medium Priority
14. TODO-015: LLM client singleton fix (4h)
15. TODO-016: LLM caching (4h)
16. TODO-018: Request queuing (2h)
17. TODO-019: Config validation (4h)
18. TODO-028: Prometheus metrics (6h)

### Week 7-8: Testing & Documentation
19. TODO-027: Comprehensive tests (40h)
20. TODO-031: CI/CD pipeline (4h)
21. TODO-032: Load testing (4h)

---

## ✅ CHECKLIST FOR PRODUCTION DEPLOYMENT

Before deploying to production, ensure:

- [ ] All P0 todos completed
- [ ] JWT_SECRET_KEY set and secure
- [ ] CORS_ORIGINS configured (not wildcard)
- [ ] Request timeouts configured
- [ ] Agent pooling enabled
- [ ] Rate limiting working
- [ ] Health checks responding
- [ ] Structured logging configured
- [ ] Metrics endpoint accessible
- [ ] Redis/PostgreSQL configured for state
- [ ] Load testing passed (100+ concurrent users)
- [ ] Test coverage > 80%
- [ ] CI/CD pipeline passing
- [ ] Documentation updated
- [ ] Monitoring/alerting set up (Grafana, etc.)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-17
**Maintainer**: Development Team
