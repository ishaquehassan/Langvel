# Langvel ↔️ Laravel: Framework Comparison

This document shows how Langvel concepts map directly to Laravel patterns you already know and love.

## 📁 Directory Structure

### Laravel
```
app/
├── Http/
│   ├── Controllers/     # Handle requests
│   ├── Middleware/      # Intercept requests
│   └── Requests/        # Validate input
├── Models/              # Eloquent models
└── Providers/           # Service providers

config/                  # Configuration files
routes/                  # Route definitions
  ├── web.php
  └── api.php
storage/                 # Logs, cache, uploads
tests/                   # PHPUnit tests
```

### Langvel
```
app/
├── agents/              # Agent "controllers"
├── middleware/          # Intercept execution
├── models/              # State models (Pydantic)
├── tools/               # Custom tools
└── providers/           # Service providers

config/                  # Configuration files
routes/
  └── agent.py          # Agent routes
storage/                 # Logs, checkpoints
tests/                   # Pytest tests
```

## 🎯 Core Concepts

| Laravel | Langvel | Purpose |
|---------|---------|---------|
| `Controller` | `Agent` | Handle business logic |
| `Model` | `StateModel` | Data structures & validation |
| `Route::get()` | `@router.flow()` | Define routes |
| `Middleware` | `Middleware` | Intercept & modify |
| `Eloquent` | State models | ORM-like state management |
| `Service Provider` | Provider | Bootstrap services |
| `Artisan` | `langvel` CLI | Command-line tool |
| `config/*.php` | `config/*.py` | Configuration |
| `Request` | State input | Input validation |
| `Response` | State output | Output formatting |

## 💻 Code Examples

### 1. Routing

**Laravel:**
```php
// routes/web.php
Route::get('/users', [UserController::class, 'index'])
    ->middleware(['auth', 'throttle:60,1']);

Route::prefix('admin')
    ->middleware('admin')
    ->group(function () {
        Route::get('/dashboard', [AdminController::class, 'dashboard']);
    });
```

**Langvel:**
```python
# routes/agent.py
@router.flow('/users', middleware=['auth', 'rate_limit'])
class UserAgent(Agent):
    pass

with router.group(prefix='/admin', middleware=['admin']):
    @router.flow('/dashboard')
    class AdminDashboard(Agent):
        pass
```

### 2. Controllers/Agents

**Laravel:**
```php
// app/Http/Controllers/UserController.php
class UserController extends Controller
{
    public function __construct()
    {
        $this->middleware(['auth', 'verified']);
    }

    public function index(Request $request)
    {
        $users = User::where('active', true)->get();
        return response()->json($users);
    }
}
```

**Langvel:**
```python
# app/agents/user_agent.py
class UserAgent(Agent):
    state_model = UserState
    middleware = ['auth', 'verified']

    def build_graph(self):
        return self.start().then(self.index).end()

    async def index(self, state: UserState):
        state.users = await self.get_active_users()
        return state
```

### 3. Middleware

**Laravel:**
```php
// app/Http/Middleware/RateLimiter.php
class RateLimiter
{
    public function handle($request, Closure $next)
    {
        if ($this->limiter->tooManyAttempts($request->user()->id)) {
            throw new TooManyRequestsException;
        }

        return $next($request);
    }
}
```

**Langvel:**
```python
# app/middleware/rate_limiter.py
class RateLimiter(Middleware):
    async def before(self, state):
        if self.too_many_attempts(state.user_id):
            raise TooManyRequestsException()
        return state

    async def after(self, state):
        return state
```

### 4. Models

**Laravel:**
```php
// app/Models/User.php
class User extends Model
{
    protected $fillable = ['name', 'email'];

    protected $casts = [
        'email_verified_at' => 'datetime',
        'is_admin' => 'boolean',
    ];

    public function posts()
    {
        return $this->hasMany(Post::class);
    }
}
```

**Langvel:**
```python
# app/models/user_state.py
class UserState(StateModel):
    name: str
    email: EmailStr
    email_verified_at: Optional[datetime] = None
    is_admin: bool = False

    class Config:
        checkpointer = 'postgres'
        interrupts = ['before_email']
```

### 5. Validation

**Laravel:**
```php
// app/Http/Requests/StoreUserRequest.php
class StoreUserRequest extends FormRequest
{
    public function rules()
    {
        return [
            'name' => 'required|string|max:255',
            'email' => 'required|email|unique:users',
            'age' => 'integer|min:18',
        ];
    }
}
```

**Langvel:**
```python
# app/models/user_state.py
from pydantic import Field, validator

class UserState(StateModel):
    name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    age: int = Field(ge=18)

    @validator('email')
    def email_must_be_unique(cls, v):
        # Check uniqueness
        return v
```

### 6. Service Providers

**Laravel:**
```php
// app/Providers/AppServiceProvider.php
class AppServiceProvider extends ServiceProvider
{
    public function register()
    {
        $this->app->singleton(PaymentService::class);
    }

    public function boot()
    {
        // Bootstrap services
    }
}
```

**Langvel:**
```python
# app/providers/app_provider.py
class AppProvider(Provider):
    def register(self):
        self.app.singleton(PaymentService)

    def boot(self):
        # Bootstrap services
        pass
```

### 7. Artisan Commands

**Laravel:**
```bash
# Create controller
php artisan make:controller UserController

# Create model
php artisan make:model User

# Create middleware
php artisan make:middleware RateLimiter

# Serve application
php artisan serve

# Run tests
php artisan test
```

**Langvel:**
```bash
# Create agent
langvel make:agent UserAgent

# Create state model
langvel make:state UserState

# Create middleware
langvel make:middleware RateLimiter

# Serve application
langvel agent serve

# Run tests
pytest
```

## 🎨 Design Patterns

### Facades (Laravel) → Managers (Langvel)

**Laravel:**
```php
use Illuminate\Support\Facades\Cache;

Cache::put('key', 'value', 600);
$value = Cache::get('key');
```

**Langvel:**
```python
from langvel.rag.manager import RAGManager

rag = RAGManager()
results = await rag.retrieve('docs', 'query')
```

### Events & Listeners (Laravel) → Tool Decorators (Langvel)

**Laravel:**
```php
event(new UserRegistered($user));

// Listener
class SendWelcomeEmail {
    public function handle(UserRegistered $event) {
        // Send email
    }
}
```

**Langvel:**
```python
@tool(description="Send welcome email")
async def send_welcome_email(self, user: User):
    # Send email
    pass
```

### Jobs & Queues (Laravel) → Async Tools (Langvel)

**Laravel:**
```php
ProcessPodcast::dispatch($podcast);
```

**Langvel:**
```python
@tool(description="Process podcast")
async def process_podcast(self, podcast: Podcast):
    # Process in background
    pass
```

## 🔧 Configuration

### Laravel
```php
// config/database.php
return [
    'default' => env('DB_CONNECTION', 'mysql'),
    'connections' => [
        'mysql' => [
            'host' => env('DB_HOST', '127.0.0.1'),
        ]
    ]
];
```

### Langvel
```python
# config/langvel.py
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/langvel')
STATE_CHECKPOINTER = os.getenv('STATE_CHECKPOINTER', 'memory')
```

## 🌐 API Responses

### Laravel
```php
return response()->json([
    'data' => $users,
    'message' => 'Success'
], 200);
```

### Langvel
```python
return AgentResponse(
    output={'users': users},
    metadata={'message': 'Success'}
)
```

## 🔐 Authentication

### Laravel
```php
Route::middleware('auth')->group(function () {
    Route::get('/profile', [ProfileController::class, 'show']);
});
```

### Langvel
```python
@router.flow('/profile', middleware=['auth'])
class ProfileAgent(Agent):
    @requires_auth
    async def show(self, state):
        pass
```

## 📊 Where Langvel Extends Laravel

While maintaining Laravel's elegance, Langvel adds AI-specific features:

| Feature | Laravel Equivalent | Langvel Addition |
|---------|-------------------|------------------|
| RAG Tools | Database queries | Vector search with embeddings |
| MCP Servers | HTTP clients | Protocol for external AI tools |
| LLM Tools | API calls | Direct LLM integration |
| State Graphs | Sequential logic | Complex branching workflows |
| Streaming | Chunked responses | Real-time agent streaming |
| Checkpointers | Database sessions | State persistence across runs |

## 🎯 Philosophy Alignment

### Laravel's Philosophy
- Convention over configuration
- Developer happiness
- Expressive, elegant syntax
- Comprehensive tooling
- Strong community

### Langvel's Philosophy
- ✅ Convention over configuration (same!)
- ✅ Developer happiness (same!)
- ✅ Expressive, elegant syntax (same!)
- ✅ Comprehensive tooling (same!)
- ✅ + AI-first patterns
- ✅ + LangGraph power

## 💡 Key Takeaways

1. **Familiar Structure**: If you know Laravel, you know where everything goes in Langvel
2. **Same Patterns**: Routes, controllers (agents), middleware, models - all work the same way
3. **Enhanced with AI**: All the Laravel goodness + RAG, MCP, LLM tools, and graph workflows
4. **Beautiful DX**: Same attention to developer experience that makes Laravel loved

## 🚀 Next Steps

If you're coming from Laravel:

1. **Think Controllers → Agents**: Your business logic containers
2. **Think Models → State Models**: Your data structures
3. **Think Routes → Flow Routes**: Your endpoint definitions
4. **Add AI Tools**: RAG, MCP, LLM - the new superpowers

The rest is Laravel! 🎉

---

**You already know how to use Langvel. It's just Laravel for AI agents!**
