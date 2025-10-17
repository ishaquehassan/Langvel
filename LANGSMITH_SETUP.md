# LangSmith Tracing Setup Guide

This guide will help you set up LangSmith tracing for your Langvel agents.

🔗 **[LangSmith Official Platform](https://smith.langchain.com)** - Sign up and view your traces here!

## Prerequisites

1. **LangSmith Account**
   - Sign up at [https://smith.langchain.com](https://smith.langchain.com)
   - Free tier available with generous limits

2. **Anthropic API Key** (for Claude LLM)
   - Sign up at [https://console.anthropic.com](https://console.anthropic.com)
   - Get your API key from the dashboard

## Step 1: Get Your LangSmith API Key

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Sign in or create an account
3. Click on your profile (top right)
4. Go to "Settings" → "API Keys"
5. Click "Create API Key"
6. Copy the API key (it will only be shown once!)

## Step 2: Configure Environment Variables

Update your `.env` file with the following configuration:

```bash
# Anthropic API Key (required for Claude LLM)
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx  # Replace with your actual key

# LangSmith Configuration
LANGSMITH_API_KEY=lsv2_pt_xxxxx  # Replace with your actual LangSmith key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langvel-demo  # Or any project name you prefer
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### Environment Variable Explanation

- **ANTHROPIC_API_KEY**: Your Anthropic API key for Claude LLM
- **LANGSMITH_API_KEY**: Your LangSmith API key for tracing
- **LANGCHAIN_TRACING_V2**: Enables LangChain tracing (set to `true`)
- **LANGCHAIN_PROJECT**: Project name in LangSmith dashboard
- **LANGSMITH_ENDPOINT**: LangSmith API endpoint (usually don't need to change)

## Step 3: Run the Test Script

Once you've configured your API keys, run the test script:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the LangSmith demo
python test_langsmith.py
```

## Step 4: View Traces in LangSmith

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Select your project (e.g., "langvel-demo")
3. You'll see all traces from your agent executions

## What You'll See in LangSmith

Each trace includes:

### 1. **Overall Execution**
- Total execution time
- Input and output data
- Success/failure status

### 2. **Detailed Spans**
- Each node in the agent workflow
- Function calls and their duration
- Input/output for each step

### 3. **LLM Calls**
- Model used (e.g., claude-3-5-sonnet-20241022)
- Prompts sent to the LLM
- Responses received
- Token usage and costs

### 4. **Tool Executions**
- Tool name and type
- Input parameters
- Output results
- Success/failure status
- Execution time

### 5. **Workflow Visualization**
- Graph showing the execution flow
- Which paths were taken
- Where errors occurred (if any)

## Example Trace Structure

```
CustomerSupportAgent (15.2s)
├─ classify_request (0.1s)
│  └─ Input: query="I'm having trouble..."
│  └─ Output: category="technical"
├─ analyze_sentiment (0.05s)
│  └─ sentiment=0.4
├─ search_knowledge (0.3s)
│  └─ RAG retrieval
├─ handle_technical (0.02s)
├─ generate_response (14.5s)
│  └─ LLM Call
│     ├─ Model: claude-3-5-sonnet-20241022
│     ├─ Prompt: "Customer Query..."
│     ├─ Response: "I understand you're having..."
│     ├─ Tokens: 245 input, 180 output
│     └─ Cost: $0.003
└─ notify_slack (0.1s)
```

## Advanced Features

### Custom Metadata

You can add custom metadata to traces:

```python
# In your agent code
from langvel.observability.tracer import get_observability_manager

obs_manager = get_observability_manager()
obs_manager.start_trace(
    name="my_agent",
    input_data=input_data,
    metadata={
        "user_id": "user_123",
        "environment": "production",
        "version": "v1.2.3"
    }
)
```

### Multiple Projects

You can organize traces into different projects:

```bash
# In .env
LANGCHAIN_PROJECT=langvel-production  # For production
# or
LANGCHAIN_PROJECT=langvel-development  # For development
```

### Filtering and Search

In LangSmith dashboard:
- Filter by status (success/failure)
- Search by input/output content
- Filter by execution time
- Group by metadata fields

## Troubleshooting

### Error: "403 Forbidden"

**Problem**: Invalid or expired API key

**Solution**:
1. Check that LANGSMITH_API_KEY is correct
2. Regenerate API key in LangSmith dashboard
3. Update .env file with new key

### Error: "401 Unauthorized" for Anthropic

**Problem**: Invalid Anthropic API key

**Solution**:
1. Check ANTHROPIC_API_KEY in .env
2. Verify key is active in Anthropic console
3. Check you have sufficient credits

### Traces Not Appearing

**Problem**: Traces not showing up in dashboard

**Solution**:
1. Verify LANGCHAIN_TRACING_V2=true
2. Check LANGCHAIN_PROJECT name
3. Wait a few seconds (traces can take time to upload)
4. Check for errors in console output

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'langsmith'`

**Solution**:
```bash
source venv/bin/activate
pip install langsmith
```

## Cost Optimization

LangSmith is free for:
- Up to 5,000 traces/month
- 30 days of retention

For larger usage:
- **Plus Plan**: $39/month for 50K traces
- **Enterprise**: Custom pricing

Tips to reduce costs:
1. Use separate projects for dev/prod
2. Filter out health check traces
3. Sample production traffic (e.g., trace 10% of requests)

## Next Steps

1. ✅ Set up API keys
2. ✅ Run test script
3. ✅ View traces in dashboard
4. 📊 Add custom metadata
5. 🔍 Analyze performance bottlenecks
6. 🚀 Deploy with tracing enabled

## Resources

- **LangSmith Docs**: [https://docs.smith.langchain.com](https://docs.smith.langchain.com)
- **Anthropic Docs**: [https://docs.anthropic.com](https://docs.anthropic.com)
- **Langvel Docs**: See `/docs` folder

## Support

Having issues? Check:
1. This setup guide
2. Console output for error messages
3. LangSmith documentation
4. GitHub issues: [langvel/issues](https://github.com/ishaquehassan/langvel/issues)
