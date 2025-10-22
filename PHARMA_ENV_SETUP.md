# Pharma Intelligence Platform - Environment Setup

This document describes the environment variables needed for the Pharmaceutical Intelligence Platform.

## Required Environment Variables

Create a `backend/.env` file with the following configuration:

### 1. LLM Provider (Required)

Choose ONE of the following:

```bash
# Option A: OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Option B: OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### 2. RAG / Vector Search (Optional)

For enhanced product intelligence with curated pharma knowledge:

```bash
# Enable RAG (Retrieval-Augmented Generation)
ENABLE_RAG=1

# OpenAI Embeddings (required if ENABLE_RAG=1)
OPENAI_EMBED_MODEL=text-embedding-3-small
```

### 3. Observability (Optional)

For production monitoring and tracing:

```bash
ARIZE_SPACE_ID=your_arize_space_id
ARIZE_API_KEY=your_arize_api_key
```

## Optional API Keys (For Real-Time Data)

### Web Search APIs

These enable real-time web search for news, legislation, and product information. Without these keys, the system falls back to LLM-generated responses.

#### Tavily (Recommended)
- **Best for:** AI applications with semantic search
- **Free tier:** 1,000 searches/month
- **Sign up:** https://tavily.com

```bash
TAVILY_API_KEY=your_tavily_api_key
```

#### SerpAPI (Alternative)
- **Best for:** Traditional Google search results
- **Free tier:** 100 searches/month
- **Sign up:** https://serpapi.com

```bash
SERPAPI_API_KEY=your_serpapi_key
```

### Pharma-Specific APIs

#### NewsAPI
- **Purpose:** Market news and company announcements
- **Free tier:** 100 requests/day (development only)
- **Paid tier:** Recommended for production ($449/month)
- **Sign up:** https://newsapi.org

```bash
NEWSAPI_KEY=your_newsapi_key
```

#### OpenCorporates
- **Purpose:** Entity resolution fallback for unknown companies
- **Used when:** Entity not found in curated database
- **Free tier:** 500 requests/month
- **Sign up:** https://opencorporates.com

```bash
OPENCORPORATES_API_KEY=your_opencorporates_key
```

**Note:** FDA and ClinicalTrials.gov APIs are public and do not require keys.

## Complete Example `.env` File

```bash
# ============================================================================
# PHARMA INTELLIGENCE PLATFORM - ENVIRONMENT CONFIGURATION
# ============================================================================

# LLM Provider
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx

# RAG / Embeddings
ENABLE_RAG=1
OPENAI_EMBED_MODEL=text-embedding-3-small

# Observability
ARIZE_SPACE_ID=your_space_id
ARIZE_API_KEY=your_api_key

# Web Search (optional)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx

# Pharma APIs (optional)
NEWSAPI_KEY=xxxxxxxxxxxxxxxx
OPENCORPORATES_API_KEY=xxxxxxxxxxxxxxxx

# Testing
TEST_MODE=0
```

## Minimal Setup (MVP)

To get started quickly, you only need:

```bash
# Minimal .env for testing
OPENAI_API_KEY=your_openai_key_here
ENABLE_RAG=0
```

This configuration will:
- ✅ Enable entity search and fuzzy matching
- ✅ Run all three intelligence agents
- ✅ Generate executive summaries
- ⚠️ Use LLM fallback for all tool responses (no real-time data)

## Recommended Production Setup

For production use with real-time data:

```bash
# Production .env
OPENAI_API_KEY=your_openai_key_here
ENABLE_RAG=1
OPENAI_EMBED_MODEL=text-embedding-3-small

# Real-time data sources
TAVILY_API_KEY=your_tavily_key
NEWSAPI_KEY=your_newsapi_key

# Observability
ARIZE_SPACE_ID=your_space_id
ARIZE_API_KEY=your_api_key
```

This configuration provides:
- ✅ Real-time market news from NewsAPI
- ✅ Web search for legislative and product data
- ✅ RAG-enhanced product intelligence
- ✅ Full observability and tracing

## API Cost Estimates

| Service | Free Tier | Production Cost | Monthly Est. |
|---------|-----------|-----------------|--------------|
| OpenAI GPT-3.5 | $5 credit | $0.002/1K tokens | ~$50-100 |
| Tavily Search | 1K/month | $50/10K searches | $0-50 |
| NewsAPI | 100/day | $449/month | $449 |
| OpenCorporates | 500/month | $50/10K calls | $0-50 |
| **Total** | **Free tier** | - | **~$550-650/month** |

## Fallback Strategy

The platform is designed with graceful degradation:

1. **With API keys:** Real-time data from authoritative sources
2. **Without API keys:** LLM-generated responses based on training data
3. **With ENABLE_RAG=1:** Curated pharma knowledge injected into responses
4. **Without RAG:** Pure LLM knowledge

This ensures the platform works in any configuration, from local development to production deployment.

## Setup Instructions

1. **Copy this file's example to `.env`:**
   ```bash
   cd backend
   cp ../PHARMA_ENV_SETUP.md .env
   # Edit .env to add your actual keys
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. **Test the setup:**
   ```bash
   cd ..
   python "test scripts/test_pharma_intelligence.py"
   ```

## Troubleshooting

### "401 Unauthorized" errors
- Check that your API keys are correct and active
- Ensure no extra spaces in the `.env` file

### "Module not found" errors
- Run `pip install -r requirements.txt`
- Check that you're in the correct virtual environment

### Slow responses without API keys
- Expected behavior - LLM fallback is slower than API calls
- Add TAVILY_API_KEY for faster responses

### No RAG results
- Ensure `ENABLE_RAG=1` in `.env`
- Verify `pharma_knowledge.json` exists in `backend/data/`
- Check that OPENAI_API_KEY is set (needed for embeddings)

