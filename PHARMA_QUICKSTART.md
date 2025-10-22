# Pharma Intelligence Platform - Quick Start Guide

Welcome to the Pharmaceutical & PBM Intelligence Platform! This guide will get you up and running in 5 minutes.

## What You'll Build

A platform that generates executive intelligence summaries for pharmaceutical manufacturers and PBMs, including:
- 📰 **Market News:** Recent announcements, earnings, M&A activity
- ⚖️ **Legislative Impact:** Federal/state bills, CMS policies, regulatory changes
- 💊 **Product Pipeline:** FDA approvals, clinical trials, drug development

## Prerequisites

- Python 3.10+
- OpenAI API key (required)
- Optional: Tavily API key (recommended for real-time data)

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This installs:
- FastAPI, LangChain, LangGraph (agent framework)
- RapidFuzz (fuzzy entity matching)
- OpenAI SDK, httpx (API calls)

### Step 2: Configure Environment

Create `backend/.env`:

```bash
# Minimal setup
OPENAI_API_KEY=sk-your-openai-key-here
ENABLE_RAG=0

# Optional: For real-time data (recommended)
TAVILY_API_KEY=tvly-your-tavily-key-here
```

### Step 3: Start the Server

```bash
# From project root
./start.sh

# OR manually:
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
Loaded 60 entities (XXX total names/aliases)
```

### Step 4: Test Entity Search

Open http://localhost:8000/docs in your browser to see the interactive API documentation.

**Try the autocomplete endpoint:**

```bash
curl "http://localhost:8000/search-entities?q=Pfizer&limit=3"
```

Response:
```json
{
  "results": [
    {
      "id": "pfizer",
      "name": "Pfizer Inc.",
      "type": "manufacturer",
      "ticker": "PFE",
      "score": 100
    }
  ]
}
```

### Step 5: Generate Intelligence Summary

```bash
curl -X POST "http://localhost:8000/generate-intelligence" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "pfizer",
    "entity_name": "Pfizer Inc.",
    "entity_type": "manufacturer",
    "date_range": "30 days"
  }'
```

**This will:**
1. ⏱️ Take 30-60 seconds to complete
2. 🤖 Run 3 agents in parallel (News, Legislative, Product)
3. 📊 Return a comprehensive executive summary

Response structure:
```json
{
  "entity": "Pfizer Inc.",
  "summary": "Executive Overview: Pfizer continues to navigate...",
  "sections": {
    "news": "Recent market developments for Pfizer...",
    "legislative": "Federal and state legislation affecting...",
    "products": "FDA approvals and clinical trial updates..."
  },
  "tool_calls": [
    {"agent": "news", "tool": "market_news", "args": {...}},
    {"agent": "legislative", "tool": "federal_legislation", "args": {...}}
  ]
}
```

## Architecture Overview

```
User Request: "Pfizer"
       ↓
[Entity Resolver]
   Fuzzy Match → entity_id, entity_name, entity_type
       ↓
[LangGraph Orchestration]
       ↓
   ┌───┴───┬────────┬────────┐
   │       │        │        │
[News]  [Leg.]  [Product]  (parallel)
   │       │        │        │
   └───┬───┴────────┴────────┘
       ↓
[Synthesizer Agent]
       ↓
  Executive Summary
```

### Agent Responsibilities

1. **News Agent**
   - Tools: `market_news()`, `company_announcements()`, `sentiment_analysis()`
   - Sources: NewsAPI, Tavily, company press releases
   - Output: Top 5 market developments with sentiment

2. **Legislative Agent**
   - Tools: `federal_legislation()`, `state_bills()`, `cms_updates()`
   - Sources: Web search for Congress.gov, state legislatures, CMS
   - Output: Regulatory impacts and policy changes

3. **Product Agent**
   - Tools: `fda_approvals()`, `clinical_trials()`, `pipeline_status()`
   - Sources: Web search for FDA, ClinicalTrials.gov
   - RAG: Injects curated pharma knowledge (if ENABLE_RAG=1)
   - Output: Product pipeline and FDA activity

4. **Synthesizer Agent**
   - Input: All agent outputs
   - Output: Structured executive summary for C-suite audience

## Testing the Full System

Run the automated test suite:

```bash
python "test scripts/test_pharma_intelligence.py"
```

This tests:
- ✅ Entity search with autocomplete
- ✅ Fuzzy matching (typos, aliases)
- ✅ End-to-end intelligence generation
- ✅ Tool call execution
- ✅ Response timing (<60 seconds)

## Supported Entities

**50 Pharmaceutical Manufacturers:**
- Top 20: Pfizer, J&J, Eli Lilly, AbbVie, Novartis, Merck, Roche, AstraZeneca, BMS, Novo Nordisk, Sanofi, GSK, Gilead, Amgen, Biogen, Bayer, Takeda, Boehringer Ingelheim, Moderna, Regeneron
- Plus 30 more including Vertex, BioNTech, Viatris, Organon, etc.

**10 Major PBMs:**
- CVS Caremark, Express Scripts, OptumRx, Humana Pharmacy, Prime Therapeutics, Magellan Rx, MedImpact, Navitus, Elixir, EnvolveRx

All entities include:
- Primary name and aliases (e.g., "CVS" → "CVS Caremark")
- Stock ticker symbols
- Entity type (manufacturer or PBM)
- Headquarters location

## Fuzzy Matching Examples

The entity resolver handles:
- **Typos:** "Fizer" → Pfizer (score: 91)
- **Abbreviations:** "J&J" → Johnson & Johnson (score: 100)
- **Partial names:** "Lilly" → Eli Lilly and Company (score: 95)
- **Case insensitive:** "cvs" → CVS Caremark (score: 100)
- **Common aliases:** "BMS" → Bristol Myers Squibb (score: 100)

## API Endpoints

### 1. Entity Search (GET)
```
GET /search-entities?q={query}&limit={limit}
```

Returns top matching entities with scores.

### 2. Generate Intelligence (POST)
```
POST /generate-intelligence
```

Body:
```json
{
  "entity_id": "pfizer",
  "entity_name": "Pfizer Inc.",
  "entity_type": "manufacturer",
  "date_range": "30 days",
  "session_id": "optional_session_id"
}
```

### 3. Health Check (GET)
```
GET /health
```

Returns server status.

### 4. Trip Planner (Legacy)
```
POST /plan-trip
```

Original trip planning endpoint (still functional).

## Enabling Real-Time Data

Without API keys, the system uses LLM-generated responses (slower, based on training data).

**To enable real-time data:**

1. Add Tavily API key to `.env`:
```bash
TAVILY_API_KEY=tvly-your-key-here
```

2. Restart the server

3. All tools now fetch live data from web search

**Benefits:**
- ✅ Current news from last 24-48 hours
- ✅ Recent FDA approvals and clinical trials
- ✅ Latest legislative activity
- ✅ Real-time market sentiment

**Without API keys:**
- ⚠️ Responses based on LLM training data (up to April 2024)
- ⚠️ Slower response times
- ✅ Still functional for testing and development

## Enabling RAG (Knowledge Base)

The Product Agent can use Retrieval-Augmented Generation to inject curated pharmaceutical knowledge:

1. Enable RAG in `.env`:
```bash
ENABLE_RAG=1
OPENAI_API_KEY=your-key-here  # Required for embeddings
```

2. Restart server

3. Product Agent now includes context from:
   - FDA approval processes
   - Drug Supply Chain Security Act (DSCSA)
   - 340B Drug Pricing Program
   - PBM rebate structures
   - Biosimilars and interchangeability
   - Gene therapy regulations
   - 20+ curated pharma topics

## Observability with Arize

Track agent performance, tool calls, and response times:

1. Sign up at https://app.arize.com

2. Add to `.env`:
```bash
ARIZE_SPACE_ID=your_space_id
ARIZE_API_KEY=your_api_key
```

3. View traces in Arize dashboard:
   - Agent execution times
   - Tool call frequency
   - LLM token usage
   - Error rates

## Common Issues

### "Cannot connect to server"
- Ensure server is running: `uvicorn main:app --reload`
- Check port 8000 is not in use

### "Module not found: entity_resolver"
- Install dependencies: `pip install -r requirements.txt`
- Ensure you're in the backend directory when running uvicorn

### "401 Unauthorized" from OpenAI
- Check OPENAI_API_KEY in `.env`
- Ensure key has no extra spaces or quotes

### Slow intelligence generation (>90 seconds)
- Expected without API keys (LLM fallback)
- Add TAVILY_API_KEY for faster responses
- Check OpenAI API rate limits

### Empty search results
- Check entity database loaded: Look for "Loaded X entities" in logs
- Verify `backend/data/pharma_entities.json` exists
- Try exact company names first

## Next Steps

1. **Add More Entities:** Edit `backend/data/pharma_entities.json`
2. **Customize Agents:** Modify agent prompts in `backend/main.py`
3. **Add New Tools:** Create tools in `backend/pharma_tools.py`
4. **Build UI:** Update `frontend/index.html` for pharma use case
5. **Deploy:** Use `render.yaml` for production deployment

## Example Use Cases

### Investment Analysis
"Get the latest intelligence on Novo Nordisk for earnings call prep"
- Market news: Recent Ozempic/Wegovy sales
- Legislative: Medicare negotiation impacts
- Products: GLP-1 pipeline updates

### Competitive Intelligence
"Compare Pfizer vs. Moderna's mRNA vaccine developments"
- Search both entities
- Generate intelligence for each
- Compare product pipelines and FDA activity

### Policy Monitoring
"Track PBM reform legislation affecting Express Scripts"
- Legislative agent focuses on federal/state bills
- CMS policy updates
- Regulatory changes

### M&A Research
"Analyze AbbVie's acquisition strategy and pipeline"
- Recent acquisitions in news
- Post-merger integration updates
- Combined product pipeline

## Support

- 📖 **Full Documentation:** See `PRD_PHARMA_INTELLIGENCE.md`
- 🔧 **Environment Setup:** See `PHARMA_ENV_SETUP.md`
- 🧪 **Testing:** Run `python "test scripts/test_pharma_intelligence.py"`
- 🚀 **Original Project:** See `README.md` for AI Trip Planner architecture

---

**Ready to build?** Start the server and visit http://localhost:8000/docs 🚀

