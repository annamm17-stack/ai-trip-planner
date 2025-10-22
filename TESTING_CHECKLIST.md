# Pharma Intelligence Platform - Testing Checklist

Use this checklist to verify that the Entity Search feature is working correctly.

## Prerequisites

- [ ] Python 3.10+ installed
- [ ] Dependencies installed: `pip install -r backend/requirements.txt`
- [ ] `.env` file created in `backend/` directory with at least `OPENAI_API_KEY`

## Phase 1: Server Startup ✓

### 1.1 Start the Server

```bash
cd "/Users/walkerfam/Documents/AI Trip Planner /ai-trip-planner"
./start.sh
```

**Expected Output:**
```
Loaded 60 entities (XXX total names/aliases)
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Checklist:**
- [ ] Server starts without errors
- [ ] "Loaded 60 entities" message appears
- [ ] No import errors or module not found
- [ ] Port 8000 is accessible

### 1.2 Verify Health Endpoint

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{"status": "healthy", "service": "ai-trip-planner"}
```

- [ ] Health endpoint returns 200 OK
- [ ] JSON response is valid

---

## Phase 2: Entity Search Testing ✓

### 2.1 Basic Search

```bash
curl "http://localhost:8000/search-entities?q=Pfizer"
```

**Expected:**
- [ ] Returns HTTP 200
- [ ] JSON contains "results" array
- [ ] First result has "name": "Pfizer Inc."
- [ ] Score is 100 (exact match)
- [ ] Response time < 100ms

### 2.2 Fuzzy Matching

Test with typo:
```bash
curl "http://localhost:8000/search-entities?q=Fizer"
```

**Expected:**
- [ ] Returns results (fuzzy match)
- [ ] First result is Pfizer
- [ ] Score is > 80

### 2.3 Alias Matching

```bash
curl "http://localhost:8000/search-entities?q=CVS"
curl "http://localhost:8000/search-entities?q=J&J"
curl "http://localhost:8000/search-entities?q=BMS"
```

**Expected:**
- [ ] "CVS" matches "CVS Caremark"
- [ ] "J&J" matches "Johnson & Johnson"
- [ ] "BMS" matches "Bristol Myers Squibb"
- [ ] All have high scores (90-100)

### 2.4 Case Insensitivity

```bash
curl "http://localhost:8000/search-entities?q=novo"
curl "http://localhost:8000/search-entities?q=NOVO"
```

**Expected:**
- [ ] Both return "Novo Nordisk"
- [ ] Same score regardless of case

### 2.5 PBM Search

```bash
curl "http://localhost:8000/search-entities?q=Express%20Scripts"
curl "http://localhost:8000/search-entities?q=OptumRx"
```

**Expected:**
- [ ] Finds PBM entities
- [ ] type field is "pbm" (not "manufacturer")

### 2.6 Limit Parameter

```bash
curl "http://localhost:8000/search-entities?q=pharma&limit=3"
```

**Expected:**
- [ ] Returns maximum 3 results
- [ ] Results are ranked by score

### 2.7 Empty Query

```bash
curl "http://localhost:8000/search-entities?q="
```

**Expected:**
- [ ] Returns empty results array
- [ ] No server error

---

## Phase 3: Intelligence Generation Testing ✓

### 3.1 Basic Intelligence Request

Create test file `test_request.json`:
```json
{
  "entity_id": "pfizer",
  "entity_name": "Pfizer Inc.",
  "entity_type": "manufacturer",
  "date_range": "30 days"
}
```

Run:
```bash
curl -X POST "http://localhost:8000/generate-intelligence" \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

**Expected:**
- [ ] Returns HTTP 200
- [ ] Response time: 30-60 seconds
- [ ] Response contains:
  - [ ] "entity": "Pfizer Inc."
  - [ ] "summary": (executive summary text)
  - [ ] "sections": {news, legislative, products}
  - [ ] "tool_calls": (array of tool executions)
  - [ ] "citations": (array, may be empty)

### 3.2 Verify Agent Execution

Check the response for:
- [ ] "news" section is not empty
- [ ] "legislative" section is not empty
- [ ] "products" section is not empty
- [ ] "summary" synthesizes all three sections

### 3.3 Tool Call Verification

In the response, verify `tool_calls`:
- [ ] Contains 3-9 tool calls
- [ ] At least one from "news" agent
- [ ] At least one from "legislative" agent
- [ ] At least one from "product" agent

Example tool_calls:
```json
[
  {"agent": "news", "tool": "market_news", "args": {...}},
  {"agent": "legislative", "tool": "federal_legislation", "args": {...}},
  {"agent": "product", "tool": "fda_approvals", "args": {...}}
]
```

### 3.4 Test Different Entity Types

**Manufacturer:**
```json
{"entity_id": "eli-lilly", "entity_name": "Eli Lilly and Company", "entity_type": "manufacturer"}
```

**PBM:**
```json
{"entity_id": "cvs-caremark", "entity_name": "CVS Caremark", "entity_type": "pbm"}
```

**Expected:**
- [ ] Both complete successfully
- [ ] Responses differ appropriately (manufacturer vs PBM context)

---

## Phase 4: Automated Test Suite ✓

### 4.1 Run Test Script

```bash
python "test scripts/test_pharma_intelligence.py"
```

**Expected Output:**
```
PHARMACEUTICAL INTELLIGENCE PLATFORM - TEST SUITE
================================================================================
✅ Server is running

TEST 1: Entity Search Autocomplete
================================================================================
✅ Found X matches
...

TEST 2: Fuzzy Matching
================================================================================
✅ Matched: ...

TEST 3: Intelligence Generation
================================================================================
✅ Intelligence generated in X.X seconds
...

ALL TESTS COMPLETED
```

**Checklist:**
- [ ] All tests pass
- [ ] No exceptions or errors
- [ ] Intelligence generation completes in <60 seconds

---

## Phase 5: API Documentation ✓

### 5.1 Interactive Docs

Open in browser: http://localhost:8000/docs

**Checklist:**
- [ ] Swagger UI loads
- [ ] See `/search-entities` endpoint
- [ ] See `/generate-intelligence` endpoint
- [ ] Can test endpoints interactively
- [ ] Request/response schemas are visible

### 5.2 Test via Swagger UI

**Test /search-entities:**
- [ ] Click "Try it out"
- [ ] Enter query: "Pfizer"
- [ ] Click "Execute"
- [ ] See results in response body

**Test /generate-intelligence:**
- [ ] Click "Try it out"
- [ ] Use example request body
- [ ] Click "Execute"
- [ ] Wait 30-60 seconds
- [ ] See full intelligence report

---

## Phase 6: Optional Features ✓

### 6.1 Test with Real-Time Data (Requires API Keys)

Add to `.env`:
```bash
TAVILY_API_KEY=your_key_here
NEWSAPI_KEY=your_key_here
```

Restart server and test:
```bash
curl -X POST "http://localhost:8000/generate-intelligence" \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "pfizer", "entity_name": "Pfizer Inc.", "entity_type": "manufacturer"}'
```

**Expected:**
- [ ] Response includes real news URLs
- [ ] Citations are more specific
- [ ] Response time may be faster

### 6.2 Test RAG (Requires OpenAI Key)

Add to `.env`:
```bash
ENABLE_RAG=1
```

Restart server and test with a manufacturer:

**Expected:**
- [ ] Server logs show RAG retrieval
- [ ] Product section includes regulatory context
- [ ] References to FDA processes, regulations appear

### 6.3 Test Observability (Requires Arize Account)

Add to `.env`:
```bash
ARIZE_SPACE_ID=your_space_id
ARIZE_API_KEY=your_api_key
```

Restart server, generate intelligence, then:

**In Arize Dashboard:**
- [ ] See traces for intelligence requests
- [ ] See agent spans (news, legislative, product, synthesizer)
- [ ] See tool call spans
- [ ] Metadata includes entity name, agent type

---

## Phase 7: Error Handling ✓

### 7.1 Invalid Entity ID

```bash
curl -X POST "http://localhost:8000/generate-intelligence" \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "invalid", "entity_name": "Invalid Company", "entity_type": "manufacturer"}'
```

**Expected:**
- [ ] System handles gracefully (doesn't crash)
- [ ] Returns valid response (may be generic)

### 7.2 Missing Required Fields

```bash
curl -X POST "http://localhost:8000/generate-intelligence" \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "pfizer"}'
```

**Expected:**
- [ ] Returns HTTP 422 (validation error)
- [ ] Error message indicates missing fields

### 7.3 Malformed JSON

```bash
curl -X POST "http://localhost:8000/generate-intelligence" \
  -H "Content-Type: application/json" \
  -d '{invalid json}'
```

**Expected:**
- [ ] Returns HTTP 422 or 400
- [ ] Error message indicates JSON parsing error

---

## Phase 8: Performance Testing ✓

### 8.1 Response Time Benchmarking

Run 5 intelligence requests and measure time:

```bash
for i in {1..5}; do
  echo "Request $i:"
  time curl -X POST "http://localhost:8000/generate-intelligence" \
    -H "Content-Type: application/json" \
    -d '{"entity_id": "pfizer", "entity_name": "Pfizer Inc.", "entity_type": "manufacturer"}' \
    -o /dev/null -s
  echo ""
done
```

**Expected:**
- [ ] All requests complete in <60 seconds
- [ ] Average time: 30-45 seconds (with APIs)
- [ ] Average time: 45-60 seconds (LLM fallback)
- [ ] No timeouts or errors

### 8.2 Concurrent Requests

Test with 3 simultaneous requests (in separate terminals):

Terminal 1:
```bash
curl -X POST "http://localhost:8000/generate-intelligence" \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "pfizer", "entity_name": "Pfizer Inc.", "entity_type": "manufacturer"}'
```

Terminal 2:
```bash
curl -X POST "http://localhost:8000/generate-intelligence" \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "eli-lilly", "entity_name": "Eli Lilly and Company", "entity_type": "manufacturer"}'
```

Terminal 3:
```bash
curl -X POST "http://localhost:8000/generate-intelligence" \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "cvs-caremark", "entity_name": "CVS Caremark", "entity_type": "pbm"}'
```

**Expected:**
- [ ] All three complete successfully
- [ ] No significant slowdown
- [ ] No server crashes or errors

---

## Phase 9: Legacy Compatibility ✓

### 9.1 Trip Planner Still Works

Test original trip planner endpoint:

```bash
curl -X POST "http://localhost:8000/plan-trip" \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Paris, France",
    "duration": "5 days",
    "budget": "2000",
    "interests": "art, food"
  }'
```

**Expected:**
- [ ] Returns trip itinerary
- [ ] Original functionality unchanged
- [ ] Both systems coexist

### 9.2 Frontend Loads

Open: http://localhost:8000/

**Expected:**
- [ ] Frontend HTML loads
- [ ] No 404 errors
- [ ] Original trip planner UI intact

---

## Troubleshooting Common Issues

### Issue: "Module not found: entity_resolver"

**Solution:**
```bash
cd backend
pip install -r requirements.txt
# Ensure you're running uvicorn from the backend directory
```

### Issue: "Loaded 0 entities"

**Solution:**
- Check that `backend/data/pharma_entities.json` exists
- Verify JSON is valid (no syntax errors)
- Check file permissions

### Issue: "401 Unauthorized" from OpenAI

**Solution:**
- Verify `OPENAI_API_KEY` in `.env`
- Remove any extra spaces or quotes
- Test key: `curl https://api.openai.com/v1/models -H "Authorization: Bearer YOUR_KEY"`

### Issue: Slow responses (>90 seconds)

**Solution:**
- This is normal without API keys (LLM fallback is slower)
- Add `TAVILY_API_KEY` for faster responses
- Check your internet connection
- Verify OpenAI API is not rate-limited

### Issue: Empty sections in response

**Solution:**
- Check server logs for tool errors
- Verify API keys are correct
- LLM fallback may produce shorter responses (expected)

---

## Success Criteria

✅ **All tests pass if:**
- Entity search returns results in <100ms
- Fuzzy matching handles typos and aliases
- Intelligence generation completes in <60 seconds
- All three agent sections are populated
- Tool calls are tracked
- No server crashes or errors
- Automated test suite passes
- Response structure matches schema

---

## Final Checklist

- [ ] All Phase 1-5 tests pass (required)
- [ ] Phase 6 tests pass (if API keys configured)
- [ ] Phase 7 error handling works
- [ ] Phase 8 performance meets requirements
- [ ] Phase 9 legacy compatibility verified
- [ ] Documentation reviewed
- [ ] Environment variables documented
- [ ] Ready for demo/pilot deployment

---

**Testing completed? Mark as done! ✅**

See `IMPLEMENTATION_SUMMARY.md` for full implementation details.
See `PHARMA_QUICKSTART.md` for usage examples.
See `PHARMA_ENV_SETUP.md` for environment configuration.

