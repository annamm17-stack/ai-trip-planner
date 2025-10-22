"""Test script for Pharmaceutical Intelligence Platform.

Tests:
1. Entity search with autocomplete
2. Entity resolution with fuzzy matching
3. End-to-end intelligence generation
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 120  # 2 minutes for full intelligence generation


def test_entity_search():
    """Test the entity search autocomplete endpoint."""
    print("\n" + "="*80)
    print("TEST 1: Entity Search Autocomplete")
    print("="*80)
    
    test_queries = [
        ("Pfizer", "Should find Pfizer Inc."),
        ("CVS", "Should find CVS Caremark"),
        ("Lilly", "Should find Eli Lilly"),
        ("Express Scripts", "Should find Express Scripts"),
        ("novo", "Should find Novo Nordisk (case insensitive)")
    ]
    
    for query, description in test_queries:
        print(f"\n🔍 Searching for: '{query}' ({description})")
        
        try:
            response = requests.get(
                f"{API_BASE_URL}/search-entities",
                params={"q": query, "limit": 3},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    print(f"✅ Found {len(results)} matches:")
                    for idx, result in enumerate(results, 1):
                        print(f"   {idx}. {result['name']} ({result['type']}) - Score: {result['score']}")
                        if result.get('ticker'):
                            print(f"      Ticker: {result['ticker']}")
                else:
                    print(f"⚠️  No matches found")
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
        
        except Exception as e:
            print(f"❌ Exception: {str(e)}")


def test_fuzzy_matching():
    """Test fuzzy matching with typos and variations."""
    print("\n" + "="*80)
    print("TEST 2: Fuzzy Matching")
    print("="*80)
    
    test_cases = [
        ("Fizer", "Typo for Pfizer"),
        ("J&J", "Alias for Johnson & Johnson"),
        ("BMS", "Alias for Bristol Myers Squibb"),
        ("Optum", "Should match OptumRx")
    ]
    
    for query, description in test_cases:
        print(f"\n🔍 Testing: '{query}' ({description})")
        
        try:
            response = requests.get(
                f"{API_BASE_URL}/search-entities",
                params={"q": query, "limit": 1},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    result = results[0]
                    print(f"✅ Matched: {result['name']} (Score: {result['score']})")
                else:
                    print(f"⚠️  No match found")
            else:
                print(f"❌ Error: HTTP {response.status_code}")
        
        except Exception as e:
            print(f"❌ Exception: {str(e)}")


def test_intelligence_generation(entity_id: str, entity_name: str, entity_type: str):
    """Test full intelligence generation pipeline."""
    print("\n" + "="*80)
    print(f"TEST 3: Intelligence Generation for {entity_name}")
    print("="*80)
    
    request_payload = {
        "entity_id": entity_id,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "date_range": "30 days",
        "session_id": "test_session_001",
        "user_id": "test_user"
    }
    
    print(f"\n📋 Request payload:")
    print(json.dumps(request_payload, indent=2))
    
    print(f"\n⏳ Generating intelligence (this may take 30-60 seconds)...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-intelligence",
            json=request_payload,
            timeout=TIMEOUT
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n✅ Intelligence generated in {elapsed_time:.1f} seconds")
            print(f"\n{'='*80}")
            print(f"EXECUTIVE SUMMARY")
            print(f"{'='*80}")
            print(data.get("summary", "No summary available"))
            
            print(f"\n{'='*80}")
            print(f"DETAILED SECTIONS")
            print(f"{'='*80}")
            
            sections = data.get("sections", {})
            for section_name, section_content in sections.items():
                print(f"\n--- {section_name.upper()} ---")
                print(section_content[:500])  # Print first 500 chars
                if len(section_content) > 500:
                    print("... (truncated)")
            
            print(f"\n{'='*80}")
            print(f"METADATA")
            print(f"{'='*80}")
            
            tool_calls = data.get("tool_calls", [])
            print(f"Tool calls: {len(tool_calls)}")
            for call in tool_calls:
                print(f"  - {call['agent']}: {call['tool']}")
            
            citations = data.get("citations", [])
            print(f"Citations: {len(citations)}")
            
            print(f"\n✅ Test completed successfully!")
            
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.Timeout:
        print(f"❌ Request timed out after {TIMEOUT} seconds")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHARMACEUTICAL INTELLIGENCE PLATFORM - TEST SUITE")
    print("="*80)
    
    # Check if server is running
    print("\n🔌 Checking server connectivity...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"⚠️  Server returned HTTP {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server at {API_BASE_URL}")
        print(f"   Error: {str(e)}")
        print(f"\n💡 Please start the server first:")
        print(f"   cd backend && uvicorn main:app --reload")
        return
    
    # Run tests
    test_entity_search()
    test_fuzzy_matching()
    
    # Test intelligence generation with a real entity
    print("\n" + "="*80)
    print("Select an entity for intelligence generation test:")
    print("="*80)
    
    # First, search for Pfizer
    response = requests.get(
        f"{API_BASE_URL}/search-entities",
        params={"q": "Pfizer", "limit": 1},
        timeout=5
    )
    
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            entity = results[0]
            test_intelligence_generation(
                entity_id=entity["id"],
                entity_name=entity["name"],
                entity_type=entity["type"]
            )
        else:
            print("⚠️  Could not find Pfizer in database")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()

