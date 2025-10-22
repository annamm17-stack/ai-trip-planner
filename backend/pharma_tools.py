"""Pharmaceutical intelligence tools for News, Legislative, and Product agents.

Each tool follows the graceful degradation pattern:
1. Try real API call (if key configured)
2. Fall back to LLM generation (if no API key)
3. Return formatted string with citations
"""

import os
import httpx
from typing import Optional, List
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage


# Timeout for API calls
API_TIMEOUT = 10.0


def _compact(text: str, limit: int = 300) -> str:
    """Compact text to a maximum length, truncating at word boundaries."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.;- ")


def _llm_fallback(instruction: str, entity: str, llm) -> str:
    """Use the LLM to generate a response when APIs aren't available."""
    prompt = f"Respond with 300 characters or less.\n{instruction}\nEntity: {entity}"
    try:
        response = llm.invoke([
            SystemMessage(content="You are a pharmaceutical industry analyst."),
            HumanMessage(content=prompt),
        ])
        return _compact(response.content)
    except Exception:
        return f"Unable to retrieve information for {entity}."


# ============================================================================
# NEWS AGENT TOOLS
# ============================================================================

@tool
def market_news(entity: str, days: int = 30) -> str:
    """Get recent market news and announcements for a pharmaceutical entity.
    
    Args:
        entity: Name of pharmaceutical company or PBM
        days: Number of days to look back (default: 30)
        
    Returns:
        Summary of recent news with sources
    """
    from main import llm  # Import here to avoid circular dependency
    
    newsapi_key = os.getenv("NEWSAPI_KEY")
    
    if newsapi_key:
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": entity,
                        "apiKey": newsapi_key,
                        "from": from_date.strftime("%Y-%m-%d"),
                        "to": to_date.strftime("%Y-%m-%d"),
                        "language": "en",
                        "sortBy": "relevancy",
                        "pageSize": 5
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])
                    
                    if articles:
                        summaries = []
                        for article in articles[:5]:
                            title = article.get("title", "")
                            description = article.get("description", "")
                            url = article.get("url", "")
                            source = article.get("source", {}).get("name", "")
                            
                            summaries.append(f"• {title} ({source}) - {url}")
                        
                        return f"Recent news for {entity} (last {days} days):\n" + "\n".join(summaries)
        
        except Exception as e:
            print(f"NewsAPI error: {e}")
    
    # Fallback to Tavily if available
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": f"{entity} pharmaceutical news recent",
                        "max_results": 5,
                        "search_depth": "basic",
                        "include_answer": True,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    results = data.get("results", [])
                    
                    summaries = [answer] if answer else []
                    for result in results[:3]:
                        title = result.get("title", "")
                        url = result.get("url", "")
                        summaries.append(f"• {title} - {url}")
                    
                    return f"Recent market news for {entity}:\n" + "\n".join(summaries)
        
        except Exception as e:
            print(f"Tavily error: {e}")
    
    # LLM fallback
    instruction = f"Summarize recent market news and developments for {entity} in the pharmaceutical industry."
    return _llm_fallback(instruction, entity, llm)


@tool
def company_announcements(entity: str, months: int = 6) -> str:
    """Get recent company announcements, press releases, and investor updates.
    
    Args:
        entity: Name of pharmaceutical company or PBM
        months: Number of months to look back (default: 6)
        
    Returns:
        Summary of major announcements with sources
    """
    from main import llm
    
    # Try web search for press releases
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": f"{entity} press release announcement earnings investor",
                        "max_results": 5,
                        "search_depth": "basic",
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        summaries = []
                        for result in results[:5]:
                            title = result.get("title", "")
                            url = result.get("url", "")
                            snippet = result.get("content", "")
                            summaries.append(f"• {title}\n  {_compact(snippet, 100)}\n  Source: {url}")
                        
                        return f"Company announcements for {entity}:\n\n" + "\n\n".join(summaries)
        
        except Exception as e:
            print(f"Search error: {e}")
    
    # LLM fallback
    instruction = f"Summarize recent major announcements, press releases, and investor updates for {entity}."
    return _llm_fallback(instruction, entity, llm)


@tool
def sentiment_analysis(entity: str) -> str:
    """Analyze market sentiment and analyst opinions for a pharmaceutical entity.
    
    Args:
        entity: Name of pharmaceutical company or PBM
        
    Returns:
        Summary of market sentiment with sources
    """
    from main import llm
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": f"{entity} stock analyst rating outlook sentiment",
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": True,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    if answer:
                        return f"Market sentiment for {entity}:\n{answer}"
        
        except Exception as e:
            print(f"Search error: {e}")
    
    # LLM fallback
    instruction = f"Provide a brief market sentiment analysis and analyst outlook for {entity}."
    return _llm_fallback(instruction, entity, llm)


# ============================================================================
# LEGISLATIVE AGENT TOOLS
# ============================================================================

@tool
def federal_legislation(entity: str) -> str:
    """Get relevant federal legislation and regulatory actions affecting the entity.
    
    Args:
        entity: Name of pharmaceutical company or PBM
        
    Returns:
        Summary of relevant federal bills and regulations
    """
    from main import llm
    
    # Try searching for legislative mentions
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": f"{entity} federal legislation congress bill regulation FDA",
                        "max_results": 5,
                        "search_depth": "basic",
                        "include_answer": True,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    results = data.get("results", [])
                    
                    summaries = [f"Legislative overview: {answer}"] if answer else []
                    for result in results[:3]:
                        title = result.get("title", "")
                        url = result.get("url", "")
                        summaries.append(f"• {title} - {url}")
                    
                    return f"Federal legislation affecting {entity}:\n" + "\n".join(summaries)
        
        except Exception as e:
            print(f"Search error: {e}")
    
    # LLM fallback
    instruction = f"Summarize recent federal legislation and regulatory actions affecting {entity} in the pharmaceutical industry."
    return _llm_fallback(instruction, entity, llm)


@tool
def state_bills(entity: str) -> str:
    """Get relevant state-level legislation affecting pharmaceutical pricing and access.
    
    Args:
        entity: Name of pharmaceutical company or PBM
        
    Returns:
        Summary of relevant state legislation
    """
    from main import llm
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": f"{entity} state legislation pricing transparency PBM reform",
                        "max_results": 5,
                        "search_depth": "basic",
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        summaries = []
                        for result in results[:5]:
                            title = result.get("title", "")
                            url = result.get("url", "")
                            summaries.append(f"• {title} - {url}")
                        
                        return f"State legislation affecting {entity}:\n" + "\n".join(summaries)
        
        except Exception as e:
            print(f"Search error: {e}")
    
    # LLM fallback
    instruction = f"Summarize recent state-level legislation affecting {entity}, particularly around pricing and PBM reform."
    return _llm_fallback(instruction, entity, llm)


@tool
def cms_updates(entity: str) -> str:
    """Get CMS (Centers for Medicare & Medicaid Services) policy updates affecting the entity.
    
    Args:
        entity: Name of pharmaceutical company or PBM
        
    Returns:
        Summary of relevant CMS policy changes
    """
    from main import llm
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": f"{entity} CMS Medicare Medicaid policy coverage reimbursement",
                        "max_results": 5,
                        "search_depth": "basic",
                        "include_answer": True,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    if answer:
                        return f"CMS policy updates for {entity}:\n{answer}"
        
        except Exception as e:
            print(f"Search error: {e}")
    
    # LLM fallback
    instruction = f"Summarize recent CMS policy changes and Medicare/Medicaid updates affecting {entity}."
    return _llm_fallback(instruction, entity, llm)


# ============================================================================
# PRODUCT AGENT TOOLS
# ============================================================================

@tool
def fda_approvals(entity: str, months: int = 12) -> str:
    """Get recent FDA approvals, clearances, and regulatory submissions for the entity.
    
    Args:
        entity: Name of pharmaceutical company
        months: Number of months to look back (default: 12)
        
    Returns:
        Summary of FDA approvals and submissions
    """
    from main import llm
    
    # FDA API is public, but search can be complex
    # Use Tavily to find FDA-related news
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": f"{entity} FDA approval clearance submission drug device",
                        "max_results": 5,
                        "search_depth": "basic",
                        "include_answer": True,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    results = data.get("results", [])
                    
                    summaries = [f"FDA activity overview: {answer}"] if answer else []
                    for result in results[:4]:
                        title = result.get("title", "")
                        url = result.get("url", "")
                        summaries.append(f"• {title} - {url}")
                    
                    return f"FDA approvals and submissions for {entity} (last {months} months):\n" + "\n".join(summaries)
        
        except Exception as e:
            print(f"Search error: {e}")
    
    # LLM fallback
    instruction = f"Summarize recent FDA approvals, clearances, and regulatory submissions for {entity} in the last {months} months."
    return _llm_fallback(instruction, entity, llm)


@tool
def clinical_trials(entity: str, phase: str = "3") -> str:
    """Get information about clinical trials for the entity's drugs.
    
    Args:
        entity: Name of pharmaceutical company
        phase: Trial phase to focus on (default: "3" for Phase III)
        
    Returns:
        Summary of relevant clinical trials
    """
    from main import llm
    
    # ClinicalTrials.gov has a public API
    # For MVP, use web search to find trial information
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": f"{entity} clinical trial phase {phase} study results",
                        "max_results": 5,
                        "search_depth": "basic",
                        "include_answer": True,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    results = data.get("results", [])
                    
                    summaries = [f"Clinical trials overview: {answer}"] if answer else []
                    for result in results[:4]:
                        title = result.get("title", "")
                        url = result.get("url", "")
                        summaries.append(f"• {title} - {url}")
                    
                    return f"Clinical trials for {entity} (Phase {phase}):\n" + "\n".join(summaries)
        
        except Exception as e:
            print(f"Search error: {e}")
    
    # LLM fallback
    instruction = f"Summarize recent Phase {phase} clinical trials and results for {entity}."
    return _llm_fallback(instruction, entity, llm)


@tool
def pipeline_status(entity: str) -> str:
    """Get the product pipeline status and development programs for the entity.
    
    Args:
        entity: Name of pharmaceutical company
        
    Returns:
        Summary of product pipeline and development programs
    """
    from main import llm
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=API_TIMEOUT) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": f"{entity} drug pipeline development program candidate",
                        "max_results": 5,
                        "search_depth": "basic",
                        "include_answer": True,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    results = data.get("results", [])
                    
                    summaries = [f"Pipeline overview: {answer}"] if answer else []
                    for result in results[:4]:
                        title = result.get("title", "")
                        url = result.get("url", "")
                        summaries.append(f"• {title} - {url}")
                    
                    return f"Product pipeline for {entity}:\n" + "\n".join(summaries)
        
        except Exception as e:
            print(f"Search error: {e}")
    
    # LLM fallback
    instruction = f"Summarize the product pipeline and development programs for {entity}."
    return _llm_fallback(instruction, entity, llm)

