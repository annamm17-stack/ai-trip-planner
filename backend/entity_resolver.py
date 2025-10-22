"""Entity resolution module for pharmaceutical manufacturers and PBMs.

Provides fuzzy matching with RapidFuzz and fallback to OpenCorporates API
for entities not in the curated database.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
from rapidfuzz import fuzz, process


class EntityResolver:
    """Resolves pharmaceutical entities using fuzzy matching and API fallback."""
    
    def __init__(self, data_path: Path):
        """Initialize resolver with entity database.
        
        Args:
            data_path: Path to pharma_entities.json file
        """
        self.entities: List[Dict[str, Any]] = []
        self.name_to_entity: Dict[str, Dict[str, Any]] = {}
        self._load_entities(data_path)
    
    def _load_entities(self, data_path: Path) -> None:
        """Load entities from JSON file and create lookup index."""
        if not data_path.exists():
            print(f"Warning: Entity database not found at {data_path}")
            return
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Combine manufacturers and PBMs into single list
            all_entities = data.get("manufacturers", []) + data.get("pbms", [])
            
            for entity in all_entities:
                self.entities.append(entity)
                # Index by primary name
                self.name_to_entity[entity["name"].lower()] = entity
                # Index by all aliases
                for alias in entity.get("aliases", []):
                    if alias:
                        self.name_to_entity[alias.lower()] = entity
            
            print(f"Loaded {len(self.entities)} entities ({len(self.name_to_entity)} total names/aliases)")
        
        except Exception as e:
            print(f"Error loading entities: {e}")
    
    def fuzzy_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fuzzy search for entities matching query.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching entities with scores
        """
        if not query or not query.strip():
            return []
        
        query = query.strip()
        
        # Exact match check first
        if query.lower() in self.name_to_entity:
            entity = self.name_to_entity[query.lower()]
            return [{
                "id": entity["id"],
                "name": entity["name"],
                "type": entity["type"],
                "ticker": entity.get("ticker"),
                "headquarters": entity.get("headquarters"),
                "score": 100
            }]
        
        # Fuzzy matching across all names and aliases
        all_names = list(self.name_to_entity.keys())
        
        # Use RapidFuzz to find best matches
        matches = process.extract(
            query.lower(),
            all_names,
            scorer=fuzz.ratio,
            limit=limit * 2  # Get more to deduplicate
        )
        
        # Deduplicate results (same entity can match via multiple aliases)
        seen_ids = set()
        results = []
        
        for match_name, score, _ in matches:
            entity = self.name_to_entity[match_name]
            entity_id = entity["id"]
            
            # Skip if we've already included this entity
            if entity_id in seen_ids:
                continue
            
            # Only include matches with reasonable scores
            if score >= 60:
                seen_ids.add(entity_id)
                results.append({
                    "id": entity_id,
                    "name": entity["name"],
                    "type": entity["type"],
                    "ticker": entity.get("ticker"),
                    "headquarters": entity.get("headquarters"),
                    "score": round(score, 1)
                })
            
            if len(results) >= limit:
                break
        
        return results
    
    def resolve_entity(self, query: str) -> Optional[Dict[str, Any]]:
        """Resolve a single entity with high confidence.
        
        Args:
            query: Entity name or alias
            
        Returns:
            Best matching entity or None if no good match
        """
        matches = self.fuzzy_search(query, limit=1)
        
        if not matches:
            # Try API fallback for unknown entities
            return self._fallback_api_search(query)
        
        best_match = matches[0]
        
        # Require high confidence for resolution
        if best_match["score"] >= 80:
            return best_match
        
        # If score is low, try API fallback
        return self._fallback_api_search(query)
    
    def _fallback_api_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Search OpenCorporates API for unknown entities.
        
        Args:
            query: Company name to search
            
        Returns:
            Entity data if found, None otherwise
        """
        import os
        
        api_key = os.getenv("OPENCORPORATES_API_KEY")
        
        # If no API key, return None (graceful degradation)
        if not api_key:
            return None
        
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    "https://api.opencorporates.com/v0.4/companies/search",
                    params={
                        "q": query,
                        "api_token": api_key,
                        "jurisdiction_code": "us",
                        "per_page": 1
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", {}).get("companies", [])
                    
                    if results:
                        company = results[0].get("company", {})
                        
                        # Check if it's healthcare/pharma related
                        industry_keywords = ["pharmaceutical", "pharma", "biotech", "health", "medical"]
                        company_name = company.get("name", "").lower()
                        
                        is_relevant = any(kw in company_name for kw in industry_keywords)
                        
                        if is_relevant:
                            return {
                                "id": company_name.replace(" ", "-").lower(),
                                "name": company.get("name"),
                                "type": "manufacturer",  # Default assumption
                                "ticker": None,
                                "headquarters": company.get("registered_address_in_full"),
                                "score": 75,
                                "source": "opencorporates"
                            }
        
        except Exception as e:
            print(f"OpenCorporates API error: {e}")
        
        return None
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by its ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity data or None if not found
        """
        for entity in self.entities:
            if entity["id"] == entity_id:
                return entity
        return None


# Global instance (initialized at module level)
_resolver: Optional[EntityResolver] = None


def get_resolver() -> EntityResolver:
    """Get or create global EntityResolver instance."""
    global _resolver
    if _resolver is None:
        data_dir = Path(__file__).parent / "data"
        _resolver = EntityResolver(data_dir / "pharma_entities.json")
    return _resolver

