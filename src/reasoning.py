"""
reasoning.py — Hierarchical Reasoning Pipeline

Implements multi-layer reasoning over user data projected into
the learned latent space.

Layers:
  1. Planning    — Query decomposition
  2. Retrieval   — Cluster-aware semantic search
  3. Evidence    — Cross-source relationship construction
  4. Validation  — Consistency and conflict detection
"""

import os
import re
from src.llm_wrapper import LLMProvider


class HierarchicalReasoner:
    """
    Processes user queries through a multi-layer reasoning pipeline
    before passing validated evidence to the generative synthesis stage.
    """

    def __init__(self):
        provider_name = os.getenv("LLM_PROVIDER", "huggingface")
        self.llm = LLMProvider(provider=provider_name)

    # =========================================================
    # Layer 1: Planning — Query Decomposition
    # =========================================================
    def decompose_query(self, query):
        """
        Break a complex query into sub-queries for targeted retrieval.
        Returns a list of sub-queries.
        """
        # For simple queries, return as-is
        if len(query.split()) < 10:
            return [query]

        system_prompt = "You are a query decomposition expert. Break the given query into 2-4 simpler, focused sub-queries. Return ONLY the sub-queries, one per line, numbered."
        
        try:
            response = self.llm.generate(
                f"Decompose this query into sub-queries:\n\n{query}",
                system_prompt
            )
            # Parse numbered sub-queries
            sub_queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                if cleaned and len(cleaned) > 5:
                    sub_queries.append(cleaned)
            
            return sub_queries if sub_queries else [query]
        except Exception:
            return [query]

    # =========================================================
    # Layer 2: Retrieval — Cluster-Aware Semantic Search
    # =========================================================
    def cluster_aware_retrieve(self, query_embedding, query_cluster,
                                vector_store, all_clusters, n_results=5):
        """
        Retrieve documents with cluster-aware scoring.
        Documents in the same cluster as the query get priority.
        """
        # Standard retrieval
        raw_results = vector_store.query(query_embedding, n_results=n_results * 2)
        
        # Re-rank: boost same-cluster results
        scored_results = []
        for i, (doc, dist) in enumerate(raw_results):
            cluster_boost = 0.0
            if all_clusters is not None and i < len(all_clusters):
                if all_clusters[i] == query_cluster:
                    cluster_boost = -0.3  # Lower distance = higher priority
            
            adjusted_score = dist + cluster_boost
            scored_results.append((doc, adjusted_score))
        
        # Sort by adjusted score (lower is better)
        scored_results.sort(key=lambda x: x[1])
        return scored_results[:n_results]

    # =========================================================
    # Layer 3: Evidence Linking
    # =========================================================
    def link_evidence(self, retrieved_docs):
        """
        Build cross-source relationships between retrieved documents.
        Groups documents by source type and identifies connections.
        """
        evidence_map = {
            "csv_data": [],
            "pdf_docs": [],
            "log_entries": [],
            "text_docs": [],
        }
        
        for doc, score in retrieved_docs:
            doc_lower = doc.lower()
            if any(kw in doc_lower for kw in ['revenue', 'profit', 'cost', 'quarter', '$', 'record']):
                evidence_map["csv_data"].append({"content": doc, "score": score})
            elif any(kw in doc_lower for kw in ['log', 'error', 'warn', 'info', 'server']):
                evidence_map["log_entries"].append({"content": doc, "score": score})
            elif any(kw in doc_lower for kw in ['abstract', 'research', 'study', 'paper']):
                evidence_map["pdf_docs"].append({"content": doc, "score": score})
            else:
                evidence_map["text_docs"].append({"content": doc, "score": score})
        
        # Remove empty categories
        evidence_map = {k: v for k, v in evidence_map.items() if v}
        
        return evidence_map

    # =========================================================
    # Layer 4: Validation — Consistency & Conflict Detection
    # =========================================================
    def validate_evidence(self, evidence_map):
        """
        Check evidence for consistency and flag potential conflicts.
        Returns validated evidence with confidence metadata.
        """
        validation_result = {
            "validated_evidence": [],
            "conflicts": [],
            "confidence": 1.0,
            "source_count": 0,
            "cross_source": False,
        }
        
        source_types = list(evidence_map.keys())
        validation_result["source_count"] = len(source_types)
        validation_result["cross_source"] = len(source_types) > 1
        
        # Collect all evidence
        all_evidence = []
        for source_type, docs in evidence_map.items():
            for doc in docs:
                doc["source_type"] = source_type
                all_evidence.append(doc)
        
        # Simple conflict detection: check if any docs contradict each other
        # (Basic heuristic: if scores are very different, flag as lower confidence)
        if all_evidence:
            scores = [d["score"] for d in all_evidence]
            score_variance = max(scores) - min(scores) if len(scores) > 1 else 0
            
            if score_variance > 5.0:
                validation_result["confidence"] = 0.6
                validation_result["conflicts"].append(
                    "High variance in retrieval scores — evidence may be inconsistent."
                )
            elif score_variance > 2.0:
                validation_result["confidence"] = 0.8
        
        validation_result["validated_evidence"] = all_evidence
        return validation_result

    # =========================================================
    # Full Pipeline
    # =========================================================
    def reason(self, query, query_embedding, query_cluster,
               vector_store, all_clusters=None):
        """
        Execute the full hierarchical reasoning pipeline.
        Returns validated, structured evidence ready for synthesis.
        """
        # Layer 1: Decompose query
        sub_queries = self.decompose_query(query)
        
        # Layer 2: Retrieve for each sub-query (use main embedding for now)
        all_retrieved = self.cluster_aware_retrieve(
            query_embedding, query_cluster, vector_store, all_clusters
        )
        
        # Layer 3: Link evidence across sources
        evidence_map = self.link_evidence(all_retrieved)
        
        # Layer 4: Validate
        validation = self.validate_evidence(evidence_map)
        
        return {
            "sub_queries": sub_queries,
            "evidence_map": evidence_map,
            "validation": validation,
        }
