"""
evaluator.py — Evaluation Layer

Metrics:
  - Confidence Score (VAE reconstruction probability)
  - Anomaly Likelihood (distance from cluster centroid)
  - Evidence Coverage
  - LLM-based faithfulness audit
"""

import os
import json
import re
import numpy as np
from src.llm_wrapper import LLMProvider


def evaluate_narrative(narrative, context_docs, vae=None, clusterer=None,
                       user_latent=None, user_embeddings=None):
    """
    Comprehensive evaluation of the generated narrative.
    Combines statistical metrics with LLM-based auditing.
    """
    metrics = {}

    # =========================================================
    # Metric 1: VAE Reconstruction Probability
    # =========================================================
    if vae is not None and user_embeddings is not None:
        try:
            recon_loss = vae.get_reconstruction_probability(user_embeddings)
            avg_recon = float(np.mean(recon_loss))
            # Normalize: lower loss = higher confidence
            confidence = max(0, min(1, 1.0 - (avg_recon / 10.0)))
            metrics["vae_confidence"] = round(confidence, 3)
            metrics["avg_reconstruction_loss"] = round(avg_recon, 4)
        except Exception as e:
            metrics["vae_confidence"] = None
            metrics["vae_error"] = str(e)

    # =========================================================
    # Metric 2: Anomaly Likelihood (Cluster Distance)
    # =========================================================
    if clusterer is not None and user_latent is not None:
        try:
            distances = clusterer.get_distance_to_centroid(user_latent)
            avg_dist = float(np.mean(distances))
            # Normalize: closer to centroid = less anomalous
            anomaly_score = min(1.0, avg_dist / 10.0)
            metrics["anomaly_likelihood"] = round(anomaly_score, 3)
            metrics["avg_cluster_distance"] = round(avg_dist, 4)
        except Exception as e:
            metrics["anomaly_likelihood"] = None
            metrics["cluster_error"] = str(e)

    # =========================================================
    # Metric 3: Evidence Coverage
    # =========================================================
    if context_docs:
        # Check how many context docs are referenced in the narrative
        referenced = 0
        for doc in context_docs:
            # Check if key phrases from the document appear in the narrative
            key_words = doc.split()[:5]  # First 5 words as identifier
            key_phrase = " ".join(key_words)
            if any(word.lower() in narrative.lower() for word in key_words if len(word) > 3):
                referenced += 1
        
        coverage = referenced / len(context_docs) if context_docs else 0
        metrics["evidence_coverage"] = round(coverage, 3)
        metrics["docs_referenced"] = referenced
        metrics["total_docs"] = len(context_docs)

    # =========================================================
    # Metric 4: LLM-based Faithfulness Audit
    # =========================================================
    try:
        provider_name = os.getenv("LLM_PROVIDER", "huggingface")
        llm = LLMProvider(provider=provider_name)

        context_text = "\n---\n".join(context_docs[:3])  # Limit context

        audit_prompt = f"""
        NARRATIVE TO AUDIT:
        {narrative[:1000]}

        SOURCE CONTEXT:
        {context_text[:1000]}

        TASK: Rate the narrative's faithfulness to the source on a scale of 0-1.
        Return ONLY valid JSON: {{"score": 0.0, "critique": "..."}}
        """

        response_text = llm.generate(
            audit_prompt,
            "You are a specialized AI Auditor. Return ONLY valid JSON."
        )

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            audit = json.loads(json_match.group(0))
            metrics["faithfulness_score"] = audit.get("score", 0.5)
            metrics["critique"] = audit.get("critique", "N/A")
        else:
            metrics["faithfulness_score"] = 0.5
            metrics["critique"] = "Could not parse auditor response."
    except Exception as e:
        metrics["faithfulness_score"] = None
        metrics["audit_error"] = str(e)

    return metrics
