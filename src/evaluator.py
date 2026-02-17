import os
import openai
from src.llm_wrapper import LLMProvider

def evaluate_narrative(narrative, context_docs):
    """
    Perform a 'second pass' to evaluate the narrative for faithfulness and consistency.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"score": 0, "critique": "API Key missing"}

    # Initialize provider
    provider_name = os.getenv("LLM_PROVIDER", "huggingface") 
    llm = LLMProvider(provider=provider_name)

    response_text = llm.generate(prompt, "You are a specialized AI Auditor focus on Fact-Checking and Hallucination detection. Return ONLY valid JSON.")
    
    try:
        # Attempt to find JSON in response if it's wrapped in text
        import json
        import re
        
        # Look for { ... } structure
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return {"score": 0.5, "critique": "Could not parse auditor response.", "is_valid": False, "raw_response": response_text}
            
    except Exception as e:
        return {"error": str(e), "is_valid": False}
